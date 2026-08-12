use std::collections::BTreeSet;
use std::net::IpAddr;
use std::str;
use std::time::Duration;

use dom_query::Document;
use futures_util::StreamExt;
use reqwest::{Client, Response, StatusCode, header::HeaderMap};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::event::AgentEvent;
use crate::protocol::{
    ChatMessage, FunctionCall, MessageRole, ToolCall, ToolDefinition, Usage, WebSearchAction,
};

const MAX_REQUEST_ATTEMPTS: usize = 4;
const MAX_STREAM_ATTEMPTS: usize = 2;

#[derive(Debug, thiserror::Error)]
pub enum OpenAiError {
    #[error("request cancelled")]
    Cancelled,
    #[error("OpenAI request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid OpenAI URL: {0}")]
    Url(#[from] url::ParseError),
    #[error("failed to encode OpenAI request: {0}")]
    Json(#[from] serde_json::Error),
    #[error("API 返回 HTTP {status}：{body}{request_id}", request_id = request_id.as_deref().map(|id| format!("（请求 ID：{id}）")).unwrap_or_default())]
    Api {
        status: u16,
        body: String,
        request_id: Option<String>,
    },
    #[error("invalid OpenAI stream: {0}")]
    Protocol(String),
    #[error("OpenAI stream was interrupted: {message}{request_id}", request_id = request_id.as_deref().map(|id| format!(" (request id: {id})")).unwrap_or_default())]
    Stream {
        message: String,
        request_id: Option<String>,
    },
}

impl OpenAiError {
    #[must_use]
    pub fn is_context_overflow(&self) -> bool {
        match self {
            Self::Api { status, body, .. } => {
                matches!(status, 400 | 413 | 422) && is_context_overflow_text(body)
            }
            Self::Protocol(message) => is_context_overflow_text(message),
            Self::Cancelled
            | Self::Http(_)
            | Self::Url(_)
            | Self::Json(_)
            | Self::Stream { .. } => false,
        }
    }

    fn is_retryable_stream_failure(&self) -> bool {
        matches!(self, Self::Stream { .. })
    }
}

pub type Result<T> = std::result::Result<T, OpenAiError>;

#[derive(Debug, Clone)]
pub(crate) struct OpenAiModelConfig {
    pub provider: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub max_output_tokens: Option<u64>,
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    http: Client,
    endpoint: Url,
    provider: String,
    api_key: Option<String>,
    model: String,
    max_output_tokens: Option<u64>,
    reasoning_effort: Option<String>,
    prompt_cache_key: String,
    idle_timeout: Duration,
}

#[derive(Debug, Clone, Copy)]
struct StreamFeatures {
    native_web_search: bool,
    require_local_tool: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantTurn {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub response_items: Vec<serde_json::Value>,
    pub usage: Option<Usage>,
    pub stop_reason: AssistantStopReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssistantStopReason {
    Stop,
    Length,
    ToolUse,
}

impl OpenAiClient {
    pub(crate) fn new(
        config: OpenAiModelConfig,
        prompt_cache_key: String,
        timeout: Duration,
    ) -> Result<Self> {
        let endpoint = api_endpoint_url(&config.base_url)?;
        let http = build_http_client(&endpoint, timeout)?;
        Ok(Self {
            http,
            endpoint,
            provider: config.provider,
            api_key: config.api_key,
            model: config.model,
            max_output_tokens: config.max_output_tokens,
            reasoning_effort: config.reasoning_effort,
            prompt_cache_key,
            idle_timeout: timeout,
        })
    }

    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    #[must_use]
    pub fn endpoint(&self) -> &Url {
        &self.endpoint
    }

    #[must_use]
    pub const fn max_output_tokens(&self) -> Option<u64> {
        self.max_output_tokens
    }

    pub(crate) fn reconfigure(&mut self, config: OpenAiModelConfig) -> Result<()> {
        let endpoint = api_endpoint_url(&config.base_url)?;
        self.http = build_http_client(&endpoint, self.idle_timeout)?;
        self.endpoint = endpoint;
        self.provider = config.provider;
        self.api_key = config.api_key;
        self.model = config.model;
        self.max_output_tokens = config.max_output_tokens;
        self.reasoning_effort = config.reasoning_effort;
        Ok(())
    }

    pub fn set_reasoning_effort(&mut self, reasoning_effort: Option<String>) {
        self.reasoning_effort = reasoning_effort;
    }

    pub fn set_prompt_cache_key(&mut self, prompt_cache_key: String) {
        self.prompt_cache_key = prompt_cache_key;
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn reasoning_effort(&self) -> Option<&str> {
        self.reasoning_effort.as_deref()
    }

    async fn send_stream_request<T: Serialize + Sync + ?Sized>(
        &self,
        body: &T,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> Result<Response> {
        for attempt in 0..MAX_REQUEST_ATTEMPTS {
            let mut request = self
                .http
                .post(self.endpoint.clone())
                .header("Accept", "text/event-stream")
                .json(body);
            if let Some(api_key) = &self.api_key {
                request = request.bearer_auth(api_key);
            }

            let response = tokio::select! {
                () = cancel.cancelled() => return Err(OpenAiError::Cancelled),
                response = request.send() => response,
            };
            match response {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let status = response.status();
                    let request_id = response_request_id(response.headers());
                    let retry_after = retry_after_delay(response.headers());
                    let body = tokio::select! {
                        () = cancel.cancelled() => return Err(OpenAiError::Cancelled),
                        text = response.text() => text.unwrap_or_else(|error| format!("<failed to read response: {error}>")),
                    };
                    let error = OpenAiError::Api {
                        status: status.as_u16(),
                        body: summarize_error_body(&body),
                        request_id,
                    };
                    if !is_retryable_status(status) || attempt + 1 == MAX_REQUEST_ATTEMPTS {
                        return Err(error);
                    }
                    let _ = events.send(AgentEvent::AssistantRetrying {
                        attempt: attempt + 2,
                        max_attempts: MAX_REQUEST_ATTEMPTS,
                        message: error.to_string(),
                    });
                    wait_before_retry(
                        retry_after.unwrap_or_else(|| retry_backoff(attempt)),
                        cancel,
                    )
                    .await?;
                }
                Err(error) => {
                    if error.is_builder() || attempt + 1 == MAX_REQUEST_ATTEMPTS {
                        return Err(OpenAiError::Http(error));
                    }
                    let _ = events.send(AgentEvent::AssistantRetrying {
                        attempt: attempt + 2,
                        max_attempts: MAX_REQUEST_ATTEMPTS,
                        message: error.to_string(),
                    });
                    wait_before_retry(retry_backoff(attempt), cancel).await?;
                }
            }
        }
        unreachable!("request retry loop always returns")
    }

    pub async fn stream_response(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> Result<AssistantTurn> {
        self.stream_response_with_options(
            messages,
            tools,
            events,
            cancel,
            self.max_output_tokens,
            StreamFeatures {
                native_web_search: true,
                require_local_tool: false,
            },
        )
        .await
    }

    pub async fn stream_response_requiring_local_tool(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> Result<AssistantTurn> {
        self.stream_response_with_options(
            messages,
            tools,
            events,
            cancel,
            self.max_output_tokens,
            StreamFeatures {
                native_web_search: false,
                require_local_tool: true,
            },
        )
        .await
    }

    pub async fn stream_response_with_max_tokens(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        max_tokens: Option<u64>,
    ) -> Result<AssistantTurn> {
        self.stream_response_with_options(
            messages,
            tools,
            events,
            cancel,
            max_tokens,
            StreamFeatures {
                native_web_search: false,
                require_local_tool: false,
            },
        )
        .await
    }

    async fn stream_response_with_options(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        max_tokens: Option<u64>,
        features: StreamFeatures,
    ) -> Result<AssistantTurn> {
        for attempt in 0..MAX_STREAM_ATTEMPTS {
            let result = self
                .stream_responses(messages, tools, events, cancel, max_tokens, features)
                .await;
            match result {
                Ok(turn) => return Ok(turn),
                Err(error)
                    if error.is_retryable_stream_failure() && attempt + 1 < MAX_STREAM_ATTEMPTS =>
                {
                    let _ = events.send(AgentEvent::AssistantRetrying {
                        attempt: attempt + 2,
                        max_attempts: MAX_STREAM_ATTEMPTS,
                        message: error.to_string(),
                    });
                    wait_before_retry(retry_backoff(attempt), cancel).await?;
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("stream retry loop always returns")
    }

    async fn stream_responses(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        max_tokens: Option<u64>,
        features: StreamFeatures,
    ) -> Result<AssistantTurn> {
        let (instructions, input) = responses_input(messages)?;
        let response_tools = responses_tools(tools, features.native_web_search);
        let has_tools = !response_tools.is_empty();
        let body = ResponsesRequest {
            model: &self.model,
            instructions: &instructions,
            input: &input,
            tools: &response_tools,
            tool_choice: has_tools.then_some(if features.require_local_tool {
                "required"
            } else {
                "auto"
            }),
            parallel_tool_calls: has_tools.then_some(true),
            reasoning: self
                .reasoning_effort
                .as_deref()
                .map(|effort| ResponsesReasoning {
                    effort,
                    summary: (self.provider == "xai").then_some("auto"),
                }),
            max_output_tokens: max_tokens,
            prompt_cache_key: (self.provider == "xai").then_some(self.prompt_cache_key.as_str()),
            include: (self.provider == "xai").then_some(["reasoning.encrypted_content"].as_slice()),
            store: false,
            stream: true,
        };

        let response = self.send_stream_request(&body, events, cancel).await?;
        let request_id = response_request_id(response.headers());

        let mut bytes = response.bytes_stream();
        let mut decoder = SseDecoder::default();
        let mut state = ResponsesAccumulator::default();
        let mut done = false;
        while !done {
            let chunk = tokio::select! {
                () = cancel.cancelled() => return Err(OpenAiError::Cancelled),
                chunk = bytes.next() => chunk,
            };
            let Some(chunk) = chunk else {
                break;
            };
            let chunk =
                chunk.map_err(|error| stream_transport_error(&error, request_id.as_deref()))?;
            for data in decoder
                .push(&chunk)
                .map_err(|error| attach_request_id(error, request_id.as_deref()))?
            {
                if data.trim() == "[DONE]" {
                    done = true;
                    break;
                }
                apply_responses_stream_event(&data, &mut state, events)
                    .map_err(|error| attach_request_id(error, request_id.as_deref()))?;
                if state.completed {
                    done = true;
                    break;
                }
            }
        }
        for data in decoder
            .finish()
            .map_err(|error| attach_request_id(error, request_id.as_deref()))?
        {
            if data.trim() != "[DONE]" {
                apply_responses_stream_event(&data, &mut state, events)
                    .map_err(|error| attach_request_id(error, request_id.as_deref()))?;
            }
        }
        if !state.completed {
            return Err(stream_error(
                "Responses stream ended before a terminal response event",
                request_id.as_deref(),
            ));
        }

        let reasoning = if state.raw_reasoning.is_empty() {
            state.reasoning
        } else {
            state.raw_reasoning
        };
        let stop_reason = state.stop_reason.unwrap_or({
            if state.tool_calls.is_empty() {
                AssistantStopReason::Stop
            } else {
                AssistantStopReason::ToolUse
            }
        });
        Ok(AssistantTurn {
            content: (!state.content.is_empty()).then_some(state.content),
            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls: state.tool_calls,
            response_items: state.response_items,
            usage: state.usage,
            stop_reason,
        })
    }
}

#[derive(Debug, Serialize)]
struct ResponsesRequest<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "str::is_empty")]
    instructions: &'a str,
    input: &'a [serde_json::Value],
    #[serde(skip_serializing_if = "<[ResponsesTool]>::is_empty")]
    tools: &'a [ResponsesTool],
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoning<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<&'a [&'static str]>,
    store: bool,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ResponsesReasoning<'a> {
    effort: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<&'static str>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesInputItem {
    Message {
        role: MessageRole,
        content: Vec<ResponsesInputContent>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesInputContent {
    InputText { text: String },
    InputImage { image_url: String },
    OutputText { text: String },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesTool {
    Function {
        name: String,
        description: String,
        parameters: serde_json::Value,
    },
    WebSearch,
}

fn responses_input(messages: &[ChatMessage]) -> Result<(String, Vec<serde_json::Value>)> {
    let mut instructions = Vec::new();
    let mut input = Vec::new();
    for message in messages {
        match message.role {
            MessageRole::System => {
                if let Some(content) = message.content.as_deref().filter(|text| !text.is_empty()) {
                    instructions.push(content.to_string());
                }
            }
            MessageRole::User => {
                let mut content = Vec::with_capacity(message.images.len() + 1);
                if let Some(text) = message.content.as_deref().filter(|text| !text.is_empty()) {
                    content.push(ResponsesInputContent::InputText {
                        text: text.to_string(),
                    });
                }
                content.extend(message.images.iter().map(|image| {
                    ResponsesInputContent::InputImage {
                        image_url: image.data_url(),
                    }
                }));
                if !content.is_empty() {
                    input.push(serde_json::to_value(ResponsesInputItem::Message {
                        role: MessageRole::User,
                        content,
                    })?);
                }
            }
            MessageRole::Assistant => {
                if !message.response_items.is_empty() {
                    input.extend(message.response_items.iter().cloned());
                    continue;
                }
                if let Some(text) = message.content.as_deref().filter(|text| !text.is_empty()) {
                    input.push(serde_json::to_value(ResponsesInputItem::Message {
                        role: MessageRole::Assistant,
                        content: vec![ResponsesInputContent::OutputText {
                            text: text.to_string(),
                        }],
                    })?);
                }
                input.extend(
                    message
                        .tool_calls
                        .iter()
                        .map(|call| {
                            serde_json::to_value(ResponsesInputItem::FunctionCall {
                                call_id: call.id.clone(),
                                name: call.function.name.clone(),
                                arguments: call.function.arguments.clone(),
                            })
                        })
                        .collect::<std::result::Result<Vec<_>, _>>()?,
                );
            }
            MessageRole::Tool => {
                let call_id = message.tool_call_id.clone().ok_or_else(|| {
                    OpenAiError::Protocol("tool message is missing tool_call_id".to_string())
                })?;
                input.push(serde_json::to_value(
                    ResponsesInputItem::FunctionCallOutput {
                        call_id,
                        output: message.content.clone().unwrap_or_default(),
                    },
                )?);
            }
        }
    }
    Ok((instructions.join("\n\n"), input))
}

fn responses_tools(definitions: &[ToolDefinition], include_web_search: bool) -> Vec<ResponsesTool> {
    let mut tools = definitions
        .iter()
        .map(|definition| ResponsesTool::Function {
            name: definition.function.name.clone(),
            description: definition.function.description.clone(),
            parameters: definition.function.parameters.clone(),
        })
        .collect::<Vec<_>>();
    if include_web_search {
        tools.push(ResponsesTool::WebSearch);
    }
    tools
}

#[derive(Debug, Default)]
struct ResponsesAccumulator {
    content: String,
    reasoning: String,
    raw_reasoning: String,
    reasoning_summary_index: Option<usize>,
    tool_calls: Vec<ToolCall>,
    response_items: Vec<serde_json::Value>,
    seen_output_item_ids: BTreeSet<String>,
    seen_tool_calls: BTreeSet<String>,
    started_searches: BTreeSet<String>,
    cited_urls: BTreeSet<String>,
    usage: Option<Usage>,
    completed: bool,
    stop_reason: Option<AssistantStopReason>,
}

#[derive(Debug, Deserialize)]
struct ResponsesStreamEvent {
    #[serde(rename = "type")]
    kind: String,
    response: Option<serde_json::Value>,
    item: Option<serde_json::Value>,
    delta: Option<String>,
    text: Option<String>,
    summary_index: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesOutputItem {
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    WebSearchCall {
        id: String,
        #[serde(default)]
        action: Option<WebSearchAction>,
    },
    Message {
        #[serde(default)]
        content: Vec<ResponsesOutputContent>,
    },
    Reasoning {
        #[serde(default)]
        summary: Vec<ResponsesReasoningSummary>,
        #[serde(default)]
        content: Vec<ResponsesReasoningContent>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesOutputContent {
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<ResponsesAnnotation>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesAnnotation {
    UrlCitation {
        url: String,
        #[serde(default)]
        title: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesReasoningSummary {
    SummaryText {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponsesReasoningContent {
    ReasoningText {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct ResponsesTerminal {
    usage: Option<ResponsesUsage>,
    #[serde(default)]
    output: Vec<serde_json::Value>,
    incomplete_details: Option<ResponsesIncompleteDetails>,
}

#[derive(Debug, Deserialize)]
struct ResponsesIncompleteDetails {
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponsesUsage {
    #[serde(default, rename = "input_tokens")]
    input: u64,
    #[serde(default, rename = "output_tokens")]
    output: u64,
    #[serde(default, rename = "total_tokens")]
    total: u64,
    input_tokens_details: Option<CachedTokenDetails>,
}

fn apply_responses_stream_event(
    data: &str,
    state: &mut ResponsesAccumulator,
    events: &mpsc::UnboundedSender<AgentEvent>,
) -> Result<()> {
    let event: ResponsesStreamEvent = serde_json::from_str(data)
        .map_err(|error| OpenAiError::Protocol(format!("invalid Responses JSON event: {error}")))?;
    match event.kind.as_str() {
        "response.output_text.delta" => {
            if let Some(text) = event.delta {
                state.content.push_str(&text);
                let _ = events.send(AgentEvent::TextDelta { text });
            }
        }
        "response.reasoning_text.delta" => {
            if let Some(text) = event.delta {
                state.raw_reasoning.push_str(&text);
            }
        }
        "response.reasoning_text.done" => {
            if let Some(text) = event.text {
                merge_completed_text(&mut state.raw_reasoning, &text);
            }
        }
        "response.reasoning_summary_part.added" => {
            if let Some(index) = event.summary_index {
                begin_reasoning_summary_part(index, state, events);
            }
        }
        "response.reasoning_summary_text.delta" => {
            if let Some(text) = event.delta {
                if let Some(index) = event.summary_index {
                    begin_reasoning_summary_part(index, state, events);
                }
                state.reasoning.push_str(&text);
                let _ = events.send(AgentEvent::ReasoningSummaryDelta { text });
            }
        }
        "response.output_item.added" => {
            if let Some(ResponsesOutputItem::WebSearchCall { id, .. }) =
                parse_responses_item(event.item)?
            {
                start_web_search(&id, state, events);
            }
        }
        "response.output_item.done" => {
            if let Some(item) = event.item {
                apply_raw_responses_output_item(item, state, events)?;
            }
        }
        "response.completed" => {
            let response = event.response.ok_or_else(|| {
                OpenAiError::Protocol("response.completed is missing response".to_string())
            })?;
            let completed: ResponsesTerminal =
                serde_json::from_value(response).map_err(|error| {
                    OpenAiError::Protocol(format!("invalid response.completed payload: {error}"))
                })?;
            apply_responses_terminal(completed, state, events)?;
            state.completed = true;
        }
        "response.incomplete" => {
            let response = event.response.ok_or_else(|| {
                OpenAiError::Protocol("response.incomplete is missing response".to_string())
            })?;
            let incomplete: ResponsesTerminal =
                serde_json::from_value(response).map_err(|error| {
                    OpenAiError::Protocol(format!("invalid response.incomplete payload: {error}"))
                })?;
            let reason = incomplete
                .incomplete_details
                .as_ref()
                .and_then(|details| details.reason.as_deref());
            if reason != Some("max_output_tokens") {
                return Err(OpenAiError::Protocol(match reason {
                    Some(reason) => format!("provider returned incomplete response: {reason}"),
                    None => "provider returned incomplete response without a reason".to_string(),
                }));
            }
            apply_responses_terminal(incomplete, state, events)?;
            state.stop_reason = Some(AssistantStopReason::Length);
            state.completed = true;
        }
        "response.failed" | "error" => {
            let detail = event
                .response
                .map_or_else(|| data.to_string(), |response| response.to_string());
            return Err(OpenAiError::Protocol(format!(
                "provider returned {}: {detail}",
                event.kind
            )));
        }
        _ => {}
    }
    Ok(())
}

fn apply_responses_terminal(
    terminal: ResponsesTerminal,
    state: &mut ResponsesAccumulator,
    events: &mpsc::UnboundedSender<AgentEvent>,
) -> Result<()> {
    for item in &terminal.output {
        apply_raw_responses_output_item(item.clone(), state, events)?;
    }
    if !terminal.output.is_empty() {
        state.response_items = terminal.output;
    }
    state.usage = terminal.usage.map(|usage| Usage {
        prompt_tokens: usage.input,
        completion_tokens: usage.output,
        total_tokens: usage.total,
        cached_prompt_tokens: usage
            .input_tokens_details
            .and_then(|details| details.cached_tokens),
    });
    Ok(())
}

fn apply_raw_responses_output_item(
    value: serde_json::Value,
    state: &mut ResponsesAccumulator,
    events: &mpsc::UnboundedSender<AgentEvent>,
) -> Result<()> {
    let id = value
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string);
    let is_duplicate = id
        .as_ref()
        .is_some_and(|id| state.seen_output_item_ids.contains(id))
        || (id.is_none() && state.response_items.contains(&value));
    if is_duplicate {
        return Ok(());
    }
    if let Some(id) = id {
        state.seen_output_item_ids.insert(id);
    }
    let parsed = parse_responses_item(Some(value.clone()))?;
    state.response_items.push(value);
    if let Some(item) = parsed {
        apply_responses_output_item(item, state, events);
    }
    Ok(())
}

fn parse_responses_item(value: Option<serde_json::Value>) -> Result<Option<ResponsesOutputItem>> {
    value
        .map(|value| {
            serde_json::from_value(value).map_err(|error| {
                OpenAiError::Protocol(format!("invalid Responses output item: {error}"))
            })
        })
        .transpose()
}

fn apply_responses_output_item(
    item: ResponsesOutputItem,
    state: &mut ResponsesAccumulator,
    events: &mpsc::UnboundedSender<AgentEvent>,
) {
    match item {
        ResponsesOutputItem::FunctionCall {
            call_id,
            name,
            arguments,
        } => {
            if state.seen_tool_calls.insert(call_id.clone()) {
                state.tool_calls.push(ToolCall {
                    id: call_id,
                    kind: "function".to_string(),
                    function: FunctionCall { name, arguments },
                });
            }
        }
        ResponsesOutputItem::WebSearchCall { id, action } => {
            start_web_search(&id, state, events);
            let _ = events.send(AgentEvent::WebSearchFinished {
                id,
                action: action.unwrap_or(WebSearchAction::Other),
            });
        }
        ResponsesOutputItem::Message { content } => {
            let mut citations = Vec::new();
            let mut completed_text = String::new();
            for part in content {
                if let ResponsesOutputContent::OutputText { text, annotations } = part {
                    completed_text.push_str(&text);
                    citations.extend(annotations.into_iter().filter_map(|annotation| {
                        let ResponsesAnnotation::UrlCitation { url, title } = annotation else {
                            return None;
                        };
                        let title = title
                            .filter(|title| !title.trim().is_empty())
                            .unwrap_or_else(|| url.clone());
                        Some((url, title))
                    }));
                }
            }
            let missing_text = if state.content.is_empty() {
                completed_text
            } else {
                completed_text
                    .strip_prefix(&state.content)
                    .unwrap_or_default()
                    .to_string()
            };
            if !missing_text.is_empty() {
                state.content.push_str(&missing_text);
                let _ = events.send(AgentEvent::TextDelta { text: missing_text });
            }
            append_citations(citations, state, events);
        }
        ResponsesOutputItem::Reasoning { summary, content } => {
            let completed_reasoning = content
                .into_iter()
                .filter_map(|part| match part {
                    ResponsesReasoningContent::ReasoningText { text } => Some(text),
                    ResponsesReasoningContent::Other => None,
                })
                .collect::<String>();
            if !completed_reasoning.is_empty() {
                merge_completed_text(&mut state.raw_reasoning, &completed_reasoning);
            }
            let parts = summary
                .into_iter()
                .filter_map(|part| match part {
                    ResponsesReasoningSummary::SummaryText { text } => Some(text),
                    ResponsesReasoningSummary::Other => None,
                })
                .collect::<Vec<_>>();
            let has_summary = !state.reasoning.is_empty() || !parts.is_empty();
            if state.reasoning.is_empty() {
                for (index, text) in parts.into_iter().enumerate() {
                    begin_reasoning_summary_part(index, state, events);
                    state.reasoning.push_str(&text);
                    let _ = events.send(AgentEvent::ReasoningSummaryDelta { text });
                }
            }
            state.reasoning_summary_index = None;
            if has_summary {
                let _ = events.send(AgentEvent::ReasoningSummaryFinished);
            }
        }
        ResponsesOutputItem::Other => {}
    }
}

fn merge_completed_text(partial: &mut String, completed: &str) {
    if partial == completed {
        return;
    }
    if let Some(suffix) = completed.strip_prefix(partial.as_str()) {
        partial.push_str(suffix);
    } else {
        partial.clear();
        partial.push_str(completed);
    }
}

fn begin_reasoning_summary_part(
    index: usize,
    state: &mut ResponsesAccumulator,
    events: &mpsc::UnboundedSender<AgentEvent>,
) {
    if state.reasoning_summary_index == Some(index) {
        return;
    }
    if !state.reasoning.is_empty() && !state.reasoning.ends_with("\n\n") {
        if state.reasoning.ends_with('\n') {
            state.reasoning.push('\n');
        } else {
            state.reasoning.push_str("\n\n");
        }
    }
    state.reasoning_summary_index = Some(index);
    let _ = events.send(AgentEvent::ReasoningSummaryPartAdded { index });
}

fn start_web_search(
    id: &str,
    state: &mut ResponsesAccumulator,
    events: &mpsc::UnboundedSender<AgentEvent>,
) {
    if state.started_searches.insert(id.to_string()) {
        let _ = events.send(AgentEvent::WebSearchStarted { id: id.to_string() });
    }
}

fn append_citations(
    citations: Vec<(String, String)>,
    state: &mut ResponsesAccumulator,
    events: &mpsc::UnboundedSender<AgentEvent>,
) {
    let sources = citations
        .into_iter()
        .filter(|(url, _)| state.cited_urls.insert(url.clone()))
        .map(|(url, title)| {
            let title = title.replace([']', '\n', '\r'], " ");
            format!("- [{title}]({url})")
        })
        .collect::<Vec<_>>();
    if sources.is_empty() {
        return;
    }
    let text = format!("\n\nSources:\n{}", sources.join("\n"));
    state.content.push_str(&text);
    let _ = events.send(AgentEvent::TextDelta { text });
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn build_http_client(
    endpoint: &Url,
    idle_timeout: Duration,
) -> std::result::Result<Client, reqwest::Error> {
    let mut builder = Client::builder()
        .connect_timeout(idle_timeout.min(Duration::from_secs(20)))
        .read_timeout(idle_timeout)
        .user_agent(crate::USER_AGENT);
    if endpoint.host_str().is_some_and(is_loopback_host) {
        builder = builder.no_proxy();
    }
    builder.build()
}

fn response_request_id(headers: &HeaderMap) -> Option<String> {
    ["x-request-id", "request-id", "openai-request-id"]
        .into_iter()
        .find_map(|name| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
        })
}

fn retry_after_delay(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get("retry-after")?.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let retry_at = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    let retry_at = std::time::SystemTime::from(retry_at);
    retry_at.duration_since(std::time::SystemTime::now()).ok()
}

fn is_retryable_status(status: StatusCode) -> bool {
    status.as_u16() == 429 || status.is_server_error()
}

fn retry_backoff(attempt: usize) -> Duration {
    let exponent = u32::try_from(attempt).unwrap_or(u32::MAX).min(8);
    Duration::from_millis(200_u64.saturating_mul(2_u64.saturating_pow(exponent)))
}

async fn wait_before_retry(delay: Duration, cancel: &CancellationToken) -> Result<()> {
    let delay = delay.min(Duration::from_mins(1));
    tokio::select! {
        () = cancel.cancelled() => Err(OpenAiError::Cancelled),
        () = tokio::time::sleep(delay) => Ok(()),
    }
}

fn stream_error(message: &str, request_id: Option<&str>) -> OpenAiError {
    OpenAiError::Stream {
        message: message.to_string(),
        request_id: request_id.map(ToString::to_string),
    }
}

fn attach_request_id(error: OpenAiError, request_id: Option<&str>) -> OpenAiError {
    let Some(request_id) = request_id else {
        return error;
    };
    match error {
        OpenAiError::Protocol(message) => {
            OpenAiError::Protocol(format!("{message} (request id: {request_id})"))
        }
        error => error,
    }
}

fn stream_transport_error(error: &reqwest::Error, request_id: Option<&str>) -> OpenAiError {
    OpenAiError::Stream {
        message: format!("failed to read OpenAI stream: {error}"),
        request_id: request_id.map(ToString::to_string),
    }
}

#[derive(Debug, Deserialize)]
struct CachedTokenDetails {
    cached_tokens: Option<u64>,
}
fn api_endpoint_url(base_url: &str) -> Result<Url> {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(OpenAiError::Protocol(
            "base URL cannot be empty".to_string(),
        ));
    }
    Ok(Url::parse(&format!("{trimmed}/responses"))?)
}

fn summarize_error_body(body: &str) -> String {
    let body = body.trim();
    if body.is_empty() {
        return "响应正文为空".to_string();
    }

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        let message = value
            .pointer("/error/message")
            .and_then(serde_json::Value::as_str)
            .or_else(|| value.get("message").and_then(serde_json::Value::as_str))
            .or_else(|| value.get("detail").and_then(serde_json::Value::as_str))
            .or_else(|| value.get("error").and_then(serde_json::Value::as_str));
        if let Some(message) = message {
            let code = value
                .pointer("/error/code")
                .or_else(|| value.pointer("/error/type"))
                .or_else(|| value.get("code"))
                .and_then(|code| {
                    code.as_str()
                        .map(str::to_string)
                        .or_else(|| code.as_i64().map(|code| code.to_string()))
                });
            let summary = code.map_or_else(
                || message.to_string(),
                |code| format!("{message}（代码：{code}）"),
            );
            return truncate_error_summary(&collapse_whitespace(&summary));
        }
        return truncate_error_summary(&collapse_whitespace(&value.to_string()));
    }

    if looks_like_html(body) {
        return summarize_html_error(body);
    }

    truncate_error_summary(&collapse_whitespace(body))
}

fn looks_like_html(body: &str) -> bool {
    let prefix = body
        .chars()
        .take(256)
        .collect::<String>()
        .to_ascii_lowercase();
    prefix.contains("<!doctype html") || prefix.contains("<html") || prefix.contains("<head")
}

fn summarize_html_error(body: &str) -> String {
    let document = Document::from(body);
    let mut parts = ["h1", "h2"]
        .into_iter()
        .map(|selector| collapse_whitespace(document.select(selector).first().text().as_ref()))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        let title = collapse_whitespace(document.select("title").first().text().as_ref());
        if !title.is_empty() {
            parts.push(title);
        }
    }
    let description = if parts.is_empty() {
        "服务器返回了 HTML 错误页".to_string()
    } else {
        parts.join(" · ")
    };
    let prefix = if body.to_ascii_lowercase().contains("cloudflare") {
        "Cloudflare 拒绝了请求"
    } else {
        "HTML 错误页"
    };
    truncate_error_summary(&format!("{prefix}：{description}"))
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_error_summary(text: &str) -> String {
    const MAX_CHARS: usize = 500;
    let mut chars = text.chars();
    let prefix = chars.by_ref().take(MAX_CHARS).collect::<String>();
    if chars.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

fn is_context_overflow_text(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "context length exceeded",
        "context_length_exceeded",
        "exceeds the context window",
        "maximum context length",
        "max context length",
        "context window exceeded",
        "prompt is too long",
        "prompt too long",
        "input is too long",
        "input length",
        "too many tokens",
    ]
    .iter()
    .any(|pattern| message.contains(pattern))
}

#[derive(Debug, Default)]
struct SseDecoder {
    pending: Vec<u8>,
    data_lines: Vec<String>,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        self.pending.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(newline) = self.pending.iter().position(|byte| byte.eq(&b'\n')) {
            let mut line = self.pending.drain(..=newline).collect::<Vec<_>>();
            line.pop();
            if line.last().is_some_and(|byte| byte.eq(&b'\r')) {
                line.pop();
            }
            let line = str::from_utf8(&line).map_err(|error| {
                OpenAiError::Protocol(format!("SSE line is not valid UTF-8: {error}"))
            })?;
            let owned = line.to_string();
            self.apply_line(&owned, &mut events);
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<String>> {
        let mut events = Vec::new();
        if !self.pending.is_empty() {
            let line = str::from_utf8(&self.pending).map_err(|error| {
                OpenAiError::Protocol(format!("SSE tail is not valid UTF-8: {error}"))
            })?;
            let owned = line.trim_end_matches('\r').to_string();
            self.pending.clear();
            self.apply_line(&owned, &mut events);
        }
        self.flush_event(&mut events);
        Ok(events)
    }

    fn apply_line(&mut self, line: &str, events: &mut Vec<String>) {
        if line.is_empty() {
            self.flush_event(events);
        } else if let Some(data) = line.strip_prefix("data:") {
            self.data_lines
                .push(data.strip_prefix(' ').unwrap_or(data).to_string());
        }
    }

    fn flush_event(&mut self, events: &mut Vec<String>) {
        if !self.data_lines.is_empty() {
            events.push(self.data_lines.join("\n"));
            self.data_lines.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizes_structured_and_html_error_bodies() {
        let json = r#"{"error":{"message":"Invalid API key","code":"invalid_api_key"}}"#;
        assert_eq!(
            summarize_error_body(json),
            "Invalid API key（代码：invalid_api_key）"
        );

        let html = r"<!DOCTYPE html><html><head><title>Attention Required! | Cloudflare</title></head><body><h1>Sorry, you have been blocked</h1><h2>You are unable to access mengying.eu.org</h2><script>ignored</script></body></html>";
        let summary = summarize_error_body(html);
        assert_eq!(
            summary,
            "Cloudflare 拒绝了请求：Sorry, you have been blocked · You are unable to access mengying.eu.org"
        );
        assert!(!summary.contains("<html>"));
        assert!(!summary.contains("ignored"));
    }

    #[test]
    fn builds_responses_web_search_tool() {
        assert_eq!(
            serde_json::to_value(responses_tools(&[], true)).unwrap(),
            serde_json::json!([{"type": "web_search"}])
        );
        assert!(responses_tools(&[], false).is_empty());
    }

    #[test]
    fn classifies_only_max_output_responses_as_length_stops() {
        let (events, _receiver) = mpsc::unbounded_channel();
        let mut state = ResponsesAccumulator::default();
        apply_responses_stream_event(
            r#"{"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":200,"output_tokens":30,"total_tokens":230},"output":[]}}"#,
            &mut state,
            &events,
        )
        .unwrap();
        assert!(state.completed);
        assert_eq!(state.stop_reason, Some(AssistantStopReason::Length));
        assert_eq!(state.usage.unwrap().completion_tokens, 30);

        let mut state = ResponsesAccumulator::default();
        let error = apply_responses_stream_event(
            r#"{"type":"response.incomplete","response":{"incomplete_details":{"reason":"content_filter"},"output":[]}}"#,
            &mut state,
            &events,
        )
        .unwrap_err();
        assert!(error.to_string().contains("content_filter"));
    }

    #[test]
    fn parses_cached_tokens_from_responses_usage() {
        let (events, _receiver) = mpsc::unbounded_channel();
        let mut state = ResponsesAccumulator::default();
        apply_responses_stream_event(
            r#"{"type":"response.completed","response":{"usage":{"input_tokens":200,"output_tokens":30,"total_tokens":230,"input_tokens_details":{"cached_tokens":160}},"output":[]}}"#,
            &mut state,
            &events,
        )
        .unwrap();
        let usage = state.usage.unwrap();
        assert_eq!(usage.cached_prompt_tokens, Some(160));
        assert_eq!(usage.saturating_add(usage).cached_prompt_tokens, Some(320));
    }
}
