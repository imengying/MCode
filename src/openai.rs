use std::collections::BTreeMap;
use std::net::IpAddr;
use std::str;
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::event::AgentEvent;
use crate::protocol::{ChatMessage, FunctionCall, MessageRole, ToolCall, ToolDefinition, Usage};

const CODEX_USER_AGENT: &str = "codex_cli_rs/0.145.0";

#[derive(Debug, thiserror::Error)]
pub enum OpenAiError {
    #[error("request cancelled")]
    Cancelled,
    #[error("OpenAI request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid OpenAI URL: {0}")]
    Url(#[from] url::ParseError),
    #[error("OpenAI API returned HTTP {status}: {body}")]
    Api { status: u16, body: String },
    #[error("invalid OpenAI stream: {0}")]
    Protocol(String),
}

pub type Result<T> = std::result::Result<T, OpenAiError>;

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    http: Client,
    endpoint: Url,
    api_key: Option<String>,
    model: String,
    reasoning_effort: Option<String>,
    supports_reasoning_effort: bool,
    supports_usage_in_streaming: bool,
    timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantTurn {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
}

impl OpenAiClient {
    pub fn new(
        base_url: &str,
        api_key: Option<String>,
        model: impl Into<String>,
        reasoning_effort: Option<String>,
        supports_reasoning_effort: bool,
        supports_usage_in_streaming: bool,
        timeout: Duration,
    ) -> Result<Self> {
        let endpoint = chat_completions_url(base_url)?;
        let http = build_http_client(&endpoint)?;
        Ok(Self {
            http,
            endpoint,
            api_key,
            model: model.into(),
            reasoning_effort,
            supports_reasoning_effort,
            supports_usage_in_streaming,
            timeout,
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

    pub fn reconfigure(
        &mut self,
        base_url: &str,
        api_key: Option<String>,
        model: impl Into<String>,
        reasoning_effort: Option<String>,
        supports_reasoning_effort: bool,
        supports_usage_in_streaming: bool,
    ) -> Result<()> {
        let endpoint = chat_completions_url(base_url)?;
        self.http = build_http_client(&endpoint)?;
        self.endpoint = endpoint;
        self.api_key = api_key;
        self.model = model.into();
        self.reasoning_effort = reasoning_effort;
        self.supports_reasoning_effort = supports_reasoning_effort;
        self.supports_usage_in_streaming = supports_usage_in_streaming;
        Ok(())
    }

    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
    }

    pub fn set_reasoning_effort(&mut self, reasoning_effort: Option<String>) {
        self.reasoning_effort = reasoning_effort;
    }

    pub async fn stream_chat(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> Result<AssistantTurn> {
        let api_messages: Vec<ApiMessage<'_>> = messages.iter().map(ApiMessage::from).collect();
        let body = ChatRequest {
            model: &self.model,
            messages: &api_messages,
            tools,
            reasoning_effort: self
                .reasoning_effort
                .as_deref()
                .filter(|_| self.supports_reasoning_effort),
            stream: true,
            stream_options: self.supports_usage_in_streaming.then_some(StreamOptions {
                include_usage: true,
            }),
        };

        let mut request = self
            .http
            .post(self.endpoint.clone())
            .timeout(self.timeout)
            .header("Accept", "text/event-stream")
            .json(&body);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = tokio::select! {
            () = cancel.cancelled() => return Err(OpenAiError::Cancelled),
            response = request.send() => response?,
        };
        let status = response.status();
        if !status.is_success() {
            let body = tokio::select! {
                () = cancel.cancelled() => return Err(OpenAiError::Cancelled),
                text = response.text() => text.unwrap_or_else(|error| format!("<failed to read response: {error}>")),
            };
            return Err(OpenAiError::Api {
                status: status.as_u16(),
                body: truncate_error_body(&body),
            });
        }

        let mut bytes = response.bytes_stream();
        let mut decoder = SseDecoder::default();
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = BTreeMap::<usize, ToolCallBuilder>::new();
        let mut usage = None;
        let mut done = false;

        while !done {
            let chunk = tokio::select! {
                () = cancel.cancelled() => return Err(OpenAiError::Cancelled),
                chunk = bytes.next() => chunk,
            };
            let Some(chunk) = chunk else {
                break;
            };
            let chunk = chunk?;
            for data in decoder.push(&chunk)? {
                if data.trim() == "[DONE]" {
                    done = true;
                    break;
                }
                apply_stream_chunk(
                    &data,
                    &mut content,
                    &mut reasoning,
                    &mut tool_calls,
                    &mut usage,
                    events,
                )?;
            }
        }

        for data in decoder.finish()? {
            if data.trim() != "[DONE]" {
                apply_stream_chunk(
                    &data,
                    &mut content,
                    &mut reasoning,
                    &mut tool_calls,
                    &mut usage,
                    events,
                )?;
            }
        }

        let tool_calls = tool_calls
            .into_values()
            .map(ToolCallBuilder::finish)
            .collect::<Result<Vec<_>>>()?;

        Ok(AssistantTurn {
            content: (!content.is_empty()).then_some(content),
            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls,
            usage,
        })
    }
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn build_http_client(endpoint: &Url) -> std::result::Result<Client, reqwest::Error> {
    let mut builder = Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .user_agent(CODEX_USER_AGENT);
    if endpoint.host_str().is_some_and(is_loopback_host) {
        builder = builder.no_proxy();
    }
    builder.build()
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ApiMessage<'a>],
    #[serde(skip_serializing_if = "<[ToolDefinition]>::is_empty")]
    tools: &'a [ToolDefinition],
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct ApiMessage<'a> {
    role: MessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<ApiContent<'a>>,
    #[serde(skip_serializing_if = "<[ToolCall]>::is_empty")]
    tool_calls: &'a [ToolCall],
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ApiContent<'a> {
    Text(&'a str),
    Parts(Vec<ApiContentPart<'a>>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiContentPart<'a> {
    Text { text: &'a str },
    ImageUrl { image_url: ApiImageUrl },
}

#[derive(Debug, Serialize)]
struct ApiImageUrl {
    url: String,
}

impl<'a> From<&'a ChatMessage> for ApiMessage<'a> {
    fn from(message: &'a ChatMessage) -> Self {
        let content = if message.images.is_empty() {
            message.content.as_deref().map(ApiContent::Text)
        } else {
            let mut parts = Vec::with_capacity(message.images.len() + 1);
            if let Some(text) = message.content.as_deref().filter(|text| !text.is_empty()) {
                parts.push(ApiContentPart::Text { text });
            }
            parts.extend(message.images.iter().map(|image| ApiContentPart::ImageUrl {
                image_url: ApiImageUrl {
                    url: image.data_url(),
                },
            }));
            Some(ApiContent::Parts(parts))
        };
        Self {
            role: message.role,
            content,
            tool_calls: &message.tool_calls,
            tool_call_id: message.tool_call_id.as_deref(),
            reasoning_content: message.reasoning_content.as_deref(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    usage: Option<Usage>,
    error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
}

#[derive(Debug, Default, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct ToolCallBuilder {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallBuilder {
    fn apply(&mut self, delta: ToolCallDelta) {
        if let Some(id) = delta.id {
            self.id.push_str(&id);
        }
        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                self.name.push_str(&name);
            }
            if let Some(arguments) = function.arguments {
                self.arguments.push_str(&arguments);
            }
        }
    }

    fn finish(self) -> Result<ToolCall> {
        if self.id.is_empty() {
            return Err(OpenAiError::Protocol(
                "tool call completed without an id".to_string(),
            ));
        }
        if self.name.is_empty() {
            return Err(OpenAiError::Protocol(format!(
                "tool call {} completed without a function name",
                self.id
            )));
        }
        Ok(ToolCall {
            id: self.id,
            kind: "function".to_string(),
            function: FunctionCall {
                name: self.name,
                arguments: self.arguments,
            },
        })
    }
}

fn apply_stream_chunk(
    data: &str,
    content: &mut String,
    reasoning: &mut String,
    tool_calls: &mut BTreeMap<usize, ToolCallBuilder>,
    usage: &mut Option<Usage>,
    events: &mpsc::UnboundedSender<AgentEvent>,
) -> Result<()> {
    let chunk: StreamChunk = serde_json::from_str(data)
        .map_err(|error| OpenAiError::Protocol(format!("invalid JSON event: {error}")))?;
    if let Some(error) = chunk.error {
        return Err(OpenAiError::Protocol(format!(
            "provider returned a stream error: {error}"
        )));
    }
    if let Some(next_usage) = chunk.usage {
        *usage = Some(next_usage);
    }
    for choice in chunk.choices {
        if let Some(text) = choice.delta.content {
            content.push_str(&text);
            let _ = events.send(AgentEvent::TextDelta { text });
        }
        if let Some(text) = choice.delta.reasoning_content.or(choice.delta.reasoning) {
            reasoning.push_str(&text);
            let _ = events.send(AgentEvent::ReasoningDelta { text });
        }
        for delta in choice.delta.tool_calls {
            tool_calls.entry(delta.index).or_default().apply(delta);
        }
    }
    Ok(())
}

fn chat_completions_url(base_url: &str) -> Result<Url> {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(OpenAiError::Protocol(
            "base URL cannot be empty".to_string(),
        ));
    }
    let endpoint = if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    };
    Ok(Url::parse(&endpoint)?)
}

fn truncate_error_body(body: &str) -> String {
    const MAX_CHARS: usize = 8_000;
    let mut chars = body.chars();
    let truncated: String = chars.by_ref().take(MAX_CHARS).collect();
    if chars.next().is_some() {
        format!("{truncated}\n... response truncated")
    } else {
        truncated
    }
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
    fn builds_chat_completion_endpoint() {
        assert_eq!(
            chat_completions_url("https://api.example.test/v1/")
                .unwrap()
                .as_str(),
            "https://api.example.test/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://api.example.test/v1/chat/completions")
                .unwrap()
                .as_str(),
            "https://api.example.test/v1/chat/completions"
        );
    }

    #[test]
    fn decodes_sse_across_arbitrary_chunks() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"da").unwrap().is_empty());
        assert_eq!(
            decoder
                .push(b"ta: {\"choices\":[]}\r\n\r\ndata: [DO")
                .unwrap(),
            vec!["{\"choices\":[]}"]
        );
        assert_eq!(decoder.push(b"NE]\n\n").unwrap(), vec!["[DONE]"]);
    }

    #[test]
    fn combines_fragmented_tool_calls() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls = BTreeMap::new();
        let mut usage = None;
        apply_stream_chunk(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_","function":{"name":"read_","arguments":"{\"pa"}}]}}]}"#,
            &mut content,
            &mut reasoning,
            &mut calls,
            &mut usage,
            &tx,
        )
        .unwrap();
        apply_stream_chunk(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"1","function":{"name":"file","arguments":"th\":\"README.md\"}"}}]}}]}"#,
            &mut content,
            &mut reasoning,
            &mut calls,
            &mut usage,
            &tx,
        )
        .unwrap();
        let call = calls.into_values().next().unwrap().finish().unwrap();
        assert_eq!(call.id, "call_1");
        assert_eq!(call.function.name, "read_file");
        assert_eq!(call.function.arguments, r#"{"path":"README.md"}"#);
    }

    #[test]
    fn serializes_images_as_chat_completion_content_parts() {
        let message = ChatMessage::user_with_images(
            "inspect this",
            vec![crate::protocol::ImageAttachment {
                name: "screen.png".to_string(),
                mime_type: "image/png".to_string(),
                data: "AQID".to_string(),
            }],
        );
        let encoded = serde_json::to_value(ApiMessage::from(&message)).unwrap();
        assert_eq!(encoded["content"][0]["type"], "text");
        assert_eq!(encoded["content"][0]["text"], "inspect this");
        assert_eq!(encoded["content"][1]["type"], "image_url");
        assert_eq!(
            encoded["content"][1]["image_url"]["url"],
            "data:image/png;base64,AQID"
        );
    }
}
