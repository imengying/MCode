use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::config::{AppConfig, ModelProfile, ReasoningEffort, find_model_profile};
use crate::event::AgentEvent;
use crate::openai::{AssistantTurn, OpenAiClient, OpenAiError};
use crate::protocol::{ChatMessage, ImageAttachment, Usage};
use crate::session::Session;
use crate::tools::ToolRegistry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Completed,
    Cancelled,
}

pub struct Agent {
    client: OpenAiClient,
    model_profiles: Vec<ModelProfile>,
    provider: Option<String>,
    reasoning_effort: ReasoningEffort,
    context_window: u64,
    context_tokens: u64,
    usage_estimated: bool,
    tools: ToolRegistry,
    session: Session,
    system_prompt: String,
    max_tool_turns: usize,
    total_usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelChoice {
    pub provider: String,
    pub id: String,
    pub name: Option<String>,
    pub context_window: u64,
    pub reasoning: bool,
}

impl Agent {
    pub async fn new(config: &AppConfig, session: Session) -> Result<Self> {
        let client = OpenAiClient::new(
            &config.base_url,
            config.api_key.clone(),
            &config.model,
            config.reasoning_value.clone(),
            config.supports_reasoning_effort,
            config.supports_usage_in_streaming,
            Duration::from_secs(config.request_timeout_secs),
        )?;
        let tools = ToolRegistry::with_mcp(&config.cwd, &config.mcp_servers).await?;
        let system_prompt = build_system_prompt(&config.cwd);
        Ok(Self {
            client,
            model_profiles: config.model_profiles.clone(),
            provider: config.provider.clone(),
            reasoning_effort: config.reasoning_effort,
            context_window: config.context_window,
            context_tokens: 0,
            usage_estimated: false,
            tools,
            session,
            system_prompt,
            max_tool_turns: config.max_tool_turns,
            total_usage: Usage::default(),
        })
    }

    pub async fn run(
        &mut self,
        prompt: &str,
        images: Vec<ImageAttachment>,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> Result<RunStatus> {
        if prompt.trim().is_empty() {
            bail!("prompt cannot be empty");
        }
        let _ = events.send(AgentEvent::RunStarted);
        self.session
            .append(ChatMessage::user_with_images(prompt, images))?;
        let definitions = self.tools.definitions().to_vec();

        for _ in 0..self.max_tool_turns {
            if cancel.is_cancelled() {
                let _ = events.send(AgentEvent::Cancelled);
                return Ok(RunStatus::Cancelled);
            }

            let _ = events.send(AgentEvent::AssistantStarted);
            let mut context = Vec::with_capacity(self.session.messages().len() + 1);
            context.push(ChatMessage::system(&self.system_prompt));
            context.extend_from_slice(self.session.messages());
            let turn = match self
                .client
                .stream_chat(&context, &definitions, events, cancel)
                .await
            {
                Ok(turn) => turn,
                Err(OpenAiError::Cancelled) => {
                    let _ = events.send(AgentEvent::Cancelled);
                    return Ok(RunStatus::Cancelled);
                }
                Err(error) => return Err(error.into()),
            };

            let (usage, estimated) = turn.usage.map_or_else(
                || (estimate_usage(&context, &turn), true),
                |usage| (normalize_usage(usage), false),
            );
            self.total_usage.prompt_tokens = self
                .total_usage
                .prompt_tokens
                .saturating_add(usage.prompt_tokens);
            self.total_usage.completion_tokens = self
                .total_usage
                .completion_tokens
                .saturating_add(usage.completion_tokens);
            self.total_usage.total_tokens = self
                .total_usage
                .total_tokens
                .saturating_add(usage.total_tokens);
            self.context_tokens = usage.total_tokens;
            self.usage_estimated = estimated;
            let _ = events.send(AgentEvent::Usage {
                usage,
                context_tokens: self.context_tokens,
                context_window: self.context_window,
                estimated,
            });

            if turn.content.is_none() && turn.tool_calls.is_empty() {
                return Err(anyhow!("provider completed without text or tool calls"));
            }
            let tool_calls = turn.tool_calls.clone();
            self.session.append(ChatMessage::assistant(
                turn.content,
                turn.reasoning_content,
                turn.tool_calls,
            ))?;

            if tool_calls.is_empty() {
                let _ = events.send(AgentEvent::RunFinished);
                return Ok(RunStatus::Completed);
            }

            for call in tool_calls {
                let _ = events.send(AgentEvent::ToolStarted {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    arguments: call.function.arguments.clone(),
                });
                let execution = self.tools.execute(&call, cancel).await;
                let _ = events.send(AgentEvent::ToolFinished {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    output: execution.output.clone(),
                    is_error: execution.is_error,
                });
                self.session
                    .append(ChatMessage::tool(call.id, execution.output))?;

                if cancel.is_cancelled() {
                    let _ = events.send(AgentEvent::Cancelled);
                    return Ok(RunStatus::Cancelled);
                }
            }
        }

        Err(anyhow!(
            "agent exceeded the maximum of {} tool turns",
            self.max_tool_turns
        ))
    }

    #[must_use]
    pub fn messages(&self) -> &[ChatMessage] {
        self.session.messages()
    }

    #[must_use]
    pub fn model(&self) -> &str {
        self.client.model()
    }

    #[must_use]
    pub fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }

    #[must_use]
    pub const fn reasoning_effort(&self) -> ReasoningEffort {
        self.reasoning_effort
    }

    #[must_use]
    pub const fn context_window(&self) -> u64 {
        self.context_window
    }

    #[must_use]
    pub const fn context_tokens(&self) -> u64 {
        self.context_tokens
    }

    #[must_use]
    pub const fn usage_estimated(&self) -> bool {
        self.usage_estimated
    }

    #[must_use]
    pub fn model_choices(&self) -> Vec<ModelChoice> {
        self.model_profiles
            .iter()
            .map(|profile| ModelChoice {
                provider: profile.provider.clone(),
                id: profile.id.clone(),
                name: profile.name.clone(),
                context_window: profile.context_window,
                reasoning: profile.reasoning,
            })
            .collect()
    }

    #[must_use]
    pub fn available_reasoning_efforts(&self) -> Vec<ReasoningEffort> {
        self.current_profile().map_or_else(
            || ReasoningEffort::ALL.to_vec(),
            ModelProfile::supported_reasoning_efforts,
        )
    }

    pub fn select_model(&mut self, query: &str) -> Result<()> {
        let profile =
            find_model_profile(&self.model_profiles, self.provider.as_deref(), query)?.cloned();
        if let Some(profile) = profile {
            let effective_effort = profile.clamp_reasoning_effort(self.reasoning_effort);
            let reasoning_value = profile.reasoning_value(effective_effort)?;
            self.client.reconfigure(
                &profile.base_url,
                profile.api_key.clone(),
                &profile.id,
                reasoning_value,
                profile.supports_reasoning_effort,
                profile.supports_usage_in_streaming,
            )?;
            self.provider = Some(profile.provider);
            self.reasoning_effort = effective_effort;
            self.context_window = profile.context_window;
        } else {
            let query = query.trim();
            if query.is_empty() {
                bail!("model cannot be empty");
            }
            self.client.set_model(query);
            self.client.set_reasoning_effort(
                (self.reasoning_effort != ReasoningEffort::Off)
                    .then(|| self.reasoning_effort.as_str().to_string()),
            );
        }
        self.session.set_model(self.client.model())?;
        self.session.set_reasoning_effort(self.reasoning_effort)?;
        self.context_tokens = self.estimated_context_tokens();
        self.usage_estimated = true;
        Ok(())
    }

    pub fn set_reasoning_effort(&mut self, effort: ReasoningEffort) -> Result<()> {
        let effective_effort = self
            .current_profile()
            .map_or(effort, |profile| profile.clamp_reasoning_effort(effort));
        let reasoning_value = self.current_profile().map_or_else(
            || {
                Ok((effective_effort != ReasoningEffort::Off)
                    .then(|| effective_effort.as_str().to_string()))
            },
            |profile| profile.reasoning_value(effective_effort),
        )?;
        self.client.set_reasoning_effort(reasoning_value);
        self.session.set_reasoning_effort(effective_effort)?;
        self.reasoning_effort = effective_effort;
        Ok(())
    }

    pub fn new_session(&mut self) -> Result<()> {
        self.session = self
            .session
            .fresh(self.client.model(), self.reasoning_effort)?;
        self.total_usage = Usage::default();
        self.context_tokens = 0;
        self.usage_estimated = false;
        Ok(())
    }

    pub fn delete_session(&mut self) -> Result<uuid::Uuid> {
        self.session.delete_current()
    }

    #[must_use]
    pub fn endpoint(&self) -> &str {
        self.client.endpoint().as_str()
    }

    #[must_use]
    pub fn session(&self) -> &Session {
        &self.session
    }

    #[must_use]
    pub fn total_usage(&self) -> Usage {
        self.total_usage
    }

    #[must_use]
    pub fn mcp_server_count(&self) -> usize {
        self.tools.mcp_server_count()
    }

    #[must_use]
    pub fn mcp_tool_count(&self) -> usize {
        self.tools.mcp_tool_count()
    }

    fn current_profile(&self) -> Option<&ModelProfile> {
        self.model_profiles.iter().find(|profile| {
            profile.id == self.client.model()
                && self
                    .provider
                    .as_deref()
                    .is_some_and(|provider| provider == profile.provider)
        })
    }

    fn estimated_context_tokens(&self) -> u64 {
        estimate_text_tokens(&self.system_prompt)
            .saturating_add(
                self.session
                    .messages()
                    .iter()
                    .map(estimate_message_tokens)
                    .sum::<u64>(),
            )
            .saturating_add(4)
    }
}

fn build_system_prompt(cwd: &std::path::Path) -> String {
    format!(
        "You are MCode, a focused coding agent running in a terminal.\n\
         Work directly in the user's repository and complete requested changes end to end.\n\
         Use read_file before editing unfamiliar code. Use edit_file for precise changes, \
         write_file for new or fully replaced files, and shell for searches, builds, and tests.\n\
         Prefer rg for text search when available. Keep changes scoped to the request.\n\
         Never claim a command succeeded unless you observed its output.\n\
         The working directory is {}. File tools cannot access paths outside it.",
        cwd.display()
    )
}

fn normalize_usage(mut usage: Usage) -> Usage {
    if usage.total_tokens == 0 {
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);
    }
    usage
}

fn estimate_usage(context: &[ChatMessage], turn: &AssistantTurn) -> Usage {
    let prompt_tokens = context
        .iter()
        .map(estimate_message_tokens)
        .sum::<u64>()
        .saturating_add(4);
    let completion_tokens = turn
        .content
        .as_deref()
        .map_or(0, estimate_text_tokens)
        .saturating_add(
            turn.reasoning_content
                .as_deref()
                .map_or(0, estimate_text_tokens),
        )
        .saturating_add(
            turn.tool_calls
                .iter()
                .map(|call| {
                    estimate_text_tokens(&call.function.name)
                        .saturating_add(estimate_text_tokens(&call.function.arguments))
                        .saturating_add(4)
                })
                .sum::<u64>(),
        )
        .max(1);
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    }
}

fn estimate_message_tokens(message: &ChatMessage) -> u64 {
    const IMAGE_TOKEN_ESTIMATE: u64 = 1_024;

    let mut tokens = 4_u64;
    if let Some(content) = &message.content {
        tokens = tokens.saturating_add(estimate_text_tokens(content));
    }
    if let Some(reasoning) = &message.reasoning_content {
        tokens = tokens.saturating_add(estimate_text_tokens(reasoning));
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        tokens = tokens.saturating_add(estimate_text_tokens(tool_call_id));
    }
    for call in &message.tool_calls {
        tokens = tokens
            .saturating_add(estimate_text_tokens(&call.function.name))
            .saturating_add(estimate_text_tokens(&call.function.arguments))
            .saturating_add(4);
    }
    tokens = tokens.saturating_add(
        u64::try_from(message.images.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(IMAGE_TOKEN_ESTIMATE),
    );
    tokens
}

fn estimate_text_tokens(text: &str) -> u64 {
    let characters = u64::try_from(text.chars().count()).unwrap_or(u64::MAX);
    characters.div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimate_is_nonzero_and_scales_with_text() {
        assert_eq!(estimate_text_tokens(""), 0);
        assert_eq!(estimate_text_tokens("abcd"), 1);
        assert_eq!(estimate_text_tokens("abcde"), 2);
        assert!(
            estimate_message_tokens(&ChatMessage::user("a longer message"))
                > estimate_message_tokens(&ChatMessage::user("a"))
        );
    }
}
