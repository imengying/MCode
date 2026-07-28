use std::collections::BTreeSet;
use std::fs;
use std::io::ErrorKind;
use std::ops::Range;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::approval::{ApprovalDecision, ApprovalGate};
use crate::compaction::{
    CompactionPreparation, SUMMARIZATION_SYSTEM_PROMPT, append_file_operations, combine_usage,
    estimate_message_tokens, estimate_text_tokens, history_summary_request, prepare_compaction,
    should_compact, turn_prefix_summary_request,
};
use crate::config::{
    ApiProtocol, AppConfig, CompactionSettings, ModelProfile, ReasoningEffort, WebSearchMode,
    find_model_profile,
};
use crate::event::{AgentEvent, CompactionReason};
use crate::openai::{AssistantTurn, OpenAiClient, OpenAiError, OpenAiModelConfig};
use crate::protocol::{ChatMessage, ImageAttachment, MessageRole, ToolDefinition, Usage};
use crate::session::{PendingToolCall, RunOutcome, Session, ToolReplayPolicy};
use crate::tools::{McpStartupFailure, ToolRegistry};

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
    max_input_tokens: u64,
    context_tokens: u64,
    compaction_settings: CompactionSettings,
    usage_estimated: bool,
    tools: ToolRegistry,
    session: Session,
    system_prompt: String,
    max_tool_turns: usize,
    total_usage: Usage,
    approved_tools: BTreeSet<String>,
    last_context_dropped: usize,
    auto_compaction_failed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelChoice {
    pub provider: String,
    pub id: String,
    pub name: Option<String>,
    pub api: ApiProtocol,
    pub context_window: u64,
    pub max_input_tokens: u64,
    pub reasoning: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionResult {
    pub summary: String,
    pub first_kept_message_index: usize,
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub usage: Option<Usage>,
}

impl Agent {
    pub async fn new(config: &AppConfig, mut session: Session) -> Result<Self> {
        session.set_model(config.provider.as_deref(), &config.model, config.api)?;
        session.set_reasoning_effort(config.reasoning_effort)?;
        session.set_web_search_mode(config.web_search.mode)?;
        let client = OpenAiClient::new(
            OpenAiModelConfig {
                base_url: config.base_url.clone(),
                api_key: config.api_key.clone(),
                model: config.model.clone(),
                api: config.api,
                reasoning_effort: config.reasoning_value.clone(),
                supports_reasoning_effort: config.supports_reasoning_effort,
                supports_usage_in_streaming: config.supports_usage_in_streaming,
                supports_strict_tools: config.supports_strict_tools,
            },
            config.web_search.clone(),
            Duration::from_secs(config.request_timeout_secs),
        )?;
        let tools = ToolRegistry::with_mcp(
            &config.cwd,
            &config.mcp_servers,
            config.web_search.clone(),
            config.api,
        )
        .await?;
        let system_prompt = build_system_prompt(&config.cwd)?;
        let total_usage = session.total_usage();
        let mut agent = Self {
            client,
            model_profiles: config.model_profiles.clone(),
            provider: config.provider.clone(),
            reasoning_effort: config.reasoning_effort,
            context_window: config.context_window,
            max_input_tokens: config.max_input_tokens,
            context_tokens: 0,
            compaction_settings: config.compaction,
            usage_estimated: false,
            tools,
            session,
            system_prompt,
            max_tool_turns: config.max_tool_turns,
            total_usage,
            approved_tools: BTreeSet::new(),
            last_context_dropped: 0,
            auto_compaction_failed: false,
        };
        if !agent.session.messages().is_empty() {
            agent.context_tokens = agent.estimated_context_tokens();
            agent.usage_estimated = true;
        }
        Ok(agent)
    }

    pub async fn run(
        &mut self,
        prompt: &str,
        images: Vec<ImageAttachment>,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        approvals: &ApprovalGate,
    ) -> Result<RunStatus> {
        if prompt.trim().is_empty() {
            bail!("prompt cannot be empty");
        }
        if self.session.has_pending_run() {
            bail!("session has an unfinished run; resume it before starting another prompt");
        }
        let _ = events.send(AgentEvent::RunStarted);
        self.auto_compaction_failed = false;
        let _ = self
            .maybe_auto_compact(CompactionReason::Threshold, events, cancel)
            .await;
        if cancel.is_cancelled() {
            let _ = events.send(AgentEvent::Cancelled);
            return Ok(RunStatus::Cancelled);
        }
        let run_id = self
            .session
            .begin_run(ChatMessage::user_with_images(prompt, images))?;
        let result = self.drive_run(run_id, events, cancel, approvals).await;
        self.finish_run_result(run_id, result, events)
    }

    pub async fn resume_pending(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        approvals: &ApprovalGate,
    ) -> Result<RunStatus> {
        let run_id = self
            .session
            .active_run_id()
            .ok_or_else(|| anyhow!("session has no unfinished run"))?;
        let _ = events.send(AgentEvent::RunResumed);
        self.auto_compaction_failed = false;

        let result = if self.session.active_run_has_final_response() {
            Ok(RunStatus::Completed)
        } else {
            match self
                .execute_tool_calls(
                    run_id,
                    self.session.pending_tool_calls()?,
                    events,
                    cancel,
                    approvals,
                )
                .await?
            {
                RunStatus::Cancelled => Ok(RunStatus::Cancelled),
                RunStatus::Completed => self.drive_run(run_id, events, cancel, approvals).await,
            }
        };
        self.finish_run_result(run_id, result, events)
    }

    async fn drive_run(
        &mut self,
        run_id: uuid::Uuid,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        approvals: &ApprovalGate,
    ) -> Result<RunStatus> {
        let definitions = self.tools.definitions().to_vec();
        let mut overflow_recovery_attempted = false;

        while self.session.active_run_completed_steps() < self.max_tool_turns {
            if cancel.is_cancelled() {
                return Ok(RunStatus::Cancelled);
            }

            let (turn, context) = loop {
                let _ = self
                    .maybe_auto_compact(CompactionReason::Threshold, events, cancel)
                    .await;
                if cancel.is_cancelled() {
                    return Ok(RunStatus::Cancelled);
                }

                let active_messages = self.session.context_messages();
                let selection = select_context(
                    &self.system_prompt,
                    &active_messages,
                    &definitions,
                    self.max_input_tokens,
                )?;
                if selection.dropped_messages > 0
                    && selection.dropped_messages != self.last_context_dropped
                {
                    let _ = events.send(AgentEvent::ContextTrimmed {
                        dropped_messages: selection.dropped_messages,
                        dropped_turns: selection.dropped_turns,
                        estimated_tokens: selection.estimated_tokens,
                    });
                }
                self.last_context_dropped = selection.dropped_messages;
                let context = selection.messages;
                self.session.start_generation(
                    run_id,
                    self.provider.as_deref(),
                    self.client.model(),
                    self.client.api(),
                )?;
                let _ = events.send(AgentEvent::AssistantStarted);
                match self
                    .client
                    .stream_chat(&context, &definitions, events, cancel)
                    .await
                {
                    Ok(turn) => break (turn, context),
                    Err(OpenAiError::Cancelled) => return Ok(RunStatus::Cancelled),
                    Err(error) if error.is_context_overflow() && !overflow_recovery_attempted => {
                        overflow_recovery_attempted = true;
                        if self
                            .compact_for_reason(None, CompactionReason::Overflow, events, cancel)
                            .await
                            .is_ok()
                        {
                            continue;
                        }
                        return Err(error.into());
                    }
                    Err(error) => return Err(error.into()),
                }
            };

            if turn.content.is_none() && turn.tool_calls.is_empty() {
                return Err(anyhow!("provider completed without text or tool calls"));
            }
            let (usage, estimated) = turn.usage.map_or_else(
                || (estimate_usage(&context, &definitions, &turn), true),
                |usage| (normalize_usage(usage), false),
            );
            let tool_calls = turn.tool_calls.clone();
            self.session.complete_generation(
                run_id,
                ChatMessage::assistant_with_response_items(
                    turn.content,
                    turn.reasoning_content,
                    turn.tool_calls,
                    turn.response_items,
                ),
                usage,
                estimated,
            )?;
            add_usage(&mut self.total_usage, usage);
            self.context_tokens = usage.total_tokens;
            self.usage_estimated = estimated;
            let _ = events.send(AgentEvent::Usage {
                usage,
                context_tokens: self.context_tokens,
                context_window: self.context_window,
                max_input_tokens: self.max_input_tokens,
                estimated,
            });

            if tool_calls.is_empty() {
                let _ = self
                    .maybe_auto_compact(CompactionReason::Threshold, events, cancel)
                    .await;
                if cancel.is_cancelled() {
                    return Ok(RunStatus::Cancelled);
                }
                return Ok(RunStatus::Completed);
            }
            let pending = tool_calls
                .into_iter()
                .map(|call| PendingToolCall { call, intent: None })
                .collect();
            if self
                .execute_tool_calls(run_id, pending, events, cancel, approvals)
                .await?
                == RunStatus::Cancelled
            {
                return Ok(RunStatus::Cancelled);
            }
        }

        Err(anyhow!(
            "agent exceeded the maximum of {} tool turns",
            self.max_tool_turns
        ))
    }

    async fn execute_tool_calls(
        &mut self,
        run_id: uuid::Uuid,
        pending: Vec<PendingToolCall>,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        approvals: &ApprovalGate,
    ) -> Result<RunStatus> {
        for pending in pending {
            let call = pending.call;
            let recovered_intent = pending.intent;
            let was_recovered = recovered_intent.is_some();
            let requires_approval = recovered_intent.is_none()
                && self.tools.requires_approval(&call.function.name)
                && !self.approved_tools.contains(&call.function.name)
                && !approvals.bypasses_approval();
            if requires_approval {
                let _ = events.send(AgentEvent::ApprovalRequested {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    arguments: call.function.arguments.clone(),
                });
                let Some(decision) = approvals
                    .decide(
                        &call.id,
                        &call.function.name,
                        &call.function.arguments,
                        cancel,
                    )
                    .await
                else {
                    return Ok(RunStatus::Cancelled);
                };
                let approved = matches!(
                    decision,
                    ApprovalDecision::ApproveOnce | ApprovalDecision::ApproveForSession
                );
                let for_session = decision == ApprovalDecision::ApproveForSession;
                let _ = events.send(AgentEvent::ApprovalResolved {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    approved,
                    for_session,
                });
                if for_session {
                    self.approved_tools.insert(call.function.name.clone());
                }
                if !approved {
                    let output = format!(
                        "tool execution denied: {} requires user approval",
                        call.function.name
                    );
                    self.session
                        .append(ChatMessage::tool(call.id.clone(), output.clone()))?;
                    let _ = events.send(AgentEvent::ToolFinished {
                        id: call.id,
                        name: call.function.name,
                        output,
                        is_error: true,
                    });
                    continue;
                }
            }

            let _ = events.send(AgentEvent::ToolStarted {
                id: call.id.clone(),
                name: call.function.name.clone(),
                arguments: call.function.arguments.clone(),
            });
            let intent = match recovered_intent {
                Some(intent) => intent,
                None => self.session.start_tool(
                    run_id,
                    call.clone(),
                    self.tools.replay_policy(&call.function.name),
                )?,
            };
            let execution = if intent.replay == ToolReplayPolicy::Never && was_recovered {
                crate::tools::ToolExecution {
                    output: format!(
                        "tool error: {} may have run before MCode was interrupted; it was not replayed",
                        call.function.name
                    ),
                    is_error: true,
                }
            } else {
                self.tools.execute(&call, cancel).await
            };
            self.session.complete_tool(
                &intent,
                ChatMessage::tool(call.id.clone(), execution.output.clone()),
            )?;
            let _ = events.send(AgentEvent::ToolFinished {
                id: call.id,
                name: call.function.name,
                output: execution.output,
                is_error: execution.is_error,
            });

            if cancel.is_cancelled() {
                return Ok(RunStatus::Cancelled);
            }
        }
        Ok(RunStatus::Completed)
    }

    fn finish_run_result(
        &mut self,
        run_id: uuid::Uuid,
        result: Result<RunStatus>,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<RunStatus> {
        match result {
            Ok(RunStatus::Completed) => {
                self.session.finish_run(run_id, RunOutcome::Completed)?;
                let _ = events.send(AgentEvent::RunFinished);
                Ok(RunStatus::Completed)
            }
            Ok(RunStatus::Cancelled) => {
                self.session.finish_run(run_id, RunOutcome::Cancelled)?;
                let _ = events.send(AgentEvent::Cancelled);
                Ok(RunStatus::Cancelled)
            }
            Err(error) => {
                if let Err(finish_error) = self.session.finish_run(run_id, RunOutcome::Failed) {
                    return Err(error.context(format!(
                        "the run also could not be closed durably: {finish_error:#}"
                    )));
                }
                Err(error)
            }
        }
    }

    pub async fn compact(
        &mut self,
        custom_instructions: Option<&str>,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> Result<CompactionResult> {
        self.compact_for_reason(
            custom_instructions,
            CompactionReason::Manual,
            events,
            cancel,
        )
        .await
    }

    async fn maybe_auto_compact(
        &mut self,
        reason: CompactionReason,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> bool {
        if self.auto_compaction_failed || self.session.is_compacted_at_tip() {
            return false;
        }
        let context_tokens = self.estimated_context_tokens();
        self.context_tokens = context_tokens;
        self.usage_estimated = true;
        if !should_compact(
            context_tokens,
            self.max_input_tokens,
            self.compaction_settings,
        ) {
            return false;
        }
        if self
            .compact_for_reason(None, reason, events, cancel)
            .await
            .is_ok()
        {
            true
        } else {
            self.auto_compaction_failed = true;
            false
        }
    }

    async fn compact_for_reason(
        &mut self,
        custom_instructions: Option<&str>,
        reason: CompactionReason,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> Result<CompactionResult> {
        let _ = events.send(AgentEvent::CompactionStarted { reason });
        let result = self.perform_compaction(custom_instructions, cancel).await;
        match result {
            Ok(result) => {
                let _ = events.send(AgentEvent::CompactionFinished {
                    reason,
                    summary: result.summary.clone(),
                    first_kept_message_index: result.first_kept_message_index,
                    tokens_before: result.tokens_before,
                    tokens_after: result.tokens_after,
                    usage: result.usage,
                });
                Ok(result)
            }
            Err(error) => {
                let _ = events.send(AgentEvent::CompactionFailed {
                    reason,
                    message: format!("{error:#}"),
                });
                Err(error)
            }
        }
    }

    async fn perform_compaction(
        &mut self,
        custom_instructions: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<CompactionResult> {
        if cancel.is_cancelled() {
            bail!("compaction cancelled");
        }
        let tokens_before = self.estimated_context_tokens();
        let preparation =
            prepare_compaction(&self.session, self.compaction_settings, tokens_before).ok_or_else(
                || {
                    if self.session.is_compacted_at_tip() {
                        anyhow!("already compacted")
                    } else {
                        anyhow!("nothing to compact (session too small)")
                    }
                },
            )?;
        let (summary, usage) = self
            .generate_compaction_summary(&preparation, custom_instructions, cancel)
            .await?;
        if cancel.is_cancelled() {
            bail!("compaction cancelled");
        }
        let summary = append_file_operations(
            summary,
            &preparation.read_files,
            &preparation.modified_files,
        );
        self.session.append_compaction(
            summary.clone(),
            preparation.first_kept_message_index,
            preparation.tokens_before,
            Some(usage),
            preparation.read_files,
            preparation.modified_files,
        )?;
        add_usage(&mut self.total_usage, usage);
        self.context_tokens = self.estimated_context_tokens();
        self.usage_estimated = true;
        self.last_context_dropped = 0;
        self.auto_compaction_failed = false;
        Ok(CompactionResult {
            summary,
            first_kept_message_index: preparation.first_kept_message_index,
            tokens_before: preparation.tokens_before,
            tokens_after: self.context_tokens,
            usage: Some(usage),
        })
    }

    async fn generate_compaction_summary(
        &self,
        preparation: &CompactionPreparation,
        custom_instructions: Option<&str>,
        cancel: &CancellationToken,
    ) -> Result<(String, Usage)> {
        if preparation.is_split_turn && !preparation.turn_prefix_messages.is_empty() {
            let (history, history_usage) = if preparation.messages_to_summarize.is_empty() {
                (
                    preparation
                        .previous_summary
                        .clone()
                        .unwrap_or_else(|| "No prior history.".to_string()),
                    None,
                )
            } else {
                let request = history_summary_request(
                    &preparation.messages_to_summarize,
                    preparation.settings.reserve_tokens,
                    custom_instructions,
                    preparation.previous_summary.as_deref(),
                );
                let (summary, usage) = self.complete_summary(request, cancel).await?;
                (summary, Some(usage))
            };
            let request = turn_prefix_summary_request(
                &preparation.turn_prefix_messages,
                preparation.settings.reserve_tokens,
            );
            let (prefix, prefix_usage) = self.complete_summary(request, cancel).await?;
            let usage = history_usage.map_or(prefix_usage, |history_usage| {
                combine_usage(history_usage, prefix_usage)
            });
            return Ok((
                format!("{history}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix}"),
                usage,
            ));
        }

        let request = history_summary_request(
            &preparation.messages_to_summarize,
            preparation.settings.reserve_tokens,
            custom_instructions,
            preparation.previous_summary.as_deref(),
        );
        self.complete_summary(request, cancel).await
    }

    async fn complete_summary(
        &self,
        request: crate::compaction::SummaryRequest,
        cancel: &CancellationToken,
    ) -> Result<(String, Usage)> {
        let context = vec![
            ChatMessage::system(SUMMARIZATION_SYSTEM_PROMPT),
            ChatMessage::user(request.prompt),
        ];
        let (sink, _discarded) = mpsc::unbounded_channel();
        let turn = self
            .client
            .stream_chat_with_max_tokens(&context, &[], &sink, cancel, Some(request.max_tokens))
            .await?;
        if !turn.tool_calls.is_empty() {
            bail!("summarization returned an unexpected tool call");
        }
        let summary = turn
            .content
            .as_deref()
            .map(str::trim)
            .filter(|content| !content.is_empty())
            .ok_or_else(|| anyhow!("summarization completed without text"))?
            .to_string();
        let usage = turn
            .usage
            .map_or_else(|| estimate_usage(&context, &[], &turn), normalize_usage);
        Ok((summary, usage))
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
    pub const fn max_input_tokens(&self) -> u64 {
        self.max_input_tokens
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
                api: profile.api,
                context_window: profile.context_window,
                max_input_tokens: profile.max_input_tokens,
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
            self.client.reconfigure(OpenAiModelConfig {
                base_url: profile.base_url.clone(),
                api_key: profile.api_key.clone(),
                model: profile.id.clone(),
                api: profile.api,
                reasoning_effort: reasoning_value,
                supports_reasoning_effort: profile.compat.reasoning_effort,
                supports_usage_in_streaming: profile.compat.usage_in_streaming,
                supports_strict_tools: profile.compat.strict_tools,
            })?;
            self.provider = Some(profile.provider);
            self.reasoning_effort = effective_effort;
            self.context_window = profile.context_window;
            self.max_input_tokens = profile.max_input_tokens;
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
        self.tools.set_api(self.client.api());
        let provider = self.provider.clone();
        let model = self.client.model().to_string();
        self.session
            .set_model(provider.as_deref(), &model, self.client.api())?;
        self.session.set_reasoning_effort(self.reasoning_effort)?;
        self.context_tokens = self.estimated_context_tokens();
        self.usage_estimated = true;
        Ok(())
    }

    pub fn set_web_search_mode(&mut self, mode: WebSearchMode) -> Result<()> {
        self.session.set_web_search_mode(mode)?;
        self.client.set_web_search_mode(mode);
        self.tools.set_web_search_mode(mode);
        Ok(())
    }

    #[must_use]
    pub const fn web_search_mode(&self) -> WebSearchMode {
        self.client.web_search_mode()
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
        self.session = self.session.fresh()?;
        self.total_usage = Usage::default();
        self.context_tokens = 0;
        self.usage_estimated = false;
        self.approved_tools.clear();
        self.last_context_dropped = 0;
        self.auto_compaction_failed = false;
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
    pub const fn api(&self) -> ApiProtocol {
        self.client.api()
    }

    #[must_use]
    pub fn session(&self) -> &Session {
        &self.session
    }

    #[must_use]
    pub fn has_pending_run(&self) -> bool {
        self.session.has_pending_run()
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

    #[must_use]
    pub fn mcp_startup_failures(&self) -> &[McpStartupFailure] {
        self.tools.mcp_startup_failures()
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
                    .context_messages()
                    .iter()
                    .map(estimate_message_tokens)
                    .sum::<u64>(),
            )
            .saturating_add(estimate_tool_definitions(self.tools.definitions()))
            .saturating_add(4)
    }
}

fn build_system_prompt(cwd: &Path) -> Result<String> {
    const MAX_PROJECT_INSTRUCTIONS_BYTES: u64 = 64 * 1024;

    let mut prompt = format!(
        "You are MCode, a focused coding agent running in a terminal.\n\
         Work directly in the user's repository and complete requested changes end to end.\n\
         Use read_file before editing unfamiliar code. Use edit_file for precise changes, \
         write_file for new or fully replaced files, and shell for searches, builds, and tests.\n\
         Prefer rg for text search when available. Keep changes scoped to the request.\n\
         Never claim a command succeeded unless you observed its output.\n\
         The working directory is {}. File tools cannot access paths outside it.",
        cwd.display()
    );
    let instructions_path = cwd.join("AGENTS.md");
    let metadata = match fs::symlink_metadata(&instructions_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(prompt),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to inspect {}", instructions_path.display()));
        }
    };
    if !metadata.file_type().is_file() {
        return Ok(prompt);
    }
    if metadata.len() > MAX_PROJECT_INSTRUCTIONS_BYTES {
        bail!(
            "project instructions exceed the 64 KiB limit: {}",
            instructions_path.display()
        );
    }
    let instructions = fs::read_to_string(&instructions_path)
        .with_context(|| format!("failed to read {}", instructions_path.display()))?;
    if !instructions.trim().is_empty() {
        prompt.push_str("\n\nProject instructions from AGENTS.md:\n\n");
        prompt.push_str(instructions.trim());
    }
    Ok(prompt)
}

fn normalize_usage(mut usage: Usage) -> Usage {
    if usage.total_tokens == 0 {
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);
    }
    usage
}

fn add_usage(total: &mut Usage, usage: Usage) {
    total.prompt_tokens = total.prompt_tokens.saturating_add(usage.prompt_tokens);
    total.completion_tokens = total
        .completion_tokens
        .saturating_add(usage.completion_tokens);
    total.total_tokens = total.total_tokens.saturating_add(usage.total_tokens);
}

fn estimate_usage(
    context: &[ChatMessage],
    definitions: &[ToolDefinition],
    turn: &AssistantTurn,
) -> Usage {
    let prompt_tokens = context
        .iter()
        .map(estimate_message_tokens)
        .sum::<u64>()
        .saturating_add(estimate_tool_definitions(definitions))
        .saturating_add(4);
    let semantic_completion_tokens = turn
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
        );
    let response_item_tokens = turn
        .response_items
        .iter()
        .map(|item| estimate_text_tokens(&item.to_string()))
        .sum::<u64>();
    let completion_tokens = semantic_completion_tokens.max(response_item_tokens).max(1);
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    }
}

fn estimate_tool_definitions(definitions: &[ToolDefinition]) -> u64 {
    serde_json::to_string(definitions)
        .map_or(0, |json| estimate_text_tokens(&json))
        .saturating_add(8)
}

struct ContextSelection {
    messages: Vec<ChatMessage>,
    dropped_messages: usize,
    dropped_turns: usize,
    estimated_tokens: u64,
}

fn select_context(
    system_prompt: &str,
    messages: &[ChatMessage],
    definitions: &[ToolDefinition],
    max_input_tokens: u64,
) -> Result<ContextSelection> {
    const INPUT_BUDGET_PERCENT: u64 = 80;
    const OMISSION_NOTICE: &str =
        "Earlier conversation turns were omitted automatically to fit the model context window.";

    let budget = max_input_tokens
        .saturating_mul(INPUT_BUDGET_PERCENT)
        .checked_div(100)
        .unwrap_or_default();
    let system = ChatMessage::system(system_prompt);
    let base_tokens = estimate_message_tokens(&system)
        .saturating_add(estimate_tool_definitions(definitions))
        .saturating_add(4);
    if base_tokens > budget {
        bail!(
            "system prompt and tool definitions need about {base_tokens} tokens, exceeding the safe input budget of {budget} for a {max_input_tokens}-token model input limit"
        );
    }

    let message_tokens = messages
        .iter()
        .map(estimate_message_tokens)
        .collect::<Vec<_>>();
    let full_tokens = base_tokens.saturating_add(message_tokens.iter().sum::<u64>());
    if full_tokens <= budget {
        let mut context = Vec::with_capacity(messages.len() + 1);
        context.push(system);
        context.extend_from_slice(messages);
        return Ok(ContextSelection {
            messages: context,
            dropped_messages: 0,
            dropped_turns: 0,
            estimated_tokens: full_tokens,
        });
    }

    let turns = conversation_turns(messages);
    let Some(latest) = turns.last() else {
        return Ok(ContextSelection {
            messages: vec![system],
            dropped_messages: 0,
            dropped_turns: 0,
            estimated_tokens: base_tokens,
        });
    };
    let trimmed_system = ChatMessage::system(format!("{system_prompt}\n\n{OMISSION_NOTICE}"));
    let fixed_tokens = estimate_message_tokens(&trimmed_system)
        .saturating_add(estimate_tool_definitions(definitions))
        .saturating_add(4);
    let latest_tokens = message_tokens[latest.clone()].iter().sum::<u64>();
    if fixed_tokens.saturating_add(latest_tokens) > budget {
        bail!(
            "the current conversation turn needs about {} input tokens, exceeding the safe budget of {budget}; start a new session or reduce the prompt, images, or tool output",
            fixed_tokens.saturating_add(latest_tokens)
        );
    }

    let mut first_turn = turns.len() - 1;
    let mut selected_tokens = fixed_tokens.saturating_add(latest_tokens);
    for index in (0..first_turn).rev() {
        let turn_tokens = message_tokens[turns[index].clone()].iter().sum::<u64>();
        if selected_tokens.saturating_add(turn_tokens) > budget {
            break;
        }
        selected_tokens = selected_tokens.saturating_add(turn_tokens);
        first_turn = index;
    }

    let first_message = turns[first_turn].start;
    let mut context = Vec::with_capacity(messages.len() - first_message + 1);
    context.push(trimmed_system);
    context.extend_from_slice(&messages[first_message..]);
    Ok(ContextSelection {
        messages: context,
        dropped_messages: first_message,
        dropped_turns: first_turn,
        estimated_tokens: selected_tokens,
    })
}

fn conversation_turns(messages: &[ChatMessage]) -> Vec<Range<usize>> {
    if messages.is_empty() {
        return Vec::new();
    }
    let mut starts = vec![0];
    for (index, message) in messages.iter().enumerate().skip(1) {
        if message.role == MessageRole::User {
            starts.push(index);
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(index, start)| *start..starts.get(index + 1).copied().unwrap_or(messages.len()))
        .collect()
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

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

    #[test]
    fn context_trimming_keeps_a_contiguous_suffix_of_complete_turns() {
        let messages = vec![
            ChatMessage::user("old ".repeat(160)),
            ChatMessage::assistant(Some("old answer ".repeat(80)), None, Vec::new()),
            ChatMessage::user("recent question"),
            ChatMessage::assistant(Some("recent answer".to_string()), None, Vec::new()),
            ChatMessage::user("current question"),
        ];
        let selection = select_context("system", &messages, &[], 300).unwrap();

        assert!(selection.dropped_messages >= 2);
        assert!(selection.dropped_turns >= 1);
        assert_eq!(selection.messages.last(), messages.last());
        assert_eq!(selection.messages[0].role, MessageRole::System);
        assert!(
            selection.messages[0]
                .content
                .as_deref()
                .unwrap()
                .contains("omitted automatically")
        );
        assert_eq!(selection.messages[1].role, MessageRole::User);
        assert!(selection.estimated_tokens <= 240);
    }

    #[test]
    fn context_trimming_rejects_an_oversized_current_turn() {
        let messages = vec![ChatMessage::user("x".repeat(2_000))];
        let error = select_context("system", &messages, &[], 200)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("current conversation turn"));
    }

    #[test]
    fn token_estimate_is_conservative_for_non_ascii_text() {
        assert_eq!(estimate_text_tokens("abcd"), 1);
        assert_eq!(estimate_text_tokens("中文"), 4);
    }

    #[test]
    fn project_instructions_load_only_from_a_regular_agents_file() {
        let project = tempdir().unwrap();
        fs::write(
            project.path().join("AGENTS.md"),
            "  Keep the project sentinel intact.  \n",
        )
        .unwrap();

        let prompt = build_system_prompt(project.path()).unwrap();
        assert!(prompt.contains("Project instructions from AGENTS.md:"));
        assert!(prompt.ends_with("Keep the project sentinel intact."));

        fs::remove_file(project.path().join("AGENTS.md")).unwrap();
        fs::create_dir(project.path().join("AGENTS.md")).unwrap();
        let prompt = build_system_prompt(project.path()).unwrap();
        assert!(!prompt.contains("Project instructions from AGENTS.md:"));
    }

    #[cfg(unix)]
    #[test]
    fn project_instructions_do_not_follow_an_agents_symlink() {
        use std::os::unix::fs::symlink;

        let project = tempdir().unwrap();
        fs::write(project.path().join("instructions.txt"), "do not load me").unwrap();
        symlink(
            project.path().join("instructions.txt"),
            project.path().join("AGENTS.md"),
        )
        .unwrap();

        let prompt = build_system_prompt(project.path()).unwrap();
        assert!(!prompt.contains("do not load me"));
        assert!(!prompt.contains("Project instructions from AGENTS.md:"));
    }
}
