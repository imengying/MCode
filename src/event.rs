use serde::Serialize;

use crate::protocol::{FileChangeSummary, Usage, WebSearchAction};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompactionReason {
    Manual,
    Threshold,
    Overflow,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    RunStarted,
    RunResumed,
    AssistantStarted,
    AssistantRetrying {
        attempt: usize,
        max_attempts: usize,
        message: String,
    },
    TextDelta {
        text: String,
    },
    ReasoningSummaryDelta {
        text: String,
    },
    ReasoningSummaryPartAdded {
        index: usize,
    },
    ReasoningSummaryFinished,
    ToolStarted {
        id: String,
        name: String,
        arguments: String,
    },
    ApprovalRequested {
        id: String,
        name: String,
        arguments: String,
    },
    ApprovalResolved {
        id: String,
        name: String,
        approved: bool,
        for_session: bool,
    },
    ToolFinished {
        id: String,
        name: String,
        output: String,
        is_error: bool,
        file_change: Option<FileChangeSummary>,
    },
    WebSearchStarted {
        id: String,
    },
    WebSearchFinished {
        id: String,
        action: WebSearchAction,
    },
    Usage {
        usage: Usage,
        context_tokens: u64,
        context_window: u64,
        max_input_tokens: u64,
        estimated: bool,
    },
    ContextTrimmed {
        dropped_messages: usize,
        dropped_turns: usize,
        estimated_tokens: u64,
    },
    CompactionStarted {
        reason: CompactionReason,
    },
    CompactionFinished {
        reason: CompactionReason,
        summary: String,
        first_kept_message_index: usize,
        tokens_before: u64,
        tokens_after: u64,
        usage: Option<Usage>,
    },
    CompactionFailed {
        reason: CompactionReason,
        message: String,
    },
    RunFinished,
    Cancelled,
    Error {
        message: String,
    },
}
