use serde::Serialize;

use crate::protocol::Usage;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    RunStarted,
    AssistantStarted,
    TextDelta {
        text: String,
    },
    ReasoningDelta {
        text: String,
    },
    ToolStarted {
        id: String,
        name: String,
        arguments: String,
    },
    ToolFinished {
        id: String,
        name: String,
        output: String,
        is_error: bool,
    },
    Usage {
        usage: Usage,
        context_tokens: u64,
        context_window: u64,
        estimated: bool,
    },
    RunFinished,
    Cancelled,
    Error {
        message: String,
    },
}
