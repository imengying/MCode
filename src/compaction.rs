use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::config::CompactionSettings;
use crate::protocol::{ChatMessage, MessageRole, Usage};
use crate::session::Session;

const TOOL_RESULT_MAX_CHARS: usize = 2_000;
const ESTIMATED_IMAGE_TOKENS: u64 = 1_024;

pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

const SUMMARIZATION_PROMPT: &str = r#"The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."#;

const UPDATE_SUMMARIZATION_PROMPT: &str = r#"The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."#;

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = r"This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.";

#[derive(Debug, Clone)]
pub struct CompactionPreparation {
    pub first_kept_message_index: usize,
    pub messages_to_summarize: Vec<ChatMessage>,
    pub turn_prefix_messages: Vec<ChatMessage>,
    pub is_split_turn: bool,
    pub tokens_before: u64,
    pub previous_summary: Option<String>,
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
    pub settings: CompactionSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryRequest {
    pub prompt: String,
    pub max_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CutPoint {
    first_kept_message_index: usize,
    turn_start_index: Option<usize>,
    is_split_turn: bool,
}

#[must_use]
pub fn should_compact(
    context_tokens: u64,
    context_window: u64,
    settings: CompactionSettings,
) -> bool {
    settings.enabled && context_tokens > context_window.saturating_sub(settings.reserve_tokens)
}

#[must_use]
pub fn prepare_compaction(
    session: &Session,
    settings: CompactionSettings,
    tokens_before: u64,
) -> Option<CompactionPreparation> {
    if session.is_compacted_at_tip() {
        return None;
    }

    let messages = session.messages();
    let previous = session.latest_compaction();
    let boundary_start = previous.map_or(0, |checkpoint| {
        checkpoint.first_kept_message_index.min(messages.len())
    });
    let cut_point = find_cut_point(
        messages,
        boundary_start,
        messages.len(),
        settings.keep_recent_tokens,
    );
    let history_end = if cut_point.is_split_turn {
        cut_point
            .turn_start_index
            .unwrap_or(cut_point.first_kept_message_index)
    } else {
        cut_point.first_kept_message_index
    };
    let messages_to_summarize = messages[boundary_start..history_end].to_vec();
    let turn_prefix_messages = if cut_point.is_split_turn {
        messages[history_end..cut_point.first_kept_message_index].to_vec()
    } else {
        Vec::new()
    };
    if messages_to_summarize.is_empty() && turn_prefix_messages.is_empty() {
        return None;
    }

    let mut read_files = previous
        .map(|checkpoint| checkpoint.read_files.iter().cloned().collect())
        .unwrap_or_default();
    let mut modified_files = previous
        .map(|checkpoint| checkpoint.modified_files.iter().cloned().collect())
        .unwrap_or_default();
    for message in messages_to_summarize.iter().chain(&turn_prefix_messages) {
        extract_file_operations(message, &mut read_files, &mut modified_files);
    }
    read_files.retain(|path| !modified_files.contains(path));

    Some(CompactionPreparation {
        first_kept_message_index: cut_point.first_kept_message_index,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn: cut_point.is_split_turn,
        tokens_before,
        previous_summary: previous.map(|checkpoint| checkpoint.summary.clone()),
        read_files: read_files.into_iter().collect(),
        modified_files: modified_files.into_iter().collect(),
        settings,
    })
}

#[must_use]
pub fn history_summary_request(
    messages: &[ChatMessage],
    reserve_tokens: u64,
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
) -> SummaryRequest {
    let conversation = serialize_conversation(messages);
    let mut prompt = format!("<conversation>\n{conversation}\n</conversation>\n\n");
    if let Some(previous_summary) = previous_summary {
        prompt.push_str("<previous-summary>\n");
        prompt.push_str(previous_summary);
        prompt.push_str("\n</previous-summary>\n\n");
    }
    prompt.push_str(if previous_summary.is_some() {
        UPDATE_SUMMARIZATION_PROMPT
    } else {
        SUMMARIZATION_PROMPT
    });
    if let Some(instructions) = custom_instructions.filter(|value| !value.trim().is_empty()) {
        prompt.push_str("\n\nAdditional focus: ");
        prompt.push_str(instructions.trim());
    }
    SummaryRequest {
        prompt,
        max_tokens: reserve_tokens
            .saturating_mul(4)
            .checked_div(5)
            .unwrap_or(1)
            .max(1),
    }
}

#[must_use]
pub fn turn_prefix_summary_request(
    messages: &[ChatMessage],
    reserve_tokens: u64,
) -> SummaryRequest {
    let conversation = serialize_conversation(messages);
    SummaryRequest {
        prompt: format!(
            "<conversation>\n{conversation}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
        ),
        max_tokens: reserve_tokens.checked_div(2).unwrap_or(1).max(1),
    }
}

#[must_use]
pub fn append_file_operations(
    mut summary: String,
    read_files: &[String],
    modified_files: &[String],
) -> String {
    if !read_files.is_empty() {
        summary.push_str("\n\n<read-files>\n");
        summary.push_str(&read_files.join("\n"));
        summary.push_str("\n</read-files>");
    }
    if !modified_files.is_empty() {
        summary.push_str("\n\n<modified-files>\n");
        summary.push_str(&modified_files.join("\n"));
        summary.push_str("\n</modified-files>");
    }
    summary
}

#[must_use]
pub fn combine_usage(first: Usage, second: Usage) -> Usage {
    Usage {
        prompt_tokens: first.prompt_tokens.saturating_add(second.prompt_tokens),
        completion_tokens: first
            .completion_tokens
            .saturating_add(second.completion_tokens),
        total_tokens: first.total_tokens.saturating_add(second.total_tokens),
    }
}

#[must_use]
pub fn serialize_conversation(messages: &[ChatMessage]) -> String {
    let mut parts = Vec::new();
    for message in messages {
        match message.role {
            MessageRole::System => {}
            MessageRole::User => {
                let mut content = message.content.clone().unwrap_or_default();
                if !message.images.is_empty() {
                    let names = message
                        .images
                        .iter()
                        .map(|image| image.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let _ = write!(
                        content,
                        "\n[Image attachments ({}): {names}]",
                        message.images.len()
                    );
                }
                if !content.is_empty() {
                    parts.push(format!("[User]: {content}"));
                }
            }
            MessageRole::Assistant => {
                if let Some(reasoning) = message
                    .reasoning_content
                    .as_deref()
                    .filter(|value| !value.is_empty())
                {
                    parts.push(format!("[Assistant thinking]: {reasoning}"));
                }
                if let Some(content) = message.content.as_deref().filter(|value| !value.is_empty())
                {
                    parts.push(format!("[Assistant]: {content}"));
                }
                if !message.tool_calls.is_empty() {
                    let calls = message
                        .tool_calls
                        .iter()
                        .map(|call| format_tool_call(&call.function.name, &call.function.arguments))
                        .collect::<Vec<_>>()
                        .join("; ");
                    parts.push(format!("[Assistant tool calls]: {calls}"));
                }
            }
            MessageRole::Tool => {
                if let Some(content) = message.content.as_deref().filter(|value| !value.is_empty())
                {
                    parts.push(format!(
                        "[Tool result]: {}",
                        truncate_for_summary(content, TOOL_RESULT_MAX_CHARS)
                    ));
                }
            }
        }
    }
    parts.join("\n\n")
}

#[must_use]
pub fn estimate_message_tokens(message: &ChatMessage) -> u64 {
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
    let semantic_tokens = tokens.saturating_add(
        u64::try_from(message.images.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(ESTIMATED_IMAGE_TOKENS),
    );
    let response_item_tokens = 4_u64.saturating_add(
        message
            .response_items
            .iter()
            .map(|item| estimate_text_tokens(&item.to_string()))
            .sum::<u64>(),
    );
    semantic_tokens.max(response_item_tokens)
}

#[must_use]
pub fn estimate_text_tokens(text: &str) -> u64 {
    let (ascii, non_ascii) = text.chars().fold((0_u64, 0_u64), |counts, character| {
        if character.is_ascii() {
            (counts.0.saturating_add(1), counts.1)
        } else {
            (counts.0, counts.1.saturating_add(1))
        }
    });
    ascii
        .div_ceil(4)
        .saturating_add(non_ascii.saturating_mul(2))
}

fn find_cut_point(
    messages: &[ChatMessage],
    start_index: usize,
    end_index: usize,
    keep_recent_tokens: u64,
) -> CutPoint {
    let cut_points = (start_index..end_index)
        .filter(|index| is_cut_point(&messages[*index]))
        .collect::<Vec<_>>();
    let Some(default_cut) = cut_points.first().copied() else {
        return CutPoint {
            first_kept_message_index: start_index,
            turn_start_index: None,
            is_split_turn: false,
        };
    };

    let mut accumulated_tokens = 0_u64;
    let mut cut_index = default_cut;
    for index in (start_index..end_index).rev() {
        accumulated_tokens =
            accumulated_tokens.saturating_add(estimate_message_tokens(&messages[index]));
        if accumulated_tokens >= keep_recent_tokens {
            if let Some(point) = cut_points.iter().copied().find(|point| *point >= index) {
                cut_index = point;
            }
            break;
        }
    }

    let starts_turn = messages[cut_index].role == MessageRole::User;
    let turn_start_index = (!starts_turn).then(|| {
        (start_index..=cut_index)
            .rev()
            .find(|index| messages[*index].role == MessageRole::User)
    });
    let turn_start_index = turn_start_index.flatten();
    CutPoint {
        first_kept_message_index: cut_index,
        turn_start_index,
        is_split_turn: !starts_turn && turn_start_index.is_some(),
    }
}

fn is_cut_point(message: &ChatMessage) -> bool {
    matches!(message.role, MessageRole::User | MessageRole::Assistant)
}

fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let total_chars = text.chars().count();
    if total_chars <= max_chars {
        return text.to_string();
    }
    let prefix = text.chars().take(max_chars).collect::<String>();
    format!(
        "{prefix}\n\n[... {} more characters truncated]",
        total_chars - max_chars
    )
}

fn format_tool_call(name: &str, arguments: &str) -> String {
    let Ok(serde_json::Value::Object(arguments)) = serde_json::from_str(arguments) else {
        return format!("{name}({arguments})");
    };
    let arguments = arguments
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{name}({arguments})")
}

fn extract_file_operations(
    message: &ChatMessage,
    read_files: &mut BTreeSet<String>,
    modified_files: &mut BTreeSet<String>,
) {
    if message.role != MessageRole::Assistant {
        return;
    }
    for call in &message.tool_calls {
        let Ok(arguments) = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
        else {
            continue;
        };
        let Some(path) = arguments.get("path").and_then(serde_json::Value::as_str) else {
            continue;
        };
        match call.function.name.as_str() {
            "read" | "read_file" => {
                read_files.insert(path.to_string());
            }
            "write" | "write_file" | "edit" | "edit_file" => {
                modified_files.insert(path.to_string());
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::config::ReasoningEffort;
    use crate::protocol::{FunctionCall, ToolCall};

    fn settings(keep_recent_tokens: u64) -> CompactionSettings {
        CompactionSettings {
            enabled: true,
            reserve_tokens: 100,
            keep_recent_tokens,
        }
    }

    #[test]
    fn threshold_matches_pi_reserve_formula() {
        let settings = settings(20);
        assert!(!should_compact(900, 1_000, settings));
        assert!(should_compact(901, 1_000, settings));
        assert!(!should_compact(
            1_000,
            1_000,
            CompactionSettings {
                enabled: false,
                ..settings
            }
        ));
    }

    #[test]
    fn cut_point_keeps_a_suffix_at_a_complete_turn_boundary() {
        let messages = vec![
            ChatMessage::user("old request"),
            ChatMessage::assistant(Some("old answer".into()), None, Vec::new()),
            ChatMessage::user("recent request"),
            ChatMessage::assistant(Some("recent answer".into()), None, Vec::new()),
            ChatMessage::user("current request"),
        ];
        let cut = find_cut_point(&messages, 0, messages.len(), 18);
        assert_eq!(cut.first_kept_message_index, 2);
        assert!(!cut.is_split_turn);
    }

    #[test]
    fn oversized_turn_is_split_without_cutting_at_a_tool_result() {
        let messages = vec![
            ChatMessage::user("one large turn"),
            ChatMessage::assistant(
                None,
                None,
                vec![ToolCall {
                    id: "call_1".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: r#"{"path":"src/lib.rs"}"#.into(),
                    },
                }],
            ),
            ChatMessage::tool("call_1", "x".repeat(500)),
            ChatMessage::assistant(Some("recent suffix".into()), None, Vec::new()),
        ];
        let cut = find_cut_point(&messages, 0, messages.len(), 20);
        assert_eq!(cut.first_kept_message_index, 3);
        assert_eq!(cut.turn_start_index, Some(0));
        assert!(cut.is_split_turn);

        let project = tempdir().unwrap();
        let mut session = Session::create(
            project.path(),
            crate::session::SessionMetadata::local("model", ReasoningEffort::Off),
            false,
        )
        .unwrap();
        for message in &messages {
            session.append(message.clone()).unwrap();
        }
        let preparation = prepare_compaction(&session, settings(20), 1_000).unwrap();
        assert!(preparation.is_split_turn);
        assert!(preparation.messages_to_summarize.is_empty());
        assert_eq!(preparation.turn_prefix_messages, messages[..3]);
        assert_eq!(preparation.first_kept_message_index, 3);
        assert_eq!(preparation.read_files, ["src/lib.rs"]);
    }

    #[test]
    fn repeated_compaction_starts_at_the_previous_kept_boundary() {
        let project = tempdir().unwrap();
        let mut session = Session::create(
            project.path(),
            crate::session::SessionMetadata::local("model", ReasoningEffort::Off),
            false,
        )
        .unwrap();
        session
            .append(ChatMessage::user("already summarized"))
            .unwrap();
        session
            .append(ChatMessage::assistant(Some("old".into()), None, Vec::new()))
            .unwrap();
        session
            .append(ChatMessage::user("kept before".repeat(8)))
            .unwrap();
        session
            .append_compaction(
                "previous summary".into(),
                2,
                500,
                None,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
        session
            .append(ChatMessage::assistant(
                Some("next".repeat(20)),
                None,
                Vec::new(),
            ))
            .unwrap();
        session
            .append(ChatMessage::user("new recent".repeat(8)))
            .unwrap();

        let preparation = prepare_compaction(&session, settings(20), 600).unwrap();
        assert_eq!(
            preparation.previous_summary.as_deref(),
            Some("previous summary")
        );
        assert!(preparation.first_kept_message_index >= 3);
        assert_eq!(preparation.messages_to_summarize[0], session.messages()[2]);
    }

    #[test]
    fn summary_serialization_truncates_tool_results() {
        let serialized = serialize_conversation(&[ChatMessage::tool("call", "x".repeat(2_500))]);
        assert!(serialized.contains("500 more characters truncated"));
        assert!(serialized.len() < 2_100);
    }

    #[test]
    fn update_prompt_carries_the_previous_summary_and_custom_focus() {
        let request = history_summary_request(
            &[ChatMessage::user("new work")],
            100,
            Some("focus on tests"),
            Some("old checkpoint"),
        );
        assert!(
            request
                .prompt
                .contains("<previous-summary>\nold checkpoint")
        );
        assert!(request.prompt.contains("PRESERVE all existing information"));
        assert!(request.prompt.contains("Additional focus: focus on tests"));
        assert_eq!(request.max_tokens, 80);
    }
}
