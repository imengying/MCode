use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

pub(crate) const MAX_IMAGE_BYTES: u64 = 20 * 1024 * 1024;
const SUPPORTED_IMAGE_MIME_TYPES: [&str; 4] =
    ["image/png", "image/jpeg", "image/gif", "image/webp"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: MessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageAttachment>,
    /// Raw Responses API output items that must be replayed unchanged for
    /// stateless multi-turn reasoning and tool calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub response_items: Vec<serde_json::Value>,
    /// UI-only metadata for an applied built-in file change. API request
    /// conversion intentionally ignores this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_change: Option<FileChangeSummary>,
}

impl ChatMessage {
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(MessageRole::System, content)
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::text(MessageRole::User, content)
    }

    #[must_use]
    pub fn user_with_images(content: impl Into<String>, images: Vec<ImageAttachment>) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            reasoning_content: None,
            images,
            response_items: Vec::new(),
            file_change: None,
        }
    }

    #[must_use]
    pub fn assistant(
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self::assistant_with_response_items(content, reasoning_content, tool_calls, Vec::new())
    }

    #[must_use]
    pub fn assistant_with_response_items(
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<ToolCall>,
        response_items: Vec<serde_json::Value>,
    ) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
            tool_calls,
            tool_call_id: None,
            tool_name: None,
            reasoning_content,
            images: Vec::new(),
            response_items,
            file_change: None,
        }
    }

    #[must_use]
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::tool_with_file_change(tool_call_id, content, None)
    }

    #[must_use]
    pub fn tool_with_file_change(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        file_change: Option<FileChangeSummary>,
    ) -> Self {
        Self::tool_message(tool_call_id, None, content, file_change)
    }

    #[must_use]
    pub fn named_tool_with_file_change(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
        file_change: Option<FileChangeSummary>,
    ) -> Self {
        Self::tool_message(tool_call_id, Some(tool_name.into()), content, file_change)
    }

    fn tool_message(
        tool_call_id: impl Into<String>,
        tool_name: Option<String>,
        content: impl Into<String>,
        file_change: Option<FileChangeSummary>,
    ) -> Self {
        Self {
            role: MessageRole::Tool,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            tool_name,
            reasoning_content: None,
            images: Vec::new(),
            response_items: Vec::new(),
            file_change,
        }
    }

    fn text(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            reasoning_content: None,
            images: Vec::new(),
            response_items: Vec::new(),
            file_change: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FileChangeKind {
    Added,
    Updated,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FileChangeLineKind {
    Context,
    Added,
    Removed,
    Omitted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileChangeLine {
    pub kind: FileChangeLineKind,
    pub line_number: usize,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileChangeSummary {
    pub path: String,
    pub kind: FileChangeKind,
    pub added_lines: usize,
    pub removed_lines: usize,
    pub preview: Vec<FileChangeLine>,
    pub preview_truncated: bool,
}

/// Removes terminal control sequences from untrusted model, tool, MCP, and
/// provider text while preserving ordinary layout characters.
#[must_use]
pub fn sanitize_terminal_text(text: &str) -> String {
    text.chars()
        .filter_map(|character| match character {
            '\n' | '\t' => Some(character),
            '\r' => Some('\n'),
            character if character.is_control() => None,
            character => Some(character),
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageAttachment {
    pub name: String,
    pub mime_type: String,
    pub data: String,
}

impl ImageAttachment {
    pub fn load(path: &Path, cwd: &Path) -> Result<Self> {
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            cwd.join(path)
        };
        let path = path
            .canonicalize()
            .with_context(|| format!("failed to resolve image: {}", path.display()))?;
        let metadata = path
            .metadata()
            .with_context(|| format!("failed to inspect image: {}", path.display()))?;
        if !metadata.is_file() {
            bail!("image path is not a file: {}", path.display());
        }
        if metadata.len() > MAX_IMAGE_BYTES {
            bail!("image exceeds the 20 MiB limit: {}", path.display());
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read image: {}", path.display()))?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("image")
            .to_string();
        Self::from_encoded_bytes(name, bytes)
            .with_context(|| format!("invalid image: {}", path.display()))
    }

    pub fn from_encoded_bytes(name: impl Into<String>, bytes: Vec<u8>) -> Result<Self> {
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_IMAGE_BYTES {
            bail!("image exceeds the 20 MiB limit");
        }
        let mime_type = infer::get(&bytes)
            .map(|kind| kind.mime_type())
            .filter(|mime| SUPPORTED_IMAGE_MIME_TYPES.contains(mime))
            .ok_or_else(|| {
                anyhow::anyhow!("unsupported image format; use PNG, JPEG, GIF, or WebP")
            })?;
        Ok(Self {
            name: name.into(),
            mime_type: mime_type.to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
        })
    }

    #[must_use]
    pub fn data_url(&self) -> String {
        format!("data:{};base64,{}", self.mime_type, self.data)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebSearchAction {
    Search {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        query: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        queries: Vec<String>,
    },
    OpenPage {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
    },
    FindInPage {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pattern: Option<String>,
    },
    #[serde(other)]
    Other,
}

impl WebSearchAction {
    #[must_use]
    pub fn description_zh(&self) -> String {
        match self {
            Self::Search { query, queries } => query
                .clone()
                .or_else(|| (!queries.is_empty()).then(|| queries.join("、")))
                .unwrap_or_else(|| "搜索完成".to_string()),
            Self::OpenPage { url } => url
                .as_deref()
                .map_or_else(|| "已打开页面".to_string(), |url| format!("已打开 {url}")),
            Self::FindInPage { url, pattern } => match (pattern, url) {
                (Some(pattern), Some(url)) => format!("在 {url} 中找到 {pattern:?}"),
                (Some(pattern), None) => format!("在页面中找到 {pattern:?}"),
                (None, Some(url)) => format!("已在 {url} 中搜索"),
                (None, None) => "已在页面内搜索".to_string(),
            },
            Self::Other => "网页搜索完成".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_prompt_tokens: Option<u64>,
}

impl Usage {
    #[must_use]
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            prompt_tokens: self.prompt_tokens.saturating_add(other.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .saturating_add(other.completion_tokens),
            total_tokens: self.total_tokens.saturating_add(other.total_tokens),
            cached_prompt_tokens: match (self.cached_prompt_tokens, other.cached_prompt_tokens) {
                (Some(left), Some(right)) => Some(left.saturating_add(right)),
                (Some(cached), None) | (None, Some(cached)) => Some(cached),
                (None, None) => None,
            },
        }
    }
}
