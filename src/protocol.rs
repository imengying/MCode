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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageAttachment>,
    /// Raw Responses API output items that must be replayed unchanged for
    /// stateless multi-turn reasoning and tool calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub response_items: Vec<serde_json::Value>,
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
            reasoning_content: None,
            images,
            response_items: Vec::new(),
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
            reasoning_content,
            images: Vec::new(),
            response_items,
        }
    }

    #[must_use]
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            reasoning_content: None,
            images: Vec::new(),
            response_items: Vec::new(),
        }
    }

    fn text(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning_content: None,
            images: Vec::new(),
            response_items: Vec::new(),
        }
    }
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
    pub fn description(&self) -> String {
        match self {
            Self::Search { query, queries } => query
                .clone()
                .or_else(|| (!queries.is_empty()).then(|| queries.join(", ")))
                .unwrap_or_else(|| "search completed".to_string()),
            Self::OpenPage { url } => url.as_deref().map_or_else(
                || "opened a page".to_string(),
                |url| format!("opened {url}"),
            ),
            Self::FindInPage { url, pattern } => match (pattern, url) {
                (Some(pattern), Some(url)) => format!("found {pattern:?} in {url}"),
                (Some(pattern), None) => format!("found {pattern:?} in page"),
                (None, Some(url)) => format!("searched within {url}"),
                (None, None) => "searched within a page".to_string(),
            },
            Self::Other => "web search completed".to_string(),
        }
    }

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

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn loads_and_serializes_a_png_attachment() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("pixel.png");
        fs::write(
            &path,
            [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0],
        )
        .unwrap();

        let image = ImageAttachment::load(&path, temp.path()).unwrap();
        assert_eq!(image.name, "pixel.png");
        assert_eq!(image.mime_type, "image/png");
        assert!(image.data_url().starts_with("data:image/png;base64,"));

        let encoded =
            serde_json::to_value(ChatMessage::user_with_images("look", vec![image])).unwrap();
        assert_eq!(encoded["images"][0]["name"], "pixel.png");
    }

    #[test]
    fn accepts_encoded_clipboard_images() {
        let image = ImageAttachment::from_encoded_bytes(
            "clipboard.png",
            vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0],
        )
        .unwrap();
        assert_eq!(image.name, "clipboard.png");
        assert_eq!(image.mime_type, "image/png");
        assert!(image.data_url().starts_with("data:image/png;base64,"));
        assert!(ImageAttachment::from_encoded_bytes("bad.png", vec![0; 12]).is_err());
    }

    #[test]
    fn strips_terminal_control_sequences_from_untrusted_text() {
        assert_eq!(
            sanitize_terminal_text("safe\u{1b}]52;c;secret\u{7}\r\ntext\tend"),
            "safe]52;c;secret\n\ntext\tend"
        );
    }

    #[test]
    fn localizes_web_search_action_descriptions() {
        let action = WebSearchAction::OpenPage {
            url: Some("https://example.com".to_string()),
        };
        assert_eq!(action.description_zh(), "已打开 https://example.com");
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}
