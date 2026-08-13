use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

const REDACTED: &str = "<redacted>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    ApproveOnce,
    ApproveForSession,
    Deny,
}

#[must_use]
pub fn format_tool_arguments(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments).map_or_else(
        |_| redact_sensitive_text(arguments),
        |mut value| {
            redact_json_value(&mut value);
            serde_json::to_string_pretty(&value)
                .unwrap_or_else(|_| redact_sensitive_text(arguments))
        },
    )
}

/// Redacts credentials from text before it is rendered or written to a terminal log.
#[must_use]
pub fn redact_sensitive_text(text: &str) -> String {
    let mut redactor = SensitiveTextRedactor::default();
    let mut output = redactor.push(text);
    output.push_str(&redactor.finish());
    output
}

#[derive(Debug, Default)]
pub struct SensitiveTextRedactor {
    pending: String,
    expectation: SecretExpectation,
}

impl SensitiveTextRedactor {
    pub fn push(&mut self, text: &str) -> String {
        self.pending.push_str(text);
        let mut output = String::new();
        while let Some((whitespace_start, whitespace_end)) =
            self.pending.char_indices().find_map(|(index, character)| {
                character
                    .is_whitespace()
                    .then_some((index, index + character.len_utf8()))
            })
        {
            let token = self.pending[..whitespace_start].to_string();
            let whitespace = self.pending[whitespace_start..whitespace_end].to_string();
            let (redacted, expectation) = redact_token(&token, self.expectation);
            output.push_str(&redacted);
            output.push_str(&whitespace);
            self.expectation = expectation;
            self.pending.drain(..whitespace_end);
        }
        output
    }

    pub fn finish(&mut self) -> String {
        let pending = std::mem::take(&mut self.pending);
        let (redacted, _) = redact_token(&pending, self.expectation);
        self.expectation = SecretExpectation::None;
        redacted
    }
}

#[derive(Debug, Default, Clone, Copy)]
enum SecretExpectation {
    #[default]
    None,
    Authorization,
    Value,
}

fn redact_json_value(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            for (key, value) in object {
                if is_sensitive_key(key) {
                    *value = serde_json::Value::String(REDACTED.to_string());
                } else {
                    redact_json_value(value);
                }
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                redact_json_value(value);
            }
        }
        serde_json::Value::String(text) => *text = redact_sensitive_text(text),
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

fn redact_token(token: &str, expectation: SecretExpectation) -> (String, SecretExpectation) {
    if token.is_empty() {
        return (String::new(), expectation);
    }

    match expectation {
        SecretExpectation::Value => {
            return (
                redacted_token_with_punctuation(token),
                SecretExpectation::None,
            );
        }
        SecretExpectation::Authorization => {
            if token
                .trim_matches(shell_punctuation)
                .eq_ignore_ascii_case("bearer")
            {
                return (token.to_string(), SecretExpectation::Value);
            }
            return (
                redacted_token_with_punctuation(token),
                SecretExpectation::None,
            );
        }
        SecretExpectation::None => {}
    }

    let core = token.trim_matches(shell_punctuation);
    if core.eq_ignore_ascii_case("bearer") {
        return (token.to_string(), SecretExpectation::Value);
    }
    if core.eq_ignore_ascii_case("authorization:")
        || core.eq_ignore_ascii_case("proxy-authorization:")
    {
        return (token.to_string(), SecretExpectation::Authorization);
    }

    if let Some((key, _)) = core.split_once('=')
        && is_sensitive_cli_key(key)
    {
        return (
            replace_token_value(token, core, key.len().saturating_add(1)),
            SecretExpectation::None,
        );
    }
    if is_sensitive_flag(core) {
        return (token.to_string(), SecretExpectation::Value);
    }
    if looks_like_secret(core) {
        return (
            redacted_token_with_punctuation(token),
            SecretExpectation::None,
        );
    }

    (token.to_string(), SecretExpectation::None)
}

fn replace_token_value(token: &str, core: &str, value_start: usize) -> String {
    let core_start = token.find(core).unwrap_or(0);
    let suffix_start = core_start + core.len();
    format!(
        "{}{}{}{}",
        &token[..core_start],
        &core[..value_start],
        REDACTED,
        &token[suffix_start..]
    )
}

fn redacted_token_with_punctuation(token: &str) -> String {
    let prefix_len = token
        .char_indices()
        .find_map(|(index, character)| (!shell_punctuation(character)).then_some(index))
        .unwrap_or(token.len());
    let suffix_start = token
        .char_indices()
        .rev()
        .find_map(|(index, character)| {
            (!shell_punctuation(character)).then_some(index + character.len_utf8())
        })
        .unwrap_or(prefix_len);
    format!(
        "{}{}{}",
        &token[..prefix_len],
        REDACTED,
        &token[suffix_start..]
    )
}

fn shell_punctuation(character: char) -> bool {
    matches!(
        character,
        '\'' | '"' | '`' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}'
    )
}

fn is_sensitive_cli_key(key: &str) -> bool {
    is_sensitive_key(key.trim_start_matches('-'))
}

fn is_sensitive_flag(token: &str) -> bool {
    token.starts_with('-') && is_sensitive_cli_key(token)
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|character| character.to_ascii_lowercase())
        .collect::<String>();
    matches!(
        normalized.as_str(),
        "apikey"
            | "authorization"
            | "proxyauthorization"
            | "accesstoken"
            | "refreshtoken"
            | "idtoken"
            | "authtoken"
            | "password"
            | "passwd"
            | "secret"
            | "clientsecret"
            | "cookie"
            | "setcookie"
            | "credential"
            | "credentials"
    ) || [
        "apikey",
        "authorization",
        "password",
        "passwd",
        "secret",
        "credential",
        "cookie",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
        || normalized.ends_with("token")
}

fn looks_like_secret(token: &str) -> bool {
    token.len() >= 12
        && ["sk-", "xai-", "key-"]
            .iter()
            .any(|prefix| token.to_ascii_lowercase().starts_with(prefix))
}

pub struct ApprovalRequest {
    pub id: String,
    pub name: String,
    pub arguments: String,
    response: oneshot::Sender<ApprovalDecision>,
}

impl ApprovalRequest {
    pub fn resolve(self, decision: ApprovalDecision) {
        let _ = self.response.send(decision);
    }
}

#[derive(Clone, Default)]
pub struct ApprovalGate {
    mode: ApprovalMode,
}

#[derive(Clone, Default)]
enum ApprovalMode {
    AllowAll,
    Request(mpsc::UnboundedSender<ApprovalRequest>),
    #[default]
    Deny,
}

impl ApprovalGate {
    #[must_use]
    pub fn allow_all() -> Self {
        Self {
            mode: ApprovalMode::AllowAll,
        }
    }

    #[must_use]
    pub fn channel() -> (Self, mpsc::UnboundedReceiver<ApprovalRequest>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            Self {
                mode: ApprovalMode::Request(sender),
            },
            receiver,
        )
    }

    #[must_use]
    pub(crate) fn bypasses_approval(&self) -> bool {
        matches!(self.mode, ApprovalMode::AllowAll)
    }

    pub(crate) async fn decide(
        &self,
        id: &str,
        name: &str,
        arguments: &str,
        cancel: &CancellationToken,
    ) -> Option<ApprovalDecision> {
        match &self.mode {
            ApprovalMode::AllowAll => Some(ApprovalDecision::ApproveOnce),
            ApprovalMode::Deny => Some(ApprovalDecision::Deny),
            ApprovalMode::Request(sender) => {
                let (response, decision) = oneshot::channel();
                if sender
                    .send(ApprovalRequest {
                        id: id.to_string(),
                        name: name.to_string(),
                        arguments: arguments.to_string(),
                        response,
                    })
                    .is_err()
                {
                    return Some(ApprovalDecision::Deny);
                }
                tokio::select! {
                    () = cancel.cancelled() => None,
                    result = decision => Some(result.unwrap_or(ApprovalDecision::Deny)),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SensitiveTextRedactor, format_tool_arguments, redact_sensitive_text};

    #[test]
    fn redacts_structured_and_command_credentials() {
        let formatted = format_tool_arguments(
            r#"{"api_key":"sk-secret-value","nested":{"password":"hello"},"command":"curl -H 'Authorization: Bearer abc123' --token=xyz"}"#,
        );
        assert!(!formatted.contains("sk-secret-value"));
        assert!(!formatted.contains("hello"));
        assert!(!formatted.contains("abc123"));
        assert!(!formatted.contains("xyz"));
        assert!(formatted.contains("<redacted>"));
    }

    #[test]
    fn preserves_non_secret_command_arguments() {
        assert_eq!(
            redact_sensitive_text("cargo test --package mcode"),
            "cargo test --package mcode"
        );
    }

    #[test]
    fn redacts_credentials_split_across_stream_chunks() {
        let mut redactor = SensitiveTextRedactor::default();
        assert_eq!(
            redactor.push("Authorization: Bearer "),
            "Authorization: Bearer "
        );
        let mut output = redactor.push("abc123\nnext line");
        output.push_str(&redactor.finish());
        assert!(!output.contains("abc123"));
        assert!(output.contains("<redacted>"));
        assert!(output.ends_with("next line"));
    }
}
