use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    ApproveOnce,
    ApproveForSession,
    Deny,
}

#[must_use]
pub fn format_tool_arguments(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| arguments.to_string())
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
