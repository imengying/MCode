use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::config::{AppConfig, ReasoningEffort, mcode_home_dir};
use crate::protocol::{ChatMessage, MessageRole, ToolCall, Usage};

const SESSION_VERSION: u32 = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionMetadata {
    pub provider: String,
    pub model: String,
    pub reasoning_effort: ReasoningEffort,
}

impl From<&AppConfig> for SessionMetadata {
    fn from(config: &AppConfig) -> Self {
        Self {
            provider: config.provider.clone(),
            model: config.model.clone(),
            reasoning_effort: config.reasoning_effort,
        }
    }
}

#[derive(Debug)]
pub struct Session {
    id: Uuid,
    cwd: PathBuf,
    metadata: SessionMetadata,
    created_at: u64,
    persistence_base: Option<PathBuf>,
    path: Option<PathBuf>,
    writer: Option<SessionWriter>,
    messages: Vec<ChatMessage>,
    context_excluded_messages: BTreeSet<usize>,
    latest_compaction: Option<CompactionCheckpoint>,
    total_usage: Usage,
    active_run: Option<ActiveRun>,
}

#[derive(Debug)]
struct SessionWriter(File);

impl Deref for SessionWriter {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SessionWriter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for SessionWriter {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolReplayPolicy {
    Safe,
    Never,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolIntent {
    pub run_id: Uuid,
    pub call: ToolCall,
    pub result_id: Uuid,
    pub replay: ToolReplayPolicy,
    pub created_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingToolCall {
    pub call: ToolCall,
    pub intent: Option<ToolIntent>,
}

#[derive(Debug)]
struct ActiveRun {
    id: Uuid,
    message_start: usize,
    generation_attempts: usize,
    tools: BTreeMap<String, ToolIntent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompactionCheckpoint {
    pub summary: String,
    pub first_kept_message_index: usize,
    pub message_count: usize,
    pub tokens_before: u64,
    pub created_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub read_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modified_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SessionSummary {
    pub id: Uuid,
    pub created_at: u64,
    pub provider: String,
    pub model: String,
    pub reasoning_effort: ReasoningEffort,
    pub message_count: usize,
    pub total_usage: Usage,
    pub has_pending_run: bool,
    pub path: PathBuf,
}

#[derive(Debug)]
struct SessionCandidate {
    path: PathBuf,
    id: Uuid,
    created_at: u64,
    has_messages: bool,
}

#[derive(Deserialize)]
struct SessionRecordTag {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, thiserror::Error)]
#[error("session is already open by another MCode process: {path} ({source})")]
struct SessionAlreadyOpen {
    path: PathBuf,
    #[source]
    source: io::Error,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionRecord {
    Session {
        version: u32,
        id: Uuid,
        cwd: PathBuf,
        #[serde(flatten)]
        metadata: SessionMetadata,
        created_at: u64,
    },
    Message {
        message: ChatMessage,
    },
    RunStarted {
        run_id: Uuid,
        message: ChatMessage,
        created_at: u64,
    },
    GenerationStarted {
        run_id: Uuid,
        attempt: usize,
        created_at: u64,
    },
    AssistantCompleted {
        run_id: Uuid,
        message: ChatMessage,
        usage: Usage,
    },
    AssistantDiscarded {
        run_id: Uuid,
        message: ChatMessage,
        usage: Usage,
    },
    ToolStarted {
        #[serde(flatten)]
        intent: ToolIntent,
    },
    ToolCompleted {
        run_id: Uuid,
        result_id: Uuid,
        message: ChatMessage,
    },
    RunFinished {
        run_id: Uuid,
        outcome: RunOutcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        created_at: u64,
    },
    ModelChanged {
        provider: String,
        model: String,
    },
    ReasoningChanged {
        reasoning_effort: ReasoningEffort,
    },
    Compaction {
        #[serde(flatten)]
        checkpoint: CompactionCheckpoint,
    },
}

impl Session {
    pub fn create(cwd: &Path, metadata: SessionMetadata, persist: bool) -> Result<Self> {
        let cwd = cwd
            .canonicalize()
            .with_context(|| format!("invalid session directory: {}", cwd.display()))?;
        let persistence_base = persist.then(default_session_base).transpose()?;
        Ok(Self::new_pending(cwd, metadata, persistence_base))
    }

    pub fn create_in(base: &Path, cwd: &Path, metadata: SessionMetadata) -> Result<Self> {
        let cwd = cwd
            .canonicalize()
            .with_context(|| format!("invalid session directory: {}", cwd.display()))?;
        let mut session = Self::new_pending(cwd, metadata, Some(base.to_path_buf()));
        session.materialize()?;
        Ok(session)
    }

    fn new_pending(
        cwd: PathBuf,
        metadata: SessionMetadata,
        persistence_base: Option<PathBuf>,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            cwd,
            metadata,
            created_at: unix_timestamp(),
            persistence_base,
            path: None,
            writer: None,
            messages: Vec::new(),
            context_excluded_messages: BTreeSet::new(),
            latest_compaction: None,
            total_usage: Usage::default(),
            active_run: None,
        }
    }

    pub fn resume(cwd: &Path, selector: Option<&str>) -> Result<Self> {
        let base = default_session_base()?;
        Self::resume_in(&base, cwd, selector)
    }

    pub fn resume_in(base: &Path, cwd: &Path, selector: Option<&str>) -> Result<Self> {
        let cwd = cwd
            .canonicalize()
            .with_context(|| format!("invalid session directory: {}", cwd.display()))?;
        let directory = project_session_dir(base, &cwd);
        if let Some(selector) = selector {
            let direct = Path::new(selector);
            if direct.is_file() {
                return Self::load_for_project(direct, &directory, &cwd);
            }
        }

        let mut candidates = session_candidates(&directory)?;
        if let Some(selector) = selector.filter(|value| !value.eq_ignore_ascii_case("last")) {
            candidates.retain(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains(selector))
            });
        }
        let prefer_non_empty = selector.is_none_or(|value| value.eq_ignore_ascii_case("last"));
        let mut load_error = None;
        let mut sessions = Vec::new();
        for path in candidates {
            match scan_session_candidate(&path, &cwd) {
                Ok(session) => sessions.push(session),
                Err(error) if prefer_non_empty => load_error = Some(error),
                Err(error) => return Err(error),
            }
        }
        sessions.sort_by_key(|session| Reverse((session.created_at, session.id)));
        if prefer_non_empty {
            for has_messages in [true, false] {
                for session in sessions
                    .iter()
                    .filter(|session| session.has_messages == has_messages)
                {
                    match Self::load_for_project(&session.path, &directory, &cwd) {
                        Ok(session) => return Ok(session),
                        Err(error) if error.downcast_ref::<SessionAlreadyOpen>().is_some() => {
                            return Err(error);
                        }
                        Err(error) => load_error = Some(error),
                    }
                }
            }
        } else if let Some(session) = sessions.first() {
            return Self::load_for_project(&session.path, &directory, &cwd);
        }
        if sessions.is_empty() || prefer_non_empty {
            if let Some(error) = load_error {
                return Err(error);
            }
            return Err(if let Some(selector) = selector {
                anyhow!("no session matching {selector:?} for {}", cwd.display())
            } else {
                anyhow!("no previous session for {}", cwd.display())
            });
        }
        unreachable!("a non-empty explicit candidate list returned above")
    }

    pub fn list(cwd: &Path) -> Result<Vec<SessionSummary>> {
        let base = default_session_base()?;
        Self::list_in(&base, cwd)
    }

    pub fn list_in(base: &Path, cwd: &Path) -> Result<Vec<SessionSummary>> {
        let cwd = cwd
            .canonicalize()
            .with_context(|| format!("invalid session directory: {}", cwd.display()))?;
        let directory = project_session_dir(base, &cwd);
        let mut sessions = session_candidates(&directory)?
            .into_iter()
            .map(|path| {
                let session = Self::load_readonly(&path)?;
                if session.cwd != cwd {
                    bail!(
                        "session {} belongs to {}, not {}",
                        session.id,
                        session.cwd.display(),
                        cwd.display()
                    );
                }
                Ok(SessionSummary {
                    id: session.id,
                    created_at: session.created_at,
                    provider: session.metadata.provider,
                    model: session.metadata.model,
                    reasoning_effort: session.metadata.reasoning_effort,
                    message_count: session.messages.len(),
                    total_usage: session.total_usage,
                    has_pending_run: session.active_run.is_some(),
                    path,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        sessions.sort_by_key(|session| std::cmp::Reverse(session.created_at));
        Ok(sessions)
    }

    pub fn storage_directory(cwd: &Path) -> Result<PathBuf> {
        let cwd = cwd
            .canonicalize()
            .with_context(|| format!("invalid session directory: {}", cwd.display()))?;
        Ok(project_session_dir(&default_session_base()?, &cwd))
    }

    pub fn delete(cwd: &Path, selector: &str) -> Result<Uuid> {
        let base = default_session_base()?;
        Self::delete_in(&base, cwd, selector)
    }

    pub fn delete_in(base: &Path, cwd: &Path, selector: &str) -> Result<Uuid> {
        let cwd = cwd
            .canonicalize()
            .with_context(|| format!("invalid session directory: {}", cwd.display()))?;
        let selector = selector.trim();
        if selector.is_empty() {
            bail!("session selector cannot be empty");
        }
        let directory = project_session_dir(base, &cwd);
        let mut matches = session_candidates(&directory)?
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains(selector))
            })
            .collect::<Vec<_>>();
        match matches.len() {
            0 => bail!("no session matching {selector:?} for {}", cwd.display()),
            1 => {}
            count => bail!(
                "session selector {selector:?} matches {count} sessions; use the complete UUID"
            ),
        }
        let path = matches
            .pop()
            .ok_or_else(|| anyhow!("session match disappeared while resolving {selector:?}"))?;
        let mut session = Self::load_writable(&path)?;
        if session.cwd != cwd {
            bail!(
                "session {} belongs to a different working directory",
                session.id
            );
        }
        let id = session.id;
        session.writer.take();
        drop(session);
        remove_session_files(&path)?;
        remove_project_session_directory_if_empty(&directory)?;
        Ok(id)
    }

    fn load_readonly(path: &Path) -> Result<Self> {
        Self::load_with_writer(path, None)
    }

    fn load_writable(path: &Path) -> Result<Self> {
        let writer = open_session_writer(path)?;
        Self::load_with_writer(path, Some(writer))
    }

    fn load_with_writer(path: &Path, mut writer: Option<SessionWriter>) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        if bytes.is_empty() {
            bail!("session is empty: {}", path.display());
        }
        let utf8_len = match std::str::from_utf8(&bytes) {
            Ok(_) => bytes.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(error) => {
                return Err(anyhow!(error))
                    .with_context(|| format!("session is not valid UTF-8: {}", path.display()));
            }
        };
        let text = std::str::from_utf8(&bytes[..utf8_len])
            .with_context(|| format!("session is not valid UTF-8: {}", path.display()))?;
        let mut lines = text.split_inclusive('\n').enumerate();
        let (_, header) = lines
            .next()
            .ok_or_else(|| anyhow!("session is empty: {}", path.display()))?;
        let header = header.trim_end_matches(['\r', '\n']);
        let header: SessionRecord = serde_json::from_str(header)
            .with_context(|| format!("invalid session header: {}", path.display()))?;
        let SessionRecord::Session {
            version,
            id,
            cwd,
            mut metadata,
            created_at,
        } = header
        else {
            bail!("first session record is not a header: {}", path.display());
        };
        if version != SESSION_VERSION {
            bail!("unsupported session version {version}; expected {SESSION_VERSION}");
        }

        let mut messages = Vec::new();
        let mut context_excluded_messages = BTreeSet::new();
        let mut latest_compaction = None;
        let mut total_usage = Usage::default();
        let mut active_run: Option<ActiveRun> = None;
        let mut valid_len = text.split_inclusive('\n').next().map_or(0, str::len);
        let mut recovered_tail = utf8_len != bytes.len();
        for (index, line) in lines {
            let terminated = line.ends_with('\n');
            let record_text = line.trim_end_matches(['\r', '\n']);
            if record_text.trim().is_empty() {
                valid_len = valid_len.saturating_add(line.len());
                continue;
            }
            let record: SessionRecord = match serde_json::from_str(record_text) {
                Ok(record) => record,
                Err(_) if !terminated => {
                    recovered_tail = true;
                    break;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("invalid session record at {}:{}", path.display(), index + 1)
                    });
                }
            };
            valid_len = valid_len.saturating_add(line.len());
            match record {
                SessionRecord::Message { message } => messages.push(message),
                SessionRecord::RunStarted {
                    run_id,
                    message,
                    created_at: _,
                } => {
                    if active_run.is_some() {
                        bail!("overlapping runs at {}:{}", path.display(), index + 1);
                    }
                    if message.role != MessageRole::User {
                        bail!(
                            "run start is not a user message at {}:{}",
                            path.display(),
                            index + 1
                        );
                    }
                    let message_start = messages.len();
                    messages.push(message);
                    active_run = Some(ActiveRun {
                        id: run_id,
                        message_start,
                        generation_attempts: 0,
                        tools: BTreeMap::new(),
                    });
                }
                SessionRecord::GenerationStarted {
                    run_id,
                    attempt,
                    created_at: _,
                } => {
                    let active = active_run.as_mut().ok_or_else(|| {
                        anyhow!(
                            "generation outside a run at {}:{}",
                            path.display(),
                            index + 1
                        )
                    })?;
                    if active.id != run_id || attempt != active.generation_attempts + 1 {
                        bail!(
                            "invalid generation sequence at {}:{}",
                            path.display(),
                            index + 1
                        );
                    }
                    active.generation_attempts = attempt;
                }
                SessionRecord::AssistantCompleted {
                    run_id,
                    message,
                    usage,
                } => {
                    let active = active_run.as_mut().ok_or_else(|| {
                        anyhow!(
                            "assistant completion outside a run at {}:{}",
                            path.display(),
                            index + 1
                        )
                    })?;
                    if active.id != run_id
                        || active.generation_attempts == 0
                        || message.role != MessageRole::Assistant
                    {
                        bail!(
                            "invalid assistant completion at {}:{}",
                            path.display(),
                            index + 1
                        );
                    }
                    active.generation_attempts = 0;
                    total_usage = total_usage.saturating_add(usage);
                    messages.push(message);
                }
                SessionRecord::AssistantDiscarded {
                    run_id,
                    message,
                    usage,
                } => {
                    let active = active_run.as_ref().ok_or_else(|| {
                        anyhow!(
                            "discarded assistant response outside a run at {}:{}",
                            path.display(),
                            index + 1
                        )
                    })?;
                    if active.id != run_id
                        || active.generation_attempts == 0
                        || message.role != MessageRole::Assistant
                    {
                        bail!(
                            "invalid discarded assistant response at {}:{}",
                            path.display(),
                            index + 1
                        );
                    }
                    total_usage = total_usage.saturating_add(usage);
                    context_excluded_messages.insert(messages.len());
                    messages.push(message);
                }
                SessionRecord::ToolStarted { intent } => {
                    let active = active_run.as_mut().ok_or_else(|| {
                        anyhow!(
                            "tool intent outside a run at {}:{}",
                            path.display(),
                            index + 1
                        )
                    })?;
                    if active.id != intent.run_id || active.tools.contains_key(&intent.call.id) {
                        bail!("invalid tool intent at {}:{}", path.display(), index + 1);
                    }
                    active.tools.insert(intent.call.id.clone(), intent);
                }
                SessionRecord::ToolCompleted {
                    run_id,
                    result_id,
                    message,
                } => {
                    let active = active_run.as_mut().ok_or_else(|| {
                        anyhow!(
                            "tool result outside a run at {}:{}",
                            path.display(),
                            index + 1
                        )
                    })?;
                    let call_id = message.tool_call_id.as_deref().ok_or_else(|| {
                        anyhow!(
                            "tool result is missing its call id at {}:{}",
                            path.display(),
                            index + 1
                        )
                    })?;
                    let intent = active.tools.get(call_id).ok_or_else(|| {
                        anyhow!(
                            "tool result has no matching intent at {}:{}",
                            path.display(),
                            index + 1
                        )
                    })?;
                    if active.id != run_id
                        || intent.result_id != result_id
                        || message.role != MessageRole::Tool
                    {
                        bail!("invalid tool result at {}:{}", path.display(), index + 1);
                    }
                    active.tools.remove(call_id);
                    messages.push(message);
                }
                SessionRecord::RunFinished {
                    run_id,
                    outcome: _,
                    error: _,
                    created_at: _,
                } => {
                    let active = active_run.as_ref().ok_or_else(|| {
                        anyhow!(
                            "run finish without a start at {}:{}",
                            path.display(),
                            index + 1
                        )
                    })?;
                    if active.id != run_id || !active.tools.is_empty() {
                        bail!("invalid run finish at {}:{}", path.display(), index + 1);
                    }
                    active_run = None;
                }
                SessionRecord::ModelChanged { provider, model } => {
                    metadata.provider = provider;
                    metadata.model = model;
                }
                SessionRecord::ReasoningChanged {
                    reasoning_effort: next_effort,
                } => metadata.reasoning_effort = next_effort,
                SessionRecord::Compaction { checkpoint } => {
                    if checkpoint.message_count != messages.len() {
                        bail!(
                            "invalid compaction message count at {}:{}: expected {}, found {}",
                            path.display(),
                            index + 1,
                            messages.len(),
                            checkpoint.message_count
                        );
                    }
                    if checkpoint.first_kept_message_index > checkpoint.message_count {
                        bail!(
                            "invalid compaction boundary at {}:{}",
                            path.display(),
                            index + 1
                        );
                    }
                    if let Some(usage) = checkpoint.usage {
                        total_usage = total_usage.saturating_add(usage);
                    }
                    latest_compaction = Some(checkpoint);
                }
                SessionRecord::Session { .. } => {
                    bail!(
                        "unexpected session header at {}:{}",
                        path.display(),
                        index + 1
                    );
                }
            }
        }

        let missing_newline = valid_len > 0 && bytes.get(valid_len - 1) != Some(&b'\n');
        if (recovered_tail || missing_newline)
            && let Some(file) = writer.as_mut()
        {
            repair_session_tail(file, path, valid_len, missing_newline)?;
        }

        Ok(Self {
            id,
            cwd,
            metadata,
            created_at,
            persistence_base: Some(
                path.parent()
                    .and_then(Path::parent)
                    .ok_or_else(|| anyhow!("session path has no storage base: {}", path.display()))?
                    .to_path_buf(),
            ),
            path: Some(path.to_path_buf()),
            writer,
            messages,
            context_excluded_messages,
            latest_compaction,
            total_usage,
            active_run,
        })
    }

    fn load_for_project(path: &Path, directory: &Path, cwd: &Path) -> Result<Self> {
        let path = path
            .canonicalize()
            .with_context(|| format!("failed to resolve session path: {}", path.display()))?;
        if !directory.is_dir() {
            bail!(
                "session path is outside the current project session directory: {}",
                path.display()
            );
        }
        let directory = directory.canonicalize().with_context(|| {
            format!(
                "failed to resolve current project session directory: {}",
                directory.display()
            )
        })?;
        if path.parent() != Some(directory.as_path()) {
            bail!(
                "session path is outside the current project session directory: {}",
                path.display()
            );
        }
        let session = Self::load_writable(&path)?;
        if session.cwd != cwd {
            bail!(
                "session {} belongs to {}, not the current working directory {}",
                session.id,
                session.cwd.display(),
                cwd.display()
            );
        }
        Ok(session)
    }

    pub fn append(&mut self, message: ChatMessage) -> Result<()> {
        self.persist_record(&SessionRecord::Message {
            message: message.clone(),
        })?;
        self.messages.push(message);
        Ok(())
    }

    pub fn append_compaction(
        &mut self,
        summary: String,
        first_kept_message_index: usize,
        tokens_before: u64,
        usage: Option<Usage>,
        read_files: Vec<String>,
        modified_files: Vec<String>,
    ) -> Result<()> {
        if first_kept_message_index > self.messages.len() {
            bail!(
                "compaction boundary {first_kept_message_index} exceeds the session message count {}",
                self.messages.len()
            );
        }
        let checkpoint = CompactionCheckpoint {
            summary,
            first_kept_message_index,
            message_count: self.messages.len(),
            tokens_before,
            created_at: unix_timestamp(),
            usage,
            read_files,
            modified_files,
        };
        self.persist_record(&SessionRecord::Compaction {
            checkpoint: checkpoint.clone(),
        })?;
        if let Some(usage) = checkpoint.usage {
            self.total_usage = self.total_usage.saturating_add(usage);
        }
        self.latest_compaction = Some(checkpoint);
        Ok(())
    }

    pub fn set_model(&mut self, provider: &str, model: &str) -> Result<()> {
        if self.metadata.provider == provider && self.metadata.model == model {
            return Ok(());
        }
        if self.path.is_none() {
            self.metadata.provider = provider.to_string();
            self.metadata.model = model.to_string();
            return Ok(());
        }
        self.persist_record(&SessionRecord::ModelChanged {
            provider: provider.to_string(),
            model: model.to_string(),
        })?;
        self.metadata.provider = provider.to_string();
        self.metadata.model = model.to_string();
        Ok(())
    }

    pub fn set_reasoning_effort(&mut self, effort: ReasoningEffort) -> Result<()> {
        if self.metadata.reasoning_effort == effort {
            return Ok(());
        }
        if self.path.is_none() {
            self.metadata.reasoning_effort = effort;
            return Ok(());
        }
        self.persist_record(&SessionRecord::ReasoningChanged {
            reasoning_effort: effort,
        })?;
        self.metadata.reasoning_effort = effort;
        Ok(())
    }

    pub fn begin_run(&mut self, message: ChatMessage) -> Result<Uuid> {
        if self.active_run.is_some() {
            bail!("session already has an unfinished run");
        }
        if message.role != MessageRole::User {
            bail!("a run must start with a user message");
        }
        let run_id = Uuid::now_v7();
        let created_at = unix_timestamp();
        self.persist_record(&SessionRecord::RunStarted {
            run_id,
            message: message.clone(),
            created_at,
        })?;
        let message_start = self.messages.len();
        self.messages.push(message);
        self.active_run = Some(ActiveRun {
            id: run_id,
            message_start,
            generation_attempts: 0,
            tools: BTreeMap::new(),
        });
        Ok(run_id)
    }

    pub fn start_generation(&mut self, run_id: Uuid) -> Result<()> {
        let active = self
            .active_run
            .as_ref()
            .ok_or_else(|| anyhow!("cannot start a generation without an active run"))?;
        if active.id != run_id {
            bail!("generation belongs to a different run");
        }
        let attempt = active.generation_attempts.saturating_add(1);
        self.persist_record(&SessionRecord::GenerationStarted {
            run_id,
            attempt,
            created_at: unix_timestamp(),
        })?;
        if let Some(active) = self.active_run.as_mut() {
            active.generation_attempts = attempt;
        }
        Ok(())
    }

    pub fn complete_generation(
        &mut self,
        run_id: Uuid,
        message: ChatMessage,
        usage: Usage,
    ) -> Result<()> {
        let active = self
            .active_run
            .as_ref()
            .ok_or_else(|| anyhow!("cannot complete a generation without an active run"))?;
        if active.id != run_id || active.generation_attempts == 0 {
            bail!("generation completion does not match the active run");
        }
        if message.role != MessageRole::Assistant {
            bail!("generation completion must contain an assistant message");
        }
        self.persist_record(&SessionRecord::AssistantCompleted {
            run_id,
            message: message.clone(),
            usage,
        })?;
        if let Some(active) = self.active_run.as_mut() {
            active.generation_attempts = 0;
        }
        self.total_usage = self.total_usage.saturating_add(usage);
        self.messages.push(message);
        Ok(())
    }

    pub fn discard_generation(
        &mut self,
        run_id: Uuid,
        message: ChatMessage,
        usage: Usage,
    ) -> Result<()> {
        let active = self
            .active_run
            .as_ref()
            .ok_or_else(|| anyhow!("cannot discard a generation without an active run"))?;
        if active.id != run_id || active.generation_attempts == 0 {
            bail!("discarded generation does not match the active run");
        }
        if message.role != MessageRole::Assistant {
            bail!("discarded generation must contain an assistant message");
        }
        self.persist_record(&SessionRecord::AssistantDiscarded {
            run_id,
            message: message.clone(),
            usage,
        })?;
        self.total_usage = self.total_usage.saturating_add(usage);
        self.context_excluded_messages.insert(self.messages.len());
        self.messages.push(message);
        Ok(())
    }

    pub fn start_tool(
        &mut self,
        run_id: Uuid,
        call: ToolCall,
        replay: ToolReplayPolicy,
    ) -> Result<ToolIntent> {
        let active = self
            .active_run
            .as_ref()
            .ok_or_else(|| anyhow!("cannot start a tool without an active run"))?;
        if active.id != run_id || active.tools.contains_key(&call.id) {
            bail!("tool call does not match the active run");
        }
        let intent = ToolIntent {
            run_id,
            call,
            result_id: Uuid::now_v7(),
            replay,
            created_at: unix_timestamp(),
        };
        self.persist_record(&SessionRecord::ToolStarted {
            intent: intent.clone(),
        })?;
        if let Some(active) = self.active_run.as_mut() {
            active.tools.insert(intent.call.id.clone(), intent.clone());
        }
        Ok(intent)
    }

    pub fn complete_tool(&mut self, intent: &ToolIntent, message: ChatMessage) -> Result<()> {
        let active = self
            .active_run
            .as_ref()
            .ok_or_else(|| anyhow!("cannot complete a tool without an active run"))?;
        let current = active
            .tools
            .get(&intent.call.id)
            .ok_or_else(|| anyhow!("tool call has no durable start record"))?;
        if current != intent
            || message.role != MessageRole::Tool
            || message.tool_call_id.as_deref() != Some(intent.call.id.as_str())
        {
            bail!("tool result does not match its durable start record");
        }
        self.persist_record(&SessionRecord::ToolCompleted {
            run_id: intent.run_id,
            result_id: intent.result_id,
            message: message.clone(),
        })?;
        if let Some(active) = self.active_run.as_mut() {
            active.tools.remove(&intent.call.id);
        }
        self.messages.push(message);
        Ok(())
    }

    pub(crate) fn save_tool_output(
        &self,
        intent: &ToolIntent,
        output: &str,
    ) -> Result<Option<PathBuf>> {
        let Some(session_path) = self.path.as_deref() else {
            return Ok(None);
        };
        let directory = session_artifact_directory(session_path).join("tool-results");
        create_private_directory(&directory)?;
        let path = directory.join(format!("{}.txt", intent.result_id));
        write_private_file(&path, output.as_bytes())?;
        Ok(Some(path))
    }

    pub fn finish_run(&mut self, run_id: Uuid, outcome: RunOutcome) -> Result<()> {
        self.finish_run_with_error(run_id, outcome, None)
    }

    pub fn finish_run_with_error(
        &mut self,
        run_id: Uuid,
        outcome: RunOutcome,
        error: Option<String>,
    ) -> Result<()> {
        let active = self
            .active_run
            .as_ref()
            .ok_or_else(|| anyhow!("cannot finish a run that is not active"))?;
        if active.id != run_id {
            bail!("run finish belongs to a different run");
        }
        if !active.tools.is_empty() {
            bail!("cannot finish a run with unresolved tool calls");
        }
        self.persist_record(&SessionRecord::RunFinished {
            run_id,
            outcome,
            error,
            created_at: unix_timestamp(),
        })?;
        self.active_run = None;
        Ok(())
    }

    #[must_use]
    pub fn active_run_id(&self) -> Option<Uuid> {
        self.active_run.as_ref().map(|run| run.id)
    }

    #[must_use]
    pub fn has_pending_run(&self) -> bool {
        self.active_run.is_some()
    }

    #[must_use]
    pub fn active_run_has_final_response(&self) -> bool {
        let Some(active) = &self.active_run else {
            return false;
        };
        self.messages
            .iter()
            .enumerate()
            .skip(active.message_start)
            .filter(|(index, _)| !self.context_excluded_messages.contains(index))
            .map(|(_, message)| message)
            .next_back()
            .is_some_and(|message| {
                message.role == MessageRole::Assistant && message.tool_calls.is_empty()
            })
    }

    pub fn pending_tool_calls(&self) -> Result<Vec<PendingToolCall>> {
        let Some(active) = &self.active_run else {
            return Ok(Vec::new());
        };
        let active_messages = self
            .messages
            .get(active.message_start..)
            .ok_or_else(|| anyhow!("active run message boundary is invalid"))?;
        let Some((assistant_index, assistant)) =
            active_messages
                .iter()
                .enumerate()
                .rev()
                .find(|(index, message)| {
                    message.role == MessageRole::Assistant
                        && !self
                            .context_excluded_messages
                            .contains(&(active.message_start + index))
                })
        else {
            if active.tools.is_empty() {
                return Ok(Vec::new());
            }
            bail!("active run contains tool intents without an assistant message");
        };
        if assistant.tool_calls.is_empty() {
            if active.tools.is_empty() {
                return Ok(Vec::new());
            }
            bail!("active run contains tool intents after a final assistant response");
        }
        let completed = active_messages[assistant_index + 1..]
            .iter()
            .filter(|message| message.role == MessageRole::Tool)
            .filter_map(|message| message.tool_call_id.as_deref())
            .collect::<std::collections::BTreeSet<_>>();
        let pending = assistant
            .tool_calls
            .iter()
            .filter(|call| !completed.contains(call.id.as_str()))
            .map(|call| PendingToolCall {
                call: call.clone(),
                intent: active.tools.get(&call.id).cloned(),
            })
            .collect::<Vec<_>>();
        if active
            .tools
            .keys()
            .any(|call_id| !pending.iter().any(|pending| pending.call.id == *call_id))
        {
            bail!("active run contains an orphaned tool intent");
        }
        Ok(pending)
    }

    #[must_use]
    pub const fn total_usage(&self) -> Usage {
        self.total_usage
    }

    pub fn fresh_with_reasoning_effort(&self, reasoning_effort: ReasoningEffort) -> Result<Self> {
        let mut metadata = self.metadata.clone();
        metadata.reasoning_effort = reasoning_effort;
        Ok(Self::new_pending(
            self.cwd.clone(),
            metadata,
            self.persistence_base.clone(),
        ))
    }

    pub fn delete_current(&mut self) -> Result<Uuid> {
        let base = default_session_base()?;
        self.delete_current_in(&base)
    }

    fn delete_current_in(&mut self, base: &Path) -> Result<Uuid> {
        let Some(path) = self.path.clone() else {
            if self.persistence_base.take().is_some() {
                return Ok(self.id);
            }
            bail!("this session is not persisted");
        };
        let expected_directory = project_session_dir(base, &self.cwd);
        let actual_directory = path
            .parent()
            .ok_or_else(|| anyhow!("session path has no parent: {}", path.display()))?;
        let expected_directory = expected_directory.canonicalize().with_context(|| {
            format!(
                "failed to resolve session directory: {}",
                expected_directory.display()
            )
        })?;
        let actual_directory = actual_directory.canonicalize().with_context(|| {
            format!(
                "failed to resolve session directory: {}",
                actual_directory.display()
            )
        })?;
        if actual_directory != expected_directory {
            bail!("refusing to delete a session outside the current project session directory");
        }
        self.writer.take();
        remove_session_files(&path)?;
        self.path = None;
        self.persistence_base = None;
        remove_project_session_directory_if_empty(&expected_directory)?;
        Ok(self.id)
    }

    #[must_use]
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    #[must_use]
    pub fn context_messages(&self) -> Vec<ChatMessage> {
        let Some(compaction) = &self.latest_compaction else {
            return self
                .messages
                .iter()
                .enumerate()
                .filter(|(index, _)| !self.context_excluded_messages.contains(index))
                .map(|(_, message)| message.clone())
                .collect();
        };
        let first_kept = compaction.first_kept_message_index.min(self.messages.len());
        let mut messages = Vec::with_capacity(self.messages.len() - first_kept + 1);
        messages.push(ChatMessage::user(format!(
            "The conversation history before this point was compacted into the following summary:\n\n<summary>\n{}\n</summary>",
            compaction.summary
        )));
        messages.extend(
            self.messages[first_kept..]
                .iter()
                .enumerate()
                .filter(|(offset, _)| {
                    !self
                        .context_excluded_messages
                        .contains(&(first_kept + offset))
                })
                .map(|(_, message)| message.clone()),
        );
        messages
    }

    #[must_use]
    pub(crate) fn message_is_in_context(&self, index: usize) -> bool {
        index < self.messages.len() && !self.context_excluded_messages.contains(&index)
    }

    #[must_use]
    pub const fn latest_compaction(&self) -> Option<&CompactionCheckpoint> {
        self.latest_compaction.as_ref()
    }

    #[must_use]
    pub fn is_compacted_at_tip(&self) -> bool {
        self.latest_compaction.as_ref().is_some_and(|checkpoint| {
            checkpoint.message_count <= self.messages.len()
                && (checkpoint.message_count..self.messages.len())
                    .all(|index| self.context_excluded_messages.contains(&index))
        })
    }

    #[must_use]
    pub fn id(&self) -> Uuid {
        self.id
    }

    #[must_use]
    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    #[must_use]
    pub fn model(&self) -> &str {
        &self.metadata.model
    }

    #[must_use]
    pub fn provider(&self) -> &str {
        &self.metadata.provider
    }

    #[must_use]
    pub const fn reasoning_effort(&self) -> ReasoningEffort {
        self.metadata.reasoning_effort
    }

    #[must_use]
    pub fn model_selector(&self) -> String {
        format!("{}/{}", self.provider(), self.model())
    }

    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub(crate) fn persistence_path(&self) -> Result<Option<PathBuf>> {
        if let Some(path) = &self.path {
            return Ok(Some(path.clone()));
        }
        let Some(base) = self.persistence_base.as_deref() else {
            return Ok(None);
        };
        Ok(Some(
            project_session_dir(base, &self.cwd).join(rollout_filename(self.created_at, self.id)?),
        ))
    }

    fn materialize(&mut self) -> Result<()> {
        if self.writer.is_some() {
            return Ok(());
        }
        let Some(base) = self.persistence_base.as_deref() else {
            return Ok(());
        };
        if let Some(path) = &self.path {
            bail!(
                "persisted session does not hold its writer lock: {}",
                path.display()
            );
        }

        let directory = project_session_dir(base, &self.cwd);
        create_private_directory(&directory)?;
        let path = directory.join(rollout_filename(self.created_at, self.id)?);
        let header = SessionRecord::Session {
            version: SESSION_VERSION,
            id: self.id,
            cwd: self.cwd.clone(),
            metadata: self.metadata.clone(),
            created_at: self.created_at,
        };
        let writer = create_session_file(&path, &header)?;
        self.path = Some(path);
        self.writer = Some(writer);
        Ok(())
    }

    fn persist_record(&mut self, record: &SessionRecord) -> Result<()> {
        if self.persistence_base.is_none() {
            return Ok(());
        }
        self.materialize()?;
        let path = self
            .path
            .as_deref()
            .ok_or_else(|| anyhow!("persisted session has no path"))?;
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| anyhow!("persisted session does not hold its writer lock"))?;
        append_record(writer, path, record)
    }
}

fn create_private_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("failed to create {}", path.display()))?;
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("failed to protect session directory: {}", path.display()))?;
    Ok(())
}

fn write_private_file(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("tool output path has no parent: {}", path.display()))?;
    let temporary = parent.join(format!(".{}.tmp", Uuid::now_v7()));
    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options
            .open(&temporary)
            .with_context(|| format!("failed to create tool output: {}", temporary.display()))?;
        file.write_all(contents)
            .with_context(|| format!("failed to write tool output: {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to flush tool output: {}", temporary.display()))?;
        drop(file);
        fs::rename(&temporary, path)
            .with_context(|| format!("failed to save tool output: {}", path.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn create_session_file(path: &Path, header: &SessionRecord) -> Result<SessionWriter> {
    let mut options = OpenOptions::new();
    options.read(true).append(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options
        .open(path)
        .with_context(|| format!("failed to create session: {}", path.display()))?;
    file.try_lock()
        .map_err(|error| anyhow!("failed to lock new session {}: {error}", path.display()))?;
    append_record(&mut file, path, header)?;
    Ok(SessionWriter(file))
}

fn open_session_writer(path: &Path) -> Result<SessionWriter> {
    let file = OpenOptions::new()
        .read(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open session: {}", path.display()))?;
    file.try_lock().map_err(|source| SessionAlreadyOpen {
        path: path.to_path_buf(),
        source: source.into(),
    })?;
    Ok(SessionWriter(file))
}

fn append_record(file: &mut File, path: &Path, record: &SessionRecord) -> Result<()> {
    let mut encoded = serde_json::to_vec(record)
        .with_context(|| format!("failed to serialize session: {}", path.display()))?;
    encoded.push(b'\n');
    file.write_all(&encoded)
        .with_context(|| format!("failed to write session: {}", path.display()))?;
    file.sync_data()
        .with_context(|| format!("failed to flush session: {}", path.display()))
}

fn repair_session_tail(
    file: &mut File,
    path: &Path,
    valid_len: usize,
    add_newline: bool,
) -> Result<()> {
    let valid_len = u64::try_from(valid_len).context("session is too large to recover")?;
    file.set_len(valid_len).with_context(|| {
        format!(
            "failed to truncate damaged session tail: {}",
            path.display()
        )
    })?;
    file.sync_data()
        .with_context(|| format!("failed to flush recovered session: {}", path.display()))?;
    if add_newline {
        file.write_all(b"\n")
            .with_context(|| format!("failed to finish session recovery: {}", path.display()))?;
        file.sync_data()
            .with_context(|| format!("failed to flush recovered session: {}", path.display()))?;
    }
    Ok(())
}

fn default_session_base() -> Result<PathBuf> {
    mcode_home_dir()
        .map(|home| home.join("sessions"))
        .ok_or_else(|| anyhow!("could not determine home directory for session storage"))
}

fn project_session_dir(base: &Path, cwd: &Path) -> PathBuf {
    use std::fmt::Write as _;

    let digest = Sha256::digest(cwd.to_string_lossy().as_bytes());
    let mut key = String::with_capacity(16);
    for byte in &digest[..8] {
        let _ = write!(key, "{byte:02x}");
    }
    base.join(key)
}

fn session_artifact_directory(path: &Path) -> PathBuf {
    path.with_extension("")
}

fn remove_session_artifacts(path: &Path) -> Result<()> {
    let directory = session_artifact_directory(path);
    let metadata = match fs::symlink_metadata(&directory) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to inspect session artifacts: {}",
                    directory.display()
                )
            });
        }
    };
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        bail!(
            "refusing to delete invalid session artifact directory: {}",
            directory.display()
        );
    }
    fs::remove_dir_all(&directory).with_context(|| {
        format!(
            "failed to delete session artifacts: {}",
            directory.display()
        )
    })
}

fn remove_session_files(path: &Path) -> Result<()> {
    remove_session_artifacts(path)?;
    fs::remove_file(path).with_context(|| format!("failed to delete session: {}", path.display()))
}

fn remove_project_session_directory_if_empty(directory: &Path) -> Result<()> {
    match fs::remove_dir(directory) {
        Ok(()) => Ok(()),
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::DirectoryNotEmpty
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "failed to remove empty session directory: {}",
                directory.display()
            )
        }),
    }
}

fn session_candidates(directory: &Path) -> Result<Vec<PathBuf>> {
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let entries = fs::read_dir(directory)
        .with_context(|| format!("failed to list sessions: {}", directory.display()))?;
    Ok(entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "jsonl")
        })
        .collect())
}

fn scan_session_candidate(path: &Path, expected_cwd: &Path) -> Result<SessionCandidate> {
    let file = File::open(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    if reader
        .read_until(b'\n', &mut line)
        .with_context(|| format!("failed to read {}", path.display()))?
        == 0
    {
        bail!("session is empty: {}", path.display());
    }
    let header = trim_jsonl_line(&line);
    let header: SessionRecord = serde_json::from_slice(header)
        .with_context(|| format!("invalid session header: {}", path.display()))?;
    let SessionRecord::Session {
        version,
        id,
        cwd,
        created_at,
        ..
    } = header
    else {
        bail!("first session record is not a header: {}", path.display());
    };
    if version != SESSION_VERSION {
        bail!("unsupported session version {version}; expected {SESSION_VERSION}");
    }
    if cwd != expected_cwd {
        bail!(
            "session {id} belongs to {}, not {}",
            cwd.display(),
            expected_cwd.display()
        );
    }

    let mut has_messages = false;
    let mut line_number = 1usize;
    loop {
        line.clear();
        let read = reader
            .read_until(b'\n', &mut line)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        line_number = line_number.saturating_add(1);
        let terminated = line.ends_with(b"\n");
        let record = trim_jsonl_line(&line);
        if record.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let tag: SessionRecordTag = match serde_json::from_slice(record) {
            Ok(tag) => tag,
            Err(_) if !terminated => break,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("invalid session record at {}:{line_number}", path.display())
                });
            }
        };
        match tag.kind.as_str() {
            "message"
            | "run_started"
            | "assistant_completed"
            | "assistant_discarded"
            | "tool_completed" => {
                has_messages = true;
            }
            "generation_started" | "tool_started" | "run_finished" | "model_changed"
            | "reasoning_changed" | "compaction" => {}
            "session" if !terminated => break,
            "session" => bail!(
                "duplicate session header at {}:{line_number}",
                path.display()
            ),
            _ if !terminated => break,
            unknown => bail!(
                "unknown session record {unknown:?} at {}:{line_number}",
                path.display()
            ),
        }
    }

    Ok(SessionCandidate {
        path: path.to_path_buf(),
        id,
        created_at,
        has_messages,
    })
}

fn trim_jsonl_line(mut line: &[u8]) -> &[u8] {
    if line.ends_with(b"\n") {
        line = &line[..line.len() - 1];
    }
    if line.ends_with(b"\r") {
        line = &line[..line.len() - 1];
    }
    line
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn rollout_filename(created_at: u64, id: Uuid) -> Result<String> {
    let seconds = i64::try_from(created_at).context("session timestamp is out of range")?;
    let timestamp = DateTime::<Utc>::from_timestamp(seconds, 0)
        .ok_or_else(|| anyhow!("session timestamp is out of range: {created_at}"))?;
    Ok(format!(
        "rollout-{}-{id}.jsonl",
        timestamp.format("%Y-%m-%dT%H-%M-%S")
    ))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn persistent_session_is_created_only_after_the_first_user_message() {
        let base = tempdir().unwrap();
        let project = tempdir().unwrap();
        let cwd = project.path().canonicalize().unwrap();
        let directory = project_session_dir(base.path(), &cwd);
        let metadata = SessionMetadata {
            provider: "deepseek".to_string(),
            model: "test-model".to_string(),
            reasoning_effort: ReasoningEffort::Low,
        };

        let blank = Session::new_pending(
            cwd.clone(),
            metadata.clone(),
            Some(base.path().to_path_buf()),
        );
        assert!(blank.path().is_none());
        drop(blank);
        assert!(!directory.exists());

        let mut session = Session::new_pending(cwd, metadata, Some(base.path().to_path_buf()));
        session.set_model("xai", "grok-test").unwrap();
        session.set_reasoning_effort(ReasoningEffort::High).unwrap();
        assert!(session.path().is_none());
        assert!(!directory.exists());

        let id = session.id();
        session.begin_run(ChatMessage::user("hello")).unwrap();
        assert!(session.path().is_some_and(Path::is_file));
        drop(session);

        let resumed =
            Session::resume_in(base.path(), project.path(), Some(&id.to_string())).unwrap();
        assert_eq!(resumed.provider(), "xai");
        assert_eq!(resumed.model(), "grok-test");
        assert_eq!(resumed.reasoning_effort(), ReasoningEffort::High);
        assert_eq!(resumed.messages(), [ChatMessage::user("hello")]);
    }

    #[test]
    fn saves_full_tool_output_privately_and_deletes_it_with_the_session() {
        let base = tempdir().unwrap();
        let project = tempdir().unwrap();
        let metadata = SessionMetadata {
            provider: "deepseek".to_string(),
            model: "test-model".to_string(),
            reasoning_effort: ReasoningEffort::High,
        };
        let mut session = Session::create_in(base.path(), project.path(), metadata).unwrap();
        let id = session.id();
        let run_id = session.begin_run(ChatMessage::user("run a tool")).unwrap();
        let intent = session
            .start_tool(
                run_id,
                ToolCall {
                    id: "call_output".to_string(),
                    kind: "function".to_string(),
                    function: crate::protocol::FunctionCall {
                        name: "read_file".to_string(),
                        arguments: r#"{"path":"large.txt","offset":null,"limit":null}"#.to_string(),
                    },
                },
                ToolReplayPolicy::Safe,
            )
            .unwrap();
        let session_path = session.path().unwrap().to_path_buf();
        let output_path = session
            .save_tool_output(&intent, "complete output")
            .unwrap()
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(&output_path).unwrap(),
            "complete output"
        );
        #[cfg(unix)]
        assert_eq!(
            std::fs::metadata(&output_path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        session
            .save_tool_output(&intent, "replacement output")
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(&output_path).unwrap(),
            "replacement output"
        );
        drop(session);

        Session::delete_in(base.path(), project.path(), &id.to_string()).unwrap();
        assert!(!session_path.exists());
        assert!(!session_artifact_directory(&session_path).exists());
    }

    #[test]
    fn discarded_assistant_is_persisted_but_excluded_from_context() {
        let base = tempdir().unwrap();
        let project = tempdir().unwrap();
        let metadata = SessionMetadata {
            provider: "deepseek".to_string(),
            model: "test-model".to_string(),
            reasoning_effort: ReasoningEffort::High,
        };
        let mut session = Session::create_in(base.path(), project.path(), metadata).unwrap();
        let id = session.id();
        let run_id = session.begin_run(ChatMessage::user("continue")).unwrap();
        session.start_generation(run_id).unwrap();
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 10,
            total_tokens: 110,
            cached_prompt_tokens: Some(80),
        };
        session
            .discard_generation(
                run_id,
                ChatMessage::assistant(Some("partial".to_string()), None, Vec::new()),
                usage,
            )
            .unwrap();
        assert_eq!(session.messages().len(), 2);
        assert_eq!(session.context_messages(), [ChatMessage::user("continue")]);
        assert!(!session.active_run_has_final_response());
        assert_eq!(session.total_usage(), usage);
        drop(session);

        let resumed =
            Session::resume_in(base.path(), project.path(), Some(&id.to_string())).unwrap();
        assert_eq!(resumed.messages().len(), 2);
        assert_eq!(resumed.context_messages(), [ChatMessage::user("continue")]);
        assert!(!resumed.active_run_has_final_response());
        assert_eq!(resumed.total_usage(), usage);
    }

    #[test]
    fn persists_run_failure_details_without_affecting_resume() {
        let base = tempdir().unwrap();
        let project = tempdir().unwrap();
        let metadata = SessionMetadata {
            provider: "deepseek".to_string(),
            model: "test-model".to_string(),
            reasoning_effort: ReasoningEffort::High,
        };
        let mut session = Session::create_in(base.path(), project.path(), metadata).unwrap();
        let id = session.id();
        let run_id = session.begin_run(ChatMessage::user("写入文件")).unwrap();
        let path = session.path().unwrap().to_path_buf();
        session
            .finish_run_with_error(
                run_id,
                RunOutcome::Failed,
                Some("provider completed without a tool call".to_string()),
            )
            .unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"error\":\"provider completed without a tool call\""));
        drop(session);

        let resumed =
            Session::resume_in(base.path(), project.path(), Some(&id.to_string())).unwrap();
        assert_eq!(resumed.messages(), [ChatMessage::user("写入文件")]);
    }
}
