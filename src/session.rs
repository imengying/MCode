use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::config::ReasoningEffort;
use crate::protocol::ChatMessage;

const SESSION_VERSION: u32 = 1;

#[derive(Debug)]
pub struct Session {
    id: Uuid,
    cwd: PathBuf,
    model: String,
    reasoning_effort: ReasoningEffort,
    created_at: u64,
    path: Option<PathBuf>,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionRecord {
    Session {
        version: u32,
        id: Uuid,
        cwd: PathBuf,
        model: String,
        #[serde(default)]
        reasoning_effort: ReasoningEffort,
        created_at: u64,
    },
    Message {
        message: ChatMessage,
    },
    ModelChanged {
        model: String,
    },
    ReasoningChanged {
        reasoning_effort: ReasoningEffort,
    },
}

impl Session {
    pub fn create(
        cwd: &Path,
        model: &str,
        reasoning_effort: ReasoningEffort,
        persist: bool,
    ) -> Result<Self> {
        let cwd = cwd
            .canonicalize()
            .with_context(|| format!("invalid session directory: {}", cwd.display()))?;
        if !persist {
            return Ok(Self {
                id: Uuid::new_v4(),
                cwd,
                model: model.to_string(),
                reasoning_effort,
                created_at: unix_timestamp(),
                path: None,
                messages: Vec::new(),
            });
        }
        let base = default_session_base()?;
        Self::create_in(&base, &cwd, model, reasoning_effort)
    }

    pub fn create_in(
        base: &Path,
        cwd: &Path,
        model: &str,
        reasoning_effort: ReasoningEffort,
    ) -> Result<Self> {
        let cwd = cwd
            .canonicalize()
            .with_context(|| format!("invalid session directory: {}", cwd.display()))?;
        let id = Uuid::new_v4();
        let created_at = unix_timestamp();
        let directory = project_session_dir(base, &cwd);
        fs::create_dir_all(&directory)
            .with_context(|| format!("failed to create {}", directory.display()))?;
        let path = directory.join(format!("{created_at}-{id}.jsonl"));
        let session = Self {
            id,
            cwd,
            model: model.to_string(),
            reasoning_effort,
            created_at,
            path: Some(path),
            messages: Vec::new(),
        };
        session.write_header()?;
        Ok(session)
    }

    pub fn resume(cwd: &Path, selector: Option<&str>) -> Result<Self> {
        let base = default_session_base()?;
        Self::resume_in(&base, cwd, selector)
    }

    pub fn resume_in(base: &Path, cwd: &Path, selector: Option<&str>) -> Result<Self> {
        let cwd = cwd
            .canonicalize()
            .with_context(|| format!("invalid session directory: {}", cwd.display()))?;
        if let Some(selector) = selector {
            let direct = Path::new(selector);
            if direct.is_file() {
                return Self::load(direct);
            }
        }

        let directory = project_session_dir(base, &cwd);
        let mut candidates = session_candidates(&directory)?;
        if let Some(selector) = selector.filter(|value| !value.eq_ignore_ascii_case("last")) {
            candidates.retain(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains(selector))
            });
        }
        let path = candidates
            .into_iter()
            .max_by_key(|path| {
                path.metadata()
                    .and_then(|metadata| metadata.modified())
                    .unwrap_or(UNIX_EPOCH)
            })
            .ok_or_else(|| {
                if let Some(selector) = selector {
                    anyhow!("no session matching {selector:?} for {}", cwd.display())
                } else {
                    anyhow!("no previous session for {}", cwd.display())
                }
            })?;
        Self::load(&path)
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
        let session = Self::load(&path)?;
        if session.cwd != cwd {
            bail!(
                "session {} belongs to a different working directory",
                session.id
            );
        }
        fs::remove_file(&path)
            .with_context(|| format!("failed to delete session: {}", path.display()))?;
        Ok(session.id)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut lines = BufReader::new(file).lines();
        let header = lines
            .next()
            .transpose()
            .with_context(|| format!("failed to read {}", path.display()))?
            .ok_or_else(|| anyhow!("session is empty: {}", path.display()))?;
        let header: SessionRecord = serde_json::from_str(&header)
            .with_context(|| format!("invalid session header: {}", path.display()))?;
        let SessionRecord::Session {
            version,
            id,
            cwd,
            mut model,
            mut reasoning_effort,
            created_at,
        } = header
        else {
            bail!("first session record is not a header: {}", path.display());
        };
        if version != SESSION_VERSION {
            bail!("unsupported session version {version}; expected {SESSION_VERSION}");
        }

        let mut messages = Vec::new();
        for (index, line) in lines.enumerate() {
            let line = line.with_context(|| {
                format!("failed to read {} at line {}", path.display(), index + 2)
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let record: SessionRecord = serde_json::from_str(&line).with_context(|| {
                format!("invalid session record at {}:{}", path.display(), index + 2)
            })?;
            match record {
                SessionRecord::Message { message } => messages.push(message),
                SessionRecord::ModelChanged { model: next_model } => model = next_model,
                SessionRecord::ReasoningChanged {
                    reasoning_effort: next_effort,
                } => reasoning_effort = next_effort,
                SessionRecord::Session { .. } => {
                    bail!(
                        "unexpected session header at {}:{}",
                        path.display(),
                        index + 2
                    );
                }
            }
        }

        Ok(Self {
            id,
            cwd,
            model,
            reasoning_effort,
            created_at,
            path: Some(path.to_path_buf()),
            messages,
        })
    }

    pub fn append(&mut self, message: ChatMessage) -> Result<()> {
        if let Some(path) = &self.path {
            append_record(
                path,
                &SessionRecord::Message {
                    message: message.clone(),
                },
            )?;
        }
        self.messages.push(message);
        Ok(())
    }

    pub fn set_model(&mut self, model: &str) -> Result<()> {
        if self.model == model {
            return Ok(());
        }
        if let Some(path) = &self.path {
            append_record(
                path,
                &SessionRecord::ModelChanged {
                    model: model.to_string(),
                },
            )?;
        }
        self.model = model.to_string();
        Ok(())
    }

    pub fn set_reasoning_effort(&mut self, effort: ReasoningEffort) -> Result<()> {
        if self.reasoning_effort == effort {
            return Ok(());
        }
        if let Some(path) = &self.path {
            append_record(
                path,
                &SessionRecord::ReasoningChanged {
                    reasoning_effort: effort,
                },
            )?;
        }
        self.reasoning_effort = effort;
        Ok(())
    }

    pub fn fresh(&self, model: &str, reasoning_effort: ReasoningEffort) -> Result<Self> {
        Self::create(&self.cwd, model, reasoning_effort, self.path.is_some())
    }

    pub fn delete_current(&mut self) -> Result<Uuid> {
        let base = default_session_base()?;
        self.delete_current_in(&base)
    }

    fn delete_current_in(&mut self, base: &Path) -> Result<Uuid> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| anyhow!("this session is not persisted"))?;
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
        fs::remove_file(path)
            .with_context(|| format!("failed to delete session: {}", path.display()))?;
        self.path = None;
        Ok(self.id)
    }

    #[must_use]
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
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
        &self.model
    }

    #[must_use]
    pub const fn reasoning_effort(&self) -> ReasoningEffort {
        self.reasoning_effort
    }

    #[must_use]
    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    fn write_header(&self) -> Result<()> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| anyhow!("cannot persist an in-memory session"))?;
        let record = SessionRecord::Session {
            version: SESSION_VERSION,
            id: self.id,
            cwd: self.cwd.clone(),
            model: self.model.clone(),
            reasoning_effort: self.reasoning_effort,
            created_at: self.created_at,
        };
        let mut file = File::create(path)
            .with_context(|| format!("failed to create session: {}", path.display()))?;
        serde_json::to_writer(&mut file, &record)
            .with_context(|| format!("failed to serialize session: {}", path.display()))?;
        file.write_all(b"\n")
            .with_context(|| format!("failed to write session: {}", path.display()))?;
        file.sync_data()
            .with_context(|| format!("failed to flush session: {}", path.display()))
    }
}

fn append_record(path: &Path, record: &SessionRecord) -> Result<()> {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open session: {}", path.display()))?;
    serde_json::to_writer(&mut file, record)
        .with_context(|| format!("failed to serialize session: {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("failed to write session: {}", path.display()))?;
    file.sync_data()
        .with_context(|| format!("failed to flush session: {}", path.display()))
}

fn default_session_base() -> Result<PathBuf> {
    dirs::home_dir()
        .map(|home| home.join(".mcode/sessions"))
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

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn persists_and_resumes_latest_session() {
        let base = tempdir().unwrap();
        let project = tempdir().unwrap();
        let mut session = Session::create_in(
            base.path(),
            project.path(),
            "test-model",
            ReasoningEffort::Low,
        )
        .unwrap();
        let id = session.id();
        session.append(ChatMessage::user("hello")).unwrap();
        session
            .append(ChatMessage::assistant(
                Some("world".into()),
                None,
                Vec::new(),
            ))
            .unwrap();
        session.set_model("next-model").unwrap();
        session.set_reasoning_effort(ReasoningEffort::High).unwrap();

        let loaded = Session::resume_in(base.path(), project.path(), None).unwrap();
        assert_eq!(loaded.id(), id);
        assert_eq!(loaded.model(), "next-model");
        assert_eq!(loaded.reasoning_effort(), ReasoningEffort::High);
        assert_eq!(loaded.messages().len(), 2);
        assert_eq!(loaded.messages()[0], ChatMessage::user("hello"));
    }

    #[test]
    fn deletes_only_the_exact_project_session() {
        let base = tempdir().unwrap();
        let project = tempdir().unwrap();
        let first = Session::create_in(
            base.path(),
            project.path(),
            "test-model",
            ReasoningEffort::Low,
        )
        .unwrap();
        let second = Session::create_in(
            base.path(),
            project.path(),
            "test-model",
            ReasoningEffort::Low,
        )
        .unwrap();
        let first_path = first.path().unwrap().to_path_buf();
        let second_path = second.path().unwrap().to_path_buf();

        let deleted =
            Session::delete_in(base.path(), project.path(), &first.id().to_string()).unwrap();
        assert_eq!(deleted, first.id());
        assert!(!first_path.exists());
        assert!(second_path.exists());
    }

    #[test]
    fn deleting_current_session_rejects_another_storage_root() {
        let base = tempdir().unwrap();
        let other_base = tempdir().unwrap();
        let project = tempdir().unwrap();
        let mut session = Session::create_in(
            base.path(),
            project.path(),
            "test-model",
            ReasoningEffort::Low,
        )
        .unwrap();

        assert!(session.delete_current_in(other_base.path()).is_err());
        assert!(session.path().unwrap().exists());
    }
}
