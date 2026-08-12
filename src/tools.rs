use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::RunningService;
use rmcp::transport::TokioChildProcess;
use rmcp::{RoleClient, ServiceExt};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{ChildStderr, Command};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::config::McpServerConfig;
use crate::event::AgentEvent;
use crate::protocol::{
    FileChangeKind, FileChangeLine, FileChangeLineKind, FileChangeSummary, FunctionDefinition,
    ToolCall, ToolDefinition,
};
use crate::sandbox::{PermissionProfile, shell_command};
use crate::session::ToolReplayPolicy;
use crate::web_access::WebAccess;

pub(crate) const MAX_TOOL_OUTPUT_CHARS: usize = 60_000;
const MAX_FILE_CHANGE_PREVIEW_LINES: usize = 5;
const MCP_CALL_TIMEOUT: Duration = Duration::from_mins(2);
const MAX_MCP_STDERR_BYTES: usize = 16_000;
const SHELL_EXIT_PIPE_IDLE_GRACE: Duration = Duration::from_millis(100);

pub struct ToolRegistry {
    root: PathBuf,
    permission_profile: PermissionProfile,
    definitions: Vec<ToolDefinition>,
    web_access: WebAccess,
    mcp_servers: Vec<McpService>,
    mcp_routes: BTreeMap<String, McpToolRoute>,
    mcp_startup_failures: Vec<McpStartupFailure>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ApprovalScope {
    ShellCommand(String),
    McpTool(String),
}

type McpService = RunningService<RoleClient, ()>;

#[derive(Debug, Clone)]
struct McpToolRoute {
    server_index: usize,
    tool_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpStartupFailure {
    pub server: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ToolExecution {
    pub output: String,
    pub is_error: bool,
    pub file_change: Option<FileChangeSummary>,
}

impl ToolExecution {
    fn success(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
            file_change: None,
        }
    }

    fn success_with_file_change(output: impl Into<String>, file_change: FileChangeSummary) -> Self {
        Self {
            output: output.into(),
            is_error: false,
            file_change: Some(file_change),
        }
    }

    fn error(error: impl std::fmt::Display) -> Self {
        Self {
            output: format!("tool error: {error}"),
            is_error: true,
            file_change: None,
        }
    }
}

impl ToolRegistry {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root
            .as_ref()
            .canonicalize()
            .with_context(|| format!("invalid tool root: {}", root.as_ref().display()))?;
        let web_access = WebAccess::new();
        let mut definitions = Self::builtin_definitions();
        definitions.extend(WebAccess::definitions());
        Ok(Self {
            root,
            permission_profile: PermissionProfile::default(),
            definitions,
            web_access,
            mcp_servers: Vec::new(),
            mcp_routes: BTreeMap::new(),
            mcp_startup_failures: Vec::new(),
        })
    }

    pub async fn with_mcp(root: impl AsRef<Path>, servers: &[McpServerConfig]) -> Result<Self> {
        let mut registry = Self::new(root)?;
        for server in servers {
            let result = async {
                let service = connect_mcp_server(server, &registry.root).await?;
                registry.register_mcp_server(&server.name, service).await
            }
            .await;
            if let Err(error) = result {
                registry.mcp_startup_failures.push(McpStartupFailure {
                    server: server.name.clone(),
                    message: format!("{error:#}"),
                });
            }
        }
        Ok(registry)
    }

    #[must_use]
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    #[must_use]
    pub const fn permission_profile(&self) -> PermissionProfile {
        self.permission_profile
    }

    pub const fn set_permission_profile(&mut self, profile: PermissionProfile) {
        self.permission_profile = profile;
    }

    #[must_use]
    pub fn mcp_server_count(&self) -> usize {
        self.mcp_servers.len()
    }

    #[must_use]
    pub fn mcp_tool_count(&self) -> usize {
        self.mcp_routes.len()
    }

    #[must_use]
    pub fn mcp_startup_failures(&self) -> &[McpStartupFailure] {
        &self.mcp_startup_failures
    }

    #[must_use]
    pub(crate) fn approval_scope(&self, call: &ToolCall) -> Option<ApprovalScope> {
        if call.function.name == "shell" {
            let command = parse_args::<ShellArgs>(&call.function.arguments).map_or_else(
                |_| call.function.arguments.trim().to_string(),
                |args| args.command.trim().to_string(),
            );
            return Some(ApprovalScope::ShellCommand(command));
        }
        self.mcp_routes
            .contains_key(&call.function.name)
            .then(|| ApprovalScope::McpTool(call.function.name.clone()))
    }

    #[must_use]
    pub fn replay_policy(&self, name: &str) -> ToolReplayPolicy {
        if name == "read_file" {
            ToolReplayPolicy::Safe
        } else {
            ToolReplayPolicy::Never
        }
    }

    fn builtin_definitions() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                kind: "function".to_string(),
                function: FunctionDefinition {
                    name: "read_file".to_string(),
                    description: "Read a UTF-8 text file inside the working directory with line numbers.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "offset": {"type": ["integer", "null"], "minimum": 1},
                            "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 2000}
                        },
                        "required": ["path", "offset", "limit"],
                        "additionalProperties": false
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: FunctionDefinition {
                    name: "write_file".to_string(),
                    description: "Create or overwrite a UTF-8 text file inside the working directory.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"],
                        "additionalProperties": false
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: FunctionDefinition {
                    name: "edit_file".to_string(),
                    description: "Replace an exact text fragment in a UTF-8 file. The match must be unique unless replace_all is true.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                            "replace_all": {"type": "boolean"}
                        },
                        "required": ["path", "old_text", "new_text", "replace_all"],
                        "additionalProperties": false
                    }),
                },
            },
            ToolDefinition {
                kind: "function".to_string(),
                function: FunctionDefinition {
                    name: "shell".to_string(),
                    description: "Run a shell command in the working directory and return its exit status, stdout, and stderr.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout_seconds": {
                                "type": ["integer", "null"],
                                "minimum": 1,
                                "maximum": 1800
                            }
                        },
                        "required": ["command", "timeout_seconds"],
                        "additionalProperties": false
                    }),
                },
            },
        ]
    }

    pub async fn execute(
        &self,
        call: &ToolCall,
        cancel: &CancellationToken,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolExecution {
        if call.function.name == "$web_search" {
            return ToolExecution::success(call.function.arguments.clone());
        }
        if is_web_access_tool(&call.function.name)
            && self
                .definitions
                .iter()
                .any(|definition| definition.function.name == call.function.name)
        {
            return match self
                .web_access
                .execute(&call.function.name, &call.function.arguments, cancel)
                .await
            {
                Ok(output) => ToolExecution::success(output),
                Err(error) => ToolExecution::error(error),
            };
        }
        if let Some(route) = self.mcp_routes.get(&call.function.name) {
            return self
                .execute_mcp(route, &call.function.arguments, cancel)
                .await;
        }
        let result = match call.function.name.as_str() {
            "read_file" => match parse_args(&call.function.arguments) {
                Ok(args) => self.read_file(args).await,
                Err(error) => Err(error),
            },
            "write_file" => match parse_args(&call.function.arguments) {
                Ok(args) => {
                    if !self.permission_profile.allows_file_writes() {
                        return ToolExecution::error(
                            "当前权限为只读；请通过 /permissions 更改权限",
                        );
                    }
                    return match self.write_file(args).await {
                        Ok((output, change)) => {
                            ToolExecution::success_with_file_change(output, change)
                        }
                        Err(error) => ToolExecution::error(error),
                    };
                }
                Err(error) => return ToolExecution::error(error),
            },
            "edit_file" => match parse_args(&call.function.arguments) {
                Ok(args) => {
                    if !self.permission_profile.allows_file_writes() {
                        return ToolExecution::error(
                            "当前权限为只读；请通过 /permissions 更改权限",
                        );
                    }
                    return match self.edit_file(args).await {
                        Ok((output, change)) => {
                            ToolExecution::success_with_file_change(output, change)
                        }
                        Err(error) => ToolExecution::error(error),
                    };
                }
                Err(error) => return ToolExecution::error(error),
            },
            "shell" => match parse_args(&call.function.arguments) {
                Ok(args) => self.shell(args, &call.id, cancel, events).await,
                Err(error) => Err(error),
            },
            unknown => Err(anyhow!("unknown tool: {unknown}")),
        };

        match result {
            Ok(output) => ToolExecution::success(output),
            Err(error) => ToolExecution::error(error),
        }
    }

    async fn register_mcp_server(&mut self, server_name: &str, service: McpService) -> Result<()> {
        const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(15);

        let tools = tokio::time::timeout(DISCOVERY_TIMEOUT, service.peer().list_all_tools())
            .await
            .with_context(|| {
                format!(
                    "MCP server {server_name:?} did not list tools within {} seconds",
                    DISCOVERY_TIMEOUT.as_secs()
                )
            })?
            .with_context(|| format!("failed to list tools from MCP server {server_name:?}"))?;
        let server_index = self.mcp_servers.len();
        let mut used_names = self
            .definitions
            .iter()
            .map(|definition| definition.function.name.clone())
            .collect::<BTreeSet<_>>();
        for tool in tools {
            let exposed_name = unique_mcp_tool_name(server_name, &tool.name, &used_names);
            used_names.insert(exposed_name.clone());
            let description = tool.description.as_deref().map_or_else(
                || format!("MCP tool {} from server {server_name}.", tool.name),
                ToString::to_string,
            );
            self.definitions.push(ToolDefinition {
                kind: "function".to_string(),
                function: FunctionDefinition {
                    name: exposed_name.clone(),
                    description,
                    parameters: serde_json::Value::Object((*tool.input_schema).clone()),
                },
            });
            self.mcp_routes.insert(
                exposed_name,
                McpToolRoute {
                    server_index,
                    tool_name: tool.name.into_owned(),
                },
            );
        }
        self.mcp_servers.push(service);
        Ok(())
    }

    async fn execute_mcp(
        &self,
        route: &McpToolRoute,
        arguments: &str,
        cancel: &CancellationToken,
    ) -> ToolExecution {
        let arguments = match serde_json::from_str::<serde_json::Value>(arguments) {
            Ok(serde_json::Value::Object(arguments)) => arguments,
            Ok(_) => return ToolExecution::error("MCP tool arguments must be a JSON object"),
            Err(error) => {
                return ToolExecution::error(format!("invalid MCP tool arguments: {error}"));
            }
        };
        let request = CallToolRequestParams::new(route.tool_name.clone()).with_arguments(arguments);
        let result = tokio::select! {
            () = cancel.cancelled() => return ToolExecution::error("MCP tool call cancelled"),
            result = tokio::time::timeout(
                MCP_CALL_TIMEOUT,
                self.mcp_servers[route.server_index].peer().call_tool(request),
            ) => result,
        };
        match result {
            Ok(Ok(result)) => mcp_result_to_execution(result),
            Ok(Err(error)) => ToolExecution::error(format!("MCP tool call failed: {error}")),
            Err(_) => ToolExecution::error(format!(
                "MCP tool call timed out after {} seconds",
                MCP_CALL_TIMEOUT.as_secs()
            )),
        }
    }

    async fn read_file(&self, args: ReadFileArgs) -> Result<String> {
        let path = self.resolve_path(&args.path)?;
        let text = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("failed to read {}", display_relative(&self.root, &path)))?;
        let offset = args.offset.unwrap_or(1);
        let limit = args.limit.unwrap_or(400).min(2000);
        let mut output = String::new();
        let lines = text.lines().skip(offset.saturating_sub(1)).take(limit);
        for (index, line) in lines.enumerate() {
            use std::fmt::Write as _;
            let _ = writeln!(output, "{:>6} | {line}", offset + index);
        }
        if output.is_empty() {
            output.push_str("<empty or offset beyond end of file>");
        }
        Ok(output)
    }

    async fn write_file(&self, args: WriteFileArgs) -> Result<(String, FileChangeSummary)> {
        let path = self.resolve_path(&args.path)?;
        let previous = match tokio::fs::read_to_string(&path).await {
            Ok(content) => Some(content),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to read {}", display_relative(&self.root, &path))
                });
            }
        };
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        atomic_write(&path, args.content.as_bytes())
            .await
            .with_context(|| format!("failed to write {}", display_relative(&self.root, &path)))?;
        let display_path = display_relative(&self.root, &path);
        let change = summarize_file_change(&display_path, previous.as_deref(), &args.content);
        Ok((
            format!("wrote {} bytes to {display_path}", args.content.len()),
            change,
        ))
    }

    async fn edit_file(&self, args: EditFileArgs) -> Result<(String, FileChangeSummary)> {
        if args.old_text.is_empty() {
            bail!("old_text cannot be empty");
        }
        let path = self.resolve_path(&args.path)?;
        let text = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("failed to read {}", display_relative(&self.root, &path)))?;
        let matches = text.matches(&args.old_text).count();
        if matches == 0 {
            bail!(
                "old_text was not found in {}",
                display_relative(&self.root, &path)
            );
        }
        if matches > 1 && !args.replace_all {
            bail!(
                "old_text matched {matches} locations in {}; provide a unique fragment or set replace_all",
                display_relative(&self.root, &path)
            );
        }
        let updated = if args.replace_all {
            text.replace(&args.old_text, &args.new_text)
        } else {
            text.replacen(&args.old_text, &args.new_text, 1)
        };
        atomic_write(&path, updated.as_bytes())
            .await
            .with_context(|| format!("failed to write {}", display_relative(&self.root, &path)))?;
        let replacement_count = if args.replace_all { matches } else { 1 };
        let display_path = display_relative(&self.root, &path);
        let change = summarize_file_change(&display_path, Some(&text), &updated);
        Ok((
            format!(
                "updated {replacement_count} replacement{} in {display_path}",
                if args.replace_all && matches != 1 {
                    "s"
                } else {
                    ""
                }
            ),
            change,
        ))
    }

    async fn shell(
        &self,
        args: ShellArgs,
        tool_id: &str,
        cancel: &CancellationToken,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<String> {
        if args.command.trim().is_empty() {
            bail!("command cannot be empty");
        }
        let timeout = Duration::from_secs(args.timeout_seconds.unwrap_or(120).clamp(1, 1800));
        let mut command = shell_command(&args.command, &self.root, self.permission_profile)?;
        let mut child = command.spawn().context("failed to start shell command")?;
        let stdout = child
            .stdout
            .take()
            .context("failed to capture shell stdout")?;
        let stderr = child
            .stderr
            .take()
            .context("failed to capture shell stderr")?;
        let (pipe_tx, mut pipe_rx) = mpsc::unbounded_channel();
        let stdout_task = tokio::spawn(read_shell_pipe(stdout, ShellPipe::Stdout, pipe_tx.clone()));
        let stderr_task = tokio::spawn(read_shell_pipe(stderr, ShellPipe::Stderr, pipe_tx.clone()));
        drop(pipe_tx);
        let mut wait_task = tokio::spawn(async move { child.wait().await });
        let mut wait_pending = true;
        let deadline = tokio::time::sleep(timeout);
        tokio::pin!(deadline);
        let exit_pipe_idle = tokio::time::sleep(SHELL_EXIT_PIPE_IDLE_GRACE);
        tokio::pin!(exit_pipe_idle);

        let mut status = None;
        let mut pipes_closed = false;
        let mut exit_pipe_idle_armed = false;
        let mut pipe_error = None;
        let mut stdout_bytes = Vec::new();
        let mut stderr_bytes = Vec::new();
        let mut stdout_decoder = StreamingUtf8::default();
        let mut stderr_decoder = StreamingUtf8::default();
        let mut stderr_started = false;

        loop {
            if status.is_some() && pipes_closed {
                break;
            }
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    if wait_pending {
                        wait_task.abort();
                        let _ = wait_task.await;
                    }
                    stdout_task.abort();
                    stderr_task.abort();
                    bail!("command cancelled");
                }
                () = &mut deadline => {
                    if wait_pending {
                        wait_task.abort();
                        let _ = wait_task.await;
                    }
                    stdout_task.abort();
                    stderr_task.abort();
                    bail!("command timed out after {} seconds", timeout.as_secs());
                }
                result = &mut wait_task, if wait_pending => {
                    wait_pending = false;
                    status = Some(
                        result
                            .context("shell wait task failed")?
                            .context("failed to wait for shell command")?,
                    );
                    exit_pipe_idle
                        .as_mut()
                        .reset(tokio::time::Instant::now() + SHELL_EXIT_PIPE_IDLE_GRACE);
                    exit_pipe_idle_armed = true;
                }
                next = pipe_rx.recv(), if !pipes_closed => {
                    match next {
                        Some(ShellPipeEvent::Chunk { pipe: ShellPipe::Stdout, bytes }) => {
                            stdout_bytes.extend_from_slice(&bytes);
                            emit_shell_delta(tool_id, &stdout_decoder.push(&bytes), events);
                        }
                        Some(ShellPipeEvent::Chunk { pipe: ShellPipe::Stderr, bytes }) => {
                            stderr_bytes.extend_from_slice(&bytes);
                            let text = stderr_decoder.push(&bytes);
                            if !text.is_empty() {
                                let prefix = if stderr_started { "" } else { "\nstderr:\n" };
                                stderr_started = true;
                                emit_shell_delta(tool_id, &format!("{prefix}{text}"), events);
                            }
                        }
                        Some(ShellPipeEvent::Error(error)) => pipe_error = Some(error),
                        None => pipes_closed = true,
                    }
                    if status.is_some() && !pipes_closed {
                        exit_pipe_idle
                            .as_mut()
                            .reset(tokio::time::Instant::now() + SHELL_EXIT_PIPE_IDLE_GRACE);
                        exit_pipe_idle_armed = true;
                    }
                }
                () = &mut exit_pipe_idle, if exit_pipe_idle_armed => break,
            }
        }

        emit_shell_delta(tool_id, &stdout_decoder.finish(), events);
        let stderr_tail = stderr_decoder.finish();
        if !stderr_tail.is_empty() {
            let prefix = if stderr_started { "" } else { "\nstderr:\n" };
            emit_shell_delta(tool_id, &format!("{prefix}{stderr_tail}"), events);
        }
        if pipes_closed {
            stdout_task.await.context("stdout reader task failed")?;
            stderr_task.await.context("stderr reader task failed")?;
        } else {
            stdout_task.abort();
            stderr_task.abort();
            let _ = stdout_task.await;
            let _ = stderr_task.await;
        }
        if let Some(error) = pipe_error {
            bail!("failed to read shell output: {error}");
        }

        let status = status.context("shell command ended without an exit status")?;
        let stdout = String::from_utf8_lossy(&stdout_bytes);
        let stderr = String::from_utf8_lossy(&stderr_bytes);
        Ok(format!(
            "exit code: {}\nstdout:\n{}\nstderr:\n{}",
            status.code().map_or_else(
                || "terminated by signal".to_string(),
                |code| code.to_string()
            ),
            if stdout.is_empty() {
                "<empty>"
            } else {
                &stdout
            },
            if stderr.is_empty() {
                "<empty>"
            } else {
                &stderr
            }
        ))
    }

    fn resolve_path(&self, raw: &str) -> Result<PathBuf> {
        if raw.trim().is_empty() {
            bail!("path cannot be empty");
        }
        let input = Path::new(raw);
        let candidate = if input.is_absolute() {
            normalize_path(input)?
        } else {
            normalize_path(&self.root.join(input))?
        };
        if !candidate.starts_with(&self.root) {
            bail!("path escapes the working directory: {raw}");
        }

        let mut ancestor = candidate.as_path();
        while !ancestor.exists() {
            ancestor = ancestor
                .parent()
                .ok_or_else(|| anyhow!("path has no existing ancestor: {raw}"))?;
        }
        let canonical_ancestor = ancestor
            .canonicalize()
            .with_context(|| format!("failed to resolve {}", ancestor.display()))?;
        if !canonical_ancestor.starts_with(&self.root) {
            bail!("path resolves outside the working directory: {raw}");
        }
        Ok(candidate)
    }
}

async fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("file path has no parent: {}", path.display()))?;
    let existing_permissions = tokio::fs::metadata(path)
        .await
        .ok()
        .map(|metadata| metadata.permissions());
    let temporary = parent.join(format!(".mcode-write-{}.tmp", uuid::Uuid::now_v7()));
    let result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .await
            .with_context(|| format!("failed to create temporary file in {}", parent.display()))?;
        file.write_all(contents)
            .await
            .with_context(|| format!("failed to write temporary file in {}", parent.display()))?;
        file.flush()
            .await
            .with_context(|| format!("failed to flush temporary file in {}", parent.display()))?;
        file.sync_all()
            .await
            .with_context(|| format!("failed to sync temporary file in {}", parent.display()))?;
        drop(file);
        if let Some(permissions) = existing_permissions {
            tokio::fs::set_permissions(&temporary, permissions)
                .await
                .with_context(|| {
                    format!("failed to preserve permissions for {}", path.display())
                })?;
        }
        tokio::fs::rename(&temporary, path)
            .await
            .with_context(|| format!("failed to replace {}", path.display()))?;
        Ok(())
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&temporary).await;
    }
    result
}

async fn connect_mcp_server(server: &McpServerConfig, root: &Path) -> Result<McpService> {
    const STARTUP_TIMEOUT: Duration = Duration::from_secs(15);

    let mut command = Command::new(&server.command);
    command
        .args(&server.args)
        .envs(&server.env)
        .env("AI_AGENT", "mcode")
        .current_dir(root);
    let (transport, stderr) = TokioChildProcess::builder(command)
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "failed to start MCP server {:?} with command {:?}",
                server.name, server.command
            )
        })?;
    let stderr_task = stderr.map(|stderr| tokio::spawn(capture_mcp_stderr(stderr)));
    let startup = tokio::time::timeout(STARTUP_TIMEOUT, ().serve(transport)).await;
    match startup {
        Ok(Ok(service)) => Ok(service),
        result => {
            let stderr = collect_mcp_stderr(stderr_task).await;
            let detail = match result {
                Err(_) => format!(
                    "MCP server {:?} did not initialize within {} seconds",
                    server.name,
                    STARTUP_TIMEOUT.as_secs()
                ),
                Ok(Err(error)) => {
                    format!("failed to initialize MCP server {:?}: {error}", server.name)
                }
                Ok(Ok(_)) => unreachable!("successful MCP startup returned above"),
            };
            if stderr.is_empty() {
                bail!(detail);
            }
            bail!("{detail}; stderr: {stderr}");
        }
    }
}

async fn capture_mcp_stderr(mut stderr: ChildStderr) -> String {
    let mut captured = Vec::new();
    let mut buffer = [0_u8; 2048];
    while let Ok(read) = stderr.read(&mut buffer).await {
        if read == 0 {
            break;
        }
        let remaining = MAX_MCP_STDERR_BYTES.saturating_sub(captured.len());
        captured.extend_from_slice(&buffer[..read.min(remaining)]);
    }
    let mut text = String::from_utf8_lossy(&captured).trim().to_string();
    if captured.len() == MAX_MCP_STDERR_BYTES {
        text.push_str(" ... <stderr truncated>");
    }
    text
}

async fn collect_mcp_stderr(task: Option<tokio::task::JoinHandle<String>>) -> String {
    let Some(task) = task else {
        return String::new();
    };
    tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .ok()
        .and_then(std::result::Result::ok)
        .unwrap_or_default()
}

fn unique_mcp_tool_name(server: &str, tool: &str, used: &BTreeSet<String>) -> String {
    let base = format!(
        "mcp__{}__{}",
        sanitize_tool_component(server),
        sanitize_tool_component(tool)
    );
    if base.len() <= 64 && !used.contains(&base) {
        return base;
    }

    for collision in 0_u32.. {
        let hash_input = format!("{server}\0{tool}\0{collision}");
        let suffix = short_hash(&hash_input);
        let prefix_len = 64 - suffix.len() - 1;
        let prefix = &base[..base.len().min(prefix_len)];
        let candidate = format!("{prefix}_{suffix}");
        if !used.contains(&candidate) {
            return candidate;
        }
    }
    unreachable!("the collision counter is unbounded")
}

fn is_web_access_tool(name: &str) -> bool {
    name == "fetch_content"
}

fn sanitize_tool_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "unnamed".to_string()
    } else {
        sanitized
    }
}

fn short_hash(value: &str) -> String {
    use std::fmt::Write as _;

    let digest = Sha256::digest(value.as_bytes());
    let mut hash = String::with_capacity(8);
    for byte in &digest[..4] {
        let _ = write!(hash, "{byte:02x}");
    }
    hash
}

fn mcp_result_to_execution(result: CallToolResult) -> ToolExecution {
    let is_error = result.is_error.unwrap_or(false);
    let mut parts = result
        .content
        .into_iter()
        .map(|content| {
            content.as_text().map_or_else(
                || {
                    serde_json::to_string(&content)
                        .unwrap_or_else(|error| format!("<invalid MCP content: {error}>"))
                },
                |text| text.text.clone(),
            )
        })
        .collect::<Vec<_>>();
    if let Some(structured) = result.structured_content {
        parts.push(
            serde_json::to_string_pretty(&structured)
                .unwrap_or_else(|error| format!("<invalid MCP structured content: {error}>")),
        );
    }
    if parts.is_empty() {
        parts.push("<empty MCP result>".to_string());
    }
    ToolExecution {
        output: parts.join("\n"),
        is_error,
        file_change: None,
    }
}

fn summarize_file_change(path: &str, old: Option<&str>, new: &str) -> FileChangeSummary {
    let difference = diffy::create_patch(old.unwrap_or_default(), new);
    let mut added_lines = 0usize;
    let mut removed_lines = 0usize;
    let mut preview = Vec::new();
    let mut preview_truncated = false;

    for (hunk_index, hunk) in difference.hunks().iter().enumerate() {
        if hunk_index > 0 {
            push_file_change_preview_line(
                &mut preview,
                &mut preview_truncated,
                FileChangeLine {
                    kind: FileChangeLineKind::Omitted,
                    line_number: 0,
                    content: String::new(),
                },
            );
        }
        let mut old_line = hunk.old_range().start();
        let mut new_line = hunk.new_range().start();
        for line in hunk.lines() {
            let (kind, line_number, content) = match line {
                diffy::Line::Context(content) => {
                    let line_number = new_line;
                    old_line = old_line.saturating_add(1);
                    new_line = new_line.saturating_add(1);
                    (FileChangeLineKind::Context, line_number, *content)
                }
                diffy::Line::Delete(content) => {
                    let line_number = old_line;
                    old_line = old_line.saturating_add(1);
                    removed_lines = removed_lines.saturating_add(1);
                    (FileChangeLineKind::Removed, line_number, *content)
                }
                diffy::Line::Insert(content) => {
                    let line_number = new_line;
                    new_line = new_line.saturating_add(1);
                    added_lines = added_lines.saturating_add(1);
                    (FileChangeLineKind::Added, line_number, *content)
                }
            };
            push_file_change_preview_line(
                &mut preview,
                &mut preview_truncated,
                FileChangeLine {
                    kind,
                    line_number,
                    content: content.trim_end_matches('\n').to_string(),
                },
            );
        }
    }

    FileChangeSummary {
        path: path.to_string(),
        kind: if old.is_some() {
            FileChangeKind::Updated
        } else {
            FileChangeKind::Added
        },
        added_lines,
        removed_lines,
        preview,
        preview_truncated,
    }
}

fn push_file_change_preview_line(
    preview: &mut Vec<FileChangeLine>,
    preview_truncated: &mut bool,
    line: FileChangeLine,
) {
    if preview.len() < MAX_FILE_CHANGE_PREVIEW_LINES {
        preview.push(line);
    } else {
        *preview_truncated = true;
    }
}

#[derive(Debug, Clone, Copy)]
enum ShellPipe {
    Stdout,
    Stderr,
}

enum ShellPipeEvent {
    Chunk { pipe: ShellPipe, bytes: Vec<u8> },
    Error(String),
}

async fn read_shell_pipe<R>(
    mut reader: R,
    pipe: ShellPipe,
    events: mpsc::UnboundedSender<ShellPipeEvent>,
) where
    R: AsyncRead + Unpin,
{
    let mut buffer = vec![0; 8 * 1024];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => {
                if events
                    .send(ShellPipeEvent::Chunk {
                        pipe,
                        bytes: buffer[..read].to_vec(),
                    })
                    .is_err()
                {
                    break;
                }
            }
            Err(error) => {
                let _ = events.send(ShellPipeEvent::Error(error.to_string()));
                break;
            }
        }
    }
}

#[derive(Default)]
struct StreamingUtf8 {
    pending: Vec<u8>,
}

impl StreamingUtf8 {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut output = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    output.push_str(text);
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    if valid > 0 {
                        output.push_str(
                            std::str::from_utf8(&self.pending[..valid])
                                .expect("UTF-8 validator reported a valid prefix"),
                        );
                        self.pending.drain(..valid);
                    }
                    let Some(invalid) = error.error_len() else {
                        break;
                    };
                    output.push('\u{fffd}');
                    self.pending.drain(..invalid);
                }
            }
        }
        output
    }

    fn finish(&mut self) -> String {
        let pending = std::mem::take(&mut self.pending);
        String::from_utf8_lossy(&pending).into_owned()
    }
}

fn emit_shell_delta(tool_id: &str, delta: &str, events: &mpsc::UnboundedSender<AgentEvent>) {
    if !delta.is_empty() {
        let _ = events.send(AgentEvent::ToolOutputDelta {
            id: tool_id.to_string(),
            delta: delta.to_string(),
        });
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileArgs {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EditFileArgs {
    path: String,
    old_text: String,
    new_text: String,
    #[serde(default)]
    replace_all: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellArgs {
    command: String,
    timeout_seconds: Option<u64>,
}

fn parse_args<T>(arguments: &str) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_str(arguments).context("invalid tool arguments")
}

fn normalize_path(path: &Path) -> Result<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    bail!("path traverses above filesystem root");
                }
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    Ok(normalized)
}

fn display_relative<'a>(root: &'a Path, path: &'a Path) -> std::borrow::Cow<'a, str> {
    path.strip_prefix(root).unwrap_or(path).to_string_lossy()
}

pub(crate) fn summarize_oversized_tool_output(output: &str, saved_path: Option<&Path>) -> String {
    let location = saved_path.map_or_else(
        || "full output was not saved because session persistence is disabled".to_string(),
        |path| format!("full output: {}", path.display()),
    );
    let marker = format!("... tool output truncated; {location} ...");
    let content_budget = MAX_TOOL_OUTPUT_CHARS
        .saturating_sub(marker.chars().count())
        .saturating_sub(2);
    let head_budget = content_budget / 2;
    let tail_budget = content_budget.saturating_sub(head_budget);
    let head: String = output.chars().take(head_budget).collect();
    let tail: String = output
        .chars()
        .rev()
        .take(tail_budget)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}\n{marker}\n{tail}")
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn shell_session_approval_is_bound_to_the_exact_command() {
        let project = tempdir().unwrap();
        let tools = ToolRegistry::new(project.path()).unwrap();
        let call = |command: &str, timeout_seconds: u64| ToolCall {
            id: "call_shell".to_string(),
            kind: "function".to_string(),
            function: crate::protocol::FunctionCall {
                name: "shell".to_string(),
                arguments: serde_json::json!({
                    "command": command,
                    "timeout_seconds": timeout_seconds
                })
                .to_string(),
            },
        };

        assert_eq!(
            tools.approval_scope(&call("cargo test", 30)),
            tools.approval_scope(&call(" cargo test ", 120))
        );
        assert_ne!(
            tools.approval_scope(&call("cargo test", 30)),
            tools.approval_scope(&call("cargo test --release", 30))
        );
    }

    #[test]
    fn summarized_tool_output_references_the_saved_full_output() {
        let output = "x".repeat(MAX_TOOL_OUTPUT_CHARS + 1);
        let path = Path::new("/tmp/tool-results/result.txt");

        let summary = summarize_oversized_tool_output(&output, Some(path));

        assert!(summary.contains("full output: /tmp/tool-results/result.txt"));
        assert!(summary.chars().count() <= MAX_TOOL_OUTPUT_CHARS);
    }

    #[tokio::test]
    async fn read_only_profile_rejects_builtin_file_writes() {
        let project = tempdir().unwrap();
        let mut tools = ToolRegistry::new(project.path()).unwrap();
        tools.set_permission_profile(PermissionProfile::ReadOnly);
        let call = ToolCall {
            id: "call_write".to_string(),
            kind: "function".to_string(),
            function: crate::protocol::FunctionCall {
                name: "write_file".to_string(),
                arguments: serde_json::json!({"path": "blocked.txt", "content": "no"}).to_string(),
            },
        };
        let (tx, _rx) = mpsc::unbounded_channel();

        let result = tools.execute(&call, &CancellationToken::new(), &tx).await;

        assert!(result.is_error);
        assert!(!project.path().join("blocked.txt").exists());
    }

    #[tokio::test]
    async fn shell_emits_incremental_output_events() {
        let project = tempdir().unwrap();
        let mut tools = ToolRegistry::new(project.path()).unwrap();
        tools.set_permission_profile(PermissionProfile::FullAccess);
        let call = ToolCall {
            id: "call_shell".to_string(),
            kind: "function".to_string(),
            function: crate::protocol::FunctionCall {
                name: "shell".to_string(),
                arguments: serde_json::json!({
                    "command": "printf alpha; printf beta; printf :$AI_AGENT"
                })
                .to_string(),
            },
        };
        let (tx, mut rx) = mpsc::unbounded_channel();

        let result = tools.execute(&call, &CancellationToken::new(), &tx).await;

        assert!(!result.is_error);
        assert!(result.output.contains("alphabeta:mcode"));
        let deltas = std::iter::from_fn(|| rx.try_recv().ok())
            .filter_map(|event| match event {
                AgentEvent::ToolOutputDelta { id, delta } if id == "call_shell" => Some(delta),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(deltas, "alphabeta:mcode");
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn shell_stops_waiting_for_quiet_inherited_pipes_after_exit() {
        let project = tempdir().unwrap();
        let mut tools = ToolRegistry::new(project.path()).unwrap();
        tools.set_permission_profile(PermissionProfile::FullAccess);
        let call = ToolCall {
            id: "call_shell".to_string(),
            kind: "function".to_string(),
            function: crate::protocol::FunctionCall {
                name: "shell".to_string(),
                arguments: serde_json::json!({
                    "command": "sh -c 'sleep 2 &' && printf done",
                    "timeout_seconds": 1
                })
                .to_string(),
            },
        };
        let (tx, _rx) = mpsc::unbounded_channel();

        let result = tools.execute(&call, &CancellationToken::new(), &tx).await;

        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("done"));
    }
}
