use std::collections::{BTreeMap, BTreeSet};
use std::env;
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
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{ChildStderr, Command};
use tokio_util::sync::CancellationToken;

use crate::config::{ApiProtocol, McpServerConfig, WebSearchMode, WebSearchSettings};
use crate::protocol::{FunctionDefinition, ToolCall, ToolDefinition};
use crate::session::ToolReplayPolicy;
use crate::web_access::WebAccess;

const MAX_TOOL_OUTPUT_CHARS: usize = 60_000;
const MCP_CALL_TIMEOUT: Duration = Duration::from_mins(2);
const MAX_MCP_STDERR_BYTES: usize = 16_000;

pub struct ToolRegistry {
    root: PathBuf,
    definitions: Vec<ToolDefinition>,
    api: ApiProtocol,
    web_access: WebAccess,
    mcp_servers: Vec<McpService>,
    mcp_routes: BTreeMap<String, McpToolRoute>,
    mcp_startup_failures: Vec<McpStartupFailure>,
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
}

impl ToolExecution {
    fn success(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
        }
    }

    fn error(error: impl std::fmt::Display) -> Self {
        Self {
            output: format!("tool error: {error}"),
            is_error: true,
        }
    }
}

impl ToolRegistry {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        Self::with_web_access(
            root,
            WebSearchSettings::default(),
            ApiProtocol::ChatCompletions,
        )
    }

    fn with_web_access(
        root: impl AsRef<Path>,
        web_search: WebSearchSettings,
        api: ApiProtocol,
    ) -> Result<Self> {
        let root = root
            .as_ref()
            .canonicalize()
            .with_context(|| format!("invalid tool root: {}", root.as_ref().display()))?;
        let web_access = WebAccess::new(web_search)?;
        let mut definitions = Self::builtin_definitions();
        definitions.extend(web_access.definitions(api));
        Ok(Self {
            root,
            definitions,
            api,
            web_access,
            mcp_servers: Vec::new(),
            mcp_routes: BTreeMap::new(),
            mcp_startup_failures: Vec::new(),
        })
    }

    pub async fn with_mcp(
        root: impl AsRef<Path>,
        servers: &[McpServerConfig],
        web_search: WebSearchSettings,
        api: ApiProtocol,
    ) -> Result<Self> {
        let mut registry = Self::with_web_access(root, web_search, api)?;
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

    pub fn set_web_search_mode(&mut self, mode: WebSearchMode) {
        self.web_access.set_mode(mode);
        self.refresh_web_access_definitions();
    }

    pub fn set_api(&mut self, api: ApiProtocol) {
        if self.api == api {
            return;
        }
        self.api = api;
        self.refresh_web_access_definitions();
    }

    #[must_use]
    pub fn requires_approval(&self, name: &str) -> bool {
        name == "shell" || self.mcp_routes.contains_key(name)
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
                    strict: Some(true),
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
                    strict: Some(true),
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
                    strict: Some(true),
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
                    strict: Some(true),
                },
            },
        ]
    }

    pub async fn execute(&self, call: &ToolCall, cancel: &CancellationToken) -> ToolExecution {
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
                Ok(output) => ToolExecution::success(truncate_output(&output)),
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
                Ok(args) => self.write_file(args).await,
                Err(error) => Err(error),
            },
            "edit_file" => match parse_args(&call.function.arguments) {
                Ok(args) => self.edit_file(args).await,
                Err(error) => Err(error),
            },
            "shell" => match parse_args(&call.function.arguments) {
                Ok(args) => self.shell(args, cancel).await,
                Err(error) => Err(error),
            },
            unknown => Err(anyhow!("unknown tool: {unknown}")),
        };

        match result {
            Ok(output) => ToolExecution::success(truncate_output(&output)),
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
                    strict: None,
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

    fn refresh_web_access_definitions(&mut self) {
        self.definitions
            .retain(|definition| !is_web_access_tool(&definition.function.name));
        let insert_at = self
            .definitions
            .iter()
            .position(|definition| self.mcp_routes.contains_key(&definition.function.name))
            .unwrap_or(self.definitions.len());
        self.definitions
            .splice(insert_at..insert_at, self.web_access.definitions(self.api));
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

    async fn write_file(&self, args: WriteFileArgs) -> Result<String> {
        let path = self.resolve_path(&args.path)?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        atomic_write(&path, args.content.as_bytes())
            .await
            .with_context(|| format!("failed to write {}", display_relative(&self.root, &path)))?;
        Ok(format!(
            "wrote {} bytes to {}",
            args.content.len(),
            display_relative(&self.root, &path)
        ))
    }

    async fn edit_file(&self, args: EditFileArgs) -> Result<String> {
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
        Ok(format!(
            "updated {} replacement{} in {}",
            if args.replace_all { matches } else { 1 },
            if args.replace_all && matches != 1 {
                "s"
            } else {
                ""
            },
            display_relative(&self.root, &path)
        ))
    }

    async fn shell(&self, args: ShellArgs, cancel: &CancellationToken) -> Result<String> {
        if args.command.trim().is_empty() {
            bail!("command cannot be empty");
        }
        let timeout = Duration::from_secs(args.timeout_seconds.unwrap_or(120).clamp(1, 1800));
        let mut command = shell_command(&args.command);
        command
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let child = command.spawn().context("failed to start shell command")?;
        let output = tokio::select! {
            () = cancel.cancelled() => bail!("command cancelled"),
            result = tokio::time::timeout(timeout, child.wait_with_output()) => {
                match result {
                    Ok(output) => output.context("failed to wait for shell command")?,
                    Err(_) => bail!("command timed out after {} seconds", timeout.as_secs()),
                }
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Ok(format!(
            "exit code: {}\nstdout:\n{}\nstderr:\n{}",
            output.status.code().map_or_else(
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
    matches!(name, "web_search" | "fetch_content")
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
        output: truncate_output(&parts.join("\n")),
        is_error,
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

fn truncate_output(output: &str) -> String {
    const HALF: usize = MAX_TOOL_OUTPUT_CHARS / 2;

    if output.chars().count() <= MAX_TOOL_OUTPUT_CHARS {
        return output.to_string();
    }
    let head: String = output.chars().take(HALF).collect();
    let tail: String = output
        .chars()
        .rev()
        .take(HALF)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}\n... tool output truncated ...\n{tail}")
}

#[cfg(windows)]
fn shell_command(command: &str) -> Command {
    let mut process = Command::new("cmd");
    process.arg("/C").arg(command);
    process
}

#[cfg(not(windows))]
fn shell_command(command: &str) -> Command {
    let shell = env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let mut process = Command::new(shell);
    process.arg("-lc").arg(command);
    process
}
