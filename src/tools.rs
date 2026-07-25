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
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::config::McpServerConfig;
use crate::protocol::{FunctionDefinition, ToolCall, ToolDefinition};

const MAX_TOOL_OUTPUT_CHARS: usize = 60_000;

pub struct ToolRegistry {
    root: PathBuf,
    definitions: Vec<ToolDefinition>,
    mcp_servers: Vec<McpService>,
    mcp_routes: BTreeMap<String, McpToolRoute>,
}

type McpService = RunningService<RoleClient, ()>;

#[derive(Debug, Clone)]
struct McpToolRoute {
    server_index: usize,
    tool_name: String,
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
        let root = root
            .as_ref()
            .canonicalize()
            .with_context(|| format!("invalid tool root: {}", root.as_ref().display()))?;
        Ok(Self {
            root,
            definitions: Self::builtin_definitions(),
            mcp_servers: Vec::new(),
            mcp_routes: BTreeMap::new(),
        })
    }

    pub async fn with_mcp(root: impl AsRef<Path>, servers: &[McpServerConfig]) -> Result<Self> {
        let mut registry = Self::new(root)?;
        for server in servers {
            let service = connect_mcp_server(server, &registry.root).await?;
            registry.register_mcp_server(&server.name, service).await?;
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
                            "offset": {"type": "integer", "minimum": 1},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 2000}
                        },
                        "required": ["path"],
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
                            "replace_all": {"type": "boolean", "default": false}
                        },
                        "required": ["path", "old_text", "new_text"],
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
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 1800,
                                "default": 120
                            }
                        },
                        "required": ["command"],
                        "additionalProperties": false
                    }),
                },
            },
        ]
    }

    pub async fn execute(&self, call: &ToolCall, cancel: &CancellationToken) -> ToolExecution {
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
            result = self.mcp_servers[route.server_index].peer().call_tool(request) => result,
        };
        match result {
            Ok(result) => mcp_result_to_execution(result),
            Err(error) => ToolExecution::error(format!("MCP tool call failed: {error}")),
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
        tokio::fs::write(&path, args.content.as_bytes())
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
        tokio::fs::write(&path, updated.as_bytes())
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

async fn connect_mcp_server(server: &McpServerConfig, root: &Path) -> Result<McpService> {
    const STARTUP_TIMEOUT: Duration = Duration::from_secs(15);

    let mut command = Command::new(&server.command);
    command
        .args(&server.args)
        .envs(&server.env)
        .current_dir(root);
    let (transport, _) = TokioChildProcess::builder(command)
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| {
            format!(
                "failed to start MCP server {:?} with command {:?}",
                server.name, server.command
            )
        })?;
    tokio::time::timeout(STARTUP_TIMEOUT, ().serve(transport))
        .await
        .with_context(|| {
            format!(
                "MCP server {:?} did not initialize within {} seconds",
                server.name,
                STARTUP_TIMEOUT.as_secs()
            )
        })?
        .with_context(|| format!("failed to initialize MCP server {:?}", server.name))
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

#[cfg(test)]
mod tests {
    use rmcp::model::{
        CallToolResult as McpCallToolResult, ContentBlock, ListToolsResult, PaginatedRequestParams,
        ServerCapabilities, ServerInfo, Tool,
    };
    use rmcp::service::RequestContext;
    use rmcp::{ErrorData, RoleServer, ServerHandler};
    use tempfile::tempdir;

    use super::*;
    use crate::protocol::FunctionCall;

    fn call(name: &str, arguments: &serde_json::Value) -> ToolCall {
        ToolCall {
            id: "call-1".to_string(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    #[derive(Clone)]
    struct FixtureMcpServer;

    impl ServerHandler for FixtureMcpServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> std::result::Result<ListToolsResult, ErrorData> {
            let schema = json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": false
            })
            .as_object()
            .expect("schema is an object")
            .clone();
            Ok(ListToolsResult::with_all_items(vec![Tool::new(
                "echo",
                "Echo a value from the fixture server.",
                schema,
            )]))
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> std::result::Result<McpCallToolResult, ErrorData> {
            let value = request
                .arguments
                .as_ref()
                .and_then(|arguments| arguments.get("value"))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| ErrorData::invalid_params("value is required", None))?;
            Ok(McpCallToolResult::success(vec![ContentBlock::text(
                format!("echo:{value}"),
            )]))
        }
    }

    #[tokio::test]
    async fn writes_reads_and_edits_files() {
        let temp = tempdir().unwrap();
        let registry = ToolRegistry::new(temp.path()).unwrap();
        let cancel = CancellationToken::new();

        let result = registry
            .execute(
                &call(
                    "write_file",
                    &json!({"path": "src/lib.rs", "content": "fn old() {}\n"}),
                ),
                &cancel,
            )
            .await;
        assert!(!result.is_error, "{}", result.output);

        let result = registry
            .execute(
                &call(
                    "edit_file",
                    &json!({
                        "path": "src/lib.rs",
                        "old_text": "old",
                        "new_text": "new"
                    }),
                ),
                &cancel,
            )
            .await;
        assert!(!result.is_error, "{}", result.output);

        let result = registry
            .execute(&call("read_file", &json!({"path": "src/lib.rs"})), &cancel)
            .await;
        assert!(!result.is_error, "{}", result.output);
        assert!(result.output.contains("fn new()"));
    }

    #[tokio::test]
    async fn rejects_paths_outside_root() {
        let temp = tempdir().unwrap();
        let registry = ToolRegistry::new(temp.path()).unwrap();
        let result = registry
            .execute(
                &call("read_file", &json!({"path": "../outside.txt"})),
                &CancellationToken::new(),
            )
            .await;
        assert!(result.is_error);
        assert!(result.output.contains("escapes the working directory"));
    }

    #[tokio::test]
    async fn discovers_and_executes_an_mcp_tool() {
        let (server_transport, client_transport) = tokio::io::duplex(4096);
        let server_task = tokio::spawn(async move {
            let server = FixtureMcpServer
                .serve(server_transport)
                .await
                .expect("fixture MCP server should start");
            let _ = server.waiting().await;
        });
        let client = ().serve(client_transport).await.expect("fixture MCP client should start");
        let temp = tempdir().unwrap();
        let mut registry = ToolRegistry::new(temp.path()).unwrap();
        registry
            .register_mcp_server("fixture.server", client)
            .await
            .unwrap();

        assert_eq!(registry.mcp_server_count(), 1);
        assert_eq!(registry.mcp_tool_count(), 1);
        assert!(
            registry
                .definitions()
                .iter()
                .any(|definition| definition.function.name == "mcp__fixture_server__echo")
        );
        let result = registry
            .execute(
                &call("mcp__fixture_server__echo", &json!({"value": "ok"})),
                &CancellationToken::new(),
            )
            .await;
        assert!(!result.is_error, "{}", result.output);
        assert_eq!(result.output, "echo:ok");

        drop(registry);
        server_task.abort();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn starts_and_calls_a_configured_stdio_mcp_server() {
        let temp = tempdir().unwrap();
        let script = temp.path().join("mcp-fixture.sh");
        std::fs::write(
            &script,
            r#"while IFS= read -r line; do
  case "$line" in
    *'"method":"initialize"'*)
      rest=${line#*\"id\":}; id=${rest%%,*}
      version_rest=${line#*\"protocolVersion\":\"}; version=${version_rest%%\"*}
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"%s","capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"1.0.0"}}}\n' "$id" "$version"
      ;;
    *'"method":"tools/list"'*)
      rest=${line#*\"id\":}; id=${rest%%,*}
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"stdio fixture","inputSchema":{"type":"object","properties":{},"additionalProperties":true}}]}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      rest=${line#*\"id\":}; id=${rest%%,*}
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"stdio-ok"}],"isError":false}}\n' "$id"
      ;;
  esac
done
"#,
        )
        .unwrap();
        let server = McpServerConfig {
            name: "stdio".to_string(),
            command: "/bin/sh".to_string(),
            args: vec![script.to_string_lossy().into_owned()],
            env: BTreeMap::new(),
        };
        let registry = ToolRegistry::with_mcp(temp.path(), &[server])
            .await
            .unwrap();

        let result = registry
            .execute(
                &call("mcp__stdio__echo", &json!({"value": "ok"})),
                &CancellationToken::new(),
            )
            .await;
        assert!(!result.is_error, "{}", result.output);
        assert_eq!(result.output, "stdio-ok");
    }
}
