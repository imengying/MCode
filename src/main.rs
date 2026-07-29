use std::env;
use std::io::{self, IsTerminal, Read, Write};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use mcode::agent::{Agent, RunStatus};
use mcode::approval::{ApprovalDecision, ApprovalGate, ApprovalRequest, format_tool_arguments};
use mcode::cli::{Cli, Command, join_prompt};
use mcode::config::{AppConfig, ConfigOverrides, McpServerConfig, WebSearchMode};
use mcode::event::{AgentEvent, CompactionReason};
use mcode::protocol::{ImageAttachment, sanitize_terminal_text};
use mcode::session::{Session, SessionMetadata};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("错误：{error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse_localized();
    if matches!(&cli.command, Some(Command::Update)) {
        return mcode::update::run().await;
    }
    let root_prompt = cli.root_prompt();
    let model_was_overridden =
        cli.model.is_some() || env::var("OPENAI_MODEL").is_ok_and(|value| !value.trim().is_empty());
    let reasoning_was_overridden = cli.reasoning_effort.is_some()
        || env::var("OPENAI_REASONING_EFFORT").is_ok_and(|value| !value.trim().is_empty());
    let search_was_overridden = cli.search;
    let overrides = ConfigOverrides {
        model: cli.model.clone(),
        reasoning_effort: cli.reasoning_effort,
        base_url: cli.base_url.clone(),
        api_key_env: cli.api_key_env.clone(),
        context_window: cli.context_window,
        max_input_tokens: cli.max_input_tokens,
        cwd: cli.cwd.clone(),
        request_timeout_secs: cli.request_timeout,
        web_search: cli.search.then_some(WebSearchMode::Live),
    };
    let mut config = AppConfig::load(&overrides)?;
    let bypass_approvals = cli.dangerously_bypass_approvals;

    match cli.command {
        Some(Command::Exec(args)) => {
            let prompt = resolve_exec_prompt(&args.prompt)?;
            let images = load_images(&cli.images, &config.cwd)?;
            prepare_mcp_servers(&mut config, bypass_approvals, !args.json);
            let session = Session::create(&config.cwd, SessionMetadata::from(&config), false)?;
            let agent = Agent::new(&config, session).await?;
            run_exec(agent, prompt, images, args.json, bypass_approvals).await
        }
        Some(Command::Resume(args)) => {
            let session = Session::resume(&config.cwd, args.session.as_deref())?;
            if !search_was_overridden {
                config.web_search.mode = session.web_search_mode();
            }
            let saved_model = session.model_selector();
            match (model_was_overridden, reasoning_was_overridden) {
                (false, false) => {
                    config.select_model_and_reasoning(&saved_model, session.reasoning_effort())?;
                }
                (false, true) => config.select_model(&saved_model)?,
                (true, false) => {
                    config.select_reasoning_effort(session.reasoning_effort())?;
                }
                (true, true) => {}
            }
            if !model_was_overridden {
                config.api = session.api();
            }
            let prompt = join_prompt(&args.prompt);
            let images = load_images(&cli.images, &config.cwd)?;
            let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
            prepare_mcp_servers(&mut config, bypass_approvals, interactive);
            let agent = Agent::new(&config, session).await?;
            if interactive {
                mcode::ui::run_interactive(agent, prompt, images, bypass_approvals)
            } else if agent.has_pending_run() {
                if prompt.is_some() {
                    bail!("此会话存在中断的任务；请先在不提供新提示词的情况下恢复它");
                }
                run_exec(agent, String::new(), Vec::new(), false, bypass_approvals).await
            } else {
                let prompt = prompt.ok_or_else(|| {
                    anyhow::anyhow!("标准输入不是交互终端时，resume 需要提供提示词")
                })?;
                run_exec(agent, prompt, images, false, bypass_approvals).await
            }
        }
        Some(Command::Delete(args)) => {
            if args.force && Uuid::parse_str(&args.session).is_err() {
                bail!("--force 需要完整的会话 UUID");
            }
            if !args.force && !confirm_session_delete(&args.session)? {
                println!("已取消删除。");
                return Ok(());
            }
            let id = Session::delete(&config.cwd, &args.session)?;
            println!("已删除会话 {id}。");
            Ok(())
        }
        Some(Command::Sessions(args)) => list_sessions(&config, args.json),
        Some(Command::Doctor(args)) => run_doctor(&config, args.json),
        Some(Command::Update) => unreachable!("update 已在加载配置前处理"),
        None if io::stdin().is_terminal() && io::stdout().is_terminal() => {
            prepare_mcp_servers(&mut config, bypass_approvals, true);
            let session =
                Session::create(&config.cwd, SessionMetadata::from(&config), !cli.no_session)?;
            let agent = Agent::new(&config, session).await?;
            let images = load_images(&cli.images, &config.cwd)?;
            mcode::ui::run_interactive(agent, root_prompt, images, bypass_approvals)
        }
        None => {
            let prompt = match root_prompt {
                Some(prompt) => prompt,
                None => read_stdin_prompt()?,
            };
            let session = Session::create(&config.cwd, SessionMetadata::from(&config), false)?;
            prepare_mcp_servers(&mut config, bypass_approvals, false);
            let agent = Agent::new(&config, session).await?;
            let images = load_images(&cli.images, &config.cwd)?;
            run_exec(agent, prompt, images, false, bypass_approvals).await
        }
    }
}

fn list_sessions(config: &AppConfig, json: bool) -> Result<()> {
    let sessions = Session::list(&config.cwd)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&sessions).context("将会话编码为 JSON 失败")?
        );
        return Ok(());
    }
    if sessions.is_empty() {
        println!("{} 没有已保存的会话。", config.cwd.display());
        return Ok(());
    }
    for session in sessions {
        let timestamp = i64::try_from(session.created_at)
            .ok()
            .and_then(|seconds| chrono::DateTime::<chrono::Utc>::from_timestamp(seconds, 0))
            .map_or_else(
                || session.created_at.to_string(),
                |timestamp| timestamp.format("%Y-%m-%d %H:%M:%SZ").to_string(),
            );
        let model = session.provider.as_deref().map_or_else(
            || session.model.clone(),
            |provider| format!("{provider}/{}", session.model),
        );
        println!(
            "{}  {}  {}  {}  {} 条消息，{} 个 token{}\n    {}",
            session.id,
            timestamp,
            sanitize_terminal_text(&model),
            session.api,
            session.message_count,
            session.total_usage.total_tokens,
            if session.has_pending_run {
                " [已中断]"
            } else {
                ""
            },
            sanitize_terminal_text(&session.path.to_string_lossy())
        );
    }
    Ok(())
}

fn run_doctor(config: &AppConfig, json: bool) -> Result<()> {
    let mut checks = Vec::new();
    checks.push(serde_json::json!({
        "status": "ok",
        "name": "version",
        "detail": format!("mcode {}", mcode::VERSION),
    }));
    checks.push(serde_json::json!({
        "status": "ok",
        "name": "working_directory",
        "detail": config.cwd,
    }));
    checks.push(serde_json::json!({
        "status": "ok",
        "name": "model",
        "detail": {
            "provider": config.provider,
            "id": config.model,
            "api": config.api.to_string(),
            "contextWindow": config.context_window,
            "maxInputTokens": config.max_input_tokens,
            "webSearch": config.web_search.mode.to_string(),
        },
    }));
    let endpoint_status = url::Url::parse(&config.base_url).map_or_else(
        |error| ("error", format!("{}: {error}", config.base_url)),
        |url| {
            if matches!(url.scheme(), "http" | "https") {
                ("ok", url.to_string())
            } else {
                ("error", format!("unsupported URL scheme: {}", url.scheme()))
            }
        },
    );
    checks.push(serde_json::json!({
        "status": endpoint_status.0,
        "name": "endpoint",
        "detail": endpoint_status.1,
    }));
    checks.push(serde_json::json!({
        "status": if config.api_key.is_some() { "ok" } else { "warning" },
        "name": "api_key",
        "detail": if config.api_key.is_some() {
            "configured (value hidden)"
        } else {
            "not configured; this is valid only for endpoints that do not require authentication"
        },
    }));
    let session_directory = Session::storage_directory(&config.cwd)?;
    let storage_ancestor = existing_ancestor(&session_directory);
    let storage_status = storage_ancestor.as_ref().map_or("error", |path| {
        path.metadata().map_or("error", |metadata| {
            if metadata.permissions().readonly() {
                "warning"
            } else {
                "ok"
            }
        })
    });
    checks.push(serde_json::json!({
        "status": storage_status,
        "name": "session_storage",
        "detail": session_directory,
    }));
    match Session::list(&config.cwd) {
        Ok(sessions) => checks.push(serde_json::json!({
            "status": "ok",
            "name": "sessions",
            "detail": format!("{} readable session(s)", sessions.len()),
        })),
        Err(error) => checks.push(serde_json::json!({
            "status": "error",
            "name": "sessions",
            "detail": format!("{error:#}"),
        })),
    }
    for server in &config.mcp_servers {
        let executable = resolve_executable(&server.command, &config.cwd);
        checks.push(serde_json::json!({
            "status": if executable.is_some() { "ok" } else { "warning" },
            "name": format!("mcp:{}", server.name),
            "detail": executable.map_or_else(
                || format!("command not found: {}", server.command),
                |path| path.to_string_lossy().into_owned(),
            ),
        }));
    }

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&checks).context("将诊断报告编码为 JSON 失败")?
        );
    } else {
        println!("MCode 诊断");
        for check in &checks {
            let status = check["status"].as_str().unwrap_or("error");
            let name = check["name"].as_str().unwrap_or("unknown");
            let detail = localized_doctor_detail(status, name, &check["detail"]);
            println!(
                "[{}] {}: {}",
                localized_doctor_status(status),
                localized_doctor_name(name),
                sanitize_terminal_text(&detail)
            );
        }
    }
    Ok(())
}

fn localized_doctor_status(status: &str) -> &'static str {
    match status {
        "ok" => "正常",
        "warning" => "警告",
        _ => "错误",
    }
}

fn localized_doctor_name(name: &str) -> &str {
    match name {
        "version" => "版本",
        "working_directory" => "工作目录",
        "model" => "模型",
        "endpoint" => "端点",
        "api_key" => "API 密钥",
        "session_storage" => "会话存储",
        "sessions" => "会话",
        _ => name,
    }
}

fn localized_doctor_detail(status: &str, name: &str, detail: &serde_json::Value) -> String {
    match name {
        "model" => format!(
            "提供商 {}，模型 {}，API {}，上下文窗口 {}，最大输入 {}，网页搜索 {}",
            detail["provider"].as_str().unwrap_or("未指定"),
            detail["id"].as_str().unwrap_or("未知"),
            detail["api"].as_str().unwrap_or("未知"),
            detail["contextWindow"],
            detail["maxInputTokens"],
            localized_web_search_mode(detail["webSearch"].as_str().unwrap_or(""))
        ),
        "api_key" if detail.as_str() == Some("configured (value hidden)") => {
            "已配置（值已隐藏）".to_string()
        }
        "api_key" => "未配置；仅无需身份验证的端点可正常使用".to_string(),
        "sessions" if status == "ok" => detail
            .as_str()
            .and_then(|value| value.split_whitespace().next())
            .map_or_else(
                || detail.to_string(),
                |count| format!("可读取 {count} 个会话"),
            ),
        _ => detail
            .as_str()
            .map_or_else(|| detail.to_string(), ToString::to_string),
    }
}

fn localized_web_search_mode(mode: &str) -> &str {
    match mode {
        "disabled" => "禁用",
        "cached" => "缓存",
        "live" => "实时",
        _ => "未知",
    }
}

fn existing_ancestor(path: &std::path::Path) -> Option<std::path::PathBuf> {
    path.ancestors()
        .find(|candidate| candidate.exists())
        .map(std::path::Path::to_path_buf)
}

fn resolve_executable(command: &str, cwd: &std::path::Path) -> Option<std::path::PathBuf> {
    let command_path = std::path::Path::new(command);
    if command_path.components().count() > 1 {
        let path = if command_path.is_absolute() {
            command_path.to_path_buf()
        } else {
            cwd.join(command_path)
        };
        return path.is_file().then_some(path);
    }
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|directory| directory.join(command))
            .find(|candidate| candidate.is_file())
    })
}

fn prepare_mcp_servers(config: &mut AppConfig, bypass_approvals: bool, may_prompt: bool) {
    if bypass_approvals || config.mcp_servers.is_empty() {
        return;
    }
    let can_prompt = may_prompt && io::stdin().is_terminal() && io::stderr().is_terminal();
    let mut approved = Vec::new();
    for server in std::mem::take(&mut config.mcp_servers) {
        let details = mcp_startup_details(&server);
        let decision = if can_prompt {
            confirm_tool_execution(&format!("MCP 服务器 {}", server.name), &details)
        } else {
            ApprovalDecision::Deny
        };
        if matches!(
            decision,
            ApprovalDecision::ApproveOnce | ApprovalDecision::ApproveForSession
        ) {
            approved.push(server);
        } else {
            eprintln!("MCP 服务器 {:?} 因启动未获批准而被禁用", server.name);
        }
    }
    config.mcp_servers = approved;
}

fn mcp_startup_details(server: &McpServerConfig) -> String {
    let environment_variables = server.env.keys().cloned().collect::<Vec<_>>();
    serde_json::json!({
        "command": &server.command,
        "args": &server.args,
        "environmentVariables": environment_variables,
    })
    .to_string()
}

fn load_images(
    paths: &[std::path::PathBuf],
    cwd: &std::path::Path,
) -> Result<Vec<ImageAttachment>> {
    paths
        .iter()
        .map(|path| ImageAttachment::load(path, cwd))
        .collect()
}

fn confirm_session_delete(session: &str) -> Result<bool> {
    if !(io::stdin().is_terminal() && io::stderr().is_terminal()) {
        bail!("无法在非终端环境中确认删除；请使用 --force 和完整会话 UUID 重试");
    }
    eprintln!("永久删除会话 {session}？");
    eprintln!("此操作无法撤销。");
    eprint!("继续？[y/N]：");
    io::stderr().flush().context("刷新删除确认提示失败")?;
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .context("读取删除确认失败")?;
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn resolve_exec_prompt(parts: &[String]) -> Result<String> {
    if parts.is_empty() || (parts.len() == 1 && parts[0] == "-") {
        return read_stdin_prompt();
    }
    join_prompt(parts).ok_or_else(|| anyhow::anyhow!("提示词不能为空"))
}

fn read_stdin_prompt() -> Result<String> {
    if io::stdin().is_terminal() {
        bail!("未提供提示词；请传入文本或通过标准输入管道提供提示词");
    }
    let mut prompt = String::new();
    io::stdin()
        .read_to_string(&mut prompt)
        .context("从标准输入读取提示词失败")?;
    if prompt.trim().is_empty() {
        bail!("标准输入中没有提示词");
    }
    Ok(prompt)
}

async fn run_exec(
    mut agent: Agent,
    prompt: String,
    images: Vec<ImageAttachment>,
    json: bool,
    bypass_approvals: bool,
) -> Result<()> {
    let resume_pending = agent.has_pending_run();
    for failure in agent.mcp_startup_failures() {
        eprintln!(
            "警告：MCP 服务器 {:?} 启动失败，已禁用：{}",
            sanitize_terminal_text(&failure.server),
            sanitize_terminal_text(&failure.message)
        );
    }
    let (tx, mut rx) = mpsc::unbounded_channel();
    let (approvals, approval_task) = if bypass_approvals {
        (ApprovalGate::allow_all(), None)
    } else {
        let (gate, requests) = ApprovalGate::channel();
        let can_prompt = !json && io::stdin().is_terminal() && io::stderr().is_terminal();
        let task = tokio::spawn(handle_exec_approvals(requests, can_prompt));
        (gate, Some(task))
    };
    let cancel = CancellationToken::new();
    let signal_cancel = cancel.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_cancel.cancel();
        }
    });

    let run_tx = tx.clone();
    let run_cancel = cancel.clone();
    let task = tokio::spawn(async move {
        let result = if resume_pending {
            agent.resume_pending(&run_tx, &run_cancel, &approvals).await
        } else {
            agent
                .run(&prompt, images, &run_tx, &run_cancel, &approvals)
                .await
        };
        if let Err(error) = &result {
            let _ = run_tx.send(AgentEvent::Error {
                message: format!("{error:#}"),
            });
        }
        result
    });
    drop(tx);

    let mut streamed_text = false;
    while let Some(event) = rx.recv().await {
        if json {
            println!(
                "{}",
                serde_json::to_string(&event).context("编码 JSON 事件失败")?
            );
            continue;
        }
        match event {
            AgentEvent::TextDelta { text } => {
                print!("{}", sanitize_terminal_text(&text));
                io::stdout().flush().context("刷新标准输出失败")?;
                streamed_text = true;
            }
            AgentEvent::AssistantRetrying {
                attempt,
                max_attempts,
                message,
            } => {
                if streamed_text {
                    println!();
                    streamed_text = false;
                }
                eprintln!(
                    "[响应重试 {attempt}/{max_attempts}] {}",
                    sanitize_terminal_text(&message)
                );
            }
            AgentEvent::ToolStarted { name, .. } => {
                if streamed_text {
                    println!();
                    streamed_text = false;
                }
                eprintln!("[工具] {}", sanitize_terminal_text(&name));
            }
            AgentEvent::ToolFinished { name, is_error, .. } => {
                eprintln!(
                    "[{}] {}",
                    if is_error {
                        "工具错误"
                    } else {
                        "工具完成"
                    },
                    sanitize_terminal_text(&name)
                );
            }
            AgentEvent::WebSearchStarted { .. } => {
                if streamed_text {
                    println!();
                    streamed_text = false;
                }
                eprintln!("[网页搜索] 正在搜索");
            }
            AgentEvent::WebSearchFinished { action, .. } => {
                eprintln!(
                    "[网页搜索完成] {}",
                    sanitize_terminal_text(&action.description_zh())
                );
            }
            AgentEvent::ApprovalRequested { name, .. } => {
                eprintln!("[需要审批] {}", sanitize_terminal_text(&name));
            }
            AgentEvent::ApprovalResolved {
                name,
                approved,
                for_session,
                ..
            } => {
                let result = if approved {
                    if for_session {
                        "本次会话已允许"
                    } else {
                        "已允许一次"
                    }
                } else {
                    "已拒绝"
                };
                eprintln!("[审批：{result}] {}", sanitize_terminal_text(&name));
            }
            AgentEvent::ContextTrimmed {
                dropped_messages,
                dropped_turns,
                ..
            } => eprintln!("[上下文] 已省略较早 {dropped_turns} 轮中的 {dropped_messages} 条消息"),
            AgentEvent::CompactionStarted { reason } => {
                eprintln!("[上下文] 开始压缩（{}）", compaction_reason_label(reason));
            }
            AgentEvent::CompactionFinished {
                reason,
                tokens_before,
                tokens_after,
                ..
            } => eprintln!(
                "[上下文] 压缩完成（{}）：预计 token {tokens_before} -> {tokens_after}",
                compaction_reason_label(reason)
            ),
            AgentEvent::CompactionFailed { reason, message } => eprintln!(
                "[上下文] 压缩失败（{}），已回退为硬裁剪：{}",
                compaction_reason_label(reason),
                sanitize_terminal_text(&message)
            ),
            AgentEvent::Error { message } => {
                eprintln!("错误：{}", sanitize_terminal_text(&message));
            }
            AgentEvent::RunFinished | AgentEvent::Cancelled if streamed_text => {
                println!();
                streamed_text = false;
            }
            AgentEvent::RunStarted
            | AgentEvent::RunResumed
            | AgentEvent::AssistantStarted
            | AgentEvent::ReasoningSummaryDelta { .. }
            | AgentEvent::ReasoningSummaryPartAdded { .. }
            | AgentEvent::ReasoningSummaryFinished
            | AgentEvent::Usage { .. }
            | AgentEvent::RunFinished
            | AgentEvent::Cancelled => {}
        }
    }

    let status = task.await.context("Agent 任务失败")??;
    if let Some(approval_task) = approval_task {
        approval_task.await.context("审批处理任务失败")?;
    }
    if status == RunStatus::Cancelled {
        bail!("已取消");
    }
    Ok(())
}

async fn handle_exec_approvals(
    mut requests: mpsc::UnboundedReceiver<ApprovalRequest>,
    can_prompt: bool,
) {
    while let Some(request) = requests.recv().await {
        let decision = if can_prompt {
            let name = request.name.clone();
            let arguments = request.arguments.clone();
            tokio::task::spawn_blocking(move || confirm_tool_execution(&name, &arguments))
                .await
                .unwrap_or(ApprovalDecision::Deny)
        } else {
            ApprovalDecision::Deny
        };
        request.resolve(decision);
    }
}

fn confirm_tool_execution(name: &str, arguments: &str) -> ApprovalDecision {
    let arguments = format_tool_arguments(arguments);
    eprintln!("\n{name} 请求使用以下参数执行：");
    eprintln!("{arguments}");
    eprint!("是否允许？[y] 允许一次 / [a] 本次会话始终允许 / [N] 拒绝：");
    if io::stderr().flush().is_err() {
        return ApprovalDecision::Deny;
    }
    let mut answer = String::new();
    if io::stdin().read_line(&mut answer).is_err() {
        return ApprovalDecision::Deny;
    }
    match answer.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => ApprovalDecision::ApproveOnce,
        "a" | "always" => ApprovalDecision::ApproveForSession,
        _ => ApprovalDecision::Deny,
    }
}

fn compaction_reason_label(reason: CompactionReason) -> &'static str {
    match reason {
        CompactionReason::Manual => "手动",
        CompactionReason::Threshold => "达到阈值",
        CompactionReason::Overflow => "上下文溢出",
    }
}
