use std::io::{self, Read as _};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::cursor::{Hide, Show};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use pulldown_cmark::{
    Event as MarkdownEvent, HeadingLevel, Options as MarkdownOptions, Parser, Tag, TagEnd,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use unicode_width::UnicodeWidthChar;
use wl_clipboard_rs::paste::{ClipboardType, MimeType, Seat, get_contents, get_mime_types};

use crate::agent::{Agent, ModelChoice};
use crate::approval::{ApprovalDecision, ApprovalGate, ApprovalRequest, format_tool_arguments};
use crate::config::{ApiProtocol, ReasoningEffort, WebSearchMode};
use crate::event::{AgentEvent, CompactionReason};
use crate::protocol::{
    ChatMessage, ImageAttachment, MAX_IMAGE_BYTES, MessageRole, Usage, sanitize_terminal_text,
};

const APPROVAL_HEIGHT: u16 = 6;
const DELETE_CONFIRMATION_HEIGHT: u16 = 5;
const COLLAPSED_PASTE_CHAR_THRESHOLD: usize = 1_000;
const COLLAPSED_PASTE_LINE_THRESHOLD: usize = 8;
const INPUT_PREFIX_WIDTH: u16 = 2;
const MAX_INPUT_HEIGHT: u16 = 5;
const MAX_SLASH_SUGGESTIONS: u16 = 8;
const TOOL_ARGUMENT_PREVIEW_LINES: usize = 2;
const TOOL_OUTPUT_PREVIEW_LINES: usize = 5;
const TOOL_PREVIEW_LINE_CHARS: usize = 240;
const FRAME_INTERVAL: Duration = Duration::from_millis(50);
const CLIPBOARD_IMAGE_TYPES: [(&str, &str); 4] = [
    ("image/png", "png"),
    ("image/jpeg", "jpg"),
    ("image/gif", "gif"),
    ("image/webp", "webp"),
];

pub fn run_interactive(
    agent: Agent,
    initial_prompt: Option<String>,
    initial_images: Vec<ImageAttachment>,
    bypass_approvals: bool,
) -> Result<()> {
    let historical_compaction = agent.session().latest_compaction().cloned();
    let historical_messages = historical_compaction.as_ref().map_or_else(
        || agent.messages().to_vec(),
        |checkpoint| {
            agent.messages()[checkpoint
                .first_kept_message_index
                .min(agent.messages().len())..]
                .to_vec()
        },
    );
    let model = agent.model().to_string();
    let endpoint = agent.endpoint().to_string();
    let cwd = agent.session().cwd().to_path_buf();
    let has_pending_run = agent.has_pending_run();
    let mut state = UiState::new(model, endpoint, cwd);
    state.pending_images = initial_images;
    state.sync_from_agent(&agent);
    for failure in agent.mcp_startup_failures() {
        state.push_error(format!(
            "MCP 服务器 {:?} 启动失败，已禁用：{}",
            failure.server, failure.message
        ));
    }
    let agent = Arc::new(Mutex::new(agent));
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let (channel_gate, mut approval_rx) = ApprovalGate::channel();
    let approvals = if bypass_approvals {
        ApprovalGate::allow_all()
    } else {
        channel_gate
    };

    let screen = ScreenGuard::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend).context("初始化终端失败")?;
    terminal.clear().context("清空终端失败")?;

    if let Some(checkpoint) = historical_compaction {
        state.push_notice(format!(
            "此会话已从约 {} 个 token 压缩。\n\n{}",
            format_tokens(checkpoint.tokens_before),
            checkpoint.summary
        ));
    }
    for message in historical_messages {
        state.push_history(message);
    }
    let mut active_cancel = None;
    if has_pending_run {
        if let Some(prompt) = initial_prompt.filter(|prompt| !prompt.trim().is_empty()) {
            state.editor.insert_str(&prompt);
            state.push_notice("正在优先恢复中断的任务；提供的提示词已保留在输入框中。");
        }
        start_resume(
            Arc::clone(&agent),
            &event_tx,
            approvals.clone(),
            &mut state,
            &mut active_cancel,
        );
    } else if let Some(prompt) = initial_prompt.filter(|prompt| !prompt.trim().is_empty()) {
        let images = state.take_pending_images();
        start_run(
            Arc::clone(&agent),
            prompt,
            images,
            &event_tx,
            approvals.clone(),
            &mut state,
            &mut active_cancel,
        );
    }

    let mut deleted_session = None;
    let mut pending_approval: Option<ApprovalRequest> = None;
    let mut last_frame = Instant::now();
    'ui: loop {
        while let Ok(agent_event) = event_rx.try_recv() {
            state.apply_agent_event(agent_event);
            if !state.running {
                active_cancel = None;
                if let Some(request) = pending_approval.take() {
                    request.resolve(ApprovalDecision::Deny);
                }
                state.clear_pending_approval();
            }
        }
        if pending_approval.is_none()
            && let Ok(request) = approval_rx.try_recv()
        {
            state.set_pending_approval(&request);
            pending_approval = Some(request);
        }

        if last_frame.elapsed() >= FRAME_INTERVAL {
            state.spinner_frame = state.spinner_frame.wrapping_add(1);
            terminal
                .draw(|frame| render(frame, &mut state))
                .context("绘制终端界面失败")?;
            last_frame = Instant::now();
        }

        if !event::poll(Duration::from_millis(20)).context("轮询终端事件失败")? {
            continue;
        }
        match event::read().context("读取终端事件失败")? {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                match handle_key(key, &mut state, active_cancel.as_ref()) {
                    UiAction::None => {}
                    UiAction::Quit => break,
                    UiAction::Submit { prompt, images } => {
                        start_run(
                            Arc::clone(&agent),
                            prompt,
                            images,
                            &event_tx,
                            approvals.clone(),
                            &mut state,
                            &mut active_cancel,
                        );
                    }
                    UiAction::SelectModel(query) => match agent.try_lock() {
                        Ok(mut agent) => match agent.select_model(&query) {
                            Ok(()) => {
                                state.sync_from_agent(&agent);
                                state.push_notice(format!("模型已切换为 {}。", state.model));
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state.push_error("Agent 正忙，请等待当前任务完成。"),
                    },
                    UiAction::SetReasoning(effort) => match agent.try_lock() {
                        Ok(mut agent) => match agent.set_reasoning_effort(effort) {
                            Ok(()) => {
                                state.sync_from_agent(&agent);
                                state.push_notice(format!(
                                    "effort 已切换为 {}。",
                                    state.reasoning_effort
                                ));
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state.push_error("Agent 正忙，请等待当前任务完成。"),
                    },
                    UiAction::SetWebSearch(mode) => match agent.try_lock() {
                        Ok(mut agent) => match agent.set_web_search_mode(mode) {
                            Ok(()) => {
                                state.sync_from_agent(&agent);
                                state.push_notice(format!(
                                    "网页搜索模式已切换为 {}。",
                                    state.web_search_mode.label_zh()
                                ));
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state.push_error("Agent 正忙，请等待当前任务完成。"),
                    },
                    UiAction::Compact(instructions) => {
                        start_compaction(
                            Arc::clone(&agent),
                            instructions,
                            &event_tx,
                            &mut state,
                            &mut active_cancel,
                        );
                    }
                    UiAction::NewSession => match agent.try_lock() {
                        Ok(mut agent) => match agent.new_session() {
                            Ok(()) => {
                                state.reset_session();
                                state.sync_from_agent(&agent);
                                state.push_notice("已新建会话。");
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state.push_error("Agent 正忙，请等待当前任务完成。"),
                    },
                    UiAction::DeleteSession => match agent.try_lock() {
                        Ok(mut agent) => match agent.delete_session() {
                            Ok(id) => {
                                deleted_session = Some(id);
                                break 'ui;
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state.push_error("Agent 正忙，请等待当前任务完成。"),
                    },
                    UiAction::PasteClipboard => paste_from_clipboard(&mut state),
                    UiAction::ResolveApproval(decision) => {
                        if let Some(request) = pending_approval.take() {
                            request.resolve(decision);
                        }
                        state.clear_pending_approval();
                    }
                }
            }
            Event::Paste(text)
                if state.pending_approval.is_none()
                    && state.delete_confirmation == DeleteConfirmation::None =>
            {
                paste_text_or_image(&mut state, &text);
            }
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => state.scroll_lines_up(3),
                MouseEventKind::ScrollDown => state.scroll_lines_down(3),
                _ => {}
            },
            _ => {}
        }
    }

    drop(terminal);
    drop(screen);
    if let Some(id) = deleted_session {
        println!("已删除会话 {id}。");
    }
    Ok(())
}

fn start_resume(
    agent: Arc<Mutex<Agent>>,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    approvals: ApprovalGate,
    state: &mut UiState,
    active_cancel: &mut Option<CancellationToken>,
) {
    state.running = true;
    state.status = "正在恢复中断任务".to_string();
    let cancel = CancellationToken::new();
    *active_cancel = Some(cancel.clone());
    let tx = event_tx.clone();
    tokio::spawn(async move {
        let result = agent
            .lock()
            .await
            .resume_pending(&tx, &cancel, &approvals)
            .await;
        if let Err(error) = result {
            let _ = tx.send(AgentEvent::Error {
                message: format!("{error:#}"),
            });
        }
    });
}

fn start_run(
    agent: Arc<Mutex<Agent>>,
    prompt: String,
    images: Vec<ImageAttachment>,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    approvals: ApprovalGate,
    state: &mut UiState,
    active_cancel: &mut Option<CancellationToken>,
) {
    state.push_user(prompt.clone(), &images);
    state.running = true;
    let cancel = CancellationToken::new();
    *active_cancel = Some(cancel.clone());
    let tx = event_tx.clone();
    tokio::spawn(async move {
        let result = agent
            .lock()
            .await
            .run(&prompt, images, &tx, &cancel, &approvals)
            .await;
        if let Err(error) = result {
            let _ = tx.send(AgentEvent::Error {
                message: format!("{error:#}"),
            });
        }
    });
}

fn start_compaction(
    agent: Arc<Mutex<Agent>>,
    instructions: String,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    state: &mut UiState,
    active_cancel: &mut Option<CancellationToken>,
) {
    state.running = true;
    state.status = "正在压缩上下文".to_string();
    let cancel = CancellationToken::new();
    *active_cancel = Some(cancel.clone());
    let tx = event_tx.clone();
    tokio::spawn(async move {
        let instructions = (!instructions.trim().is_empty()).then_some(instructions);
        let _ = agent
            .lock()
            .await
            .compact(instructions.as_deref(), &tx, &cancel)
            .await;
    });
}

#[derive(Debug)]
enum UiAction {
    None,
    Quit,
    Submit {
        prompt: String,
        images: Vec<ImageAttachment>,
    },
    SelectModel(String),
    SetReasoning(ReasoningEffort),
    SetWebSearch(WebSearchMode),
    Compact(String),
    NewSession,
    DeleteSession,
    PasteClipboard,
    ResolveApproval(ApprovalDecision),
}

#[derive(Debug, Clone, Copy)]
struct SlashCommand {
    name: &'static str,
    accepts_argument: bool,
    description: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SlashSuggestion {
    label: String,
    replacement: String,
    description: String,
}

const SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "model",
        accepts_argument: true,
        description: "查看或切换模型",
    },
    SlashCommand {
        name: "effort",
        accepts_argument: true,
        description: "查看或设置 effort",
    },
    SlashCommand {
        name: "thinking",
        accepts_argument: true,
        description: "显示或隐藏思考过程",
    },
    SlashCommand {
        name: "search",
        accepts_argument: true,
        description: "查看或设置网页搜索模式",
    },
    SlashCommand {
        name: "compact",
        accepts_argument: true,
        description: "压缩当前上下文",
    },
    SlashCommand {
        name: "status",
        accepts_argument: false,
        description: "显示会话状态",
    },
    SlashCommand {
        name: "new",
        accepts_argument: false,
        description: "新建会话",
    },
    SlashCommand {
        name: "delete",
        accepts_argument: false,
        description: "删除当前会话",
    },
    SlashCommand {
        name: "clear",
        accepts_argument: false,
        description: "清空对话视图",
    },
    SlashCommand {
        name: "help",
        accepts_argument: false,
        description: "显示命令帮助",
    },
    SlashCommand {
        name: "exit",
        accepts_argument: false,
        description: "退出 MCode",
    },
];

fn slash_input(text: &str) -> Option<(&str, Option<&str>)> {
    let input = text.strip_prefix('/')?;
    if input.contains('\r') || input.contains('\n') {
        return None;
    }
    let Some((name, argument)) = input.split_once(char::is_whitespace) else {
        return Some((input, None));
    };
    let argument = argument.trim_start();
    (!argument.chars().any(char::is_whitespace)).then_some((name, Some(argument)))
}

fn slash_suggestions(state: &UiState) -> Vec<SlashSuggestion> {
    let text = state.editor.text();
    let Some((name, argument)) = slash_input(&text) else {
        return Vec::new();
    };
    let Some(argument) = argument else {
        let query = name.to_ascii_lowercase();
        return SLASH_COMMANDS
            .iter()
            .filter(|command| command.name.starts_with(&query))
            .map(|command| {
                let trailing_space = if command.accepts_argument { " " } else { "" };
                SlashSuggestion {
                    label: format!("/{}", command.name),
                    replacement: format!("/{}{trailing_space}", command.name),
                    description: command.description.to_string(),
                }
            })
            .collect();
    };

    let query = argument.to_ascii_lowercase();
    match name.to_ascii_lowercase().as_str() {
        "model" => state
            .model_choices
            .iter()
            .filter_map(|choice| {
                let qualified = format!("{}/{}", choice.provider, choice.id);
                let matches = qualified.to_ascii_lowercase().starts_with(&query)
                    || choice.id.to_ascii_lowercase().starts_with(&query)
                    || choice
                        .name
                        .as_deref()
                        .is_some_and(|value| value.to_ascii_lowercase().contains(&query));
                if !matches {
                    return None;
                }
                let qualified = sanitize_terminal_text(&qualified);
                let current = choice.id == state.model
                    && state.provider.as_deref() == Some(choice.provider.as_str());
                let detail = choice
                    .name
                    .as_deref()
                    .map_or_else(|| choice.api.to_string(), sanitize_terminal_text);
                let description = if current {
                    format!("当前 · {detail}")
                } else {
                    detail
                };
                Some(SlashSuggestion {
                    label: format!("/model {qualified}"),
                    replacement: format!("/model {qualified}"),
                    description,
                })
            })
            .collect(),
        "effort" => state
            .reasoning_choices
            .iter()
            .filter(|effort| effort.as_str().starts_with(&query))
            .map(|effort| {
                let mut markers = Vec::new();
                if *effort == state.reasoning_effort {
                    markers.push("当前");
                }
                if *effort == state.default_reasoning_effort {
                    markers.push("默认");
                }
                SlashSuggestion {
                    label: format!("/effort {effort}"),
                    replacement: format!("/effort {effort}"),
                    description: if markers.is_empty() {
                        "思考强度".to_string()
                    } else {
                        markers.join("、")
                    },
                }
            })
            .collect(),
        "thinking" => [
            ("show", "显示思考过程"),
            ("hide", "隐藏思考过程"),
            ("toggle", "切换显示状态"),
        ]
        .into_iter()
        .filter(|(value, _)| value.starts_with(&query))
        .map(|(value, description)| SlashSuggestion {
            label: format!("/thinking {value}"),
            replacement: format!("/thinking {value}"),
            description: description.to_string(),
        })
        .collect(),
        "search" => WebSearchMode::ALL
            .into_iter()
            .filter(|mode| mode.as_str().starts_with(&query))
            .map(|mode| SlashSuggestion {
                label: format!("/search {mode}"),
                replacement: format!("/search {mode}"),
                description: mode.label_zh().to_string(),
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn complete_slash_suggestion(state: &mut UiState) -> bool {
    let suggestions = slash_suggestions(state);
    let Some(suggestion) = suggestions.get(
        state
            .slash_selection
            .min(suggestions.len().saturating_sub(1)),
    ) else {
        return false;
    };
    state.editor.set_text(&suggestion.replacement);
    state.slash_selection = 0;
    true
}

fn handle_key(
    key: KeyEvent,
    state: &mut UiState,
    active_cancel: Option<&CancellationToken>,
) -> UiAction {
    if state.pending_approval.is_some() {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            if let Some(cancel) = active_cancel {
                cancel.cancel();
                state.status = "正在取消".to_string();
            }
            return UiAction::ResolveApproval(ApprovalDecision::Deny);
        }
        if key.code == KeyCode::Esc {
            if let Some(cancel) = active_cancel {
                cancel.cancel();
                state.status = "正在取消".to_string();
            }
            return UiAction::ResolveApproval(ApprovalDecision::Deny);
        }
        if key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
        {
            return UiAction::None;
        }
        return match key.code {
            KeyCode::Up | KeyCode::BackTab => {
                if let Some(approval) = state.pending_approval.as_mut() {
                    approval.selection = approval.selection.previous();
                }
                UiAction::None
            }
            KeyCode::Down | KeyCode::Tab => {
                if let Some(approval) = state.pending_approval.as_mut() {
                    approval.selection = approval.selection.next();
                }
                UiAction::None
            }
            KeyCode::Enter => state
                .pending_approval
                .as_ref()
                .map_or(UiAction::None, |approval| {
                    UiAction::ResolveApproval(approval.selection.decision())
                }),
            KeyCode::Char('1' | 'y' | 'Y') => {
                UiAction::ResolveApproval(ApprovalDecision::ApproveOnce)
            }
            KeyCode::Char('2' | 'a' | 'A') => {
                UiAction::ResolveApproval(ApprovalDecision::ApproveForSession)
            }
            KeyCode::Char('3' | 'n' | 'N') => UiAction::ResolveApproval(ApprovalDecision::Deny),
            _ => UiAction::None,
        };
    }

    if let DeleteConfirmation::Selecting(selection) = state.delete_confirmation {
        match key.code {
            KeyCode::Left | KeyCode::Up => {
                state.delete_confirmation = DeleteConfirmation::Selecting(DeleteChoice::Yes);
            }
            KeyCode::Right | KeyCode::Down => {
                state.delete_confirmation = DeleteConfirmation::Selecting(DeleteChoice::No);
            }
            KeyCode::Tab => {
                let selection = match selection {
                    DeleteChoice::Yes => DeleteChoice::No,
                    DeleteChoice::No => DeleteChoice::Yes,
                };
                state.delete_confirmation = DeleteConfirmation::Selecting(selection);
            }
            KeyCode::Char('y' | 'Y') => {
                state.delete_confirmation = DeleteConfirmation::None;
                return UiAction::DeleteSession;
            }
            KeyCode::Enter if selection == DeleteChoice::Yes => {
                state.delete_confirmation = DeleteConfirmation::None;
                return UiAction::DeleteSession;
            }
            KeyCode::Enter | KeyCode::Esc | KeyCode::Char('n' | 'N') => {
                state.delete_confirmation = DeleteConfirmation::None;
                state.push_notice("已取消删除。");
            }
            KeyCode::Char('c' | 'C') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                state.delete_confirmation = DeleteConfirmation::None;
                state.push_notice("已取消删除。");
            }
            _ => {}
        }
        return UiAction::None;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('c') => {
                if state.running {
                    if let Some(cancel) = active_cancel {
                        cancel.cancel();
                        state.status = "正在取消".to_string();
                    }
                    return UiAction::None;
                }
                return UiAction::Quit;
            }
            KeyCode::Char('d') if state.editor.is_empty() && !state.running => {
                return UiAction::Quit;
            }
            KeyCode::Char('j') => {
                state.editor.insert('\n');
                return UiAction::None;
            }
            KeyCode::Char('v') => return UiAction::PasteClipboard,
            KeyCode::Home => {
                state.scroll_lines_up(usize::MAX);
                return UiAction::None;
            }
            KeyCode::End => {
                state.scroll_lines_down(usize::MAX);
                return UiAction::None;
            }
            _ => {}
        }
    }

    let suggestions = slash_suggestions(state);
    if !suggestions.is_empty() && key.modifiers.is_empty() {
        state.slash_selection = state
            .slash_selection
            .min(suggestions.len().saturating_sub(1));
        match key.code {
            KeyCode::Up => {
                state.slash_selection = if state.slash_selection == 0 {
                    suggestions.len() - 1
                } else {
                    state.slash_selection - 1
                };
                return UiAction::None;
            }
            KeyCode::Down => {
                state.slash_selection = (state.slash_selection + 1) % suggestions.len();
                return UiAction::None;
            }
            KeyCode::Tab => {
                complete_slash_suggestion(state);
                return UiAction::None;
            }
            KeyCode::Enter if !state.running => {
                complete_slash_suggestion(state);
                return submit_editor(state);
            }
            _ => {}
        }
    }

    match key.code {
        KeyCode::Esc if state.running => {
            if let Some(cancel) = active_cancel {
                cancel.cancel();
                state.status = "正在取消".to_string();
            }
        }
        KeyCode::Enter
            if key.modifiers.contains(KeyModifiers::SHIFT)
                || key.modifiers.contains(KeyModifiers::ALT) =>
        {
            state.editor.insert('\n');
        }
        KeyCode::Enter if !state.running => {
            return submit_editor(state);
        }
        KeyCode::Char(character)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            state.editor.insert(character);
            state.slash_selection = 0;
        }
        KeyCode::Backspace => {
            if state.editor.is_empty() {
                if state.pending_images.pop().is_some() {
                    state.status = attachment_status(state.pending_images.len());
                }
            } else {
                state.editor.backspace();
            }
            state.slash_selection = 0;
        }
        KeyCode::Delete => {
            state.editor.delete();
            state.slash_selection = 0;
        }
        KeyCode::Left => state.editor.move_left(),
        KeyCode::Right => state.editor.move_right(),
        KeyCode::Home => state.editor.move_home(),
        KeyCode::End => state.editor.move_end(),
        KeyCode::Up => state.scroll_lines_up(1),
        KeyCode::Down => state.scroll_lines_down(1),
        KeyCode::PageUp => state.scroll_up(),
        KeyCode::PageDown => state.scroll_down(),
        _ => {}
    }
    UiAction::None
}

fn submit_editor(state: &mut UiState) -> UiAction {
    state.slash_selection = 0;
    let prompt = state.editor.take();
    if prompt.trim().is_empty() {
        return UiAction::None;
    }
    let trimmed = prompt.trim();
    let Some(command) = trimmed.strip_prefix('/') else {
        state.delete_confirmation = DeleteConfirmation::None;
        return UiAction::Submit {
            prompt,
            images: state.take_pending_images(),
        };
    };
    let (name, argument) = command
        .split_once(char::is_whitespace)
        .map_or((command, ""), |(name, argument)| (name, argument.trim()));
    if name != "delete" {
        state.delete_confirmation = DeleteConfirmation::None;
    }
    match name {
        "exit" => UiAction::Quit,
        "clear" => {
            state.messages.clear();
            state.follow_tail = true;
            UiAction::None
        }
        "new" => UiAction::NewSession,
        "compact" => UiAction::Compact(argument.to_string()),
        "delete" if argument.is_empty() => {
            state.delete_confirmation = DeleteConfirmation::Selecting(DeleteChoice::No);
            UiAction::None
        }
        "delete" => {
            state.push_error("用法：/delete");
            UiAction::None
        }
        "model" if argument.is_empty() => {
            let notice = state.model_list_notice();
            state.push_notice(notice);
            UiAction::None
        }
        "model" => UiAction::SelectModel(argument.to_string()),
        "effort" if argument.is_empty() => {
            let notice = state.effort_list_notice();
            state.push_notice(notice);
            UiAction::None
        }
        "effort" => {
            if let Some(effort) = parse_reasoning_effort(argument)
                && state.reasoning_choices.contains(&effort)
            {
                return UiAction::SetReasoning(effort);
            }
            let choices = state
                .reasoning_choices
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("、");
            state.push_error(format!(
                "当前模型 {} 未配置 effort {argument:?}。可选值：{choices}。",
                state.qualified_model()
            ));
            UiAction::None
        }
        "thinking" => {
            match argument.to_ascii_lowercase().as_str() {
                "" | "toggle" => state.toggle_thinking(),
                "show" => state.set_thinking_visible(true),
                "hide" => state.set_thinking_visible(false),
                _ => state.push_error("用法：/thinking、/thinking show 或 /thinking hide"),
            }
            UiAction::None
        }
        "search" if argument.is_empty() => {
            state.push_notice(format!(
                "网页搜索：{}\n使用 /search <disabled|cached|live> 选择。",
                state.web_search_mode.label_zh()
            ));
            UiAction::None
        }
        "search" => {
            if let Some(mode) = parse_web_search_mode(argument) {
                return UiAction::SetWebSearch(mode);
            }
            state.push_error(format!(
                "未知的网页搜索模式 {argument:?}。可用值：disabled、cached、live。"
            ));
            UiAction::None
        }
        "status" => {
            let notice = state.status_notice();
            state.push_notice(notice);
            UiAction::None
        }
        "help" => {
            state.push_notice(
                "命令：/model [ID]、/effort [级别]、/thinking [show|hide]、/search [模式]、/compact [说明]、/status、/new、/delete、/clear、/help、/exit",
            );
            UiAction::None
        }
        _ => {
            state.push_error(format!("未知命令：/{name}"));
            UiAction::None
        }
    }
}

fn paste_from_clipboard(state: &mut UiState) {
    let clipboard = ClipboardType::Regular;
    let seat = Seat::Unspecified;
    let mime_types = match get_mime_types(clipboard, seat) {
        Ok(mime_types) => mime_types,
        Err(error) => {
            state.push_error(format!("无法访问 Wayland 剪贴板：{error}"));
            return;
        }
    };

    if let Some(&(mime_type, extension)) = CLIPBOARD_IMAGE_TYPES
        .iter()
        .find(|(mime_type, _)| mime_types.contains(*mime_type))
    {
        let (reader, _) = match get_contents(clipboard, seat, MimeType::Specific(mime_type)) {
            Ok(contents) => contents,
            Err(error) => {
                state.push_error(format!("无法读取 Wayland 剪贴板图片：{error}"));
                return;
            }
        };
        let bytes = match read_limited(reader, MAX_IMAGE_BYTES) {
            Ok(bytes) => bytes,
            Err(error) => {
                state.push_error(format!("无法读取 Wayland 剪贴板图片：{error}"));
                return;
            }
        };
        let name = format!("clipboard-{}.{}", state.pending_images.len() + 1, extension);
        match ImageAttachment::from_encoded_bytes(name, bytes) {
            Ok(image) => attach_image(state, image),
            Err(error) => state.push_error(format!("无法附加剪贴板图片：{error:#}")),
        }
        return;
    }

    let (reader, _) = match get_contents(clipboard, seat, MimeType::Text) {
        Ok(contents) => contents,
        Err(error) => {
            state.push_error(format!("Wayland 剪贴板中没有可粘贴的图片或文本：{error}"));
            return;
        }
    };
    match read_limited(reader, MAX_IMAGE_BYTES).and_then(|bytes| {
        String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }) {
        Ok(text) if !text.is_empty() => paste_text_or_image(state, &text),
        Ok(_) => state.push_error("Wayland 剪贴板为空。"),
        Err(error) => state.push_error(format!("无法读取 Wayland 剪贴板文本：{error}")),
    }
}

fn read_limited(reader: impl io::Read, max_bytes: u64) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "剪贴板内容超过 20 MiB 限制",
        ));
    }
    Ok(bytes)
}

fn paste_text_or_image(state: &mut UiState, text: &str) {
    let image =
        pasted_image_path(text).and_then(|path| ImageAttachment::load(&path, &state.cwd).ok());
    if let Some(image) = image {
        attach_image(state, image);
    } else {
        state.editor.insert_paste(text);
    }
    state.slash_selection = 0;
}

fn pasted_image_path(text: &str) -> Option<PathBuf> {
    if text.contains(['\r', '\n', '\0']) {
        return None;
    }
    let value = text.trim();
    if value.is_empty() {
        return None;
    }
    let value = match value.as_bytes() {
        [b'\'', .., b'\''] | [b'"', .., b'"'] if value.len() >= 2 => &value[1..value.len() - 1],
        _ => value,
    };
    if value.starts_with("file://") {
        return url::Url::parse(value).ok()?.to_file_path().ok();
    }
    Some(PathBuf::from(value))
}

fn attach_image(state: &mut UiState, image: ImageAttachment) {
    state.pending_images.push(image);
    state.status = attachment_status(state.pending_images.len());
}

fn attachment_status(count: usize) -> String {
    if count == 0 {
        "就绪".to_string()
    } else {
        format!("已附加 {count} 张图片")
    }
}

fn parse_reasoning_effort(value: &str) -> Option<ReasoningEffort> {
    ReasoningEffort::ALL
        .into_iter()
        .find(|effort| effort.as_str().eq_ignore_ascii_case(value))
}

fn parse_web_search_mode(value: &str) -> Option<WebSearchMode> {
    WebSearchMode::ALL
        .into_iter()
        .find(|mode| mode.as_str().eq_ignore_ascii_case(value))
}

#[derive(Debug)]
enum EditorItem {
    Character(char),
    Paste { content: String, label: String },
}

#[derive(Debug, Default)]
struct Editor {
    items: Vec<EditorItem>,
    cursor: usize,
}

impl Editor {
    fn insert(&mut self, character: char) {
        self.items
            .insert(self.cursor, EditorItem::Character(character));
        self.cursor += 1;
    }

    fn insert_str(&mut self, text: &str) {
        for character in text.chars() {
            self.insert(character);
        }
    }

    fn insert_paste(&mut self, text: &str) {
        let character_count = text.chars().count();
        let line_count = text.bytes().filter(|byte| byte == &b'\n').count() + 1;
        if character_count < COLLAPSED_PASTE_CHAR_THRESHOLD
            && line_count < COLLAPSED_PASTE_LINE_THRESHOLD
        {
            self.insert_str(text);
            return;
        }
        self.items.insert(
            self.cursor,
            EditorItem::Paste {
                content: text.to_string(),
                label: format!("[已粘贴 {character_count} 个字符]"),
            },
        );
        self.cursor += 1;
    }

    fn set_text(&mut self, text: &str) {
        self.items = text.chars().map(EditorItem::Character).collect();
        self.cursor = self.items.len();
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.items.remove(self.cursor);
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.items.len() {
            self.items.remove(self.cursor);
        }
    }

    fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    fn move_right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.items.len());
    }

    fn move_home(&mut self) {
        self.cursor = self.items[..self.cursor]
            .iter()
            .rposition(|item| matches!(item, EditorItem::Character('\n')))
            .map_or(0, |index| index + 1);
    }

    fn move_end(&mut self) {
        self.cursor = self.items[self.cursor..]
            .iter()
            .position(|item| matches!(item, EditorItem::Character('\n')))
            .map_or(self.items.len(), |index| self.cursor + index);
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn text(&self) -> String {
        let mut text = String::new();
        for item in &self.items {
            match item {
                EditorItem::Character(character) => text.push(*character),
                EditorItem::Paste { label, .. } => text.push_str(label),
            }
        }
        text
    }

    fn take(&mut self) -> String {
        self.cursor = 0;
        let mut text = String::new();
        for item in self.items.drain(..) {
            match item {
                EditorItem::Character(character) => text.push(character),
                EditorItem::Paste { content, .. } => text.push_str(&content),
            }
        }
        text
    }

    fn position_at(&self, end: usize, width: u16) -> (usize, usize) {
        let width = usize::from(width.max(1));
        let mut row = 0usize;
        let mut column = 0usize;
        let mut advance = |character: char| {
            if character == '\n' {
                row += 1;
                column = 0;
                return;
            }
            let character_width = character.width().unwrap_or(0).max(1);
            if column + character_width > width {
                row += 1;
                column = 0;
            }
            column += character_width;
            if column >= width {
                row += 1;
                column = 0;
            }
        };
        for item in self.items.iter().take(end) {
            match item {
                EditorItem::Character(character) => advance(*character),
                EditorItem::Paste { label, .. } => {
                    for character in label.chars() {
                        advance(character);
                    }
                }
            }
        }
        (row, column)
    }

    fn rendered_height(&self, width: u16) -> u16 {
        let (row, _) = self.position_at(self.items.len(), width);
        u16::try_from(row.saturating_add(1)).unwrap_or(u16::MAX)
    }

    fn cursor_layout(&self, width: u16, visible_height: u16) -> (u16, u16, u16) {
        let (row, column) = self.position_at(self.cursor, width);
        let visible_height = usize::from(visible_height.max(1));
        let scroll = row.saturating_sub(visible_height - 1);
        (
            u16::try_from(column).unwrap_or(u16::MAX),
            u16::try_from(row.saturating_sub(scroll)).unwrap_or(u16::MAX),
            u16::try_from(scroll).unwrap_or(u16::MAX),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewRole {
    User,
    Assistant,
    Tool,
    Notice,
    Error,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum DeleteConfirmation {
    #[default]
    None,
    Selecting(DeleteChoice),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum DeleteChoice {
    Yes,
    #[default]
    No,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ThinkingDisplay {
    #[default]
    Hidden,
    Shown,
}

#[derive(Debug)]
struct ViewMessage {
    role: ViewRole,
    title: String,
    content: String,
    reasoning: String,
    tool_arguments: Option<String>,
    tool_id: Option<String>,
    running: bool,
}

#[derive(Debug)]
struct ApprovalView {
    name: String,
    arguments: String,
    selection: ApprovalChoice,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ApprovalChoice {
    #[default]
    ApproveOnce,
    ApproveForSession,
    Deny,
}

impl ApprovalChoice {
    fn previous(self) -> Self {
        match self {
            Self::ApproveOnce | Self::ApproveForSession => Self::ApproveOnce,
            Self::Deny => Self::ApproveForSession,
        }
    }

    fn next(self) -> Self {
        match self {
            Self::ApproveOnce => Self::ApproveForSession,
            Self::ApproveForSession | Self::Deny => Self::Deny,
        }
    }

    fn decision(self) -> ApprovalDecision {
        match self {
            Self::ApproveOnce => ApprovalDecision::ApproveOnce,
            Self::ApproveForSession => ApprovalDecision::ApproveForSession,
            Self::Deny => ApprovalDecision::Deny,
        }
    }
}

#[derive(Debug)]
struct UiState {
    model: String,
    provider: Option<String>,
    api: ApiProtocol,
    reasoning_effort: ReasoningEffort,
    default_reasoning_effort: ReasoningEffort,
    web_search_mode: WebSearchMode,
    endpoint: String,
    cwd: std::path::PathBuf,
    model_choices: Vec<ModelChoice>,
    reasoning_choices: Vec<ReasoningEffort>,
    messages: Vec<ViewMessage>,
    editor: Editor,
    running: bool,
    current_assistant: Option<usize>,
    generation_start: Option<usize>,
    status: String,
    usage: Usage,
    context_tokens: u64,
    context_window: u64,
    max_input_tokens: u64,
    usage_estimated: bool,
    spinner_frame: usize,
    scroll: usize,
    max_scroll: usize,
    viewport_height: usize,
    follow_tail: bool,
    delete_confirmation: DeleteConfirmation,
    pending_images: Vec<ImageAttachment>,
    slash_selection: usize,
    thinking_display: ThinkingDisplay,
    mcp_server_count: usize,
    mcp_tool_count: usize,
    pending_approval: Option<ApprovalView>,
}

impl UiState {
    fn new(model: String, endpoint: String, cwd: std::path::PathBuf) -> Self {
        Self {
            model,
            provider: None,
            api: ApiProtocol::ChatCompletions,
            reasoning_effort: ReasoningEffort::Off,
            default_reasoning_effort: ReasoningEffort::Off,
            web_search_mode: WebSearchMode::Disabled,
            endpoint,
            cwd,
            model_choices: Vec::new(),
            reasoning_choices: ReasoningEffort::ALL.to_vec(),
            messages: Vec::new(),
            editor: Editor::default(),
            running: false,
            current_assistant: None,
            generation_start: None,
            status: "就绪".to_string(),
            usage: Usage::default(),
            context_tokens: 0,
            context_window: 128_000,
            max_input_tokens: 128_000,
            usage_estimated: false,
            spinner_frame: 0,
            scroll: 0,
            max_scroll: 0,
            viewport_height: 1,
            follow_tail: true,
            delete_confirmation: DeleteConfirmation::None,
            pending_images: Vec::new(),
            slash_selection: 0,
            thinking_display: ThinkingDisplay::Hidden,
            mcp_server_count: 0,
            mcp_tool_count: 0,
            pending_approval: None,
        }
    }

    fn sync_from_agent(&mut self, agent: &Agent) {
        self.model = sanitize_terminal_text(agent.model());
        self.provider = agent.provider().map(sanitize_terminal_text);
        self.api = agent.api();
        self.reasoning_effort = agent.reasoning_effort();
        self.default_reasoning_effort = agent.default_reasoning_effort();
        self.web_search_mode = agent.web_search_mode();
        self.endpoint = sanitize_terminal_text(agent.endpoint());
        self.model_choices = agent.model_choices();
        self.reasoning_choices = agent.available_reasoning_efforts();
        self.usage = agent.total_usage();
        self.context_tokens = agent.context_tokens();
        self.context_window = agent.context_window();
        self.max_input_tokens = agent.max_input_tokens();
        self.usage_estimated = agent.usage_estimated();
        self.mcp_server_count = agent.mcp_server_count();
        self.mcp_tool_count = agent.mcp_tool_count();
    }

    fn push_history(&mut self, message: ChatMessage) {
        match message.role {
            MessageRole::System => {}
            MessageRole::User => {
                let content = sanitize_terminal_text(&format_user_content(
                    message.content.unwrap_or_default(),
                    &message.images,
                ));
                self.messages.push(ViewMessage {
                    role: ViewRole::User,
                    title: String::new(),
                    content,
                    reasoning: String::new(),
                    tool_arguments: None,
                    tool_id: None,
                    running: false,
                });
            }
            MessageRole::Assistant => {
                let content = sanitize_terminal_text(&message.content.unwrap_or_default());
                let reasoning =
                    sanitize_terminal_text(&message.reasoning_content.unwrap_or_default());
                if !content.is_empty() || !reasoning.is_empty() {
                    self.messages.push(ViewMessage {
                        role: ViewRole::Assistant,
                        title: String::new(),
                        content,
                        reasoning,
                        tool_arguments: None,
                        tool_id: None,
                        running: false,
                    });
                }
                for call in message.tool_calls {
                    let name = sanitize_terminal_text(&call.function.name);
                    self.messages.push(ViewMessage {
                        role: ViewRole::Tool,
                        title: name.clone(),
                        content: String::new(),
                        reasoning: String::new(),
                        tool_arguments: Some(format_tool_input(&name, &call.function.arguments)),
                        tool_id: Some(call.id),
                        running: false,
                    });
                }
            }
            MessageRole::Tool => {
                let tool_id = message.tool_call_id;
                let content = truncate_for_ui(&sanitize_terminal_text(
                    &message.content.unwrap_or_default(),
                ));
                if let Some(existing) = tool_id.as_deref().and_then(|id| {
                    self.messages
                        .iter_mut()
                        .rev()
                        .find(|view| view.tool_id.as_deref() == Some(id))
                }) {
                    existing.content = content;
                } else {
                    self.messages.push(ViewMessage {
                        role: ViewRole::Tool,
                        title: "工具".to_string(),
                        content,
                        reasoning: String::new(),
                        tool_arguments: None,
                        tool_id,
                        running: false,
                    });
                }
            }
        }
        self.follow_tail = true;
        self.delete_confirmation = DeleteConfirmation::None;
    }

    fn push_user(&mut self, prompt: String, images: &[ImageAttachment]) {
        self.messages.push(ViewMessage {
            role: ViewRole::User,
            title: String::new(),
            content: sanitize_terminal_text(&format_user_content(prompt, images)),
            reasoning: String::new(),
            tool_arguments: None,
            tool_id: None,
            running: false,
        });
        self.follow_tail = true;
    }

    fn take_pending_images(&mut self) -> Vec<ImageAttachment> {
        std::mem::take(&mut self.pending_images)
    }

    fn push_notice(&mut self, content: impl AsRef<str>) {
        self.messages.push(ViewMessage {
            role: ViewRole::Notice,
            title: "MCode".to_string(),
            content: sanitize_terminal_text(content.as_ref()),
            reasoning: String::new(),
            tool_arguments: None,
            tool_id: None,
            running: false,
        });
        self.follow_tail = true;
    }

    fn push_error(&mut self, content: impl AsRef<str>) {
        self.messages.push(ViewMessage {
            role: ViewRole::Error,
            title: "错误".to_string(),
            content: sanitize_terminal_text(content.as_ref()),
            reasoning: String::new(),
            tool_arguments: None,
            tool_id: None,
            running: false,
        });
        self.follow_tail = true;
    }

    fn set_pending_approval(&mut self, request: &ApprovalRequest) {
        let name = sanitize_terminal_text(&request.name);
        self.pending_approval = Some(ApprovalView {
            arguments: format_tool_input(&name, &request.arguments),
            name,
            selection: ApprovalChoice::default(),
        });
        self.status = "等待审批".to_string();
    }

    fn clear_pending_approval(&mut self) {
        self.pending_approval = None;
        if self.running && self.status == "等待审批" {
            self.status = "处理中".to_string();
        }
    }

    fn reset_session(&mut self) {
        self.messages.clear();
        self.usage = Usage::default();
        self.context_tokens = 0;
        self.usage_estimated = false;
        self.status = "就绪".to_string();
        self.follow_tail = true;
        self.delete_confirmation = DeleteConfirmation::None;
        self.pending_images.clear();
        self.pending_approval = None;
    }

    fn model_list_notice(&self) -> String {
        if self.model_choices.is_empty() {
            return format!(
                "当前模型：{}\n~/.mcode/agent/models.json 中没有模型列表；仍可使用 /model <ID> 选择当前端点上的模型。",
                self.model
            );
        }
        let mut lines = vec!["已配置的模型：".to_string()];
        for choice in &self.model_choices {
            let selected = if choice.id == self.model
                && self.provider.as_deref() == Some(choice.provider.as_str())
            {
                "*"
            } else {
                " "
            };
            let name = choice.name.as_deref().map_or_else(String::new, |name| {
                format!(" ({})", sanitize_terminal_text(name))
            });
            let reasoning = if choice.reasoning {
                "，支持思考"
            } else {
                ""
            };
            let limits = if choice.max_input_tokens == choice.context_window {
                format!("{} 上下文/输入", format_tokens(choice.context_window))
            } else {
                format!(
                    "{} 上下文，{} 最大输入",
                    format_tokens(choice.context_window),
                    format_tokens(choice.max_input_tokens)
                )
            };
            lines.push(format!(
                "{selected} {}/{}{} - {limits}, {}{}",
                sanitize_terminal_text(&choice.provider),
                sanitize_terminal_text(&choice.id),
                name,
                choice.api,
                reasoning
            ));
        }
        lines.push("使用 /model <提供商/模型> 选择。".to_string());
        lines.join("\n")
    }

    fn effort_list_notice(&self) -> String {
        let mut lines = vec![format!(
            "当前模型 {} 配置的 effort 级别：",
            self.qualified_model()
        )];
        for effort in &self.reasoning_choices {
            let selected = if *effort == self.reasoning_effort {
                "*"
            } else {
                " "
            };
            let default = if *effort == self.default_reasoning_effort {
                "（默认）"
            } else {
                ""
            };
            lines.push(format!("{selected} {effort}{default}"));
        }
        let choices = self
            .reasoning_choices
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("|");
        lines.push(format!("使用 /effort <{choices}> 选择。"));
        lines.join("\n")
    }

    fn qualified_model(&self) -> String {
        self.provider.as_deref().map_or_else(
            || self.model.clone(),
            |provider| format!("{provider}/{}", self.model),
        )
    }

    fn toggle_thinking(&mut self) {
        let visible = self.thinking_display == ThinkingDisplay::Hidden;
        self.set_thinking_visible(visible);
    }

    fn set_thinking_visible(&mut self, visible: bool) {
        self.thinking_display = if visible {
            ThinkingDisplay::Shown
        } else {
            ThinkingDisplay::Hidden
        };
        self.follow_tail = true;
        self.push_notice(if visible {
            "已显示思考过程。"
        } else {
            "已隐藏思考过程。"
        });
    }

    fn status_notice(&self) -> String {
        let qualified_model = self.qualified_model();
        let estimate = if self.usage_estimated { "~" } else { "" };
        let percent = format_context_percent(self.context_tokens, self.max_input_tokens);
        format!(
            "模型：{qualified_model}\nAPI：{}\neffort：{}\n网页搜索：{}\n输入：{estimate}{}/{}（{percent}%）\n模型上下文窗口：{}\nToken：{estimate}输入 {}，输出 {}\nMCP：{} 个服务器，{} 个工具\n端点：{}\n工作目录：{}",
            self.api,
            self.reasoning_effort,
            self.web_search_mode.label_zh(),
            format_tokens(self.context_tokens),
            format_tokens(self.max_input_tokens),
            format_tokens(self.context_window),
            format_tokens(self.usage.prompt_tokens),
            format_tokens(self.usage.completion_tokens),
            self.mcp_server_count,
            self.mcp_tool_count,
            self.endpoint,
            sanitize_terminal_text(&self.cwd.to_string_lossy())
        )
    }

    fn apply_agent_event(&mut self, event: AgentEvent) {
        let follow_tail = self.follow_tail;
        match event {
            AgentEvent::RunStarted | AgentEvent::RunResumed => {
                self.status = "处理中".to_string();
                self.running = true;
            }
            AgentEvent::AssistantStarted => {
                let index = self.start_assistant_message();
                self.generation_start = Some(index);
            }
            AgentEvent::AssistantRetrying {
                attempt,
                max_attempts,
                message,
            } => {
                let index = self
                    .generation_start
                    .filter(|index| *index < self.messages.len())
                    .unwrap_or_else(|| {
                        let index = self.start_assistant_message();
                        self.generation_start = Some(index);
                        index
                    });
                self.messages.truncate(index + 1);
                if let Some(assistant) = self.messages.get_mut(index) {
                    assistant.title.clear();
                    assistant.content.clear();
                    assistant.reasoning.clear();
                    assistant.running = true;
                }
                self.current_assistant = Some(index);
                self.status = format!(
                    "正在重试响应（{attempt}/{max_attempts}）：{}",
                    sanitize_terminal_text(&message)
                );
            }
            AgentEvent::TextDelta { text } => {
                let index = self.ensure_assistant_message();
                if let Some(message) = self.messages.get_mut(index) {
                    message.content.push_str(&sanitize_terminal_text(&text));
                }
            }
            AgentEvent::ReasoningDelta { text } => {
                let index = self.ensure_assistant_message();
                if let Some(message) = self.messages.get_mut(index) {
                    message.reasoning.push_str(&sanitize_terminal_text(&text));
                }
            }
            AgentEvent::ApprovalRequested {
                id,
                name,
                arguments,
            } => {
                self.finish_current_assistant_for_tool();
                let name = sanitize_terminal_text(&name);
                self.messages.push(ViewMessage {
                    role: ViewRole::Tool,
                    title: format!("等待审批：{name}"),
                    content: String::new(),
                    reasoning: String::new(),
                    tool_arguments: Some(format_tool_input(&name, &arguments)),
                    tool_id: Some(id),
                    running: true,
                });
                self.status = "等待审批".to_string();
            }
            AgentEvent::ApprovalResolved {
                id,
                name,
                approved,
                for_session,
            } => {
                let name = sanitize_terminal_text(&name);
                if let Some(message) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|message| message.tool_id.as_deref() == Some(id.as_str()))
                {
                    message.title = if approved {
                        format!(
                            "已允许{}：{name}",
                            if for_session { "（本次会话）" } else { "" }
                        )
                    } else {
                        format!("已拒绝：{name}")
                    };
                    message.running = false;
                    if !approved {
                        message.role = ViewRole::Error;
                    }
                }
                self.clear_pending_approval();
            }
            AgentEvent::ToolStarted {
                id,
                name,
                arguments,
            } => {
                self.finish_current_assistant_for_tool();
                let name = sanitize_terminal_text(&name);
                let arguments = format_tool_input(&name, &arguments);
                if let Some(message) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|message| message.tool_id.as_deref() == Some(id.as_str()))
                {
                    message.role = ViewRole::Tool;
                    message.title = name;
                    message.content.clear();
                    message.tool_arguments = Some(arguments);
                    message.running = true;
                } else {
                    self.messages.push(ViewMessage {
                        role: ViewRole::Tool,
                        title: name,
                        content: String::new(),
                        reasoning: String::new(),
                        tool_arguments: Some(arguments),
                        tool_id: Some(id),
                        running: true,
                    });
                }
            }
            AgentEvent::ToolFinished {
                id,
                name,
                output,
                is_error,
            } => {
                if let Some(message) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|message| message.tool_id.as_deref() == Some(id.as_str()))
                {
                    message.title = sanitize_terminal_text(&name);
                    message.content = truncate_for_ui(&sanitize_terminal_text(&output));
                    message.running = false;
                    if is_error {
                        message.role = ViewRole::Error;
                    }
                }
            }
            AgentEvent::WebSearchStarted { id } => {
                self.finish_current_assistant_for_tool();
                self.messages.push(ViewMessage {
                    role: ViewRole::Tool,
                    title: "网页搜索".to_string(),
                    content: "正在搜索...".to_string(),
                    reasoning: String::new(),
                    tool_arguments: None,
                    tool_id: Some(id),
                    running: true,
                });
                self.status = "正在搜索网页".to_string();
            }
            AgentEvent::WebSearchFinished { id, action } => {
                if let Some(message) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|message| message.tool_id.as_deref() == Some(id.as_str()))
                {
                    message.content = sanitize_terminal_text(&action.description_zh());
                    message.running = false;
                }
                self.status = "处理中".to_string();
            }
            AgentEvent::Usage {
                usage,
                context_tokens,
                context_window,
                max_input_tokens,
                estimated,
            } => {
                self.usage.prompt_tokens =
                    self.usage.prompt_tokens.saturating_add(usage.prompt_tokens);
                self.usage.completion_tokens = self
                    .usage
                    .completion_tokens
                    .saturating_add(usage.completion_tokens);
                self.usage.total_tokens =
                    self.usage.total_tokens.saturating_add(usage.total_tokens);
                self.context_tokens = context_tokens;
                self.context_window = context_window;
                self.max_input_tokens = max_input_tokens;
                self.usage_estimated = estimated;
            }
            AgentEvent::ContextTrimmed {
                dropped_messages,
                dropped_turns,
                estimated_tokens,
            } => self.push_notice(format!(
                "达到上下文限制：已省略较早 {dropped_turns} 轮中的 {dropped_messages} 条消息；预计输入为 {} 个 token。",
                format_tokens(estimated_tokens)
            )),
            AgentEvent::CompactionStarted { .. } => {
                self.finish_current_assistant_for_tool();
                self.status = "正在压缩上下文".to_string();
            }
            AgentEvent::CompactionFinished {
                reason,
                summary,
                tokens_before,
                tokens_after,
                usage,
                ..
            } => {
                if let Some(usage) = usage {
                    self.usage.prompt_tokens = self
                        .usage
                        .prompt_tokens
                        .saturating_add(usage.prompt_tokens);
                    self.usage.completion_tokens = self
                        .usage
                        .completion_tokens
                        .saturating_add(usage.completion_tokens);
                    self.usage.total_tokens =
                        self.usage.total_tokens.saturating_add(usage.total_tokens);
                }
                self.context_tokens = tokens_after;
                self.usage_estimated = true;
                self.push_notice(format!(
                    "上下文已压缩（预计 token：{} -> {}）。\n\n{summary}",
                    format_tokens(tokens_before),
                    format_tokens(tokens_after)
                ));
                if reason == CompactionReason::Manual {
                    self.finish_run("就绪");
                } else {
                    self.status = "处理中".to_string();
                }
            }
            AgentEvent::CompactionFailed { reason, message } => {
                if reason == CompactionReason::Manual {
                    self.finish_run("错误");
                    self.push_error(format!("上下文压缩失败：{message}"));
                } else {
                    self.status = "处理中".to_string();
                    self.push_error(format!(
                        "自动压缩失败，已回退为硬裁剪上下文：{message}"
                    ));
                }
            }
            AgentEvent::RunFinished => self.finish_run("就绪"),
            AgentEvent::Cancelled => self.finish_run("已取消"),
            AgentEvent::Error { message } => {
                self.finish_run("错误");
                self.messages.push(ViewMessage {
                    role: ViewRole::Error,
                    title: "错误".to_string(),
                    content: sanitize_terminal_text(&message),
                    reasoning: String::new(),
                    tool_arguments: None,
                    tool_id: None,
                    running: false,
                });
            }
        }
        self.follow_tail = follow_tail;
    }

    fn finish_run(&mut self, status: &str) {
        for message in &mut self.messages {
            message.running = false;
        }
        self.current_assistant = None;
        self.generation_start = None;
        self.running = false;
        self.status = status.to_string();
        self.pending_approval = None;
    }

    fn start_assistant_message(&mut self) -> usize {
        self.messages.push(ViewMessage {
            role: ViewRole::Assistant,
            title: String::new(),
            content: String::new(),
            reasoning: String::new(),
            tool_arguments: None,
            tool_id: None,
            running: true,
        });
        let index = self.messages.len() - 1;
        self.current_assistant = Some(index);
        index
    }

    fn ensure_assistant_message(&mut self) -> usize {
        self.current_assistant
            .unwrap_or_else(|| self.start_assistant_message())
    }

    fn finish_current_assistant_for_tool(&mut self) {
        if let Some(index) = self.current_assistant {
            if self
                .messages
                .get(index)
                .is_some_and(|message| message.content.is_empty() && message.reasoning.is_empty())
            {
                self.messages.remove(index);
            } else if let Some(message) = self.messages.get_mut(index) {
                message.running = false;
            }
        }
        self.current_assistant = None;
    }

    fn scroll_up(&mut self) {
        let amount = self.viewport_height.saturating_sub(2).max(1);
        self.scroll_lines_up(amount);
    }

    fn scroll_down(&mut self) {
        let amount = self.viewport_height.saturating_sub(2).max(1);
        self.scroll_lines_down(amount);
    }

    fn scroll_lines_up(&mut self, amount: usize) {
        self.follow_tail = false;
        self.scroll = self.scroll.saturating_sub(amount.max(1));
    }

    fn scroll_lines_down(&mut self, amount: usize) {
        self.scroll = self
            .scroll
            .saturating_add(amount.max(1))
            .min(self.max_scroll);
        if self.scroll >= self.max_scroll {
            self.follow_tail = true;
        }
    }
}

fn render(frame: &mut Frame<'_>, state: &mut UiState) {
    let area = frame.area();
    if area.width < 24 || area.height < 11 {
        frame.render_widget(
            Paragraph::new("终端窗口过小")
                .style(Style::default().fg(Color::Red))
                .block(Block::default().borders(Borders::ALL)),
            area,
        );
        return;
    }

    let input_height = if state.pending_approval.is_some() {
        APPROVAL_HEIGHT
    } else if state.delete_confirmation != DeleteConfirmation::None {
        DELETE_CONFIRMATION_HEIGHT
    } else {
        let editor_height = state
            .editor
            .rendered_height(area.width.saturating_sub(INPUT_PREFIX_WIDTH))
            .clamp(1, MAX_INPUT_HEIGHT);
        editor_height.saturating_add(u16::from(!state.pending_images.is_empty()))
    };
    let suggestion_count = if state.pending_approval.is_some()
        || state.delete_confirmation != DeleteConfirmation::None
    {
        0
    } else {
        slash_suggestions(state).len()
    };
    let reserved_height = 1_u16
        .saturating_add(3)
        .saturating_add(input_height)
        .saturating_add(1);
    let suggestion_height = u16::try_from(suggestion_count)
        .unwrap_or(u16::MAX)
        .min(MAX_SLASH_SUGGESTIONS)
        .min(area.height.saturating_sub(reserved_height));
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(suggestion_height),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, state, areas[0]);
    render_conversation(frame, state, areas[1]);
    render_slash_suggestions(frame, state, areas[2]);
    render_input(frame, state, areas[3]);
    render_footer(frame, state, areas[4]);
}

fn render_header(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    let spinner = ["-", "\\", "|", "/"][state.spinner_frame % 4];
    let (activity, activity_color) = if state.running {
        (spinner, Color::Yellow)
    } else {
        ("•", Color::Rgb(103, 232, 163))
    };
    let line = Line::from(vec![
        Span::styled(
            " MCode ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Rgb(103, 232, 163))
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(activity, Style::default().fg(activity_color)),
        Span::raw(" "),
        Span::styled(
            state.status.clone(),
            Style::default().fg(if state.running {
                Color::White
            } else {
                Color::Gray
            }),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(Color::Rgb(24, 27, 32))),
        area,
    );
}

fn render_conversation(frame: &mut Frame<'_>, state: &mut UiState, area: Rect) {
    let lines = conversation_lines(state);
    let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    let total_height = paragraph.line_count(area.width);
    state.viewport_height = usize::from(area.height.max(1));
    state.max_scroll = total_height
        .saturating_sub(state.viewport_height)
        .min(usize::from(u16::MAX));
    if state.follow_tail {
        state.scroll = state.max_scroll;
    } else {
        state.scroll = state.scroll.min(state.max_scroll);
    }
    let scroll = u16::try_from(state.scroll).unwrap_or(u16::MAX);
    frame.render_widget(paragraph.scroll((scroll, 0)), area);
}

fn render_slash_suggestions(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    if area.height == 0 {
        return;
    }
    let suggestions = slash_suggestions(state);
    if suggestions.is_empty() {
        return;
    }
    let selected = state
        .slash_selection
        .min(suggestions.len().saturating_sub(1));
    let visible = usize::from(area.height);
    let start = selected
        .saturating_add(1)
        .saturating_sub(visible)
        .min(suggestions.len().saturating_sub(visible));
    let end = start.saturating_add(visible).min(suggestions.len());
    let available = usize::from(area.width.saturating_sub(2));
    let label_column = suggestions[start..end]
        .iter()
        .map(|suggestion| display_width(&suggestion.label))
        .max()
        .unwrap_or(0)
        .min(available.saturating_mul(3) / 5);
    let lines = suggestions
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .map(|(index, suggestion)| {
            let is_selected = index == selected;
            let marker = if is_selected { "› " } else { "  " };
            let label = truncate_width(&suggestion.label, label_column.saturating_add(1));
            let padding = " ".repeat(
                label_column
                    .saturating_sub(display_width(&label))
                    .saturating_add(2),
            );
            let command_style = if is_selected {
                Style::default()
                    .fg(Color::Rgb(126, 200, 255))
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            Line::from(vec![
                Span::styled(
                    marker,
                    Style::default()
                        .fg(Color::Rgb(126, 200, 255))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(label, command_style),
                Span::raw(padding),
                Span::styled(
                    suggestion.description.clone(),
                    Style::default().fg(if is_selected {
                        Color::Gray
                    } else {
                        Color::DarkGray
                    }),
                ),
            ])
            .style(if is_selected {
                Style::default().bg(Color::Rgb(31, 35, 41))
            } else {
                Style::default()
            })
        })
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(lines), area);
}

fn render_input(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    if let Some(approval) = &state.pending_approval {
        let arguments = approval.arguments.replace(['\r', '\n'], " ");
        let argument_prefix = if approval.name == "shell" {
            "$ "
        } else {
            "↳ "
        };
        let details = truncate_width(
            &format!("{argument_prefix}{arguments}"),
            usize::from(area.width.saturating_sub(2)),
        );
        let lines = vec![
            Line::from(Span::styled(
                format!("是否允许运行 {}？", approval.name),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(details, Style::default().fg(Color::Gray))),
            approval_option_line(
                ApprovalChoice::ApproveOnce,
                approval.selection,
                "1. 允许一次",
            ),
            approval_option_line(
                ApprovalChoice::ApproveForSession,
                approval.selection,
                "2. 本次会话内始终允许",
            ),
            approval_option_line(ApprovalChoice::Deny, approval.selection, "3. 拒绝"),
            Line::from(Span::styled(
                "  ↑/↓ 选择 · Enter 确认 · Esc 取消任务",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        frame.render_widget(
            Paragraph::new(lines).style(Style::default().bg(Color::Rgb(24, 27, 32))),
            area,
        );
        return;
    }

    if let DeleteConfirmation::Selecting(selection) = state.delete_confirmation {
        let yes_style = if selection == DeleteChoice::Yes {
            Style::default()
                .fg(Color::Red)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let no_style = if selection == DeleteChoice::No {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let yes_marker = if selection == DeleteChoice::Yes {
            ">"
        } else {
            " "
        };
        let no_marker = if selection == DeleteChoice::No {
            ">"
        } else {
            " "
        };
        let lines = vec![
            Line::from(Span::styled(
                "删除当前对话？此操作无法撤销。",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("{yes_marker} Yes  删除并退出"),
                yes_style,
            )),
            Line::from(Span::styled(format!("{no_marker} No   返回"), no_style)),
            Line::from(Span::styled(
                "使用方向键选择，按 Enter 确认",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), area);
        return;
    }

    let mut editor_area = area;
    if !state.pending_images.is_empty() {
        let attachment_area = Rect::new(area.x, area.y, area.width, 1.min(area.height));
        render_pending_images(frame, state, attachment_area);
        editor_area = Rect::new(
            area.x,
            area.y.saturating_add(1),
            area.width,
            area.height.saturating_sub(1),
        );
    }

    let prompt_color = if state.running {
        Color::Yellow
    } else {
        Color::Rgb(126, 200, 255)
    };
    let prefix_width = INPUT_PREFIX_WIDTH.min(editor_area.width);
    let prefix_area = Rect::new(
        editor_area.x,
        editor_area.y,
        prefix_width,
        1.min(editor_area.height),
    );
    frame.render_widget(
        Paragraph::new(Span::styled(
            "> ",
            Style::default()
                .fg(prompt_color)
                .add_modifier(Modifier::BOLD),
        )),
        prefix_area,
    );
    let input_area = Rect::new(
        editor_area.x.saturating_add(prefix_width),
        editor_area.y,
        editor_area.width.saturating_sub(prefix_width),
        editor_area.height,
    );
    let (cursor_x, cursor_y, scroll) = state
        .editor
        .cursor_layout(input_area.width, input_area.height);
    frame.render_widget(
        Paragraph::new(state.editor.text())
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        input_area,
    );
    frame.set_cursor_position(Position::new(
        input_area
            .x
            .saturating_add(cursor_x.min(input_area.width.saturating_sub(1))),
        input_area
            .y
            .saturating_add(cursor_y.min(input_area.height.saturating_sub(1))),
    ));
}

fn render_pending_images(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    let names = state
        .pending_images
        .iter()
        .take(3)
        .map(|image| sanitize_terminal_text(&image.name))
        .collect::<Vec<_>>()
        .join("、");
    let hidden = state.pending_images.len().saturating_sub(3);
    let suffix = if hidden == 0 {
        names
    } else {
        format!("{names} 等 {} 张", state.pending_images.len())
    };
    let label = format!("图片 {}  ", state.pending_images.len());
    let available = usize::from(area.width).saturating_sub(display_width(&label) + 4);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "  + ",
                Style::default()
                    .fg(Color::Rgb(103, 232, 163))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                label,
                Style::default()
                    .fg(Color::Rgb(103, 232, 163))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                truncate_width(&suffix, available.saturating_add(1)),
                Style::default().fg(Color::Gray),
            ),
        ])),
        area,
    );
}

fn render_footer(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    frame.render_widget(
        Paragraph::new(footer_line(state, usize::from(area.width)))
            .style(Style::default().bg(Color::Rgb(24, 27, 32))),
        area,
    );
}

fn approval_option_line(
    choice: ApprovalChoice,
    selected: ApprovalChoice,
    label: &'static str,
) -> Line<'static> {
    let is_selected = choice == selected;
    let style = if is_selected {
        Style::default()
            .fg(Color::Rgb(126, 200, 255))
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    };
    Line::from(vec![
        Span::styled(
            if is_selected { "› " } else { "  " },
            Style::default()
                .fg(Color::Rgb(126, 200, 255))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(label, style),
    ])
}

fn footer_line(state: &UiState, width: usize) -> Line<'static> {
    let estimate = if state.usage_estimated { "~" } else { "" };
    let context_full = format!(
        "{estimate}{}/{} ({}%)",
        format_tokens(state.context_tokens),
        format_tokens(state.max_input_tokens),
        format_context_percent(state.context_tokens, state.max_input_tokens)
    );
    let context_compact = format!(
        "{estimate}{}%",
        format_context_percent(state.context_tokens, state.max_input_tokens)
    );
    let usage = format!(
        " | 输入 {} 输出 {}",
        format_tokens(state.usage.prompt_tokens),
        format_tokens(state.usage.completion_tokens)
    );
    let effort = state.reasoning_effort.to_string();
    let context_label = " 上下文 ";
    let full_right_width = display_width(&state.model)
        .saturating_add(display_width(" | effort "))
        .saturating_add(display_width(&effort))
        .saturating_add(1);
    let model_right_width = display_width(&state.model).saturating_add(1);
    let full_left_width = display_width(context_label)
        .saturating_add(display_width(&context_full))
        .saturating_add(display_width(&usage));
    let context_left_width =
        display_width(context_label).saturating_add(display_width(&context_full));
    let compact_left_width =
        display_width(context_label).saturating_add(display_width(&context_compact));

    let (context, usage, show_effort, model) = if full_left_width
        .saturating_add(full_right_width)
        .saturating_add(2)
        <= width
    {
        (context_full, Some(usage), true, state.model.clone())
    } else if context_left_width
        .saturating_add(full_right_width)
        .saturating_add(2)
        <= width
    {
        (context_full, None, true, state.model.clone())
    } else if compact_left_width
        .saturating_add(full_right_width)
        .saturating_add(2)
        <= width
    {
        (context_compact, None, true, state.model.clone())
    } else {
        let available = width.saturating_sub(compact_left_width).saturating_sub(3);
        (
            context_compact,
            None,
            false,
            truncate_width(&state.model, available.saturating_add(1)),
        )
    };

    let left_width = display_width(context_label)
        .saturating_add(display_width(&context))
        .saturating_add(usage.as_deref().map_or(0, display_width));
    let right_width = if model.is_empty() {
        0
    } else if show_effort {
        full_right_width
    } else {
        model_right_width.min(display_width(&model).saturating_add(1))
    };
    let padding = width.saturating_sub(left_width.saturating_add(right_width));
    let mut spans = vec![
        Span::styled(
            context_label,
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            context,
            Style::default()
                .fg(context_usage_color(
                    state.context_tokens,
                    state.max_input_tokens,
                ))
                .add_modifier(Modifier::BOLD),
        ),
    ];
    if let Some(usage) = usage {
        spans.push(Span::styled(usage, Style::default().fg(Color::DarkGray)));
    }
    spans.push(Span::raw(" ".repeat(padding)));
    if !model.is_empty() {
        spans.push(Span::styled(
            model,
            Style::default()
                .fg(Color::Rgb(126, 200, 255))
                .add_modifier(Modifier::BOLD),
        ));
        if show_effort {
            spans.push(Span::styled(
                " | effort ",
                Style::default().fg(Color::DarkGray),
            ));
            spans.push(Span::styled(
                effort,
                Style::default()
                    .fg(Color::Rgb(245, 190, 78))
                    .add_modifier(Modifier::BOLD),
            ));
        }
        spans.push(Span::raw(" "));
    }
    Line::from(spans)
}

fn context_usage_color(tokens: u64, limit: u64) -> Color {
    if limit == 0 {
        return Color::Gray;
    }
    let percent = u128::from(tokens).saturating_mul(100) / u128::from(limit);
    if percent >= 90 {
        Color::LightRed
    } else if percent >= 70 {
        Color::Yellow
    } else {
        Color::Rgb(103, 232, 163)
    }
}

fn format_tokens(count: u64) -> String {
    if count < 1_000 {
        return count.to_string();
    }
    if count < 10_000 {
        return format!("{}.{:01}k", count / 1_000, (count % 1_000) / 100);
    }
    if count < 1_000_000 {
        return format!("{}k", count.saturating_add(500) / 1_000);
    }
    if count < 10_000_000 {
        return format!(
            "{}.{:01}M",
            count / 1_000_000,
            (count % 1_000_000) / 100_000
        );
    }
    format!("{}M", count.saturating_add(500_000) / 1_000_000)
}

fn format_user_content(mut content: String, images: &[ImageAttachment]) -> String {
    for image in images {
        if !content.is_empty() {
            content.push('\n');
        }
        content.push_str("[图片：");
        content.push_str(&image.name);
        content.push(']');
    }
    content
}

fn format_context_percent(tokens: u64, window: u64) -> String {
    if window == 0 {
        return "0.0".to_string();
    }
    let tenths = tokens.saturating_mul(1_000) / window;
    format!("{}.{:01}", tenths / 10, tenths % 10)
}

fn display_width(text: &str) -> usize {
    text.chars()
        .map(|character| character.width().unwrap_or(0))
        .sum()
}

fn conversation_lines(state: &UiState) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for message in &state.messages {
        if message.role == ViewRole::Assistant {
            if !message.reasoning.is_empty() {
                let reasoning_in_progress = message.running && message.content.is_empty();
                let show_reasoning = state.thinking_display == ThinkingDisplay::Shown;
                if show_reasoning {
                    let running = if reasoning_in_progress {
                        "（思考中）"
                    } else {
                        ""
                    };
                    lines.push(Line::from(vec![
                        Span::styled(
                            "• 思考",
                            Style::default()
                                .fg(Color::DarkGray)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(running, Style::default().fg(Color::DarkGray)),
                    ]));
                    for line in message.reasoning.lines() {
                        lines.push(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(
                                line.to_string(),
                                Style::default()
                                    .fg(Color::DarkGray)
                                    .add_modifier(Modifier::ITALIC),
                            ),
                        ]));
                    }
                } else {
                    let line_count = message
                        .reasoning
                        .bytes()
                        .filter(|byte| byte == &b'\n')
                        .count()
                        + 1;
                    let character_count = message.reasoning.chars().count();
                    lines.push(Line::from(vec![
                        Span::styled(
                            "• 思考",
                            Style::default()
                                .fg(Color::DarkGray)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("（已隐藏，{line_count} 行，{character_count} 个字符）"),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]));
                }
                if !message.content.is_empty() {
                    lines.push(Line::default());
                }
            }
            if !message.content.is_empty() || message.reasoning.is_empty() {
                if message.content.is_empty() && message.running {
                    lines.push(Line::from(Span::styled(
                        "• 正在回复...",
                        Style::default().fg(Color::DarkGray),
                    )));
                } else {
                    append_markdown_lines(
                        &mut lines,
                        &message.content,
                        Style::default().fg(Color::White),
                    );
                }
            }
            lines.push(Line::default());
            continue;
        }

        if matches!(message.role, ViewRole::Tool | ViewRole::Error) && message.tool_id.is_some() {
            append_tool_message(&mut lines, message);
            lines.push(Line::default());
            continue;
        }

        if message.role == ViewRole::User {
            append_user_message(&mut lines, &message.content);
            lines.push(Line::default());
            continue;
        }

        let (label_color, content_style) = match message.role {
            ViewRole::Tool => (Color::Rgb(245, 190, 78), Style::default().fg(Color::Gray)),
            ViewRole::Notice => (Color::Cyan, Style::default().fg(Color::Gray)),
            ViewRole::Error => (Color::Red, Style::default().fg(Color::LightRed)),
            ViewRole::User | ViewRole::Assistant => unreachable!(),
        };
        let running = if message.running { "  运行中" } else { "" };
        lines.push(Line::from(vec![
            Span::styled(
                message.title.clone(),
                Style::default()
                    .fg(label_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(running, Style::default().fg(Color::DarkGray)),
        ]));
        append_markdown_lines(&mut lines, &message.content, content_style);
        lines.push(Line::default());
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "就绪。",
            Style::default().fg(Color::DarkGray),
        )));
    }
    lines
}

fn append_user_message(lines: &mut Vec<Line<'static>>, content: &str) {
    let mut content_lines = content.lines();
    let first = content_lines.next().unwrap_or_default();
    lines.push(Line::from(vec![
        Span::styled(
            "› ",
            Style::default()
                .fg(Color::Rgb(126, 200, 255))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(first.to_string(), Style::default().fg(Color::White)),
    ]));
    for line in content_lines {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(line.to_string(), Style::default().fg(Color::White)),
        ]));
    }
}

fn append_tool_message(lines: &mut Vec<Line<'static>>, message: &ViewMessage) {
    let failed = message.role == ViewRole::Error;
    let color = if failed {
        Color::LightRed
    } else {
        Color::Rgb(245, 190, 78)
    };
    let status = if message.running {
        "（运行中）"
    } else {
        ""
    };
    lines.push(Line::from(vec![
        Span::styled(
            "• ",
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            message.title.clone(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(status, Style::default().fg(Color::DarkGray)),
    ]));

    if let Some(arguments) = message
        .tool_arguments
        .as_deref()
        .filter(|arguments| !arguments.is_empty())
    {
        let first_prefix = if message.title == "shell" {
            "  $ "
        } else {
            "  ↳ "
        };
        append_collapsed_plain_lines(
            lines,
            arguments,
            TOOL_ARGUMENT_PREVIEW_LINES,
            first_prefix,
            "    ",
            Style::default().fg(Color::Gray),
        );
    }
    if !message.content.is_empty() {
        append_collapsed_plain_lines(
            lines,
            &message.content,
            TOOL_OUTPUT_PREVIEW_LINES,
            "  └ ",
            "    ",
            if failed {
                Style::default().fg(Color::LightRed)
            } else {
                Style::default().fg(Color::DarkGray)
            },
        );
    }
}

fn append_collapsed_plain_lines(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    limit: usize,
    first_prefix: &str,
    continuation_prefix: &str,
    style: Style,
) {
    let content_lines = content.lines().collect::<Vec<_>>();
    for (index, line) in content_lines.iter().take(limit).enumerate() {
        let prefix = if index == 0 {
            first_prefix
        } else {
            continuation_prefix
        };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::DarkGray)),
            Span::styled(truncate_tool_preview_line(line), style),
        ]));
    }
    let hidden = content_lines.len().saturating_sub(limit);
    if hidden > 0 {
        lines.push(Line::from(vec![
            Span::styled(
                continuation_prefix.to_string(),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                format!("… 已折叠 {hidden} 行"),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ),
        ]));
    }
}

fn truncate_tool_preview_line(line: &str) -> String {
    let mut characters = line.chars();
    let preview = characters
        .by_ref()
        .take(TOOL_PREVIEW_LINE_CHARS)
        .collect::<String>();
    if characters.next().is_some() {
        format!("{preview}…")
    } else {
        preview
    }
}

fn format_tool_input(name: &str, arguments: &str) -> String {
    let parsed = serde_json::from_str::<serde_json::Value>(arguments).ok();
    let concise = parsed.as_ref().and_then(|value| match name {
        "shell" => value.get("command")?.as_str().map(ToString::to_string),
        "read_file" | "write_file" | "edit_file" => {
            value.get("path")?.as_str().map(ToString::to_string)
        }
        "web_search" => value
            .get("query")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string),
        "fetch_content" => value
            .get("url")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string),
        _ => None,
    });
    let formatted = concise.unwrap_or_else(|| format_tool_arguments(arguments));
    truncate_for_ui(&sanitize_terminal_text(&formatted))
}

fn append_markdown_lines(lines: &mut Vec<Line<'static>>, content: &str, base: Style) {
    lines.extend(MarkdownRenderer::new(base).render(content));
}

struct MarkdownList {
    next: Option<u64>,
}

struct MarkdownRenderer {
    lines: Vec<Line<'static>>,
    current: Vec<Span<'static>>,
    base: Style,
    heading: Option<HeadingLevel>,
    strong_depth: usize,
    emphasis_depth: usize,
    strikethrough_depth: usize,
    code_block: bool,
    quote_depth: usize,
    lists: Vec<MarkdownList>,
    item_continuations: Vec<String>,
    pending_item_prefix: Option<String>,
    links: Vec<String>,
}

impl MarkdownRenderer {
    fn new(base: Style) -> Self {
        Self {
            lines: Vec::new(),
            current: Vec::new(),
            base,
            heading: None,
            strong_depth: 0,
            emphasis_depth: 0,
            strikethrough_depth: 0,
            code_block: false,
            quote_depth: 0,
            lists: Vec::new(),
            item_continuations: Vec::new(),
            pending_item_prefix: None,
            links: Vec::new(),
        }
    }

    fn render(mut self, content: &str) -> Vec<Line<'static>> {
        for event in Parser::new_ext(content, MarkdownOptions::all()) {
            self.event(event);
        }
        self.flush_line(false);
        if self.lines.is_empty() {
            self.lines.push(Line::default());
        }
        self.lines
    }

    fn event(&mut self, event: MarkdownEvent<'_>) {
        match event {
            MarkdownEvent::Start(tag) => self.start_tag(tag),
            MarkdownEvent::End(tag) => self.end_tag(tag),
            MarkdownEvent::Text(text)
            | MarkdownEvent::Html(text)
            | MarkdownEvent::InlineHtml(text) => {
                self.push_text(&text, self.current_style());
            }
            MarkdownEvent::Code(code) => self.push_text(
                &code,
                Style::default()
                    .fg(Color::Rgb(215, 220, 230))
                    .bg(Color::Rgb(31, 35, 41)),
            ),
            MarkdownEvent::InlineMath(math) => {
                self.push_text(&format!("${math}$"), self.current_style());
            }
            MarkdownEvent::DisplayMath(math) => {
                self.flush_line(false);
                self.push_text(&format!("$${math}$$"), self.current_style());
                self.flush_line(false);
            }
            MarkdownEvent::FootnoteReference(label) => {
                self.push_text(&format!("[^{label}]"), self.current_style());
            }
            MarkdownEvent::SoftBreak | MarkdownEvent::HardBreak => self.flush_line(false),
            MarkdownEvent::Rule => {
                self.flush_line(false);
                self.push_text(
                    "────────────────────────",
                    Style::default().fg(Color::DarkGray),
                );
                self.flush_line(false);
            }
            MarkdownEvent::TaskListMarker(checked) => {
                self.push_text(if checked { "[x] " } else { "[ ] " }, self.current_style());
            }
        }
    }

    fn start_tag(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph | Tag::TableRow => self.flush_line(false),
            Tag::Heading { level, .. } => {
                self.flush_line(false);
                self.heading = Some(level);
            }
            Tag::BlockQuote(_) => {
                self.flush_line(false);
                self.quote_depth = self.quote_depth.saturating_add(1);
            }
            Tag::CodeBlock(_) => {
                self.flush_line(false);
                self.code_block = true;
            }
            Tag::List(start) => {
                self.flush_line(false);
                self.lists.push(MarkdownList { next: start });
            }
            Tag::Item => {
                self.flush_line(false);
                let depth = self.lists.len().saturating_sub(1);
                let marker = self.lists.last_mut().map_or_else(
                    || "- ".to_string(),
                    |list| match list.next.as_mut() {
                        Some(next) => {
                            let marker = format!("{next}. ");
                            *next = next.saturating_add(1);
                            marker
                        }
                        None => "- ".to_string(),
                    },
                );
                let indentation = "  ".repeat(depth);
                self.pending_item_prefix = Some(format!("{indentation}{marker}"));
                self.item_continuations.push(format!(
                    "{indentation}{}",
                    " ".repeat(display_width(&marker))
                ));
            }
            Tag::FootnoteDefinition(label) => {
                self.flush_line(false);
                self.pending_item_prefix = Some(format!("[^{label}] "));
            }
            Tag::DefinitionListTitle | Tag::Strong => {
                self.strong_depth = self.strong_depth.saturating_add(1);
            }
            Tag::DefinitionListDefinition => {
                self.flush_line(false);
                self.pending_item_prefix = Some(": ".to_string());
            }
            Tag::TableCell => {
                if !self.current.is_empty() {
                    self.push_text(" | ", Style::default().fg(Color::DarkGray));
                }
            }
            Tag::Emphasis => self.emphasis_depth = self.emphasis_depth.saturating_add(1),
            Tag::Strikethrough => {
                self.strikethrough_depth = self.strikethrough_depth.saturating_add(1);
            }
            Tag::Link { dest_url, .. } => self.links.push(dest_url.into_string()),
            Tag::Image { dest_url, .. } => {
                self.push_text("图片：", Style::default().fg(Color::DarkGray));
                self.links.push(dest_url.into_string());
            }
            Tag::HtmlBlock
            | Tag::Table(_)
            | Tag::TableHead
            | Tag::DefinitionList
            | Tag::Superscript
            | Tag::Subscript
            | Tag::MetadataBlock(_) => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph | TagEnd::Heading(_) | TagEnd::TableRow => {
                self.flush_line(false);
                if matches!(tag, TagEnd::Heading(_)) {
                    self.heading = None;
                }
            }
            TagEnd::BlockQuote(_) => {
                self.flush_line(false);
                self.quote_depth = self.quote_depth.saturating_sub(1);
            }
            TagEnd::CodeBlock => {
                self.flush_line(false);
                self.code_block = false;
            }
            TagEnd::List(_) => {
                self.flush_line(false);
                self.lists.pop();
            }
            TagEnd::Item => {
                self.flush_line(false);
                self.pending_item_prefix = None;
                self.item_continuations.pop();
            }
            TagEnd::FootnoteDefinition | TagEnd::DefinitionListDefinition => {
                self.flush_line(false);
                self.pending_item_prefix = None;
            }
            TagEnd::DefinitionListTitle => {
                self.strong_depth = self.strong_depth.saturating_sub(1);
                self.flush_line(false);
            }
            TagEnd::Emphasis => self.emphasis_depth = self.emphasis_depth.saturating_sub(1),
            TagEnd::Strong => self.strong_depth = self.strong_depth.saturating_sub(1),
            TagEnd::Strikethrough => {
                self.strikethrough_depth = self.strikethrough_depth.saturating_sub(1);
            }
            TagEnd::Link | TagEnd::Image => {
                if let Some(destination) = self.links.pop()
                    && !destination.is_empty()
                {
                    self.push_text(
                        &format!(" ({destination})"),
                        Style::default()
                            .fg(Color::Rgb(126, 200, 255))
                            .add_modifier(Modifier::UNDERLINED),
                    );
                }
            }
            TagEnd::TableCell
            | TagEnd::HtmlBlock
            | TagEnd::Table
            | TagEnd::TableHead
            | TagEnd::DefinitionList
            | TagEnd::Superscript
            | TagEnd::Subscript
            | TagEnd::MetadataBlock(_) => {}
        }
    }

    fn current_style(&self) -> Style {
        let mut style = if self.code_block {
            Style::default()
                .fg(Color::Rgb(215, 220, 230))
                .bg(Color::Rgb(31, 35, 41))
        } else {
            self.base
        };
        if let Some(level) = self.heading {
            style = style
                .fg(match level {
                    HeadingLevel::H1 => Color::Rgb(103, 232, 163),
                    HeadingLevel::H2 => Color::Rgb(126, 200, 255),
                    _ => Color::White,
                })
                .add_modifier(Modifier::BOLD);
        }
        if self.strong_depth > 0 {
            style = style.add_modifier(Modifier::BOLD);
        }
        if self.emphasis_depth > 0 {
            style = style.add_modifier(Modifier::ITALIC);
        }
        if self.strikethrough_depth > 0 {
            style = style.add_modifier(Modifier::CROSSED_OUT);
        }
        if !self.links.is_empty() {
            style = style
                .fg(Color::Rgb(126, 200, 255))
                .add_modifier(Modifier::UNDERLINED);
        }
        style
    }

    fn push_text(&mut self, text: &str, style: Style) {
        for (index, part) in text.split('\n').enumerate() {
            if index > 0 {
                self.flush_line(true);
            }
            if !part.is_empty() {
                self.ensure_prefix();
                self.current.push(Span::styled(part.to_string(), style));
            }
        }
    }

    fn ensure_prefix(&mut self) {
        if !self.current.is_empty() {
            return;
        }
        if self.quote_depth > 0 {
            self.current.push(Span::styled(
                "│ ".repeat(self.quote_depth),
                Style::default().fg(Color::DarkGray),
            ));
        }
        let prefix = self
            .pending_item_prefix
            .take()
            .or_else(|| self.item_continuations.last().cloned());
        if let Some(prefix) = prefix {
            self.current
                .push(Span::styled(prefix, Style::default().fg(Color::DarkGray)));
        }
    }

    fn flush_line(&mut self, force: bool) {
        if self.current.is_empty() && force {
            self.ensure_prefix();
        }
        if !self.current.is_empty() {
            self.lines
                .push(Line::from(std::mem::take(&mut self.current)));
        }
    }
}

fn truncate_for_ui(text: &str) -> String {
    const LIMIT: usize = 4_000;
    let mut chars = text.chars();
    let prefix: String = chars.by_ref().take(LIMIT).collect();
    if chars.next().is_some() {
        format!("{prefix}\n... 界面中的输出已截短；完整结果已发送给模型")
    } else {
        prefix
    }
}

fn truncate_width(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut output = String::new();
    let mut used = 0usize;
    for character in text.chars() {
        let character_width = character.width().unwrap_or(0);
        if used + character_width >= width {
            break;
        }
        output.push(character);
        used += character_width;
    }
    output
}

struct ScreenGuard;

impl ScreenGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("启用终端原始模式失败")?;
        execute!(
            io::stdout(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture,
            Hide
        )
        .context("进入终端备用屏幕失败")?;
        Ok(Self)
    }
}

impl Drop for ScreenGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            Show,
            DisableMouseCapture,
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
    }
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;

    use super::*;

    fn rendered_terminal(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    #[test]
    fn editor_handles_unicode_and_lines() {
        let mut editor = Editor::default();
        editor.insert_str("ab");
        editor.insert('中');
        editor.insert('\n');
        editor.insert('z');
        assert_eq!(editor.text(), "ab中\nz");
        editor.move_left();
        editor.backspace();
        assert_eq!(editor.text(), "ab中z");
    }

    #[test]
    fn editor_collapses_large_pastes_and_restores_them_on_submit() {
        let pasted = "x".repeat(COLLAPSED_PASTE_CHAR_THRESHOLD);
        let mut editor = Editor::default();
        editor.insert_str("before ");
        editor.insert_paste(&pasted);
        editor.insert_str(" after");

        assert_eq!(
            editor.text(),
            format!("before [已粘贴 {COLLAPSED_PASTE_CHAR_THRESHOLD} 个字符] after")
        );
        assert_eq!(editor.take(), format!("before {pasted} after"));
    }

    #[test]
    fn editor_keeps_short_pastes_editable_and_deletes_collapsed_pastes_as_one_item() {
        let mut editor = Editor::default();
        editor.insert_paste("short\npaste");
        assert_eq!(editor.text(), "short\npaste");

        let pasted = (0..COLLAPSED_PASTE_LINE_THRESHOLD)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        editor.insert_paste(&pasted);
        assert!(editor.text().contains("[已粘贴 "));
        editor.backspace();
        assert_eq!(editor.text(), "short\npaste");
    }

    #[test]
    fn submitting_a_collapsed_paste_sends_its_full_contents() {
        let pasted = "full pasted content\n".repeat(COLLAPSED_PASTE_LINE_THRESHOLD);
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.editor.insert_paste(&pasted);
        assert!(state.editor.text().starts_with("[已粘贴 "));

        let action = handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert!(matches!(
            action,
            UiAction::Submit { prompt, images } if prompt == pasted && images.is_empty()
        ));
    }

    #[test]
    fn ctrl_v_requests_a_wayland_clipboard_paste() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );

        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL),
                &mut state,
                None,
            ),
            UiAction::PasteClipboard
        ));
    }

    #[test]
    fn pasted_image_paths_become_removable_attachments() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join("pixel.png"),
            [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0],
        )
        .unwrap();
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            temp.path().to_path_buf(),
        );

        paste_text_or_image(&mut state, "'pixel.png'");
        assert!(state.editor.is_empty());
        assert_eq!(state.pending_images.len(), 1);
        assert_eq!(state.pending_images[0].name, "pixel.png");

        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let rendered = rendered_terminal(&terminal);
        assert!(rendered.replace(' ', "").contains("图片1"));
        assert!(rendered.contains("pixel.png"));

        handle_key(
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert!(state.pending_images.is_empty());
        assert_eq!(state.status, "就绪");
    }

    #[test]
    fn clipboard_reads_are_bounded() {
        let exact = read_limited(io::Cursor::new(vec![1; 4]), 4).unwrap();
        assert_eq!(exact, vec![1; 4]);
        assert!(read_limited(io::Cursor::new(vec![1; 5]), 4).is_err());
    }

    #[test]
    fn renders_collapsed_paste_without_exposing_its_contents() {
        let pasted = format!(
            "private-start{}",
            "x".repeat(COLLAPSED_PASTE_CHAR_THRESHOLD)
        );
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.editor.insert_paste(&pasted);
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        assert!(rendered.replace(' ', "").contains("[已粘贴"));
        assert!(!rendered.contains("private-start"));
    }

    #[test]
    fn renders_commonmark_as_terminal_styles() {
        let content = "# Heading\n\n**bold** and *italic* with `code` and [docs](https://example.com).\n\n- one\n- two\n\n> quote\n\n```rust\nlet value = 1;\n```";
        let mut lines = Vec::new();
        append_markdown_lines(&mut lines, content, Style::default().fg(Color::White));
        let rendered = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("Heading"));
        assert!(rendered.contains("- one"));
        assert!(rendered.contains("│ quote"));
        assert!(rendered.contains("docs (https://example.com)"));
        assert!(rendered.contains("let value = 1;"));
        assert!(!rendered.contains("# Heading"));
        assert!(!rendered.contains("**bold**"));
        assert!(!rendered.contains("```"));

        let spans = lines.iter().flat_map(|line| line.spans.iter());
        let bold = spans
            .clone()
            .find(|span| span.content.as_ref() == "bold")
            .unwrap();
        assert!(bold.style.add_modifier.contains(Modifier::BOLD));
        let italic = spans
            .clone()
            .find(|span| span.content.as_ref() == "italic")
            .unwrap();
        assert!(italic.style.add_modifier.contains(Modifier::ITALIC));
        let code = spans
            .clone()
            .find(|span| span.content.as_ref() == "code")
            .unwrap();
        assert_eq!(code.style.bg, Some(Color::Rgb(31, 35, 41)));
    }

    #[test]
    fn folds_tool_arguments_and_output() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.apply_agent_event(AgentEvent::ToolStarted {
            id: "call_shell".to_string(),
            name: "shell".to_string(),
            arguments: serde_json::json!({
                "command": "first\nsecond\nthird",
                "timeout_seconds": 30
            })
            .to_string(),
        });
        state.apply_agent_event(AgentEvent::ToolFinished {
            id: "call_shell".to_string(),
            name: "shell".to_string(),
            output: (0..12)
                .map(|index| format!("output {index}"))
                .collect::<Vec<_>>()
                .join("\n"),
            is_error: false,
        });

        let rendered = conversation_lines(&state)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("• shell"));
        assert!(rendered.contains("$ first"));
        assert!(rendered.contains("second"));
        assert!(!rendered.contains("third"));
        assert!(rendered.contains("output 0"));
        assert!(rendered.contains("output 4"));
        assert!(!rendered.contains("output 5"));
        assert!(!rendered.contains("output 11"));
        assert!(rendered.contains("已折叠 1 行"));
        assert!(rendered.contains("已折叠 7 行"));
    }

    #[test]
    fn rebuilds_folded_tool_views_from_session_history() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.push_history(ChatMessage::assistant(
            None,
            None,
            vec![crate::protocol::ToolCall {
                id: "call_read".to_string(),
                kind: "function".to_string(),
                function: crate::protocol::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({
                        "path": "README.md",
                        "offset": 1,
                        "limit": 20
                    })
                    .to_string(),
                },
            }],
        ));
        state.push_history(ChatMessage::tool(
            "call_read",
            (0..8)
                .map(|index| format!("line {index}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ));

        let rendered = conversation_lines(&state)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("• read_file"));
        assert!(rendered.contains("README.md"));
        assert!(rendered.contains("line 0"));
        assert!(!rendered.contains("line 7"));
        assert!(rendered.contains("已折叠 3 行"));
    }

    #[test]
    fn footer_truncation_respects_character_width() {
        assert_eq!(truncate_width("ab中文", 5), "ab中");
    }

    #[test]
    fn renders_conversation_input_and_status_without_overlap() {
        let mut state = UiState::new(
            "test-model".to_string(),
            "http://localhost:8000/v1/chat/completions".to_string(),
            std::path::PathBuf::from("/tmp/project"),
        );
        state.push_user("Please inspect the project".to_string(), &[]);
        state.apply_agent_event(AgentEvent::AssistantStarted);
        state.apply_agent_event(AgentEvent::TextDelta {
            text: "I found the relevant module.".to_string(),
        });
        state.apply_agent_event(AgentEvent::WebSearchStarted {
            id: "ws_1".to_string(),
        });
        state.apply_agent_event(AgentEvent::WebSearchFinished {
            id: "ws_1".to_string(),
            action: crate::protocol::WebSearchAction::Search {
                query: Some("current release".to_string()),
                queries: Vec::new(),
            },
        });
        state.apply_agent_event(AgentEvent::TextDelta {
            text: "The current release is available.".to_string(),
        });

        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let mut rendered = String::new();
        for cell in &terminal.backend().buffer().content {
            rendered.push_str(cell.symbol());
        }

        assert!(rendered.contains("MCode"));
        assert!(rendered.contains("test-model"));
        assert!(rendered.contains("Please inspect the project"));
        assert!(rendered.contains("I found the relevant module."));
        assert!(rendered.replace(' ', "").contains("网页搜索"));
        assert!(rendered.contains("current release"));
        assert!(rendered.contains("The current release is available."));
        assert!(!rendered.contains("你"));
        assert!(!rendered.contains("助手"));
        assert!(rendered.contains('>'));
        assert!(!rendered.contains("prompt"));
        assert!(!rendered.contains('┌'));
        assert!(!rendered.contains("/tmp/project"));
        assert!(rendered.contains("effort off"));
        let compact = rendered.replace(' ', "");
        assert!(compact.contains("输入0输出0"));
        assert!(compact.contains("上下文0/128k"));
    }

    #[test]
    fn renders_small_supported_terminal() {
        let mut state = UiState::new(
            "m".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        let backend = TestBackend::new(24, 11);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        assert!(!rendered.contains("终端窗口过小"));
    }

    #[test]
    fn hides_active_and_completed_thinking_by_default() {
        let mut state = UiState::new(
            "reasoning-model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("/tmp/project"),
        );
        state.reasoning_effort = ReasoningEffort::High;
        state.apply_agent_event(AgentEvent::AssistantStarted);
        state.apply_agent_event(AgentEvent::ReasoningDelta {
            text: "Inspecting the request.\nChecking the implementation.".to_string(),
        });

        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let rendered = rendered_terminal(&terminal);
        assert!(rendered.replace(' ', "").contains("思考（已隐藏，2行"));
        assert!(!rendered.contains("Inspecting the request."));
        assert!(!rendered.contains("Checking the implementation."));

        state.apply_agent_event(AgentEvent::TextDelta {
            text: "Final response.".to_string(),
        });

        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let rendered = rendered_terminal(&terminal);
        let compact = rendered.replace(' ', "");
        assert!(compact.contains("思考"));
        assert!(compact.contains("已隐藏，2行"));
        assert!(!rendered.contains("Inspecting the request."));
        assert!(!compact.contains("助手"));
        assert!(rendered.contains("Final response."));
        assert!(rendered.contains("effort high"));
    }

    #[test]
    fn thinking_command_shows_and_hides_reasoning() {
        let mut state = UiState::new(
            "reasoning-model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.messages.push(ViewMessage {
            role: ViewRole::Assistant,
            title: "assistant".to_string(),
            content: "Done.".to_string(),
            reasoning: "Hidden reasoning.".to_string(),
            tool_arguments: None,
            tool_id: None,
            running: false,
        });

        let lines = conversation_lines(&state);
        let rendered = lines.iter().map(Line::to_string).collect::<String>();
        assert!(rendered.contains("思考（已隐藏"));
        assert!(!rendered.contains("Hidden reasoning."));

        state.editor.insert_str("/thinking show");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        assert_eq!(state.thinking_display, ThinkingDisplay::Shown);
        let lines = conversation_lines(&state);
        let rendered = lines.iter().map(Line::to_string).collect::<String>();
        assert!(rendered.contains("Hidden reasoning."));

        state.editor.insert_str("/thinking hide");
        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert_eq!(state.thinking_display, ThinkingDisplay::Hidden);
        let lines = conversation_lines(&state);
        let rendered = lines.iter().map(Line::to_string).collect::<String>();
        assert!(rendered.contains("已隐藏"));
        assert!(!rendered.contains("Hidden reasoning."));
    }

    #[test]
    fn renders_and_filters_slash_command_suggestions() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.editor.insert('/');
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        let compact = rendered.replace(' ', "");
        assert!(compact.contains("/model"));
        assert!(compact.contains("查看或切换模型"));
        assert!(compact.contains("/effort"));
        assert!(rendered.contains("/thinking"));
        assert!(!rendered.contains("[提供商/模型]"));
        assert!(!rendered.contains("[show|hide|toggle]"));

        let mut filtered = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        filtered.editor.insert_str("/ex");
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut filtered)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        assert!(rendered.contains("/exit"));
        assert!(!rendered.contains("/model"));
    }

    #[test]
    fn navigates_and_completes_slash_command_suggestions() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.editor.insert('/');
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        assert_eq!(state.editor.text(), "/effort ");
        assert_eq!(slash_suggestions(&state).len(), ReasoningEffort::ALL.len());

        state.editor.set_text("/sta");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        assert!(state.editor.is_empty());
        assert!(state.messages.last().unwrap().content.contains("模型"));
    }

    #[test]
    fn completes_slash_arguments_from_runtime_choices() {
        let mut state = UiState::new(
            "grok-4".to_string(),
            "http://localhost/v1/responses".to_string(),
            std::path::PathBuf::from("."),
        );
        state.provider = Some("xai".to_string());
        state.reasoning_choices = vec![ReasoningEffort::Low, ReasoningEffort::High];
        state.reasoning_effort = ReasoningEffort::Low;
        state.default_reasoning_effort = ReasoningEffort::High;
        state.model_choices = vec![ModelChoice {
            provider: "xai".to_string(),
            id: "grok-4".to_string(),
            name: Some("Grok 4".to_string()),
            api: ApiProtocol::Responses,
            context_window: 256_000,
            max_input_tokens: 256_000,
            reasoning: true,
        }];

        state.editor.set_text("/effort ");
        handle_key(
            KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
            &mut state,
            None,
        );
        handle_key(
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert_eq!(state.editor.text(), "/effort high");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::SetReasoning(ReasoningEffort::High)
        ));

        state.editor.set_text("/search l");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::SetWebSearch(WebSearchMode::Live)
        ));
        assert!(state.editor.is_empty());

        state.editor.set_text("/thinking s");
        handle_key(
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert_eq!(state.editor.text(), "/thinking show");

        state.editor.set_text("/model gro");
        handle_key(
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert_eq!(state.editor.text(), "/model xai/grok-4");
    }

    #[test]
    fn slash_runtime_controls_return_independent_actions() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.editor.insert_str("/effort high");
        let action = handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert!(matches!(
            action,
            UiAction::SetReasoning(ReasoningEffort::High)
        ));

        state.editor.insert_str("/search live");
        let action = handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert!(matches!(
            action,
            UiAction::SetWebSearch(WebSearchMode::Live)
        ));

        state.editor.insert_str("/reasoning high");
        let action = handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert!(matches!(action, UiAction::None));
        assert_eq!(
            state
                .messages
                .last()
                .map(|message| message.content.as_str()),
            Some("未知命令：/reasoning")
        );
    }

    #[test]
    fn slash_effort_lists_and_accepts_only_the_current_model_configuration() {
        let mut state = UiState::new(
            "grok".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.provider = Some("xai".to_string());
        state.reasoning_effort = ReasoningEffort::Medium;
        state.default_reasoning_effort = ReasoningEffort::High;
        state.reasoning_choices = vec![
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
        ];

        state.editor.insert_str("/effort");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        let notice = &state.messages.last().unwrap().content;
        assert!(notice.contains("xai/grok"));
        assert!(notice.contains("* medium"));
        assert!(notice.contains("high（默认）"));
        assert!(notice.contains("/effort <low|medium|high>"));
        assert!(!notice.contains("xhigh"));

        state.editor.insert_str("/effort xhigh");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        let error = &state.messages.last().unwrap().content;
        assert!(error.contains("未配置 effort \"xhigh\""));
        assert!(error.contains("可选值：low、medium、high"));

        state.editor.insert_str("/effort low");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::SetReasoning(ReasoningEffort::Low)
        ));
    }

    #[test]
    fn slash_exit_is_the_only_exit_command() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.editor.insert_str("/exit");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::Quit
        ));

        state.editor.insert_str("/quit");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        assert_eq!(
            state
                .messages
                .last()
                .map(|message| message.content.as_str()),
            Some("未知命令：/quit")
        );
    }

    #[test]
    fn slash_compact_preserves_optional_focus_instructions() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state
            .editor
            .insert_str("/compact focus on unresolved tests");
        let action = handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert!(matches!(
            action,
            UiAction::Compact(instructions) if instructions == "focus on unresolved tests"
        ));
    }

    #[test]
    fn slash_delete_uses_a_keyboard_confirmation_defaulting_to_no() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.editor.insert_str("/delete");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        assert_eq!(
            state.delete_confirmation,
            DeleteConfirmation::Selecting(DeleteChoice::No)
        );

        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        assert_eq!(state.delete_confirmation, DeleteConfirmation::None);
        assert_eq!(
            state
                .messages
                .last()
                .map(|message| message.content.as_str()),
            Some("已取消删除。")
        );

        state.editor.insert_str("/delete");
        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
            None,
        );
        handle_key(
            KeyEvent::new(KeyCode::Left, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert_eq!(
            state.delete_confirmation,
            DeleteConfirmation::Selecting(DeleteChoice::Yes)
        );
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::DeleteSession
        ));
        assert_eq!(state.delete_confirmation, DeleteConfirmation::None);
    }

    #[test]
    fn scrolling_up_stays_detached_while_streaming_output_arrives() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.apply_agent_event(AgentEvent::AssistantStarted);
        state.apply_agent_event(AgentEvent::TextDelta {
            text: "输出行\n".repeat(80),
        });
        let backend = TestBackend::new(50, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        assert!(state.max_scroll > 0);
        assert_eq!(state.scroll, state.max_scroll);

        handle_key(
            KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE),
            &mut state,
            None,
        );
        let scroll = state.scroll;
        assert!(!state.follow_tail);
        assert!(scroll < state.max_scroll);

        state.apply_agent_event(AgentEvent::TextDelta {
            text: "后续流式输出\n".repeat(10),
        });
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        assert!(!state.follow_tail);
        assert_eq!(state.scroll, scroll);

        handle_key(
            KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL),
            &mut state,
            None,
        );
        assert!(state.follow_tail);
        assert_eq!(state.scroll, state.max_scroll);
    }

    #[test]
    fn approval_prompt_supports_vertical_selection_and_shortcuts() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.pending_approval = Some(ApprovalView {
            name: "shell".to_string(),
            arguments: r#"{"command":"cargo test"}"#.to_string(),
            selection: ApprovalChoice::ApproveOnce,
        });

        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        assert_eq!(
            state.pending_approval.as_ref().unwrap().selection,
            ApprovalChoice::ApproveForSession
        );
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::ResolveApproval(ApprovalDecision::ApproveForSession)
        ));

        handle_key(
            KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert_eq!(
            state.pending_approval.as_ref().unwrap().selection,
            ApprovalChoice::Deny
        );
        handle_key(
            KeyEvent::new(KeyCode::Up, KeyModifiers::NONE),
            &mut state,
            None,
        );
        handle_key(
            KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert_eq!(
            state.pending_approval.as_ref().unwrap().selection,
            ApprovalChoice::ApproveForSession
        );
        handle_key(
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            &mut state,
            None,
        );
        assert_eq!(
            state.pending_approval.as_ref().unwrap().selection,
            ApprovalChoice::ApproveOnce
        );

        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::ResolveApproval(ApprovalDecision::ApproveOnce)
        ));
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::ResolveApproval(ApprovalDecision::ApproveForSession)
        ));
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::ResolveApproval(ApprovalDecision::Deny)
        ));
    }

    #[test]
    fn renders_tool_approval_without_overlapping_the_footer() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("/tmp/project"),
        );
        state.pending_approval = Some(ApprovalView {
            name: "shell".to_string(),
            arguments: r#"{"command":"cargo test --all-targets"}"#.to_string(),
            selection: ApprovalChoice::ApproveOnce,
        });
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        let compact = rendered.replace(' ', "");
        assert!(compact.contains("是否允许运行shell？"));
        assert!(rendered.contains("shell"));
        assert!(compact.contains("›1.允许一次"));
        assert!(compact.contains("2.本次会话内始终允许"));
        assert!(compact.contains("↑/↓选择·Enter确认"));
        assert!(!rendered.contains("/tmp/project"));
        assert!(rendered.contains("effort off"));
    }

    #[test]
    fn formats_compact_token_counts_and_context_percent() {
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1_250), "1.2k");
        assert_eq!(format_tokens(12_500), "13k");
        assert_eq!(format_context_percent(24_000, 128_000), "18.7");
    }

    #[test]
    fn footer_highlights_context_model_and_effort() {
        let mut state = UiState::new(
            "test-model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.context_tokens = 24_000;
        state.max_input_tokens = 128_000;
        state.reasoning_effort = ReasoningEffort::High;

        let line = footer_line(&state, 80);
        let context = line
            .spans
            .iter()
            .find(|span| span.content.contains("24k/128k"))
            .unwrap();
        assert_eq!(context.style.fg, Some(Color::Rgb(103, 232, 163)));
        assert!(context.style.add_modifier.contains(Modifier::BOLD));

        let model = line
            .spans
            .iter()
            .find(|span| span.content.as_ref() == "test-model")
            .unwrap();
        assert_eq!(model.style.fg, Some(Color::Rgb(126, 200, 255)));
        assert!(model.style.add_modifier.contains(Modifier::BOLD));

        let effort = line
            .spans
            .iter()
            .find(|span| span.content.as_ref() == "high")
            .unwrap();
        assert_eq!(effort.style.fg, Some(Color::Rgb(245, 190, 78)));

        state.context_tokens = 100_000;
        assert_eq!(context_usage_color(100_000, 128_000), Color::Yellow);
        assert_eq!(context_usage_color(120_000, 128_000), Color::LightRed);

        let compact = footer_line(&state, 32).to_string();
        assert!(compact.contains("上下文"));
        assert!(compact.contains("test-model"));
        assert!(display_width(&compact) <= 32);
    }
}
