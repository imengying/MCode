use std::io::{self, Read as _};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::cursor::{Hide, Show};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    Clear, ClearType as CrosstermClearType, disable_raw_mode, enable_raw_mode,
};
use pulldown_cmark::{
    CodeBlockKind, Event as MarkdownEvent, HeadingLevel, Options as MarkdownOptions, Parser, Tag,
    TagEnd,
};
use ratatui::backend::{Backend, ClearType as BackendClearType, CrosstermBackend, WindowSize};
use ratatui::buffer::Cell;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect, Size};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};
use ratatui::{Frame, Terminal, TerminalOptions, Viewport};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use unicode_width::UnicodeWidthChar;
use wl_clipboard_rs::paste::{ClipboardType, MimeType, Seat, get_contents, get_mime_types};

use crate::agent::{Agent, ModelChoice};
use crate::approval::{ApprovalDecision, ApprovalGate, ApprovalRequest, format_tool_arguments};
use crate::config::{ApiProtocol, ReasoningEffort, WebSearchMode};
use crate::event::{AgentEvent, CompactionReason};
use crate::highlight::highlight_code;
use crate::protocol::{
    ChatMessage, ImageAttachment, MAX_IMAGE_BYTES, MessageRole, Usage, sanitize_terminal_text,
};

const APPROVAL_HEIGHT: u16 = 6;
const DELETE_CONFIRMATION_HEIGHT: u16 = 5;
const COLLAPSED_PASTE_CHAR_THRESHOLD: usize = 1_000;
const COLLAPSED_PASTE_LINE_THRESHOLD: usize = 8;
const INLINE_VIEWPORT_HEIGHT: u16 = 10;
const INPUT_PREFIX_WIDTH: u16 = 2;
const MAX_INPUT_HEIGHT: u16 = 5;
const MAX_INPUT_HISTORY: usize = 100;
const MAX_SLASH_SUGGESTIONS: u16 = 8;
const PREVIEW_LINE_CHARS: usize = 240;
const TOOL_ARGUMENT_PREVIEW_LINES: usize = 2;
const TOOL_OUTPUT_PREVIEW_LINES: usize = 5;
const FRAME_INTERVAL: Duration = Duration::from_millis(50);
const THEME_BASE: Color = Color::Rgb(30, 30, 46);
const THEME_MANTLE: Color = Color::Rgb(24, 24, 37);
const THEME_SURFACE: Color = Color::Rgb(49, 50, 68);
const THEME_TEXT: Color = Color::Rgb(205, 214, 244);
const THEME_SUBTEXT: Color = Color::Rgb(186, 194, 222);
const THEME_MUTED: Color = Color::Rgb(108, 112, 134);
const THEME_BLUE: Color = Color::Rgb(137, 180, 250);
const THEME_GREEN: Color = Color::Rgb(166, 227, 161);
const THEME_YELLOW: Color = Color::Rgb(249, 226, 175);
const THEME_RED: Color = Color::Rgb(243, 139, 168);
const THEME_MAUVE: Color = Color::Rgb(203, 166, 247);
const THEME_TEAL: Color = Color::Rgb(148, 226, 213);
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
    let pending_tool_ids = if has_pending_run {
        agent
            .session()
            .pending_tool_calls()?
            .into_iter()
            .map(|pending| pending.call.id)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
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
    let backend = UiBackend::new(io::stdout());
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(INLINE_VIEWPORT_HEIGHT),
        },
    )
    .context("初始化终端失败")?;
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
    state.hold_pending_tools(&pending_tool_ids);
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
    let now = Instant::now();
    let mut last_frame = now.checked_sub(FRAME_INTERVAL).unwrap_or(now);
    let mut needs_draw = true;
    'ui: loop {
        while let Ok(agent_event) = event_rx.try_recv() {
            needs_draw = true;
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
            needs_draw = true;
            state.set_pending_approval(&request);
            pending_approval = Some(request);
        }

        let history_inserted = flush_finalized_history(&mut terminal, &mut state)?;
        needs_draw |= history_inserted;
        if needs_draw && (history_inserted || last_frame.elapsed() >= FRAME_INTERVAL) {
            terminal
                .draw(|frame| render(frame, &mut state))
                .context("绘制终端界面失败")?;
            last_frame = Instant::now();
            needs_draw = false;
        }

        if !event::poll(Duration::from_millis(20)).context("轮询终端事件失败")? {
            continue;
        }
        match event::read().context("读取终端事件失败")? {
            Event::Key(key)
                if key.kind != KeyEventKind::Release
                    && matches!(key.code, KeyCode::PageUp | KeyCode::PageDown) => {}
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                needs_draw = true;
                match handle_key(key, &mut state, active_cancel.as_ref()) {
                    UiAction::None => {}
                    UiAction::Quit => {
                        if let Some(cancel) = active_cancel.take() {
                            cancel.cancel();
                        }
                        if let Some(request) = pending_approval.take() {
                            request.resolve(ApprovalDecision::Deny);
                        }
                        state.clear_pending_approval();
                        break;
                    }
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
                                clear_terminal_history(&mut terminal)?;
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
                    UiAction::Clear => {
                        state.clear_view();
                        clear_terminal_history(&mut terminal)?;
                    }
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
                needs_draw = true;
                paste_text_or_image(&mut state, &text);
            }
            Event::Resize(..) => needs_draw = true,
            _ => {}
        }
    }

    terminal.clear().context("清理终端输入区失败")?;
    drop(terminal);
    drop(screen);
    if let Some(id) = deleted_session {
        println!("已删除会话 {id}。");
    }
    Ok(())
}

// Some PTYs do not answer cursor-position queries; retain Ratatui's last position as a fallback.
struct UiBackend {
    inner: CrosstermBackend<io::Stdout>,
    cursor: Option<Position>,
}

impl UiBackend {
    fn new(stdout: io::Stdout) -> Self {
        Self {
            inner: CrosstermBackend::new(stdout),
            cursor: None,
        }
    }
}

impl Backend for UiBackend {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        self.inner.draw(content)
    }

    fn append_lines(&mut self, count: u16) -> io::Result<()> {
        self.inner.append_lines(count)
    }

    fn hide_cursor(&mut self) -> io::Result<()> {
        self.inner.hide_cursor()
    }

    fn show_cursor(&mut self) -> io::Result<()> {
        self.inner.show_cursor()
    }

    fn get_cursor_position(&mut self) -> io::Result<Position> {
        if let Some(cursor) = self.cursor {
            return Ok(cursor);
        }
        let cursor = self.inner.get_cursor_position().unwrap_or(Position::ORIGIN);
        self.cursor = Some(cursor);
        Ok(cursor)
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
        let position = position.into();
        self.inner.set_cursor_position(position)?;
        self.cursor = Some(position);
        Ok(())
    }

    fn clear(&mut self) -> io::Result<()> {
        self.inner.clear()
    }

    fn clear_region(&mut self, clear_type: BackendClearType) -> io::Result<()> {
        self.inner.clear_region(clear_type)
    }

    fn size(&self) -> io::Result<Size> {
        self.inner.size()
    }

    fn window_size(&mut self) -> io::Result<WindowSize> {
        self.inner.window_size()
    }

    fn flush(&mut self) -> io::Result<()> {
        Backend::flush(&mut self.inner)
    }
}

type UiTerminal = Terminal<UiBackend>;

fn flush_finalized_history(terminal: &mut UiTerminal, state: &mut UiState) -> Result<bool> {
    let size = terminal.size().context("读取终端尺寸失败")?;
    if size.width < 24 || size.height < INLINE_VIEWPORT_HEIGHT {
        return Ok(false);
    }
    let heights = ui_section_heights(state, size.width, INLINE_VIEWPORT_HEIGHT);

    let messages = state.take_finalized_messages(size.width, heights.conversation);
    if messages.is_empty() {
        return Ok(false);
    }

    for message in messages {
        insert_history_lines(
            terminal,
            conversation_lines_for_messages(&[message]),
            size.width,
        )?;
    }
    Ok(true)
}

fn insert_history_lines(
    terminal: &mut UiTerminal,
    lines: Vec<Line<'static>>,
    width: u16,
) -> Result<()> {
    if lines.is_empty() {
        return Ok(());
    }
    let paragraph = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(THEME_TEXT).bg(THEME_BASE))
        .wrap(Wrap { trim: false });
    let height =
        u16::try_from(paragraph.line_count(width)).context("单条终端消息超过可显示的最大高度")?;
    terminal
        .insert_before(height, |buffer| {
            let area = buffer.area;
            paragraph.render(area, buffer);
        })
        .context("写入终端历史失败")
}

fn clear_terminal_history(terminal: &mut UiTerminal) -> Result<()> {
    execute!(
        io::stdout(),
        Clear(CrosstermClearType::Purge),
        Clear(CrosstermClearType::All)
    )
    .context("清空终端历史失败")?;
    terminal.clear().context("重置终端视口失败")
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
    Clear,
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
    state.detach_input_history();
    state.editor.set_text(&suggestion.replacement);
    state.slash_selection = 0;
    true
}

fn handle_key(
    key: KeyEvent,
    state: &mut UiState,
    active_cancel: Option<&CancellationToken>,
) -> UiAction {
    if matches!(key.code, KeyCode::Enter)
        && !key
            .modifiers
            .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT | KeyModifiers::CONTROL)
        && state.editor.text().trim().eq_ignore_ascii_case("/exit")
    {
        state.editor.set_text("");
        return UiAction::Quit;
    }

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
                state.detach_input_history();
                state.editor.insert('\n');
                return UiAction::None;
            }
            KeyCode::Char('v') => return UiAction::PasteClipboard,
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
            state.detach_input_history();
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
            state.detach_input_history();
            state.editor.insert(character);
            state.slash_selection = 0;
        }
        KeyCode::Backspace => {
            if state.editor.is_empty() {
                if state.pending_images.pop().is_some() {
                    state.status = attachment_status(state.pending_images.len());
                }
            } else {
                state.detach_input_history();
                state.editor.backspace();
            }
            state.slash_selection = 0;
        }
        KeyCode::Delete => {
            state.detach_input_history();
            state.editor.delete();
            state.slash_selection = 0;
        }
        KeyCode::Left => state.editor.move_left(),
        KeyCode::Right => state.editor.move_right(),
        KeyCode::Home => state.editor.move_home(),
        KeyCode::End => state.editor.move_end(),
        KeyCode::Up => state.previous_input(),
        KeyCode::Down => state.next_input(),
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
    state.record_input(prompt.clone());
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
        "clear" => UiAction::Clear,
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
                "命令：/model [ID]、/effort [级别]、/search [模式]、/compact [说明]、/status、/new、/delete、/clear、/help、/exit",
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
    state.detach_input_history();
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
                label: "…".to_string(),
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

    fn content(&self) -> String {
        let mut text = String::new();
        for item in &self.items {
            match item {
                EditorItem::Character(character) => text.push(*character),
                EditorItem::Paste { content, .. } => text.push_str(content),
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ReasoningActivityState {
    #[default]
    Inactive,
    Active,
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
    delete_confirmation: DeleteConfirmation,
    pending_images: Vec<ImageAttachment>,
    slash_selection: usize,
    input_history: Vec<String>,
    input_history_index: Option<usize>,
    input_history_draft: Option<String>,
    reasoning_buffer: String,
    reasoning_summary_parts: Vec<String>,
    reasoning_header: Option<String>,
    reasoning_activity_header: Option<String>,
    reasoning_activity_state: ReasoningActivityState,
    mcp_server_count: usize,
    mcp_tool_count: usize,
    pending_approval: Option<ApprovalView>,
    show_welcome: bool,
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
            delete_confirmation: DeleteConfirmation::None,
            pending_images: Vec::new(),
            slash_selection: 0,
            input_history: Vec::new(),
            input_history_index: None,
            input_history_draft: None,
            reasoning_buffer: String::new(),
            reasoning_summary_parts: Vec::new(),
            reasoning_header: None,
            reasoning_activity_header: None,
            reasoning_activity_state: ReasoningActivityState::Inactive,
            mcp_server_count: 0,
            mcp_tool_count: 0,
            pending_approval: None,
            show_welcome: true,
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
        self.show_welcome = false;
        match message.role {
            MessageRole::System => {}
            MessageRole::User => {
                let prompt = message.content.unwrap_or_default();
                self.record_input(prompt.clone());
                let content = sanitize_terminal_text(&format_user_content(prompt, &message.images));
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
                let reasoning = if self.api == ApiProtocol::Responses {
                    let mut parts = response_reasoning_summary_parts(&message.response_items);
                    if parts.is_empty()
                        && let Some(summary) = message.reasoning_content.as_deref()
                    {
                        parts.push(summary.to_string());
                    }
                    visible_reasoning_summary(&parts).unwrap_or_default()
                } else {
                    String::new()
                };
                let content = sanitize_terminal_text(&message.content.unwrap_or_default());
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
        self.delete_confirmation = DeleteConfirmation::None;
    }

    fn push_user(&mut self, prompt: String, images: &[ImageAttachment]) {
        self.show_welcome = false;
        self.record_input(prompt.clone());
        self.messages.push(ViewMessage {
            role: ViewRole::User,
            title: String::new(),
            content: sanitize_terminal_text(&format_user_content(prompt, images)),
            reasoning: String::new(),
            tool_arguments: None,
            tool_id: None,
            running: false,
        });
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
        self.current_assistant = None;
        self.generation_start = None;
        self.usage = Usage::default();
        self.context_tokens = 0;
        self.usage_estimated = false;
        self.status = "就绪".to_string();
        self.delete_confirmation = DeleteConfirmation::None;
        self.pending_images.clear();
        self.pending_approval = None;
        self.show_welcome = true;
        self.input_history.clear();
        self.detach_input_history();
        self.reset_reasoning_summary();
    }

    fn clear_view(&mut self) {
        self.messages.clear();
        self.current_assistant = None;
        self.generation_start = None;
        self.delete_confirmation = DeleteConfirmation::None;
        self.reset_reasoning_summary();
    }

    fn take_finalized_messages(&mut self, width: u16, visible_height: u16) -> Vec<ViewMessage> {
        let first_running = self
            .messages
            .iter()
            .position(|message| message.running)
            .unwrap_or(self.messages.len());
        if first_running == 0 || self.messages.is_empty() {
            return Vec::new();
        }

        let mut retained_height = 0usize;
        let mut keep_from = first_running;
        for index in (0..self.messages.len()).rev() {
            let message_height = Paragraph::new(Text::from(conversation_lines_for_messages(
                std::slice::from_ref(&self.messages[index]),
            )))
            .wrap(Wrap { trim: false })
            .line_count(width);
            let must_keep = index >= first_running || keep_from == self.messages.len();
            if !must_keep
                && retained_height.saturating_add(message_height) > usize::from(visible_height)
            {
                break;
            }
            retained_height = retained_height.saturating_add(message_height);
            keep_from = index;
        }

        let count = keep_from.min(first_running);
        if count == 0 {
            return Vec::new();
        }

        let finalized = self.messages.drain(..count).collect();
        self.current_assistant = shift_message_index(self.current_assistant, count);
        self.generation_start = shift_message_index(self.generation_start, count);
        finalized
    }

    fn hold_pending_tools(&mut self, tool_ids: &[String]) {
        for message in &mut self.messages {
            if message
                .tool_id
                .as_ref()
                .is_some_and(|id| tool_ids.contains(id))
            {
                message.running = true;
            }
        }
    }

    fn start_reasoning_summary(&mut self) {
        self.reset_reasoning_summary();
        self.reasoning_activity_state = ReasoningActivityState::Active;
    }

    fn begin_reasoning_summary_part(&mut self) {
        if !self.reasoning_buffer.is_empty() {
            self.reasoning_summary_parts
                .push(std::mem::take(&mut self.reasoning_buffer));
        }
        self.reasoning_header = None;
        self.reasoning_activity_state = ReasoningActivityState::Active;
    }

    fn append_reasoning_summary(&mut self, text: &str) {
        self.reasoning_activity_state = ReasoningActivityState::Active;
        self.reasoning_buffer
            .push_str(&sanitize_terminal_text(text));
        if self.reasoning_header.is_none() {
            self.reasoning_header = extract_first_bold(&self.reasoning_buffer);
            if let Some(header) = &self.reasoning_header {
                self.reasoning_activity_header = Some(header.clone());
            }
        }
    }

    fn finish_reasoning_summary(&mut self) {
        if !self.reasoning_buffer.is_empty() {
            self.reasoning_summary_parts
                .push(std::mem::take(&mut self.reasoning_buffer));
        }
        let parts = std::mem::take(&mut self.reasoning_summary_parts);
        if let Some(reasoning) = visible_reasoning_summary(&parts) {
            let index = self.ensure_assistant_message();
            if let Some(message) = self.messages.get_mut(index) {
                message.reasoning = reasoning;
            }
        }
        self.reasoning_header = None;
        self.reasoning_activity_header = None;
        self.reasoning_activity_state = ReasoningActivityState::Inactive;
    }

    fn reset_reasoning_summary(&mut self) {
        self.reasoning_buffer.clear();
        self.reasoning_summary_parts.clear();
        self.reasoning_header = None;
        self.reasoning_activity_header = None;
        self.reasoning_activity_state = ReasoningActivityState::Inactive;
    }

    fn reasoning_activity(&self) -> Option<String> {
        (self.running && self.reasoning_activity_state == ReasoningActivityState::Active).then(
            || {
                self.reasoning_activity_header
                    .clone()
                    .unwrap_or_else(|| "正在思考…".to_string())
            },
        )
    }

    fn record_input(&mut self, input: String) {
        self.detach_input_history();
        if input.trim().is_empty() || self.input_history.last() == Some(&input) {
            return;
        }
        if self.input_history.len() == MAX_INPUT_HISTORY {
            self.input_history.remove(0);
        }
        self.input_history.push(input);
    }

    fn previous_input(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        if self.input_history_index.is_none() {
            self.input_history_draft = Some(self.editor.content());
        }
        let index = self
            .input_history_index
            .map_or(self.input_history.len() - 1, |index| {
                index.saturating_sub(1)
            });
        self.input_history_index = Some(index);
        self.editor.set_text(&self.input_history[index]);
        self.slash_selection = 0;
    }

    fn next_input(&mut self) {
        let Some(index) = self.input_history_index else {
            return;
        };
        if index + 1 < self.input_history.len() {
            let next = index + 1;
            self.input_history_index = Some(next);
            self.editor.set_text(&self.input_history[next]);
        } else {
            self.input_history_index = None;
            self.editor
                .set_text(&self.input_history_draft.take().unwrap_or_default());
        }
        self.slash_selection = 0;
    }

    fn detach_input_history(&mut self) {
        self.input_history_index = None;
        self.input_history_draft = None;
    }

    fn model_list_notice(&self) -> String {
        if self.model_choices.is_empty() {
            return format!(
                "当前模型：{}\n~/.mcode/models.json 中没有模型列表；仍可使用 /model <ID> 选择当前端点上的模型。",
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
        match event {
            AgentEvent::RunStarted | AgentEvent::RunResumed => {
                self.status = "处理中".to_string();
                self.running = true;
            }
            AgentEvent::AssistantStarted => {
                let index = self.start_assistant_message();
                self.generation_start = Some(index);
                self.start_reasoning_summary();
                self.status = "正在思考".to_string();
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
                self.start_reasoning_summary();
                self.status = format!(
                    "正在重试响应（{attempt}/{max_attempts}）：{}",
                    sanitize_terminal_text(&message)
                );
            }
            AgentEvent::TextDelta { text } => {
                self.finish_reasoning_summary();
                let index = self.ensure_assistant_message();
                if let Some(message) = self.messages.get_mut(index) {
                    message.content.push_str(&sanitize_terminal_text(&text));
                }
                self.status = "正在生成回复".to_string();
            }
            AgentEvent::ReasoningSummaryDelta { text } => {
                if self.api == ApiProtocol::Responses {
                    self.append_reasoning_summary(&text);
                }
                self.status = "正在思考".to_string();
            }
            AgentEvent::ReasoningSummaryPartAdded { .. } => {
                if self.api == ApiProtocol::Responses {
                    self.begin_reasoning_summary_part();
                }
            }
            AgentEvent::ReasoningSummaryFinished => self.finish_reasoning_summary(),
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
                    message.running = true;
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
                self.status = tool_running_status(&name);
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
                self.status = "处理中".to_string();
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
            AgentEvent::RunFinished => {
                self.finish_reasoning_summary();
                self.finish_run("就绪");
            }
            AgentEvent::Cancelled => {
                self.reset_reasoning_summary();
                self.finish_run("已取消");
            }
            AgentEvent::Error { message } => {
                self.reset_reasoning_summary();
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
    }

    fn finish_run(&mut self, status: &str) {
        for message in &mut self.messages {
            message.running = false;
        }
        self.messages.retain(|message| {
            message.role != ViewRole::Assistant
                || !message.content.is_empty()
                || !message.reasoning.is_empty()
        });
        self.current_assistant = None;
        self.generation_start = None;
        self.running = false;
        self.status = status.to_string();
        self.pending_approval = None;
        self.reset_reasoning_summary();
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
        self.finish_reasoning_summary();
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
}

fn shift_message_index(index: Option<usize>, removed: usize) -> Option<usize> {
    index.and_then(|index| index.checked_sub(removed))
}

fn render(frame: &mut Frame<'_>, state: &mut UiState) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().fg(THEME_TEXT).bg(THEME_BASE)),
        area,
    );
    if area.width < 24 || area.height < INLINE_VIEWPORT_HEIGHT {
        frame.render_widget(
            Paragraph::new("终端窗口过小")
                .style(Style::default().fg(THEME_RED))
                .block(Block::default().borders(Borders::ALL)),
            area,
        );
        return;
    }

    let heights = ui_section_heights(state, area.width, area.height);
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(heights.conversation),
            Constraint::Length(heights.suggestions),
            Constraint::Length(heights.activity),
            Constraint::Length(heights.input),
            Constraint::Length(1),
        ])
        .split(area);
    render_conversation(frame, state, areas[0]);
    render_slash_suggestions(frame, state, areas[1]);
    render_reasoning_activity(frame, state, areas[2]);
    render_input(frame, state, areas[3]);
    render_footer(frame, state, areas[4]);
}

#[derive(Debug, Clone, Copy)]
struct UiSectionHeights {
    conversation: u16,
    suggestions: u16,
    activity: u16,
    input: u16,
}

fn ui_section_heights(state: &UiState, width: u16, height: u16) -> UiSectionHeights {
    let input = if state.pending_approval.is_some() {
        APPROVAL_HEIGHT
    } else if state.delete_confirmation != DeleteConfirmation::None {
        DELETE_CONFIRMATION_HEIGHT
    } else {
        let editor_height = state
            .editor
            .rendered_height(width.saturating_sub(INPUT_PREFIX_WIDTH))
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
    let activity = u16::from(state.reasoning_activity().is_some());
    let reserved_height = 3_u16
        .saturating_add(activity)
        .saturating_add(input)
        .saturating_add(1);
    let suggestions = u16::try_from(suggestion_count)
        .unwrap_or(u16::MAX)
        .min(MAX_SLASH_SUGGESTIONS)
        .min(height.saturating_sub(reserved_height));
    let conversation = height
        .saturating_sub(suggestions)
        .saturating_sub(activity)
        .saturating_sub(input)
        .saturating_sub(1);
    UiSectionHeights {
        conversation,
        suggestions,
        activity,
        input,
    }
}

fn render_reasoning_activity(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    let Some(activity) = state.reasoning_activity() else {
        return;
    };
    let available = usize::from(area.width).saturating_sub(2);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("• ", Style::default().fg(THEME_YELLOW)),
            Span::styled(
                truncate_width(&activity, available.saturating_add(1)),
                Style::default().fg(THEME_SUBTEXT),
            ),
        ])),
        area,
    );
}

fn render_conversation(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    let mut lines = Vec::new();
    if state.show_welcome {
        lines.extend(welcome_lines(state, area.width));
        if !state.messages.is_empty() {
            lines.push(Line::default());
        }
    }
    lines.extend(conversation_lines(state));
    if lines.is_empty() || area.height == 0 || area.width == 0 {
        return;
    }
    let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    let total_height = paragraph.line_count(area.width);
    let visible_height = total_height.min(usize::from(area.height));
    let render_area = Rect::new(
        area.x,
        area.bottom()
            .saturating_sub(u16::try_from(visible_height).unwrap_or(area.height)),
        area.width,
        u16::try_from(visible_height).unwrap_or(area.height),
    );
    let scroll = u16::try_from(total_height.saturating_sub(visible_height)).unwrap_or(u16::MAX);
    frame.render_widget(paragraph.scroll((scroll, 0)), render_area);
}

fn welcome_lines(state: &UiState, width: u16) -> Vec<Line<'static>> {
    let Some(inner_width) = usize::from(width).checked_sub(4).map(|width| width.min(56)) else {
        return Vec::new();
    };
    let title = format!(">_ MCode (v{})", crate::VERSION);
    let model = format!(
        "model: {} {}   /model 切换",
        state.qualified_model(),
        state.reasoning_effort
    );
    let title = truncate_width(&title, inner_width.saturating_add(1));
    let model = truncate_width(&model, inner_width.saturating_add(1));
    let bordered = [
        Line::from(vec![
            Span::styled(">_ ", Style::default().fg(THEME_MUTED)),
            Span::styled(
                title.strip_prefix(">_ ").unwrap_or(&title).to_string(),
                Style::default().fg(THEME_TEXT).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::default(),
        Line::from(vec![
            Span::styled("model: ", Style::default().fg(THEME_MUTED)),
            Span::styled(
                model.strip_prefix("model: ").unwrap_or(&model).to_string(),
                Style::default().fg(THEME_BLUE),
            ),
        ]),
    ];
    let mut lines = Vec::with_capacity(bordered.len().saturating_add(4));
    lines.push(Line::from(Span::styled(
        format!("╭{}╮", "─".repeat(inner_width.saturating_add(2))),
        Style::default().fg(THEME_MUTED),
    )));
    for line in bordered {
        let used = line
            .spans
            .iter()
            .map(|span| display_width(span.content.as_ref()))
            .sum::<usize>();
        let mut spans = Vec::with_capacity(line.spans.len().saturating_add(3));
        spans.push(Span::styled("│ ", Style::default().fg(THEME_MUTED)));
        spans.extend(line.spans);
        spans.push(Span::raw(" ".repeat(inner_width.saturating_sub(used))));
        spans.push(Span::styled(" │", Style::default().fg(THEME_MUTED)));
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(Span::styled(
        format!("╰{}╯", "─".repeat(inner_width.saturating_add(2))),
        Style::default().fg(THEME_MUTED),
    )));
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        "  描述任务，或输入 / 查看命令",
        Style::default().fg(THEME_MUTED),
    )));
    lines
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
                Style::default().fg(THEME_BLUE).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(THEME_SUBTEXT)
            };
            Line::from(vec![
                Span::styled(
                    marker,
                    Style::default().fg(THEME_BLUE).add_modifier(Modifier::BOLD),
                ),
                Span::styled(label, command_style),
                Span::raw(padding),
                Span::styled(
                    suggestion.description.clone(),
                    Style::default().fg(if is_selected {
                        THEME_SUBTEXT
                    } else {
                        THEME_MUTED
                    }),
                ),
            ])
            .style(if is_selected {
                Style::default().bg(THEME_SURFACE)
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
                    .fg(THEME_YELLOW)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(details, Style::default().fg(THEME_SUBTEXT))),
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
                Style::default().fg(THEME_MUTED),
            )),
        ];
        frame.render_widget(
            Paragraph::new(lines).style(Style::default().bg(THEME_MANTLE)),
            area,
        );
        return;
    }

    if let DeleteConfirmation::Selecting(selection) = state.delete_confirmation {
        let yes_style = if selection == DeleteChoice::Yes {
            Style::default()
                .fg(THEME_RED)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default().fg(THEME_MUTED)
        };
        let no_style = if selection == DeleteChoice::No {
            Style::default()
                .fg(THEME_GREEN)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default().fg(THEME_MUTED)
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
                    .fg(THEME_YELLOW)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("{yes_marker} Yes  删除并退出"),
                yes_style,
            )),
            Line::from(Span::styled(format!("{no_marker} No   返回"), no_style)),
            Line::from(Span::styled(
                "使用方向键选择，按 Enter 确认",
                Style::default().fg(THEME_MUTED),
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
        THEME_YELLOW
    } else {
        THEME_BLUE
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
                    .fg(THEME_GREEN)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                label,
                Style::default()
                    .fg(THEME_GREEN)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                truncate_width(&suffix, available.saturating_add(1)),
                Style::default().fg(THEME_SUBTEXT),
            ),
        ])),
        area,
    );
}

fn render_footer(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    frame.render_widget(
        Paragraph::new(footer_line(state, usize::from(area.width)))
            .style(Style::default().bg(THEME_MANTLE)),
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
        Style::default().fg(THEME_BLUE).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(THEME_SUBTEXT)
    };
    Line::from(vec![
        Span::styled(
            if is_selected { "› " } else { "  " },
            Style::default().fg(THEME_BLUE).add_modifier(Modifier::BOLD),
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
                .fg(THEME_SUBTEXT)
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
        spans.push(Span::styled(usage, Style::default().fg(THEME_MUTED)));
    }
    spans.push(Span::raw(" ".repeat(padding)));
    if !model.is_empty() {
        spans.push(Span::styled(
            model,
            Style::default().fg(THEME_BLUE).add_modifier(Modifier::BOLD),
        ));
        if show_effort {
            spans.push(Span::styled(" | effort ", Style::default().fg(THEME_MUTED)));
            spans.push(Span::styled(
                effort,
                Style::default()
                    .fg(THEME_YELLOW)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        spans.push(Span::raw(" "));
    }
    Line::from(spans)
}

fn context_usage_color(tokens: u64, limit: u64) -> Color {
    if limit == 0 {
        return THEME_SUBTEXT;
    }
    let percent = u128::from(tokens).saturating_mul(100) / u128::from(limit);
    if percent >= 90 {
        THEME_RED
    } else if percent >= 70 {
        THEME_YELLOW
    } else {
        THEME_GREEN
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

fn extract_first_bold(text: &str) -> Option<String> {
    let (_, after_open) = text.split_once("**")?;
    let (header, _) = after_open.split_once("**")?;
    let header = header.split_whitespace().collect::<Vec<_>>().join(" ");
    (!header.is_empty()).then_some(header)
}

fn response_reasoning_summary_parts(items: &[serde_json::Value]) -> Vec<String> {
    items
        .iter()
        .filter(|item| item.get("type").and_then(serde_json::Value::as_str) == Some("reasoning"))
        .flat_map(|item| {
            item.get("summary")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter(|part| part.get("type").and_then(serde_json::Value::as_str) == Some("summary_text"))
        .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
        .map(ToString::to_string)
        .collect()
}

fn visible_reasoning_summary(parts: &[String]) -> Option<String> {
    let parts = parts
        .iter()
        .map(|part| sanitize_terminal_text(part))
        .collect::<Vec<_>>();
    let (header, content) = split_reasoning_summary_parts(&parts);
    (!header.is_empty() && !content.trim().is_empty()).then(|| content.trim().to_string())
}

fn split_reasoning_summary_parts(parts: &[String]) -> (String, String) {
    let mut placeholder_header = None;
    let mut content_parts = Vec::with_capacity(parts.len());

    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let header_end = part.strip_prefix("**").and_then(|after_open| {
            after_open
                .find("**")
                .and_then(|close| (close > 0).then_some(close + 4))
        });
        let body = header_end.map_or(part, |header_end| &part[header_end..]);
        if body.trim() == "<!-- -->" {
            if content_parts.is_empty()
                && placeholder_header.is_none()
                && let Some(header_end) = header_end
            {
                placeholder_header = Some(part[..header_end].to_string());
            }
            continue;
        }
        content_parts.push(part);
    }

    let content = content_parts.join("\n\n");
    if content.is_empty() {
        return (placeholder_header.unwrap_or_default(), content);
    }
    if let Some(after_open) = content.strip_prefix("**")
        && let Some(close) = after_open.find("**")
    {
        let after_close = 2 + close + 2;
        if matches!(content[after_close..].chars().next(), Some('\n' | '\r')) {
            return (
                content[..after_close].to_string(),
                content[after_close..].to_string(),
            );
        }
    }
    (placeholder_header.unwrap_or_default(), content)
}

fn conversation_lines(state: &UiState) -> Vec<Line<'static>> {
    conversation_lines_for_messages(&state.messages)
}

fn conversation_lines_for_messages(messages: &[ViewMessage]) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for message in messages {
        if message.role == ViewRole::Assistant {
            if message.reasoning.is_empty() && message.content.is_empty() {
                continue;
            }
            if !message.reasoning.is_empty() {
                append_reasoning_summary(&mut lines, &message.reasoning);
                if !message.content.is_empty() {
                    lines.push(Line::default());
                }
            }
            if !message.content.is_empty() {
                append_markdown_lines(
                    &mut lines,
                    &message.content,
                    Style::default().fg(THEME_TEXT),
                );
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
            ViewRole::Tool => (THEME_YELLOW, Style::default().fg(THEME_SUBTEXT)),
            ViewRole::Notice => (THEME_TEAL, Style::default().fg(THEME_SUBTEXT)),
            ViewRole::Error => (THEME_RED, Style::default().fg(THEME_RED)),
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
            Span::styled(running, Style::default().fg(THEME_MUTED)),
        ]));
        append_markdown_lines(&mut lines, &message.content, content_style);
        lines.push(Line::default());
    }
    lines
}

fn append_reasoning_summary(lines: &mut Vec<Line<'static>>, reasoning: &str) {
    for (index, mut line) in MarkdownRenderer::new(Style::default())
        .render(reasoning)
        .into_iter()
        .enumerate()
    {
        for span in &mut line.spans {
            span.style = span.style.fg(THEME_SUBTEXT).add_modifier(Modifier::ITALIC);
        }
        let mut spans = Vec::with_capacity(line.spans.len().saturating_add(1));
        spans.push(Span::styled(
            if index == 0 { "• " } else { "  " },
            Style::default().fg(THEME_MUTED),
        ));
        spans.extend(line.spans);
        lines.push(Line::from(spans));
    }
}

fn append_user_message(lines: &mut Vec<Line<'static>>, content: &str) {
    let mut content_lines = content.lines();
    let first = content_lines.next().unwrap_or_default();
    lines.push(Line::from(vec![
        Span::styled(
            "› ",
            Style::default().fg(THEME_BLUE).add_modifier(Modifier::BOLD),
        ),
        Span::styled(first.to_string(), Style::default().fg(THEME_TEXT)),
    ]));
    for line in content_lines {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(line.to_string(), Style::default().fg(THEME_TEXT)),
        ]));
    }
}

fn append_tool_message(lines: &mut Vec<Line<'static>>, message: &ViewMessage) {
    let failed = message.role == ViewRole::Error;
    let color = if failed {
        THEME_RED
    } else if message.running {
        THEME_YELLOW
    } else {
        THEME_GREEN
    };
    lines.push(Line::from(vec![
        Span::styled(
            "• ",
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            tool_action_title(&message.title, message.running, failed),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
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
        if message.title == "shell" {
            append_code_preview_lines(
                lines,
                arguments,
                "bash",
                TOOL_ARGUMENT_PREVIEW_LINES,
                first_prefix,
                "    ",
            );
        } else {
            append_preview_lines(
                lines,
                arguments,
                TOOL_ARGUMENT_PREVIEW_LINES,
                first_prefix,
                "    ",
                Style::default().fg(THEME_SUBTEXT),
            );
        }
    }
    if !message.content.is_empty() {
        append_preview_lines(
            lines,
            &message.content,
            TOOL_OUTPUT_PREVIEW_LINES,
            "  └ ",
            "    ",
            if failed {
                Style::default().fg(THEME_RED)
            } else {
                Style::default().fg(THEME_MUTED)
            },
        );
    }
}

fn tool_running_status(name: &str) -> String {
    tool_action_title(name, true, false)
}

fn tool_action_title(name: &str, running: bool, failed: bool) -> String {
    if name.starts_with("等待审批：") || name.starts_with("已允许") || name.starts_with("已拒绝：")
    {
        return name.to_string();
    }
    let labels = match name {
        "shell" => ("正在运行命令", "已运行命令", "命令运行失败"),
        "read_file" => ("正在读取文件", "已读取文件", "读取文件失败"),
        "write_file" => ("正在写入文件", "已写入文件", "写入文件失败"),
        "edit_file" => ("正在编辑文件", "已编辑文件", "编辑文件失败"),
        "web_search" | "网页搜索" => ("正在搜索网页", "已搜索网页", "网页搜索失败"),
        "fetch_content" => ("正在读取网页", "已读取网页", "读取网页失败"),
        _ => {
            return if failed {
                format!("{name} 运行失败")
            } else if running {
                format!("正在运行 {name}")
            } else {
                format!("已运行 {name}")
            };
        }
    };
    if failed {
        labels.2
    } else if running {
        labels.0
    } else {
        labels.1
    }
    .to_string()
}

fn append_preview_lines(
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
            Span::styled(prefix.to_string(), Style::default().fg(THEME_MUTED)),
            Span::styled(truncate_preview_line(line), style),
        ]));
    }
    if content_lines.len() > limit {
        lines.push(Line::from(vec![
            Span::styled(
                continuation_prefix.to_string(),
                Style::default().fg(THEME_MUTED),
            ),
            Span::styled(
                "…",
                Style::default()
                    .fg(THEME_MUTED)
                    .add_modifier(Modifier::ITALIC),
            ),
        ]));
    }
}

fn append_code_preview_lines(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    language: &str,
    limit: usize,
    first_prefix: &str,
    continuation_prefix: &str,
) {
    let content_lines = content.lines().collect::<Vec<_>>();
    let preview = content_lines
        .iter()
        .take(limit)
        .map(|line| truncate_preview_line(line))
        .collect::<Vec<_>>()
        .join("\n");
    for (index, highlighted) in highlight_code(&preview, language).into_iter().enumerate() {
        let prefix = if index == 0 {
            first_prefix
        } else {
            continuation_prefix
        };
        let mut spans = Vec::with_capacity(highlighted.spans.len().saturating_add(1));
        spans.push(Span::styled(
            prefix.to_string(),
            Style::default().fg(THEME_MUTED),
        ));
        spans.extend(highlighted.spans);
        lines.push(Line::from(spans));
    }
    if content_lines.len() > limit {
        lines.push(Line::from(vec![
            Span::styled(
                continuation_prefix.to_string(),
                Style::default().fg(THEME_MUTED),
            ),
            Span::styled(
                "…",
                Style::default()
                    .fg(THEME_MUTED)
                    .add_modifier(Modifier::ITALIC),
            ),
        ]));
    }
}

fn truncate_preview_line(line: &str) -> String {
    let mut characters = line.chars();
    let preview = characters
        .by_ref()
        .take(PREVIEW_LINE_CHARS)
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
    code_block_language: Option<String>,
    code_block_buffer: String,
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
            code_block_language: None,
            code_block_buffer: String::new(),
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
            MarkdownEvent::Text(text) if self.code_block && self.code_block_language.is_some() => {
                self.code_block_buffer.push_str(&text);
            }
            MarkdownEvent::Text(text)
            | MarkdownEvent::Html(text)
            | MarkdownEvent::InlineHtml(text) => {
                self.push_text(&text, self.current_style());
            }
            MarkdownEvent::Code(code) => {
                self.push_text(&code, Style::default().fg(THEME_TEAL));
            }
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
                self.push_text("────────────────────────", Style::default().fg(THEME_MUTED));
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
            Tag::CodeBlock(kind) => {
                self.flush_line(false);
                self.code_block = true;
                self.code_block_language = match kind {
                    CodeBlockKind::Fenced(info) => info
                        .split([',', ' ', '\t'])
                        .next()
                        .filter(|language| !language.is_empty())
                        .map(ToString::to_string),
                    CodeBlockKind::Indented => None,
                };
                self.code_block_buffer.clear();
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
                    self.push_text(" | ", Style::default().fg(THEME_MUTED));
                }
            }
            Tag::Emphasis => self.emphasis_depth = self.emphasis_depth.saturating_add(1),
            Tag::Strikethrough => {
                self.strikethrough_depth = self.strikethrough_depth.saturating_add(1);
            }
            Tag::Link { dest_url, .. } => self.links.push(dest_url.into_string()),
            Tag::Image { dest_url, .. } => {
                self.push_text("图片：", Style::default().fg(THEME_MUTED));
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
                if let Some(language) = self.code_block_language.take() {
                    let code = std::mem::take(&mut self.code_block_buffer);
                    for mut line in highlight_code(&code, &language) {
                        for span in &mut line.spans {
                            if span.style.fg.is_none() {
                                span.style = span.style.fg(THEME_TEXT);
                            }
                        }
                        self.push_styled_line(line);
                    }
                }
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
                            .fg(THEME_BLUE)
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
            Style::default().fg(THEME_TEXT)
        } else {
            self.base
        };
        if let Some(level) = self.heading {
            style = style
                .fg(match level {
                    HeadingLevel::H1 => THEME_MAUVE,
                    HeadingLevel::H2 => THEME_BLUE,
                    _ => THEME_TEXT,
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
            style = style.fg(THEME_BLUE).add_modifier(Modifier::UNDERLINED);
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

    fn push_styled_line(&mut self, mut line: Line<'static>) {
        self.ensure_prefix();
        self.current.append(&mut line.spans);
        if self.current.is_empty() {
            self.lines.push(Line::default());
        } else {
            self.flush_line(false);
        }
    }

    fn ensure_prefix(&mut self) {
        if !self.current.is_empty() {
            return;
        }
        if self.quote_depth > 0 {
            self.current.push(Span::styled(
                "│ ".repeat(self.quote_depth),
                Style::default().fg(THEME_MUTED),
            ));
        }
        let prefix = self
            .pending_item_prefix
            .take()
            .or_else(|| self.item_continuations.last().cloned());
        if let Some(prefix) = prefix {
            self.current
                .push(Span::styled(prefix, Style::default().fg(THEME_MUTED)));
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
        format!("{prefix}\n…")
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
        if let Err(error) = execute!(io::stdout(), EnableBracketedPaste, Hide) {
            let _ = disable_raw_mode();
            return Err(error).context("配置终端模式失败");
        }
        Ok(Self)
    }
}

impl Drop for ScreenGuard {
    fn drop(&mut self) {
        let _ = execute!(io::stdout(), Show, DisableBracketedPaste);
        let _ = disable_raw_mode();
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
    fn submitting_a_collapsed_paste_sends_its_full_contents() {
        let pasted = "full pasted content\n".repeat(COLLAPSED_PASTE_LINE_THRESHOLD);
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.editor.insert_paste(&pasted);
        assert_eq!(state.editor.text(), "…");

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
    fn slash_exit_quits_immediately_while_an_agent_is_running() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.running = true;
        state.editor.insert_str("/exit");

        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::Quit
        ));
        assert!(state.editor.is_empty());
    }

    #[test]
    fn renders_a_welcome_card_for_a_new_session() {
        let mut state = UiState::new(
            "gpt-test".to_string(),
            "http://localhost/v1/responses".to_string(),
            std::path::PathBuf::from("."),
        );
        state.provider = Some("openai".to_string());
        state.reasoning_effort = ReasoningEffort::High;
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut state)).unwrap();

        let rendered = rendered_terminal(&terminal);
        assert!(rendered.contains("MCode"));
        assert!(rendered.contains("openai/gpt-test high"));
        assert!(rendered.contains("描"));
        assert!(rendered.contains("命"));
        assert!(rendered.contains("╭"));
    }

    #[test]
    fn aligns_a_short_conversation_directly_above_the_input() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.push_user("紧凑布局".to_string(), &[]);
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut state)).unwrap();

        let buffer = terminal.backend().buffer();
        let message_y = buffer
            .content
            .iter()
            .enumerate()
            .find_map(|(index, cell)| (cell.symbol() == "紧").then_some(index / 80))
            .unwrap();
        let input_y = buffer
            .content
            .iter()
            .enumerate()
            .find_map(|(index, cell)| (cell.symbol() == ">").then_some(index / 80))
            .unwrap();
        assert!(input_y.saturating_sub(message_y) <= 2);
        assert!(message_y > 0);
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
    fn renders_commonmark_as_terminal_styles() {
        let content = "# Heading\n\n**bold** and *italic* with `code` and [docs](https://example.com).\n\n- one\n- two\n\n> quote\n\n```rust\nlet value = 1;\n```";
        let mut lines = Vec::new();
        append_markdown_lines(&mut lines, content, Style::default().fg(THEME_TEXT));
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
        assert_eq!(code.style.fg, Some(THEME_TEAL));
        assert_eq!(code.style.bg, None);

        let rust_line = lines
            .iter()
            .find(|line| line.to_string().contains("let value = 1;"))
            .unwrap();
        let mut rust_colors = rust_line
            .spans
            .iter()
            .filter_map(|span| span.style.fg)
            .collect::<Vec<_>>();
        rust_colors.dedup();
        assert!(rust_colors.len() > 1);
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
        let running = conversation_lines(&state)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(running.contains("• 正在运行命令"));
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
        assert!(rendered.contains("• 已运行命令"));
        assert!(rendered.contains("$ first"));
        assert!(rendered.contains("second"));
        assert!(!rendered.contains("third"));
        assert!(rendered.contains("output 0"));
        assert!(rendered.contains("output 4"));
        assert!(!rendered.contains("output 5"));
        assert!(!rendered.contains("output 11"));
        assert_eq!(
            rendered.lines().filter(|line| line.trim() == "…").count(),
            2
        );
        assert!(!rendered.contains("折叠"));
    }

    #[test]
    fn renders_codex_style_responses_summaries_without_raw_chat_reasoning() {
        let mut chat_state = UiState::new(
            "reasoning-model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("/tmp/project"),
        );
        chat_state.apply_agent_event(AgentEvent::RunStarted);
        chat_state.apply_agent_event(AgentEvent::AssistantStarted);
        chat_state.apply_agent_event(AgentEvent::ReasoningSummaryDelta {
            text: "Private raw reasoning.".to_string(),
        });
        let active = conversation_lines(&chat_state)
            .iter()
            .map(Line::to_string)
            .collect::<String>();
        assert!(active.is_empty());
        assert_eq!(
            chat_state.reasoning_activity().as_deref(),
            Some("正在思考…")
        );
        assert!(!active.contains("Private raw reasoning."));

        chat_state.apply_agent_event(AgentEvent::TextDelta {
            text: "Final response.".to_string(),
        });
        let completed = conversation_lines(&chat_state)
            .iter()
            .map(Line::to_string)
            .collect::<String>();
        assert!(completed.contains("Final response."));
        assert!(!completed.contains("正在思考"));

        let mut responses_state = UiState::new(
            "reasoning-model".to_string(),
            "http://localhost/v1/responses".to_string(),
            std::path::PathBuf::from("."),
        );
        responses_state.api = ApiProtocol::Responses;
        responses_state.apply_agent_event(AgentEvent::RunStarted);
        responses_state.apply_agent_event(AgentEvent::AssistantStarted);
        responses_state.apply_agent_event(AgentEvent::ReasoningSummaryPartAdded { index: 0 });
        responses_state.apply_agent_event(AgentEvent::ReasoningSummaryDelta {
            text: "**Inspecting files**\n\nReading the relevant modules.".to_string(),
        });
        assert_eq!(
            responses_state.reasoning_activity().as_deref(),
            Some("Inspecting files")
        );
        assert!(conversation_lines(&responses_state).is_empty());
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render(frame, &mut responses_state))
            .unwrap();
        let active = rendered_terminal(&terminal);
        assert!(active.contains("• Inspecting files"));
        assert!(!active.contains("Reading the relevant modules."));

        responses_state.apply_agent_event(AgentEvent::ReasoningSummaryPartAdded { index: 1 });
        assert_eq!(
            responses_state.reasoning_activity().as_deref(),
            Some("Inspecting files")
        );
        responses_state.apply_agent_event(AgentEvent::ReasoningSummaryDelta {
            text: "**Running".to_string(),
        });
        assert_eq!(
            responses_state.reasoning_activity().as_deref(),
            Some("Inspecting files")
        );
        responses_state.apply_agent_event(AgentEvent::ReasoningSummaryDelta {
            text: " checks**\n\nVerifying the behavior.".to_string(),
        });
        assert_eq!(
            responses_state.reasoning_activity().as_deref(),
            Some("Running checks")
        );
        responses_state.apply_agent_event(AgentEvent::ReasoningSummaryFinished);

        let summary = conversation_lines(&responses_state)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(responses_state.reasoning_activity(), None);
        assert!(summary.contains("• Reading the relevant modules."));
        assert!(summary.contains("Running checks"));
        assert!(summary.contains("Verifying the behavior."));
        assert!(!summary.contains("Inspecting files"));
        assert!(!summary.contains('…'));
        assert!(!summary.contains("隐藏"));
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
    fn finalized_history_keeps_live_messages_in_the_viewport() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.push_user("检查项目".to_string(), &[]);
        state.apply_agent_event(AgentEvent::AssistantStarted);
        state.apply_agent_event(AgentEvent::TextDelta {
            text: "正在处理".to_string(),
        });
        let finalized = state.take_finalized_messages(80, 3);
        assert_eq!(finalized.len(), 1);
        assert_eq!(finalized[0].role, ViewRole::User);
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.current_assistant, Some(0));
        assert_eq!(state.generation_start, Some(0));

        state.apply_agent_event(AgentEvent::ToolStarted {
            id: "tool-1".to_string(),
            name: "shell".to_string(),
            arguments: r#"{"command":"cargo check"}"#.to_string(),
        });
        let finalized = state.take_finalized_messages(80, 3);
        assert_eq!(finalized.len(), 1);
        assert_eq!(finalized[0].role, ViewRole::Assistant);
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].role, ViewRole::Tool);
        assert!(state.messages[0].running);
        assert_eq!(state.current_assistant, None);
        assert_eq!(state.generation_start, None);

        let mut resumed_state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        resumed_state.push_notice("已恢复的历史");
        resumed_state.messages.push(ViewMessage {
            role: ViewRole::Tool,
            title: "shell".to_string(),
            content: String::new(),
            reasoning: String::new(),
            tool_arguments: Some("cargo check".to_string()),
            tool_id: Some("tool-1".to_string()),
            running: false,
        });

        resumed_state.hold_pending_tools(&["tool-1".to_string()]);
        let finalized = resumed_state.take_finalized_messages(80, 3);
        assert_eq!(finalized.len(), 1);
        assert_eq!(resumed_state.messages.len(), 1);
        assert!(resumed_state.messages[0].running);

        resumed_state.apply_agent_event(AgentEvent::ToolFinished {
            id: "tool-1".to_string(),
            name: "shell".to_string(),
            output: "完成".to_string(),
            is_error: false,
        });
        assert!(resumed_state.take_finalized_messages(80, 1).is_empty());
        assert_eq!(resumed_state.messages.len(), 1);
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
}
