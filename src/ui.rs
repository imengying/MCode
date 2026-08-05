use std::collections::VecDeque;
use std::io::{self, Read as _, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{
    self, DisableBracketedPaste, DisableFocusChange, EnableBracketedPaste, EnableFocusChange,
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    Clear, ClearType as TerminalClearType, disable_raw_mode, enable_raw_mode,
};
use pulldown_cmark::{
    CodeBlockKind, Event as MarkdownEvent, HeadingLevel, Options as MarkdownOptions, Parser, Tag,
    TagEnd,
};
use ratatui::backend::{Backend, ClearType as BackendClearType, CrosstermBackend, WindowSize};
use ratatui::buffer::Cell;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Position, Rect, Size};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};
use ratatui::{Frame, Terminal, TerminalOptions, Viewport};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
use wl_clipboard_rs::paste::{ClipboardType, MimeType, Seat, get_contents, get_mime_types};

use crate::agent::{Agent, ModelChoice};
use crate::approval::{ApprovalDecision, ApprovalGate, ApprovalRequest, format_tool_arguments};
use crate::compaction::{estimate_message_tokens, estimate_text_tokens};
use crate::config::{ApiProtocol, ReasoningEffort};
use crate::event::{AgentEvent, CompactionReason};
use crate::highlight::highlight_code;
use crate::protocol::{
    ChatMessage, FileChangeKind, FileChangeLineKind, FileChangeSummary, ImageAttachment,
    MAX_IMAGE_BYTES, MessageRole, Usage, sanitize_terminal_text,
};

const APPROVAL_HEIGHT: u16 = 6;
const DELETE_CONFIRMATION_HEIGHT: u16 = 5;
const COLLAPSED_PASTE_CHAR_THRESHOLD: usize = 1_000;
const COLLAPSED_PASTE_LINE_THRESHOLD: usize = 8;
const MIN_TERMINAL_HEIGHT: u16 = 10;
const INPUT_PREFIX_WIDTH: u16 = 2;
const INPUT_FOOTER_GAP: u16 = 1;
const INPUT_PLACEHOLDER: &str = "描述任务，或输入 / 查看命令";
const MAX_INPUT_HEIGHT: u16 = 5;
const MAX_INPUT_HISTORY: usize = 100;
const MAX_QUEUED_SUBMISSIONS: usize = 8;
const MAX_SLASH_SUGGESTIONS: u16 = 8;
const PREVIEW_LINE_CHARS: usize = 240;
const TOOL_ARGUMENT_PREVIEW_LINES: usize = 2;
const TOOL_OUTPUT_PREVIEW_LINES: usize = 5;
const X11_CLIPBOARD_TIMEOUT: Duration = Duration::from_millis(500);
const FRAME_INTERVAL: Duration = Duration::from_millis(50);
const ELAPSED_REFRESH_INTERVAL: Duration = Duration::from_secs(1);
const QUIT_SHORTCUT_TIMEOUT: Duration = Duration::from_secs(1);
// Match Codex's terminal-native palette: inherit the configured foreground/background and use
// ANSI semantic colors so accents remain legible across terminal themes.
const THEME_BASE: Color = Color::Reset;
const THEME_MANTLE: Color = Color::Reset;
const THEME_SURFACE: Color = Color::DarkGray;
const THEME_TEXT: Color = Color::Reset;
const THEME_SUBTEXT: Color = Color::Gray;
const THEME_MUTED: Color = Color::DarkGray;
const THEME_BLUE: Color = Color::Cyan;
const THEME_GREEN: Color = Color::Green;
const THEME_YELLOW: Color = Color::Yellow;
const THEME_RED: Color = Color::Red;
const THEME_DIFF_ADD_BG: Color = Color::Rgb(33, 58, 43);
const THEME_DIFF_REMOVE_BG: Color = Color::Rgb(74, 34, 29);
const THEME_MAUVE: Color = Color::Magenta;
const THEME_TEAL: Color = Color::Cyan;
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
    let mut resume_candidate = agent
        .session()
        .persistence_path()?
        .map(|path| (agent.session().id(), path));
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
    let viewport_height = backend.size().context("读取终端尺寸失败")?.height;
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(viewport_height),
        },
    )
    .context("初始化终端失败")?;
    clear_terminal_view(&mut terminal)?;

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
        if state.expire_quit_shortcut() {
            needs_draw = true;
        }
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
        if !state.running
            && pending_approval.is_none()
            && let Some(queued) = state.queued_submissions.pop_front()
        {
            needs_draw = true;
            start_run(
                Arc::clone(&agent),
                queued.prompt,
                queued.images,
                &event_tx,
                approvals.clone(),
                &mut state,
                &mut active_cancel,
            );
        }

        if needs_draw {
            archive_transcript_overflow(&mut terminal, &mut state)?;
        }
        if state.run_started_at.is_some() && last_frame.elapsed() >= ELAPSED_REFRESH_INTERVAL {
            needs_draw = true;
        }
        if needs_draw && last_frame.elapsed() >= FRAME_INTERVAL {
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
                    UiAction::ListModels => match agent.try_lock() {
                        Ok(mut agent) => match agent.refresh_model_profiles() {
                            Ok(()) => {
                                state.sync_from_agent(&agent);
                                let notice = state.model_list_notice();
                                state.push_notice(notice);
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state.push_error("Agent 正忙，请等待当前任务完成。"),
                    },
                    UiAction::SelectModel(query) => match agent.try_lock() {
                        Ok(mut agent) => match agent
                            .refresh_model_profiles()
                            .and_then(|()| agent.select_model(&query))
                        {
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
                                resume_candidate = agent
                                    .session()
                                    .persistence_path()?
                                    .map(|path| (agent.session().id(), path));
                                state.reset_session();
                                state.sync_from_agent(&agent);
                                clear_terminal_view(&mut terminal)?;
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
                        clear_terminal_view(&mut terminal)?;
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
            Event::FocusGained => needs_draw = true,
            Event::Resize(..) => {
                needs_draw = true;
            }
            _ => {}
        }
    }

    clear_terminal_view(&mut terminal)?;
    drop(terminal);
    drop(screen);
    if let Some(id) = deleted_session {
        println!("已删除会话 {id}。");
    } else if let Some((id, _)) =
        resume_candidate.filter(|(_, path)| session_path_is_resumable(path))
    {
        println!("继续此会话：mcode resume {id}");
    }
    Ok(())
}

fn session_path_is_resumable(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
}

// Some PTYs do not answer cursor-position queries; retain Ratatui's last position as a fallback.
struct UiBackend<W: Write> {
    inner: CrosstermBackend<W>,
    cursor: Option<Position>,
}

impl<W: Write> UiBackend<W> {
    fn new(writer: W) -> Self {
        Self {
            inner: CrosstermBackend::new(writer),
            cursor: None,
        }
    }
}

impl<W: Write> Backend for UiBackend<W> {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        let mut positions = DrawPositionAdapter::default();
        self.inner.draw(content.filter_map(|(x, y, cell)| {
            positions
                .map(x, y, cell.symbol())
                .map(|position| (position.x, position.y, cell))
        }))
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

type UiTerminal = Terminal<UiBackend<io::Stdout>>;

fn clear_terminal_view(terminal: &mut UiTerminal) -> Result<()> {
    execute!(
        io::stdout(),
        MoveTo(0, 0),
        Clear(TerminalClearType::All),
        Clear(TerminalClearType::Purge),
        MoveTo(0, 0)
    )
    .context("清空终端滚动历史失败")?;
    terminal.backend_mut().cursor = Some(Position::ORIGIN);
    terminal.clear().context("重置终端视口失败")
}

// Normal diffs omit wide-glyph trailing cells, but history insertion can yield them. Skip those
// cells and report adjacent virtual positions so Crossterm neither inserts spaces nor repositions.
#[derive(Debug, Default)]
struct DrawPositionAdapter {
    actual_end: Option<Position>,
    reported: Option<Position>,
}

impl DrawPositionAdapter {
    fn map(&mut self, x: u16, y: u16, symbol: &str) -> Option<Position> {
        let actual = Position::new(x, y);
        if self
            .actual_end
            .is_some_and(|end| end.y == y && actual.x < end.x)
        {
            return None;
        }
        let reported = if self.actual_end == Some(actual) {
            Position::new(
                self.reported
                    .map_or(x, |position| position.x.saturating_add(1)),
                y,
            )
        } else {
            actual
        };
        let width = u16::try_from(UnicodeWidthStr::width(symbol).max(1)).unwrap_or(u16::MAX);
        self.actual_end = Some(Position::new(x.saturating_add(width), y));
        self.reported = Some(reported);
        Some(reported)
    }
}

fn archive_transcript_overflow(terminal: &mut UiTerminal, state: &mut UiState) -> Result<()> {
    terminal.autoresize().context("调整终端视口失败")?;
    let size = terminal.size().context("读取终端尺寸失败")?;
    let conversation_height =
        usize::from(ui_section_heights(state, size.width, size.height).conversation);
    let width = size.width.max(1);
    let count = transcript_archive_count(state, width, conversation_height);
    if count == 0 {
        return Ok(());
    }

    let lines = conversation_lines_for_messages(&state.messages[..count]);
    if !lines.is_empty() {
        let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
        let height = paragraph.line_count(width);
        let height = u16::try_from(height).context("单条终端历史记录过长")?;
        terminal
            .insert_before(height, move |buffer| {
                paragraph.render(buffer.area, buffer);
            })
            .context("写入终端滚动历史失败")?;
    }

    state.messages.drain(..count);
    state.current_assistant = state
        .current_assistant
        .and_then(|index| index.checked_sub(count));
    state.generation_start = state
        .generation_start
        .and_then(|index| index.checked_sub(count));
    state.protected_turn_start = state
        .protected_turn_start
        .and_then(|index| index.checked_sub(count));
    Ok(())
}

fn transcript_archive_count(state: &UiState, width: u16, conversation_height: usize) -> usize {
    if transcript_line_count(state, &state.messages, width) <= conversation_height {
        return 0;
    }
    let archivable = state
        .messages
        .iter()
        .take_while(|message| !message.running)
        .count()
        .min(state.protected_turn_start.unwrap_or(usize::MAX));
    let mut count = 0;
    while count < archivable
        && transcript_line_count(state, &state.messages[count..], width) > conversation_height
    {
        count += 1;
    }
    count
}

fn start_resume(
    agent: Arc<Mutex<Agent>>,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    approvals: ApprovalGate,
    state: &mut UiState,
    active_cancel: &mut Option<CancellationToken>,
) {
    state.protect_resumed_turn();
    state.begin_live_usage(state.context_tokens);
    state.begin_run("正在恢复中断任务");
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
    let prompt_tokens = estimate_message_tokens(&ChatMessage::user_with_images(
        prompt.clone(),
        images.clone(),
    ));
    state.protect_new_turn();
    state.begin_live_usage(state.context_tokens.saturating_add(prompt_tokens));
    state.push_user(prompt.clone(), &images);
    state.begin_run("处理中");
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
    state.begin_live_usage(state.context_tokens);
    state.begin_run("正在压缩上下文");
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
    ListModels,
    SelectModel(String),
    SetReasoning(ReasoningEffort),
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
    if state.dismissed_slash_input.as_deref() == Some(text.as_str()) {
        return Vec::new();
    }
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
                let current = choice.id == state.model && state.provider == choice.provider;
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
    state.dismissed_slash_input = None;
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

    let is_ctrl_c = matches!(key.code, KeyCode::Char('c' | 'C'))
        && key.modifiers.contains(KeyModifiers::CONTROL);
    if !is_ctrl_c {
        state.clear_quit_shortcut();
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

    if key.code == KeyCode::Esc && !slash_suggestions(state).is_empty() {
        state.dismissed_slash_input = Some(state.editor.text());
        state.slash_selection = 0;
        return UiAction::None;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('c' | 'C') => {
                if !state.editor.is_empty() || !state.pending_images.is_empty() {
                    let draft = state.editor.content();
                    state.record_input(draft);
                    state.editor.clear();
                    state.pending_images.clear();
                    state.prepare_input_edit();
                    state.arm_quit_shortcut();
                    return UiAction::None;
                }
                if state.quit_shortcut_active() {
                    return UiAction::Quit;
                }
                state.arm_quit_shortcut();
                if state.running
                    && let Some(cancel) = active_cancel
                {
                    cancel.cancel();
                    state.status = "正在取消".to_string();
                }
                return UiAction::None;
            }
            KeyCode::Char('d')
                if state.editor.is_empty() && state.pending_images.is_empty() && !state.running =>
            {
                return UiAction::Quit;
            }
            KeyCode::Char('a') => state.editor.move_home(),
            KeyCode::Char('b') => state.editor.move_left(),
            KeyCode::Char('e') => state.editor.move_end(),
            KeyCode::Char('f') => state.editor.move_right(),
            KeyCode::Char('j' | 'm') => {
                state.prepare_input_edit();
                state.editor.insert('\n');
            }
            KeyCode::Char('h') => {
                state.prepare_input_edit();
                state.editor.backspace();
            }
            KeyCode::Char('k') => {
                state.prepare_input_edit();
                state.editor.kill_line_end();
            }
            KeyCode::Char('n') => state.next_input(),
            KeyCode::Char('p') => state.previous_input(),
            KeyCode::Char('u') => {
                state.prepare_input_edit();
                state.editor.kill_line_start();
            }
            KeyCode::Char('v') => return UiAction::PasteClipboard,
            KeyCode::Char('w') | KeyCode::Backspace => {
                state.prepare_input_edit();
                state.editor.delete_backward_word();
            }
            KeyCode::Char('y') => {
                state.prepare_input_edit();
                state.editor.yank();
            }
            KeyCode::Char('d') => {
                state.prepare_input_edit();
                state.editor.delete();
            }
            KeyCode::Left => state.editor.move_word_left(),
            KeyCode::Right => state.editor.move_word_right(),
            KeyCode::Delete => {
                state.prepare_input_edit();
                state.editor.delete_forward_word();
            }
            _ => return UiAction::None,
        }
        return UiAction::None;
    }

    if key.modifiers.contains(KeyModifiers::ALT) {
        match key.code {
            KeyCode::Enter => {
                state.prepare_input_edit();
                state.editor.insert('\n');
            }
            KeyCode::Char('b') | KeyCode::Left => state.editor.move_word_left(),
            KeyCode::Char('f') | KeyCode::Right => state.editor.move_word_right(),
            KeyCode::Char('d') | KeyCode::Delete => {
                state.prepare_input_edit();
                state.editor.delete_forward_word();
            }
            KeyCode::Backspace => {
                state.prepare_input_edit();
                state.editor.delete_backward_word();
            }
            _ => return UiAction::None,
        }
        return UiAction::None;
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
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
            state.prepare_input_edit();
            state.editor.insert('\n');
        }
        KeyCode::Enter if !state.running => {
            return submit_editor(state);
        }
        KeyCode::Enter => queue_editor(state),
        KeyCode::Tab if state.running => queue_editor(state),
        KeyCode::Tab => return submit_editor(state),
        KeyCode::Char(character)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            state.prepare_input_edit();
            state.editor.insert(character);
        }
        KeyCode::Backspace => {
            if state.editor.is_empty() {
                if state.pending_images.pop().is_some() {
                    state.status = attachment_status(state.pending_images.len());
                }
            } else {
                state.prepare_input_edit();
                state.editor.backspace();
            }
            state.slash_selection = 0;
            state.dismissed_slash_input = None;
        }
        KeyCode::Delete => {
            state.prepare_input_edit();
            state.editor.delete();
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

fn queue_editor(state: &mut UiState) {
    if state.queued_submissions.len() >= MAX_QUEUED_SUBMISSIONS {
        state.push_error("最多排队 8 条消息。");
        return;
    }
    let prompt = state.editor.content();
    if prompt.trim().is_empty() || prompt.trim_start().starts_with('/') {
        return;
    }
    let prompt = state.editor.take();
    state.record_input(prompt.clone());
    let images = state.take_pending_images();
    state
        .queued_submissions
        .push_back(QueuedSubmission { prompt, images });
    state.slash_selection = 0;
    state.dismissed_slash_input = None;
}

fn submit_editor(state: &mut UiState) -> UiAction {
    state.slash_selection = 0;
    state.dismissed_slash_input = None;
    state.clear_quit_shortcut();
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
        "model" if argument.is_empty() => UiAction::ListModels,
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
        "status" => {
            let notice = state.status_notice();
            state.push_notice(notice);
            UiAction::None
        }
        "help" => {
            state.push_notice(
                "命令：/model [ID]、/effort [级别]、/compact [说明]、/status、/new、/delete、/clear、/help、/exit",
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
    let content = match read_wayland_clipboard() {
        Ok(content) => content,
        Err(wayland_error) => match read_x11_clipboard(state) {
            Ok(content) => content,
            Err(x11_error) => {
                state.push_error(format!(
                    "无法读取剪贴板：Wayland：{wayland_error:#}；X11：{x11_error:#}"
                ));
                return;
            }
        },
    };
    match content {
        ClipboardContent::Image { extension, bytes } => {
            let name = format!("clipboard-{}.{}", state.pending_images.len() + 1, extension);
            match ImageAttachment::from_encoded_bytes(name, bytes) {
                Ok(image) => attach_image(state, image),
                Err(error) => state.push_error(format!("无法附加剪贴板图片：{error:#}")),
            }
        }
        ClipboardContent::Text(text) => paste_text_or_image(state, &text),
    }
}

enum ClipboardContent {
    Image {
        extension: &'static str,
        bytes: Vec<u8>,
    },
    Text(String),
}

fn read_wayland_clipboard() -> Result<ClipboardContent> {
    let clipboard = ClipboardType::Regular;
    let seat = Seat::Unspecified;
    let mime_types = get_mime_types(clipboard, seat).context("无法访问 Wayland 剪贴板")?;

    if let Some(&(mime_type, extension)) = CLIPBOARD_IMAGE_TYPES
        .iter()
        .find(|(mime_type, _)| mime_types.contains(*mime_type))
    {
        let (reader, _) = get_contents(clipboard, seat, MimeType::Specific(mime_type))
            .context("无法读取 Wayland 剪贴板图片")?;
        let bytes = read_limited(reader, MAX_IMAGE_BYTES).context("无法读取 Wayland 剪贴板图片")?;
        return Ok(ClipboardContent::Image { extension, bytes });
    }

    let (reader, _) = get_contents(clipboard, seat, MimeType::Text)
        .context("Wayland 剪贴板中没有可粘贴的图片或文本")?;
    let text = read_limited(reader, MAX_IMAGE_BYTES).and_then(|bytes| {
        String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    })?;
    if text.is_empty() {
        bail!("Wayland 剪贴板为空");
    }
    Ok(ClipboardContent::Text(text))
}

fn read_x11_clipboard(state: &mut UiState) -> Result<ClipboardContent> {
    let clipboard = x11_clipboard(state)?;
    let selection = clipboard.getter.atoms.clipboard;
    let property = clipboard.getter.atoms.property;
    let mut last_error = None;

    for &(mime_type, extension) in &CLIPBOARD_IMAGE_TYPES {
        let target = clipboard
            .getter
            .get_atom(mime_type)
            .with_context(|| format!("无法查询 X11 剪贴板格式 {mime_type}"))?;
        match clipboard.load(selection, target, property, X11_CLIPBOARD_TIMEOUT) {
            Ok(bytes) if !bytes.is_empty() => {
                ensure_clipboard_size(&bytes)?;
                return Ok(ClipboardContent::Image { extension, bytes });
            }
            Ok(_) => {}
            Err(error) => last_error = Some(error.to_string()),
        }
    }

    for target in [
        clipboard.getter.atoms.utf8_string,
        clipboard.getter.atoms.string,
    ] {
        match clipboard.load(selection, target, property, X11_CLIPBOARD_TIMEOUT) {
            Ok(bytes) if !bytes.is_empty() => {
                ensure_clipboard_size(&bytes)?;
                let text = String::from_utf8_lossy(&bytes).into_owned();
                return Ok(ClipboardContent::Text(text));
            }
            Ok(_) => {}
            Err(error) => last_error = Some(error.to_string()),
        }
    }

    if let Some(error) = last_error {
        bail!("X11 剪贴板中没有可粘贴的图片或文本：{error}");
    }
    bail!("X11 剪贴板为空");
}

fn ensure_clipboard_size(bytes: &[u8]) -> Result<()> {
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_IMAGE_BYTES {
        bail!("剪贴板内容超过 20 MiB 限制");
    }
    Ok(())
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
    state.dismissed_slash_input = None;
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

#[derive(Debug, Clone)]
enum EditorItem {
    Character(char),
    Paste { content: String, label: String },
}

#[derive(Debug, Default)]
struct Editor {
    items: Vec<EditorItem>,
    cursor: usize,
    kill_buffer: Vec<EditorItem>,
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

    fn clear(&mut self) {
        self.items.clear();
        self.cursor = 0;
    }

    fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    fn move_right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.items.len());
    }

    fn move_word_left(&mut self) {
        while self.cursor > 0 && Self::is_whitespace(&self.items[self.cursor - 1]) {
            self.cursor -= 1;
        }
        while self.cursor > 0 && !Self::is_whitespace(&self.items[self.cursor - 1]) {
            self.cursor -= 1;
        }
    }

    fn move_word_right(&mut self) {
        while self.cursor < self.items.len() && Self::is_whitespace(&self.items[self.cursor]) {
            self.cursor += 1;
        }
        while self.cursor < self.items.len() && !Self::is_whitespace(&self.items[self.cursor]) {
            self.cursor += 1;
        }
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

    fn delete_backward_word(&mut self) {
        let end = self.cursor;
        self.move_word_left();
        self.kill_range(self.cursor, end);
    }

    fn delete_forward_word(&mut self) {
        let start = self.cursor;
        self.move_word_right();
        let end = self.cursor;
        self.cursor = start;
        self.kill_range(start, end);
    }

    fn kill_line_start(&mut self) {
        let end = self.cursor;
        self.move_home();
        self.kill_range(self.cursor, end);
    }

    fn kill_line_end(&mut self) {
        let start = self.cursor;
        self.move_end();
        let mut end = self.cursor;
        if end == start
            && self
                .items
                .get(end)
                .is_some_and(|item| matches!(item, EditorItem::Character('\n')))
        {
            end += 1;
        }
        self.cursor = start;
        self.kill_range(start, end);
    }

    fn yank(&mut self) {
        if self.kill_buffer.is_empty() {
            return;
        }
        let count = self.kill_buffer.len();
        self.items
            .splice(self.cursor..self.cursor, self.kill_buffer.clone());
        self.cursor += count;
    }

    fn kill_range(&mut self, start: usize, end: usize) {
        if start >= end || end > self.items.len() {
            self.cursor = start.min(self.items.len());
            return;
        }
        self.kill_buffer = self.items.drain(start..end).collect();
        self.cursor = start;
    }

    fn is_whitespace(item: &EditorItem) -> bool {
        matches!(item, EditorItem::Character(character) if character.is_whitespace())
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
    RunCompleted,
    RunCancelled,
    RunFailed,
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
    file_change: Option<FileChangeSummary>,
    running: bool,
}

#[derive(Debug)]
struct QueuedSubmission {
    prompt: String,
    images: Vec<ImageAttachment>,
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

struct X11Clipboard(x11_clipboard::Clipboard);

impl std::fmt::Debug for X11Clipboard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("X11Clipboard(..)")
    }
}

#[derive(Debug)]
struct UiState {
    model: String,
    provider: String,
    api: ApiProtocol,
    reasoning_effort: ReasoningEffort,
    default_reasoning_effort: ReasoningEffort,
    endpoint: String,
    cwd: std::path::PathBuf,
    model_choices: Vec<ModelChoice>,
    reasoning_choices: Vec<ReasoningEffort>,
    messages: Vec<ViewMessage>,
    editor: Editor,
    running: bool,
    run_started_at: Option<Instant>,
    run_elapsed_before_pause: Duration,
    current_assistant: Option<usize>,
    generation_start: Option<usize>,
    protected_turn_start: Option<usize>,
    status: String,
    usage: Usage,
    live_prompt_tokens: u64,
    live_completion: String,
    context_tokens: u64,
    context_window: u64,
    max_input_tokens: u64,
    usage_estimated: bool,
    delete_confirmation: DeleteConfirmation,
    pending_images: Vec<ImageAttachment>,
    queued_submissions: VecDeque<QueuedSubmission>,
    slash_selection: usize,
    dismissed_slash_input: Option<String>,
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
    quit_shortcut_expires_at: Option<Instant>,
    show_welcome: bool,
    x11_clipboard: Option<X11Clipboard>,
}

impl UiState {
    fn new(model: String, endpoint: String, cwd: std::path::PathBuf) -> Self {
        Self {
            model,
            provider: "xai".to_string(),
            api: ApiProtocol::ChatCompletions,
            reasoning_effort: ReasoningEffort::Off,
            default_reasoning_effort: ReasoningEffort::Off,
            endpoint,
            cwd,
            model_choices: Vec::new(),
            reasoning_choices: ReasoningEffort::ALL.to_vec(),
            messages: Vec::new(),
            editor: Editor::default(),
            running: false,
            run_started_at: None,
            run_elapsed_before_pause: Duration::ZERO,
            current_assistant: None,
            generation_start: None,
            protected_turn_start: None,
            status: "就绪".to_string(),
            usage: Usage::default(),
            live_prompt_tokens: 0,
            live_completion: String::new(),
            context_tokens: 0,
            context_window: 128_000,
            max_input_tokens: 128_000,
            usage_estimated: false,
            delete_confirmation: DeleteConfirmation::None,
            pending_images: Vec::new(),
            queued_submissions: VecDeque::new(),
            slash_selection: 0,
            dismissed_slash_input: None,
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
            quit_shortcut_expires_at: None,
            show_welcome: true,
            x11_clipboard: None,
        }
    }

    fn sync_from_agent(&mut self, agent: &Agent) {
        self.model = sanitize_terminal_text(agent.model());
        self.provider = sanitize_terminal_text(agent.provider());
        self.api = agent.api();
        self.reasoning_effort = agent.reasoning_effort();
        self.default_reasoning_effort = agent.default_reasoning_effort();
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

    fn begin_run(&mut self, status: impl Into<String>) {
        self.quit_shortcut_expires_at = None;
        if !self.running {
            self.run_elapsed_before_pause = Duration::ZERO;
            self.run_started_at = Some(Instant::now());
        }
        self.running = true;
        self.status = status.into();
    }

    fn protect_new_turn(&mut self) {
        self.protected_turn_start = Some(self.messages.len());
    }

    fn protect_resumed_turn(&mut self) {
        self.protected_turn_start = self
            .messages
            .iter()
            .rposition(|message| message.role == ViewRole::User);
    }

    fn begin_live_usage(&mut self, prompt_tokens: u64) {
        self.live_prompt_tokens = prompt_tokens;
        self.live_completion.clear();
    }

    fn append_live_completion(&mut self, text: &str) {
        self.live_completion.push_str(text);
    }

    fn clear_live_usage(&mut self) {
        self.live_prompt_tokens = 0;
        self.live_completion.clear();
    }

    fn live_completion_tokens(&self) -> u64 {
        estimate_text_tokens(&self.live_completion)
    }

    fn displayed_usage(&self) -> Usage {
        let completion_tokens = self.live_completion_tokens();
        self.usage.saturating_add(Usage {
            prompt_tokens: self.live_prompt_tokens,
            completion_tokens,
            total_tokens: self.live_prompt_tokens.saturating_add(completion_tokens),
            cached_prompt_tokens: None,
        })
    }

    fn displayed_context_tokens(&self) -> u64 {
        if self.live_prompt_tokens == 0 {
            self.context_tokens
        } else {
            self.live_prompt_tokens
                .saturating_add(self.live_completion_tokens())
        }
    }

    fn displayed_usage_is_estimated(&self) -> bool {
        self.usage_estimated || self.live_prompt_tokens > 0 || !self.live_completion.is_empty()
    }

    fn run_elapsed(&self) -> Duration {
        self.run_elapsed_before_pause.saturating_add(
            self.run_started_at
                .map_or(Duration::ZERO, |started_at| started_at.elapsed()),
        )
    }

    fn pause_run_timer(&mut self) {
        if let Some(started_at) = self.run_started_at.take() {
            self.run_elapsed_before_pause = self
                .run_elapsed_before_pause
                .saturating_add(started_at.elapsed());
        }
    }

    fn resume_run_timer(&mut self) {
        if self.running && self.run_started_at.is_none() {
            self.run_started_at = Some(Instant::now());
        }
    }

    fn activity_label(&self) -> Option<String> {
        if !self.running || self.pending_approval.is_some() {
            return None;
        }
        let activity = self
            .reasoning_activity()
            .or_else(|| (!self.status.is_empty()).then(|| self.status.clone()))?;
        if self.queued_submissions.is_empty() {
            Some(activity)
        } else {
            Some(format!(
                "{activity} · 已排队 {}",
                self.queued_submissions.len()
            ))
        }
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
                    file_change: None,
                    running: false,
                });
            }
            MessageRole::Assistant => {
                let reasoning = if self.api == ApiProtocol::Responses {
                    let mut parts = response_reasoning_summary_parts(&message.response_items);
                    if parts.is_empty()
                        && !has_response_reasoning_item(&message.response_items)
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
                        file_change: None,
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
                        file_change: None,
                        running: false,
                    });
                }
            }
            MessageRole::Tool => {
                let tool_id = message.tool_call_id;
                let file_change = message.file_change;
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
                    existing.file_change = file_change;
                } else {
                    self.messages.push(ViewMessage {
                        role: ViewRole::Tool,
                        title: "工具".to_string(),
                        content,
                        reasoning: String::new(),
                        tool_arguments: None,
                        tool_id,
                        file_change,
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
            file_change: None,
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
            file_change: None,
            running: false,
        });
    }

    fn push_error(&mut self, content: impl AsRef<str>) {
        self.messages.push(ViewMessage {
            role: ViewRole::Error,
            title: "错误".to_string(),
            content: truncate_error_for_ui(&sanitize_terminal_text(content.as_ref())),
            reasoning: String::new(),
            tool_arguments: None,
            tool_id: None,
            file_change: None,
            running: false,
        });
    }

    fn set_pending_approval(&mut self, request: &ApprovalRequest) {
        self.pause_run_timer();
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
        self.resume_run_timer();
    }

    fn reset_session(&mut self) {
        self.messages.clear();
        self.run_started_at = None;
        self.run_elapsed_before_pause = Duration::ZERO;
        self.current_assistant = None;
        self.generation_start = None;
        self.protected_turn_start = None;
        self.usage = Usage::default();
        self.clear_live_usage();
        self.context_tokens = 0;
        self.usage_estimated = false;
        self.status = "就绪".to_string();
        self.delete_confirmation = DeleteConfirmation::None;
        self.pending_images.clear();
        self.queued_submissions.clear();
        self.dismissed_slash_input = None;
        self.pending_approval = None;
        self.quit_shortcut_expires_at = None;
        self.show_welcome = true;
        self.input_history.clear();
        self.detach_input_history();
        self.reset_reasoning_summary();
    }

    fn clear_view(&mut self) {
        self.messages.clear();
        self.current_assistant = None;
        self.generation_start = None;
        self.protected_turn_start = None;
        self.delete_confirmation = DeleteConfirmation::None;
        self.reset_reasoning_summary();
    }

    fn arm_quit_shortcut(&mut self) {
        self.quit_shortcut_expires_at = Instant::now().checked_add(QUIT_SHORTCUT_TIMEOUT);
    }

    fn quit_shortcut_active(&self) -> bool {
        self.quit_shortcut_expires_at
            .is_some_and(|expires_at| Instant::now() < expires_at)
    }

    fn clear_quit_shortcut(&mut self) {
        self.quit_shortcut_expires_at = None;
    }

    fn expire_quit_shortcut(&mut self) -> bool {
        if self
            .quit_shortcut_expires_at
            .is_some_and(|expires_at| Instant::now() >= expires_at)
        {
            self.quit_shortcut_expires_at = None;
            true
        } else {
            false
        }
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
        self.dismissed_slash_input = None;
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
        self.dismissed_slash_input = None;
    }

    fn detach_input_history(&mut self) {
        self.input_history_index = None;
        self.input_history_draft = None;
    }

    fn prepare_input_edit(&mut self) {
        self.detach_input_history();
        self.slash_selection = 0;
        self.dismissed_slash_input = None;
    }

    fn model_list_notice(&self) -> String {
        if self.model_choices.is_empty() {
            return format!(
                "当前模型：{}\n~/.mcode/models.json 中没有可用模型。",
                self.model
            );
        }
        let mut lines = vec!["已配置的模型：".to_string()];
        for choice in &self.model_choices {
            let selected = if choice.id == self.model && self.provider == choice.provider {
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
        format!("{}/{}", self.provider, self.model)
    }

    fn status_notice(&self) -> String {
        let qualified_model = self.qualified_model();
        let usage = self.displayed_usage();
        let context_tokens = self.displayed_context_tokens();
        let estimate = if self.displayed_usage_is_estimated() {
            "~"
        } else {
            ""
        };
        let percent = format_context_percent(context_tokens, self.max_input_tokens);
        let cache = self
            .usage
            .cached_prompt_tokens
            .map_or_else(String::new, |cached| {
                format!(
                    "\n缓存命中：{}（输入的 {}%）",
                    format_tokens(cached),
                    format_context_percent(cached, usage.prompt_tokens)
                )
            });
        format!(
            "模型：{qualified_model}\nAPI：{}\neffort：{}\n网页搜索：原生开启\n输入：{estimate}{}/{}（{percent}%）\n模型上下文窗口：{}\nToken：{estimate}输入 {}，输出 {}{cache}\nMCP：{} 个服务器，{} 个工具\n端点：{}\n工作目录：{}",
            self.api,
            self.reasoning_effort,
            format_tokens(context_tokens),
            format_tokens(self.max_input_tokens),
            format_tokens(self.context_window),
            format_tokens(usage.prompt_tokens),
            format_tokens(usage.completion_tokens),
            self.mcp_server_count,
            self.mcp_tool_count,
            self.endpoint,
            sanitize_terminal_text(&self.cwd.to_string_lossy())
        )
    }

    fn apply_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::RunStarted | AgentEvent::RunResumed => {
                self.begin_run("处理中");
            }
            AgentEvent::AssistantStarted => {
                if self.live_prompt_tokens == 0 {
                    self.begin_live_usage(self.context_tokens);
                } else {
                    self.live_completion.clear();
                }
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
                self.live_completion.clear();
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
                let text = sanitize_terminal_text(&text);
                self.append_live_completion(&text);
                let index = self.ensure_assistant_message();
                if let Some(message) = self.messages.get_mut(index) {
                    message.content.push_str(&text);
                }
                self.status = "正在生成回复".to_string();
            }
            AgentEvent::ReasoningSummaryDelta { text } => {
                if self.api == ApiProtocol::Responses {
                    self.append_live_completion(&text);
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
            AgentEvent::ResponseTruncated { had_tool_calls } => self.push_error(if had_tool_calls {
                "模型输出达到上限，工具参数可能不完整，未执行。"
            } else {
                "模型输出达到上限，回复可能不完整。"
            }),
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
                    file_change: None,
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
                    message.file_change = None;
                    message.running = true;
                } else {
                    self.messages.push(ViewMessage {
                        role: ViewRole::Tool,
                        title: name,
                        content: String::new(),
                        reasoning: String::new(),
                        tool_arguments: Some(arguments),
                        tool_id: Some(id),
                        file_change: None,
                        running: true,
                    });
                }
            }
            AgentEvent::ToolOutputDelta { id, delta } => {
                if let Some(message) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|message| message.tool_id.as_deref() == Some(id.as_str()))
                {
                    message.content.push_str(&sanitize_terminal_text(&delta));
                    trim_live_tool_output(&mut message.content);
                }
            }
            AgentEvent::ToolFinished {
                id,
                name,
                output,
                is_error,
                file_change,
            } => {
                if let Some(message) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|message| message.tool_id.as_deref() == Some(id.as_str()))
                {
                    message.title = sanitize_terminal_text(&name);
                    message.content = truncate_for_ui(&sanitize_terminal_text(&output));
                    message.file_change = file_change;
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
                    file_change: None,
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
                self.usage = self.usage.saturating_add(usage);
                self.context_tokens = context_tokens;
                self.context_window = context_window;
                self.max_input_tokens = max_input_tokens;
                self.usage_estimated = estimated;
                self.clear_live_usage();
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
                    self.usage = self.usage.saturating_add(usage);
                }
                self.context_tokens = tokens_after;
                self.usage_estimated = true;
                self.push_notice(format!(
                    "上下文已压缩（预计 token：{} -> {}）。\n\n{summary}",
                    format_tokens(tokens_before),
                    format_tokens(tokens_after)
                ));
                if reason == CompactionReason::Manual {
                    let elapsed = self.finish_run();
                    self.push_run_summary(ViewRole::RunCompleted, elapsed);
                } else {
                    self.status = "处理中".to_string();
                }
            }
            AgentEvent::CompactionFailed { reason, message } => {
                if reason == CompactionReason::Manual {
                    let elapsed = self.finish_run();
                    self.push_error(format!("上下文压缩失败：{message}"));
                    self.push_run_summary(ViewRole::RunFailed, elapsed);
                } else {
                    self.status = "处理中".to_string();
                    self.push_error(format!(
                        "自动压缩失败，已回退为硬裁剪上下文：{message}"
                    ));
                }
            }
            AgentEvent::RunFinished => {
                self.finish_reasoning_summary();
                let elapsed = self.finish_run();
                self.push_run_summary(ViewRole::RunCompleted, elapsed);
            }
            AgentEvent::Cancelled => {
                self.reset_reasoning_summary();
                let elapsed = self.finish_run();
                self.push_run_summary(ViewRole::RunCancelled, elapsed);
            }
            AgentEvent::Error { message } => {
                self.reset_reasoning_summary();
                let elapsed = self.finish_run();
                self.messages.push(ViewMessage {
                    role: ViewRole::Error,
                    title: "错误".to_string(),
                    content: truncate_error_for_ui(&sanitize_terminal_text(&message)),
                    reasoning: String::new(),
                    tool_arguments: None,
                    tool_id: None,
                    file_change: None,
                    running: false,
                });
                self.push_run_summary(ViewRole::RunFailed, elapsed);
            }
        }
    }

    fn finish_run(&mut self) -> Duration {
        let elapsed = self.run_elapsed();
        self.run_started_at = None;
        self.run_elapsed_before_pause = Duration::ZERO;
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
        self.pending_approval = None;
        self.clear_live_usage();
        self.reset_reasoning_summary();
        elapsed
    }

    fn push_run_summary(&mut self, role: ViewRole, elapsed: Duration) {
        debug_assert!(matches!(
            role,
            ViewRole::RunCompleted | ViewRole::RunCancelled | ViewRole::RunFailed
        ));
        self.messages.push(ViewMessage {
            role,
            title: String::new(),
            content: format_elapsed_compact(elapsed.as_secs()),
            reasoning: String::new(),
            tool_arguments: None,
            tool_id: None,
            file_change: None,
            running: false,
        });
    }

    fn start_assistant_message(&mut self) -> usize {
        self.messages.push(ViewMessage {
            role: ViewRole::Assistant,
            title: String::new(),
            content: String::new(),
            reasoning: String::new(),
            tool_arguments: None,
            tool_id: None,
            file_change: None,
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

fn x11_clipboard(state: &mut UiState) -> Result<&x11_clipboard::Clipboard> {
    if state.x11_clipboard.is_none() {
        state.x11_clipboard = Some(X11Clipboard(
            x11_clipboard::Clipboard::new().context("无法连接 X11 剪贴板")?,
        ));
    }
    Ok(&state
        .x11_clipboard
        .as_ref()
        .expect("X11 clipboard was initialized above")
        .0)
}

fn render(frame: &mut Frame<'_>, state: &mut UiState) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().fg(THEME_TEXT).bg(THEME_BASE)),
        area,
    );
    if area.width < 24 || area.height < MIN_TERMINAL_HEIGHT {
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
            Constraint::Length(INPUT_FOOTER_GAP),
            Constraint::Length(1),
        ])
        .split(area);
    render_conversation(frame, state, areas[0]);
    render_slash_suggestions(frame, state, areas[1]);
    render_activity_status(frame, state, areas[2]);
    render_input(frame, state, areas[3]);
    render_footer(frame, state, areas[5]);
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
    let activity = u16::from(state.activity_label().is_some());
    let reserved_height = 3_u16
        .saturating_add(activity)
        .saturating_add(input)
        .saturating_add(INPUT_FOOTER_GAP)
        .saturating_add(1);
    let suggestions = u16::try_from(suggestion_count)
        .unwrap_or(u16::MAX)
        .min(MAX_SLASH_SUGGESTIONS)
        .min(height.saturating_sub(reserved_height));
    let conversation = height
        .saturating_sub(suggestions)
        .saturating_sub(activity)
        .saturating_sub(input)
        .saturating_sub(INPUT_FOOTER_GAP)
        .saturating_sub(1);
    UiSectionHeights {
        conversation,
        suggestions,
        activity,
        input,
    }
}

fn render_activity_status(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    let Some(activity) = state.activity_label() else {
        return;
    };
    let elapsed = format_elapsed_compact(state.run_elapsed().as_secs());
    let suffix = format!(" ({elapsed} · Esc 取消)");
    let available = usize::from(area.width)
        .saturating_sub(2)
        .saturating_sub(display_width(&suffix));
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("• ", Style::default().fg(THEME_YELLOW)),
            Span::styled(
                truncate_width(&activity, available.saturating_add(1)),
                Style::default()
                    .fg(THEME_SUBTEXT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(suffix, Style::default().fg(THEME_MUTED)),
        ])),
        area,
    );
}

fn render_conversation(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    let content_width = area.width.max(1);
    let lines = transcript_lines(state, content_width);
    if lines.is_empty() || area.height == 0 || area.width == 0 {
        return;
    }
    let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    let total_height = paragraph.line_count(content_width);
    let visible_height = total_height.min(usize::from(area.height));
    let render_height = u16::try_from(visible_height).unwrap_or(area.height);
    let render_y = if state.show_welcome && state.messages.is_empty() {
        area.y
            .saturating_add(area.height.saturating_sub(render_height) / 2)
    } else {
        area.bottom().saturating_sub(render_height)
    };
    let render_area = Rect::new(area.x, render_y, content_width, render_height);
    let scroll = u16::try_from(total_height.saturating_sub(visible_height)).unwrap_or(u16::MAX);
    frame.render_widget(paragraph.scroll((scroll, 0)), render_area);
}

fn transcript_lines(state: &UiState, content_width: u16) -> Vec<Line<'static>> {
    transcript_lines_for_messages(state, &state.messages, content_width)
}

fn transcript_lines_for_messages(
    state: &UiState,
    messages: &[ViewMessage],
    content_width: u16,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if state.show_welcome {
        lines.extend(welcome_lines(state, content_width));
        if !messages.is_empty() {
            lines.push(Line::default());
        }
    }
    lines.extend(conversation_lines_for_messages(messages));
    lines
}

fn transcript_line_count(state: &UiState, messages: &[ViewMessage], width: u16) -> usize {
    let lines = transcript_lines_for_messages(state, messages, width);
    if lines.is_empty() {
        return 0;
    }
    Paragraph::new(Text::from(lines))
        .wrap(Wrap { trim: false })
        .line_count(width.max(1))
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
    let mut lines = Vec::with_capacity(bordered.len().saturating_add(2));
    lines.push(
        Line::from(Span::styled(
            format!("╭{}╮", "─".repeat(inner_width.saturating_add(2))),
            Style::default().fg(THEME_MUTED),
        ))
        .alignment(Alignment::Center),
    );
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
        lines.push(Line::from(spans).alignment(Alignment::Center));
    }
    lines.push(
        Line::from(Span::styled(
            format!("╰{}╯", "─".repeat(inner_width.saturating_add(2))),
            Style::default().fg(THEME_MUTED),
        ))
        .alignment(Alignment::Center),
    );
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

    frame.render_widget(
        Block::default().style(Style::default().bg(THEME_MANTLE)),
        area,
    );

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

    let editor_area = if state.pending_images.is_empty() {
        area
    } else {
        let attachment_area = Rect::new(area.x, area.y, area.width, 1.min(area.height));
        render_pending_images(frame, state, attachment_area);
        Rect::new(
            area.x,
            area.y.saturating_add(1),
            area.width,
            area.height.saturating_sub(1),
        )
    };

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
    let editor = if state.editor.is_empty() {
        Paragraph::new(Span::styled(
            truncate_width(
                INPUT_PLACEHOLDER,
                usize::from(input_area.width).saturating_add(1),
            ),
            Style::default().fg(THEME_MUTED),
        ))
    } else {
        Paragraph::new(state.editor.text())
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0))
    };
    frame.render_widget(editor, input_area);
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
    if state.quit_shortcut_active() {
        return Line::from(Span::styled(
            truncate_width("再按 Ctrl+C 退出", width),
            Style::default()
                .fg(THEME_YELLOW)
                .add_modifier(Modifier::BOLD),
        ));
    }
    let usage_values = state.displayed_usage();
    let context_tokens = state.displayed_context_tokens();
    let estimate = if state.displayed_usage_is_estimated() {
        "~"
    } else {
        ""
    };
    let context_full = format!(
        "{estimate}{}/{} ({}%)",
        format_tokens(context_tokens),
        format_tokens(state.max_input_tokens),
        format_context_percent(context_tokens, state.max_input_tokens)
    );
    let context_compact = format!(
        "{estimate}{}%",
        format_context_percent(context_tokens, state.max_input_tokens)
    );
    let usage = format!(
        " | 输入 {} 输出 {}",
        format_tokens(usage_values.prompt_tokens),
        format_tokens(usage_values.completion_tokens)
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
                .fg(context_usage_color(context_tokens, state.max_input_tokens))
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

fn format_elapsed_compact(elapsed_secs: u64) -> String {
    if elapsed_secs < 60 {
        return format!("{elapsed_secs}s");
    }
    if elapsed_secs < 3_600 {
        return format!("{}m {:02}s", elapsed_secs / 60, elapsed_secs % 60);
    }
    format!(
        "{}h {:02}m {:02}s",
        elapsed_secs / 3_600,
        (elapsed_secs % 3_600) / 60,
        elapsed_secs % 60
    )
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
    UnicodeWidthStr::width(text)
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

fn has_response_reasoning_item(items: &[serde_json::Value]) -> bool {
    items
        .iter()
        .any(|item| item.get("type").and_then(serde_json::Value::as_str) == Some("reasoning"))
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

#[cfg(test)]
fn conversation_lines(state: &UiState) -> Vec<Line<'static>> {
    conversation_lines_for_messages(&state.messages)
}

fn conversation_lines_for_messages(messages: &[ViewMessage]) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for message in messages {
        if matches!(
            message.role,
            ViewRole::RunCompleted | ViewRole::RunCancelled | ViewRole::RunFailed
        ) {
            append_run_summary(&mut lines, message);
            lines.push(Line::default());
            continue;
        }

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
            ViewRole::User
            | ViewRole::Assistant
            | ViewRole::RunCompleted
            | ViewRole::RunCancelled
            | ViewRole::RunFailed => unreachable!(),
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

fn append_run_summary(lines: &mut Vec<Line<'static>>, message: &ViewMessage) {
    let (label, color) = match message.role {
        ViewRole::RunCompleted => ("已完成", THEME_GREEN),
        ViewRole::RunCancelled => ("已取消", THEME_YELLOW),
        ViewRole::RunFailed => ("失败", THEME_RED),
        _ => return,
    };
    lines.push(Line::from(vec![
        Span::styled("─ ", Style::default().fg(THEME_MUTED)),
        Span::styled(
            label,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" · 用时 ", Style::default().fg(THEME_MUTED)),
        Span::styled(message.content.clone(), Style::default().fg(THEME_SUBTEXT)),
    ]));
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
        Span::styled(first.to_string(), Style::default().fg(THEME_TEAL)),
    ]));
    for line in content_lines {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(line.to_string(), Style::default().fg(THEME_TEAL)),
        ]));
    }
}

fn append_tool_message(lines: &mut Vec<Line<'static>>, message: &ViewMessage) {
    let failed = message.role == ViewRole::Error;
    if !failed
        && !message.running
        && let Some(change) = &message.file_change
    {
        append_file_change(lines, change);
        return;
    }
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
        let style = if failed {
            Style::default().fg(THEME_RED)
        } else {
            Style::default().fg(THEME_MUTED)
        };
        if message.running {
            append_tail_preview_lines(
                lines,
                &message.content,
                TOOL_OUTPUT_PREVIEW_LINES,
                "  └ ",
                "    ",
                style,
            );
        } else {
            append_preview_lines(
                lines,
                &message.content,
                TOOL_OUTPUT_PREVIEW_LINES,
                "  └ ",
                "    ",
                style,
            );
        }
    }
}

fn append_file_change(lines: &mut Vec<Line<'static>>, change: &FileChangeSummary) {
    let action = match change.kind {
        FileChangeKind::Added => "已新增",
        FileChangeKind::Updated => "已编辑",
    };
    lines.push(Line::from(vec![
        Span::styled("• ", Style::default().fg(THEME_MUTED)),
        Span::styled(
            action,
            Style::default()
                .fg(THEME_GREEN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(
            sanitize_terminal_text(&change.path),
            Style::default().fg(THEME_TEXT),
        ),
        Span::raw(" ("),
        Span::styled(
            format!("+{}", change.added_lines),
            Style::default().fg(THEME_GREEN),
        ),
        Span::raw(" "),
        Span::styled(
            format!("-{}", change.removed_lines),
            Style::default().fg(THEME_RED),
        ),
        Span::raw(")"),
    ]));

    let highlighted = std::path::Path::new(&change.path)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|extension| {
            let source = change
                .preview
                .iter()
                .filter(|line| line.kind != FileChangeLineKind::Omitted)
                .map(|line| truncate_preview_line(&sanitize_terminal_text(&line.content)))
                .collect::<Vec<_>>()
                .join("\n");
            highlight_code(&source, extension)
        })
        .unwrap_or_default();
    let mut highlighted_index = 0usize;
    for line in &change.preview {
        if line.kind == FileChangeLineKind::Omitted {
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled("⋮", Style::default().fg(THEME_MUTED)),
            ]));
            continue;
        }
        let (sign, color, background) = match line.kind {
            FileChangeLineKind::Context => (' ', THEME_SUBTEXT, None),
            FileChangeLineKind::Added => ('+', THEME_GREEN, Some(THEME_DIFF_ADD_BG)),
            FileChangeLineKind::Removed => ('-', THEME_RED, Some(THEME_DIFF_REMOVE_BG)),
            FileChangeLineKind::Omitted => unreachable!(),
        };
        let style = background.map_or_else(
            || Style::default().fg(color),
            |background| Style::default().fg(color).bg(background),
        );
        let content = sanitize_terminal_text(&line.content);
        let mut spans = vec![
            Span::styled(
                format!("  {:>3} ", line.line_number),
                Style::default().fg(THEME_MUTED),
            ),
            Span::styled(sign.to_string(), style),
        ];
        if let Some(highlighted_line) = highlighted.get(highlighted_index) {
            spans.extend(highlighted_line.spans.iter().cloned().map(|mut span| {
                if let Some(background) = background {
                    span.style = span.style.bg(background);
                }
                span
            }));
        } else {
            spans.push(Span::styled(truncate_preview_line(&content), style));
        }
        highlighted_index = highlighted_index.saturating_add(1);
        lines.push(Line::from(spans).style(
            background.map_or_else(Style::default, |background| Style::default().bg(background)),
        ));
    }
    if change.preview_truncated {
        lines.push(Line::from(vec![
            Span::raw("      "),
            Span::styled("…", Style::default().fg(THEME_MUTED)),
        ]));
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
        "web_search" | "$web_search" | "网页搜索" => {
            ("正在搜索网页", "已搜索网页", "网页搜索失败")
        }
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

fn append_tail_preview_lines(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    limit: usize,
    first_prefix: &str,
    continuation_prefix: &str,
    style: Style,
) {
    let content_lines = content.lines().collect::<Vec<_>>();
    let truncated = content_lines.len() > limit;
    if truncated {
        lines.push(Line::from(vec![
            Span::styled(first_prefix.to_string(), Style::default().fg(THEME_MUTED)),
            Span::styled(
                "…",
                Style::default()
                    .fg(THEME_MUTED)
                    .add_modifier(Modifier::ITALIC),
            ),
        ]));
    }
    let start = content_lines.len().saturating_sub(limit);
    for (index, line) in content_lines[start..].iter().enumerate() {
        let prefix = if index == 0 && !truncated {
            first_prefix
        } else {
            continuation_prefix
        };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(THEME_MUTED)),
            Span::styled(truncate_preview_line(line), style),
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
        "web_search" | "$web_search" => value
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

fn truncate_error_for_ui(text: &str) -> String {
    const MAX_CHARS: usize = 240;
    const MAX_LINES: usize = 3;
    let mut output = String::new();
    let mut truncated = false;
    let mut remaining = MAX_CHARS;
    let mut lines = text.lines();
    for (index, line) in lines.by_ref().take(MAX_LINES).enumerate() {
        if index > 0 {
            output.push('\n');
        }
        let prefix = line.chars().take(remaining).collect::<String>();
        let count = prefix.chars().count();
        output.push_str(&prefix);
        remaining = remaining.saturating_sub(count);
        if count < line.chars().count() || remaining == 0 {
            truncated = true;
            break;
        }
    }
    if lines.next().is_some() {
        truncated = true;
    }
    if truncated {
        output.push_str("\n…");
    }
    output
}

fn trim_live_tool_output(text: &mut String) {
    const LIMIT: usize = 4_000;
    let count = text.chars().count();
    if count <= LIMIT {
        return;
    }
    let tail = text.chars().skip(count - LIMIT).collect::<String>();
    *text = format!("…\n{tail}");
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
        if let Err(error) = execute!(io::stdout(), EnableFocusChange, EnableBracketedPaste, Hide) {
            let _ = execute!(
                io::stdout(),
                DisableBracketedPaste,
                DisableFocusChange,
                Show
            );
            let _ = disable_raw_mode();
            return Err(error).context("配置终端模式失败");
        }
        Ok(Self)
    }
}

impl Drop for ScreenGuard {
    fn drop(&mut self) {
        let _ = execute!(
            io::stdout(),
            DisableBracketedPaste,
            DisableFocusChange,
            Show
        );
        let _ = disable_raw_mode();
    }
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;

    use super::*;

    #[derive(Clone, Default)]
    struct CaptureWriter(std::rc::Rc<std::cell::RefCell<Vec<u8>>>);

    impl std::io::Write for CaptureWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

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
    fn resume_hint_requires_a_nonempty_session_file() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("rollout.jsonl");

        assert!(!session_path_is_resumable(&path));
        std::fs::File::create(&path).unwrap();
        assert!(!session_path_is_resumable(&path));
        std::fs::write(&path, "{}\n").unwrap();
        assert!(session_path_is_resumable(&path));
        assert!(!session_path_is_resumable(temp.path()));
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
    fn supports_codex_style_editor_shortcuts_and_yank() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/responses".to_string(),
            std::path::PathBuf::from("."),
        );
        state.editor.insert_str("alpha beta");

        handle_key(
            KeyEvent::new(KeyCode::Char('w'), KeyModifiers::CONTROL),
            &mut state,
            None,
        );
        assert_eq!(state.editor.text(), "alpha ");
        handle_key(
            KeyEvent::new(KeyCode::Char('y'), KeyModifiers::CONTROL),
            &mut state,
            None,
        );
        assert_eq!(state.editor.text(), "alpha beta");
        handle_key(
            KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL),
            &mut state,
            None,
        );
        handle_key(
            KeyEvent::new(KeyCode::Char('f'), KeyModifiers::ALT),
            &mut state,
            None,
        );
        handle_key(
            KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL),
            &mut state,
            None,
        );
        assert_eq!(state.editor.text(), "alpha");
    }

    #[test]
    fn prioritizes_popup_dismissal_and_queues_follow_up_input() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/responses".to_string(),
            std::path::PathBuf::from("."),
        );
        state.running = true;
        state.editor.insert('/');
        let cancel = CancellationToken::new();

        handle_key(
            KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
            &mut state,
            Some(&cancel),
        );
        assert!(!cancel.is_cancelled());
        assert!(slash_suggestions(&state).is_empty());
        handle_key(
            KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
            &mut state,
            Some(&cancel),
        );
        assert!(cancel.is_cancelled());

        state.running = true;
        state.editor.set_text("follow up");
        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
            None,
        );
        assert!(state.editor.is_empty());
        assert_eq!(state.queued_submissions.len(), 1);
        assert_eq!(
            state.queued_submissions.front().unwrap().prompt,
            "follow up"
        );
        assert!(state.activity_label().unwrap().contains("已排队 1"));

        state.running = false;
        state.editor.set_text("draft");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
                &mut state,
                None,
            ),
            UiAction::None
        ));
        assert!(state.editor.is_empty());
        assert_eq!(
            state.input_history.last().map(String::as_str),
            Some("draft")
        );
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
                &mut state,
                None,
            ),
            UiAction::Quit
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
    fn renders_live_elapsed_time_and_a_final_run_status() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.begin_run("处理中");
        state.run_started_at = Instant::now().checked_sub(Duration::from_secs(322));
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut state)).unwrap();

        let active = rendered_terminal(&terminal);
        assert!(active.contains("5m 22s"));
        assert!(active.contains("Esc"));

        state.pause_run_timer();
        assert!(state.run_started_at.is_none());
        assert!((322..=323).contains(&state.run_elapsed().as_secs()));
        state.resume_run_timer();
        assert!(state.run_started_at.is_some());

        state.apply_agent_event(AgentEvent::RunFinished);
        let completed = conversation_lines(&state)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(completed.contains("已完成"));
        assert!(completed.contains("5m 22s"));
        assert_eq!(format_elapsed_compact(3_723), "1h 02m 03s");
    }

    #[test]
    fn keeps_the_live_turn_contiguous_and_refreshes_usage() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/responses".to_string(),
            std::path::PathBuf::from("."),
        );
        state.protect_new_turn();
        state.begin_live_usage(120);
        state.push_user("继续".to_string(), &[]);
        state.begin_run("处理中");
        state.apply_agent_event(AgentEvent::AssistantStarted);
        state.apply_agent_event(AgentEvent::TextDelta {
            text: format!("{}最终一行", "较长的中文输出。\n".repeat(40)),
        });

        let live_usage = state.displayed_usage();
        assert_eq!(live_usage.prompt_tokens, 120);
        assert!(live_usage.completion_tokens > 0);
        assert!(footer_line(&state, 80).to_string().contains('~'));

        state.apply_agent_event(AgentEvent::Usage {
            usage: Usage {
                prompt_tokens: 150,
                completion_tokens: 90,
                total_tokens: 240,
                cached_prompt_tokens: None,
            },
            context_tokens: 240,
            context_window: 1_000,
            max_input_tokens: 1_000,
            estimated: false,
        });
        state.apply_agent_event(AgentEvent::RunFinished);

        assert_eq!(state.displayed_usage().prompt_tokens, 150);
        assert_eq!(state.displayed_usage().completion_tokens, 90);
        assert_eq!(transcript_archive_count(&state, 40, 6), 0);

        let backend = TestBackend::new(40, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let buffer = terminal.backend().buffer();
        let final_line_y = buffer
            .content
            .iter()
            .enumerate()
            .find_map(|(index, cell)| (cell.symbol() == "最").then_some(index / 40))
            .unwrap();
        let completed_y = buffer
            .content
            .iter()
            .enumerate()
            .find_map(|(index, cell)| (cell.symbol() == "已").then_some(index / 40))
            .unwrap();
        assert!(completed_y.saturating_sub(final_line_y) <= 2);
    }

    #[test]
    fn keeps_a_large_error_and_failure_status_in_the_current_view() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/responses".to_string(),
            std::path::PathBuf::from("."),
        );
        state.push_user("你是？".to_string(), &[]);
        state.begin_run("处理中");
        state.apply_agent_event(AgentEvent::Error {
            message: "Cloudflare error body ".repeat(100),
        });
        let error = state
            .messages
            .iter()
            .find(|message| message.role == ViewRole::Error)
            .unwrap();
        assert!(error.content.chars().count() <= 242);
        assert!(error.content.ends_with('…'));
        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut state)).unwrap();

        let rendered = rendered_terminal(&terminal);
        // Wide CJK glyphs occupy two backend cells, so inspect their leading cells here.
        assert!(rendered.contains('错'));
        assert!(rendered.contains('失'));
        assert!(rendered.contains('…'));
    }

    #[test]
    fn renders_a_welcome_card_and_input_placeholder_for_a_new_session() {
        let mut state = UiState::new(
            "grok-test".to_string(),
            "http://localhost/v1/responses".to_string(),
            std::path::PathBuf::from("."),
        );
        state.reasoning_effort = ReasoningEffort::High;
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut state)).unwrap();

        let rendered = rendered_terminal(&terminal);
        assert!(rendered.contains("MCode"));
        assert!(rendered.contains("xai/grok-test high"));
        assert!(rendered.contains("╭"));

        let buffer = terminal.backend().buffer();
        let placeholder_y = buffer
            .content
            .iter()
            .enumerate()
            .find_map(|(index, cell)| (cell.symbol() == "描").then_some(index / 80))
            .unwrap();
        let input_y = buffer
            .content
            .iter()
            .enumerate()
            .rfind(|(_, cell)| cell.symbol() == ">")
            .map(|(index, _)| index / 80)
            .unwrap();
        assert_eq!(placeholder_y, input_y);
        let border_y = buffer
            .content
            .iter()
            .enumerate()
            .find_map(|(index, cell)| (cell.symbol() == "╭").then_some(index / 80))
            .unwrap();
        let border_left = buffer
            .content
            .iter()
            .enumerate()
            .find_map(|(index, cell)| (cell.symbol() == "╭").then_some(index % 80))
            .unwrap();
        let border_right = buffer
            .content
            .iter()
            .enumerate()
            .find_map(|(index, cell)| (cell.symbol() == "╮").then_some(index % 80))
            .unwrap();
        assert_eq!(border_y, 4);
        assert!(border_left.abs_diff(79 - border_right) <= 1);
        assert_eq!(
            buffer[(79, u16::try_from(input_y).unwrap())].bg,
            Color::Reset
        );

        state.editor.insert('x');
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        assert!(!rendered_terminal(&terminal).contains("描述任务"));
    }

    #[test]
    fn archives_only_completed_overflow_and_distinguishes_user_text() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/responses".to_string(),
            std::path::PathBuf::from("."),
        );
        state.push_user("用户内容\n".repeat(12), &[]);
        state.start_assistant_message();

        assert_eq!(transcript_archive_count(&state, 40, 5), 1);
        let lines = conversation_lines_for_messages(&state.messages);
        let user_span = lines
            .iter()
            .flat_map(|line| &line.spans)
            .find(|span| span.content.contains("用户内容"))
            .unwrap();
        assert_eq!(user_span.style.fg, Some(THEME_TEAL));

        let assistant = ViewMessage {
            role: ViewRole::Assistant,
            title: String::new(),
            content: "Agent 内容".to_string(),
            reasoning: String::new(),
            tool_arguments: None,
            tool_id: None,
            file_change: None,
            running: false,
        };
        let assistant_lines = conversation_lines_for_messages(&[assistant]);
        let assistant_span = assistant_lines
            .iter()
            .flat_map(|line| &line.spans)
            .find(|span| span.content.contains("Agent 内容"))
            .unwrap();
        assert_eq!(assistant_span.style.fg, Some(THEME_TEXT));
    }

    #[test]
    fn writes_raw_history_buffers_without_cjk_spacing() {
        let mut buffer = ratatui::buffer::Buffer::empty(Rect::new(0, 0, 8, 1));
        buffer.set_string(0, 0, "你好", Style::default());
        let output = CaptureWriter::default();
        let mut backend = UiBackend::new(output.clone());
        let width = usize::from(buffer.area.width);

        backend
            .draw(buffer.content.iter().enumerate().map(|(index, cell)| {
                (
                    u16::try_from(index % width).unwrap(),
                    u16::try_from(index / width).unwrap(),
                    cell,
                )
            }))
            .unwrap();

        let bytes = output.0.borrow();
        let output = std::str::from_utf8(&bytes).unwrap();
        assert!(output.contains("你好"), "terminal output was {output:?}");
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
        let footer_y = buffer
            .content
            .iter()
            .enumerate()
            .find_map(|(index, cell)| (cell.symbol() == "上").then_some(index / 80))
            .unwrap();
        assert!(input_y.saturating_sub(message_y) <= 2);
        assert_eq!(footer_y.saturating_sub(input_y), 2);
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
            file_change: None,
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
    fn renders_file_changes_with_counts_and_colored_diff_lines() {
        let message = ViewMessage {
            role: ViewRole::Tool,
            title: "edit_file".to_string(),
            content: "updated 1 replacement in src/lib.rs".to_string(),
            reasoning: String::new(),
            tool_arguments: Some("src/lib.rs".to_string()),
            tool_id: Some("call_edit".to_string()),
            file_change: Some(FileChangeSummary {
                path: "src/lib.rs".to_string(),
                kind: FileChangeKind::Updated,
                added_lines: 1,
                removed_lines: 1,
                preview: vec![
                    crate::protocol::FileChangeLine {
                        kind: FileChangeLineKind::Removed,
                        line_number: 7,
                        content: "let old = true;".to_string(),
                    },
                    crate::protocol::FileChangeLine {
                        kind: FileChangeLineKind::Added,
                        line_number: 7,
                        content: "let new = true;".to_string(),
                    },
                ],
                preview_truncated: true,
            }),
            running: false,
        };

        let lines = conversation_lines_for_messages(&[message]);
        let rendered = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("已编辑 src/lib.rs (+1 -1)"));
        assert!(rendered.contains("7 -let old = true;"));
        assert!(rendered.contains("7 +let new = true;"));
        assert!(rendered.contains('…'));
        assert!(lines.iter().flat_map(|line| &line.spans).any(|span| {
            span.content.starts_with('-') && span.style.bg == Some(THEME_DIFF_REMOVE_BG)
        }));
        assert!(lines.iter().flat_map(|line| &line.spans).any(|span| {
            span.content.starts_with('+') && span.style.bg == Some(THEME_DIFF_ADD_BG)
        }));
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

        let mut resumed_state = UiState::new(
            "deepseek-v4-flash".to_string(),
            "https://api.deepseek.com/responses".to_string(),
            std::path::PathBuf::from("."),
        );
        resumed_state.api = ApiProtocol::Responses;
        resumed_state.push_history(ChatMessage::assistant_with_response_items(
            Some("**Final response.**".to_string()),
            Some("Private raw reasoning.".to_string()),
            Vec::new(),
            vec![serde_json::json!({
                "type": "reasoning",
                "id": "rs_deepseek",
                "content": [{
                    "type": "reasoning_text",
                    "text": "Private raw reasoning."
                }],
                "summary": []
            })],
        ));
        let resumed_lines = conversation_lines(&resumed_state);
        let resumed = resumed_lines
            .iter()
            .map(Line::to_string)
            .collect::<String>();
        assert!(resumed.contains("Final response."));
        assert!(!resumed.contains("**"));
        assert!(!resumed.contains("Private raw reasoning."));
        let final_response = resumed_lines
            .iter()
            .flat_map(|line| &line.spans)
            .find(|span| span.content == "Final response.")
            .unwrap();
        assert!(final_response.style.add_modifier.contains(Modifier::BOLD));
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
