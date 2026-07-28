use std::io;
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
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
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

use crate::agent::{Agent, ModelChoice};
use crate::approval::{ApprovalDecision, ApprovalGate, ApprovalRequest, format_tool_arguments};
use crate::config::{ApiProtocol, ReasoningEffort, WebSearchMode};
use crate::event::{AgentEvent, CompactionReason};
use crate::protocol::{ChatMessage, ImageAttachment, MessageRole, Usage, sanitize_terminal_text};

const INPUT_HEIGHT: u16 = 5;
const FRAME_INTERVAL: Duration = Duration::from_millis(50);

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
            "MCP server {:?} was disabled after a startup failure: {}",
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
    let mut terminal = Terminal::new(backend).context("failed to initialize terminal")?;
    terminal.clear().context("failed to clear terminal")?;

    if let Some(checkpoint) = historical_compaction {
        state.push_notice(format!(
            "This session was compacted from approximately {} tokens.\n\n{}",
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
            state.push_notice(
                "Resuming the interrupted run first; the supplied prompt is waiting in the editor.",
            );
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
                .context("failed to draw terminal UI")?;
            last_frame = Instant::now();
        }

        if !event::poll(Duration::from_millis(20)).context("failed to poll terminal events")? {
            continue;
        }
        match event::read().context("failed to read terminal event")? {
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
                                state.push_notice(format!("Model changed to {}.", state.model));
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state
                            .push_error("The agent is busy; wait for the current turn to finish."),
                    },
                    UiAction::SetReasoning(effort) => match agent.try_lock() {
                        Ok(mut agent) => match agent.set_reasoning_effort(effort) {
                            Ok(()) => {
                                state.sync_from_agent(&agent);
                                state.push_notice(format!(
                                    "Reasoning changed to {}.",
                                    state.reasoning_effort
                                ));
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state
                            .push_error("The agent is busy; wait for the current turn to finish."),
                    },
                    UiAction::SetWebSearch(mode) => match agent.try_lock() {
                        Ok(mut agent) => match agent.set_web_search_mode(mode) {
                            Ok(()) => {
                                state.sync_from_agent(&agent);
                                state.push_notice(format!(
                                    "Web search changed to {}.",
                                    state.web_search_mode
                                ));
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state
                            .push_error("The agent is busy; wait for the current turn to finish."),
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
                                state.push_notice("Started a new session.");
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state
                            .push_error("The agent is busy; wait for the current turn to finish."),
                    },
                    UiAction::DeleteSession => match agent.try_lock() {
                        Ok(mut agent) => match agent.delete_session() {
                            Ok(id) => {
                                deleted_session = Some(id);
                                break 'ui;
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state
                            .push_error("The agent is busy; wait for the current turn to finish."),
                    },
                    UiAction::AttachImage(path) => match ImageAttachment::load(&path, &state.cwd) {
                        Ok(image) => {
                            let name = image.name.clone();
                            state.pending_images.push(image);
                            state.push_notice(format!("Attached {name} to the next prompt."));
                        }
                        Err(error) => state.push_error(format!("{error:#}")),
                    },
                    UiAction::ResolveApproval(decision) => {
                        if let Some(request) = pending_approval.take() {
                            request.resolve(decision);
                        }
                        state.clear_pending_approval();
                    }
                }
            }
            Event::Paste(text) if state.pending_approval.is_none() => {
                state.editor.insert_str(&text);
            }
            Event::Resize(_, _) => state.follow_tail = true,
            _ => {}
        }
    }

    drop(terminal);
    drop(screen);
    if let Some(id) = deleted_session {
        println!("Deleted session {id}.");
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
    state.status = "resuming interrupted run".to_string();
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
    state.status = "compacting".to_string();
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
    AttachImage(PathBuf),
    ResolveApproval(ApprovalDecision),
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
                state.status = "cancelling".to_string();
            }
            return UiAction::ResolveApproval(ApprovalDecision::Deny);
        }
        if key.code == KeyCode::Esc {
            if let Some(cancel) = active_cancel {
                cancel.cancel();
                state.status = "cancelling".to_string();
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
            KeyCode::Char('y' | 'Y') => UiAction::ResolveApproval(ApprovalDecision::ApproveOnce),
            KeyCode::Char('a' | 'A') => {
                UiAction::ResolveApproval(ApprovalDecision::ApproveForSession)
            }
            KeyCode::Char('n' | 'N') => UiAction::ResolveApproval(ApprovalDecision::Deny),
            _ => UiAction::None,
        };
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('c') => {
                if state.running {
                    if let Some(cancel) = active_cancel {
                        cancel.cancel();
                        state.status = "cancelling".to_string();
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
            _ => {}
        }
    }

    match key.code {
        KeyCode::Esc if state.running => {
            if let Some(cancel) = active_cancel {
                cancel.cancel();
                state.status = "cancelling".to_string();
            }
        }
        KeyCode::Enter
            if key.modifiers.contains(KeyModifiers::SHIFT)
                || key.modifiers.contains(KeyModifiers::ALT) =>
        {
            state.editor.insert('\n');
        }
        KeyCode::Enter if !state.running => {
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
                "quit" | "exit" => return UiAction::Quit,
                "clear" => {
                    state.messages.clear();
                    state.follow_tail = true;
                }
                "new" => return UiAction::NewSession,
                "compact" => return UiAction::Compact(argument.to_string()),
                "delete" if argument.eq_ignore_ascii_case("confirm") => {
                    if state.delete_confirmation == DeleteConfirmation::Pending {
                        return UiAction::DeleteSession;
                    }
                    state.push_error(
                        "Run /delete first, then /delete confirm to permanently delete this session.",
                    );
                }
                "delete" if argument.is_empty() => {
                    state.delete_confirmation = DeleteConfirmation::Pending;
                    state.push_notice(
                        "Delete this session? This cannot be undone. Run /delete confirm to continue.",
                    );
                }
                "delete" => state.push_error("Use /delete or /delete confirm."),
                "image" if argument.eq_ignore_ascii_case("clear") => {
                    state.pending_images.clear();
                    state.push_notice("Cleared pending images.");
                }
                "image" if argument.is_empty() => {
                    let notice = state.image_list_notice();
                    state.push_notice(notice);
                }
                "image" => return UiAction::AttachImage(PathBuf::from(argument)),
                "model" if argument.is_empty() => {
                    let notice = state.model_list_notice();
                    state.push_notice(notice);
                }
                "model" => return UiAction::SelectModel(argument.to_string()),
                "reasoning" | "thinking" if argument.is_empty() => {
                    let notice = state.reasoning_list_notice();
                    state.push_notice(notice);
                }
                "reasoning" | "thinking" => {
                    if let Some(effort) = parse_reasoning_effort(argument) {
                        return UiAction::SetReasoning(effort);
                    }
                    state.push_error(format!(
                        "Unknown reasoning level {argument:?}. Use off, minimal, low, medium, high, xhigh, or max."
                    ));
                }
                "search" if argument.is_empty() => {
                    state.push_notice(format!(
                        "Web search: {}\nSelect with /search <disabled|cached|live>.",
                        state.web_search_mode
                    ));
                }
                "search" => {
                    if let Some(mode) = parse_web_search_mode(argument) {
                        return UiAction::SetWebSearch(mode);
                    }
                    state.push_error(format!(
                        "Unknown web search mode {argument:?}. Use disabled, cached, or live."
                    ));
                }
                "status" => {
                    let notice = state.status_notice();
                    state.push_notice(notice);
                }
                "help" => state.push_notice(
                    "Commands: /model [ID], /reasoning [LEVEL], /search [MODE], /compact [INSTRUCTIONS], /image [PATH|clear], /status, /new, /delete, /clear, /help, /quit",
                ),
                _ => state.push_error(format!("Unknown command: /{name}")),
            }
        }
        KeyCode::Char(character)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            state.editor.insert(character);
        }
        KeyCode::Backspace => state.editor.backspace(),
        KeyCode::Delete => state.editor.delete(),
        KeyCode::Left => state.editor.move_left(),
        KeyCode::Right => state.editor.move_right(),
        KeyCode::Home => state.editor.move_home(),
        KeyCode::End => state.editor.move_end(),
        KeyCode::PageUp => state.scroll_up(),
        KeyCode::PageDown => state.scroll_down(),
        _ => {}
    }
    UiAction::None
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

#[derive(Debug, Default)]
struct Editor {
    chars: Vec<char>,
    cursor: usize,
}

impl Editor {
    fn insert(&mut self, character: char) {
        self.chars.insert(self.cursor, character);
        self.cursor += 1;
    }

    fn insert_str(&mut self, text: &str) {
        for character in text.chars() {
            self.insert(character);
        }
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
        }
    }

    fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    fn move_right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.chars.len());
    }

    fn move_home(&mut self) {
        self.cursor = self.chars[..self.cursor]
            .iter()
            .rposition(|character| character == &'\n')
            .map_or(0, |index| index + 1);
    }

    fn move_end(&mut self) {
        self.cursor = self.chars[self.cursor..]
            .iter()
            .position(|character| character == &'\n')
            .map_or(self.chars.len(), |index| self.cursor + index);
    }

    fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    fn text(&self) -> String {
        self.chars.iter().collect()
    }

    fn take(&mut self) -> String {
        self.cursor = 0;
        self.chars.drain(..).collect()
    }

    fn cursor_layout(&self, width: u16, visible_height: u16) -> (u16, u16, u16) {
        let width = usize::from(width.max(1));
        let mut row = 0usize;
        let mut column = 0usize;
        for character in self.chars.iter().take(self.cursor) {
            if character == &'\n' {
                row += 1;
                column = 0;
                continue;
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
        }
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
    Pending,
}

#[derive(Debug)]
struct ViewMessage {
    role: ViewRole,
    title: String,
    content: String,
    reasoning: String,
    tool_id: Option<String>,
    running: bool,
}

#[derive(Debug)]
struct ApprovalView {
    name: String,
    arguments: String,
}

#[derive(Debug)]
struct UiState {
    model: String,
    provider: Option<String>,
    api: ApiProtocol,
    reasoning_effort: ReasoningEffort,
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
            status: "ready".to_string(),
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
                    title: "you".to_string(),
                    content,
                    reasoning: String::new(),
                    tool_id: None,
                    running: false,
                });
            }
            MessageRole::Assistant => self.messages.push(ViewMessage {
                role: ViewRole::Assistant,
                title: "assistant".to_string(),
                content: sanitize_terminal_text(&message.content.unwrap_or_default()),
                reasoning: sanitize_terminal_text(&message.reasoning_content.unwrap_or_default()),
                tool_id: None,
                running: false,
            }),
            MessageRole::Tool => self.messages.push(ViewMessage {
                role: ViewRole::Tool,
                title: "tool".to_string(),
                content: sanitize_terminal_text(&message.content.unwrap_or_default()),
                reasoning: String::new(),
                tool_id: message.tool_call_id,
                running: false,
            }),
        }
        self.follow_tail = true;
        self.delete_confirmation = DeleteConfirmation::None;
    }

    fn push_user(&mut self, prompt: String, images: &[ImageAttachment]) {
        self.messages.push(ViewMessage {
            role: ViewRole::User,
            title: "you".to_string(),
            content: sanitize_terminal_text(&format_user_content(prompt, images)),
            reasoning: String::new(),
            tool_id: None,
            running: false,
        });
        self.follow_tail = true;
    }

    fn take_pending_images(&mut self) -> Vec<ImageAttachment> {
        std::mem::take(&mut self.pending_images)
    }

    fn image_list_notice(&self) -> String {
        if self.pending_images.is_empty() {
            return "No images are attached to the next prompt.".to_string();
        }
        let names = self
            .pending_images
            .iter()
            .map(|image| format!("- {}", sanitize_terminal_text(&image.name)))
            .collect::<Vec<_>>()
            .join("\n");
        format!("Images attached to the next prompt:\n{names}")
    }

    fn push_notice(&mut self, content: impl AsRef<str>) {
        self.messages.push(ViewMessage {
            role: ViewRole::Notice,
            title: "MCode".to_string(),
            content: sanitize_terminal_text(content.as_ref()),
            reasoning: String::new(),
            tool_id: None,
            running: false,
        });
        self.follow_tail = true;
    }

    fn push_error(&mut self, content: impl AsRef<str>) {
        self.messages.push(ViewMessage {
            role: ViewRole::Error,
            title: "error".to_string(),
            content: sanitize_terminal_text(content.as_ref()),
            reasoning: String::new(),
            tool_id: None,
            running: false,
        });
        self.follow_tail = true;
    }

    fn set_pending_approval(&mut self, request: &ApprovalRequest) {
        self.pending_approval = Some(ApprovalView {
            name: sanitize_terminal_text(&request.name),
            arguments: sanitize_terminal_text(&format_tool_arguments(&request.arguments)),
        });
        self.status = "approval required".to_string();
    }

    fn clear_pending_approval(&mut self) {
        self.pending_approval = None;
        if self.running && self.status == "approval required" {
            self.status = "working".to_string();
        }
    }

    fn reset_session(&mut self) {
        self.messages.clear();
        self.usage = Usage::default();
        self.context_tokens = 0;
        self.usage_estimated = false;
        self.status = "ready".to_string();
        self.follow_tail = true;
        self.delete_confirmation = DeleteConfirmation::None;
        self.pending_images.clear();
        self.pending_approval = None;
    }

    fn model_list_notice(&self) -> String {
        if self.model_choices.is_empty() {
            return format!(
                "Current model: {}\nNo models are listed in ~/.mcode/agent/models.json; /model <ID> still selects a model on the current endpoint.",
                self.model
            );
        }
        let mut lines = vec!["Configured models:".to_string()];
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
            let reasoning = if choice.reasoning { ", reasoning" } else { "" };
            let limits = if choice.max_input_tokens == choice.context_window {
                format!("{} context/input", format_tokens(choice.context_window))
            } else {
                format!(
                    "{} context, {} max input",
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
        lines.push("Select with /model <provider/model>.".to_string());
        lines.join("\n")
    }

    fn reasoning_list_notice(&self) -> String {
        let mut lines = vec!["Reasoning levels:".to_string()];
        for effort in &self.reasoning_choices {
            let selected = if *effort == self.reasoning_effort {
                "*"
            } else {
                " "
            };
            lines.push(format!("{selected} {effort}"));
        }
        lines.push("Select with /reasoning <level>.".to_string());
        lines.join("\n")
    }

    fn status_notice(&self) -> String {
        let qualified_model = self.provider.as_deref().map_or_else(
            || self.model.clone(),
            |provider| format!("{provider}/{}", self.model),
        );
        let estimate = if self.usage_estimated { "~" } else { "" };
        let percent = format_context_percent(self.context_tokens, self.max_input_tokens);
        format!(
            "Model: {qualified_model}\nAPI: {}\nReasoning: {}\nWeb search: {}\nInput: {estimate}{}/{} ({percent}%)\nModel context window: {}\nTokens: {estimate}in {} out {}\nMCP: {} server(s), {} tool(s)\nEndpoint: {}\nWorking directory: {}",
            self.api,
            self.reasoning_effort,
            self.web_search_mode,
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
                self.status = "working".to_string();
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
                    assistant.title = format!("assistant (retry {attempt}/{max_attempts})");
                    assistant.content.clear();
                    assistant.reasoning.clear();
                    assistant.running = true;
                }
                self.current_assistant = Some(index);
                self.status = format!("retrying response: {}", sanitize_terminal_text(&message));
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
                let arguments = sanitize_terminal_text(&format_tool_arguments(&arguments));
                let name = sanitize_terminal_text(&name);
                self.messages.push(ViewMessage {
                    role: ViewRole::Tool,
                    title: format!("approval required: {name}"),
                    content: arguments,
                    reasoning: String::new(),
                    tool_id: Some(id),
                    running: true,
                });
                self.status = "approval required".to_string();
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
                            "approved{}: {name}",
                            if for_session { " for session" } else { "" }
                        )
                    } else {
                        format!("denied: {name}")
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
                let arguments = sanitize_terminal_text(&format_tool_arguments(&arguments));
                let name = sanitize_terminal_text(&name);
                if let Some(message) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|message| message.tool_id.as_deref() == Some(id.as_str()))
                {
                    message.role = ViewRole::Tool;
                    message.title = name;
                    message.content = arguments;
                    message.running = true;
                } else {
                    self.messages.push(ViewMessage {
                        role: ViewRole::Tool,
                        title: name,
                        content: arguments,
                        reasoning: String::new(),
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
                    title: "web search".to_string(),
                    content: "searching...".to_string(),
                    reasoning: String::new(),
                    tool_id: Some(id),
                    running: true,
                });
                self.status = "searching web".to_string();
            }
            AgentEvent::WebSearchFinished { id, action } => {
                if let Some(message) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|message| message.tool_id.as_deref() == Some(id.as_str()))
                {
                    message.content = sanitize_terminal_text(&action.description());
                    message.running = false;
                }
                self.status = "working".to_string();
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
                "Context limit: omitted {dropped_messages} message(s) from {dropped_turns} earlier turn(s); estimated input is {} tokens.",
                format_tokens(estimated_tokens)
            )),
            AgentEvent::CompactionStarted { .. } => {
                self.finish_current_assistant_for_tool();
                self.status = "compacting".to_string();
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
                    "Context compacted ({} -> {} estimated tokens).\n\n{summary}",
                    format_tokens(tokens_before),
                    format_tokens(tokens_after)
                ));
                if reason == CompactionReason::Manual {
                    self.finish_run("ready");
                } else {
                    self.status = "working".to_string();
                }
            }
            AgentEvent::CompactionFailed { reason, message } => {
                if reason == CompactionReason::Manual {
                    self.finish_run("error");
                    self.push_error(format!("Compaction failed: {message}"));
                } else {
                    self.status = "working".to_string();
                    self.push_error(format!(
                        "Automatic compaction failed; using hard context trimming as fallback: {message}"
                    ));
                }
            }
            AgentEvent::RunFinished => self.finish_run("ready"),
            AgentEvent::Cancelled => self.finish_run("cancelled"),
            AgentEvent::Error { message } => {
                self.finish_run("error");
                self.messages.push(ViewMessage {
                    role: ViewRole::Error,
                    title: "error".to_string(),
                    content: sanitize_terminal_text(&message),
                    reasoning: String::new(),
                    tool_id: None,
                    running: false,
                });
            }
        }
        self.follow_tail = true;
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
            title: "assistant".to_string(),
            content: String::new(),
            reasoning: String::new(),
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
        self.follow_tail = false;
        let amount = self.viewport_height.saturating_sub(2).max(1);
        self.scroll = self.scroll.saturating_sub(amount);
    }

    fn scroll_down(&mut self) {
        let amount = self.viewport_height.saturating_sub(2).max(1);
        self.scroll = (self.scroll + amount).min(self.max_scroll);
        if self.scroll >= self.max_scroll {
            self.follow_tail = true;
        }
    }
}

fn render(frame: &mut Frame<'_>, state: &mut UiState) {
    let area = frame.area();
    if area.width < 24 || area.height < 11 {
        frame.render_widget(
            Paragraph::new("Terminal too small")
                .style(Style::default().fg(Color::Red))
                .block(Block::default().borders(Borders::ALL)),
            area,
        );
        return;
    }

    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(INPUT_HEIGHT),
            Constraint::Length(2),
        ])
        .split(area);
    render_header(frame, state, areas[0]);
    render_conversation(frame, state, areas[1]);
    render_input(frame, state, areas[2]);
    render_footer(frame, state, areas[3]);
}

fn render_header(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    let spinner = ["-", "\\", "|", "/"][state.spinner_frame % 4];
    let activity = if state.running { spinner } else { " " };
    let line = Line::from(vec![
        Span::styled(
            " MCode ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Rgb(103, 232, 163))
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            state.model.clone(),
            Style::default().fg(Color::Rgb(126, 200, 255)),
        ),
        Span::raw("  "),
        Span::styled(
            format!("reasoning {}", state.reasoning_effort),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("  "),
        Span::styled(activity, Style::default().fg(Color::Yellow)),
        Span::raw(" "),
        Span::styled(state.status.clone(), Style::default().fg(Color::Gray)),
    ]);
    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(Color::Rgb(24, 27, 32))),
        area,
    );
}

fn render_conversation(frame: &mut Frame<'_>, state: &mut UiState, area: Rect) {
    let lines = conversation_lines(state);
    let width = usize::from(area.width.max(1));
    let total_height = wrapped_height(&lines, width);
    state.viewport_height = usize::from(area.height.max(1));
    state.max_scroll = total_height.saturating_sub(state.viewport_height);
    if state.follow_tail {
        state.scroll = state.max_scroll;
    } else {
        state.scroll = state.scroll.min(state.max_scroll);
    }
    let scroll = u16::try_from(state.scroll).unwrap_or(u16::MAX);
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        area,
    );
}

fn render_input(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    if let Some(approval) = &state.pending_approval {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(Span::styled(
                " approval required ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(area);
        let arguments = approval.arguments.replace(['\r', '\n'], " ");
        let details = truncate_width(&arguments, usize::from(inner.width));
        let lines = vec![
            Line::from(Span::styled(
                approval.name.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(details),
            Line::from(vec![
                Span::styled("[y]", Style::default().fg(Color::Green)),
                Span::raw(" once  "),
                Span::styled("[a]", Style::default().fg(Color::Yellow)),
                Span::raw(" session  "),
                Span::styled("[n]", Style::default().fg(Color::Red)),
                Span::raw(" deny"),
            ]),
        ];
        frame.render_widget(Paragraph::new(lines).block(block), area);
        return;
    }

    let border_color = if state.running {
        Color::Yellow
    } else {
        Color::Rgb(126, 200, 255)
    };
    let title = if state.pending_images.is_empty() {
        " prompt ".to_string()
    } else {
        format!(" prompt | {} image(s) ", state.pending_images.len())
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(
            title,
            Style::default()
                .fg(border_color)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    let (cursor_x, cursor_y, scroll) = state.editor.cursor_layout(inner.width, inner.height);
    frame.render_widget(
        Paragraph::new(state.editor.text())
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        area,
    );
    frame.set_cursor_position(Position::new(
        inner
            .x
            .saturating_add(cursor_x.min(inner.width.saturating_sub(1))),
        inner
            .y
            .saturating_add(cursor_y.min(inner.height.saturating_sub(1))),
    ));
}

fn render_footer(frame: &mut Frame<'_>, state: &UiState, area: Rect) {
    let cwd = sanitize_terminal_text(&state.cwd.to_string_lossy());
    let estimate = if state.usage_estimated { "~" } else { "" };
    let left = format!(
        " {estimate}in {} out {} | input {estimate}{}/{} ({}%)",
        format_tokens(state.usage.prompt_tokens),
        format_tokens(state.usage.completion_tokens),
        format_tokens(state.context_tokens),
        format_tokens(state.max_input_tokens),
        format_context_percent(state.context_tokens, state.max_input_tokens)
    );
    let right = format!("{} | reasoning {} ", state.model, state.reasoning_effort);
    let footer_line = align_footer_parts(&left, &right, usize::from(area.width));
    let lines = vec![
        Line::from(truncate_width(&format!(" {cwd}"), usize::from(area.width))),
        Line::from(footer_line),
    ];
    frame.render_widget(
        Paragraph::new(Text::from(lines)).style(
            Style::default()
                .fg(Color::DarkGray)
                .bg(Color::Rgb(24, 27, 32)),
        ),
        area,
    );
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
        content.push_str("[image: ");
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

fn align_footer_parts(left: &str, right: &str, width: usize) -> String {
    let left_width = display_width(left);
    let right_width = display_width(right);
    if left_width.saturating_add(right_width).saturating_add(2) <= width {
        return format!(
            "{left}{}{right}",
            " ".repeat(width - left_width - right_width)
        );
    }
    truncate_width(&format!("{left}  {right}"), width)
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
                let running = if message.running && message.content.is_empty() {
                    "  reasoning"
                } else {
                    ""
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        "thinking",
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(running, Style::default().fg(Color::DarkGray)),
                ]));
                for line in message.reasoning.lines() {
                    lines.push(Line::from(Span::styled(
                        line.to_string(),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::ITALIC),
                    )));
                }
                if !message.content.is_empty() {
                    lines.push(Line::default());
                }
            }
            if !message.content.is_empty() || message.reasoning.is_empty() {
                let running = if message.running { "  responding" } else { "" };
                lines.push(Line::from(vec![
                    Span::styled(
                        "assistant",
                        Style::default()
                            .fg(Color::Rgb(103, 232, 163))
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(running, Style::default().fg(Color::DarkGray)),
                ]));
                append_markdownish_lines(
                    &mut lines,
                    &message.content,
                    Style::default().fg(Color::White),
                );
            }
            lines.push(Line::default());
            continue;
        }

        let (label_color, content_style) = match message.role {
            ViewRole::User => (Color::Rgb(126, 200, 255), Style::default().fg(Color::White)),
            ViewRole::Tool => (Color::Rgb(245, 190, 78), Style::default().fg(Color::Gray)),
            ViewRole::Notice => (Color::Cyan, Style::default().fg(Color::Gray)),
            ViewRole::Error => (Color::Red, Style::default().fg(Color::LightRed)),
            ViewRole::Assistant => unreachable!(),
        };
        let running = if message.running { "  running" } else { "" };
        lines.push(Line::from(vec![
            Span::styled(
                message.title.clone(),
                Style::default()
                    .fg(label_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(running, Style::default().fg(Color::DarkGray)),
        ]));
        append_markdownish_lines(&mut lines, &message.content, content_style);
        lines.push(Line::default());
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "Ready.",
            Style::default().fg(Color::DarkGray),
        )));
    }
    lines
}

fn append_markdownish_lines(lines: &mut Vec<Line<'static>>, content: &str, base: Style) {
    let mut in_code = false;
    for raw_line in content.lines() {
        if raw_line.trim_start().as_bytes().starts_with(&[96, 96, 96]) {
            in_code = !in_code;
            continue;
        }
        let style = if in_code {
            Style::default()
                .fg(Color::Rgb(215, 220, 230))
                .bg(Color::Rgb(31, 35, 41))
        } else if raw_line.starts_with('#') {
            base.add_modifier(Modifier::BOLD)
        } else {
            base
        };
        lines.push(Line::from(Span::styled(raw_line.to_string(), style)));
    }
    if content.is_empty() {
        lines.push(Line::default());
    }
}

fn wrapped_height(lines: &[Line<'_>], width: usize) -> usize {
    let width = width.max(1);
    lines
        .iter()
        .map(|line| line.width().max(1).div_ceil(width))
        .sum()
}

fn truncate_for_ui(text: &str) -> String {
    const LIMIT: usize = 4_000;
    let mut chars = text.chars();
    let prefix: String = chars.by_ref().take(LIMIT).collect();
    if chars.next().is_some() {
        format!("{prefix}\n... output shortened in UI; full result was sent to the model")
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
        enable_raw_mode().context("failed to enable raw mode")?;
        execute!(
            io::stdout(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            Hide
        )
        .context("failed to enter alternate screen")?;
        Ok(Self)
    }
}

impl Drop for ScreenGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            Show,
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
    }
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;

    use super::*;

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
        assert!(rendered.contains("web search"));
        assert!(rendered.contains("current release"));
        assert!(rendered.contains("The current release is available."));
        assert!(rendered.contains("prompt"));
        assert!(rendered.contains("/tmp/project"));
        assert!(rendered.contains("reasoning off"));
        assert!(rendered.contains("in 0 out 0"));
        assert!(rendered.contains("input 0/128k"));
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
        assert!(!rendered.contains("Terminal too small"));
    }

    #[test]
    fn renders_reasoning_as_a_separate_block() {
        let mut state = UiState::new(
            "reasoning-model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("/tmp/project"),
        );
        state.reasoning_effort = ReasoningEffort::High;
        state.apply_agent_event(AgentEvent::AssistantStarted);
        state.apply_agent_event(AgentEvent::ReasoningDelta {
            text: "Inspecting the request.".to_string(),
        });
        state.apply_agent_event(AgentEvent::TextDelta {
            text: "Final response.".to_string(),
        });

        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        assert!(rendered.contains("thinking"));
        assert!(rendered.contains("Inspecting the request."));
        assert!(rendered.contains("assistant"));
        assert!(rendered.contains("Final response."));
        assert!(rendered.contains("reasoning high"));
    }

    #[test]
    fn slash_runtime_controls_return_independent_actions() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.editor.insert_str("/reasoning high");
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
    fn slash_delete_requires_a_two_step_confirmation() {
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
        assert_eq!(state.delete_confirmation, DeleteConfirmation::Pending);

        state.editor.insert_str("/delete confirm");
        assert!(matches!(
            handle_key(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut state,
                None,
            ),
            UiAction::DeleteSession
        ));
    }

    #[test]
    fn approval_prompt_accepts_once_session_or_deny_keys() {
        let mut state = UiState::new(
            "model".to_string(),
            "http://localhost/v1/chat/completions".to_string(),
            std::path::PathBuf::from("."),
        );
        state.pending_approval = Some(ApprovalView {
            name: "shell".to_string(),
            arguments: r#"{"command":"cargo test"}"#.to_string(),
        });

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
        assert!(rendered.contains("approval required"));
        assert!(rendered.contains("shell"));
        assert!(rendered.contains("[y] once"));
        assert!(rendered.contains("/tmp/project"));
    }

    #[test]
    fn formats_compact_token_counts_and_context_percent() {
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1_250), "1.2k");
        assert_eq!(format_tokens(12_500), "13k");
        assert_eq!(format_context_percent(24_000, 128_000), "18.7");
    }
}
