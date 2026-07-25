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
use crate::config::ReasoningEffort;
use crate::event::AgentEvent;
use crate::protocol::{ChatMessage, ImageAttachment, MessageRole, Usage};

const INPUT_HEIGHT: u16 = 5;
const FRAME_INTERVAL: Duration = Duration::from_millis(50);

pub fn run_interactive(
    agent: Agent,
    initial_prompt: Option<String>,
    initial_images: Vec<ImageAttachment>,
) -> Result<()> {
    let historical_messages = agent.messages().to_vec();
    let model = agent.model().to_string();
    let endpoint = agent.endpoint().to_string();
    let cwd = agent.session().cwd().to_path_buf();
    let mut state = UiState::new(model, endpoint, cwd);
    state.pending_images = initial_images;
    state.sync_from_agent(&agent);
    let agent = Arc::new(Mutex::new(agent));
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();

    let screen = ScreenGuard::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend).context("failed to initialize terminal")?;
    terminal.clear().context("failed to clear terminal")?;

    for message in historical_messages {
        state.push_history(message);
    }
    let mut active_cancel = None;
    if let Some(prompt) = initial_prompt.filter(|prompt| !prompt.trim().is_empty()) {
        let images = state.take_pending_images();
        start_run(
            Arc::clone(&agent),
            prompt,
            images,
            &event_tx,
            &mut state,
            &mut active_cancel,
        );
    }

    let mut deleted_session = None;
    let mut last_frame = Instant::now();
    'ui: loop {
        while let Ok(agent_event) = event_rx.try_recv() {
            state.apply_agent_event(agent_event);
            if !state.running {
                active_cancel = None;
            }
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
                        Err(_) => state.push_error(
                            "The agent is busy; wait for the current turn to finish.".to_string(),
                        ),
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
                        Err(_) => state.push_error(
                            "The agent is busy; wait for the current turn to finish.".to_string(),
                        ),
                    },
                    UiAction::NewSession => match agent.try_lock() {
                        Ok(mut agent) => match agent.new_session() {
                            Ok(()) => {
                                state.reset_session();
                                state.sync_from_agent(&agent);
                                state.push_notice("Started a new session.".to_string());
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state.push_error(
                            "The agent is busy; wait for the current turn to finish.".to_string(),
                        ),
                    },
                    UiAction::DeleteSession => match agent.try_lock() {
                        Ok(mut agent) => match agent.delete_session() {
                            Ok(id) => {
                                deleted_session = Some(id);
                                break 'ui;
                            }
                            Err(error) => state.push_error(format!("{error:#}")),
                        },
                        Err(_) => state.push_error(
                            "The agent is busy; wait for the current turn to finish.".to_string(),
                        ),
                    },
                    UiAction::AttachImage(path) => match ImageAttachment::load(&path, &state.cwd) {
                        Ok(image) => {
                            let name = image.name.clone();
                            state.pending_images.push(image);
                            state.push_notice(format!("Attached {name} to the next prompt."));
                        }
                        Err(error) => state.push_error(format!("{error:#}")),
                    },
                }
            }
            Event::Paste(text) => state.editor.insert_str(&text),
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

fn start_run(
    agent: Arc<Mutex<Agent>>,
    prompt: String,
    images: Vec<ImageAttachment>,
    event_tx: &mpsc::UnboundedSender<AgentEvent>,
    state: &mut UiState,
    active_cancel: &mut Option<CancellationToken>,
) {
    state.push_user(prompt.clone(), &images);
    state.running = true;
    let cancel = CancellationToken::new();
    *active_cancel = Some(cancel.clone());
    let tx = event_tx.clone();
    tokio::spawn(async move {
        let result = agent.lock().await.run(&prompt, images, &tx, &cancel).await;
        if let Err(error) = result {
            let _ = tx.send(AgentEvent::Error {
                message: format!("{error:#}"),
            });
        }
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
    NewSession,
    DeleteSession,
    AttachImage(PathBuf),
}

fn handle_key(
    key: KeyEvent,
    state: &mut UiState,
    active_cancel: Option<&CancellationToken>,
) -> UiAction {
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
                "delete" if argument.eq_ignore_ascii_case("confirm") => {
                    if state.delete_confirmation == DeleteConfirmation::Pending {
                        return UiAction::DeleteSession;
                    }
                    state.push_error(
                        "Run /delete first, then /delete confirm to permanently delete this session."
                            .to_string(),
                    );
                }
                "delete" if argument.is_empty() => {
                    state.delete_confirmation = DeleteConfirmation::Pending;
                    state.push_notice(
                        "Delete this session? This cannot be undone. Run /delete confirm to continue."
                            .to_string(),
                    );
                }
                "delete" => state.push_error("Use /delete or /delete confirm.".to_string()),
                "image" if argument.eq_ignore_ascii_case("clear") => {
                    state.pending_images.clear();
                    state.push_notice("Cleared pending images.".to_string());
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
                "status" => {
                    let notice = state.status_notice();
                    state.push_notice(notice);
                }
                "help" => state.push_notice(
                    "Commands: /model [ID], /reasoning [LEVEL], /image [PATH|clear], /status, /new, /delete, /clear, /help, /quit"
                        .to_string(),
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
struct UiState {
    model: String,
    provider: Option<String>,
    reasoning_effort: ReasoningEffort,
    endpoint: String,
    cwd: std::path::PathBuf,
    model_choices: Vec<ModelChoice>,
    reasoning_choices: Vec<ReasoningEffort>,
    messages: Vec<ViewMessage>,
    editor: Editor,
    running: bool,
    current_assistant: Option<usize>,
    status: String,
    usage: Usage,
    context_tokens: u64,
    context_window: u64,
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
}

impl UiState {
    fn new(model: String, endpoint: String, cwd: std::path::PathBuf) -> Self {
        Self {
            model,
            provider: None,
            reasoning_effort: ReasoningEffort::Off,
            endpoint,
            cwd,
            model_choices: Vec::new(),
            reasoning_choices: ReasoningEffort::ALL.to_vec(),
            messages: Vec::new(),
            editor: Editor::default(),
            running: false,
            current_assistant: None,
            status: "ready".to_string(),
            usage: Usage::default(),
            context_tokens: 0,
            context_window: 128_000,
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
        }
    }

    fn sync_from_agent(&mut self, agent: &Agent) {
        self.model = agent.model().to_string();
        self.provider = agent.provider().map(ToString::to_string);
        self.reasoning_effort = agent.reasoning_effort();
        self.endpoint = agent.endpoint().to_string();
        self.model_choices = agent.model_choices();
        self.reasoning_choices = agent.available_reasoning_efforts();
        self.usage = agent.total_usage();
        self.context_tokens = agent.context_tokens();
        self.context_window = agent.context_window();
        self.usage_estimated = agent.usage_estimated();
        self.mcp_server_count = agent.mcp_server_count();
        self.mcp_tool_count = agent.mcp_tool_count();
    }

    fn push_history(&mut self, message: ChatMessage) {
        match message.role {
            MessageRole::System => {}
            MessageRole::User => {
                let content =
                    format_user_content(message.content.unwrap_or_default(), &message.images);
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
                content: message.content.unwrap_or_default(),
                reasoning: message.reasoning_content.unwrap_or_default(),
                tool_id: None,
                running: false,
            }),
            MessageRole::Tool => self.messages.push(ViewMessage {
                role: ViewRole::Tool,
                title: "tool".to_string(),
                content: message.content.unwrap_or_default(),
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
            content: format_user_content(prompt, images),
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
            .map(|image| format!("- {}", image.name))
            .collect::<Vec<_>>()
            .join("\n");
        format!("Images attached to the next prompt:\n{names}")
    }

    fn push_notice(&mut self, content: String) {
        self.messages.push(ViewMessage {
            role: ViewRole::Notice,
            title: "MCode".to_string(),
            content,
            reasoning: String::new(),
            tool_id: None,
            running: false,
        });
        self.follow_tail = true;
    }

    fn push_error(&mut self, content: String) {
        self.messages.push(ViewMessage {
            role: ViewRole::Error,
            title: "error".to_string(),
            content,
            reasoning: String::new(),
            tool_id: None,
            running: false,
        });
        self.follow_tail = true;
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
            let name = choice
                .name
                .as_deref()
                .map_or_else(String::new, |name| format!(" ({name})"));
            let reasoning = if choice.reasoning { ", reasoning" } else { "" };
            lines.push(format!(
                "{selected} {}/{}{} - {} context{}",
                choice.provider,
                choice.id,
                name,
                format_tokens(choice.context_window),
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
        let percent = format_context_percent(self.context_tokens, self.context_window);
        format!(
            "Model: {qualified_model}\nReasoning: {}\nContext: {estimate}{}/{} ({percent}%)\nTokens: {estimate}in {} out {}\nMCP: {} server(s), {} tool(s)\nEndpoint: {}\nWorking directory: {}",
            self.reasoning_effort,
            format_tokens(self.context_tokens),
            format_tokens(self.context_window),
            format_tokens(self.usage.prompt_tokens),
            format_tokens(self.usage.completion_tokens),
            self.mcp_server_count,
            self.mcp_tool_count,
            self.endpoint,
            self.cwd.display()
        )
    }

    fn apply_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::RunStarted => {
                self.status = "working".to_string();
                self.running = true;
            }
            AgentEvent::AssistantStarted => {
                self.messages.push(ViewMessage {
                    role: ViewRole::Assistant,
                    title: "assistant".to_string(),
                    content: String::new(),
                    reasoning: String::new(),
                    tool_id: None,
                    running: true,
                });
                self.current_assistant = Some(self.messages.len() - 1);
            }
            AgentEvent::TextDelta { text } => {
                if let Some(message) = self
                    .current_assistant
                    .and_then(|index| self.messages.get_mut(index))
                {
                    message.content.push_str(&text);
                }
            }
            AgentEvent::ReasoningDelta { text } => {
                if let Some(message) = self
                    .current_assistant
                    .and_then(|index| self.messages.get_mut(index))
                {
                    message.reasoning.push_str(&text);
                }
            }
            AgentEvent::ToolStarted {
                id,
                name,
                arguments,
            } => {
                if let Some(index) = self.current_assistant {
                    if self.messages.get(index).is_some_and(|message| {
                        message.content.is_empty() && message.reasoning.is_empty()
                    }) {
                        self.messages.remove(index);
                    } else if let Some(message) = self.messages.get_mut(index) {
                        message.running = false;
                    }
                }
                self.current_assistant = None;
                let arguments = serde_json::from_str::<serde_json::Value>(&arguments)
                    .ok()
                    .and_then(|value| serde_json::to_string_pretty(&value).ok())
                    .unwrap_or(arguments);
                self.messages.push(ViewMessage {
                    role: ViewRole::Tool,
                    title: name,
                    content: arguments,
                    reasoning: String::new(),
                    tool_id: Some(id),
                    running: true,
                });
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
                    message.title = name;
                    message.content = truncate_for_ui(&output);
                    message.running = false;
                    if is_error {
                        message.role = ViewRole::Error;
                    }
                }
            }
            AgentEvent::Usage {
                usage,
                context_tokens,
                context_window,
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
                self.usage_estimated = estimated;
            }
            AgentEvent::RunFinished => self.finish_run("ready"),
            AgentEvent::Cancelled => self.finish_run("cancelled"),
            AgentEvent::Error { message } => {
                self.finish_run("error");
                self.messages.push(ViewMessage {
                    role: ViewRole::Error,
                    title: "error".to_string(),
                    content: message,
                    reasoning: String::new(),
                    tool_id: None,
                    running: false,
                });
            }
        }
        self.follow_tail = true;
    }

    fn finish_run(&mut self, status: &str) {
        if let Some(message) = self
            .current_assistant
            .and_then(|index| self.messages.get_mut(index))
        {
            message.running = false;
        }
        self.current_assistant = None;
        self.running = false;
        self.status = status.to_string();
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
    let cwd = state.cwd.to_string_lossy();
    let estimate = if state.usage_estimated { "~" } else { "" };
    let left = format!(
        " {estimate}in {} out {} | context {estimate}{}/{} ({}%)",
        format_tokens(state.usage.prompt_tokens),
        format_tokens(state.usage.completion_tokens),
        format_tokens(state.context_tokens),
        format_tokens(state.context_window),
        format_context_percent(state.context_tokens, state.context_window)
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
        assert!(rendered.contains("prompt"));
        assert!(rendered.contains("/tmp/project"));
        assert!(rendered.contains("reasoning off"));
        assert!(rendered.contains("in 0 out 0"));
        assert!(rendered.contains("context 0/128k"));
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
    fn slash_reasoning_returns_an_independent_runtime_action() {
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
    fn formats_compact_token_counts_and_context_percent() {
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1_250), "1.2k");
        assert_eq!(format_tokens(12_500), "13k");
        assert_eq!(format_context_percent(24_000, 128_000), "18.7");
    }
}
