use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::config::ReasoningEffort;

#[derive(Debug, Parser)]
#[command(
    name = "mcode",
    version,
    about = "MCode, a focused OpenAI-compatible coding agent",
    args_conflicts_with_subcommands = true,
    after_help = "Examples:\n  mcode\n  mcode -i screenshot.png \"inspect this UI\"\n  mcode --search \"check the latest dependency release\"\n  mcode exec \"fix the failing tests\"\n  mcode resume\n  mcode resume <SESSION_ID>\n  mcode sessions\n  mcode doctor\n  mcode delete <SESSION_ID>"
)]
pub struct Cli {
    /// Optional image(s) to attach to the initial prompt.
    #[arg(
        long = "image",
        short = 'i',
        value_name = "FILE",
        value_delimiter = ',',
        global = true
    )]
    pub images: Vec<PathBuf>,

    /// Model name sent to the OpenAI-compatible endpoint.
    #[arg(short = 'm', long, global = true)]
    pub model: Option<String>,

    /// Reasoning effort, selected independently from the model.
    #[arg(
        short = 'r',
        long = "reasoning",
        visible_alias = "reasoning-effort",
        global = true,
        value_enum
    )]
    pub reasoning_effort: Option<ReasoningEffort>,

    /// API root, such as <https://api.openai.com/v1>.
    #[arg(long, global = true)]
    pub base_url: Option<String>,

    /// Environment variable containing the API key.
    #[arg(long, global = true)]
    pub api_key_env: Option<String>,

    /// Context window used for status and context usage calculations.
    #[arg(long, global = true, value_name = "TOKENS")]
    pub context_window: Option<u64>,

    /// Maximum number of input tokens accepted by the model.
    #[arg(long, global = true, value_name = "TOKENS")]
    pub max_input_tokens: Option<u64>,

    /// Run as if started in this directory.
    #[arg(
        short = 'C',
        long = "cd",
        visible_alias = "cwd",
        global = true,
        value_name = "DIR"
    )]
    pub cwd: Option<PathBuf>,

    /// Do not persist a new interactive session.
    #[arg(long, global = true)]
    pub no_session: bool,

    /// Maximum model/tool cycles for one request.
    #[arg(long, global = true, value_name = "N")]
    pub max_tool_turns: Option<usize>,

    /// Provider request timeout in seconds.
    #[arg(long, global = true, value_name = "SECONDS")]
    pub request_timeout: Option<u64>,

    /// Enable live web search through the hosted Responses API.
    #[arg(long, global = true)]
    pub search: bool,

    /// Run shell and MCP tools without confirmation, disabling the execution safeguard.
    #[arg(
        long = "dangerously-bypass-approvals",
        visible_alias = "dangerously-bypass-approvals-and-sandbox",
        global = true
    )]
    pub dangerously_bypass_approvals: bool,

    #[command(subcommand)]
    pub command: Option<Command>,

    /// Optional initial prompt for interactive mode.
    #[arg(value_name = "PROMPT", trailing_var_arg = true)]
    pub prompt: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run a prompt non-interactively.
    #[command(alias = "e")]
    Exec(ExecArgs),

    /// Resume the latest or a selected session.
    Resume(ResumeArgs),

    /// Permanently delete a saved session.
    Delete(DeleteArgs),

    /// List saved sessions for the current working directory.
    Sessions(OutputArgs),

    /// Diagnose local configuration without making an API request.
    Doctor(OutputArgs),
}

#[derive(Debug, Args)]
pub struct OutputArgs {
    /// Emit machine-readable JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct ExecArgs {
    /// Emit newline-delimited JSON events.
    #[arg(long)]
    pub json: bool,

    /// Prompt text. Use '-' or omit it to read standard input.
    #[arg(value_name = "PROMPT", trailing_var_arg = true)]
    pub prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub struct ResumeArgs {
    /// Session ID fragment, JSONL path, or 'last'. Defaults to the latest session.
    pub session: Option<String>,

    /// Optional prompt to submit immediately after resuming.
    #[arg(value_name = "PROMPT", trailing_var_arg = true)]
    pub prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub struct DeleteArgs {
    /// Session UUID or an unambiguous UUID fragment.
    #[arg(value_name = "SESSION")]
    pub session: String,

    /// Delete without prompting. SESSION must be a complete UUID.
    #[arg(long, visible_alias = "yes")]
    pub force: bool,
}

impl Cli {
    #[must_use]
    pub fn root_prompt(&self) -> Option<String> {
        join_prompt(&self.prompt)
    }
}

#[must_use]
pub fn join_prompt(parts: &[String]) -> Option<String> {
    let prompt = parts.join(" ");
    (!prompt.trim().is_empty()).then_some(prompt)
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn parses_codex_style_exec() {
        let cli = Cli::parse_from([
            "mcode",
            "exec",
            "--json",
            "--model",
            "test-model",
            "fix",
            "the",
            "tests",
        ]);
        assert_eq!(cli.model.as_deref(), Some("test-model"));
        let Some(Command::Exec(exec)) = cli.command else {
            panic!("expected exec");
        };
        assert!(exec.json);
        assert_eq!(join_prompt(&exec.prompt).as_deref(), Some("fix the tests"));
    }

    #[test]
    fn parses_reasoning_separately_from_model() {
        let cli = Cli::parse_from([
            "mcode",
            "--model",
            "test-model",
            "--reasoning",
            "high",
            "--context-window",
            "200000",
        ]);
        assert_eq!(cli.model.as_deref(), Some("test-model"));
        assert_eq!(cli.reasoning_effort, Some(ReasoningEffort::High));
        assert_eq!(cli.context_window, Some(200_000));
    }

    #[test]
    fn parses_resume_session_and_prompt() {
        let cli = Cli::parse_from(["mcode", "resume", "abc123", "continue", "working"]);
        let Some(Command::Resume(resume)) = cli.command else {
            panic!("expected resume");
        };
        assert_eq!(resume.session.as_deref(), Some("abc123"));
        assert_eq!(
            join_prompt(&resume.prompt).as_deref(),
            Some("continue working")
        );
    }

    #[test]
    fn parses_sessions_and_doctor_output_modes() {
        let sessions = Cli::parse_from(["mcode", "sessions", "--json"]);
        assert!(matches!(
            sessions.command,
            Some(Command::Sessions(OutputArgs { json: true }))
        ));
        let doctor = Cli::parse_from(["mcode", "doctor"]);
        assert!(matches!(
            doctor.command,
            Some(Command::Doctor(OutputArgs { json: false }))
        ));
    }

    #[test]
    fn parses_codex_style_images_and_delete() {
        let cli = Cli::parse_from([
            "mcode",
            "delete",
            "8fd57e8e-55da-4a82-a28f-50a9f435742a",
            "--force",
            "--image",
            "one.png,two.jpg",
        ]);
        assert_eq!(
            cli.images,
            [PathBuf::from("one.png"), PathBuf::from("two.jpg")]
        );
        let Some(Command::Delete(delete)) = cli.command else {
            panic!("expected delete");
        };
        assert!(delete.force);
    }

    #[test]
    fn parses_explicit_unsafe_approval_bypass() {
        let cli = Cli::parse_from([
            "mcode",
            "exec",
            "--dangerously-bypass-approvals",
            "run tests",
        ]);
        assert!(cli.dangerously_bypass_approvals);
    }

    #[test]
    fn parses_live_web_search_flag() {
        let cli = Cli::parse_from(["mcode", "--search", "find", "current", "releases"]);
        assert!(cli.search);
        assert_eq!(cli.root_prompt().as_deref(), Some("find current releases"));
    }
}
