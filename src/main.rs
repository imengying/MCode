use std::env;
use std::io::{self, IsTerminal, Read, Write};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::Parser;
use mcode::agent::{Agent, RunStatus};
use mcode::cli::{Cli, Command, join_prompt};
use mcode::config::{AppConfig, ConfigOverrides};
use mcode::event::AgentEvent;
use mcode::protocol::ImageAttachment;
use mcode::session::Session;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let root_prompt = cli.root_prompt();
    let model_was_overridden =
        cli.model.is_some() || env::var("OPENAI_MODEL").is_ok_and(|value| !value.trim().is_empty());
    let reasoning_was_overridden = cli.reasoning_effort.is_some()
        || env::var("OPENAI_REASONING_EFFORT").is_ok_and(|value| !value.trim().is_empty());
    let overrides = ConfigOverrides {
        model: cli.model.clone(),
        reasoning_effort: cli.reasoning_effort,
        base_url: cli.base_url.clone(),
        api_key_env: cli.api_key_env.clone(),
        context_window: cli.context_window,
        cwd: cli.cwd.clone(),
        max_tool_turns: cli.max_tool_turns,
        request_timeout_secs: cli.request_timeout,
    };
    let mut config = AppConfig::load(&overrides)?;

    match cli.command {
        Some(Command::Exec(args)) => {
            let prompt = resolve_exec_prompt(&args.prompt)?;
            let images = load_images(&cli.images, &config.cwd)?;
            let session =
                Session::create(&config.cwd, &config.model, config.reasoning_effort, false)?;
            let agent = Agent::new(&config, session).await?;
            run_exec(agent, prompt, images, args.json).await
        }
        Some(Command::Resume(args)) => {
            let session = Session::resume(&config.cwd, args.session.as_deref())?;
            match (model_was_overridden, reasoning_was_overridden) {
                (false, false) => config
                    .select_model_and_reasoning(session.model(), session.reasoning_effort())?,
                (false, true) => config.select_model(session.model())?,
                (true, false) => {
                    config.select_reasoning_effort(session.reasoning_effort())?;
                }
                (true, true) => {}
            }
            let agent = Agent::new(&config, session).await?;
            let prompt = join_prompt(&args.prompt);
            let images = load_images(&cli.images, &config.cwd)?;
            if io::stdin().is_terminal() && io::stdout().is_terminal() {
                mcode::ui::run_interactive(agent, prompt, images)
            } else {
                let prompt = prompt.ok_or_else(|| {
                    anyhow::anyhow!(
                        "resume requires a prompt when standard input is not interactive"
                    )
                })?;
                run_exec(agent, prompt, images, false).await
            }
        }
        Some(Command::Delete(args)) => {
            if args.force && Uuid::parse_str(&args.session).is_err() {
                bail!("--force requires a complete session UUID");
            }
            if !args.force && !confirm_session_delete(&args.session)? {
                println!("Delete cancelled.");
                return Ok(());
            }
            let id = Session::delete(&config.cwd, &args.session)?;
            println!("Deleted session {id}.");
            Ok(())
        }
        None if io::stdin().is_terminal() && io::stdout().is_terminal() => {
            let session = Session::create(
                &config.cwd,
                &config.model,
                config.reasoning_effort,
                !cli.no_session,
            )?;
            let agent = Agent::new(&config, session).await?;
            let images = load_images(&cli.images, &config.cwd)?;
            mcode::ui::run_interactive(agent, root_prompt, images)
        }
        None => {
            let prompt = match root_prompt {
                Some(prompt) => prompt,
                None => read_stdin_prompt()?,
            };
            let session =
                Session::create(&config.cwd, &config.model, config.reasoning_effort, false)?;
            let agent = Agent::new(&config, session).await?;
            let images = load_images(&cli.images, &config.cwd)?;
            run_exec(agent, prompt, images, false).await
        }
    }
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
        bail!("cannot confirm deletion without a terminal; rerun with --force and a session UUID");
    }
    eprintln!("Permanently delete session {session}?");
    eprintln!("This cannot be undone.");
    eprint!("Continue? [y/N]: ");
    io::stderr()
        .flush()
        .context("failed to flush deletion prompt")?;
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .context("failed to read deletion confirmation")?;
    Ok(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn resolve_exec_prompt(parts: &[String]) -> Result<String> {
    if parts.is_empty() || (parts.len() == 1 && parts[0] == "-") {
        return read_stdin_prompt();
    }
    join_prompt(parts).ok_or_else(|| anyhow::anyhow!("prompt cannot be empty"))
}

fn read_stdin_prompt() -> Result<String> {
    if io::stdin().is_terminal() {
        bail!("no prompt supplied; pass text or pipe a prompt on standard input");
    }
    let mut prompt = String::new();
    io::stdin()
        .read_to_string(&mut prompt)
        .context("failed to read prompt from standard input")?;
    if prompt.trim().is_empty() {
        bail!("standard input did not contain a prompt");
    }
    Ok(prompt)
}

async fn run_exec(
    mut agent: Agent,
    prompt: String,
    images: Vec<ImageAttachment>,
    json: bool,
) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel();
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
        let result = agent.run(&prompt, images, &run_tx, &run_cancel).await;
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
                serde_json::to_string(&event).context("failed to encode JSON event")?
            );
            continue;
        }
        match event {
            AgentEvent::TextDelta { text } => {
                print!("{text}");
                io::stdout().flush().context("failed to flush stdout")?;
                streamed_text = true;
            }
            AgentEvent::ToolStarted { name, .. } => {
                if streamed_text {
                    println!();
                    streamed_text = false;
                }
                eprintln!("[tool] {name}");
            }
            AgentEvent::ToolFinished { name, is_error, .. } => {
                eprintln!(
                    "[{}] {name}",
                    if is_error { "tool error" } else { "tool done" }
                );
            }
            AgentEvent::Error { message } => eprintln!("error: {message}"),
            AgentEvent::RunFinished | AgentEvent::Cancelled if streamed_text => {
                println!();
                streamed_text = false;
            }
            AgentEvent::RunStarted
            | AgentEvent::AssistantStarted
            | AgentEvent::ReasoningDelta { .. }
            | AgentEvent::Usage { .. }
            | AgentEvent::RunFinished
            | AgentEvent::Cancelled => {}
        }
    }

    let status = task.await.context("agent task failed")??;
    if status == RunStatus::Cancelled {
        bail!("cancelled");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_exec_prompt() {
        assert_eq!(
            resolve_exec_prompt(&["fix".to_string(), "tests".to_string()]).unwrap(),
            "fix tests"
        );
    }
}
