//! Explicit, paid smoke test. It is ignored by every normal test command.

use std::process::Command;

#[test]
#[ignore = "requires OPENAI_API_KEY and makes a paid external API request"]
fn openai_responses_smoke() {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("set OPENAI_API_KEY before explicitly running this ignored test");
    let model =
        std::env::var("MCODE_REAL_API_MODEL").unwrap_or_else(|_| "gpt-5.6-terra".to_string());
    let project = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mcode"))
        .args([
            "exec",
            "--search",
            "--model",
            &model,
            "Reply with exactly: MCODE_SMOKE_OK",
        ])
        .env("OPENAI_API_KEY", api_key)
        .env_remove("OPENAI_MODEL")
        .current_dir(project.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("MCODE_SMOKE_OK"));
}
