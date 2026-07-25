use std::sync::Arc;

use mcode::agent::{Agent, RunStatus};
use mcode::config::{AppConfig, ReasoningEffort};
use mcode::event::AgentEvent;
use mcode::session::Session;
use serde_json::{Value, json};
use tempfile::tempdir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn executes_a_tool_and_continues_the_model_turn() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let server_requests = Arc::clone(&requests);

    let server = tokio::spawn(async move {
        for index in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_json_request(&mut stream).await;
            server_requests.lock().await.push(request);

            let event = if index == 0 {
                json!({
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_write",
                                "function": {
                                    "name": "write_file",
                                    "arguments": "{\"path\":\"result.txt\",\"content\":\"created by tool\"}"
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                })
            } else {
                json!({
                    "choices": [{
                        "delta": {"content": "Done."},
                        "finish_reason": "stop"
                    }]
                })
            };
            let body = format!("data: {event}\n\ndata: [DONE]\n\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
        }
    });

    let project = tempdir().unwrap();
    let config = AppConfig {
        model: "test-model".to_string(),
        provider: None,
        reasoning_effort: ReasoningEffort::High,
        reasoning_value: Some("high".to_string()),
        base_url: format!("http://{address}/v1"),
        api_key: Some("test-key".to_string()),
        context_window: 200_000,
        supports_reasoning_effort: true,
        supports_usage_in_streaming: true,
        cwd: project.path().canonicalize().unwrap(),
        max_tool_turns: 4,
        request_timeout_secs: 10,
        model_profiles: Vec::new(),
        mcp_servers: Vec::new(),
    };
    let session = Session::create(
        project.path(),
        &config.model,
        config.reasoning_effort,
        false,
    )
    .unwrap();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let status = agent
        .run(
            "create result.txt",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    drop(tx);
    server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    assert_eq!(
        std::fs::read_to_string(project.path().join("result.txt")).unwrap(),
        "created by tool"
    );

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 2);
    let second_messages = requests[1]["messages"].as_array().unwrap();
    assert!(
        second_messages
            .iter()
            .any(|message| message["role"] == "tool")
    );
    assert!(
        second_messages
            .iter()
            .any(|message| message["tool_calls"][0]["id"] == "call_write")
    );
    assert_eq!(requests[0]["model"], "test-model");
    assert_eq!(requests[0]["reasoning_effort"], "high");
    assert_eq!(requests[0]["stream_options"]["include_usage"], true);

    let mut saw_tool = false;
    let mut text = String::new();
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::ToolFinished { is_error, .. } => {
                assert!(!is_error);
                saw_tool = true;
            }
            AgentEvent::TextDelta { text: delta } => text.push_str(&delta),
            _ => {}
        }
    }
    assert!(saw_tool);
    assert_eq!(text, "Done.");
}

#[tokio::test]
async fn omits_request_fields_disabled_by_pi_compat() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let (request, user_agent) = read_http_request(&mut stream).await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Done.\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
        (request, user_agent)
    });

    let project = tempdir().unwrap();
    let config = AppConfig {
        model: "compat-model".to_string(),
        provider: None,
        reasoning_effort: ReasoningEffort::High,
        reasoning_value: Some("high".to_string()),
        base_url: format!("http://{address}/v1"),
        api_key: None,
        context_window: 128_000,
        supports_reasoning_effort: false,
        supports_usage_in_streaming: false,
        cwd: project.path().canonicalize().unwrap(),
        max_tool_turns: 1,
        request_timeout_secs: 10,
        model_profiles: Vec::new(),
        mcp_servers: Vec::new(),
    };
    let session = Session::create(
        project.path(),
        &config.model,
        config.reasoning_effort,
        false,
    )
    .unwrap();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, _rx) = mpsc::unbounded_channel();
    let status = agent
        .run(
            "test compatibility",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
        )
        .await
        .unwrap();
    let (request, user_agent) = server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    assert!(request.get("reasoning_effort").is_none());
    assert!(request.get("stream_options").is_none());
    assert_eq!(user_agent, "codex_cli_rs/0.145.0");
}

#[tokio::test]
async fn exec_command_streams_text_from_compatible_endpoint() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_json_request(&mut stream).await;
        let event = json!({
            "choices": [{
                "delta": {"content": "Hello from fixture."},
                "finish_reason": "stop"
            }]
        });
        let body = format!("data: {event}\n\ndata: [DONE]\n\n");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
        request
    });

    let project = tempdir().unwrap();
    std::fs::create_dir(project.path().join(".mcode")).unwrap();
    std::fs::write(
        project.path().join(".mcode/settings.json"),
        r#"{
            "defaultModel": "fixture-model",
            "defaultThinkingLevel": "low"
        }"#,
    )
    .unwrap();
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcode"))
        .arg("exec")
        .arg("--base-url")
        .arg(format!("http://{address}/v1"))
        .arg("say hello")
        .env("OPENAI_API_KEY", "fixture-key")
        .env_remove("OPENAI_MODEL")
        .env_remove("OPENAI_REASONING_EFFORT")
        .current_dir(project.path())
        .output()
        .await
        .unwrap();
    let request = server.await.unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "Hello from fixture.\n"
    );
    assert_eq!(request["model"], "fixture-model");
    assert_eq!(request["reasoning_effort"], "low");
    assert_eq!(request["messages"][1]["content"], "say hello");
}

#[tokio::test]
async fn exec_command_sends_image_content_parts() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_json_request(&mut stream).await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Image received.\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
        request
    });

    let project = tempdir().unwrap();
    std::fs::write(
        project.path().join("pixel.png"),
        [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0],
    )
    .unwrap();
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcode"))
        .arg("exec")
        .arg("--base-url")
        .arg(format!("http://{address}/v1"))
        .arg("--model")
        .arg("vision-model")
        .arg("--image")
        .arg("pixel.png")
        .arg("inspect this image")
        .env("OPENAI_API_KEY", "fixture-key")
        .current_dir(project.path())
        .output()
        .await
        .unwrap();
    let request = server.await.unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let content = request["messages"][1]["content"].as_array().unwrap();
    assert_eq!(
        content[0],
        json!({"type": "text", "text": "inspect this image"})
    );
    assert_eq!(content[1]["type"], "image_url");
    assert!(
        content[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
}

#[tokio::test]
async fn delete_command_removes_only_the_selected_project_session() {
    let home = tempdir().unwrap();
    let project = tempdir().unwrap();
    let base = home.path().join(".mcode/sessions");
    let selected =
        Session::create_in(&base, project.path(), "test-model", ReasoningEffort::Off).unwrap();
    let remaining =
        Session::create_in(&base, project.path(), "test-model", ReasoningEffort::Off).unwrap();
    let selected_path = selected.path().unwrap().to_path_buf();
    let remaining_path = remaining.path().unwrap().to_path_buf();

    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcode"))
        .arg("delete")
        .arg(selected.id().to_string())
        .arg("--force")
        .env("HOME", home.path())
        .current_dir(project.path())
        .output()
        .await
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!selected_path.exists());
    assert!(remaining_path.exists());
    assert!(String::from_utf8_lossy(&output.stdout).contains(&selected.id().to_string()));
}

async fn read_json_request(stream: &mut TcpStream) -> Value {
    read_http_request(stream).await.0
}

async fn read_http_request(stream: &mut TcpStream) -> (Value, String) {
    let mut bytes = Vec::new();
    let header_end = loop {
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).await.unwrap();
        assert!(read > 0, "connection closed before HTTP headers");
        bytes.extend_from_slice(&chunk[..read]);
    };

    let headers = String::from_utf8_lossy(&bytes[..header_end]);
    let user_agent = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("user-agent")
                .then(|| value.trim().to_string())
        })
        .expect("request must include User-Agent");
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .expect("request must include Content-Length");
    while bytes.len() < header_end + content_length {
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).await.unwrap();
        assert!(read > 0, "connection closed before HTTP body");
        bytes.extend_from_slice(&chunk[..read]);
    }
    (
        serde_json::from_slice(&bytes[header_end..header_end + content_length]).unwrap(),
        user_agent,
    )
}
