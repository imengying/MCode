use std::fmt::Write as _;
use std::sync::Arc;

use mcode::agent::{Agent, RunStatus};
use mcode::approval::ApprovalGate;
use mcode::config::{
    ApiProtocol, AppConfig, CompactionSettings, ConfigOverrides, ModelCompat, ReasoningEffort,
    WebSearchMode, WebSearchSettings,
};
use mcode::event::AgentEvent;
use mcode::protocol::{ChatMessage, FunctionCall, ToolCall, Usage};
use mcode::session::{Session, SessionMetadata, ToolReplayPolicy};
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
        provider: "xai".to_string(),
        api: ApiProtocol::ChatCompletions,
        reasoning_effort: ReasoningEffort::High,
        reasoning_value: Some("high".to_string()),
        base_url: format!("http://{address}/v1"),
        api_key: Some("test-key".to_string()),
        context_window: 200_000,
        max_input_tokens: 200_000,
        max_output_tokens: None,
        compat: ModelCompat {
            reasoning_effort: true,
            usage_in_streaming: true,
            finish_reason: true,
            strict_tools: false,
        },
        cwd: project.path().canonicalize().unwrap(),
        request_timeout_secs: 10,
        compaction: CompactionSettings::default(),
        web_search: WebSearchSettings::default(),
        model_profiles: Vec::new(),
        mcp_servers: Vec::new(),
        reload_overrides: ConfigOverrides::default(),
    };
    let session = Session::create(project.path(), SessionMetadata::from(&config), false).unwrap();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let approvals = ApprovalGate::default();
    let status = agent
        .run(
            "create result.txt",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
            &approvals,
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
    assert!(
        requests[0]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .all(|tool| tool["function"].get("strict").is_none())
    );

    let mut saw_file_change = false;
    let mut text = String::new();
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::ToolFinished {
                is_error,
                file_change,
                ..
            } => {
                assert!(!is_error);
                let change = file_change.expect("write_file should report a structured change");
                assert_eq!(change.path, "result.txt");
                assert_eq!((change.added_lines, change.removed_lines), (1, 0));
                saw_file_change = true;
            }
            AgentEvent::TextDelta { text: delta } => text.push_str(&delta),
            _ => {}
        }
    }
    assert!(saw_file_change);
    let persisted_change = agent
        .messages()
        .iter()
        .find_map(|message| message.file_change.as_ref())
        .expect("file change should be persisted with the tool message");
    assert_eq!(persisted_change.path, "result.txt");
    assert_eq!(text, "Done.");
}

#[tokio::test]
async fn responses_api_runs_local_tools_and_hosted_web_search() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let server_requests = Arc::clone(&requests);
    let server = tokio::spawn(async move {
        for index in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            server_requests
                .lock()
                .await
                .push(read_json_request(&mut stream).await);
            let events = if index == 0 {
                vec![
                    json!({
                        "type": "response.output_item.done",
                        "item": {
                            "type": "reasoning",
                            "id": "rs_reasoning",
                            "encrypted_content": "encrypted-fixture-state",
                            "summary": [{
                                "type": "summary_text",
                                "text": "I should inspect the requested file."
                            }]
                        }
                    }),
                    json!({
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": "fc_read",
                            "call_id": "call_read",
                            "name": "read_file",
                            "arguments": "{\"path\":\"README.md\",\"limit\":1}"
                        }
                    }),
                    responses_completed("resp_tools", 80, 12),
                ]
            } else {
                vec![
                    json!({
                        "type": "response.reasoning_text.delta",
                        "delta": "I should use the "
                    }),
                    json!({
                        "type": "response.reasoning_text.done",
                        "text": "I should use the hosted search result."
                    }),
                    json!({
                        "type": "response.output_item.done",
                        "item": {
                            "type": "reasoning",
                            "id": "rs_deepseek",
                            "content": [{
                                "type": "reasoning_text",
                                "text": "I should use the hosted search result."
                            }],
                            "summary": []
                        }
                    }),
                    json!({
                        "type": "response.output_item.added",
                        "item": {
                            "type": "web_search_call",
                            "id": "ws_current",
                            "status": "in_progress"
                        }
                    }),
                    json!({
                        "type": "response.output_item.done",
                        "item": {
                            "type": "web_search_call",
                            "id": "ws_current",
                            "status": "completed",
                            "action": {"type": "search", "query": "current Rust release"}
                        }
                    }),
                    json!({
                        "type": "response.output_text.delta",
                        "delta": "Rust is current."
                    }),
                    json!({
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": "Rust is current.",
                                "annotations": [{
                                    "type": "url_citation",
                                    "start_index": 0,
                                    "end_index": 4,
                                    "url": "https://www.rust-lang.org/",
                                    "title": "Rust Programming Language"
                                }]
                            }]
                        }
                    }),
                    responses_completed("resp_search", 120, 24),
                ]
            };
            write_responses_sse(&mut stream, events).await;
        }
    });

    let project = tempdir().unwrap();
    std::fs::write(project.path().join("README.md"), "fixture\n").unwrap();
    std::fs::write(
        project.path().join("AGENTS.md"),
        "Always preserve the fixture sentinel.",
    )
    .unwrap();
    let config = AppConfig {
        model: "deepseek-test".to_string(),
        provider: "deepseek".to_string(),
        api: ApiProtocol::Responses,
        reasoning_effort: ReasoningEffort::Low,
        reasoning_value: Some("low".to_string()),
        base_url: format!("http://{address}/v1"),
        api_key: Some("test-key".to_string()),
        context_window: 200_000,
        max_input_tokens: 200_000,
        max_output_tokens: None,
        compat: ModelCompat {
            reasoning_effort: true,
            usage_in_streaming: true,
            finish_reason: true,
            strict_tools: true,
        },
        cwd: project.path().canonicalize().unwrap(),
        request_timeout_secs: 10,
        compaction: CompactionSettings::default(),
        web_search: WebSearchSettings {
            mode: WebSearchMode::Live,
            ..WebSearchSettings::default()
        },
        model_profiles: Vec::new(),
        mcp_servers: Vec::new(),
        reload_overrides: ConfigOverrides::default(),
    };
    let session = Session::create(project.path(), SessionMetadata::from(&config), false).unwrap();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let status = agent
        .run(
            "check the README and current Rust release",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
            &ApprovalGate::default(),
        )
        .await
        .unwrap();
    drop(tx);
    server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    let requests = requests.lock().await;
    let web_tool = requests[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["type"] == "web_search")
        .unwrap();
    assert_eq!(web_tool["external_web_access"], true);
    assert!(web_tool.get("indexed_web_access").is_none());
    let read_tool = requests[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "read_file")
        .unwrap();
    assert_eq!(read_tool["type"], "function");
    assert!(read_tool.get("function").is_none());
    assert_eq!(read_tool["strict"], true);
    assert!(
        requests[0]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| { tool["type"] == "function" && tool["name"] == "fetch_content" })
    );
    assert!(
        requests[0]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|tool| tool["type"] == "function")
            .all(|tool| tool["strict"] == true)
    );
    assert_eq!(requests[0]["reasoning"]["effort"], "low");
    assert_eq!(requests[0]["reasoning"]["summary"], "auto");
    assert_eq!(
        requests[0]["include"],
        json!(["reasoning.encrypted_content"])
    );
    assert_eq!(
        requests[0]["instructions"],
        "Project instructions from AGENTS.md:\n\nAlways preserve the fixture sentinel."
    );
    let second_input = requests[1]["input"].as_array().unwrap();
    assert!(second_input.iter().any(|item| {
        item["type"] == "reasoning"
            && item["id"] == "rs_reasoning"
            && item["encrypted_content"] == "encrypted-fixture-state"
    }));
    assert!(
        second_input
            .iter()
            .any(|item| { item["type"] == "function_call" && item["call_id"] == "call_read" })
    );
    assert!(
        second_input.iter().any(|item| {
            item["type"] == "function_call_output" && item["call_id"] == "call_read"
        })
    );

    let events = drain_events(&mut rx);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::WebSearchStarted { id } if id == "ws_current"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::WebSearchFinished {
            action: mcode::protocol::WebSearchAction::Search {
                query: Some(query),
                ..
            },
            ..
        } if query == "current Rust release"
    )));
    let streamed = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TextDelta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert!(streamed.contains("Rust is current."));
    assert!(streamed.contains("[Rust Programming Language](https://www.rust-lang.org/)"));
    assert!(!events.iter().any(|event| matches!(
        event,
        AgentEvent::ReasoningSummaryDelta { text }
            if text.contains("hosted search result")
    )));
    assert!(
        agent
            .messages()
            .last()
            .and_then(|message| message.content.as_deref())
            .is_some_and(|content| content.contains("Sources:"))
    );
    let final_message = agent.messages().last().unwrap();
    assert_eq!(
        final_message.reasoning_content.as_deref(),
        Some("I should use the hosted search result.")
    );
    assert!(final_message.response_items.iter().any(|item| {
        item["type"] == "reasoning"
            && item["id"] == "rs_deepseek"
            && item["content"][0]["type"] == "reasoning_text"
    }));
}

#[tokio::test]
async fn chat_completions_exposes_local_web_access_tools() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_json_request(&mut stream).await;
        write_sse_text(&mut stream, "Search tools are available.", None).await;
        request
    });

    let project = tempdir().unwrap();
    let mut config = basic_chat_config(project.path(), address);
    config.web_search.mode = WebSearchMode::Live;
    let session = Session::create(project.path(), SessionMetadata::from(&config), false).unwrap();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, _rx) = mpsc::unbounded_channel();
    let status = agent
        .run(
            "find the current Rust release",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
            &ApprovalGate::default(),
        )
        .await
        .unwrap();
    let request = server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    let tools = request["tools"].as_array().unwrap();
    assert!(
        tools
            .iter()
            .any(|tool| tool["function"]["name"] == "web_search")
    );
    assert!(
        tools
            .iter()
            .any(|tool| tool["function"]["name"] == "fetch_content")
    );
    assert!(!tools.iter().any(|tool| tool["type"] == "web_search"));
}

#[tokio::test]
async fn shell_tool_is_denied_without_frontend_approval() {
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
                                "id": "call_shell",
                                "function": {
                                    "name": "shell",
                                    "arguments": "{\"command\":\"echo denied > should-not-exist.txt\"}"
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                })
            } else {
                json!({
                    "choices": [{
                        "delta": {"content": "The command was denied."},
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
        provider: "xai".to_string(),
        api: ApiProtocol::ChatCompletions,
        reasoning_effort: ReasoningEffort::Off,
        reasoning_value: None,
        base_url: format!("http://{address}/v1"),
        api_key: None,
        context_window: 128_000,
        max_input_tokens: 128_000,
        max_output_tokens: None,
        compat: ModelCompat {
            reasoning_effort: true,
            usage_in_streaming: true,
            finish_reason: true,
            strict_tools: false,
        },
        cwd: project.path().canonicalize().unwrap(),
        request_timeout_secs: 10,
        compaction: CompactionSettings::default(),
        web_search: WebSearchSettings::default(),
        model_profiles: Vec::new(),
        mcp_servers: Vec::new(),
        reload_overrides: ConfigOverrides::default(),
    };
    let session = Session::create(project.path(), SessionMetadata::from(&config), false).unwrap();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let status = agent
        .run(
            "run the command",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
            &ApprovalGate::default(),
        )
        .await
        .unwrap();
    drop(tx);
    server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    assert!(!project.path().join("should-not-exist.txt").exists());
    let requests = requests.lock().await;
    let tool_message = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .unwrap();
    assert!(
        tool_message["content"]
            .as_str()
            .unwrap()
            .contains("requires user approval")
    );

    let mut requested = false;
    let mut denied = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::ApprovalRequested { name, .. } if name == "shell" => requested = true,
            AgentEvent::ApprovalResolved { name, approved, .. } if name == "shell" && !approved => {
                denied = true;
            }
            _ => {}
        }
    }
    assert!(requested);
    assert!(denied);
}

#[tokio::test]
async fn omits_request_fields_disabled_by_pi_compat() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let (request, user_agent) = read_http_request(&mut stream).await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Done.\"},\"finish_reason\":\"stop\"}]}\n\n",
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
        provider: "xai".to_string(),
        api: ApiProtocol::ChatCompletions,
        reasoning_effort: ReasoningEffort::High,
        reasoning_value: Some("high".to_string()),
        base_url: format!("http://{address}/v1"),
        api_key: None,
        context_window: 128_000,
        max_input_tokens: 128_000,
        max_output_tokens: None,
        compat: ModelCompat {
            reasoning_effort: false,
            usage_in_streaming: false,
            finish_reason: true,
            strict_tools: false,
        },
        cwd: project.path().canonicalize().unwrap(),
        request_timeout_secs: 10,
        compaction: CompactionSettings::default(),
        web_search: WebSearchSettings::default(),
        model_profiles: Vec::new(),
        mcp_servers: Vec::new(),
        reload_overrides: ConfigOverrides::default(),
    };
    let session = Session::create(project.path(), SessionMetadata::from(&config), false).unwrap();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, _rx) = mpsc::unbounded_channel();
    let approvals = ApprovalGate::default();
    let status = agent
        .run(
            "test compatibility",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
            &approvals,
        )
        .await
        .unwrap();
    let (request, user_agent) = server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    assert!(request.get("reasoning_effort").is_none());
    assert!(request.get("stream_options").is_none());
    assert_eq!(user_agent, format!("mcode/{}", env!("CARGO_PKG_VERSION")));
}

#[tokio::test]
async fn recovers_from_stalled_requests_and_truncated_streams_without_persisting_partial_output() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        for attempt in 0..4 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_json_request(&mut stream).await;
            match attempt {
                0 => tokio::time::sleep(std::time::Duration::from_millis(1_100)).await,
                1 => write_http_error(&mut stream, 503, "temporarily unavailable").await,
                2 => {
                    let body = concat!(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"discard-me\"}}]}\n\n",
                        "data: [DONE]\n\n"
                    );
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    stream.write_all(response.as_bytes()).await.unwrap();
                    stream.shutdown().await.unwrap();
                }
                _ => write_sse_text(&mut stream, "kept-after-retry", Some((30, 5))).await,
            }
        }
    });

    let base = tempdir().unwrap();
    let project = tempdir().unwrap();
    let mut config = basic_chat_config(project.path(), address);
    config.request_timeout_secs = 1;
    let session =
        Session::create_in(base.path(), project.path(), SessionMetadata::from(&config)).unwrap();
    let session_path = session.path().unwrap().to_path_buf();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let status = agent
        .run(
            "retry a dropped stream",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
            &ApprovalGate::default(),
        )
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    assert_eq!(
        agent
            .messages()
            .last()
            .and_then(|message| message.content.as_deref()),
        Some("kept-after-retry")
    );
    assert_eq!(agent.total_usage().total_tokens, 35);
    let jsonl = std::fs::read_to_string(session_path).unwrap();
    assert!(!jsonl.contains("discard-me"));
    assert!(jsonl.contains("kept-after-retry"));
    let events = drain_events(&mut rx);
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::AssistantRetrying {
            attempt: 2,
            max_attempts: 4,
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::AssistantRetrying {
            attempt: 3,
            max_attempts: 4,
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::AssistantRetrying {
            attempt: 2,
            max_attempts: 2,
            ..
        }
    )));
}

#[tokio::test]
async fn exec_command_sends_image_content_parts() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_json_request(&mut stream).await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Image received.\"},\"finish_reason\":\"stop\"}]}\n\n",
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
    let mcode_home = project.path().join(".mcode-home");
    write_test_model_catalog(&mcode_home, "vision-model", true);
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcode"))
        .arg("exec")
        .arg("--base-url")
        .arg(format!("http://{address}/v1"))
        .arg("--model")
        .arg("xai/vision-model")
        .arg("--api-key-env")
        .arg("XAI_API_KEY")
        .arg("--image")
        .arg("pixel.png")
        .arg("inspect this image")
        .env("MCODE_HOME", &mcode_home)
        .env("XAI_API_KEY", "fixture-key")
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
    let content = request["messages"][0]["content"].as_array().unwrap();
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
async fn auto_compaction_summarizes_old_turns_before_the_next_request() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let server_requests = Arc::clone(&requests);
    let server = tokio::spawn(async move {
        for index in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_json_request(&mut stream).await;
            server_requests.lock().await.push(request);
            if index == 0 {
                write_sse_text(
                    &mut stream,
                    "## Goal\nContinue the fixture task\n\n## Progress\n### Done\n- [x] Old work",
                    Some((120, 30)),
                )
                .await;
            } else {
                write_sse_text(&mut stream, "Compaction worked.", Some((80, 5))).await;
            }
        }
    });

    let project = tempdir().unwrap();
    let config = compaction_test_config(project.path(), address, 4_000, 2_500, 60);
    let mut session =
        Session::create(project.path(), SessionMetadata::from(&config), false).unwrap();
    session
        .append(mcode::protocol::ChatMessage::user("old ".repeat(2_500)))
        .unwrap();
    session
        .append(mcode::protocol::ChatMessage::assistant(
            Some("old ".repeat(2_500)),
            None,
            Vec::new(),
        ))
        .unwrap();
    session
        .append(mcode::protocol::ChatMessage::user(
            "recent question ".repeat(8),
        ))
        .unwrap();
    session
        .append(mcode::protocol::ChatMessage::assistant(
            Some("recent answer ".repeat(8)),
            None,
            Vec::new(),
        ))
        .unwrap();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let status = agent
        .run(
            "continue now",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
            &ApprovalGate::default(),
        )
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    let checkpoint = agent.session().latest_compaction().unwrap();
    assert_eq!(checkpoint.first_kept_message_index, 2);
    assert_eq!(checkpoint.usage.unwrap().total_tokens, 150);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 2);
    assert!(requests[0].get("tools").is_none());
    assert_eq!(requests[0]["max_tokens"], 2_000);
    assert!(
        requests[0]["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("old old old")
    );
    let regular_messages = requests[1]["messages"].as_array().unwrap();
    assert!(regular_messages.iter().any(|message| {
        message["content"]
            .as_str()
            .is_some_and(|content| content.contains("Continue the fixture task"))
    }));
    assert!(regular_messages.iter().any(|message| {
        message["content"]
            .as_str()
            .is_some_and(|content| content.contains("recent question"))
    }));
    assert!(!regular_messages.iter().any(|message| {
        message["content"]
            .as_str()
            .is_some_and(|content| content.len() > 5_000)
    }));
    assert!(drain_events(&mut rx).into_iter().any(|event| matches!(
        event,
        AgentEvent::CompactionFinished {
            reason: mcode::event::CompactionReason::Threshold,
            ..
        }
    )));
}

#[tokio::test]
async fn context_overflow_compacts_and_retries_once() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let server_requests = Arc::clone(&requests);
    let server = tokio::spawn(async move {
        for index in 0..3 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_json_request(&mut stream).await;
            server_requests.lock().await.push(request);
            match index {
                0 => {
                    write_http_error(
                        &mut stream,
                        400,
                        r#"{"error":{"message":"maximum context length exceeded"}}"#,
                    )
                    .await;
                }
                1 => {
                    write_sse_text(&mut stream, "## Goal\nRecover from overflow", None).await;
                }
                _ => write_sse_text(&mut stream, "Recovered.", None).await,
            }
        }
    });

    let project = tempdir().unwrap();
    let config = compaction_test_config(project.path(), address, 10_000, 1_000, 60);
    let mut session =
        Session::create(project.path(), SessionMetadata::from(&config), false).unwrap();
    session
        .append(mcode::protocol::ChatMessage::user("old ".repeat(1_000)))
        .unwrap();
    session
        .append(mcode::protocol::ChatMessage::assistant(
            Some("old ".repeat(1_000)),
            None,
            Vec::new(),
        ))
        .unwrap();
    session
        .append(mcode::protocol::ChatMessage::user(
            "recent question ".repeat(8),
        ))
        .unwrap();
    session
        .append(mcode::protocol::ChatMessage::assistant(
            Some("recent answer ".repeat(8)),
            None,
            Vec::new(),
        ))
        .unwrap();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let status = agent
        .run(
            "continue",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
            &ApprovalGate::default(),
        )
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    assert!(agent.session().latest_compaction().is_some());
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 3);
    assert!(requests[0].get("tools").is_some());
    assert!(requests[1].get("tools").is_none());
    assert!(
        requests[2]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|message| {
                message["content"]
                    .as_str()
                    .is_some_and(|content| content.contains("Recover from overflow"))
            })
    );
    assert!(drain_events(&mut rx).into_iter().any(|event| matches!(
        event,
        AgentEvent::CompactionFinished {
            reason: mcode::event::CompactionReason::Overflow,
            ..
        }
    )));
}

#[tokio::test]
async fn responses_length_stop_compacts_and_retries_once() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let server_requests = Arc::clone(&requests);
    let server = tokio::spawn(async move {
        for index in 0..3 {
            let (mut stream, _) = listener.accept().await.unwrap();
            server_requests
                .lock()
                .await
                .push(read_json_request(&mut stream).await);
            let events = match index {
                0 => vec![json!({
                    "type": "response.incomplete",
                    "response": {
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "usage": {
                            "input_tokens": 5_000,
                            "output_tokens": 12,
                            "total_tokens": 5_012
                        },
                        "output": []
                    }
                })],
                1 => vec![
                    json!({
                        "type": "response.output_text.delta",
                        "delta": "## Goal\nRecover the interrupted response"
                    }),
                    responses_completed("resp_summary", 120, 20),
                ],
                _ => vec![
                    json!({
                        "type": "response.output_text.delta",
                        "delta": "Recovered response."
                    }),
                    responses_completed("resp_final", 300, 20),
                ],
            };
            write_responses_sse(&mut stream, events).await;
        }
    });

    let project = tempdir().unwrap();
    let mut config = compaction_test_config(project.path(), address, 100_000, 1_000, 60);
    config.provider = "deepseek".to_string();
    config.api = ApiProtocol::Responses;
    config.max_output_tokens = Some(100);
    config.compat.strict_tools = true;
    let mut session =
        Session::create(project.path(), SessionMetadata::from(&config), false).unwrap();
    session
        .append(ChatMessage::user("old ".repeat(500)))
        .unwrap();
    session
        .append(ChatMessage::assistant(
            Some("old ".repeat(500)),
            None,
            Vec::new(),
        ))
        .unwrap();
    session
        .append(ChatMessage::user("recent question ".repeat(8)))
        .unwrap();
    session
        .append(ChatMessage::assistant(
            Some("recent answer ".repeat(8)),
            None,
            Vec::new(),
        ))
        .unwrap();
    let mut agent = Agent::new(&config, session).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let status = agent
        .run(
            "continue",
            Vec::new(),
            &tx,
            &CancellationToken::new(),
            &ApprovalGate::default(),
        )
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0]["max_output_tokens"], 100);
    assert_eq!(requests[1]["max_output_tokens"], 800);
    assert_eq!(requests[2]["max_output_tokens"], 100);
    let retry_input = requests[2]["input"].to_string();
    assert!(retry_input.contains("Recover the interrupted response"));
    assert!(agent.messages().iter().any(|message| {
        message.role == mcode::protocol::MessageRole::Assistant && message.content.is_none()
    }));
    assert_eq!(agent.total_usage().total_tokens, 5_472);
    let events = drain_events(&mut rx);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ResponseTruncated { .. }))
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::CompactionFinished {
            reason: mcode::event::CompactionReason::Overflow,
            ..
        }
    )));
}

#[tokio::test]
async fn resume_does_not_replay_a_dangerous_tool_with_an_unknown_result() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_json_request(&mut stream).await;
        write_sse_text(&mut stream, "Recovered without replay.", Some((20, 5))).await;
        request
    });

    let base = tempdir().unwrap();
    let project = tempdir().unwrap();
    let config = basic_chat_config(project.path(), address);
    let mut session =
        Session::create_in(base.path(), project.path(), SessionMetadata::from(&config)).unwrap();
    let run_id = session
        .begin_run(ChatMessage::user("create interrupted.txt"))
        .unwrap();
    session.start_generation(run_id).unwrap();
    let call = tool_call(
        "call_interrupted_write",
        "write_file",
        r#"{"path":"interrupted.txt","content":"must not be written"}"#,
    );
    session
        .complete_generation(
            run_id,
            ChatMessage::assistant(None, None, vec![call.clone()]),
            Usage {
                prompt_tokens: 40,
                completion_tokens: 10,
                total_tokens: 50,
                cached_prompt_tokens: None,
            },
        )
        .unwrap();
    session
        .start_tool(run_id, call, ToolReplayPolicy::Never)
        .unwrap();
    drop(session);

    let resumed = Session::resume_in(base.path(), project.path(), None).unwrap();
    let mut agent = Agent::new(&config, resumed).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let status = agent
        .resume_pending(&tx, &CancellationToken::new(), &ApprovalGate::default())
        .await
        .unwrap();
    let request = server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    assert!(!project.path().join("interrupted.txt").exists());
    let tool_result = request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .unwrap();
    assert!(
        tool_result["content"]
            .as_str()
            .unwrap()
            .contains("may have run before MCode was interrupted")
    );
    assert!(drain_events(&mut rx).into_iter().any(|event| matches!(
        event,
        AgentEvent::ToolFinished {
            name,
            is_error: true,
            ..
        } if name == "write_file"
    )));
}

#[tokio::test]
async fn resume_replays_a_safe_read_file_tool() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_json_request(&mut stream).await;
        write_sse_text(&mut stream, "Read replay completed.", Some((18, 4))).await;
        request
    });

    let base = tempdir().unwrap();
    let project = tempdir().unwrap();
    std::fs::write(project.path().join("fixture.txt"), "before interruption\n").unwrap();
    let config = basic_chat_config(project.path(), address);
    let mut session =
        Session::create_in(base.path(), project.path(), SessionMetadata::from(&config)).unwrap();
    let run_id = session
        .begin_run(ChatMessage::user("read fixture.txt"))
        .unwrap();
    session.start_generation(run_id).unwrap();
    let call = tool_call(
        "call_interrupted_read",
        "read_file",
        r#"{"path":"fixture.txt","offset":null,"limit":null}"#,
    );
    session
        .complete_generation(
            run_id,
            ChatMessage::assistant(None, None, vec![call.clone()]),
            Usage {
                prompt_tokens: 30,
                completion_tokens: 8,
                total_tokens: 38,
                cached_prompt_tokens: None,
            },
        )
        .unwrap();
    session
        .start_tool(run_id, call, ToolReplayPolicy::Safe)
        .unwrap();
    drop(session);
    std::fs::write(
        project.path().join("fixture.txt"),
        "content observed during replay\n",
    )
    .unwrap();

    let resumed = Session::resume_in(base.path(), project.path(), None).unwrap();
    let mut agent = Agent::new(&config, resumed).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let status = agent
        .resume_pending(&tx, &CancellationToken::new(), &ApprovalGate::default())
        .await
        .unwrap();
    let request = server.await.unwrap();

    assert_eq!(status, RunStatus::Completed);
    let tool_result = request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .unwrap();
    assert!(
        tool_result["content"]
            .as_str()
            .unwrap()
            .contains("content observed during replay")
    );
    assert!(drain_events(&mut rx).into_iter().any(|event| matches!(
        event,
        AgentEvent::ToolFinished {
            name,
            is_error: false,
            ..
        } if name == "read_file"
    )));
}

#[tokio::test]
async fn delete_command_removes_only_the_selected_project_session() {
    let home = tempdir().unwrap();
    let project = tempdir().unwrap();
    let mcode_home = home.path().join(".mcode");
    write_test_model_catalog(&mcode_home, "test-model", false);
    let base = mcode_home.join("sessions");
    let selected =
        Session::create_in(&base, project.path(), test_session_metadata("test-model")).unwrap();
    let remaining =
        Session::create_in(&base, project.path(), test_session_metadata("test-model")).unwrap();
    let selected_path = selected.path().unwrap().to_path_buf();
    let remaining_path = remaining.path().unwrap().to_path_buf();
    let selected_id = selected.id();

    let listed = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcode"))
        .arg("sessions")
        .arg("--json")
        .env("MCODE_HOME", &mcode_home)
        .current_dir(project.path())
        .output()
        .await
        .unwrap();
    assert!(
        listed.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&listed.stderr)
    );
    let listed: Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert_eq!(listed.as_array().unwrap().len(), 2);
    drop(selected);

    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcode"))
        .arg("delete")
        .arg(selected_id.to_string())
        .arg("--force")
        .env("MCODE_HOME", &mcode_home)
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
    assert!(String::from_utf8_lossy(&output.stdout).contains(&selected_id.to_string()));
}

fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        kind: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    }
}

fn test_session_metadata(model: &str) -> SessionMetadata {
    SessionMetadata {
        provider: "xai".to_string(),
        model: model.to_string(),
        api: ApiProtocol::ChatCompletions,
        reasoning_effort: ReasoningEffort::Off,
        web_search_mode: WebSearchMode::Disabled,
    }
}

fn write_test_model_catalog(home: &std::path::Path, model: &str, supports_images: bool) {
    std::fs::create_dir_all(home).unwrap();
    let input = if supports_images {
        json!(["text", "image"])
    } else {
        json!(["text"])
    };
    let catalog = json!({
        "providers": {
            "xai": {
                "baseUrl": "http://127.0.0.1:1/v1",
                "api": "openai-completions",
                "models": [{
                    "id": model,
                    "input": input,
                    "reasoning": false,
                    "default": "off",
                    "contextWindow": 128_000,
                    "maxInputTokens": 128_000
                }]
            }
        }
    });
    std::fs::write(
        home.join("models.json"),
        serde_json::to_vec_pretty(&catalog).unwrap(),
    )
    .unwrap();
}

async fn read_json_request(stream: &mut TcpStream) -> Value {
    read_http_request(stream).await.0
}

fn drain_events(receiver: &mut mpsc::UnboundedReceiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event);
    }
    events
}

fn compaction_test_config(
    project: &std::path::Path,
    address: std::net::SocketAddr,
    context_window: u64,
    reserve_tokens: u64,
    keep_recent_tokens: u64,
) -> AppConfig {
    AppConfig {
        model: "test-model".to_string(),
        provider: "xai".to_string(),
        api: ApiProtocol::ChatCompletions,
        reasoning_effort: ReasoningEffort::Off,
        reasoning_value: None,
        base_url: format!("http://{address}/v1"),
        api_key: None,
        context_window,
        max_input_tokens: context_window,
        max_output_tokens: None,
        compat: ModelCompat {
            reasoning_effort: true,
            usage_in_streaming: true,
            finish_reason: true,
            strict_tools: false,
        },
        cwd: project.canonicalize().unwrap(),
        request_timeout_secs: 10,
        compaction: CompactionSettings {
            enabled: true,
            reserve_tokens,
            keep_recent_tokens,
        },
        web_search: WebSearchSettings::default(),
        model_profiles: Vec::new(),
        mcp_servers: Vec::new(),
        reload_overrides: ConfigOverrides::default(),
    }
}

fn basic_chat_config(project: &std::path::Path, address: std::net::SocketAddr) -> AppConfig {
    AppConfig {
        model: "test-model".to_string(),
        provider: "xai".to_string(),
        api: ApiProtocol::ChatCompletions,
        reasoning_effort: ReasoningEffort::Off,
        reasoning_value: None,
        base_url: format!("http://{address}/v1"),
        api_key: None,
        context_window: 128_000,
        max_input_tokens: 128_000,
        max_output_tokens: None,
        compat: ModelCompat {
            reasoning_effort: true,
            usage_in_streaming: true,
            finish_reason: true,
            strict_tools: false,
        },
        cwd: project.canonicalize().unwrap(),
        request_timeout_secs: 10,
        compaction: CompactionSettings::default(),
        web_search: WebSearchSettings::default(),
        model_profiles: Vec::new(),
        mcp_servers: Vec::new(),
        reload_overrides: ConfigOverrides::default(),
    }
}

async fn write_sse_text(stream: &mut TcpStream, content: &str, usage: Option<(u64, u64)>) {
    let event = json!({
        "choices": [{
            "delta": {"content": content},
            "finish_reason": "stop"
        }]
    });
    let mut body = format!("data: {event}\n\n");
    if let Some((prompt_tokens, completion_tokens)) = usage {
        let usage = json!({
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        });
        let _ = write!(body, "data: {usage}\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
}

fn responses_completed(id: &str, input_tokens: u64, output_tokens: u64) -> Value {
    json!({
        "type": "response.completed",
        "response": {
            "id": id,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
    })
}

async fn write_responses_sse(stream: &mut TcpStream, events: Vec<Value>) {
    let mut body = String::new();
    for event in events {
        let kind = event["type"].as_str().unwrap();
        let _ = write!(body, "event: {kind}\ndata: {event}\n\n");
    }
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
}

async fn write_http_error(stream: &mut TcpStream, status: u16, body: &str) {
    let response = format!(
        "HTTP/1.1 {status} Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
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
