use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;

use serde_json::{Value, json};

use super::*;
use crate::chat::{ChatMessage, ChatModel, ChatRequest, ModelError, ModelEvent};

struct Server {
    url: String,
    handle: std::thread::JoinHandle<Vec<String>>,
}

/// A loopback server answering each request with the next `(status, reason, body)`.
fn serve(responses: Vec<(u16, &'static str, String)>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let mut requests = Vec::new();
        for (status, reason, body) in responses {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut head = String::new();
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" || line == "\n" || line.is_empty() {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap_or(0);
                }
                head.push_str(&line);
            }
            let mut request = vec![0u8; length];
            reader.read_exact(&mut request).unwrap();
            requests.push(format!("{head}\r\n{}", String::from_utf8_lossy(&request)));
            let mut stream = stream;
            let reply = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(reply.as_bytes()).unwrap();
        }
        requests
    });
    Server { url, handle }
}

fn model(server: &Server, retries: u32) -> DeepSeek {
    DeepSeek::new(DeepSeekConfig::default().with_key("sk-test").with_base_url(&server.url).with_retries(retries))
}

fn ask(stream: bool) -> ChatRequest {
    let tool = json!({"type": "function", "function": {"name": "create_node", "description": "Make a layer", "parameters": {"type": "object"}}});
    let mut request = ChatRequest::new("deepseek-flash", vec![ChatMessage::user("make a frame")], vec![tool]);
    request.stream = stream;
    request
}

fn sse_chunk(value: Value) -> String {
    format!("data: {value}\n\n")
}

#[test]
fn a_streamed_answer_and_tool_call_are_assembled() {
    let body = [
        json!({"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": null}]}),
        json!({"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "create_node", "arguments": "{\"type\":"}}]}, "finish_reason": null}]}),
        json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"frame\"}"}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}),
    ]
    .into_iter()
    .map(sse_chunk)
    .collect::<String>()
        + "data: [DONE]\n\n";
    let server = serve(vec![(200, "OK", body)]);
    let model = model(&server, 0);
    let (tx, rx) = mpsc::channel();
    let mut relay = |event| tx.send(event).unwrap();
    let response = model.complete(&ask(true), &mut relay, &AtomicBool::new(false)).unwrap();
    let events: Vec<ModelEvent> = rx.try_iter().collect();

    assert_eq!(response.content, "Hello");
    assert_eq!(response.reasoning.as_deref(), Some("think"));
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].id, "call_a");
    assert_eq!(response.tool_calls[0].name, "create_node");
    assert_eq!(response.tool_calls[0].arguments, "{\"type\":\"frame\"}");
    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(response.usage.map(|u| u.total_tokens), Some(3));
    assert!(events.iter().any(|e| matches!(e, ModelEvent::Content(t) if t == "Hello")));
    assert!(events.iter().any(|e| matches!(e, ModelEvent::Reasoning(t) if t == "think")));

    let requests = server.handle.join().unwrap();
    assert_eq!(requests.len(), 1);
    let sent = &requests[0];
    let body = sent.split("\r\n\r\n").nth(1).unwrap_or_default();
    assert!(body.contains("\"stream\":true"), "{body}");
    assert!(body.contains("\"thinking\":{\"type\":\"enabled\"}"), "{body}");
    assert!(body.contains("\"tool_choice\":\"auto\"") || body.contains("\"tool_choice\": \"auto\""), "{body}");
    assert!(sent.contains("authorization: Bearer sk-test"), "the key travels in its header: {sent}");
    assert!(!body.contains("sk-test"), "the key must not be in the body: {body}");
}

#[test]
fn a_non_streamed_answer_is_parsed() {
    let body = json!({
        "choices": [{
            "message": {
                "content": "Done",
                "reasoning_content": "why",
                "tool_calls": [{"id": "c1", "function": {"name": "run_command", "arguments": "{}"}}],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
    })
    .to_string();
    let server = serve(vec![(200, "OK", body)]);
    let response = model(&server, 0).complete(&ask(false), &mut |_| {}, &AtomicBool::new(false)).unwrap();
    assert_eq!(response.content, "Done");
    assert_eq!(response.reasoning.as_deref(), Some("why"));
    assert_eq!(response.tool_calls[0].name, "run_command");
    assert_eq!(response.usage.map(|u| u.total_tokens), Some(9));
    server.handle.join().unwrap();
}

#[test]
fn a_refusal_reports_the_body_and_never_the_key() {
    let server = serve(vec![(401, "Unauthorized", "{\"error\":{\"message\":\"bad key\"}}".to_string())]);
    let error = model(&server, 0).complete(&ask(true), &mut |_| {}, &AtomicBool::new(false)).unwrap_err();
    match &error {
        ModelError::Status { code, message } => {
            assert_eq!(*code, 401);
            assert!(message.contains("bad key"), "{message}");
            assert!(!message.contains("sk-test"), "{message}");
        }
        other => panic!("expected a status error, got {other:?}"),
    }
    server.handle.join().unwrap();
}

#[test]
fn a_429_is_retried_before_the_stream_starts() {
    let body = sse_chunk(json!({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})) + "data: [DONE]\n\n";
    let server = serve(vec![(429, "Too Many Requests", "busy".to_string()), (200, "OK", body)]);
    let response = model(&server, 1).complete(&ask(true), &mut |_| {}, &AtomicBool::new(false)).unwrap();
    assert_eq!(response.content, "ok");
    assert_eq!(server.handle.join().unwrap().len(), 2);
}

#[test]
fn a_missing_key_is_refused_before_any_request() {
    let error = DeepSeek::new(DeepSeekConfig::default()).complete(&ask(true), &mut |_| {}, &AtomicBool::new(false)).unwrap_err();
    assert_eq!(error, ModelError::MissingKey);
}

#[test]
fn cancelling_before_the_request_sends_nothing() {
    let config = DeepSeekConfig::default().with_key("sk-test").with_base_url("http://127.0.0.1:1");
    let error = DeepSeek::new(config).complete(&ask(true), &mut |_| {}, &AtomicBool::new(true)).unwrap_err();
    assert_eq!(error, ModelError::Cancelled);
}

#[test]
fn an_empty_stream_is_an_error() {
    let server = serve(vec![(200, "OK", "data: [DONE]\n\n".to_string())]);
    let error = model(&server, 0).complete(&ask(true), &mut |_| {}, &AtomicBool::new(false)).unwrap_err();
    assert_eq!(error, ModelError::Empty);
    server.handle.join().unwrap();
}
