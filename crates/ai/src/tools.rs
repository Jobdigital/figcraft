//! The bridge between DeepSeek's function calls and FigCraft's MCP tools: the schemas the model
//! sees, how a tool result becomes text, and the system prompt.

use serde_json::{Value, json};

use crate::chat::ToolCall;
use crate::mode::Mode;
use crate::outcome::ToolOutcome;

pub use figcraft_mcp::{Backend, INSTRUCTIONS, ToolResult, call_tool, decode_data, tool_definitions};

/// The tools that only look: what Ask and Plan may use, so they cannot change the file even by
/// accident. `select` moves the selection (the model pointing at what it means), which is not an
/// edit. `export` is left out: it writes a file.
pub const READ_ONLY_TOOLS: &[&str] = &["list_commands", "inspect_document", "get_node", "design_context", "select", "screenshot", "inspect_ui"];

impl ToolOutcome {
    /// The text of an MCP tool result, with image blocks described rather than carried.
    pub fn from_result(result: ToolResult) -> Self {
        let mut parts = Vec::new();
        for block in &result.content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(Value::as_str) {
                        parts.push(t.to_string());
                    }
                }
                Some("image") => {
                    let mime = block.get("mimeType").and_then(Value::as_str).unwrap_or("image/png");
                    let bytes = block.get("data").and_then(Value::as_str).map(str::len).unwrap_or(0);
                    parts.push(format!(
                        "[image: {mime}, about {bytes} base64 characters; this text model can't see it — check the design with inspect_document or get_node]"
                    ));
                }
                _ => {}
            }
        }
        let text = if parts.is_empty() { "(the tool returned nothing)".to_string() } else { parts.join("\n\n") };
        Self { text, is_error: result.is_error }
    }
}

/// MCP tool definitions as OpenAI functions: `inputSchema` becomes `parameters`, and the
/// MCP-only `title` and `annotations` are left out.
pub fn openai_tools(defs: &[Value]) -> Vec<Value> {
    defs.iter()
        .filter_map(|d| {
            let name = d.get("name").and_then(Value::as_str)?;
            let description = d.get("description").and_then(Value::as_str).unwrap_or("");
            let parameters = d.get("inputSchema").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            Some(json!({
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            }))
        })
        .collect()
}

/// The MCP definitions whose names are in `names`, in the server's order.
pub fn tool_definitions_for(names: &[&str]) -> Vec<Value> {
    tool_definitions().into_iter().filter(|d| d.get("name").and_then(Value::as_str).is_some_and(|n| names.contains(&n))).collect()
}

/// The tool definitions a mode may use: every tool for Agent, only [`READ_ONLY_TOOLS`] for Ask and
/// Plan. Taking the tools away is what makes "change nothing" true, rather than a rule the model
/// could break.
pub fn tool_definitions_for_mode(mode: Mode) -> Vec<Value> {
    if mode.edits() { tool_definitions() } else { tool_definitions_for(READ_ONLY_TOOLS) }
}

/// Run one tool call against a backend. Malformed arguments become a tool error the model can
/// read and correct; nothing panics.
pub fn execute_tool<B: Backend>(backend: &mut B, call: &ToolCall) -> ToolOutcome {
    match call.arguments_value() {
        Ok(args) => ToolOutcome::from_result(call_tool(backend, &call.name, &args)),
        Err(e) => ToolOutcome::error(e),
    }
}

/// Where the Assistant works: inside the app on the open file, or through automation with the
/// file tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// The app's Assistant panel: only the file the user has open, no file or window commands.
    App,
    /// `figcraft-cli agent` and the MCP server: the automation roots apply, files may be opened and
    /// saved.
    Automation,
}

/// The system prompt: the MCP instructions every agent gets, plus what this chat connection can
/// and can't do. This is the Agent mode's prompt; [`system_prompt_for`] takes a mode.
pub fn system_prompt(scope: Scope) -> String {
    system_prompt_for(scope, Mode::Agent)
}

/// The system prompt for a mode: the shared instructions, what this chat connection can and can't
/// do, and what the mode the user picked means for the answer.
pub fn system_prompt_for(scope: Scope, mode: Mode) -> String {
    let mut prompt = String::with_capacity(INSTRUCTIONS.len() + 1_200);
    prompt.push_str(INSTRUCTIONS);
    prompt.push_str(
        "\n\nAssistant rules:\n\
         - Do the work with the tools instead of describing it; keep the final answer short.\n\
         - Work in batches until the design is finished: don't stop after one batch to say what you will do next, \
         and keep calling tools until the whole request is done.\n\
         - A screenshot comes back as text about its size and source, not the picture: check the result with \
         inspect_document or get_node.\n\
         - Answer in the language the user wrote in.",
    );
    prompt.push_str(mode.rules());
    match scope {
        Scope::App => prompt.push_str(
            "\n- Work on the file the user has open and nothing else: opening, closing, saving and quitting files are \
             not available to you, nor are preferences, windows, plugins or online sharing.",
        ),
        Scope::Automation => prompt.push_str(
            "\n- You may open and save files with the tools; everything you read or write stays inside the \
             automation roots the server was started with.",
        ),
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_tools_keep_the_schema_and_drop_mcp_metadata() {
        let defs = tool_definitions_for(&["create_node"]);
        let tools = openai_tools(&defs);
        assert_eq!(tools.len(), 1);
        let f = &tools[0]["function"];
        assert_eq!(f["name"], "create_node");
        assert!(f["parameters"].is_object());
        assert!(f.get("inputSchema").is_none());
        assert!(tools[0].get("annotations").is_none());
    }

    #[test]
    fn an_image_is_described_not_carried() {
        let result = ToolResult { content: vec![json!({"type": "image", "data": "QUJD", "mimeType": "image/png"})], is_error: false };
        let text = ToolOutcome::from_result(result).text;
        assert!(text.contains("image"), "{text}");
        assert!(!text.contains("QUJD"), "the base64 data must not reach the model: {text}");
    }

    #[test]
    fn ask_and_plan_get_no_tool_that_can_change_the_file() {
        let editing = ["create_node", "set_properties", "run_command", "batch", "export", "undo", "save_file"];
        for mode in [Mode::Ask, Mode::Plan] {
            let tools = openai_tools(&tool_definitions_for_mode(mode));
            let names: Vec<&str> = tools.iter().filter_map(|t| t["function"]["name"].as_str()).collect();
            assert!(!names.is_empty(), "{mode:?} keeps the tools that only look");
            for tool in editing {
                assert!(!names.contains(&tool), "{mode:?} must not offer {tool}: {names:?}");
            }
            assert!(names.contains(&"get_node"), "{names:?}");
        }
        let agent = openai_tools(&tool_definitions_for_mode(Mode::Agent));
        assert!(agent.len() > openai_tools(&tool_definitions_for_mode(Mode::Ask)).len(), "Agent gets more tools");
    }

    #[test]
    fn the_prompt_says_what_the_mode_allows() {
        let ask = system_prompt_for(Scope::App, Mode::Ask);
        assert!(ask.contains("Ask mode") && ask.contains("Never change the file"), "{ask}");
        let plan = system_prompt_for(Scope::App, Mode::Plan);
        assert!(plan.contains("Plan mode") && plan.contains("numbered plan"));
        let agent = system_prompt_for(Scope::App, Mode::Agent);
        assert!(agent.contains("Agent mode") && !agent.contains("Plan mode"));
        assert_eq!(system_prompt(Scope::App), agent, "the plain prompt is Agent mode");
        assert!(ask.contains("the file the user has open"), "the scope rules stay");
    }
}
