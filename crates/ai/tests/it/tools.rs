use figcraft_ai::{Scope, openai_tools, system_prompt, tool_definitions_for};

#[test]
fn every_mcp_tool_becomes_one_openai_function() {
    let defs = tool_definitions_for(&["list_commands", "run_command", "create_node", "inspect_document"]);
    let tools = openai_tools(&defs);
    let names: Vec<&str> = tools.iter().filter_map(|t| t["function"]["name"].as_str()).collect();
    assert_eq!(names, ["list_commands", "run_command", "create_node", "inspect_document"]);
    for tool in &tools {
        assert_eq!(tool["type"], "function");
        assert!(tool["function"]["parameters"].is_object(), "{tool}");
        assert!(tool["function"]["description"].as_str().is_some_and(|d| !d.is_empty()), "{tool}");
    }
}

#[test]
fn the_app_scope_explains_the_file_limit_and_the_text_screenshots() {
    let prompt = system_prompt(Scope::App);
    assert!(prompt.contains("inspect_document"));
    assert!(prompt.contains("screenshot comes back as text"), "{prompt}");
    assert!(prompt.contains("file the user has open"), "{prompt}");
    assert!(!prompt.contains("automation roots"), "the app panel doesn't open files: {prompt}");
}

#[test]
fn the_automation_scope_names_the_roots() {
    let prompt = system_prompt(Scope::Automation);
    assert!(prompt.contains("automation roots"), "{prompt}");
    assert!(prompt.contains("open and save files"), "{prompt}");
}
