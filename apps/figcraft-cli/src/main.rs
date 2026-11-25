//! `figcraft-cli`: FigCraft without a window. Inspect, export, convert and render documents, run
//! command scripts, list commands, serve MCP on stdio, publish plugins to a server's plugin
//! directory, publish code connect mappings to a design file and measure performance.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod args;
mod files;
mod perf;

use std::io::{Read as _, Write as _};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use args::{AgentArgs, Cli, Command, ConnectCommand, PluginCommand, RootArgs};
use figcraft_ai::{Agent, AgentConfig, ChatModel, DeepSeek, ModelEvent, Scope, Step};
use serde_json::{Value, json};

/// The automation roots a file subcommand was given, if any. Omitting both leaves the run
/// unconfined, as before; giving either confines every file the subcommand touches (#833).
fn roots_of(r: &RootArgs) -> CliResult<Option<figcraft_engine::file_access::Roots>> {
    if r.automation_read_root.is_empty() && r.automation_write_root.is_empty() {
        return Ok(None);
    }
    Ok(Some(figcraft_engine::file_access::Roots::new(&r.automation_read_root, &r.automation_write_root).map_err(CliError::Failed)?))
}

/// Runs `f` confined to the roots `r` carries.
fn confined<R>(r: &RootArgs, f: impl FnOnce() -> CliResult<R>) -> CliResult<R> {
    let roots = roots_of(r)?.map(std::sync::Arc::new);
    figcraft_engine::file_access::confined(roots.as_ref(), f)
}

fn main() -> ExitCode {
    // `args_os`: a non-Unicode argument is clap's usage error, not a panic.
    let cli = match args::parse(std::env::args_os().collect()) {
        Ok(cli) => cli,
        Err(e) => {
            // Help and `--version` on stdout with 0, usage errors on stderr with 2.
            let _ = e.print();
            return ExitCode::from(u8::try_from(e.exit_code()).unwrap_or(2));
        }
    };
    // Paths and imported/plugin errors may carry terminal controls; keep diagnostics printable.
    match run(cli) {
        Ok(code) => code,
        Err(CliError::Usage(msg)) => {
            eprintln!("error: {}\n\nFor more information, try '--help'.", files::plain(&msg));
            ExitCode::from(2)
        }
        Err(CliError::Failed(msg)) => {
            eprintln!("error: {}", files::plain(&msg));
            ExitCode::from(1)
        }
    }
}

#[derive(Debug)]
pub enum CliError {
    /// Bad arguments found after parsing (an output extension no format writes): exit code 2.
    Usage(String),
    /// The operation failed: exit code 1.
    Failed(String),
}

impl From<String> for CliError {
    fn from(s: String) -> Self {
        CliError::Failed(s)
    }
}

pub type CliResult<T> = Result<T, CliError>;

fn run(cli: Cli) -> CliResult<ExitCode> {
    if !cli.no_system_fonts {
        // Text lays out with the installed fonts, like the desktop app, so headless renders and
        // auto layout sizes match what users see.
        figcraft_text::scan_system_fonts_on_demand();
    } else {
        // Nothing scans them then, not even a `.sketch` import matching its fonts.
        figcraft_text::opt_out_of_system_fonts();
    }
    match cli.command {
        Command::Info(a) => confined(&a.roots, || files::info(&a.file, a.json)),
        Command::Export(a) => {
            let target =
                files::Target { print: a.print(), vd_compat: a.vd_compat, scale: a.scale.scale, format: a.format, page: a.page, node: a.node };
            confined(&a.roots, || files::export(&a.file, &a.out, &target))
        }
        Command::Convert(a) => confined(&a.roots, || files::convert(&a.input, &a.output)),
        Command::Render(a) => confined(&a.roots, || files::render(&a.file, &a.out, a.scale.scale, a.page.as_deref())),
        Command::Run(a) => confined(&a.roots, || files::run_script(&a.script, a.input.as_deref(), a.out.as_deref())),
        Command::Commands(a) => {
            commands(a.json, a.filter.as_deref());
            Ok(ExitCode::SUCCESS)
        }
        // `--attach` conflicts with the roots: the app runs the commands, only its own roots can confine them.
        Command::Mcp(a) => mcp(a.attach.as_deref(), a.timeout, roots_of(&a.roots)?),
        Command::Agent(a) => confined(&a.roots, || agent(&a)),
        Command::Plugin { command: PluginCommand::Publish(a) } => publish_plugin(&a.folder, &a.server),
        Command::Connect { command: ConnectCommand::Publish(a) } => confined(&a.roots, || files::connect_publish(&a.file, &a.mappings, a.dry_run)),
        Command::Perf(a) => {
            let sizes: Vec<usize> = a.layers.into_iter().map(|n| n.clamp(10, 2_000_000)).collect();
            let instances: Vec<usize> = a.instances.into_iter().filter(|n| *n > 0).map(|n| n.min(100_000)).collect();
            perf::run(&sizes, &instances, !cli.no_system_fonts)
        }
    }
}

fn commands(as_json: bool, filter: Option<&str>) {
    let f = filter.map(str::to_lowercase).unwrap_or_default();
    let list: Vec<&figcraft_engine::CommandSpec> = figcraft_engine::all_commands()
        .iter()
        .filter(|c| f.is_empty() || c.id.to_lowercase().contains(&f) || c.label.to_lowercase().contains(&f) || c.params.to_lowercase().contains(&f))
        .collect();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if as_json {
        let v: Vec<Value> = list
            .iter()
            .map(|c| json!({"id": c.id, "label": c.label, "params": c.params, "menu": c.menu(), "shortcut": c.shortcut(), "undoable": c.history}))
            .collect();
        let _ = writeln!(out, "{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        return;
    }
    for c in list {
        let sc = c.shortcut().map(|s| format!(" ({s})")).unwrap_or_default();
        let undo = if c.history { "*" } else { "" };
        let _ = writeln!(out, "{}{undo}  {}{sc}\n    {}", c.id, c.label, c.params);
    }
}

/// Publish the plugin folder `folder` to the plugin directory at `server` (the key from the address
/// or `FIGCRAFT_SERVER_KEY`); never prints the key.
fn publish_plugin(folder: &str, server: &str) -> CliResult<ExitCode> {
    let address = server.split('?').next().unwrap_or_default().to_string();
    let server = match std::env::var("FIGCRAFT_SERVER_KEY") {
        Ok(key) if !server.contains("key=") && !key.trim().is_empty() => format!("{address}?key={}", key.trim()),
        _ => server.to_string(),
    };
    let mut h = figcraft_mcp::Headless::new();
    use figcraft_mcp::Backend as _;
    let r = h.call("engine.execute", json!({"command": "plugin.directory.publish", "params": {"path": folder, "server": server}}))?;
    let e = &r["published"];
    println!(
        "Published {} {} ({}) to {address}: {} bytes, SHA-256 {}",
        e["name"].as_str().unwrap_or_default(),
        e["version"].as_str().unwrap_or_default(),
        e["id"].as_str().unwrap_or_default(),
        e["bytes"],
        e["sha256"].as_str().unwrap_or_default()
    );
    Ok(ExitCode::SUCCESS)
}

fn mcp(attach: Option<&str>, timeout: Option<u64>, roots: Option<figcraft_engine::file_access::Roots>) -> CliResult<ExitCode> {
    let backend: Box<dyn figcraft_mcp::Backend> = match attach {
        Some(addr) => {
            let no_app = |e: std::io::Error| format!("can't reach the FigCraft app at {addr}: {e} (start it with `figcraft --control PORT`)");
            let b = figcraft_mcp::Attached::connect(addr).map_err(no_app)?;
            match timeout {
                Some(secs) => Box::new(b.with_timeout(std::time::Duration::from_secs(secs)).map_err(no_app)?),
                None => Box::new(b),
            }
        }
        None => Box::new(figcraft_mcp::Headless::new()),
    };
    eprintln!("figcraft MCP server ({}) on stdio", backend.describe());
    if let Some(r) = &roots {
        eprintln!("file access confined to {}", r.to_json());
    }
    let mut server = figcraft_mcp::Server::new(backend).with_roots(roots);
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    server.serve(stdin.lock(), stdout.lock()).map_err(|e| format!("stdio: {e}"))?;
    Ok(ExitCode::SUCCESS)
}

/// `figcraft-cli agent`: DeepSeek drives the MCP tools, headless or attached. The key comes from
/// the environment only; the prompt from a flag, a file or standard input.
fn agent(a: &AgentArgs) -> CliResult<ExitCode> {
    let prompt = read_prompt(a)?;
    let mut config = figcraft_ai::DeepSeekConfig::from_env();
    if let Some(url) = &a.base_url {
        config = config.with_base_url(url);
    }
    if config.api_key.is_empty() {
        return Err(CliError::Failed(format!(
            "no DeepSeek API key: set {} (there is no --api-key, so the key stays out of the shell history and the process list)",
            figcraft_ai::API_KEY_ENV
        )));
    }
    let mut backend: Box<dyn figcraft_mcp::Backend> = match &a.attach {
        Some(addr) => {
            let no_app = |e: std::io::Error| format!("can't reach the FigCraft app at {addr}: {e} (start it with `figcraft --control PORT`)");
            Box::new(figcraft_mcp::Attached::connect(addr).map_err(no_app)?)
        }
        None => Box::new(figcraft_mcp::Headless::new()),
    };
    if let Some(path) = &a.input {
        let opened = figcraft_mcp::call_tool(backend.as_mut(), "open_file", &json!({"path": path}));
        if opened.is_error {
            return Err(CliError::Failed(format!("can't open {path}: {}", opened.first_text())));
        }
    }
    // The mode decides the rules and the tools: Ask and Plan never get one that changes the file.
    let mode = figcraft_ai::Mode::from_id(&a.mode).unwrap_or_default();
    let tools = figcraft_ai::openai_tools(&figcraft_ai::tool_definitions_for_mode(mode));
    let mut agent = Agent::new(
        figcraft_ai::system_prompt_for(Scope::Automation, mode),
        tools,
        AgentConfig { max_steps: a.max_steps as usize, max_output_tokens: a.max_tokens as u32, ..Default::default() },
    );
    agent.user(prompt);
    let model = DeepSeek::new(config);
    let mut as_agent = figcraft_mcp::AsAgent::new(backend.as_mut(), "DeepSeek");
    let mut streamed = false;
    loop {
        let mut request = agent.request(&a.model, !a.no_thinking);
        request.stream = !a.no_stream;
        let mut on_event = |event: ModelEvent| {
            if let ModelEvent::Content(text) = event
                && !a.json
                && !a.no_stream
            {
                streamed = true;
                print!("{text}");
                let _ = std::io::stdout().flush();
            }
        };
        let response = match model.complete(&request, &mut on_event, &AtomicBool::new(false)) {
            Ok(response) => response,
            Err(e) => {
                // The endpoint takes fewer tokens than the budget: lower it to what it says and ask
                // again rather than failing on a limit the user can't see.
                if let Some(limit) = e.max_tokens_limit().filter(|limit| *limit < agent.max_output_tokens()) {
                    agent.set_max_output_tokens(limit);
                    let text = format!("this endpoint takes at most {limit} tokens per answer: using that");
                    if a.json {
                        println!("{}", json!({"type": "note", "text": text}));
                    } else {
                        eprintln!("note: {text}");
                    }
                    continue;
                }
                return Err(CliError::Failed(e.to_string()));
            }
        };
        match agent.accept(response) {
            Step::Tools { calls, text } => {
                if a.no_stream && !text.is_empty() && !a.json {
                    println!("{text}");
                }
                for call in calls {
                    if a.json {
                        println!("{}", json!({"type": "tool", "name": call.name, "arguments": call.arguments}));
                    } else {
                        eprintln!("-> {} {}", call.name, one_line(&call.arguments));
                    }
                    let outcome = figcraft_ai::execute_tool(&mut as_agent, &call);
                    if outcome.is_error {
                        eprintln!("   error: {}", one_line(&outcome.text));
                    }
                    agent.tool_result(&call, &outcome);
                }
            }
            Step::Done { text } => {
                if a.json {
                    println!("{}", json!({"type": "done", "text": text}));
                } else if !streamed {
                    println!("{text}");
                }
                break;
            }
            Step::Stopped { text } => {
                if a.json {
                    println!("{}", json!({"type": "stopped", "text": text}));
                } else if !streamed && !text.is_empty() {
                    println!("{text}");
                }
                return Err(CliError::Failed(format!("the model used all {} turns without a final answer", a.max_steps)));
            }
            Step::Incomplete { text, cut_off } => {
                if a.json {
                    println!("{}", json!({"type": "incomplete", "text": text, "cutOff": cut_off}));
                } else if !streamed && !text.is_empty() {
                    println!("{text}");
                }
                if !agent.nudge() {
                    let why = if cut_off { "the answer ran out of output tokens" } else { "the model answered with nothing" };
                    return Err(CliError::Failed(format!("{why}, twice: stopping instead of looping")));
                }
            }
        }
    }
    if let Some(out) = &a.out {
        let saved = figcraft_mcp::call_tool(&mut as_agent, "save_file", &json!({"path": out}));
        if saved.is_error {
            return Err(CliError::Failed(format!("can't save {out}: {}", saved.first_text())));
        }
        if a.json {
            println!("{}", json!({"type": "saved", "path": out}));
        } else {
            eprintln!("wrote {out}");
        }
    } else if a.attach.is_none() {
        eprintln!("note: changes are not saved (pass --out FILE)");
    }
    Ok(ExitCode::SUCCESS)
}

/// The prompt from `--prompt`, `--prompt-file` (`-` is standard input) or standard input, at most
/// 1 MiB.
fn read_prompt(a: &AgentArgs) -> CliResult<String> {
    const MAX: u64 = 1024 * 1024;
    let text = match (&a.prompt, &a.prompt_file) {
        (Some(prompt), _) => prompt.clone(),
        (None, Some(path)) if path == "-" => read_stdin(MAX)?,
        (None, Some(path)) => {
            let bytes = figcraft_engine::file_access::read_capped(path, MAX).map_err(|e| e.to_string())?;
            String::from_utf8(bytes).map_err(|_| CliError::Failed(format!("{path}: not UTF-8")))?
        }
        (None, None) => read_stdin(MAX)?,
    };
    if text.trim().is_empty() {
        return Err(CliError::Usage("the prompt is empty (pass --prompt TEXT, --prompt-file FILE, or write it on standard input)".into()));
    }
    Ok(text)
}

fn read_stdin(max: u64) -> CliResult<String> {
    let mut text = String::new();
    std::io::stdin().take(max + 1).read_to_string(&mut text).map_err(|e| CliError::Failed(format!("standard input: {e}")))?;
    if text.len() as u64 > max {
        return Err(CliError::Usage("the prompt is larger than 1 MiB".into()));
    }
    Ok(text)
}

/// One line for a terminal: controls become spaces and the text is cut at 160 characters.
fn one_line(s: &str) -> String {
    let mut out: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).take(160).collect();
    if s.chars().count() > 160 {
        out.push('…');
    }
    out.trim().to_string()
}
