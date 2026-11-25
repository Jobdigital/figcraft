//! `cargo xtask dev`: the fast loop. Builds the app in the dev profile (optimised a little, thin
//! debug info, cargo's default incremental compilation), runs it, and rebuilds and restarts it when
//! a source file is saved.
//!
//! A release build takes about seven minutes a change here; this loop takes tens of seconds, mostly
//! the one crate that changed plus the link. Files watched: `crates/`, `apps/` and `plugins/`
//! `.rs` and `.toml`, and the root `Cargo.toml`; build output (`target/`) never is.
//!
//! `--release` loops on the release profile instead (slow, for timing and shipping checks).
//! `--no-run` only builds. `--control PORT` starts the app with `--control PORT`, and anything after
//! `--` goes to the app, such as a file to open: `cargo xtask dev -- design.figcraft`.

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

/// How often the watched files are checked, and how long a burst of saves is left to settle.
const POLL: Duration = Duration::from_millis(600);
const SETTLE: Duration = Duration::from_millis(400);

#[derive(Default)]
struct Options {
    release: bool,
    no_run: bool,
    app: Vec<String>,
}

pub fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let options = parse(args)?;
    let profile = if options.release { "release" } else { "dev" };
    // Cargo names the dev profile's folder `debug`, not `dev`.
    let folder = if options.release { "release" } else { "debug" };
    let binary = root.join("target").join(folder).join(format!("figcraft{}", std::env::consts::EXE_SUFFIX));
    eprintln!("dev: {profile} profile, watching crates/ apps/ plugins/ for changes (Ctrl+C to stop)");

    let mut app: Option<Child> = None;
    let mut watched = signature(root);
    build(root, options.release, &mut app)?;
    if !options.no_run {
        app = Some(launch(&binary, &options)?);
    }
    loop {
        std::thread::sleep(POLL);
        let now = signature(root);
        if now == watched {
            continue;
        }
        // A save often comes in bursts (an editor's temporary file, a formatter): wait for quiet.
        std::thread::sleep(SETTLE);
        watched = signature(root);
        match build(root, options.release, &mut app) {
            Ok(()) => {
                if options.no_run {
                    continue;
                }
                app = Some(launch(&binary, &options)?);
            }
            // A broken edit keeps the last good app running: the next save builds again.
            Err(why) => eprintln!("dev: build failed, the app was left as it was ({why})"),
        }
    }
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--release" => options.release = true,
            "--no-run" => options.no_run = true,
            "--control" => {
                let port = rest.next().ok_or("`--control` needs a port: `cargo xtask dev --control 7979`")?;
                options.app.push("--control".into());
                options.app.push(port.clone());
            }
            "--" => {
                options.app.extend(rest.by_ref().cloned());
                break;
            }
            other => return Err(format!("dev: unknown argument `{other}` (--release, --no-run, --control PORT, -- APP ARGS)")),
        }
    }
    Ok(options)
}

fn build(root: &Path, release: bool, app: &mut Option<Child>) -> Result<(), String> {
    // Windows locks the program while it runs, so cargo can't replace it: close the app first. A
    // failed build then leaves it closed, with the compiler's error on screen, until the next save.
    if let Some(mut child) = app.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cmd.current_dir(root).args(["build", "-p", "figcraft"]);
    if release {
        cmd.arg("--release");
    }
    eprintln!("dev: building…");
    let started = Instant::now();
    let status = cmd.status().map_err(|e| format!("can't run cargo: {e}"))?;
    if !status.success() {
        return Err("cargo build failed".into());
    }
    eprintln!("dev: built in {:.1} s", started.elapsed().as_secs_f32());
    Ok(())
}

fn launch(binary: &Path, options: &Options) -> Result<Child, String> {
    if !binary.exists() {
        return Err(format!("dev: {} is not there; was the app built?", binary.display()));
    }
    let mut cmd = Command::new(binary);
    cmd.args(&options.app).stdin(Stdio::null());
    let child = cmd.spawn().map_err(|e| format!("can't start {}: {e}", binary.display()))?;
    eprintln!("dev: running {} {}", binary.display(), options.app.join(" "));
    Ok(child)
}

/// A cheap fingerprint of every source file: its size and modification time, summed. Rebuild
/// decisions only need "did anything change", not which file.
fn signature(root: &Path) -> u64 {
    let mut sum = 0u64;
    for dir in ["crates", "apps", "plugins"] {
        walk(&root.join(dir), &mut sum);
    }
    stamp(&root.join("Cargo.toml"), &mut sum);
    sum
}

fn walk(dir: &Path, sum: &mut u64) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => walk(&path, sum),
            Ok(_) => {
                let watched = path.extension().is_some_and(|e| e == "rs" || e == "toml");
                if watched {
                    stamp(&path, sum);
                }
            }
            Err(_) => {}
        }
    }
}

fn stamp(path: &Path, sum: &mut u64) {
    let Ok(meta) = std::fs::metadata(path) else { return };
    let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let nanos = modified.duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    *sum = sum.wrapping_add(nanos ^ meta.len().wrapping_mul(0x9E37_79B9_7F4A_7C15));
    // Paths differ in length, so two files swapping mtimes still change the sum.
    let bytes = path.as_os_str().to_string_lossy();
    *sum = sum.wrapping_add(bytes.len() as u64);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn the_arguments_name_the_profile_the_app_and_what_it_opens() {
        let options = parse(&args(&["--release", "--control", "7979", "--", "design.figcraft"])).unwrap();
        assert!(options.release && !options.no_run);
        assert_eq!(options.app, ["--control", "7979", "design.figcraft"]);
        assert!(parse(&args(&["--nope"])).is_err());
        assert!(parse(&args(&["--control"])).is_err(), "a port is required");
    }

    #[test]
    fn a_saved_source_file_changes_the_signature_and_an_untouched_one_does_not() {
        let root = std::env::temp_dir().join(format!("figcraft-dev-{}", std::process::id()));
        let src = root.join("crates").join("demo").join("src");
        std::fs::create_dir_all(&src).unwrap();
        let file = src.join("lib.rs");
        std::fs::write(&file, "fn main() {}").unwrap();
        let before = signature(&root);
        assert_eq!(before, signature(&root), "nothing changed, nothing rebuilds");
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&file, "fn main() { println!(\"hi\"); }").unwrap();
        assert_ne!(before, signature(&root), "a longer file changes it");
        std::fs::write(src.join("notes.txt"), "not source").unwrap();
        let after_txt = signature(&root);
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(src.join("notes.txt"), "still not source, but longer").unwrap();
        assert_eq!(after_txt, signature(&root), "other files are ignored");
        let _ = std::fs::remove_dir_all(&root);
    }
}
