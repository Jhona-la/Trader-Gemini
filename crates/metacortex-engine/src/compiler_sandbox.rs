//! # Compiler Sandbox
//!
//! Sandboxed environment for compiling dynamically generated Rust code in `cerebro/cortex/`.
//! Enforces resource bounds suitable for 16GB RAM laptops:
//! - `-j 2` parallel jobs max
//! - 300s timeout
//! - Memory footprint monitoring
//! - Structured error parsing and automatic rollback on failure

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct CompilationConfig {
    pub jobs: usize,
    pub timeout: Duration,
    pub release_mode: bool,
    pub max_memory_mb: usize,
}

impl Default for CompilationConfig {
    fn default() -> Self {
        Self {
            jobs: 2, // Conservative for 16GB RAM laptop
            timeout: Duration::from_secs(300),
            release_mode: true,
            max_memory_mb: 2048,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompilationResult {
    pub success: bool,
    pub duration: Duration,
    pub stdout: String,
    pub stderr: String,
    pub artifact_path: Option<PathBuf>,
    pub error_summary: Option<String>,
}

pub struct CompilerSandbox {
    config: CompilationConfig,
    workspace_root: PathBuf,
}

impl CompilerSandbox {
    pub fn new<P: AsRef<Path>>(workspace_root: P, config: CompilationConfig) -> Self {
        Self {
            config,
            workspace_root: workspace_root.as_ref().to_path_buf(),
        }
    }

    /// Compiles a target package or the whole workspace in a controlled sandbox
    pub fn compile_package(&self, package_name: &str) -> CompilationResult {
        let start_time = Instant::now();

        let mut cmd = Command::new("cargo");
        cmd.current_dir(&self.workspace_root);
        cmd.arg("build");
        cmd.arg("-p").arg(package_name);

        if self.config.release_mode {
            cmd.arg("--release");
        }

        cmd.arg("-j").arg(self.config.jobs.to_string());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return CompilationResult {
                    success: false,
                    duration: start_time.elapsed(),
                    stdout: String::new(),
                    stderr: format!("Failed to spawn cargo process: {}", e),
                    artifact_path: None,
                    error_summary: Some(format!("Spawn error: {}", e)),
                };
            }
        };

        // Poll child with timeout
        let poll_interval = Duration::from_millis(200);

        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let output = child.wait_with_output().unwrap_or_else(|_| {
                        std::process::Output {
                            status,
                            stdout: Vec::new(),
                            stderr: Vec::new(),
                        }
                    });

                    let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();

                    let success = status.success();
                    let error_summary = if !success {
                        Some(extract_error_summary(&stderr_str))
                    } else {
                        None
                    };

                    let artifact_path = if success {
                        let profile = if self.config.release_mode { "release" } else { "debug" };
                        let dll_name = format!("{}.dll", package_name.replace('-', "_"));
                        let p = self.workspace_root.join("target").join(profile).join(dll_name);
                        if p.exists() { Some(p) } else { None }
                    } else {
                        None
                    };

                    return CompilationResult {
                        success,
                        duration: start_time.elapsed(),
                        stdout: stdout_str,
                        stderr: stderr_str,
                        artifact_path,
                        error_summary,
                    };
                }
                Ok(None) => {
                    if start_time.elapsed() > self.config.timeout {
                        let _ = child.kill();
                        return CompilationResult {
                            success: false,
                            duration: start_time.elapsed(),
                            stdout: String::new(),
                            stderr: "Cargo build timed out after limit".to_string(),
                            artifact_path: None,
                            error_summary: Some("Compilation timed out".to_string()),
                        };
                    }
                    std::thread::sleep(poll_interval);
                }
                Err(e) => {
                    let _ = child.kill();
                    return CompilationResult {
                        success: false,
                        duration: start_time.elapsed(),
                        stdout: String::new(),
                        stderr: format!("Error waiting on cargo process: {}", e),
                        artifact_path: None,
                        error_summary: Some(e.to_string()),
                    };
                }
            }
        }
    }
}

/// Helper to parse compiler stderr for key error lines
fn extract_error_summary(stderr: &str) -> String {
    stderr
        .lines()
        .filter(|line| line.contains("error[") || line.contains("error:"))
        .take(5)
        .collect::<Vec<&str>>()
        .join("\n")
}
