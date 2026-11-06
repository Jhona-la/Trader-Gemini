use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "dataset-admission-process-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir_all(root.join("data")).expect("create process fixture");
        Self { root }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dataset_admission"))
}

#[test]
fn real_process_preserves_help_usage_rejection_and_overwrite_exit_codes() {
    let fixture = Fixture::new();
    let help = command().arg("--help").status().expect("run --help");
    assert_eq!(help.code(), Some(0));

    let invalid = command().status().expect("run without arguments");
    assert_eq!(invalid.code(), Some(1));

    let output = fixture.root.join("receipt/manifest.json");
    let rejected = command()
        .arg("--data-root")
        .arg(fixture.root.join("data"))
        .arg("--output")
        .arg(&output)
        .status()
        .expect("run missing-data admission");
    assert_eq!(rejected.code(), Some(2));
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&output).expect("rejection must persist its manifest"))
            .expect("manifest JSON");
    assert_eq!(manifest["classification"], "INSUFICIENTE");
    assert_eq!(manifest["tapes"].as_array().map(Vec::len), Some(9));
    let before = fs::read(&output).expect("read rejected manifest");

    let existing = command()
        .arg("--data-root")
        .arg(fixture.root.join("data"))
        .arg("--output")
        .arg(&output)
        .status()
        .expect("run with preexisting receipt");
    assert_eq!(existing.code(), Some(1));
    assert_eq!(fs::read(output).expect("read preserved receipt"), before);
}
