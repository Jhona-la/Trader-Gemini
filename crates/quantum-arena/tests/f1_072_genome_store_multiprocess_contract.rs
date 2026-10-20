//! F1-072: promotion ordering across eight OS processes.
//! The fixture uses only private temporary files and child-local environment.

use quantum_arena::genome::SuperGenotype;
use quantum_arena::genome_store::GenomeEnvelope;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const WORKERS: usize = 8;
const CHILD_TEST: &str = "f1_072_multiprocess_promotion_child";
const TOKEN_ENV: &str = "TG_F1072_MULTIPROCESS_TOKEN";
const WORKER_ENV: &str = "TG_F1072_MULTIPROCESS_WORKER";
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const GO_TIMEOUT: Duration = Duration::from_secs(90);
const FINISH_TIMEOUT: Duration = Duration::from_secs(60);
const POLL_INTERVAL: Duration = Duration::from_millis(10);

struct Fixture {
    root: PathBuf,
    canonical_parent: PathBuf,
    token: String,
}

impl Fixture {
    fn new() -> Self {
        let temp_parent = std::env::temp_dir();
        assert!(temp_parent.is_absolute(), "temporary parent must be absolute");
        let canonical_parent = temp_parent.canonicalize().expect("resolve temporary parent");
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).expect("fixture clock").as_nanos();
        for attempt in 0..32_u32 {
            let token = format!("tg-f1072-mp-{}-{nonce:x}-{attempt}", std::process::id());
            let root = temp_parent.join(&token);
            match fs::create_dir(&root) {
                Ok(()) => {
                    // Keep the ordinary absolute path for TG_CONFIG_ROOT: canonicalize
                    // may introduce Windows' verbatim prefix, whereas config_join
                    // appends forward slashes. Resolution is checked separately.
                    let fixture = Self { root, canonical_parent, token };
                    assert_eq!(fixture.root.canonicalize().expect("resolve fixture").parent(), Some(fixture.canonical_parent.as_path()));
                    fixture.root.to_str().expect("TG_CONFIG_ROOT requires a UTF-8 path");
                    fs::create_dir(fixture.root.join("ready")).expect("create readiness directory");
                    fs::create_dir(fixture.root.join("receipts")).expect("create receipt directory");
                    fs::create_dir(fixture.root.join("logs")).expect("create child log directory");
                    write_new(&fixture.root.join("fixture.token"), fixture.token.as_bytes());
                    assert!(!fixture.root.join("genomes").exists(), "fixture must start without lineage");
                    return fixture;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create private fixture: {error}"),
            }
        }
        panic!("could not claim a fresh temporary directory");
    }

    fn release(&self) {
        let staging = self.root.join("go.pending");
        write_new(&staging, self.token.as_bytes());
        fs::rename(staging, self.root.join("go")).expect("publish complete barrier token");
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // The parent declares Workers after Fixture, so its children are reaped
        // before this cleanup, also during unwinding. Resolve and bound this
        // exact newly-created child of temp_dir before recursive removal.
        let owned_directory = fs::symlink_metadata(&self.root)
            .map(|m| m.is_dir() && !m.file_type().is_symlink())
            .unwrap_or(false);
        let within_parent = self.root.canonicalize()
            .map(|p| p.parent() == Some(self.canonical_parent.as_path())
                && p.file_name() == Some(std::ffi::OsStr::new(&self.token)))
            .unwrap_or(false);
        if owned_directory && within_parent {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

fn write_new(path: &Path, bytes: &[u8]) {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)
        .unwrap_or_else(|error| panic!("create {}: {error}", path.display()));
    file.write_all(bytes).expect("write private fixture file");
    file.sync_all().expect("sync private fixture file");
}

fn source_for(token: &str, worker: usize) -> String {
    format!("{token}-worker-{worker}")
}

fn log_excerpt(root: &Path, worker: usize, stream: &str) -> String {
    let path = root.join("logs").join(format!("worker-{worker}.{stream}"));
    let mut text = String::new();
    if let Ok(file) = File::open(path) {
        let _ = file.take(8_192).read_to_string(&mut text);
    }
    text
}

struct OwnedChild {
    worker: usize,
    process: Child,
}

struct Workers {
    children: Vec<OwnedChild>,
}

impl Workers {
    fn new() -> Self {
        Self { children: Vec::with_capacity(WORKERS) }
    }

    fn spawn(&mut self, fixture: &Fixture, worker: usize, executable: &Path) {
        let stdout = File::create(fixture.root.join("logs").join(format!("worker-{worker}.stdout")))
            .expect("create child stdout");
        let stderr = File::create(fixture.root.join("logs").join(format!("worker-{worker}.stderr")))
            .expect("create child stderr");
        let process = Command::new(executable)
            .args(["--exact", CHILD_TEST, "--ignored", "--nocapture", "--test-threads=1"])
            .current_dir(&fixture.root)
            .env("TG_CONFIG_ROOT", &fixture.root)
            .env_remove("TGM_CONFIG_DIR") // This override otherwise wins over TG_CONFIG_ROOT.
            .env("TG_GENOME_ENV", "demo")
            .env(TOKEN_ENV, &fixture.token)
            .env(WORKER_ENV, worker.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn().expect("spawn owned integration-test child");
        self.children.push(OwnedChild { worker, process });
    }

    fn await_ready(&mut self, fixture: &Fixture) {
        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            for child in &mut self.children {
                if let Some(status) = child.process.try_wait().expect("poll readiness child") {
                    panic!("worker {} exited before go: {status}; stdout={} stderr={}",
                        child.worker, log_excerpt(&fixture.root, child.worker, "stdout"),
                        log_excerpt(&fixture.root, child.worker, "stderr"));
                }
            }
            if (0..WORKERS).all(|worker| fixture.root.join("ready").join(format!("worker-{worker}")).is_file()) {
                return;
            }
            assert!(Instant::now() < deadline, "timeout waiting for all eight ready workers");
            thread::sleep(POLL_INTERVAL);
        }
    }

    fn await_success(&mut self, fixture: &Fixture) {
        let deadline = Instant::now() + FINISH_TIMEOUT;
        loop {
            let mut all_finished = true;
            for child in &mut self.children {
                match child.process.try_wait().expect("poll promotion child") {
                    Some(status) => assert!(status.success(),
                        "worker {} failed: {status}; stdout={} stderr={}", child.worker,
                        log_excerpt(&fixture.root, child.worker, "stdout"),
                        log_excerpt(&fixture.root, child.worker, "stderr")),
                    None => all_finished = false,
                }
            }
            if all_finished { return; }
            assert!(Instant::now() < deadline, "timeout waiting for eight successful promotions");
            thread::sleep(POLL_INTERVAL);
        }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        // Only handles created by this parent are touched. No process-name/PID
        // enumeration, shared-process cancellation, or orphaned unjoined child.
        for child in &mut self.children {
            if !matches!(child.process.try_wait(), Ok(Some(_))) {
                let _ = child.process.kill();
            }
            let _ = child.process.wait();
        }
    }
}

fn json_file(path: &Path) -> Value {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

#[test]
fn test_f1_072_multiprocess_promotions_preserve_ordered_complete_lineage() {
    let fixture = Fixture::new();
    let mut workers = Workers::new(); // Dropped before fixture on every exit path.
    let executable = std::env::current_exe().expect("current integration-test executable");
    for worker in 0..WORKERS { workers.spawn(&fixture, worker, &executable); }
    workers.await_ready(&fixture);
    fixture.release();
    workers.await_success(&fixture);

    let receipt_dir = fixture.root.join("receipts");
    assert_eq!(fs::read_dir(&receipt_dir).expect("list receipts").count(), WORKERS,
        "there must be exactly one receipt per child");
    let mut by_generation: BTreeMap<u64, Value> = BTreeMap::new();
    for worker in 0..WORKERS {
        let receipt = json_file(&receipt_dir.join(format!("worker-{worker}.json")));
        assert_eq!(receipt["token"].as_str(), Some(fixture.token.as_str()));
        assert_eq!(receipt["worker"].as_u64(), Some(worker as u64));
        assert_eq!(receipt["ok"].as_bool(), Some(true), "worker {worker}: {receipt}");
        let envelope_json = receipt.get("envelope").expect("successful promotion envelope").clone();
        let envelope: GenomeEnvelope = serde_json::from_value(envelope_json.clone())
            .expect("parse child promotion result");
        assert_eq!(envelope.source, source_for(&fixture.token, worker));
        assert_eq!(envelope.genome.btc_volatility_multiplier, 1.0 + worker as f64 / 16.0,
            "each worker must publish a distinguishable valid genome");
        assert!((1..=WORKERS as u64).contains(&envelope.generation));
        assert_eq!(envelope.parent_generation, envelope.generation - 1,
            "clean serialized promotions must descend from the preceding publication");
        assert!(by_generation.insert(envelope.generation, envelope_json).is_none(),
            "two successful processes returned the same generation");
    }
    assert_eq!(by_generation.keys().copied().collect::<Vec<_>>(), (1..=WORKERS as u64).collect::<Vec<_>>());

    // Read files directly: load_active/load_generation must not repair a missing
    // history entry and accidentally make this concurrency contract pass.
    let history_dir = fixture.root.join("genomes/demo/history");
    let history_entries: Vec<_> = fs::read_dir(&history_dir).expect("list history")
        .map(|entry| entry.expect("read history entry")).collect();
    assert_eq!(history_entries.len(), WORKERS, "exactly eight history files, no gaps or leftovers");
    for entry in &history_entries {
        assert!(entry.file_type().expect("history file type").is_file(), "history entries must be files");
    }
    for generation in 1..=WORKERS as u64 {
        let path = history_dir.join(format!("gen_{generation:06}.json"));
        let history = json_file(&path);
        let envelope: GenomeEnvelope = serde_json::from_value(history.clone()).expect("parse history envelope");
        assert_eq!(envelope.generation, generation);
        assert_eq!(envelope.parent_generation, generation - 1);
        assert_eq!(&history, &by_generation[&generation],
            "history must preserve the complete result of its owning process");
    }
    let active = json_file(&fixture.root.join("genomes/demo/active.json"));
    let active_envelope: GenomeEnvelope = serde_json::from_value(active.clone()).expect("parse active envelope");
    assert_eq!(active_envelope.generation, WORKERS as u64, "active must publish the highest completed generation");
    assert_eq!(&active, &by_generation[&(WORKERS as u64)],
        "the complete active envelope/genome must equal history of the maximum generation");
    let legacy = json_file(&fixture.root.join("genotypes/active_genome.demo.json"));
    assert_eq!(&legacy, &active["genome"],
        "the legacy mirror must finish at the same published genome");
}

#[test]
#[ignore = "child helper; only the parent supplies a private fixture and barrier token"]
fn f1_072_multiprocess_promotion_child() {
    let token = std::env::var(TOKEN_ENV).expect("child requires parent-issued token");
    assert!(token.starts_with("tg-f1072-mp-") && token.len() < 160
        && token.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'), "invalid child token");
    let worker_text = std::env::var(WORKER_ENV).expect("child requires worker identity");
    let worker: usize = worker_text.parse().expect("numeric worker identity");
    assert!(worker < WORKERS && worker_text == worker.to_string(), "invalid worker identity");
    assert_eq!(std::env::var("TG_GENOME_ENV").as_deref(), Ok("demo"));
    assert!(std::env::var_os("TGM_CONFIG_DIR").is_none(), "competing config override must be absent");
    let root = PathBuf::from(std::env::var("TG_CONFIG_ROOT").expect("child requires private config root"));
    assert!(root.is_absolute() && root.file_name() == Some(std::ffi::OsStr::new(&token)));
    assert_eq!(fs::read_to_string(root.join("fixture.token")).expect("read parent token"), token);
    assert_eq!(PathBuf::from(quantum_arena::paths::config_root()), root,
        "library must resolve the child-local configuration root");

    let mut genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    // Distinct dyadic values within this gene's [0.5, 3.0] bounds make an
    // out-of-order legacy mirror observable; TP/SL curves remain unchanged.
    genome.btc_volatility_multiplier = 1.0 + worker as f64 / 16.0;
    GenomeEnvelope::validate(&genome).expect("worker fixture genome must be valid");
    let ready = root.join("ready").join(format!("worker-{worker}"));
    // create_new also prevents two children from claiming the same worker.
    drop(OpenOptions::new().write(true).create_new(true).open(ready).expect("claim readiness slot"));
    let deadline = Instant::now() + GO_TIMEOUT;
    loop {
        match fs::read_to_string(root.join("go")) {
            Ok(go) => { assert_eq!(go, token, "barrier token mismatch"); break; }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("read private barrier: {error}"),
        }
        assert!(Instant::now() < deadline, "child timed out waiting for go");
        thread::sleep(POLL_INTERVAL);
    }

    let source = source_for(&token, worker);
    let reason = format!("F1-072 isolated multiprocess worker {worker}");
    let result = GenomeEnvelope::promote(genome, &source, &reason); // Exactly one call.
    let receipt = match &result {
        Ok(envelope) => json!({"token": token, "worker": worker, "ok": true, "envelope": envelope}),
        Err(error) => json!({"token": token, "worker": worker, "ok": false, "error": error.to_string()}),
    };
    write_new(&root.join("receipts").join(format!("worker-{worker}.json")),
        &serde_json::to_vec_pretty(&receipt).expect("serialize child receipt"));
    assert!(result.is_ok(), "the single child promotion failed: {:?}", result.err());
}
