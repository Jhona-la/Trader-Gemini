use backtest_engine::label_evidence::Provenance;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug, Serialize)]
struct ComponentReceipt {
    status: &'static str,
    reason: String,
}

#[derive(Debug, Serialize)]
struct TapeReceipt {
    symbol: String,
    month: String,
    source_path: String,
    source_status: &'static str,
    present: bool,
    bytes: Option<u64>,
    sha256: Option<String>,
    format: Option<&'static str>,
    provenance: Option<Provenance>,
    integrity: &'static str,
    rows: Option<usize>,
    first_timestamp_ms: Option<u64>,
    last_timestamp_ms: Option<u64>,
    nondecreasing_order: Option<bool>,
    equal_timestamp_rows: Option<u64>,
    duplicate_trade_identity: &'static str,
    largest_observed_gap_ms: Option<u64>,
    first_boundary_gap_ms: Option<u64>,
    last_boundary_gap_ms: Option<u64>,
    coverage: &'static str,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct DatasetManifest {
    schema: &'static str,
    classification: &'static str,
    dataset_role: &'static str,
    tapes: Vec<TapeReceipt>,
    coverage_policy: CoveragePolicy,
    order_book: ComponentReceipt,
    funding: ComponentReceipt,
    fills: ComponentReceipt,
    slippage: ComponentReceipt,
    fees: ComponentReceipt,
    wallet: ComponentReceipt,
    availability: ComponentReceipt,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct CoveragePolicy {
    max_observed_gap_ms: Option<u64>,
    max_boundary_gap_ms: Option<u64>,
}

struct TapeSpec {
    symbol: &'static str,
    month: String,
    year: i32,
    month_number: u32,
    path: PathBuf,
}

struct Config {
    data_root: PathBuf,
    output: PathBuf,
    coverage: CoveragePolicy,
}

fn tape_specs(data_root: &Path) -> Vec<TapeSpec> {
    let mut specs = Vec::with_capacity(9);
    for symbol in ["BNBUSDT", "SOLUSDT", "XRPUSDT"] {
        for (year, month_number) in [(2026, 6), (2026, 7), (2026, 8)] {
            let month = format!("{year:04}-{month_number:02}");
            specs.push(TapeSpec {
                symbol,
                path: data_root.join(format!("{symbol}_{month}_REAL.bin")),
                month,
                year,
                month_number,
            });
        }
    }
    specs
}

fn month_bounds(year: i32, month: u32) -> Result<(u64, u64), String> {
    use chrono::TimeZone;

    let (next_year, next_month) = if month == 12 {
        (year.checked_add(1).ok_or("year overflow")?, 1)
    } else {
        (year, month.checked_add(1).ok_or("month overflow")?)
    };
    let to_milliseconds = |date: chrono::LocalResult<chrono::DateTime<chrono::Utc>>| {
        let timestamp = date
            .single()
            .ok_or_else(|| "invalid UTC month boundary".to_string())?
            .timestamp_millis();
        u64::try_from(timestamp).map_err(|_| "negative timestamps are unsupported".to_string())
    };
    let start = to_milliseconds(chrono::Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0))?;
    let end = to_milliseconds(chrono::Utc.with_ymd_and_hms(next_year, next_month, 1, 0, 0, 0))?;
    Ok((start, end))
}

fn unverified_component(reason: &str) -> ComponentReceipt {
    ComponentReceipt {
        status: "unverified",
        reason: reason.to_string(),
    }
}

fn empty_tape_receipt(spec: &TapeSpec) -> TapeReceipt {
    TapeReceipt {
        symbol: spec.symbol.to_string(),
        month: spec.month.clone(),
        source_path: spec.path.to_string_lossy().into_owned(),
        source_status: "unverified",
        present: false,
        bytes: None,
        sha256: None,
        format: None,
        provenance: None,
        integrity: "missing",
        rows: None,
        first_timestamp_ms: None,
        last_timestamp_ms: None,
        nondecreasing_order: None,
        equal_timestamp_rows: None,
        duplicate_trade_identity: "unverified",
        largest_observed_gap_ms: None,
        first_boundary_gap_ms: None,
        last_boundary_gap_ms: None,
        coverage: "unverified",
        error: None,
    }
}

fn inspect_tape(spec: &TapeSpec, policy: CoveragePolicy) -> TapeReceipt {
    inspect_tape_with_hash_source(spec, policy, &spec.path)
}

fn inspect_tape_with_hash_source(
    spec: &TapeSpec,
    policy: CoveragePolicy,
    hash_source: &Path,
) -> TapeReceipt {
    inspect_tape_with_hooks(spec, policy, hash_source, || {}, || {})
}

fn inspect_tape_with_hooks<F, G>(
    spec: &TapeSpec,
    policy: CoveragePolicy,
    hash_source: &Path,
    before_parse: F,
    before_final_hash: G,
) -> TapeReceipt
where
    F: FnOnce(),
    G: FnOnce(),
{
    inspect_tape_with_readers(
        spec,
        policy,
        hash_source,
        || std::fs::read(&spec.path),
        || std::fs::read(&spec.path),
        before_parse,
        before_final_hash,
    )
}

fn inspect_tape_with_readers<F, G, H, I>(
    spec: &TapeSpec,
    policy: CoveragePolicy,
    hash_source: &Path,
    first_reader: F,
    verification_reader: G,
    before_parse: H,
    before_final_hash: I,
) -> TapeReceipt
where
    F: FnOnce() -> std::io::Result<Vec<u8>>,
    G: FnOnce() -> std::io::Result<Vec<u8>>,
    H: FnOnce(),
    I: FnOnce(),
{
    use backtest_engine::label_evidence::Tape;
    use quantum_engine::model_manifest::{sha256_file, sha256_hex};

    let mut receipt = empty_tape_receipt(spec);
    let snapshot = match first_reader() {
        Ok(bytes) => bytes,
        Err(error) => {
            if error.kind() != std::io::ErrorKind::NotFound {
                receipt.integrity = "unreadable";
            }
            receipt.error = Some(format!("source read failed: {error}"));
            return receipt;
        }
    };
    receipt.present = true;
    receipt.integrity = "invalid";
    receipt.bytes = Some(snapshot.len() as u64);
    let snapshot_sha = sha256_hex(&snapshot);
    if hash_source != spec.path {
        match sha256_file(hash_source) {
            Ok((_, external_sha)) if external_sha == snapshot_sha => {}
            Ok(_) => {
                receipt.error = Some("snapshot hash mismatch".to_string());
                receipt.sha256 = None;
                return receipt;
            }
            Err(error) => {
                receipt.integrity = "unreadable";
                receipt.error = Some(format!("source hash failed: {error}"));
                return receipt;
            }
        }
    }
    receipt.sha256 = Some(snapshot_sha);
    before_parse();

    let tape = match Tape::parse(&snapshot, false) {
        Ok(tape) => tape,
        Err(error) => {
            receipt.error = Some(error);
            return receipt;
        }
    };
    if tape.provenance != Provenance::TradeDerivedUnverifiedBook {
        receipt.error = Some(format!(
            "unsupported provenance: {:?}; synthetic or unknown tapes are not real evidence",
            tape.provenance
        ));
        return receipt;
    }
    receipt.format = Some("TGMTICK1+40B_LE");
    receipt.provenance = Some(tape.provenance);
    receipt.nondecreasing_order = Some(true);
    receipt.rows = Some(tape.len());

    let mut first = None;
    let mut last = None;
    let mut equal_timestamps = 0_u64;
    let mut largest_gap = 0_u64;
    for index in 0..tape.len() {
        let Some(record) = tape.get(index) else {
            receipt.error = Some(format!("record {index} disappeared during admission"));
            receipt.rows = None;
            return receipt;
        };
        first.get_or_insert(record.timestamp_ms);
        if let Some(previous) = last.replace(record.timestamp_ms) {
            if record.timestamp_ms == previous {
                equal_timestamps = equal_timestamps.saturating_add(1);
            }
            largest_gap = largest_gap.max(record.timestamp_ms.saturating_sub(previous));
        }
    }
    receipt.first_timestamp_ms = first;
    receipt.last_timestamp_ms = last;
    receipt.equal_timestamp_rows = Some(equal_timestamps);
    if tape.len() > 1 {
        receipt.largest_observed_gap_ms = Some(largest_gap);
    }

    let (start_ms, end_ms) = match month_bounds(spec.year, spec.month_number) {
        Ok(bounds) => bounds,
        Err(error) => {
            receipt.error = Some(error);
            return receipt;
        }
    };
    let Some(first_ms) = first else {
        receipt.error = Some("no first timestamp".to_string());
        return receipt;
    };
    let Some(last_ms) = last else {
        receipt.error = Some("no last timestamp".to_string());
        return receipt;
    };
    let Some(first_gap) = first_ms.checked_sub(start_ms) else {
        receipt.error = Some(format!(
            "first timestamp {first_ms} precedes declared month start {start_ms}"
        ));
        return receipt;
    };
    let Some(last_gap) = end_ms.checked_sub(last_ms) else {
        receipt.error = Some(format!(
            "last timestamp {last_ms} is outside declared month end {end_ms}"
        ));
        return receipt;
    };
    if last_ms >= end_ms {
        receipt.error = Some(format!(
            "last timestamp {last_ms} is outside declared month end {end_ms}"
        ));
        return receipt;
    }
    receipt.first_boundary_gap_ms = Some(first_gap);
    receipt.last_boundary_gap_ms = Some(last_gap);

    let gap_violation = match (policy.max_observed_gap_ms, receipt.largest_observed_gap_ms) {
        (Some(maximum), Some(observed)) => observed > maximum,
        _ => false,
    };
    let boundary_violation = policy
        .max_boundary_gap_ms
        .is_some_and(|maximum| first_gap > maximum || last_gap > maximum);
    if gap_violation || boundary_violation {
        receipt.error = Some("declared coverage policy exceeded".to_string());
        return receipt;
    }
    before_final_hash();
    let verification = match verification_reader() {
        Ok(bytes) => bytes,
        Err(error) => {
            let mut unreadable = empty_tape_receipt(spec);
            unreadable.present = true;
            unreadable.bytes = Some(snapshot.len() as u64);
            unreadable.integrity = "unreadable";
            unreadable.error = Some(format!("source verification read failed: {error}"));
            return unreadable;
        }
    };
    let verification_sha = sha256_hex(&verification);
    if receipt.sha256.as_deref() != Some(verification_sha.as_str()) {
        let mut invalid = empty_tape_receipt(spec);
        invalid.present = true;
        invalid.bytes = Some(snapshot.len() as u64);
        invalid.integrity = "invalid";
        invalid.error = Some("source changed during snapshot verification".to_string());
        return invalid;
    }
    receipt.integrity = "verified";
    if policy.max_observed_gap_ms.is_some()
        && policy.max_boundary_gap_ms.is_some()
        && receipt.largest_observed_gap_ms.is_some()
    {
        receipt.coverage = "policy_verified";
    }
    receipt
}

fn parse_config(args: &[String]) -> Result<Config, String> {
    let mut data_root = None;
    let mut output = None;
    let mut coverage = CoveragePolicy::default();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--data-root" => data_root = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--max-gap-ms" => {
                let maximum = value
                    .parse::<u64>()
                    .map_err(|_| "--max-gap-ms must be a positive integer".to_string())?;
                if maximum == 0 {
                    return Err("--max-gap-ms must be greater than zero".to_string());
                }
                coverage.max_observed_gap_ms = Some(maximum);
            }
            "--max-edge-gap-ms" => {
                let maximum = value
                    .parse::<u64>()
                    .map_err(|_| "--max-edge-gap-ms must be a positive integer".to_string())?;
                if maximum == 0 {
                    return Err("--max-edge-gap-ms must be greater than zero".to_string());
                }
                coverage.max_boundary_gap_ms = Some(maximum);
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
        index += 2;
    }
    Ok(Config {
        data_root: data_root.ok_or_else(|| "--data-root is required".to_string())?,
        output: output.ok_or_else(|| "--output is required".to_string())?,
        coverage,
    })
}

static NEXT_RECEIPT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct ReceiptReservation {
    output: PathBuf,
    temporary: PathBuf,
    file: Option<std::fs::File>,
}

impl ReceiptReservation {
    fn reserve(output: &Path) -> Result<Self, String> {
        if output.exists() {
            return Err(format!(
                "refusing to overwrite existing receipt: {}",
                output.display()
            ));
        }
        if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("receipt directory creation failed: {error}"))?;
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| format!("receipt temporary clock failed: {error}"))?
            .as_nanos();
        let serial = NEXT_RECEIPT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = output
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("receipt.json"))
            .to_string_lossy();
        let temporary = output.with_file_name(format!(
            ".{name}.tmp.{}.{}.{}",
            std::process::id(),
            nonce,
            serial
        ));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| {
                format!(
                    "refusing to overwrite existing temporary receipt {}: {error}",
                    temporary.display()
                )
            })?;
        Ok(Self {
            output: output.to_path_buf(),
            temporary,
            file: Some(file),
        })
    }

    fn publish(self, bytes: &[u8]) -> Result<(), String> {
        self.publish_with_hook(bytes, |_| {})
    }

    fn publish_with_hook<F>(mut self, bytes: &[u8], before_link: F) -> Result<(), String>
    where
        F: FnOnce(&Path),
    {
        use quantum_engine::model_manifest::{sha256_file, sha256_hex};
        use std::io::Write;

        let mut file = self
            .file
            .take()
            .ok_or_else(|| "receipt reservation was already consumed".to_string())?;
        file.write_all(bytes)
            .map_err(|error| format!("receipt write failed: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("receipt sync failed: {error}"))?;
        let expected_sha = sha256_hex(bytes);
        before_link(&self.temporary);
        let (_, temporary_sha) = sha256_file(&self.temporary)
            .map_err(|error| format!("receipt temporary verification failed: {error}"))?;
        if temporary_sha != expected_sha {
            return Err(
                "receipt temporary content changed; refusing to publish or delete it".to_string(),
            );
        }
        std::fs::hard_link(&self.temporary, &self.output).map_err(|error| {
            format!(
                "receipt publish without replacement failed for {}: {error}",
                self.output.display()
            )
        })?;
        let (_, published_sha) = sha256_file(&self.output)
            .map_err(|error| format!("published receipt verification failed: {error}"))?;
        if published_sha != expected_sha {
            return Err(
                "published receipt content mismatch; quarantine required, no file deleted"
                    .to_string(),
            );
        }
        drop(file);
        Ok(())
    }
}

impl Drop for ReceiptReservation {
    fn drop(&mut self) {
        drop(self.file.take());
    }
}

fn run(args: &[String]) -> Result<(PathBuf, DatasetManifest), String> {
    run_with_hook(args, |_, _| {})
}

fn run_with_hook<F>(args: &[String], before_write: F) -> Result<(PathBuf, DatasetManifest), String>
where
    F: FnOnce(&Path, &Path),
{
    let config = parse_config(args)?;
    let reservation = ReceiptReservation::reserve(&config.output)?;
    let tapes = tape_specs(&config.data_root)
        .iter()
        .map(|spec| inspect_tape(spec, config.coverage))
        .collect();
    let manifest = DatasetManifest {
        schema: "dataset_admission.v1",
        classification: "INSUFICIENTE",
        dataset_role: "development_not_final_oos",
        tapes,
        coverage_policy: config.coverage,
        order_book: unverified_component(
            "historical L2 depth and queue evidence were not supplied; trade-derived quotes do not verify the real book",
        ),
        funding: unverified_component(
            "historical funding coverage and portfolio accounting are not verified by tape admission",
        ),
        fills: unverified_component("reserved executable fill evidence is unavailable"),
        slippage: unverified_component(
            "slippage cannot be certified without historical book and reserved fills",
        ),
        fees: unverified_component("account-tier fee evidence is unavailable"),
        wallet: unverified_component("real wallet equity was not read by this data-only command"),
        availability: unverified_component(
            "source publication dates, ZIP hashes and downstream lineage remain unverified",
        ),
    };
    let encoded = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("manifest serialization failed: {error}"))?;
    before_write(&reservation.output, &reservation.temporary);
    reservation.publish(&encoded)?;
    Ok((config.output, manifest))
}

fn exit_code_for(manifest: &DatasetManifest) -> u8 {
    if manifest.classification == "COMPATIBLE" {
        0
    } else {
        2
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("dataset_admission --data-root <DIR> --output <NEW_JSON>");
        return ExitCode::SUCCESS;
    }
    match run(&args) {
        Ok((path, manifest)) => {
            println!("{}: {}", manifest.classification, path.display());
            ExitCode::from(exit_code_for(&manifest))
        }
        Err(error) => {
            eprintln!("dataset_admission: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantum_engine::model_manifest::sha256_file;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "dataset-admission-{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir_all(root.join("data")).expect("create fixture data directory");
            Self { root }
        }

        fn data_root(&self) -> PathBuf {
            self.root.join("data")
        }

        fn output(&self) -> PathBuf {
            self.root.join("receipt/manifest.json")
        }

        fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.data_root().join(name);
            fs::write(&path, bytes).expect("write fixture tape");
            path
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn record(timestamp_ms: u64, bid: f64, ask: f64) -> [u8; 40] {
        let mut bytes = [0_u8; 40];
        bytes[..8].copy_from_slice(&timestamp_ms.to_le_bytes());
        bytes[8..16].copy_from_slice(&bid.to_le_bytes());
        bytes[16..24].copy_from_slice(&ask.to_le_bytes());
        bytes[24..32].copy_from_slice(&2.0_f64.to_le_bytes());
        bytes[32..40].copy_from_slice(&3.0_f64.to_le_bytes());
        bytes
    }

    fn args(data_root: &Path, output: &Path) -> Vec<String> {
        vec![
            "--data-root".to_string(),
            data_root.to_string_lossy().into_owned(),
            "--output".to_string(),
            output.to_string_lossy().into_owned(),
        ]
    }

    #[test]
    fn truncated_tail_hashes_every_byte_without_fabricating_coverage() {
        let fixture = Fixture::new();
        let mut bytes = b"TGMTICK1".to_vec();
        bytes.extend_from_slice(&record(1_780_300_800_000, 100.0, 100.5));
        bytes.extend_from_slice(&[0xa5; 7]);
        let source = fixture.write("BNBUSDT_2026-06_REAL.bin", &bytes);
        let (_, expected_sha) = sha256_file(&source).expect("hash fixture");

        let output = fixture.output();
        let (written, manifest) = run(&args(&fixture.data_root(), &output))
            .expect("an invalid tape must produce a receipt, not an I/O failure");
        assert_eq!(manifest.classification, "INSUFICIENTE");
        assert_eq!(exit_code_for(&manifest), 2);
        assert_eq!(written, output);
        let tape = manifest
            .tapes
            .iter()
            .find(|tape| tape.symbol == "BNBUSDT" && tape.month == "2026-06")
            .expect("BNB June receipt");
        assert!(tape.present);
        assert_eq!(tape.integrity, "invalid");
        assert_eq!(tape.bytes, Some(bytes.len() as u64));
        assert_eq!(tape.sha256.as_deref(), Some(expected_sha.as_str()));
        assert_eq!(tape.rows, None);
        assert_eq!(tape.first_timestamp_ms, None);
        assert_eq!(tape.last_timestamp_ms, None);
        assert_eq!(tape.coverage, "unverified");
        assert_eq!(fs::read(&source).expect("read original"), bytes);

        let persisted: serde_json::Value = serde_json::from_slice(
            &fs::read(&written).expect("manifest must be persisted before rejection"),
        )
        .expect("valid JSON manifest");
        assert_eq!(persisted["classification"], "INSUFICIENTE");
        let bnb = persisted["tapes"]
            .as_array()
            .expect("tape array")
            .iter()
            .find(|tape| tape["symbol"] == "BNBUSDT" && tape["month"] == "2026-06")
            .expect("persisted BNB receipt");
        assert!(bnb["rows"].is_null());
        assert!(bnb["first_timestamp_ms"].is_null());
        assert!(bnb["last_timestamp_ms"].is_null());
        assert_eq!(bnb["coverage"], "unverified");
        assert_eq!(bnb["sha256"], expected_sha);
    }

    fn bnb_june(manifest: &DatasetManifest) -> &TapeReceipt {
        manifest
            .tapes
            .iter()
            .find(|tape| tape.symbol == "BNBUSDT" && tape.month == "2026-06")
            .expect("BNB June receipt")
    }

    fn tape_bytes(header: &[u8], records: &[[u8; 40]], tail: &[u8]) -> Vec<u8> {
        let mut bytes = header.to_vec();
        for record in records {
            bytes.extend_from_slice(record);
        }
        bytes.extend_from_slice(tail);
        bytes
    }

    fn overwrite_mapped_fixture(path: &Path, bytes: &[u8]) {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .expect("open mutable fixture");
        // SAFETY: the fixture stays open and the byte slice matches its mapping length.
        let mut mapping =
            unsafe { memmap2::MmapOptions::new().map_mut(&file) }.expect("map mutable fixture");
        assert_eq!(mapping.len(), bytes.len());
        mapping.copy_from_slice(bytes);
        mapping.flush().expect("flush mutable fixture");
    }

    fn write_valid_tapes(fixture: &Fixture) {
        let cases = [
            ("BNBUSDT", "2026-06", 1_780_272_000_000, 1_782_864_000_000),
            ("BNBUSDT", "2026-07", 1_782_864_000_000, 1_785_542_400_000),
            ("BNBUSDT", "2026-08", 1_785_542_400_000, 1_788_220_800_000),
            ("SOLUSDT", "2026-06", 1_780_272_000_000, 1_782_864_000_000),
            ("SOLUSDT", "2026-07", 1_782_864_000_000, 1_785_542_400_000),
            ("SOLUSDT", "2026-08", 1_785_542_400_000, 1_788_220_800_000),
            ("XRPUSDT", "2026-06", 1_780_272_000_000, 1_782_864_000_000),
            ("XRPUSDT", "2026-07", 1_782_864_000_000, 1_785_542_400_000),
            ("XRPUSDT", "2026-08", 1_785_542_400_000, 1_788_220_800_000),
        ];
        for (symbol, month, start, end) in cases {
            let bytes = tape_bytes(
                b"TGMTICK1",
                &[
                    record(start + 60_000, 100.0, 100.5),
                    record(end - 60_000, 101.0, 101.5),
                    record(end - 60_000, 101.0, 101.5),
                ],
                &[],
            );
            fixture.write(&format!("{symbol}_{month}_REAL.bin"), &bytes);
        }
    }

    #[test]
    fn valid_tapes_preserve_equal_timestamps_and_require_explicit_coverage_policy() {
        let fixture = Fixture::new();
        write_valid_tapes(&fixture);
        let output = fixture.output();
        let (_, manifest) = run(&args(&fixture.data_root(), &output)).expect("write receipt");
        assert_eq!(manifest.tapes.len(), 9);
        for tape in &manifest.tapes {
            assert_eq!(tape.integrity, "verified", "{} {}", tape.symbol, tape.month);
            assert_eq!(tape.rows, Some(3), "{} {}", tape.symbol, tape.month);
            assert_eq!(tape.equal_timestamp_rows, Some(1));
        }
        let tape = bnb_june(&manifest);
        assert_eq!(tape.integrity, "verified");
        assert_eq!(tape.rows, Some(3));
        assert_eq!(tape.equal_timestamp_rows, Some(1));
        assert_eq!(tape.duplicate_trade_identity, "unverified");
        assert_eq!(tape.nondecreasing_order, Some(true));
        assert_eq!(tape.coverage, "unverified");
        assert!(tape.largest_observed_gap_ms.is_some());
        assert!(tape.error.is_none());
        assert_eq!(manifest.classification, "INSUFICIENTE");
        for status in [
            manifest.order_book.status,
            manifest.funding.status,
            manifest.fills.status,
            manifest.slippage.status,
            manifest.fees.status,
            manifest.wallet.status,
            manifest.availability.status,
        ] {
            assert_eq!(status, "unverified");
        }

        let policy_output = fixture.root.join("receipt/policy.json");
        let mut policy_args = args(&fixture.data_root(), &policy_output);
        policy_args.extend([
            "--max-gap-ms".to_string(),
            u64::MAX.to_string(),
            "--max-edge-gap-ms".to_string(),
            "60000".to_string(),
        ]);
        let (_, policy_manifest) = run(&policy_args).expect("write policy receipt");
        let tape = bnb_june(&policy_manifest);
        assert_eq!(tape.integrity, "verified");
        assert_eq!(tape.coverage, "policy_verified");
        assert_eq!(tape.first_boundary_gap_ms, Some(60_000));
        assert_eq!(tape.last_boundary_gap_ms, Some(60_000));
        assert_eq!(policy_manifest.classification, "INSUFICIENTE");
    }

    #[test]
    fn unknown_synthetic_legacy_and_future_headers_fail_closed() {
        for (index, header) in [
            &b"UNKNOWN!"[..],
            &b"TGMSYNT1"[..],
            &b""[..],
            &b"TGMTICK2"[..],
        ]
        .into_iter()
        .enumerate()
        {
            let fixture = Fixture::new();
            let (start, _) = month_bounds(2026, 6).expect("month bounds");
            let source = fixture.write(
                "BNBUSDT_2026-06_REAL.bin",
                &tape_bytes(header, &[record(start, 100.0, 100.5)], &[]),
            );
            let before = fs::read(&source).expect("read fixture");
            let (_, manifest) = run(&args(&fixture.data_root(), &fixture.output()))
                .expect("write rejection receipt");
            let tape = bnb_june(&manifest);
            assert_eq!(tape.integrity, "invalid", "case {index}");
            assert_eq!(tape.rows, None, "case {index}");
            assert!(tape.error.is_some(), "case {index}");
            assert_eq!(tape.coverage, "unverified", "case {index}");
            assert_eq!(fs::read(&source).expect("read original"), before);
        }
    }

    #[test]
    fn nonfinite_crossed_decreasing_and_out_of_month_records_are_rejected() {
        let (start, end) = month_bounds(2026, 6).expect("month bounds");
        let nan = record(start, f64::NAN, 100.5);
        let crossed = record(start, 101.0, 100.0);
        let mut negative_quantity = record(start, 100.0, 100.5);
        negative_quantity[24..32].copy_from_slice(&(-1.0_f64).to_le_bytes());
        let cases = vec![
            tape_bytes(b"TGMTICK1", &[nan], &[]),
            tape_bytes(b"TGMTICK1", &[crossed], &[]),
            tape_bytes(b"TGMTICK1", &[negative_quantity], &[]),
            tape_bytes(
                b"TGMTICK1",
                &[
                    record(start + 2_000, 100.0, 100.5),
                    record(start + 1_000, 100.0, 100.5),
                ],
                &[],
            ),
            tape_bytes(b"TGMTICK1", &[record(start - 1, 100.0, 100.5)], &[]),
            tape_bytes(b"TGMTICK1", &[record(end, 100.0, 100.5)], &[]),
        ];
        for (index, bytes) in cases.into_iter().enumerate() {
            let fixture = Fixture::new();
            fixture.write("BNBUSDT_2026-06_REAL.bin", &bytes);
            let (_, manifest) = run(&args(&fixture.data_root(), &fixture.output()))
                .expect("write rejection receipt");
            let tape = bnb_june(&manifest);
            assert_eq!(tape.integrity, "invalid", "case {index}");
            assert!(tape.error.is_some(), "case {index}");
            assert_eq!(tape.coverage, "unverified", "case {index}");
            if index < 4 {
                assert_eq!(tape.rows, None, "case {index}");
            } else {
                assert_eq!(tape.rows, Some(1), "case {index}");
            }
        }
    }

    #[test]
    fn observed_gap_and_month_edge_policy_are_measured_not_imputed() {
        let fixture = Fixture::new();
        let (start, _) = month_bounds(2026, 6).expect("month bounds");
        fixture.write(
            "BNBUSDT_2026-06_REAL.bin",
            &tape_bytes(
                b"TGMTICK1",
                &[
                    record(start + 1_000, 100.0, 100.5),
                    record(start + 100_000, 100.0, 100.5),
                ],
                &[],
            ),
        );
        let mut gap_args = args(&fixture.data_root(), &fixture.output());
        gap_args.extend([
            "--max-gap-ms".to_string(),
            "1000".to_string(),
            "--max-edge-gap-ms".to_string(),
            u64::MAX.to_string(),
        ]);
        let (_, manifest) = run(&gap_args).expect("write gap receipt");
        let tape = bnb_june(&manifest);
        assert_eq!(tape.integrity, "invalid");
        assert_eq!(tape.largest_observed_gap_ms, Some(99_000));
        assert_eq!(tape.first_boundary_gap_ms, Some(1_000));
        assert_eq!(tape.coverage, "unverified");

        let edge_output = fixture.root.join("receipt/edge.json");
        let mut edge_args = args(&fixture.data_root(), &edge_output);
        edge_args.extend([
            "--max-gap-ms".to_string(),
            u64::MAX.to_string(),
            "--max-edge-gap-ms".to_string(),
            "500".to_string(),
        ]);
        let (_, manifest) = run(&edge_args).expect("write edge receipt");
        let tape = bnb_june(&manifest);
        assert_eq!(tape.integrity, "invalid");
        assert_eq!(tape.first_boundary_gap_ms, Some(1_000));
        assert_eq!(tape.coverage, "unverified");
    }

    #[test]
    fn missing_inputs_and_preexisting_receipt_never_approve_or_overwrite() {
        let fixture = Fixture::new();
        let (_, manifest) = run(&args(&fixture.data_root(), &fixture.output()))
            .expect("missing files still need a manifest");
        assert_eq!(manifest.tapes.len(), 9);
        for tape in &manifest.tapes {
            assert!(!tape.present);
            assert_eq!(tape.integrity, "missing");
            assert_eq!(tape.sha256, None);
            assert_eq!(tape.rows, None);
            assert_eq!(tape.coverage, "unverified");
        }
        assert_eq!(exit_code_for(&manifest), 2);

        let sentinel = b"do not overwrite";
        fs::write(fixture.output(), sentinel).expect("write sentinel");
        let error = run(&args(&fixture.data_root(), &fixture.output()))
            .expect_err("existing receipt must fail");
        assert!(error.contains("refusing to overwrite"));
        assert_eq!(fs::read(fixture.output()).expect("read sentinel"), sentinel);
    }

    #[test]
    fn invalid_arguments_and_zero_thresholds_are_usage_errors() {
        let fixture = Fixture::new();
        for extra in [
            vec!["--max-gap-ms".to_string(), "0".to_string()],
            vec!["--max-edge-gap-ms".to_string(), "0".to_string()],
            vec!["--unknown".to_string(), "1".to_string()],
        ] {
            let mut arguments = args(&fixture.data_root(), &fixture.output());
            arguments.extend(extra);
            assert!(run(&arguments).is_err());
            assert!(!fixture.output().exists());
        }
        assert!(run(&[]).is_err());
    }

    #[test]
    fn equal_length_hash_source_mismatch_is_rejected() {
        let fixture = Fixture::new();
        let (start, _) = month_bounds(2026, 6).expect("month bounds");
        let parsed = tape_bytes(b"TGMTICK1", &[record(start + 1_000, 100.0, 100.5)], &[]);
        let different_same_length =
            tape_bytes(b"TGMTICK1", &[record(start + 2_000, 101.0, 101.5)], &[]);
        assert_eq!(parsed.len(), different_same_length.len());
        let parsed_path = fixture.write("BNBUSDT_2026-06_REAL.bin", &parsed);
        let hash_path = fixture.write("different.bin", &different_same_length);
        let spec = TapeSpec {
            symbol: "BNBUSDT",
            month: "2026-06".to_string(),
            year: 2026,
            month_number: 6,
            path: parsed_path,
        };
        let receipt = inspect_tape_with_hash_source(&spec, CoveragePolicy::default(), &hash_path);
        assert_eq!(receipt.integrity, "invalid");
        assert_eq!(receipt.sha256, None);
        assert_eq!(receipt.rows, None);
        assert_eq!(receipt.coverage, "unverified");
        assert!(receipt
            .error
            .as_deref()
            .is_some_and(|error| error.contains("mutated") || error.contains("mismatch")));
    }

    #[test]
    fn one_record_cannot_verify_observed_continuity() {
        let fixture = Fixture::new();
        let (start, _) = month_bounds(2026, 6).expect("month bounds");
        fixture.write(
            "BNBUSDT_2026-06_REAL.bin",
            &tape_bytes(
                b"TGMTICK1",
                &[record(start + 86_400_000, 100.0, 100.5)],
                &[],
            ),
        );
        let mut arguments = args(&fixture.data_root(), &fixture.output());
        arguments.extend([
            "--max-gap-ms".to_string(),
            "1".to_string(),
            "--max-edge-gap-ms".to_string(),
            u64::MAX.to_string(),
        ]);
        let (_, manifest) = run(&arguments).expect("write single-row receipt");
        let tape = bnb_june(&manifest);
        assert_eq!(tape.rows, Some(1));
        assert_eq!(tape.largest_observed_gap_ms, None);
        assert_eq!(tape.coverage, "unverified");
    }

    #[test]
    fn competing_receipt_is_never_truncated_or_replaced() {
        let fixture = Fixture::new();
        let sentinel = b"competing receipt";
        let output = fixture.output();
        let temporary = std::cell::RefCell::new(None);
        let error = run_with_hook(&args(&fixture.data_root(), &output), |output, path| {
            assert!(path.exists(), "temporary receipt must be reserved");
            temporary.replace(Some(path.to_path_buf()));
            let competitor = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path);
            assert!(
                competitor.is_err(),
                "temporary must be create_new exclusive"
            );
            fs::write(output, sentinel).expect("create competing final receipt");
        })
        .expect_err("competing final receipt must not be replaced");
        assert!(error.contains("receipt") || error.contains("publish"));
        assert_eq!(fs::read(&output).expect("read competing receipt"), sentinel);
        let temporary = temporary.into_inner().expect("captured temporary");
        let retained: serde_json::Value = serde_json::from_slice(
            &fs::read(temporary).expect("owned temporary retained for reconciliation"),
        )
        .expect("retained receipt JSON");
        assert_eq!(retained["classification"], "INSUFICIENTE");
    }

    #[test]
    fn coverage_policy_is_serialized_with_the_receipt() {
        let fixture = Fixture::new();
        write_valid_tapes(&fixture);
        let mut arguments = args(&fixture.data_root(), &fixture.output());
        arguments.extend([
            "--max-gap-ms".to_string(),
            "123456".to_string(),
            "--max-edge-gap-ms".to_string(),
            "60000".to_string(),
        ]);
        let (written, _) = run(&arguments).expect("write policy receipt");
        let persisted: serde_json::Value =
            serde_json::from_slice(&fs::read(written).expect("read policy receipt"))
                .expect("policy JSON");
        assert_eq!(persisted["coverage_policy"]["max_observed_gap_ms"], 123456);
        assert_eq!(persisted["coverage_policy"]["max_boundary_gap_ms"], 60000);
    }

    #[test]
    fn unreadable_source_is_not_reported_as_missing() {
        let fixture = Fixture::new();
        fs::create_dir(fixture.data_root().join("BNBUSDT_2026-06_REAL.bin"))
            .expect("directory instead of readable tape");
        let (_, manifest) = run(&args(&fixture.data_root(), &fixture.output()))
            .expect("write unreadable-source receipt");
        assert_eq!(bnb_june(&manifest).integrity, "unreadable");
        assert_eq!(manifest.classification, "INSUFICIENTE");
    }

    #[test]
    fn aba_mutation_cannot_mix_hash_and_statistics() {
        let fixture = Fixture::new();
        let (start, _) = month_bounds(2026, 6).expect("month bounds");
        let original = tape_bytes(b"TGMTICK1", &[record(start + 1_000, 100.0, 100.5)], &[]);
        let replacement = tape_bytes(b"TGMTICK1", &[record(start + 2_000, 101.0, 101.5)], &[]);
        assert_eq!(original.len(), replacement.len());
        let source = fixture.write("BNBUSDT_2026-06_REAL.bin", &original);
        let spec = TapeSpec {
            symbol: "BNBUSDT",
            month: "2026-06".to_string(),
            year: 2026,
            month_number: 6,
            path: source.clone(),
        };
        let expected_sha = quantum_engine::model_manifest::sha256_hex(&original);
        let replacement_for_hook = replacement.clone();
        let source_for_hook = source.clone();
        let original_for_hook = original.clone();
        let restore_source = source.clone();
        let receipt = inspect_tape_with_hooks(
            &spec,
            CoveragePolicy::default(),
            &source,
            move || {
                overwrite_mapped_fixture(&source_for_hook, &replacement_for_hook);
            },
            move || {
                overwrite_mapped_fixture(&restore_source, &original_for_hook);
            },
        );
        assert_eq!(receipt.integrity, "verified");
        assert_eq!(receipt.sha256.as_deref(), Some(expected_sha.as_str()));
        assert_eq!(receipt.first_timestamp_ms, Some(start + 1_000));
        assert_eq!(receipt.rows, Some(1));
    }

    #[test]
    fn substituted_temporary_cannot_publish_or_be_deleted() {
        let fixture = Fixture::new();
        let output = fixture.output();
        let reservation = ReceiptReservation::reserve(&output).expect("reserve receipt");
        let temporary_path = reservation.temporary.clone();
        let replaced = std::cell::Cell::new(false);
        let result = reservation.publish_with_hook(b"owned", |temporary| {
            if fs::remove_file(temporary).is_ok() {
                replaced.set(true);
                fs::write(temporary, b"foreign").expect("substitute temporary");
            }
        });
        if replaced.get() {
            assert!(result.is_err(), "substituted temporary must not publish");
            assert!(!output.exists(), "foreign receipt must not be published");
            assert_eq!(
                fs::read(temporary_path).expect("foreign temporary must be preserved"),
                b"foreign"
            );
        } else {
            result.expect("open reserved file prevents substitution on this platform");
            assert_eq!(fs::read(&output).expect("read owned receipt"), b"owned");
            assert_eq!(
                fs::read(temporary_path).expect("retained owned temporary"),
                b"owned"
            );
        }
    }

    #[test]
    fn hybrid_read_is_rejected_by_independent_source_verification() {
        let fixture = Fixture::new();
        let (start, _) = month_bounds(2026, 6).expect("month bounds");
        let stable = tape_bytes(
            b"TGMTICK1",
            &[
                record(start, f64::NAN, 100.5),
                record(start + 1_000, 101.0, 101.5),
            ],
            &[],
        );
        let hybrid = tape_bytes(
            b"TGMTICK1",
            &[
                record(start, 100.0, 100.5),
                record(start + 1_000, 101.0, 101.5),
            ],
            &[],
        );
        let spec = TapeSpec {
            symbol: "BNBUSDT",
            month: "2026-06".to_string(),
            year: 2026,
            month_number: 6,
            path: fixture.data_root().join("BNBUSDT_2026-06_REAL.bin"),
        };
        let receipt = inspect_tape_with_readers(
            &spec,
            CoveragePolicy::default(),
            &spec.path,
            || Ok(hybrid.clone()),
            || Ok(stable.clone()),
            || {},
            || {},
        );
        assert_eq!(receipt.integrity, "invalid");
        assert_eq!(receipt.sha256, None);
        assert_eq!(receipt.rows, None);
        assert_eq!(receipt.coverage, "unverified");
        assert!(receipt
            .error
            .as_deref()
            .is_some_and(|error| error.contains("changed") || error.contains("mismatch")));
    }
}
