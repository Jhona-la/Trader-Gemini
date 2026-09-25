//! File admission only; these tests do not certify the legacy concurrent protocol.
use std::{
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
};
use storage_engine::mmap_bus::{MmapTelemetryBus, MmapTelemetryReader};

const FILE_SIZE: u64 = 64 + 1_000_000 * 64;

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("tg-xxx-map-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self) -> PathBuf {
        self.0.join("telemetry.dat")
    }
    fn file(&self, len: u64) -> File {
        let f = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(self.path())
            .unwrap();
        f.set_len(len).unwrap();
        f
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let exact = self.0.canonicalize().unwrap();
        assert!(exact.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        assert!(exact
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("tg-xxx-map-"));
        std::fs::remove_dir_all(exact).unwrap();
    }
}

#[test]
fn writer_rejects_nonempty_incomplete_file_without_overwriting_it() {
    let fixture = Fixture::new();
    let mut f = fixture.file(64);
    f.seek(SeekFrom::Start(16)).unwrap();
    f.write_all(b"existing evidence").unwrap();
    drop(f);
    let before = std::fs::read(fixture.path()).unwrap();
    let result = MmapTelemetryBus::new(fixture.path());
    assert!(
        result.is_err(),
        "opening an incomplete file must not repair it by zeroing existing evidence"
    );
    assert_eq!(std::fs::read(fixture.path()).unwrap(), before);
}

#[test]
fn reader_rejects_header_only_instead_of_fabricating_empty_batch() {
    let fixture = Fixture::new();
    drop(fixture.file(64)); // Safe pre-fix counterexample: complete header, missing ring.
    let mut reader = MmapTelemetryReader::new(fixture.path());
    assert!(reader.read_latest_frames().is_err());
}

#[test]
fn reader_reports_missing_file_and_can_retry_after_creation() {
    let fixture = Fixture::new();
    let mut reader = MmapTelemetryReader::new(fixture.path());
    assert_eq!(
        reader
            .read_latest_frames()
            .err()
            .expect("missing file must be an error")
            .kind(),
        std::io::ErrorKind::NotFound
    );
    drop(fixture.file(FILE_SIZE));
    assert!(reader.read_latest_frames().unwrap().is_empty());
}

#[test]
fn reader_reports_empty_file_as_invalid_not_no_observations() {
    let fixture = Fixture::new();
    drop(fixture.file(0));
    let mut reader = MmapTelemetryReader::new(fixture.path());
    assert!(reader.read_latest_frames().is_err());
}

#[test]
fn new_file_and_existing_complete_file_preserve_valid_frames() {
    let fixture = Fixture::new();
    {
        let bus = MmapTelemetryBus::new(fixture.path()).unwrap();
        bus.write_trace(12, 30, [0.6, 1.0, 0.1, -0.002, 0.005, 0.55]);
    }
    {
        let _reopened = MmapTelemetryBus::new(fixture.path()).unwrap();
        let mut reader = MmapTelemetryReader::new(fixture.path());
        let frames = reader.read_latest_frames().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].payload[3], -0.002);
        assert!(reader.read_latest_frames().unwrap().is_empty());
    }
}

#[test]
fn all_short_lengths_are_rejected_before_header_access() {
    // Added only AFTER the pre-map guard: do not run invalid dereferences as tests.
    for len in [0, 1, 7, 63, 64, 65, FILE_SIZE - 1] {
        let fixture = Fixture::new();
        drop(fixture.file(len));
        let mut reader = MmapTelemetryReader::new(fixture.path());
        let err = reader
            .read_latest_frames()
            .err()
            .expect("short map rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "len={len}");
    }
}

#[test]
fn incomplete_file_can_be_retried_after_explicit_external_recovery() {
    let fixture = Fixture::new();
    let f = fixture.file(64);
    let mut reader = MmapTelemetryReader::new(fixture.path());
    assert!(reader.read_latest_frames().is_err());
    f.set_len(FILE_SIZE).unwrap(); // explicit recovery by this fixture owner
    assert!(reader.read_latest_frames().unwrap().is_empty());
}
