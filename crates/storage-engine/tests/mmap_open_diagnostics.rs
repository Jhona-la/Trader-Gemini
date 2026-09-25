//! Passing diagnostics describe OPEN contracts, not repaired behavior.
//! Only a uniquely owned temporary file is mapped; no global telemetry bus.
use storage_engine::mmap_bus::{MmapTelemetryReader, TelemetryFrame};

#[test]
fn open_reader_permanently_skips_a_reserved_frame_committed_later() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("tg-xxix-mmap-{}-{stamp}.dat", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    file.set_len((64 + 1_000_000 * std::mem::size_of::<TelemetryFrame>()) as u64)
        .unwrap();
    {
        // Single-threaded phases: this reproduces the publish protocol without
        // concurrent raw-pointer accesses or intentionally triggering a data race.
        let mut writer = unsafe { memmap2::MmapOptions::new().map_mut(&file).unwrap() };
        writer[0..std::mem::size_of::<usize>()].copy_from_slice(&1_usize.to_ne_bytes());
        writer[64..72].copy_from_slice(&100_u64.to_ne_bytes());
        writer[76..80].copy_from_slice(&1_u32.to_ne_bytes()); // reserved/in progress
        let mut reader = MmapTelemetryReader::new(&path);
        assert!(reader.read_latest_frames().unwrap().is_empty());
        writer[80..88].copy_from_slice(&0.125_f64.to_ne_bytes());
        writer[76..80].copy_from_slice(&2_u32.to_ne_bytes()); // same reservation committed
        assert!(
            reader.read_latest_frames().unwrap().is_empty(),
            "known defect: cursor already consumed the in-progress slot"
        );
        // A fresh reader sees the now valid frame, proving it was actually committed.
        let mut fresh = MmapTelemetryReader::new(&path);
        let frames = fresh.read_latest_frames().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].payload[0], 0.125);
    }
    drop(file);
    let exact = path.canonicalize().unwrap();
    assert!(exact.starts_with(std::env::temp_dir().canonicalize().unwrap()));
    assert!(exact
        .file_name()
        .unwrap()
        .to_string_lossy()
        .starts_with("tg-xxix-mmap-"));
    std::fs::remove_file(exact).unwrap();
}
