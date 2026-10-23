//! QRA-MEM-01: a requested prefix bounds the tick allocation before loading.
//! Tiny synthetic files only; no tape, models, training or trading process.

use backtest_engine::tick_replayer::{
    load_binary_ticks, load_binary_ticks_prefix, BinTick, TICK_MAGIC_REAL, TICK_MAGIC_SYNTH,
};
use quantum_arena::TickEvent;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct TapeFixture {
    path: PathBuf,
}

impl TapeFixture {
    fn new(header: Option<&[u8; 8]>, records: &[BinTick], tail: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "qra-tick-prefix-{}-{}.bin",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed),
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("create synthetic tape");
        if let Some(header) = header {
            file.write_all(header).unwrap();
        }
        for record in records {
            // Match the complete reader's existing native-endian repr(C) format.
            file.write_all(&record.timestamp.to_ne_bytes()).unwrap();
            for value in [
                record.bid_price,
                record.ask_price,
                record.bid_qty,
                record.ask_qty,
            ] {
                file.write_all(&value.to_ne_bytes()).unwrap();
            }
        }
        file.write_all(tail).unwrap();
        drop(file);
        Self { path }
    }
}

impl Drop for TapeFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn records() -> Vec<BinTick> {
    (0..32)
        .map(|index| BinTick {
            timestamp: 1_700_000_000_000 + index,
            bid_price: 100.0 + index as f64,
            ask_price: 100.5 + index as f64,
            bid_qty: 2.0 + index as f64,
            ask_qty: 3.0 + index as f64,
        })
        .collect()
}

fn event_bits(event: &TickEvent) -> (usize, u64, [u64; 4]) {
    (
        event.coin_id,
        event.timestamp,
        [
            event.bid_price.to_bits(),
            event.ask_price.to_bits(),
            event.bid_qty.to_bits(),
            event.ask_qty.to_bits(),
        ],
    )
}

#[test]
fn prefix_matches_complete_events_and_bounds_capacity_for_all_formats() {
    let source = records();
    for header in [Some(TICK_MAGIC_REAL), Some(TICK_MAGIC_SYNTH), None] {
        let tape = TapeFixture::new(header, &source, &[]);
        let full = load_binary_ticks(&tape.path, 7).expect("complete loader preserved");
        assert_eq!(full.len(), source.len());
        for limit in [0, 1, 2, source.len(), source.len() + 5, usize::MAX] {
            let prefix = load_binary_ticks_prefix(&tape.path, 7, limit).unwrap();
            let expected = &full[..full.len().min(limit)];
            assert_eq!(
                prefix.iter().map(event_bits).collect::<Vec<_>>(),
                expected.iter().map(event_bits).collect::<Vec<_>>(),
            );
            if limit == 0 {
                assert_eq!(prefix.capacity(), 0);
            } else if limit <= 2 {
                // Do not require capacity == len: allocators may round up.
                // Loading all 32 records and then truncating must still fail.
                assert!(
                    prefix.capacity() < full.len(),
                    "a small prefix must not retain the full tape allocation",
                );
            }
        }
    }
}

#[test]
fn prefix_preserves_complete_loader_numeric_sanitization() {
    let source = [
        BinTick {
            timestamp: 11,
            bid_price: f64::NAN,
            ask_price: f64::INFINITY,
            bid_qty: -1.0,
            ask_qty: f64::NEG_INFINITY,
        },
        BinTick {
            timestamp: 12,
            bid_price: -1.0,
            ask_price: 0.0,
            bid_qty: -0.0,
            ask_qty: f64::NAN,
        },
        records()[0],
    ];
    let tape = TapeFixture::new(Some(TICK_MAGIC_REAL), &source, &[]);
    let full = load_binary_ticks(&tape.path, 9).unwrap();
    let prefix = load_binary_ticks_prefix(&tape.path, 9, 2).unwrap();
    assert_eq!(prefix[0].timestamp, 11);
    assert_eq!(prefix[1].timestamp, 12);
    assert_eq!(event_bits(&prefix[0]), (9, 11, [0; 4]));
    assert_eq!(
        event_bits(&prefix[1]),
        (9, 12, [0, 0, (-0.0f64).to_bits(), 0])
    );
    assert_eq!(
        prefix.iter().map(event_bits).collect::<Vec<_>>(),
        full[..2].iter().map(event_bits).collect::<Vec<_>>(),
    );
}

#[test]
fn prefix_validates_the_entire_alignment_even_for_limit_zero() {
    for header in [Some(TICK_MAGIC_REAL), Some(TICK_MAGIC_SYNTH), None] {
        let tape = TapeFixture::new(header, &records(), &[0xff]);
        for limit in [0, 1, usize::MAX] {
            let error = load_binary_ticks_prefix(&tape.path, 0, limit).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        }
        assert_eq!(
            load_binary_ticks(&tape.path, 0).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData,
        );
    }
}

#[test]
fn header_only_valid_tape_has_an_empty_prefix() {
    for header in [TICK_MAGIC_REAL, TICK_MAGIC_SYNTH] {
        let tape = TapeFixture::new(Some(header), &[], &[]);
        assert!(load_binary_ticks(&tape.path, 0).unwrap().is_empty());
        assert!(load_binary_ticks_prefix(&tape.path, 0, 0)
            .unwrap()
            .is_empty());
        assert!(load_binary_ticks_prefix(&tape.path, 0, 10)
            .unwrap()
            .is_empty());
    }
}

#[test]
fn zero_prefix_still_reports_file_access_errors() {
    let tape = TapeFixture::new(Some(TICK_MAGIC_REAL), &records(), &[]);
    let path = tape.path.clone();
    drop(tape);
    let error = load_binary_ticks_prefix(&path, 0, 0).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    // The fixture was removed; no input is fabricated for the zero-prefix case.
    assert!(File::open(path).is_err());
}
