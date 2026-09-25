//! Diagnostic characterizations of OPEN legacy defects (FMT-150), not repair tests.
//! A green result documents the weak contract and must not certify safe recovery.
use quantum_arena::state_continuity::{ArenaCheckpoint, PositionSnapshot, StateContinuityEngine};

fn position() -> PositionSnapshot {
    PositionSnapshot {
        coin_id: 1,
        symbol: "ETHUSDT".into(),
        is_scalp: false,
        is_long: true,
        size: 0.5,
        entry_price: 2500.0,
        highest_price: 2550.0,
        lowest_price: 2490.0,
        stop_loss: 2450.0,
        take_profit: 2600.0,
        timestamp_ms: 1000,
        checksum: StateContinuityEngine::compute_state_checksum(1, 0.5, 2500.0),
    }
}

#[test]
fn diagnostic_checksum_does_not_cover_risk_identity_or_time() {
    let mut changed = position();
    changed.symbol = "OTHER".into();
    changed.is_long = false;
    changed.is_scalp = true;
    changed.stop_loss = -1.0;
    changed.take_profit = f64::INFINITY;
    changed.highest_price = 0.0;
    changed.lowest_price = -10.0;
    changed.timestamp_ms = u64::MAX;
    assert!(StateContinuityEngine::validate_snapshot(&changed));
}

#[test]
fn diagnostic_checksum_has_structural_collisions() {
    let original = position();
    let mut changed = original.clone();
    changed.coin_id += 64;
    std::mem::swap(&mut changed.size, &mut changed.entry_price);
    assert_eq!(changed.checksum, original.checksum);
    assert!(StateContinuityEngine::validate_snapshot(&changed));
}

#[test]
fn diagnostic_loader_accepts_unsupported_version_and_unverified_global_state() {
    let checkpoint = ArenaCheckpoint {
        version: u32::MAX,
        timestamp_ms: 0,
        unified_capital: -100.0,
        positions: vec![position()],
        global_checksum: 12345,
    };
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("tg-xiii-checkpoint-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("owned-checkpoint.json");
    StateContinuityEngine::save_checkpoint(&path, &checkpoint).unwrap();
    let loaded = StateContinuityEngine::load_checkpoint(&path).unwrap();
    // Remove only this test's explicitly created file and empty directory.
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&directory).unwrap();
    assert_eq!(loaded.version, u32::MAX);
    assert_eq!(loaded.unified_capital, -100.0);
    assert_eq!(loaded.global_checksum, 12345);
}
