//! Contract test for F1-073 (Ola Ω177):
//!
//! F1-073: Atomic snapshots and Seqlock for continuous horizon curves in `QuantumConfig`
//! and `apply_to_arena`.
//! Eradicates chimera curves `(a_new, b_old)` during hot-swaps and guarantees that TP and SL
//! are read from the same genome snapshot, preserving mathematical and risk invariants.

use quantum_arena::genome::SuperGenotype;
use quantum_arena::state::GlobalArena;
use quantum_arena::temporal_spectrum::HorizonCurve;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

#[test]
fn test_f1_073_seqlock_eliminates_chimeras_during_rapid_hot_swaps() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    let mut genome_a = SuperGenotype::new_baseline(0.0002, 0.0005);
    genome_a.tp_horizon_curve = HorizonCurve {
        a: -3.5,
        b: 0.04,
    };
    genome_a.sl_horizon_curve = HorizonCurve {
        a: -4.5,
        b: 0.03,
    };
    genome_a.enforce_curve_rr();
    genome_a.sync_continuous_curves();

    let mut genome_b = SuperGenotype::new_baseline(0.0002, 0.0005);
    genome_b.tp_horizon_curve = HorizonCurve {
        a: -2.8,
        b: 0.08,
    };
    genome_b.sl_horizon_curve = HorizonCurve {
        a: -3.8,
        b: 0.06,
    };
    genome_b.enforce_curve_rr();
    genome_b.sync_continuous_curves();

    let tp_a = genome_a.tp_horizon_curve;
    let sl_a = genome_a.sl_horizon_curve;
    let tp_b = genome_b.tp_horizon_curve;
    let sl_b = genome_b.sl_horizon_curve;

    // Apply baseline A first
    genome_a.apply_to_arena(&arena);

    let stop_signal = Arc::new(AtomicBool::new(false));
    let stop_signal_clone = stop_signal.clone();

    // Reader thread: checks that NO chimera curve (a_new, b_old) is ever observed
    // and that (tp, sl) pairs are always from the same genome snapshot.
    let arena_clone = arena.clone();

    let reader_handle = thread::spawn(move || {
        let arena_ref = &arena_clone;
        let mut read_count = 0u64;

        while !stop_signal_clone.load(Ordering::Relaxed) {
            // 1. Read TP curve
            let tp = arena_ref.config.tp_curve();
            let is_tp_a = (tp.a - tp_a.a).abs() < 1e-9 && (tp.b - tp_a.b).abs() < 1e-9;
            let is_tp_b = (tp.a - tp_b.a).abs() < 1e-9 && (tp.b - tp_b.b).abs() < 1e-9;
            assert!(
                is_tp_a || is_tp_b,
                "Observed chimera TP curve: a={}, b={} (expected A=({},{}), B=({},{}))",
                tp.a, tp.b, tp_a.a, tp_a.b, tp_b.a, tp_b.b
            );

            // 2. Read SL curve
            let sl = arena_ref.config.sl_curve();
            let is_sl_a = (sl.a - sl_a.a).abs() < 1e-9 && (sl.b - sl_a.b).abs() < 1e-9;
            let is_sl_b = (sl.a - sl_b.a).abs() < 1e-9 && (sl.b - sl_b.b).abs() < 1e-9;
            assert!(
                is_sl_a || is_sl_b,
                "Observed chimera SL curve: a={}, b={} (expected A=({},{}), B=({},{}))",
                sl.a, sl.b, sl_a.a, sl_a.b, sl_b.a, sl_b.b
            );

            // 3. Read TP and SL simultaneously: must be strictly co-originating (both A or both B)
            let (tp_pair, sl_pair) = arena_ref.config.tp_sl_curves();
            let pair_is_a = (tp_pair.a - tp_a.a).abs() < 1e-9 && (sl_pair.a - sl_a.a).abs() < 1e-9;
            let pair_is_b = (tp_pair.a - tp_b.a).abs() < 1e-9 && (sl_pair.a - sl_b.a).abs() < 1e-9;
            assert!(
                pair_is_a || pair_is_b,
                "Cross-genome chimera observed! TP from one and SL from another: tp=({},{}), sl=({},{})",
                tp_pair.a, tp_pair.b, sl_pair.a, sl_pair.b
            );

            read_count += 1;
        }

        read_count
    });

    // Writer thread: rapidly alternates between genome A and genome B 10,000 times
    let iterations = 10_000;
    for i in 0..iterations {
        if i % 2 == 0 {
            genome_b.apply_to_arena(&arena);
        } else {
            genome_a.apply_to_arena(&arena);
        }
    }

    stop_signal.store(true, Ordering::Release);
    let total_reads = reader_handle.join().expect("reader thread did not panic");
    assert!(total_reads > 100, "Reader must have executed at least 100 concurrent reads, got {total_reads}");
}

#[test]
fn test_f1_073_tp_sl_at_tau_coherence() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let g = SuperGenotype::new_baseline(0.0002, 0.0005);
    g.apply_to_arena(&arena);

    let tau_30s = 30_000.0;
    let tau_12h = 12.0 * 3600.0 * 1000.0;

    let (tp_30s, sl_30s) = arena.config.tp_sl_at_tau(tau_30s);
    assert!(tp_30s > 0.0 && sl_30s > 0.0);
    assert!(tp_30s > sl_30s, "TP must exceed SL at 30s: tp={}, sl={}", tp_30s, sl_30s);

    let (tp_12h, sl_12h) = arena.config.tp_sl_at_tau(tau_12h);
    assert!(tp_12h > 0.0 && sl_12h > 0.0);
    assert!(tp_12h > sl_12h, "TP must exceed SL at 12h: tp={}, sl={}", tp_12h, sl_12h);
}

#[test]
fn test_f1_073_all_continuous_curves_read_consistently() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
    g.scalp_obi_threshold = 0.45;
    g.swing_obi_threshold = 0.35;
    g.scalp_kelly_fraction = 0.22;
    g.swing_kelly_fraction = 0.14;
    g.apply_to_arena(&arena);

    let kelly_curve = arena.config.kelly_curve();
    assert!(kelly_curve.a.is_finite() && kelly_curve.b.is_finite());

    let (mult, act, step) = arena.config.trail_curves();
    assert!(mult.a.is_finite() && act.a.is_finite() && step.a.is_finite());

    let obi_curve = arena.config.obi_curve();
    assert!(obi_curve.a.is_finite() && obi_curve.b.is_finite());

    let kelly_val = arena.config.kelly_at_tau(60_000.0);
    assert!(kelly_val > 0.0 && kelly_val <= 1.0);

    let (t_mult, t_act, t_step, t_max) = arena.config.trail_params_at_tau(60_000.0);
    assert!(t_mult > 0.0 && t_act > 0.0 && t_step > 0.0 && t_max > 0.0);

    let obi_val = arena.config.obi_threshold_at_tau(60_000.0);
    assert!(obi_val >= 0.10 && obi_val <= 0.95);
}
