//! Fast witnesses of T-1 measurement limits, not a replacement for the oracle.
#[path = "support/t1_measurement.rs"]
mod measurement;
use backtest_engine::STATS_LEN;
use measurement::*;
use quantum_arena::genome::SuperGenotype;

fn stats(x: f64) -> [f64; STATS_LEN] {
    let mut out = [0.0; STATS_LEN];
    out[6] = x;
    out
}

#[test]
fn go_diagnostic_labels_match_actual_replay_layout() {
    let values = [0.25, 8.0, 975.0, 0.05, -0.4, -10.0, -25.0, 0.50];
    assert_eq!(
        diagnostic_stats_line(&values),
        "[T1-DIAG] stats: trades=8 pnl=-25 wr=0.25 capital=975"
    );
}

#[test]
fn go_witness_legacy_fixture_noise_is_negative_not_centered() {
    let (closes, highs, lows, volumes) = serie(3_000);
    let mut previous = 60_000.0;
    let mut noise = Vec::new();
    for (i, &price) in closes.iter().enumerate() {
        let residual =
            (price / previous - 1.0 - (i as f64 / 180.0).sin() * 0.0030 - 0.00004) / 0.0040;
        noise.push(residual);
        previous = price;
        assert!(lows[i] <= price && highs[i] >= price && volumes[i] > 0.0);
    }
    // This asserts a known OPEN bias, not a property desired for a neutral fixture.
    assert!(noise.iter().all(|&u| (-0.5..0.0).contains(&u)));
    println!(
        "GO noise: n={} positive={} mean={} final_price={}",
        noise.len(),
        noise.iter().filter(|&&u| u > 0.0).count(),
        noise.iter().sum::<f64>() / noise.len() as f64,
        closes.last().unwrap()
    );
}

#[test]
fn go_one_endpoint_does_not_prove_inertia() {
    let f = |x: f64| stats(x * (1.0 - x));
    let endpoint = furthest_endpoint(0.0, 0.0, 1.0);
    assert!(!difiere(&f(0.0), &f(endpoint)));
    assert!(difiere(&f(0.0), &f(0.5)));
}

#[test]
fn go_coordinate_probes_do_not_detect_all_interactions() {
    let f = |a: f64, b: f64| stats(a * b);
    assert!(!difiere(&f(0.0, 0.0), &f(1.0, 0.0)));
    assert!(!difiere(&f(0.0, 0.0), &f(0.0, 1.0)));
    assert!(difiere(&f(0.0, 0.0), &f(1.0, 1.0)));
}

#[test]
fn go_witness_legacy_nonfinite_comparison_is_not_validity_evidence() {
    assert!(!difiere(&stats(f64::NAN), &stats(f64::INFINITY)));
    assert!(difiere(&stats(0.0), &stats(f64::NAN)));
}

#[test]
fn go_trace_realized_coordinates_after_genome_projection() {
    let base = SuperGenotype::new_baseline(0.0002, 0.0005).to_vector();
    let canonical = SuperGenotype::from_vector(&base).to_vector();
    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();
    println!(
        "GO baseline roundtrip changed: {:?}",
        changed_slots(&base, &canonical)
    );
    let mut erased = Vec::new();
    let mut coupled = Vec::new();
    for g in 0..base.len() {
        let mut request = base.clone();
        request[g] = furthest_endpoint(base[g], lo[g], hi[g]);
        let realized = SuperGenotype::from_vector(&request).to_vector();
        assert_eq!(realized.len(), SuperGenotype::DIMENSION);
        assert!(realized.iter().all(|x| x.is_finite()));
        let changed = changed_slots(&canonical, &realized);
        if changed.is_empty() {
            erased.push(g);
        }
        if changed.iter().any(|&slot| slot != g) {
            coupled.push((g, changed));
        }
    }
    // Characterization of the CURRENT projection, not global genetic inertness.
    assert!(!erased.is_empty());
    assert!(!coupled.is_empty());
    println!("GO erased requests: {erased:?}");
    println!("GO coupled requests: {coupled:?}");
}

#[test]
fn go_fixture_is_repeatable_without_reading_model_files() {
    assert_eq!(serie(3_000), serie(3_000));
}

#[test]
fn go_extraction_preserves_frozen_legacy_fixture_bit_for_bit() {
    // Independent frozen equations from main76de9935; no new distribution.
    let mut seed = 0x5DEECE66Du64;
    let mut p = 60_000.0f64;
    let (mut c, mut h, mut l, mut v) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..3_000 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
        let ciclo = (i as f64 / 180.0).sin() * 0.0030;
        p *= 1.0 + ciclo + u * 0.0040 + 0.00004;
        c.push(p);
        h.push(p * (1.0 + 0.0018 + u.abs() * 0.0012));
        l.push(p * (1.0 - 0.0018 - u.abs() * 0.0012));
        v.push(900.0 + u.abs() * 2_000.0);
    }
    let actual = serie(3_000);
    for (old, new) in [&c, &h, &l, &v]
        .into_iter()
        .zip([&actual.0, &actual.1, &actual.2, &actual.3])
    {
        assert!(old.iter().zip(new).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    assert_eq!(serie(0), (vec![], vec![], vec![], vec![]));
}

#[test]
fn go_extraction_preserves_all_legacy_perturbation_targets() {
    let base = SuperGenotype::new_baseline(0.0002, 0.0005).to_vector();
    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();
    for g in 0..base.len() {
        let dist_lo = (base[g] - lo[g]).abs();
        let dist_hi = (hi[g] - base[g]).abs();
        let old = if dist_hi >= dist_lo { hi[g] } else { lo[g] };
        assert_eq!(
            old.to_bits(),
            furthest_endpoint(base[g], lo[g], hi[g]).to_bits()
        );
    }
}
