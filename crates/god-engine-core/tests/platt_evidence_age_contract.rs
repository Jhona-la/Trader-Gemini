//! LH-12: edad individual de los cierres, con oráculo posterior de una
//! puntuación constante. No certifica calibración poblacional ni rentabilidad.
use god_engine_core::calibration::{
    PlattCalibrator, PRIOR_PSEUDO_OBSERVATIONS, SEMIVIDA_EVIDENCIA_MS,
};

fn posterior(score: f64, weighted_wins: f64, weighted_count: f64) -> f64 {
    (PRIOR_PSEUDO_OBSERVATIONS * score + weighted_wins)
        / (PRIOR_PSEUDO_OBSERVATIONS + weighted_count)
}

fn assert_posterior(c: &PlattCalibrator, wins: f64, count: f64) {
    // El ridge perturba ligeramente la solución Beta identificada.
    let expected = posterior(0.8, wins, count);
    assert!(
        (c.calibrate(0.8) - expected).abs() < 1e-3,
        "p={}, posterior ponderado={expected}",
        c.calibrate(0.8)
    );
}

#[test]
fn lh12_new_close_does_not_rejuvenate_old_evidence() {
    let mut c = PlattCalibrator::new();
    for _ in 0..40 {
        c.update_at(0.8, true, 0);
    }
    let t = SEMIVIDA_EVIDENCIA_MS as u64;
    for _ in 0..12 {
        c.update_at(0.8, false, t);
    }
    assert_eq!(c.observations(), 52);
    assert_posterior(&c, 20.0, 32.0);
}

#[test]
fn lh12_evidence_ages_while_closes_keep_arriving() {
    let h = SEMIVIDA_EVIDENCIA_MS as u64;
    let mut c = PlattCalibrator::new();
    let mut times = Vec::new();
    for i in 0..80 {
        let t = i * h / 80;
        c.update_at(0.8, true, t);
        times.push((t, true));
    }
    for i in 1..=12 {
        c.update_at(0.8, false, i * h);
        times.push((i * h, false));
    }
    let end = 12 * h;
    let (mut wins, mut count) = (0.0, 0.0);
    for (t, won) in times {
        let w = 2.0f64.powf(-((end - t) as f64) / SEMIVIDA_EVIDENCIA_MS);
        count += w;
        if won {
            wins += w;
        }
    }
    assert_posterior(&c, wins, count);
}

#[test]
fn lh12_late_close_keeps_its_age_and_does_not_rewind_clock() {
    let h = SEMIVIDA_EVIDENCIA_MS as u64;
    let mut c = PlattCalibrator::new();
    for _ in 0..64 {
        c.update_at(0.8, true, 8 * h);
    }
    c.update_at(0.8, false, 0);
    assert_posterior(&c, 64.0, 64.0 + 1.0 / 256.0);
    assert_eq!(c.calibrate_at(0.8, 8 * h), c.calibrate(0.8));
}

#[test]
fn lh12_same_timestamp_preserves_untimed_fit() {
    let (mut timed, mut legacy) = (PlattCalibrator::new(), PlattCalibrator::new());
    for i in 0..90 {
        let s = if i % 2 == 0 { 0.8 } else { 0.3 };
        timed.update_at(s, i % 3 == 0, 123);
        legacy.update(s, i % 3 == 0);
    }
    assert_eq!(timed.coefficients(), legacy.coefficients());
}

#[test]
fn lh12_age_is_invariant_to_clock_origin() {
    let h = SEMIVIDA_EVIDENCIA_MS as u64;
    let (mut a, mut b) = (PlattCalibrator::new(), PlattCalibrator::new());
    for i in 0..25 {
        a.update_at(0.8, i < 10, i * h / 4);
        b.update_at(0.8, i < 10, 1_800_000_000_000 + i * h / 4);
    }
    assert_eq!(a.coefficients(), b.coefficients());
}

#[test]
fn lh12_mixed_legacy_updates_get_a_clock_anchor_once() {
    let h = SEMIVIDA_EVIDENCIA_MS as u64;
    let mut c = PlattCalibrator::new();
    for _ in 0..10 {
        c.update(0.8, true);
    }
    c.update_at(0.8, false, 10 * h);
    c.update(0.8, false);
    c.update_at(0.8, false, 11 * h);
    assert_posterior(&c, 5.0, 5.0 + 0.5 + 0.5 + 1.0);
}

#[test]
fn lh12_invalid_score_does_not_age_or_change_evidence() {
    let mut c = PlattCalibrator::new();
    c.update_at(0.8, false, 50);
    let before = c.clone();
    for s in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        c.update_at(s, true, u64::MAX);
    }
    assert_eq!(c.observations(), before.observations());
    assert_eq!(c.coefficients(), before.coefficients());
    assert_eq!(c.calibrate_at(0.8, 100), before.calibrate_at(0.8, 100));
}

#[test]
fn lh12_two_score_bins_recover_their_different_evidence_masses() {
    // Dos bins identifican pendiente e intercepto. El posterior binomial
    // por bin es independiente del optimizador logístico; el ridge pequeño
    // perturba su solución, y la restricción a>=0 está inactiva aquí.
    let mut c = PlattCalibrator::new();
    for i in 0..64 {
        c.update_at(0.2, i < 48, 0);
    }
    for i in 0..64 {
        c.update_at(0.8, i < 48, 4 * SEMIVIDA_EVIDENCIA_MS as u64);
    }
    let half_prior = PRIOR_PSEUDO_OBSERVATIONS / 2.0;
    let low = (3.0 + half_prior * 0.2) / (4.0 + half_prior);
    let high = (48.0 + half_prior * 0.8) / (64.0 + half_prior);
    assert!(low < high);
    assert!((c.calibrate(0.2) - low).abs() < 1e-3);
    assert!((c.calibrate(0.8) - high).abs() < 1e-3);
}
