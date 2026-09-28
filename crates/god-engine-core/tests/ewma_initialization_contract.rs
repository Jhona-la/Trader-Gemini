//! Cross-component contracts for weighted EWMA initialization and input domain.
use god_engine_core::calibration::{ewma_con_peso, momentos_ewma_corregidos};
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering::Relaxed;

fn entropy_moments(observations: usize, sample: impl Fn(usize) -> f64) -> (f64, f64) {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let coin = &arena.coins[0];
    let alpha = 0.0033;
    let mut mean = coin.entropy_mean_ewma.load(Relaxed);
    let mut second = coin.entropy_sq_ewma.load(Relaxed);
    let mut weight = coin.campo_ewma_peso.load(Relaxed);
    for i in 0..observations {
        let x = sample(i);
        let (next_mean, next_weight) = ewma_con_peso(x, mean, weight, alpha);
        let (next_second, _) = ewma_con_peso(x * x, second, weight, alpha);
        mean = next_mean;
        second = next_second;
        weight = next_weight;
    }
    momentos_ewma_corregidos(mean, second, weight, alpha).expect("warm valid moments")
}

#[test]
fn arena_entropy_prior_matches_the_zero_mass_correction() {
    for n in [31, 200, 1000] {
        let (mean, sd) = entropy_moments(n, |_| 0.5);
        assert!((mean - 0.5).abs() < 1e-12, "n={n}: mean={mean}, sd={sd}");
        assert!(sd < 1e-7, "constant stream: n={n}: sd={sd}");
    }
}

#[test]
fn arena_entropy_stays_within_observed_support() {
    let (mean, sd) = entropy_moments(200, |i| if i % 2 == 0 { 0.2 } else { 0.8 });
    assert!((0.2..=0.8).contains(&mean), "mean={mean}");
    assert!((sd - 0.3).abs() < 0.001, "sd={sd}");
}

#[test]
fn corrected_moments_reject_impossible_mass_and_negative_second_moment() {
    for (mean, second, weight) in [(0.1, 0.02, 1.1), (0.0, -1.0, 1.0), (1.0, 0.1, 1.0)] {
        assert!(
            momentos_ewma_corregidos(mean, second, weight, 0.0033).is_none(),
            "mean={mean}, second={second}, weight={weight}"
        );
    }
}

#[test]
fn tiny_alpha_does_not_create_evidence_from_zero_mass() {
    assert!(momentos_ewma_corregidos(0.0, 0.0, 0.0, 1e-20).is_none());
}

#[test]
fn corrected_moments_never_return_nonfinite_mean() {
    assert!(momentos_ewma_corregidos(f64::MAX, f64::MAX, 0.1, 0.001).is_none());
}

#[test]
fn feasible_roundoff_and_signed_means_remain_usable() {
    let (mean, sd) = momentos_ewma_corregidos(-0.25, 0.125, 0.5, 0.0033).unwrap();
    assert_eq!(mean, -0.5);
    assert_eq!(sd, 0.0);
    let (_, sd) = momentos_ewma_corregidos(1.0, 1.0 - f64::EPSILON, 1.0, 0.0033).unwrap();
    assert_eq!(sd, 0.0);
}

#[test]
fn small_alpha_with_real_mass_can_still_warm_up() {
    let (mut mean, mut second, mut mass) = (0.0, 0.0, 0.0);
    for _ in 0..40 {
        let (m, w) = ewma_con_peso(0.5, mean, mass, 1e-20);
        let (s, _) = ewma_con_peso(0.25, second, mass, 1e-20);
        (mean, second, mass) = (m, s, w);
    }
    let (m, sd) = momentos_ewma_corregidos(mean, second, mass, 1e-20).unwrap();
    assert!((m - 0.5).abs() < 1e-12);
    assert!(sd < 1e-7);
}
