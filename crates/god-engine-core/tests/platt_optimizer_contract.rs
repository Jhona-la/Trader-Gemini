//! Numerical contracts of the actual MAP objective, not evidence of OOS calibration.
use god_engine_core::calibration::{PlattCalibrator, PRIOR_PSEUDO_OBSERVATIONS, WINDOW};

fn objective(obs: &[(f64, bool)], a: f64, b: f64) -> f64 {
    let q = PRIOR_PSEUDO_OBSERVATIONS / obs.len() as f64;
    let mut loss = 0.5e-4 * ((a - 1.0).powi(2) + b * b);
    for &(s, won) in obs {
        let clipped = s.clamp(1e-4, 1.0 - 1e-4);
        let z = a * (clipped / (1.0 - clipped)).ln() + b;
        let target = (f64::from(won) + q * s) / (1.0 + q);
        loss += (1.0 + q) * (z.max(0.0) - target * z + (-z.abs()).exp().ln_1p());
    }
    loss
}

fn gradient(obs: &[(f64, bool)], a: f64, b: f64) -> (f64, f64) {
    let q = PRIOR_PSEUDO_OBSERVATIONS / obs.len() as f64;
    let (mut ga, mut gb) = (1e-4 * (a - 1.0), 1e-4 * b);
    for &(s, won) in obs {
        let c = s.clamp(1e-4, 1.0 - 1e-4);
        let x = (c / (1.0 - c)).ln();
        let p = 1.0 / (1.0 + (-(a * x + b)).exp());
        let r = p - f64::from(won) + q * (p - s);
        ga += r * x;
        gb += r;
    }
    (ga, gb)
}

#[test]
fn repeated_losses_match_the_single_score_posterior() {
    let mut c = PlattCalibrator::new();
    for _ in 0..40 {
        c.update(0.8, false);
    }
    let expected = PRIOR_PSEUDO_OBSERVATIONS * 0.8 / (40.0 + PRIOR_PSEUDO_OBSERVATIONS);
    assert!(
        (c.calibrate(0.8) - expected).abs() < 1e-3,
        "p={}, posterior={expected}, coefficients={:?}",
        c.calibrate(0.8),
        c.coefficients()
    );
}

#[test]
fn inverse_ranking_hits_the_constrained_optimum_not_zero_probability() {
    let obs: Vec<_> = (0..40)
        .map(|i| {
            if i % 2 == 0 {
                (0.9, false)
            } else {
                (0.6, true)
            }
        })
        .collect();
    let mut c = PlattCalibrator::new();
    for &(s, y) in &obs {
        c.update(s, y);
    }
    let (a, b) = c.coefficients();
    let expected = (20.0 + PRIOR_PSEUDO_OBSERVATIONS * 0.75) / (40.0 + PRIOR_PSEUDO_OBSERVATIONS);
    assert!(a < 1e-8, "slope should bind: {a}");
    assert!(
        (c.calibrate(0.75) - expected).abs() < 1e-3,
        "p={}, b={b}",
        c.calibrate(0.75)
    );
    let (ga, gb) = gradient(&obs, a, b);
    assert!(ga >= -1e-6 && gb.abs() < 1e-6, "KKT residual {ga}, {gb}");
}

#[test]
fn fit_never_has_higher_map_loss_than_identity_on_extreme_scores() {
    let mut c = PlattCalibrator::new();
    let mut obs = Vec::new();
    for i in 0..80 {
        let item = if i % 3 == 0 {
            (0.9999, false)
        } else {
            (0.55, true)
        };
        obs.push(item);
        c.update(item.0, item.1);
        let (a, b) = c.coefficients();
        assert!(a.is_finite() && a >= 0.0 && b.is_finite());
        assert!(
            objective(&obs, a, b) <= objective(&obs, 1.0, 0.0) + 1e-8,
            "prefix {}: fit={}, identity={}",
            obs.len(),
            objective(&obs, a, b),
            objective(&obs, 1.0, 0.0)
        );
    }
}

#[test]
fn fitting_the_same_window_is_insensitive_to_arrival_permutation() {
    let mut first = PlattCalibrator::new();
    let mut second = PlattCalibrator::new();
    for i in 0..80 {
        first.update(if i % 2 == 0 { 0.9 } else { 0.6 }, i % 2 != 0);
    }
    for i in 0..80 {
        second.update(if i < 40 { 0.6 } else { 0.9 }, i < 40);
    }
    for s in [0.1, 0.6, 0.9] {
        assert!(
            (first.calibrate(s) - second.calibrate(s)).abs() < 1e-6,
            "{s}: {:?} vs {:?}",
            first.coefficients(),
            second.coefficients()
        );
    }
}

#[test]
fn invalid_training_scores_do_not_change_data_fit_or_clock() {
    let mut c = PlattCalibrator::new();
    c.update_at(0.8, false, 1_000);
    let before = c.clone();
    for s in [-0.1, 1.1, f64::MAX, f64::NAN, f64::INFINITY] {
        c.update_at(s, true, 50_000_000);
        c.update(s, true);
    }
    assert_eq!(c.observations(), before.observations());
    assert_eq!(c.coefficients(), before.coefficients());
    assert_eq!(
        c.calibrate_at(0.8, 60_000_000),
        before.calibrate_at(0.8, 60_000_000)
    );
}

#[test]
fn endpoints_and_single_class_windows_remain_finite() {
    for s in [0.0, 1e-4, 0.5, 0.9999, 1.0] {
        for won in [false, true] {
            let mut c = PlattCalibrator::new();
            let obs = vec![(s, won); 40];
            for &(score, y) in &obs {
                c.update(score, y);
            }
            let (a, b) = c.coefficients();
            assert!(a.is_finite() && a >= 0.0 && b.is_finite());
            let (ga, gb) = gradient(&obs, a, b);
            let projected_ga = if a == 0.0 { ga.min(0.0) } else { ga };
            assert!(
                projected_ga.abs().max(gb.abs()) < 1e-5,
                "s={s}, won={won}, KKT={ga},{gb}, a={a}, b={b}"
            );
        }
    }
}

#[test]
fn rolling_window_fits_its_contents_not_the_evicted_history() {
    let mut c = PlattCalibrator::new();
    for _ in 0..WINDOW {
        c.update(0.8, false);
    }
    for _ in 0..WINDOW {
        c.update(0.8, true);
    }
    assert_eq!(c.observations(), WINDOW);
    let expected = (WINDOW as f64 + PRIOR_PSEUDO_OBSERVATIONS * 0.8)
        / (WINDOW as f64 + PRIOR_PSEUDO_OBSERVATIONS);
    assert!(
        (c.calibrate(0.8) - expected).abs() < 1e-3,
        "p={}",
        c.calibrate(0.8)
    );
}

#[test]
fn varied_windows_satisfy_first_order_optimality() {
    for seed in 1..=24u64 {
        let mut state = seed;
        let mut obs = Vec::new();
        let mut c = PlattCalibrator::new();
        for _ in 0..64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let s = [0.0, 0.01, 0.2, 0.55, 0.8, 0.99, 1.0][((state >> 32) % 7) as usize];
            let won = state >> 63 != 0;
            obs.push((s, won));
            c.update(s, won);
        }
        let (a, b) = c.coefficients();
        let (ga, gb) = gradient(&obs, a, b);
        let projected_ga = if a == 0.0 { ga.min(0.0) } else { ga };
        assert!(
            projected_ga.abs().max(gb.abs()) < 1e-5,
            "seed={seed}, KKT={ga},{gb}, coefficients={a},{b}"
        );
        assert!(objective(&obs, a, b) <= objective(&obs, 1.0, 0.0) + 1e-8);
    }
}
