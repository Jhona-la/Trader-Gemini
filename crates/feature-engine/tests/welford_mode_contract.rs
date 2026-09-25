use feature_engine::WelfordOnline;

fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0),
        "{a} != {b}"
    );
}

#[test]
fn invalid_alpha_does_not_change_state() {
    for alpha in [2.0, f64::INFINITY, f64::NAN, 0.0, -0.1] {
        let mut s = WelfordOnline::new();
        s.update(3.0);
        s.update(7.0);
        let before = s;
        s.update_decay(100.0, alpha);
        assert_eq!(
            (s.count, s.mean, s.m2, s.is_decay),
            (before.count, before.mean, before.m2, before.is_decay)
        );
    }
}

#[test]
fn first_decayed_sample_has_no_phantom_zero_observation() {
    let mut s = WelfordOnline::new();
    s.update_decay(10.0, 0.2);
    assert_eq!(s.mean(), 10.0);
    assert_eq!(s.variance(), 0.0);
    assert_eq!(s.count, 1.0);
}

#[test]
fn returning_to_default_update_preserves_variance_representation() {
    let mut s = WelfordOnline::new();
    s.update(2.0);
    s.update(4.0);
    s.update_decay(6.0, 0.5);
    let old_mean = s.mean();
    let old_variance = s.variance();
    let alpha = 2.0 / 2001.0;
    let expected_mean = old_mean + alpha * (8.0 - old_mean);
    let expected_variance =
        (1.0 - alpha) * old_variance + alpha * (8.0 - old_mean) * (8.0 - expected_mean);
    s.update(8.0);
    assert!(s.is_decay);
    close(s.mean(), expected_mean);
    close(s.variance(), expected_variance);
}

#[test]
fn cumulative_matches_two_pass_reference() {
    let values = [10.0, 20.0, 30.0, 40.0, 50.0];
    let mut s = WelfordOnline::new();
    for x in values {
        s.update(x);
    }
    assert_eq!(s.mean(), 30.0);
    assert_eq!(s.variance(), 250.0);
    assert!(!s.is_decay);
}

#[test]
fn decayed_statistics_match_explicit_empirical_weights() {
    let values = [10.0, 4.0, 8.0, -2.0, 6.0];
    let gains = [0.2, 0.5, 0.1, 1.0, 0.25];
    let mut weights: Vec<f64> = Vec::new();
    let mut s = WelfordOnline::new();
    for i in 0..values.len() {
        let a = gains[i];
        for w in &mut weights {
            *w *= 1.0 - a;
        }
        weights.push(if i == 0 { 1.0 } else { a });
        s.update_decay(values[i], a);
        let mean: f64 = weights.iter().zip(&values).map(|(w, x)| w * x).sum();
        let variance: f64 = weights
            .iter()
            .zip(&values)
            .map(|(w, x)| w * (x - mean).powi(2))
            .sum();
        close(s.mean(), mean);
        close(s.variance(), variance);
    }
}

#[test]
fn default_transition_keeps_legacy_policy() {
    let mut s = WelfordOnline::new();
    for i in 0..2000 {
        s.update((i % 7) as f64);
    }
    assert_eq!(s.count, 2000.0);
    let before = s;
    let alpha = 2.0 / 2001.0;
    let mean = before.mean + alpha * (9.0 - before.mean);
    let var = (1.0 - alpha) * before.variance() + alpha * (9.0 - before.mean) * (9.0 - mean);
    s.update(9.0);
    close(s.mean, mean);
    close(s.variance(), var);
    assert!(s.is_decay);
    assert_eq!(s.count, 2000.0);
}

#[test]
fn nonfinite_observation_preserves_both_modes() {
    for decayed in [false, true] {
        let mut s = WelfordOnline::new();
        s.update(2.0);
        if decayed {
            s.update_decay(4.0, 0.5);
        }
        let before = s;
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            s.update(x);
            s.update_decay(x, 0.2);
        }
        assert_eq!(
            (s.count, s.mean, s.m2, s.is_decay),
            (before.count, before.mean, before.m2, before.is_decay)
        );
    }
}
