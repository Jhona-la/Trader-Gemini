//! Numerical boundary contract for the public multifractal parameter API.
//! Invalid parameter sets resolve as a whole to defaults before any observation
//! is processed; valid finite weights and documented lower bounds stay usable.

use feature_engine::{smooth_persistence_degree, MultiScaleHurstConfluence, MultifractalParams};

fn channels(output: (f64, f64, f64, f64, f64, f64)) -> [f64; 6] {
    [output.0, output.1, output.2, output.3, output.4, output.5]
}

fn price(tick: usize) -> f64 {
    let t = tick as f64;
    100.0 + 0.02 * t + (t * 0.23).sin() + 0.3 * (t * 0.71).cos()
}

fn custom_params() -> MultifractalParams {
    MultifractalParams {
        w_micro: 0.8,
        w_meso: 0.1,
        w_macro: 0.1,
        tanh_scale: 0.13,
        prior_weight_k: 2.0,
        persistence_threshold_micro: 0.08,
        persistence_threshold_macro: 0.12,
        transition_width: 0.02,
    }
}

#[test]
fn validation_rejects_finite_weights_whose_sum_overflows() {
    let params = MultifractalParams {
        w_micro: f64::MAX,
        w_meso: f64::MAX,
        w_macro: f64::MAX,
        ..MultifractalParams::default()
    };
    assert!(
        params.validate().is_err(),
        "an infinite weight sum is invalid"
    );
}

#[test]
fn invalid_params_use_whole_default_fallback_without_changing_later_state() {
    let base = custom_params();
    let cases = [
        (
            "nan weight",
            MultifractalParams {
                w_micro: f64::NAN,
                ..base
            },
        ),
        (
            "infinite weight",
            MultifractalParams {
                w_meso: f64::INFINITY,
                ..base
            },
        ),
        (
            "negative weight",
            MultifractalParams {
                w_macro: -0.1,
                ..base
            },
        ),
        (
            "zero weight sum",
            MultifractalParams {
                w_micro: 0.0,
                w_meso: 0.0,
                w_macro: 0.0,
                ..base
            },
        ),
        (
            "weight sum below lower bound",
            MultifractalParams {
                w_micro: 1e-7,
                w_meso: 0.0,
                w_macro: 0.0,
                ..base
            },
        ),
        (
            "overflowing weight sum",
            MultifractalParams {
                w_micro: f64::MAX,
                w_meso: f64::MAX,
                w_macro: f64::MAX,
                ..base
            },
        ),
        (
            "nan scale",
            MultifractalParams {
                tanh_scale: f64::NAN,
                ..base
            },
        ),
        (
            "scale below lower bound",
            MultifractalParams {
                tanh_scale: 0.5e-4,
                ..base
            },
        ),
        (
            "nan prior",
            MultifractalParams {
                prior_weight_k: f64::NAN,
                ..base
            },
        ),
        (
            "prior below lower bound",
            MultifractalParams {
                prior_weight_k: 0.05,
                ..base
            },
        ),
        (
            "nan micro threshold",
            MultifractalParams {
                persistence_threshold_micro: f64::NAN,
                ..base
            },
        ),
        (
            "zero macro threshold",
            MultifractalParams {
                persistence_threshold_macro: 0.0,
                ..base
            },
        ),
        (
            "infinite transition width",
            MultifractalParams {
                transition_width: f64::INFINITY,
                ..base
            },
        ),
        (
            "width below lower bound",
            MultifractalParams {
                transition_width: 0.5e-4,
                ..base
            },
        ),
    ];
    let defaults = MultifractalParams::default();

    for (label, invalid) in cases {
        assert!(invalid.validate().is_err(), "case must be invalid: {label}");
        let mut actual = MultiScaleHurstConfluence::new();
        let mut reference = MultiScaleHurstConfluence::new();
        // The second half checks retained state after switching back to valid
        // defaults, beyond merely checking the output of the invalid call.
        for tick in 0..384 {
            let params = if tick < 256 { &invalid } else { &defaults };
            let got = channels(actual.update_continuous_with_params(price(tick), params));
            let expected =
                channels(reference.update_continuous_with_params(price(tick), &defaults));
            for (channel, (got, expected)) in got.into_iter().zip(expected).enumerate() {
                assert!(
                    got.is_finite(),
                    "{label}, tick {tick}, channel {channel}: nonfinite output"
                );
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "{label}, tick {tick}, channel {channel}: whole-set default parity"
                );
            }
        }
    }
}

#[test]
fn large_valid_weights_preserve_the_same_convex_combination() {
    let large = f64::MAX / 8.0;
    let params = MultifractalParams {
        w_micro: large,
        w_meso: large,
        w_macro: large,
        ..MultifractalParams::default()
    };
    let normalized = MultifractalParams {
        w_micro: 1.0,
        w_meso: 1.0,
        w_macro: 1.0,
        ..MultifractalParams::default()
    };
    assert!(
        params.validate().is_ok(),
        "large finite weight sums remain valid"
    );
    let mut actual = MultiScaleHurstConfluence::new();
    let mut reference = MultiScaleHurstConfluence::new();
    let mut observed_nonzero = false;
    for tick in 0..256 {
        let got = channels(actual.update_continuous_with_params(price(tick), &params));
        let expected = channels(reference.update_continuous_with_params(price(tick), &normalized));
        observed_nonzero |= expected[3].abs() > 1e-6;
        for (got, expected) in got.into_iter().zip(expected) {
            assert!(got.is_finite());
            assert!(
                (got - expected).abs() <= 1e-12,
                "normalizing finite large weights changed the result: {got} vs {expected}"
            );
        }
    }
    assert!(
        observed_nonzero,
        "the oracle must exercise a non-neutral combination"
    );
}

#[test]
fn validated_lower_bounds_are_used_at_the_exact_boundary() {
    let params = MultifractalParams {
        w_micro: 1e-6,
        w_meso: 0.0,
        w_macro: 0.0,
        tanh_scale: 1e-4,
        prior_weight_k: 0.1,
        transition_width: 1e-4,
        ..MultifractalParams::default()
    };
    assert!(params.validate().is_ok());
    let mut actual = MultiScaleHurstConfluence::new();
    let mut exercised_scale = false;
    for tick in 0..128 {
        let (h_micro, _, _, score, _, _) =
            actual.update_continuous_with_params(price(tick), &params);
        let expected = ((h_micro - 0.5) / params.tanh_scale).tanh();
        exercised_scale |= expected.abs() > 1e-6;
        assert!(
            (score - expected).abs() <= 1e-12,
            "accepted scale lower bound must not be silently replaced"
        );
    }
    assert!(exercised_scale);

    let degree = smooth_persistence_degree(0.60005, 0.1, params.transition_width);
    assert!(
        (degree - 0.84375).abs() < 1e-10,
        "accepted width lower bound must use its narrow transition, got {degree}"
    );
}
