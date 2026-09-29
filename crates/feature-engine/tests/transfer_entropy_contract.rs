use feature_engine::transfer_entropy::{te_binaria_kt, transfer_entropy_eventos};
use feature_engine::transfer_entropy::{
    transfer_entropy_observada, ObservationWindow, TransferEntropyError,
};

// Independent oracle: four Shannon entropies from ONE smoothed joint law.
// Production uses a conditional log ratio; this detects inconsistent priors.
fn reference_cmi(x: &[u8], y: &[u8]) -> f64 {
    let mut joint = [0.5_f64; 8];
    for t in 0..x.len() - 1 {
        joint[4 * y[t + 1] as usize + 2 * y[t] as usize + x[t] as usize] += 1.0;
    }
    let mass: f64 = joint.iter().sum();
    let mut yf_y = [0.0; 4];
    let mut y_x = [0.0; 4];
    let mut y_past = [0.0; 2];
    for (i, value) in joint.iter_mut().enumerate() {
        *value /= mass;
        yf_y[i / 2] += *value;
        y_x[i % 4] += *value;
        y_past[(i / 2) % 2] += *value;
    }
    let entropy = |p: &[f64]| -p.iter().map(|p| p * p.log2()).sum::<f64>();
    entropy(&yf_y) + entropy(&y_x) - entropy(&y_past) - entropy(&joint)
}

fn symbols(n: usize) -> (Vec<u8>, Vec<u8>) {
    let x: Vec<_> = (0..n).map(|i| ((i * 37 + i / 7) % 11 < 4) as u8).collect();
    let y = (0..n)
        .map(|i| x[i.saturating_sub(1)] ^ ((i % 13 == 0) as u8))
        .collect();
    (x, y)
}

#[test]
fn cmi_matches_one_normalized_distribution() {
    for n in [64, 65, 100, 1_000] {
        let (x, y) = symbols(n);
        let expected = reference_cmi(&x, &y);
        let actual = te_binaria_kt(&x, &y).unwrap();
        assert!(
            (actual - expected).abs() < 2e-14,
            "n={n}: {actual} != {expected}"
        );
    }
}

#[test]
fn mismatched_series_are_not_silently_truncated() {
    let (x, mut y) = symbols(64);
    y.push(1);
    assert_eq!(te_binaria_kt(&x, &y), None);
    assert_eq!(te_binaria_kt(&y, &x), None);
}

#[test]
fn malformed_alphabet_is_rejected_without_panicking() {
    for value in [2, u8::MAX] {
        for at in [0, 31, 63] {
            let (mut x, y) = symbols(64);
            x[at] = value;
            assert_eq!(te_binaria_kt(&x, &y), None);
            assert_eq!(te_binaria_kt(&y, &x), None);
        }
    }
}

#[test]
fn unordered_events_are_not_folded_or_discarded() {
    let x = [0, 40, 20, 100];
    let y = [0, 50, 100];
    assert_eq!(transfer_entropy_eventos(&x, &y, 1), (None, None));
    assert_eq!(transfer_entropy_eventos(&y, &x, 1), (None, None));
}

#[test]
fn nonoverlapping_streams_are_not_padded_as_observed_zeros() {
    assert_eq!(
        transfer_entropy_eventos(&[0, 100], &[200, 300], 1),
        (None, None)
    );
}

#[test]
fn incomplete_terminal_bin_does_not_meet_minimum_support() {
    // 63 complete bins + a partial 64th. A tick is not proof of coverage
    // until the end of its bin. Old union/span+1 reports Some instead.
    assert_eq!(
        transfer_entropy_eventos(&[0, 639], &[0, 639], 10),
        (None, None)
    );
}

#[test]
fn maximum_timestamp_does_not_overflow_bin_count() {
    let (a, b) = transfer_entropy_eventos(&[0, u64::MAX], &[0, u64::MAX], 1);
    assert!(a.unwrap().is_finite() && b.unwrap().is_finite());
}

#[test]
fn alphabet_complements_preserve_conditional_information() {
    let (x, y) = symbols(512);
    let xc: Vec<_> = x.iter().map(|v| 1 - v).collect();
    let yc: Vec<_> = y.iter().map(|v| 1 - v).collect();
    let a = te_binaria_kt(&x, &y).unwrap();
    for (x, y) in [(&xc, &y), (&x, &yc), (&xc, &yc)] {
        assert!((te_binaria_kt(x, y).unwrap() - a).abs() < 2e-14);
    }
}

fn observation(start_ms: u64, windows: u64, width: u64) -> ObservationWindow {
    ObservationWindow {
        start_ms,
        end_ms: start_ms + windows * width,
        bin_width_ms: width,
        min_windows: 2,
    }
}

fn dense_counts(x: &[u8], y: &[u8]) -> [[[u64; 2]; 2]; 2] {
    let mut counts = [[[0; 2]; 2]; 2];
    for t in 0..x.len() - 1 {
        counts[y[t + 1] as usize][y[t] as usize][x[t] as usize] += 1;
    }
    counts
}

#[test]
fn sparse_counts_match_dense_oracle_across_scales_and_occupancies() {
    let mut state = 42_u64;
    // 3,840 grids, including silence, saturation, duplicates and jitter.
    for seed in 0..80 {
        for n in [2, 3, 7, 63, 64, 65, 100, 1_001] {
            for width in [1, 7, 200] {
                for origin in [0, (1_u64 << 53) + 1] {
                    let mut x = vec![0_u8; n];
                    let mut y = x.clone();
                    let mut xs = Vec::new();
                    let mut ys = Vec::new();
                    for t in 0..n {
                        for (bits, ts) in [(&mut x, &mut xs), (&mut y, &mut ys)] {
                            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            if state % 80 < seed {
                                bits[t] = 1;
                                let time = origin + t as u64 * width + (state >> 32) % width;
                                ts.extend([time, time]);
                            }
                        }
                    }
                    let result =
                        transfer_entropy_observada(&xs, &ys, observation(origin, n as u64, width))
                            .unwrap();
                    assert_eq!(result.counts_x_to_y, dense_counts(&x, &y));
                    assert_eq!(result.counts_y_to_x, dense_counts(&y, &x));
                    assert_eq!(result.transitions, n as u64 - 1);
                    assert!((result.x_to_y_bits - reference_cmi(&x, &y)).abs() < 3e-14);
                    assert!((result.y_to_x_bits - reference_cmi(&y, &x)).abs() < 3e-14);
                }
            }
        }
    }
}

#[test]
fn explicit_observation_distinguishes_silence_from_missing_coverage() {
    let result = transfer_entropy_observada(&[], &[], observation(10, 100, 1)).unwrap();
    assert_eq!(result.counts_x_to_y[0][0][0], 99);
    assert_eq!(result.counts_y_to_x, result.counts_x_to_y);
    assert!(result.x_to_y_bits.is_finite());
    // A positive smoothed value in a silent record is NOT empirical evidence.
    assert!(result.x_to_y_bits > 0.0);
    println!(
        "Silent observed record, 100 bins: {} bits (prior, not evidence)",
        result.x_to_y_bits
    );
    assert_eq!(transfer_entropy_eventos(&[], &[], 1), (None, None));
}

#[test]
fn explicit_coverage_allows_disjoint_event_extents() {
    let result =
        transfer_entropy_observada(&[0, 100], &[200, 299], observation(0, 300, 1)).unwrap();
    assert_eq!(result.transitions, 299);
    // Same event arrays are insufficient to infer coverage in the wrapper.
    assert_eq!(
        transfer_entropy_eventos(&[0, 100], &[200, 299], 1),
        (None, None)
    );
}

#[test]
fn typed_errors_distinguish_invalid_inputs_and_support() {
    use TransferEntropyError::*;
    let valid = observation(10, 64, 1);
    for (invalid, error) in [
        (
            ObservationWindow {
                bin_width_ms: 0,
                ..valid
            },
            ZeroBinWidth,
        ),
        (
            ObservationWindow {
                end_ms: 10,
                ..valid
            },
            InvalidInterval,
        ),
        (ObservationWindow { end_ms: 9, ..valid }, InvalidInterval),
        (
            ObservationWindow {
                min_windows: 1,
                ..valid
            },
            InvalidMinimumWindows,
        ),
        (
            ObservationWindow {
                min_windows: 65,
                ..valid
            },
            InsufficientWindows,
        ),
        (
            ObservationWindow {
                bin_width_ms: 1_000,
                ..valid
            },
            InsufficientWindows,
        ),
    ] {
        assert_eq!(transfer_entropy_observada(&[], &[], invalid), Err(error));
    }
    assert_eq!(
        transfer_entropy_observada(&[12, 11], &[], valid),
        Err(UnsortedTimestamps)
    );
    assert_eq!(
        transfer_entropy_observada(&[], &[12, 11], valid),
        Err(UnsortedTimestamps)
    );
}

#[test]
fn minimum_support_is_explicit_not_a_statistical_theorem() {
    let result = transfer_entropy_observada(&[0], &[1], observation(0, 2, 1)).unwrap();
    assert_eq!(result.transitions, 1);
    assert_eq!(result.counts_x_to_y[1][0][1], 1);
    assert!((result.x_to_y_bits - reference_cmi(&[1, 0], &[0, 1])).abs() < 2e-14);
}

#[test]
fn epoch_translation_and_time_unit_dilation_preserve_counts() {
    let xs = [0, 3, 18, 40, 89];
    let ys = [2, 4, 16, 39, 90];
    let base = transfer_entropy_observada(&xs, &ys, observation(0, 10, 10)).unwrap();
    for origin in [1_u64, (1_u64 << 53) + 17] {
        for scale in [1, 13] {
            let xs: Vec<_> = xs.iter().map(|t| origin + t * scale).collect();
            let ys: Vec<_> = ys.iter().map(|t| origin + t * scale).collect();
            let shifted =
                transfer_entropy_observada(&xs, &ys, observation(origin, 10, 10 * scale)).unwrap();
            assert_eq!(base, shifted);
        }
    }
}

#[test]
fn duplicate_events_do_not_change_binary_activity() {
    let xs = [2, 7, 81];
    let ys = [1, 8, 79];
    let duplicates = [2, 2, 2, 7, 7, 81, 81];
    let obs = observation(0, 100, 1);
    assert_eq!(
        transfer_entropy_observada(&xs, &ys, obs),
        transfer_entropy_observada(&duplicates, &ys, obs)
    );
}

#[test]
fn half_open_boundaries_and_partial_tail_are_auditable() {
    let obs = ObservationWindow {
        start_ms: 100,
        end_ms: 135,
        bin_width_ms: 10,
        min_windows: 2,
    };
    let result = transfer_entropy_observada(
        &[99, 100, 109, 110, 129, 130, 134, 135, u64::MAX],
        &[99, 119, 130, 135],
        obs,
    )
    .unwrap();
    assert_eq!(result.complete_windows, 3);
    assert_eq!(result.discarded_tail_ms, 5);
    assert_eq!(result.counts_x_to_y, dense_counts(&[1, 1, 1], &[0, 1, 0]));
    assert_eq!(result.counts_y_to_x, dense_counts(&[0, 1, 0], &[1, 1, 1]));
}

#[test]
fn swapping_assets_swaps_both_directional_estimates_and_counts() {
    let xs = [0, 7, 13, 61, 80];
    let ys = [1, 8, 14, 62, 81];
    let obs = observation(0, 100, 1);
    let xy = transfer_entropy_observada(&xs, &ys, obs).unwrap();
    let yx = transfer_entropy_observada(&ys, &xs, obs).unwrap();
    assert_eq!(xy.x_to_y_bits, yx.y_to_x_bits);
    assert_eq!(xy.counts_x_to_y, yx.counts_y_to_x);
    assert_eq!(xy.counts_y_to_x, yx.counts_x_to_y);
}

#[test]
fn century_silence_uses_exact_counts_without_dense_allocation() {
    let century_ms = 100 * 365_250_u64 * 86_400;
    let result = transfer_entropy_observada(&[], &[], observation(0, century_ms, 1)).unwrap();
    assert_eq!(result.transitions, century_ms - 1);
    assert_eq!(result.counts_x_to_y[0][0][0], century_ms - 1);
}

#[test]
fn upper_epoch_and_large_bin_width_have_no_arithmetic_overflow() {
    let start = u64::MAX - 1_000;
    let obs = observation(start, 100, 10);
    let result = transfer_entropy_observada(&[start, u64::MAX - 1], &[u64::MAX], obs).unwrap();
    assert_eq!(result.transitions, 99);
    assert_eq!(result.counts_x_to_y[0][0][1], 1);
    let obs = observation(0, 2, u64::MAX / 2);
    assert!(transfer_entropy_observada(&[0, u64::MAX - 1], &[], obs).is_ok());
}

#[test]
fn legacy_intersection_matches_the_explicit_contract() {
    let xs = [0, 23, 40, 500, 750, 1_000];
    let ys = [100, 180, 510, 900];
    let explicit = transfer_entropy_observada(
        &xs,
        &ys,
        ObservationWindow {
            start_ms: 100,
            end_ms: 900,
            bin_width_ms: 10,
            min_windows: 64,
        },
    )
    .unwrap();
    assert_eq!(
        transfer_entropy_eventos(&xs, &ys, 10),
        (Some(explicit.x_to_y_bits), Some(explicit.y_to_x_bits))
    );
}
