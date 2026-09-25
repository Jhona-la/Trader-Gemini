use quantum_arena::temporal_spectrum::{TemporalSpectrum, SPECTRUM_SCALES_MS};

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 2e-12, "{actual} != {expected}");
}

#[test]
fn interpolated_observables_do_not_commute_with_nonlinear_maps() {
    let mut spec = TemporalSpectrum::new();
    spec.scales[18].momentum_z = 0.0;
    spec.scales[19].momentum_z = 2.0;
    spec.scales[18].signal = 0.0;
    spec.scales[19].signal = 2.0_f64.tanh();
    let tau = (SPECTRUM_SCALES_MS[18] * SPECTRUM_SCALES_MS[19]).sqrt();
    let state = spec.state_at(tau);
    close(state.momentum_z, 1.0);
    close(state.signal, 2.0_f64.tanh() / 2.0);
    assert!((state.signal - state.momentum_z.tanh()).abs() > 0.25);
}

#[test]
fn nodal_activity_mass_can_be_positive_at_a_directional_cancellation() {
    let mut spec = TemporalSpectrum::new();
    for (i, signal) in [(18, 0.8), (19, -0.8)] {
        spec.scales[i].signal = signal;
        spec.scales[i].persistence = 0.5;
        spec.scales[i].epigenetic_gain = 1.0;
    }
    let tau = (SPECTRUM_SCALES_MS[18] * SPECTRUM_SCALES_MS[19]).sqrt();
    close(spec.signal_at(tau), 0.0);
    close(spec.continuous_energy_density(tau), 0.8);
}

#[test]
fn local_gradient_is_not_a_secant_averaging_across_a_knot() {
    let mut spec = TemporalSpectrum::new();
    spec.scales[18].signal = 0.0;
    spec.scales[19].signal = 1.0;
    spec.scales[20].signal = 1.0;
    let tau = SPECTRUM_SCALES_MS[19] * 1.1;
    close(spec.spectral_gradient_at(tau), 0.0);
}

#[test]
fn gradient_on_a_log_linear_segment_matches_its_slope() {
    let mut spec = TemporalSpectrum::new();
    spec.scales[18].signal = -0.5;
    spec.scales[19].signal = 0.7;
    let expected = 1.2 / (SPECTRUM_SCALES_MS[19] / SPECTRUM_SCALES_MS[18]).ln();
    for fraction in [0.01, 0.25, 0.5, 0.75, 0.99] {
        let tau = SPECTRUM_SCALES_MS[18] * 4.0_f64.powf(fraction);
        close(spec.spectral_gradient_at(tau), expected);
    }
}

#[test]
fn gradient_at_a_knot_uses_the_documented_right_derivative() {
    let mut spec = TemporalSpectrum::new();
    spec.scales[18].signal = 0.0;
    spec.scales[19].signal = 0.5;
    spec.scales[20].signal = -0.5;
    close(
        spec.spectral_gradient_at(SPECTRUM_SCALES_MS[19]),
        -1.0 / 4.0_f64.ln(),
    );
}

#[test]
fn gradient_of_constant_extrapolation_is_zero() {
    let mut spec = TemporalSpectrum::new();
    spec.scales[0].signal = 0.0;
    spec.scales[1].signal = 1.0;
    spec.scales[30].signal = -1.0;
    spec.scales[31].signal = 0.5;
    for tau in [
        SPECTRUM_SCALES_MS[0] * 0.9,
        SPECTRUM_SCALES_MS[31],
        SPECTRUM_SCALES_MS[31] * 1.1,
        f64::MAX,
    ] {
        close(spec.spectral_gradient_at(tau), 0.0);
    }
    close(
        spec.spectral_gradient_at(SPECTRUM_SCALES_MS[0]),
        1.0 / 4.0_f64.ln(),
    );
}

#[test]
fn invalid_gradient_queries_are_neutral() {
    let mut spec = TemporalSpectrum::new();
    for s in &mut spec.scales {
        s.signal = 0.75;
    }
    for tau in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        close(spec.spectral_gradient_at(tau), 0.0);
    }
}
