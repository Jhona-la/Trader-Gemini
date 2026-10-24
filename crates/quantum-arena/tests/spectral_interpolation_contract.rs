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
    // RET-33: state_at se retiró (sin lector); copiaba estos dos getters.
    close(spec.momentum_z_at(tau), 1.0);
    close(spec.signal_at(tau), 2.0_f64.tanh() / 2.0);
    assert!((spec.signal_at(tau) - spec.momentum_z_at(tau).tanh()).abs() > 0.25);
}

/// CL-35: sin datos no hay nada observado y la masa es 0 (D-742). El contrato
/// fija el álgebra de la interpolación sobre masas conocidas, así que primero
/// se observa la malla: 20 000 s de precio constante con un evento por segundo
/// dejan la masa del núcleo y la fracción resuelta en 1 (en f64) para las
/// escalas 18 y 19.
fn espectro_observado() -> TemporalSpectrum {
    let mut spec = TemporalSpectrum::new();
    for i in 0..20_000u64 {
        spec.update(100.0, 1_000 + i * 1_000);
    }
    spec
}

#[test]
fn nodal_activity_mass_can_be_positive_at_a_directional_cancellation() {
    let mut spec = espectro_observado();
    for (i, signal) in [(18, 0.8), (19, -0.8)] {
        spec.scales[i].signal = signal;
        spec.scales[i].persistence = 0.5;
        spec.scales[i].epigenetic_gain = 1.0;
    }
    spec.refresh_fusion();
    let tau = (SPECTRUM_SCALES_MS[18] * SPECTRUM_SCALES_MS[19]).sqrt();
    close(spec.signal_at(tau), 0.0);
    close(spec.continuous_energy_density(tau), 0.8);
}

// RET-33: spectral_gradient_at se retiró (sin lector; la pendiente global
// viva es SpectralFieldState::spectral_tilt). Sus pruebas se migran a la API
// viva que interpolaba, signal_at: lineal por tramos en ln(τ) y constante
// fuera de la malla.

#[test]
fn signal_at_is_log_linear_on_a_segment() {
    let mut spec = TemporalSpectrum::new();
    spec.scales[18].signal = -0.5;
    spec.scales[19].signal = 0.7;
    for fraction in [0.01, 0.25, 0.5, 0.75, 0.99] {
        let tau = SPECTRUM_SCALES_MS[18] * 4.0_f64.powf(fraction);
        close(spec.signal_at(tau), -0.5 + fraction * 1.2);
    }
}

#[test]
fn signal_at_returns_the_node_value_at_a_knot() {
    let mut spec = TemporalSpectrum::new();
    spec.scales[18].signal = 0.0;
    spec.scales[19].signal = 0.5;
    spec.scales[20].signal = -0.5;
    close(spec.signal_at(SPECTRUM_SCALES_MS[19]), 0.5);
}

#[test]
fn signal_at_does_not_average_across_a_knot() {
    let mut spec = TemporalSpectrum::new();
    spec.scales[18].signal = 0.0;
    spec.scales[19].signal = 1.0;
    spec.scales[20].signal = 1.0;
    close(spec.signal_at(SPECTRUM_SCALES_MS[19] * 1.1), 1.0);
}

#[test]
fn signal_at_extends_constantly_outside_the_grid() {
    let mut spec = TemporalSpectrum::new();
    spec.scales[0].signal = 0.25;
    spec.scales[1].signal = 1.0;
    spec.scales[30].signal = -1.0;
    spec.scales[31].signal = 0.5;
    close(spec.signal_at(SPECTRUM_SCALES_MS[0] * 0.9), 0.25);
    for tau in [SPECTRUM_SCALES_MS[31], SPECTRUM_SCALES_MS[31] * 1.1, f64::MAX] {
        close(spec.signal_at(tau), 0.5);
    }
}

#[test]
fn invalid_signal_queries_are_neutral() {
    let mut spec = TemporalSpectrum::new();
    for s in &mut spec.scales {
        s.signal = 0.75;
    }
    for tau in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        close(spec.signal_at(tau), 0.0);
    }
}
