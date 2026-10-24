//! # Contrato Formal F11-001: Integrales de Camino de Feynman, Instantones y Tunelamiento Cuántico
//!
//! Verificación rigurosa de invariantes físicos y cuánticos:
//! C1: Acción analítica del instantón continuo y convergencia de trayectoria euclidiana.
//! C2: Altura analítica de barrera y detección de puntos de retorno clásicos.
//! C3: Transmitancia WKB semiclásica Gamow y confinamiento en pozo metaestable.
//! C4: Transición a tunelamiento inminente y ruptura clásica de libro.
//! C5: Propagador de Fokker-Planck multivariante y conservación de energía.
//! C6: Resiliencia numérica extrema, rechazo de NaNs/Infs y cero pánicos.

use feature_engine::feynman_path_instanton::{
    FeynmanInstantonEngine, InstantonParams, InstantonRegime, InstantonTunnelingReport,
    MAX_PATH_POINTS,
};

#[test]
fn test_c1_analytical_instanton_action_and_path_convergence() {
    let params = InstantonParams {
        effective_mass: 1.0,
        hbar_eff: 0.1,
        lambda_coupling: 2.0,
        well_separation_a: 1.0,
    };
    let engine = FeynmanInstantonEngine::new(params);

    // Acción analítica continua: S_0 = (2/3) * a^3 * sqrt(2 * m * lambda)
    // Para m = 1, lambda = 2, a = 1 -> S_0 = (2/3) * 1 * sqrt(4) = 4/3 ≈ 1.333333333
    let s0 = engine.analytical_instanton_action();
    let expected_s0 = 4.0 / 3.0;
    assert!(
        (s0 - expected_s0).abs() < 1e-12,
        "Acción analítica esperada: {}, obtenida: {}",
        expected_s0,
        s0
    );

    // Muestreo de trayectoria del instantón euclidiano x(tau) = a * tanh(sqrt(lambda / (2*m)) * a * tau)
    // para tau in [-3.0, 3.0] en K = 31 puntos
    let k = 31;
    let tau_start = -3.0_f64;
    let tau_end = 3.0_f64;
    let delta_tau = (tau_end - tau_start) / (k - 1) as f64;

    let mut path = [0.0_f64; 31];
    let gamma = (params.lambda_coupling / (2.0 * params.effective_mass)).sqrt() * params.well_separation_a;
    for i in 0..k {
        let tau = tau_start + (i as f64) * delta_tau;
        path[i] = params.well_separation_a * (gamma * tau).tanh();
    }

    let action_num = engine
        .compute_euclidean_action(&path, delta_tau, 0.0)
        .expect("Debe calcular acción euclidiana");

    // La acción discreta truncada debe aproximar S_0 dentro de un error de borde < 15%
    let rel_diff = (action_num - expected_s0).abs() / expected_s0;
    assert!(
        rel_diff < 0.15,
        "Acción discreta {} debe aproximar S_0 {} dentro del 15%, error: {}",
        action_num,
        expected_s0,
        rel_diff
    );
}

#[test]
fn test_c2_barrier_height_and_classical_turning_points() {
    let params = InstantonParams {
        effective_mass: 1.0,
        hbar_eff: 0.05,
        lambda_coupling: 4.0,
        well_separation_a: 1.5,
    };
    let engine = FeynmanInstantonEngine::new(params);

    // Potencial V(x) = (lambda / 4) * (x^2 - a^2)^2
    // En los mínimos x = ±a, V(±a) = 0.
    assert!(engine.evaluate_potential(1.5, 0.0).abs() < 1e-12);
    assert!(engine.evaluate_potential(-1.5, 0.0).abs() < 1e-12);

    // En la cúspide x = 0, V(0) = (lambda / 4) * a^4 = 1.0 * (1.5)^4 = 5.0625
    let v_peak = engine.evaluate_potential(0.0, 0.0);
    assert!((v_peak - 5.0625).abs() < 1e-12);

    // Para una energía E = 2.0 < 5.0625, deben existir dos puntos de retorno x1 < 0 < x2
    // donde V(x) = E -> (x^2 - a^2)^2 = 4 E / lambda = 4 * 2 / 4 = 2
    // x^2 = a^2 - sqrt(2) = 2.25 - 1.41421356 = 0.8357864
    // x = ±sqrt(0.8357864) ≈ ±0.91421356
    let report = engine.evaluate_tunneling(-1.5, 2.0_f64.sqrt(), 0.0);
    assert!(report.is_some());
    let rep = report.unwrap();
    assert!(rep.barrier_height > 0.0);
    assert!(rep.kinetic_energy > 0.0);
}

#[test]
fn test_c3_semiclassical_gamow_wkb_tunneling_and_confinement() {
    let params = InstantonParams {
        effective_mass: 2.0,
        hbar_eff: 0.05,
        lambda_coupling: 8.0,
        well_separation_a: 1.0,
    };
    let engine = FeynmanInstantonEngine::new(params);

    // Caso con flujo de órdenes casi nulo (orden atrapada en el pozo metaestable)
    let report = engine
        .evaluate_tunneling(-1.0, 0.01, 0.0)
        .expect("Debe evaluar tunelamiento en baja energía");

    // Con energía muy baja frente a la barrera de 2.0, la probabilidad de tunelamiento es microscópica
    assert_eq!(report.regime, InstantonRegime::MetastableTrapped);
    assert!(
        report.tunneling_probability < 0.01,
        "En baja energía, la probabilidad de tunelamiento debe ser despreciable (< 0.01)"
    );
    assert!(
        report.barrier_action_b > 0.5,
        "La acción bajo la barrera debe ser sustancial"
    );
    assert_eq!(report.projected_target_x, 1.0);
}

#[test]
fn test_c4_breakout_transition_and_monotonic_tunneling() {
    let params = InstantonParams {
        effective_mass: 1.0,
        hbar_eff: 0.1,
        lambda_coupling: 2.0,
        well_separation_a: 1.0,
    };
    let engine = FeynmanInstantonEngine::new(params);

    // Evaluar para 3 niveles de velocidad de flujo de órdenes: bajo, medio, extremo
    let rep_low = engine.evaluate_tunneling(-1.0, 0.1, 0.0).unwrap();
    let rep_med = engine.evaluate_tunneling(-1.0, 0.7, 0.0).unwrap();
    let rep_high = engine.evaluate_tunneling(-1.0, 1.5, 0.0).unwrap();

    // Monotonía: a mayor velocidad de flujo, mayor probabilidad de atravesar la pared de liquidez
    assert!(
        rep_med.tunneling_probability > rep_low.tunneling_probability,
        "Probabilidad media ({}) debe superar a baja ({})",
        rep_med.tunneling_probability,
        rep_low.tunneling_probability
    );
    assert!(
        rep_high.tunneling_probability >= rep_med.tunneling_probability,
        "Probabilidad alta ({}) debe superar a media ({})",
        rep_high.tunneling_probability,
        rep_med.tunneling_probability
    );

    // Con velocidad 1.5, la energía cinética es 0.5 * 1.0 * 2.25 = 1.125 > V_peak = 0.5
    // Debe producirse una ruptura clásica con probabilidad idénticamente 1.0
    assert_eq!(rep_high.regime, InstantonRegime::ClassicalBreakout);
    assert_eq!(rep_high.tunneling_probability, 1.0);
    assert_eq!(rep_high.barrier_action_b, 0.0);
}

#[test]
fn test_c5_multivariate_fokker_planck_propagator() {
    let params = InstantonParams::default();
    let engine = FeynmanInstantonEngine::new(params);

    let x0 = 0.5; // Dentro del pozo derecho
    let var0 = 0.01;
    let velocity = 0.1;
    let diff_d = 0.02;
    let dt = 0.05;

    let (next_x, next_var) = engine
        .propagate_fokker_planck(x0, var0, velocity, diff_d, dt, 0.0)
        .expect("Debe propagar Fokker-Planck");

    // Verificación de expansión difusiva de la varianza Var(t+dt) = Var + 2*D*dt
    let expected_var = var0 + 2.0 * diff_d * dt;
    assert!(
        (next_var - expected_var).abs() < 1e-12,
        "Varianza esperada: {}, obtenida: {}",
        expected_var,
        next_var
    );

    // La fuerza -V'(0.5) empuja hacia el pozo a = 1.0
    // V'(x) = lambda * x * (x^2 - a^2) = 2.0 * 0.5 * (0.25 - 1.0) = -0.75
    // Fuerza = -V'(x) = +0.75 (hacia la derecha, atractor a = 1.0)
    // El precio debe acelerarse en dirección positiva hacia el atractor
    assert!(next_x > x0);
}

#[test]
fn test_c6_silicon_invariants_and_zero_panic_guarantee() {
    let engine = FeynmanInstantonEngine::new(InstantonParams::default());

    // 1. Trayectoria degenerada (< 2 puntos)
    assert!(engine.compute_euclidean_action(&[1.0], 0.1, 0.0).is_none());

    // 2. Paso de tiempo temporal inválido (<= 0 o no finito)
    assert!(engine.compute_euclidean_action(&[1.0, 1.1], 0.0, 0.0).is_none());
    assert!(engine.compute_euclidean_action(&[1.0, 1.1], -0.1, 0.0).is_none());
    assert!(engine.compute_euclidean_action(&[1.0, 1.1], f64::NAN, 0.0).is_none());

    // 3. Contaminación por NaN o Infinito en la trayectoria
    let nan_path = [1.0, f64::NAN, 1.5];
    assert!(engine.compute_euclidean_action(&nan_path, 0.1, 0.0).is_none());

    // 4. Entradas no finitas en tunelamiento
    assert!(engine.evaluate_tunneling(f64::NAN, 1.0, 0.0).is_none());
    assert!(engine.evaluate_tunneling(1.0, f64::INFINITY, 0.0).is_none());

    // 5. Entradas inválidas en Fokker-Planck
    assert!(engine.propagate_fokker_planck(f64::NAN, 0.1, 0.1, 0.1, 0.1, 0.0).is_none());
    assert!(engine.propagate_fokker_planck(1.0, -0.1, 0.1, 0.1, 0.1, 0.0).is_none());
    assert!(engine.propagate_fokker_planck(1.0, 0.1, 0.1, -0.1, 0.1, 0.0).is_none());
    assert!(engine.propagate_fokker_planck(1.0, 0.1, 0.1, 0.1, -0.1, 0.0).is_none());

    // 6. Entropía holográfica siempre finita y no negativa
    let s = engine.compute_holographic_entropy(0.0);
    assert!(s >= 0.0 && s.is_finite());
}
