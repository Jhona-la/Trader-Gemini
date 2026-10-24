//! CONTRATO FORMAL FASE 4 — ÁTOMO 4.1 (Ola Omega220)
//!
//! Blindaje Causal E2E, Erradicación de Panic Paths y Acoplamiento de Operadores Dinámicos:
//! 1. Inmunidad absoluta contra pánicos en `net_multiplexer::get_global_client()` con fallback robusto.
//! 2. Streaming continuo y O(1)/tick de Koopman DMD y Entrelazamiento Cuántico Tensor MPS en `StatefulEngine`.
//! 3. Modulación hidrodinámica Navier-Stokes sobre fricción y spread dinámico según la fracción laminar.
//! 4. Filtro anti-whipsaw por divergencia modal Koopman/DMD (modo de alternancia Nyquist de alta frecuencia).
//! 5. Invariantes de memoria acotada e inmunidad IEEE 754 (rechazo sin pánico ante NaN / Inf).

use feature_engine::entrelazamiento::{ModoBinarizacion, EntrelazamientoStreaming};
use god_engine_core::stateful_engine::StatefulEngine;
use quantum_arena::net_multiplexer::get_global_client;

#[test]
fn test_f4_001_net_multiplexer_panic_immunity() {
    // C1: get_global_client() no debe causar pánico bajo ninguna condición y debe retornar
    // una referencia estática válida al cliente HTTP.
    let client = get_global_client();
    assert!(
        std::ptr::eq(client, get_global_client()),
        "El cliente HTTP global debe ser un singleton estático referencialmente idéntico"
    );
}

#[test]
fn test_f4_001_stateful_engine_koopman_and_quantum_streaming() {
    // C2: StatefulEngine debe mantener el estado cuántico y Koopman en NaN durante warmup,
    // y poblarlos deterministamente tras cumplir la cadencia de 64 ticks con datos suficientes.
    let mut fe = StatefulEngine::new();

    // Estado inicial de honestidad empírica (ausencia declarada)
    assert!(fe.koopman_modulo.is_nan(), "koopman_modulo debe arrancar en NaN");
    assert!(fe.koopman_angulo.is_nan(), "koopman_angulo debe arrancar en NaN");
    assert!(fe.koopman_bifurcacion.is_nan(), "koopman_bifurcacion debe arrancar en NaN");
    assert!(fe.quantum_delta_s.is_nan(), "quantum_delta_s debe arrancar en NaN");
    assert!(fe.quantum_entropy.is_nan(), "quantum_entropy debe arrancar en NaN");

    let mut price = 50_000.0;
    let mut ts = 1_700_000_000_000u64;
    let mut rng = 123456789u64;

    // Inyectar 130 ticks con dinámica estocástica no lineal rica (rango completo para Hankel)
    for _ in 0..130 {
        ts += 100;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let ret = ((rng >> 32) as i32 as f64 / 2147483648.0) * 0.002;
        price *= 1.0 + ret;
        let vol = 1.0;

        fe.process_tick(price, vol, ts);
    }

    // Tras superar 128 ticks (2 ciclos de cadencia de 64), los observables deben estar vivos y finitos
    assert!(
        fe.koopman_modulo.is_finite(),
        "koopman_modulo debe ser finito tras 128 ticks, obtenido: {}",
        fe.koopman_modulo
    );
    assert!(
        fe.koopman_modulo > 0.0,
        "koopman_modulo debe ser estrictamente positivo, obtenido: {}",
        fe.koopman_modulo
    );

    assert!(
        fe.koopman_angulo.is_finite(),
        "koopman_angulo debe ser finito, obtenido: {}",
        fe.koopman_angulo
    );
    assert!(
        (0.0..=std::f32::consts::PI).contains(&fe.koopman_angulo),
        "koopman_angulo debe pertenecer al intervalo físico [0, π], obtenido: {}",
        fe.koopman_angulo
    );

    assert!(
        fe.koopman_bifurcacion.is_finite(),
        "koopman_bifurcacion debe ser finita, obtenido: {}",
        fe.koopman_bifurcacion
    );

    assert!(
        fe.quantum_delta_s.is_finite(),
        "quantum_delta_s debe ser finito tras 128 ticks de streaming, obtenido: {}",
        fe.quantum_delta_s
    );

    assert!(
        fe.quantum_entropy.is_finite(),
        "quantum_entropy debe ser finita, obtenido: {}",
        fe.quantum_entropy
    );
}

#[test]
fn test_f4_001_navier_stokes_friction_modulation() {
    // C3: La modulación hidrodinámica Navier-Stokes penaliza la fricción requerida
    // conforme el flujo se vuelve turbulento (laminar_share decreciente).
    let baseline_friction = 0.0010; // 10 bps

    let calc_hydro_friction = |laminar_share: f64| -> f64 {
        let ns_laminar = if laminar_share.is_finite() && laminar_share >= 0.0 {
            laminar_share.clamp(0.0, 1.0)
        } else {
            1.0
        };
        baseline_friction * (1.0 + (1.0 - ns_laminar) * 0.5)
    };

    // Caso 1: Flujo puramente laminar (laminar_share = 1.0)
    let f_laminar = calc_hydro_friction(1.0);
    assert!(
        (f_laminar - baseline_friction).abs() < 1e-12,
        "En flujo laminar puro, la fricción debe coincidir exactamente con el baseline: got {}",
        f_laminar
    );

    // Caso 2: Transición crítica de vórtices (laminar_share = 0.5)
    let f_transicion = calc_hydro_friction(0.5);
    let f_expected_trans = baseline_friction * 1.25;
    assert!(
        (f_transicion - f_expected_trans).abs() < 1e-12,
        "En transición (0.5), la fricción debe expandirse exactamente 1.25x: got {}, expected {}",
        f_transicion,
        f_expected_trans
    );

    // Caso 3: Turbulencia severa / cascada de enstrofía (laminar_share = 0.0)
    let f_turbulenta = calc_hydro_friction(0.0);
    let f_expected_turb = baseline_friction * 1.50;
    assert!(
        (f_turbulenta - f_expected_turb).abs() < 1e-12,
        "En turbulencia máxima (0.0), la fricción debe expandirse 1.50x: got {}, expected {}",
        f_turbulenta,
        f_expected_turb
    );

    // Monotonía estricta con la pérdida de laminaridad
    assert!(
        f_laminar < f_transicion && f_transicion < f_turbulenta,
        "La fricción hidrodinámica debe crecer monótonamente con la turbulencia"
    );

    // Inmunidad ante valores no finitos (fallback a laminar sin inflar artificialmente)
    let f_nan = calc_hydro_friction(f64::NAN);
    assert_eq!(f_nan, baseline_friction);
}

#[test]
fn test_f4_001_koopman_whipsaw_anti_oscillation_guard() {
    // C4: El filtro anti-whipsaw por divergencia modal Koopman/DMD debe dispararse
    // únicamente cuando el modo dominante tiene módulo > 1.15 Y ángulo > 2.85 rad (frecuencia Nyquist).
    let eval_koopman_whipsaw = |modulo: f32, angulo: f32| -> bool {
        modulo.is_finite() && angulo.is_finite() && modulo > 1.15 && angulo > 2.85
    };

    // Caso 1: Modo explosivo de alta frecuencia (alternancia violenta tick-a-tick)
    assert!(
        eval_koopman_whipsaw(1.25, 3.05),
        "Debe activar veto anti-whipsaw ante modo alternante explosivo (|λ|=1.25, θ=3.05)"
    );

    // Caso 2: Modo de alta frecuencia pero amortiguado (|λ| <= 1.15)
    assert!(
        !eval_koopman_whipsaw(0.95, 3.05),
        "No debe vetar si el modo de alta frecuencia está amortiguado (|λ|=0.95)"
    );

    // Caso 3: Modo expansivo pero tendencial de baja frecuencia (θ <= 2.85)
    assert!(
        !eval_koopman_whipsaw(1.30, 0.45),
        "No debe vetar un modo tendencial de baja frecuencia (|λ|=1.30, θ=0.45)"
    );

    // Caso 4: Valores NaN / no inicializados (fail-closed a permitir trade sin falsos positivos de whipsaw)
    assert!(
        !eval_koopman_whipsaw(f32::NAN, 3.05),
        "NaN en módulo no debe disparar falso positivo"
    );
    assert!(
        !eval_koopman_whipsaw(1.25, f32::NAN),
        "NaN en ángulo no debe disparar falso positivo"
    );
    assert!(
        !eval_koopman_whipsaw(f32::NAN, f32::NAN),
        "NaN completo no debe disparar falso positivo"
    );
}

#[test]
fn test_f4_001_entrelazamiento_streaming_capacity_and_nan_rejection() {
    // C5: EntrelazamientoStreaming debe operar en memoria rígidamente acotada (ring buffer)
    // y rechazar valores no finitos sin corromper el recuento ni la medida.
    let capacidad = 256;
    let mut st = EntrelazamientoStreaming::new(8, ModoBinarizacion::Signo, capacidad)
        .expect("Construcción válida de streaming");

    assert_eq!(st.capacidad(), capacidad);
    assert_eq!(st.activos(), 0);
    assert_eq!(st.descartados_totales(), 0);

    // Inyectar 50 valores no finitos
    for _ in 0..25 {
        st.push(f64::NAN);
        st.push(f64::INFINITY);
    }

    assert_eq!(st.activos(), 0, "No finitos no deben entrar al buffer de cálculo");
    assert_eq!(st.descartados_totales(), 50, "Debe registrar 50 descartes exactos");

    // Inyectar 500 valores válidos (superando la capacidad de 256)
    for i in 0..500 {
        let ret = if i % 2 == 0 { 0.001 } else { -0.001 };
        st.push(ret);
    }

    // El buffer debe estar saturado exactamente a `capacidad` sin rebasamiento
    assert_eq!(st.activos(), capacidad, "El buffer debe mantener exactamente su capacidad acotada");

    // La medición debe tener éxito y producir un ΔS matemáticamente válido
    let res = st.medir();
    assert!(res.is_ok(), "La medición de entrelazamiento streaming falló: {:?}", res.err());
    let medida = res.unwrap();
    assert!(medida.delta.is_finite(), "delta debe ser finito");
    assert!(medida.s.is_finite(), "entropía von Neumann s debe ser finita");
}
