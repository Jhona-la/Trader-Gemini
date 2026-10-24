//! CONTRATO FORMAL: DESACOPLAMIENTO DINÁMICO DE UMBRALES DE UTILIDAD ESPERADA Y WIN RATE EN SeniorTeleonomia (Ola Ω111).
//!
//! # Por qué existe
//! Anteriormente, `SeniorTeleonomia` evaluaba el veto por futilidad mediante dos constantes estáticas congeladas:
//! `UTILITY_VETO_FLOOR = 0.02` y `UTILITY_VETO_WR = 0.35`.
//! En micro-scalping de muy alta velocidad (s -> 0, tau ~ 30 s), la utilidad neta esperada opera legítimamente
//! en el rango de 5 a 15 bps (0.005 a 0.015). El piso rígido de 0.02 (200 bps) vetaba sistemáticamente micro-trades
//! con esperanza matemática positiva. Además, bajo asimetría matemática de payoff (RR >= 2.5), un win rate del 32%
//! posee expectativa estrictamente positiva, pero quedaba vetado por el umbral fijo de 0.35.
//!
//! # Solución (Ola Ω111)
//! 1. `compute_teleonomy_utility_thresholds` modula continuamente el piso de utilidad sobre la coordenada espectral
//!    log-lineal s in [0, 1] (0.010 en tau=30s hasta 0.020 en tau=12h) y modula el win rate admisible según la predictibilidad fractal de Hurst.
//! 2. `MarketSnapshotPayload` expone `utility_veto_floor` y `utility_veto_wr` dinámicos con validación de cotas en [0, 0.50] y [0, 1].
//! 3. `CouncilParams` incorpora `utility_veto_floor` y `utility_veto_wr` con builder `with_utility_thresholds` y propagación hacia `deliberar_traced`.
//! 4. Preservación exacta bit-a-bit de los valores canónicos (0.02, 0.35) en régimen secular (s=1.0, H=0.5).

use metacortex_engine::consejo_seniors::{
    compute_teleonomy_utility_thresholds, ConsejoDeliberacion, CouncilParams,
    MarketSnapshotPayload, SeniorAgent, SeniorRole, SeniorTeleonomia,
    UTILITY_VETO_FLOOR, UTILITY_VETO_WR,
};

fn base_payload() -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        book_imbalance: 0.1,
        hurst_exponent: 0.5,
        ml_prob: 0.5,
        fused_score: 0.0,
        persistence: 0.0,
        atr_pct: 0.001,
        loss_streak: 0,
        intended_direction: 1.0,
        do_calculus_risk: 0.0,
        causal_veto_threshold: 0.75,
        current_drawdown_pct: 0.0,
        estimated_slippage_bps: 0.0,
        dominant_tau_ms: 30_000.0, // Banda rápida tau = 30s (s = 0.0)
        whale_burst_z: 0.0,
        liquidation_severity: 0.0,
        open_interest_norm: 0.0,
        spoof_score: 0.0,
        crowd_ls_ratio: 1.0,
        crowd_taker_ratio: 1.0,
        ml_model_base: 0.5,
        ..Default::default()
    }
}

#[test]
fn pure_function_computes_expected_thresholds_across_spectral_and_hurst_scales() {
    // 1. Régimen secular swing lento (s = 1.0, H = 0.5 ruido browniano):
    let (floor_slow, wr_slow) = compute_teleonomy_utility_thresholds(0.02, 0.35, 1.0, 0.5);
    assert!((floor_slow - 0.02).abs() < 1e-12, "En s=1.0 debe reproducir 0.02 canónico");
    assert!((wr_slow - 0.35).abs() < 1e-12, "En H=0.5 debe reproducir 0.35 canónico");

    // 2. Régimen micro-scalping ultra-rápido (s = 0.0, H = 0.5):
    let (floor_fast, wr_fast) = compute_teleonomy_utility_thresholds(0.02, 0.35, 0.0, 0.5);
    assert!((floor_fast - 0.010).abs() < 1e-12, "En s=0.0 debe modular suavemente a 0.010 (10 bps)");
    assert!((wr_fast - 0.35).abs() < 1e-12, "En H=0.5 wr_factor es 1.0");

    // 3. Monotonía estricta en s in [0, 1]:
    let mut prev_floor = 0.009;
    for step in 0..=20 {
        let s = step as f64 / 20.0;
        let (floor, _) = compute_teleonomy_utility_thresholds(0.02, 0.35, s, 0.5);
        assert!(floor >= prev_floor, "Piso de utilidad debe ser monótonamente creciente en s");
        prev_floor = floor;
    }

    // 4. Modulación por persistencia fractal de Hurst:
    // Con H = 0.85 (persistencia fuerte), la predictibilidad es |0.85 - 0.5| * 2 = 0.70.
    // wr_factor = 1.0 - 0.20 * 0.70 = 0.86 -> 0.35 * 0.86 = 0.301.
    let (_, wr_persistent) = compute_teleonomy_utility_thresholds(0.02, 0.35, 0.5, 0.85);
    assert!((wr_persistent - 0.301).abs() < 1e-9, "Con H=0.85 debe modular a 0.301");

    // 5. Robustez matemática ante no-finitos y cotas:
    let (safe_floor, safe_wr) = compute_teleonomy_utility_thresholds(f64::NAN, f64::INFINITY, f64::NAN, f64::NAN);
    assert!((safe_floor - (UTILITY_VETO_FLOOR * 0.75)).abs() < 1e-12);
    assert!((safe_wr - UTILITY_VETO_WR).abs() < 1e-12);
}

#[test]
fn teleonomia_differentiates_micro_scalping_and_swing_without_arbitrary_veto() {
    let teleonomia = SeniorTeleonomia;

    // Caso de prueba:
    // Operación legítima de micro-scalping:
    // Señal espectral moderada (fused=0.15, pers=0.25), Hurst=0.60, sin ML edge (|0.5 - 0.5| = 0).
    // Produce una utilidad esperada neta en el rango de ~0.0138 (13.8 bps).
    let mut payload = base_payload();
    payload.fused_score = 0.15;
    payload.persistence = 0.25;
    payload.hurst_exponent = 0.60;
    payload.ml_prob = 0.50;
    payload.ml_model_base = 0.50;

    let wr = 0.33; // Win-rate del 33% (viable con RR >= 2.5)

    // A) En micro-scalping (tau = 30_000 ms -> s = 0.0):
    // El piso modulado es 0.010. Como expected_utility ~ 0.0138 > 0.010, NO se veta.
    payload.dominant_tau_ms = 30_000.0;
    let opinion_fast = teleonomia.evaluate(&payload, wr);
    assert!(
        !opinion_fast.is_veto,
        "En micro-scalping, trade con utilidad ~0.0138 NO debe ser vetado: justification={}",
        opinion_fast.justification
    );
    assert!(opinion_fast.signal_direction > 0.0);

    // B) En régimen swing lento (tau = 43_200_000 ms = 12 h -> s = 1.0):
    // El piso modulado es 0.020. Como expected_utility ~ 0.0138 < 0.020 y wr < 0.336, SE VETA por futilidad.
    payload.dominant_tau_ms = 43_200_000.0;
    let opinion_slow = teleonomia.evaluate(&payload, wr);
    assert!(
        opinion_slow.is_veto,
        "En swing secular, trade con utilidad < 0.020 y wr < 0.35 DEBE ser vetado: justification={}",
        opinion_slow.justification
    );
    assert_eq!(opinion_slow.signal_direction, 0.0);
}

#[test]
fn council_params_builder_and_validation_for_utility_thresholds() {
    // 1. Configuración válida con builder
    let params = CouncilParams::default()
        .with_utility_thresholds(0.015, 0.30)
        .expect("Parámetros válidos");
    assert_eq!(params.utility_veto_floor, 0.015);
    assert_eq!(params.utility_veto_wr, 0.30);

    // 2. Rechazo explícito de parámetros fuera de cotas
    assert!(CouncilParams::default().with_utility_thresholds(-0.01, 0.35).is_err());
    assert!(CouncilParams::default().with_utility_thresholds(0.55, 0.35).is_err());
    assert!(CouncilParams::default().with_utility_thresholds(0.02, -0.05).is_err());
    assert!(CouncilParams::default().with_utility_thresholds(0.02, 1.05).is_err());
    assert!(CouncilParams::default().with_utility_thresholds(f64::NAN, 0.35).is_err());
    assert!(CouncilParams::default().with_utility_thresholds(0.02, f64::INFINITY).is_err());
}

#[test]
fn consejo_deliberacion_propagates_dynamic_utility_thresholds_to_teleonomia() {
    // Construir un consejo con piso de utilidad adaptativo ultra-fino (0.005) y wr (0.25)
    let custom_params = CouncilParams::default()
        .with_utility_thresholds(0.005, 0.25)
        .expect("Parámetros válidos");

    let consejo = ConsejoDeliberacion::with_params(custom_params)
        .expect("Consejo con parámetros válidos");

    let mut payload = base_payload();
    payload.dominant_tau_ms = 30_000.0;
    payload.fused_score = 0.15;
    payload.persistence = 0.20;
    payload.hurst_exponent = 0.55;
    payload.book_imbalance = 0.80;
    payload.ml_prob = 0.65;

    // Con wr = 0.28 (menor que el canónico 0.35 pero mayor que el adaptativo 0.25),
    // deliberar_traced propaga los thresholds del consejo y Teleonomia NO veta.
    let trace = consejo.deliberar_traced(&payload, 0.28, None);
    let teleonomia_op = trace
        .opinions
        .iter()
        .find(|op| op.role == SeniorRole::Teleonomia)
        .expect("Teleonomia debe estar presente");

    assert!(
        !teleonomia_op.is_veto,
        "Teleonomia no debe vetar bajo los umbrales dinámicos del consejo: {}",
        teleonomia_op.justification
    );
}
