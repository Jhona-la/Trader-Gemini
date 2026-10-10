use metacortex_engine::consejo_seniors::{
    ConsejoDeliberacion, MarketSnapshotPayload, SeniorAgent, SeniorEnteMercado, TradingHorizon,
};
use metacortex_engine::prospect_theory::ProspectTheoryEngine;

#[test]
fn test_kahneman_tversky_value_function_asymmetry() {
    let engine = ProspectTheoryEngine::new();

    // 1. Ganancia unitaria V(1.0) = 1.0^0.88 = 1.0
    let v_gain = engine.value(1.0);
    assert!((v_gain - 1.0).abs() < 1e-6, "V(1.0) debe ser exactamente 1.0");

    // 2. Pérdida unitaria V(-1.0) = -2.25 * 1.0^0.88 = -2.25
    let v_loss = engine.value(-1.0);
    assert!((v_loss - (-2.25)).abs() < 1e-6, "V(-1.0) debe ser exactamente -2.25");

    // 3. Ratio de aversión a la pérdida: |V(-x)| / V(x) == 2.25 para cualquier x > 0
    for &x in &[0.5, 1.0, 2.0, 5.0, 10.0] {
        let g = engine.value(x);
        let l = engine.value(-x);
        let ratio = (-l) / g;
        assert!(
            (ratio - 2.25).abs() < 1e-5,
            "Ratio de aversión a pérdida en x={x} debe ser 2.25, obtenido {ratio}"
        );
    }

    // 4. Punto neutral V(0.0) == 0.0
    assert_eq!(engine.value(0.0), 0.0);
}

#[test]
fn test_prelec_probability_weighting_curvature() {
    let engine = ProspectTheoryEngine::new();

    // 1. Condiciones de frontera
    assert_eq!(engine.probability_weight(0.0), 0.0);
    assert_eq!(engine.probability_weight(1.0), 1.0);

    // 2. Sobreponderación de probabilidades pequeñas (miedo a la cola negra / efecto lotería)
    let w_rare = engine.probability_weight(0.02);
    assert!(
        w_rare > 0.02,
        "Probabilidad pequeña p=0.02 debe sobreponderarse w(0.02)={w_rare} > 0.02"
    );

    let w_crash = engine.probability_weight(0.05);
    assert!(
        w_crash > 0.05,
        "Probabilidad p=0.05 debe sobreponderarse w(0.05)={w_crash} > 0.05"
    );

    // 3. Subponderación de probabilidades moderadas/altas (complacencia tardía)
    let w_high = engine.probability_weight(0.80);
    assert!(
        w_high < 0.80,
        "Probabilidad alta p=0.80 debe subponderarse w(0.80)={w_high} < 0.80"
    );

    // 4. Monotonía estricta
    let mut prev_w = -0.01;
    for step in 0..=100 {
        let p = step as f64 * 0.01;
        let w = engine.probability_weight(p);
        assert!(w >= prev_w, "w(p) debe ser monótona no decreciente en p={p}");
        assert!((0.0..=1.0).contains(&w), "w(p) debe estar acotada en [0, 1]");
        prev_w = w;
    }
}

#[test]
fn test_prospect_pressure_computation() {
    let engine = ProspectTheoryEngine::new();

    // 1. Mercado en pánico extremo (alta probabilidad de crash o dolor masivo)
    let panic_pressure = engine.compute_prospect_pressure(0.05, 0.40, 1.0, 3.0);
    assert!(
        panic_pressure < -1.0,
        "Pánico extremo debe generar presión negativa pronunciada: {panic_pressure}"
    );

    // 2. Mercado en euforia FOMO (alta probabilidad de ganancia, complacencia ante riesgo)
    let fomo_pressure = engine.compute_prospect_pressure(0.80, 0.02, 3.0, 0.5);
    assert!(
        fomo_pressure > 1.0,
        "FOMO extremo debe generar presión positiva pronunciada: {fomo_pressure}"
    );

    // 3. Simetría de magnitudes revela la aversión a la pérdida inherente (λ = 2.25)
    let neutral_odds = engine.compute_prospect_pressure(0.50, 0.50, 1.0, 1.0);
    assert!(
        neutral_odds < 0.0,
        "A probabilidades y deltas iguales, la aversión a la pérdida genera presión psicológica neta negativa: {neutral_odds}"
    );
}

#[test]
fn test_contrarian_modulation_factor() {
    let engine = ProspectTheoryEngine::new();

    // 1. Presión neutral P = 0.0 produce factor exactamente 1.0
    let f_neutral = engine.modulation_factor(1.0, 0.0);
    assert!((f_neutral - 1.0).abs() < 1e-9, "P=0 debe dar modulación neutral 1.0");

    // 2. Smart Contrarian Long: la multitud en pánico (P = -5.0) y el sistema comprando (dir = +1.0)
    // Amplifica convicción: compras en capitulación minorista
    let f_buy_blood = engine.modulation_factor(1.0, -5.0);
    assert!(
        f_buy_blood > 1.0 && f_buy_blood <= 1.30,
        "Comprar en capitulación debe amplificar convicción contrarian [1.0, 1.30]: {f_buy_blood}"
    );

    // 3. Chasing FOMO Long: la masa eufórica (P = +5.0) y el sistema queriendo comprar (dir = +1.0)
    // Atenúa convicción: no comprar en el techo
    let f_chase_fomo = engine.modulation_factor(1.0, 5.0);
    assert!(
        f_chase_fomo >= 0.50 && f_chase_fomo < 1.0,
        "Comprar en euforia retail debe amortiguar convicción defensivamente [0.50, 1.0]: {f_chase_fomo}"
    );

    // 4. Smart Contrarian Short: la multitud en FOMO (P = +5.0) y el sistema vendiendo en corto (dir = -1.0)
    // Amplifica convicción: shortear el techo eufórico
    let f_short_euphoria = engine.modulation_factor(-1.0, 5.0);
    assert!(
        f_short_euphoria > 1.0 && f_short_euphoria <= 1.30,
        "Shortear en euforia debe amplificar convicción [1.0, 1.30]: {f_short_euphoria}"
    );
}

#[test]
fn test_fail_closed_robustness_nan_inf() {
    let engine = ProspectTheoryEngine::new();

    // Valores no finitos en value retornan 0.0 determinista fail-closed
    assert_eq!(engine.value(f64::NAN), 0.0);
    assert_eq!(engine.value(f64::INFINITY), 0.0);
    assert_eq!(engine.value(f64::NEG_INFINITY), 0.0);

    // Valores no finitos en probability_weight retornan 0.0 determinista fail-closed
    assert_eq!(engine.probability_weight(f64::NAN), 0.0);
    assert_eq!(engine.probability_weight(f64::INFINITY), 0.0);
    assert_eq!(engine.probability_weight(f64::NEG_INFINITY), 0.0);

    // Modulación con NaN/Inf es neutral fail-closed (1.0)
    assert_eq!(engine.modulation_factor(f64::NAN, -5.0), 1.0);
    assert_eq!(engine.modulation_factor(1.0, f64::NAN), 1.0);
    assert_eq!(engine.modulation_factor(f64::INFINITY, 2.0), 1.0);
}

#[test]
fn test_consejo_deliberacion_prospect_theory_integration() {
    let base_payload = MarketSnapshotPayload {
        horizon: TradingHorizon::Continuous,
        book_imbalance: 0.85,
        hurst_exponent: 0.72,
        ml_prob: 0.70,
        fused_score: 0.50,
        persistence: 0.60,
        atr_pct: 0.0010,
        loss_streak: 0,
        intended_direction: 1.0,
        do_calculus_risk: 0.5,
        causal_veto_threshold: 0.80,
        current_drawdown_pct: 0.05,
        estimated_slippage_bps: 20.0,
        dominant_tau_ms: 1_138_000.0,
        whale_burst_z: 2.0, // Fondo de actividad institucional: base_confidence < 1.0
        liquidation_severity: 0.0,
        open_interest_norm: 0.5,
        spoof_score: 0.0,
        crowd_ls_ratio: 1.0,
        crowd_taker_ratio: 1.0,
        ml_model_base: 0.5,
        hodge_curl_share: 0.0,
        yang_mills_current: 0.0,
        macro_staleness_ms: 0,
        navier_reynolds_number: 0.0,
        navier_laminar_share: 1.0,
        prospect_pressure: 0.0,
    };

    let ente = SeniorEnteMercado;

    // Caso 1: Neutral
    let op_neutral = ente.evaluate(&base_payload, 0.5);

    // Caso 2: Pánico minorista (capitulación), sistema comprando
    let mut panic_payload = base_payload.clone();
    panic_payload.prospect_pressure = -4.0;
    let op_panic = ente.evaluate(&panic_payload, 0.5);

    // Caso 3: Euforia minorista en el techo, sistema queriendo comprar
    let mut fomo_payload = base_payload.clone();
    fomo_payload.prospect_pressure = 4.0;
    let op_fomo = ente.evaluate(&fomo_payload, 0.5);

    // La convicción ante capitulación contrarian debe superar a la de euforia retail
    assert!(
        op_panic.confidence > op_neutral.confidence,
        "Capitulación retail debe amplificar convicción contrarian: panic={:.3} > neutral={:.3}",
        op_panic.confidence,
        op_neutral.confidence
    );
    assert!(
        op_neutral.confidence > op_fomo.confidence,
        "Euforia retail en techo debe amortiguar convicción de compra: neutral={:.3} > fomo={:.3}",
        op_neutral.confidence,
        op_fomo.confidence
    );

    // Deliberación completa del consejo aprueba con mayor fuerza el trade contrarian
    let consejo = ConsejoDeliberacion::new();
    let res_panic = consejo.deliberar(&panic_payload, 0.70);
    assert!(res_panic.approved, "Trade contrarian en pánico debe ser aprobado");
    assert!(res_panic.final_signal > 0.0);
}
