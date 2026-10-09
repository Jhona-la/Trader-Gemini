use metacortex_engine::consejo_seniors::{
    MarketSnapshotPayload, SeniorAgent, SeniorEnteMercado, TradingHorizon,
};

fn base_payload() -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        horizon: TradingHorizon::Continuous,
        book_imbalance: 0.2,
        hurst_exponent: 0.55,
        ml_prob: 0.60,
        fused_score: 0.40,
        persistence: 0.50,
        atr_pct: 0.0012,
        loss_streak: 0,
        intended_direction: 1.0,
        do_calculus_risk: 0.05,
        causal_veto_threshold: 0.80,
        current_drawdown_pct: 0.01,
        estimated_slippage_bps: 0.0005,
        dominant_tau_ms: 60_000.0,
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
fn ente_mercado_modulacion_whale_burst_es_continua_c1_y_monotona() {
    let ente = SeniorEnteMercado;
    let mut prev_conf: f64 = 1.01;

    // Barrido fino de z-score de ballena de 0.0 a 10.0 en pasos de 0.1
    for step in 0..=100 {
        let z = step as f64 * 0.1;
        let mut p = base_payload();
        p.whale_burst_z = z;

        let op = ente.evaluate(&p, 0.5);
        assert!(op.confidence.is_finite(), "confidence must be finite at z={z}");
        assert!(
            op.confidence <= prev_conf + 1e-12,
            "confidence must be monotonically non-increasing: z={z}, prev={prev_conf}, curr={}",
            op.confidence
        );

        // Sin saltos escalonados abruptos: la variación por paso de 0.1 debe ser suave
        if step > 0 {
            let delta: f64 = (prev_conf - op.confidence).abs();
            assert!(
                delta < 0.03,
                "step change at z={z} is too abrupt: delta={delta}"
            );
        }

        prev_conf = op.confidence;
    }

    assert!(prev_conf >= 0.30, "floor must be respected");
}

#[test]
fn ente_mercado_crowd_squeeze_modula_suavemente_en_ambas_direcciones() {
    let ente = SeniorEnteMercado;

    // 1. Long entrando contra multitud acumulada en Long
    let mut prev_conf_long: f64 = 1.01;
    for step in 10..=50 {
        let ls = step as f64 * 0.1; // 1.0 a 5.0
        let mut p = base_payload();
        p.intended_direction = 1.0;
        p.crowd_ls_ratio = ls;

        let op = ente.evaluate(&p, 0.5);
        assert!(op.confidence <= prev_conf_long + 1e-12, "long crowd must reduce confidence smoothly");
        prev_conf_long = op.confidence;
    }

    // 2. Short entrando contra multitud acumulada en Short
    let mut prev_conf_short: f64 = 1.01;
    for step in 1..=10 {
        let ls = step as f64 * 0.1; // 0.1 a 1.0 (de menor a mayor concentración contraria)
        let mut p = base_payload();
        p.intended_direction = -1.0;
        p.crowd_ls_ratio = ls; // ls < 1.0 es multitud en Short

        let op = ente.evaluate(&p, 0.5);
        // Cuando ls se acerca a 1.0, el desbalance disminuye y la confianza sube
        if step > 1 {
            assert!(op.confidence >= prev_conf_short - 1e-12, "approaching balance must restore confidence");
        }
        prev_conf_short = op.confidence;
    }
}

#[test]
fn ente_mercado_taker_exhaustion_modula_suavemente() {
    let ente = SeniorEnteMercado;
    let mut prev_conf: f64 = 1.01;

    for step in 10..=40 {
        let tk = step as f64 * 0.1; // 1.0 a 4.0
        let mut p = base_payload();
        p.intended_direction = 1.0;
        p.crowd_taker_ratio = tk;

        let op = ente.evaluate(&p, 0.5);
        assert!(op.confidence <= prev_conf + 1e-12, "exhausted taker flow must smoothly decay conviction");
        prev_conf = op.confidence;
    }
}

#[test]
fn ente_mercado_modulacion_macro_staleness_continua_y_monotona() {
    let ente = SeniorEnteMercado;
    let mut prev_conf: f64 = 1.01;

    // Con staleness < 180s (180_000 ms), no debe haber penalización (p_macro = 1.0)
    let mut p_fresh = base_payload();
    p_fresh.macro_staleness_ms = 60_000;
    let conf_fresh = ente.evaluate(&p_fresh, 0.6).confidence;
    assert_eq!(conf_fresh, 1.0);

    // De 180s a 1800s, el factor p_macro debe decaer monótonamente y de forma continua
    for step in 0..=50 {
        let stale_ms = 180_000 + step * 20_000;
        let mut p = base_payload();
        p.macro_staleness_ms = stale_ms;
        let conf = ente.evaluate(&p, 0.6).confidence;
        assert!(
            conf <= prev_conf + 1e-12,
            "Staleness debe ser monótonamente no creciente: step={} conf={} prev={}",
            step,
            conf,
            prev_conf
        );
        assert!(
            conf >= 0.40,
            "No debe caer por debajo del suelo de amortiguamiento 0.40"
        );
        prev_conf = conf;
    }
}
