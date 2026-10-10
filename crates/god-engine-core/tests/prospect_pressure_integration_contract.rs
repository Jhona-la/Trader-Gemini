use metacortex_engine::consejo_seniors::{ConsejoDeliberacion, MarketSnapshotPayload, TradingHorizon};
use metacortex_engine::prospect_theory::ProspectTheoryEngine;
use signal_engine::feynman_propagator::FeynmanPropagatorEngine;

#[test]
fn test_prospect_pressure_live_computation_in_god_engine() {
    let kt_engine = ProspectTheoryEngine::new();

    // 1. Simulación de estado vivo de mercado: p_bull moderado, liquidaciones bajas, ATR 12 bps
    let p_bull = 0.62;
    let p_crash = 0.05;
    let delta_pts = 1.2; // 1.2% = 120 bps
    let p_kt = kt_engine.compute_prospect_pressure(p_bull, p_crash, delta_pts, delta_pts);

    assert!(p_kt.is_finite());
    assert!((-50.0..=50.0).contains(&p_kt));

    // 2. Construcción de payload del consejo tal como se ejecuta en god-engine-core
    let payload = MarketSnapshotPayload {
        horizon: TradingHorizon::Continuous,
        book_imbalance: 0.35,
        hurst_exponent: 0.65,
        ml_prob: p_bull,
        fused_score: 0.40,
        persistence: 0.50,
        atr_pct: 0.0012,
        loss_streak: 0,
        intended_direction: 1.0,
        do_calculus_risk: 0.10,
        causal_veto_threshold: 0.75,
        current_drawdown_pct: 0.02,
        estimated_slippage_bps: 15.0,
        dominant_tau_ms: 120_000.0,
        whale_burst_z: 0.5,
        liquidation_severity: p_crash,
        open_interest_norm: 0.4,
        spoof_score: 0.0,
        crowd_ls_ratio: 1.2,
        crowd_taker_ratio: 1.1,
        ml_model_base: 0.5,
        hodge_curl_share: 0.1,
        yang_mills_current: 0.05,
        macro_staleness_ms: 5_000,
        navier_reynolds_number: 0.8,
        navier_laminar_share: 0.6,
        prospect_pressure: p_kt,
    };

    assert!(payload.validate().is_ok());

    let consejo = ConsejoDeliberacion::new();
    let decision = consejo.deliberar_traced(&payload, 0.65, None);

    assert!(decision.consensus.approved, "Trade con evidencia sana debe ser aprobado");
    assert!(decision.consensus.final_signal > 0.0);
    assert!(decision.consensus.vetoed_by.is_none());
}

#[test]
fn test_feynman_and_prospect_confluence_in_god_engine() {
    // 1. Feynman path integral con 32 escalas de Hilbert
    let displacements = [0.85; 32];
    let voto = FeynmanPropagatorEngine::voto_espectral(
        &displacements,
        1.0,
        1.0,
        1.0,
    );
    let (k_dom, v_dom) = voto.dominante().expect("debe existir dominante");
    assert!(v_dom > 0.20, "Voto dominante coherente debe ser significativo: {v_dom}");
    assert!(k_dom < 32);

    // 2. Prospect Theory en régimen de capitulación (comprar en pánico minorista)
    let kt = ProspectTheoryEngine::new();
    let panic_pressure = kt.compute_prospect_pressure(0.10, 0.50, 0.5, 3.5);
    assert!(panic_pressure < -1.0, "Capitulación debe producir presión negativa profunda: {panic_pressure}");

    let mod_factor = kt.modulation_factor(1.0, panic_pressure);
    assert!(mod_factor > 1.0 && mod_factor <= 1.30, "Convicción contrarian debe amplificarse: {mod_factor}");

    // 3. Deliberación integrada: coherencia cuántica + capitulación conductual
    let payload = MarketSnapshotPayload {
        horizon: TradingHorizon::Continuous,
        book_imbalance: 0.60,
        hurst_exponent: 0.70,
        ml_prob: 0.68,
        fused_score: 0.55,
        persistence: 0.65,
        atr_pct: 0.0015,
        loss_streak: 0,
        intended_direction: 1.0,
        do_calculus_risk: 0.15,
        causal_veto_threshold: 0.80,
        current_drawdown_pct: 0.03,
        estimated_slippage_bps: 20.0,
        dominant_tau_ms: 180_000.0,
        whale_burst_z: 1.5,
        liquidation_severity: 0.20,
        open_interest_norm: 0.5,
        spoof_score: 0.0,
        crowd_ls_ratio: 0.8,
        crowd_taker_ratio: 0.9,
        ml_model_base: 0.5,
        hodge_curl_share: 0.05,
        yang_mills_current: 0.0,
        macro_staleness_ms: 2_000,
        navier_reynolds_number: 0.4,
        navier_laminar_share: 0.85,
        prospect_pressure: panic_pressure,
    };

    assert!(payload.validate().is_ok());

    let consejo = ConsejoDeliberacion::new();
    let trace = consejo.deliberar_traced(&payload, 0.70, None);

    assert!(trace.consensus.approved);
    assert!(trace.consensus.final_signal > 0.0);
    assert!(trace.consensus.vetoed_by.is_none());
}

#[test]
fn test_prospect_pressure_mirror_symmetry_in_god_engine() {
    let kt = ProspectTheoryEngine::new();

    // Verificamos que para cualquier ratio Long/Short de la masa y severidad de liquidación,
    // el fallback analítico de prospect_pressure en god_engine es 100% anti-simétrico (C-10)
    for &ls in &[0.5, 0.8, 1.0, 1.25, 2.0, 4.0] {
        let p_up = kt.compute_crowd_net_prospect_pressure(ls, 0.25, 1.5);
        let p_down = kt.compute_crowd_net_prospect_pressure(1.0 / ls, 0.25, 1.5);
        assert!(
            (p_up + p_down).abs() < 1e-10,
            "C-10: Fallback analítico debe ser exactamente anti-simétrico: p_up={}, p_down={}",
            p_up,
            p_down
        );

        // La modulación contrarian para Long con euforia (+P) debe ser idéntica
        // a la modulación contrarian para Short con pánico (-P)
        let mod_long = kt.modulation_factor(1.0, p_up);
        let mod_short = kt.modulation_factor(-1.0, p_down);
        assert!(
            (mod_long - mod_short).abs() < 1e-10,
            "C-10: Modulación contrarian debe ser simétrica para Long y Short bajo inversión de sentimiento",
        );
    }
}

