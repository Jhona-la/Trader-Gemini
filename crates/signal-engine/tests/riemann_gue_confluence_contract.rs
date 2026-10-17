//! # Contrato de Integración E2E — Ola Ω140: Riemann Zeta GUE en FlowExcitationConfluenceEngine
//!
//! Certifica que la modulación de convicción por repulsión cuántica de niveles GUE
//! (`riemann_gue_conviction`) se acopla de extremo a extremo tanto en la evaluación escalar
//! (`evaluate_for_coin`) como en el consenso espectral multiescala (`voto_espectral_con_riemann_kahler_gauge_y_helmholtz`).

use omniscient_registry::OmniscientRegistry;
use signal_engine::flow_excitation_confluence::FlowExcitationConfluenceEngine;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

fn setup_base_confluence_registry() -> Arc<OmniscientRegistry> {
    let reg = Arc::new(OmniscientRegistry::new());
    reg.set("hawkes_intensity", 2.40);
    reg.set("hawkes_excitation_gene", 0.55);
    reg.set("order_book_imbalance", 0.45);
    reg.set("obi_p80_medido", 0.15);
    reg.set("obi_zscore_gene", 1.0);
    reg.set("ml_prob_motor", 0.45);
    reg.set("ml_model_base", 0.30);
    reg.set("ml_model_lift", 0.05);
    reg.set("helmholtz_conviction", 1.0);
    reg.set("yang_mills_current_absent", 1.0);
    reg.set("kahler_current_absent", 1.0);
    reg
}

#[test]
fn c1_riemann_gue_optimal_repulsion_amplifies_confluence_vote() {
    let reg = setup_base_confluence_registry();
    let mut engine = FlowExcitationConfluenceEngine::new();
    engine.init(reg.clone()).unwrap();

    // Base neutral (riemann_gue_conviction = 1.0)
    reg.set("riemann_gue_conviction", 1.0);
    let vote_neutral = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(vote_neutral > 0.10, "El voto base debe ser positivo, obtuvimos {}", vote_neutral);

    // Repulsión cuántica óptima en la línea crítica de Riemann (1.15)
    reg.set("riemann_gue_conviction", 1.15);
    let vote_gue = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(
        vote_gue > vote_neutral,
        "La repulsión óptima GUE (1.15) debe amplificar el voto ({}) sobre el neutral ({})",
        vote_gue,
        vote_neutral
    );
}

#[test]
fn c2_riemann_degenerate_level_clustering_attenuates_confluence_vote() {
    let reg = setup_base_confluence_registry();
    let mut engine = FlowExcitationConfluenceEngine::new();
    engine.init(reg.clone()).unwrap();

    reg.set("riemann_gue_conviction", 1.0);
    let vote_neutral = engine.evaluate_for_coin(0, "BTCUSDT");

    // Colapso degenerado de niveles (contagio sistémico: riemann_gue_conviction = 0.25)
    reg.set("riemann_gue_conviction", 0.25);
    let vote_degenerate = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(
        vote_degenerate < vote_neutral * 0.30,
        "El colapso degenerado GUE (0.25) debe atenuar fuertemente el voto ({}) frente al neutral ({})",
        vote_degenerate,
        vote_neutral
    );
    assert!(vote_degenerate > 0.0);
}

#[test]
fn c3_voto_espectral_32_scales_modulates_with_riemann_gue_conviction() {
    let mut desplazamientos = [0.0_f64; 32];
    for (k, val) in desplazamientos.iter_mut().enumerate() {
        *val = 0.45 + 0.01 * (k as f64);
    }

    let voto_deg =
        FlowExcitationConfluenceEngine::voto_espectral_con_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            2.5,
            0.20,
            1.0,
            0.15,
            0.15,
            0.30,
        );
    let voto_opt =
        FlowExcitationConfluenceEngine::voto_espectral_con_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            2.5,
            0.20,
            1.0,
            0.15,
            0.15,
            1.15,
        );

    for k in [2_usize, 10, 18, 28] {
        let v_deg = voto_deg.en_escala(k);
        let v_opt = voto_opt.en_escala(k);
        assert!(
            v_opt > v_deg,
            "En escala k={}, voto GUE óptimo ({}) debe superar al voto en degeneración ({})",
            k,
            v_opt,
            v_deg
        );
    }
}

#[test]
fn c4_absent_or_nan_riemann_gue_conviction_defaults_safely_to_unity() {
    let reg = setup_base_confluence_registry();
    let mut engine = FlowExcitationConfluenceEngine::new();
    engine.init(reg.clone()).unwrap();

    let vote_absent = engine.evaluate_for_coin(0, "BTCUSDT");

    reg.set("riemann_gue_conviction", f64::NAN);
    let vote_nan = engine.evaluate_for_coin(0, "BTCUSDT");

    reg.set("riemann_gue_conviction", 1.0);
    let vote_unit = engine.evaluate_for_coin(0, "BTCUSDT");

    assert!((vote_absent - vote_unit).abs() < 1e-12);
    assert!((vote_nan - vote_unit).abs() < 1e-12);
}

#[test]
fn c5_extreme_out_of_bounds_gue_conviction_is_clamped_smoothly() {
    let reg = setup_base_confluence_registry();
    let mut engine = FlowExcitationConfluenceEngine::new();
    engine.init(reg.clone()).unwrap();

    reg.set("riemann_gue_conviction", 999.0);
    let vote_huge = engine.evaluate_for_coin(0, "BTCUSDT");

    reg.set("riemann_gue_conviction", 1.15);
    let vote_cap = engine.evaluate_for_coin(0, "BTCUSDT");

    assert!(
        (vote_huge - vote_cap).abs() < 1e-12,
        "Valores extremos por encima de 1.15 deben acotarse exactamente a 1.15: {} vs {}",
        vote_huge,
        vote_cap
    );
}
