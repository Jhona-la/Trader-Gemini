//! Formal Contract Tests for Ola Ω141: Birch and Swinnerton-Dyer Elliptic Curve Confluence Modulation
//! in `FlowExcitationConfluenceEngine`.

use omniscient_registry::OmniscientRegistry;
use signal_engine::flow_excitation_confluence::FlowExcitationConfluenceEngine;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

fn build_base_registry() -> Arc<OmniscientRegistry> {
    let reg = Arc::new(OmniscientRegistry::new());
    reg.set("hawkes_intensity", 2.1);
    reg.set("order_book_imbalance", 0.65);
    reg.set("ml_prob_motor", 0.82);
    reg.set("ml_model_base", 0.30);
    reg
}

#[test]
fn contract_bsd_confluence_singular_cusp_attenuation_vs_rank_amplification() {
    let reg_singular = build_base_registry();
    reg_singular.set("bsd_conviction_modulation", 0.30);
    let mut eng_sing = FlowExcitationConfluenceEngine::new();
    eng_sing.init(reg_singular).unwrap();
    let vote_singular = eng_sing.evaluate_for_coin(0, "");

    let reg_neutral = build_base_registry();
    reg_neutral.set("bsd_conviction_modulation", 1.00);
    let mut eng_neut = FlowExcitationConfluenceEngine::new();
    eng_neut.init(reg_neutral).unwrap();
    let vote_neutral = eng_neut.evaluate_for_coin(0, "");

    let reg_high_rank = build_base_registry();
    reg_high_rank.set("bsd_conviction_modulation", 1.15);
    let mut eng_high = FlowExcitationConfluenceEngine::new();
    eng_high.init(reg_high_rank).unwrap();
    let vote_high_rank = eng_high.evaluate_for_coin(0, "");

    assert!(vote_singular > 0.0);
    assert!(
        vote_singular < vote_neutral * 0.35,
        "Singular cusp BSD conviction (0.30) must strongly attenuate vote: sing={} vs neutral={}",
        vote_singular,
        vote_neutral
    );
    assert!(
        vote_high_rank >= vote_neutral,
        "High analytic rank BSD conviction (1.15) must amplify or saturate vote: high={} vs neutral={}",
        vote_high_rank,
        vote_neutral
    );
}

#[test]
fn contract_bsd_restoring_current_directional_alignment() {
    // Use moderate base vote so saturation at 1.0 doesn't mask the +12% directional bonus
    let build_mod_reg = |bsd_curr: f64| {
        let reg = Arc::new(OmniscientRegistry::new());
        reg.set("hawkes_intensity", 2.1);
        reg.set("order_book_imbalance", 0.45);
        reg.set("ml_prob_motor", 0.50);
        reg.set("ml_model_base", 0.30);
        reg.set("bsd_conviction_modulation", 1.0);
        reg.set("bsd_restoring_current", bsd_curr);
        reg
    };

    let mut eng_aligned = FlowExcitationConfluenceEngine::new();
    eng_aligned.init(build_mod_reg(0.85)).unwrap();
    let v_aligned = eng_aligned.evaluate_for_coin(0, "");

    let mut eng_zero = FlowExcitationConfluenceEngine::new();
    eng_zero.init(build_mod_reg(0.0)).unwrap();
    let v_zero = eng_zero.evaluate_for_coin(0, "");

    let mut eng_opposed = FlowExcitationConfluenceEngine::new();
    eng_opposed.init(build_mod_reg(-0.85)).unwrap();
    let v_opposed = eng_opposed.evaluate_for_coin(0, "");

    assert!(
        v_aligned > v_zero,
        "Aligned BSD elliptic restoring current must boost confluence: aligned={} vs zero={}",
        v_aligned,
        v_zero
    );
    assert!(
        v_opposed < v_zero * 0.50,
        "Opposed BSD elliptic restoring current must smoothly attenuate confluence: opposed={} vs zero={}",
        v_opposed,
        v_zero
    );
}

#[test]
fn contract_bsd_voto_espectral_multiescala() {
    let mut desplazamientos = [0.0f64; 32];
    for k in 0..32 {
        desplazamientos[k] = 0.35;
    }

    let ve_singular =
        FlowExcitationConfluenceEngine::voto_espectral_con_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            2.0,
            0.2,
            1.0,
            0.1,
            0.1,
            1.0,
            -0.5,
            0.30,
        );
    let ve_high_rank =
        FlowExcitationConfluenceEngine::voto_espectral_con_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            2.0,
            0.2,
            1.0,
            0.1,
            0.1,
            1.0,
            0.5,
            1.15,
        );

    for k in [4usize, 12, 16, 24] {
        let s_val = ve_singular.en_escala(k);
        let h_val = ve_high_rank.en_escala(k);
        assert!(s_val.is_finite() && h_val.is_finite());
        assert!(
            h_val > s_val * 2.0,
            "Scale {} high-rank BSD vote ({}) must exceed singular BSD vote ({})",
            k,
            h_val,
            s_val
        );
    }
}

#[test]
fn contract_bsd_confluence_nan_immunity() {
    let reg = build_base_registry();
    reg.set("bsd_conviction_modulation", f64::NAN);
    reg.set("bsd_restoring_current", f64::INFINITY);
    let mut eng = FlowExcitationConfluenceEngine::new();
    eng.init(reg).unwrap();
    let v = eng.evaluate_for_coin(0, "");
    assert!(v.is_finite() && v > 0.0);
}

#[test]
fn contract_bsd_backward_compatibility_delegation() {
    let mut desplazamientos = [0.25f64; 32];
    desplazamientos[16] = 0.45;
    let v_old =
        FlowExcitationConfluenceEngine::voto_espectral_con_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            1.8,
            0.15,
            1.0,
            0.2,
            0.1,
            1.05,
        );
    let v_new =
        FlowExcitationConfluenceEngine::voto_espectral_con_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            1.8,
            0.15,
            1.0,
            0.2,
            0.1,
            1.05,
            0.0,
            1.0,
        );
    for k in 0..32 {
        assert!((v_old.en_escala(k) - v_new.en_escala(k)).abs() < 1e-12);
    }
}
