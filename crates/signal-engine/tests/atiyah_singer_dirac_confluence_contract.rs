//! Formal Contract Tests for Ola Ω142: Atiyah-Singer Index Theorem & Chiral Dirac Zero Modes
//! Confluence Modulation in `FlowExcitationConfluenceEngine`.

use omniscient_registry::OmniscientRegistry;
use signal_engine::flow_excitation_confluence::FlowExcitationConfluenceEngine;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

fn build_base_registry() -> Arc<OmniscientRegistry> {
    let reg = Arc::new(OmniscientRegistry::new());
    reg.set("hawkes_intensity", 2.1);
    reg.set("order_book_imbalance", 0.45);
    reg.set("ml_prob_motor", 0.50);
    reg.set("ml_model_base", 0.30);
    reg
}

#[test]
fn contract_dirac_confluence_whipsaw_attenuation_vs_chiral_amplification() {
    let reg_whipsaw = build_base_registry();
    reg_whipsaw.set("dirac_conviction_modulation", 0.30);
    let mut eng_whip = FlowExcitationConfluenceEngine::new();
    eng_whip.init(reg_whipsaw).unwrap();
    let vote_whipsaw = eng_whip.evaluate_for_coin(0, "");

    let reg_neutral = build_base_registry();
    reg_neutral.set("dirac_conviction_modulation", 1.00);
    let mut eng_neut = FlowExcitationConfluenceEngine::new();
    eng_neut.init(reg_neutral).unwrap();
    let vote_neutral = eng_neut.evaluate_for_coin(0, "");

    let reg_chiral = build_base_registry();
    reg_chiral.set("dirac_conviction_modulation", 1.15);
    let mut eng_chir = FlowExcitationConfluenceEngine::new();
    eng_chir.init(reg_chiral).unwrap();
    let vote_chiral = eng_chir.evaluate_for_coin(0, "");

    assert!(vote_whipsaw > 0.0);
    assert!(
        vote_whipsaw < vote_neutral * 0.35,
        "Whipsaw Dirac conviction (0.30) must strongly attenuate vote: whip={} vs neutral={}",
        vote_whipsaw,
        vote_neutral
    );
    assert!(
        vote_chiral > vote_neutral * 1.10,
        "Chiral zero-mode Dirac conviction (1.15) must amplify vote: chiral={} vs neutral={}",
        vote_chiral,
        vote_neutral
    );
}

#[test]
fn contract_dirac_chiral_current_directional_alignment() {
    let build_mod_reg = |dirac_curr: f64| {
        let reg = build_base_registry();
        reg.set("dirac_conviction_modulation", 1.0);
        reg.set("dirac_chiral_current", dirac_curr);
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
        "Aligned Adler-Bell-Jackiw chiral current must boost confluence: aligned={} vs zero={}",
        v_aligned,
        v_zero
    );
    assert!(
        v_opposed < v_zero * 0.50,
        "Opposed Adler-Bell-Jackiw chiral current must smoothly attenuate confluence: opposed={} vs zero={}",
        v_opposed,
        v_zero
    );
}

#[test]
fn contract_dirac_voto_espectral_multiescala() {
    let mut desplazamientos = [0.0f64; 32];
    for k in 0..32 {
        desplazamientos[k] = 0.35;
    }

    let ve_whipsaw =
        FlowExcitationConfluenceEngine::voto_espectral_con_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            2.0,
            0.2,
            1.0,
            0.1,
            0.1,
            1.0,
            0.0,
            1.0,
            -0.5,
            0.30,
        );
    let ve_chiral =
        FlowExcitationConfluenceEngine::voto_espectral_con_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            2.0,
            0.2,
            1.0,
            0.1,
            0.1,
            1.0,
            0.0,
            1.0,
            0.5,
            1.15,
        );

    for k in [2usize, 8, 16, 24] {
        let w_val = ve_whipsaw.en_escala(k);
        let c_val = ve_chiral.en_escala(k);
        assert!(w_val.is_finite() && c_val.is_finite());
        assert!(
            c_val > w_val * 2.0,
            "Scale {} chiral Dirac vote ({}) must exceed whipsaw Dirac vote ({})",
            k,
            c_val,
            w_val
        );
    }
}

#[test]
fn contract_dirac_confluence_nan_immunity() {
    let reg = build_base_registry();
    reg.set("dirac_conviction_modulation", f64::NAN);
    reg.set("dirac_chiral_current", f64::INFINITY);
    let mut eng = FlowExcitationConfluenceEngine::new();
    eng.init(reg).unwrap();
    let v = eng.evaluate_for_coin(0, "");
    assert!(v.is_finite() && v > 0.0);
}

#[test]
fn contract_dirac_backward_compatibility_delegation() {
    let mut desplazamientos = [0.25f64; 32];
    desplazamientos[8] = 0.45;
    let v_bsd =
        FlowExcitationConfluenceEngine::voto_espectral_con_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            1.8,
            0.15,
            1.0,
            0.2,
            0.1,
            1.05,
            0.1,
            1.05,
        );
    let v_dirac =
        FlowExcitationConfluenceEngine::voto_espectral_con_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            1.8,
            0.15,
            1.0,
            0.2,
            0.1,
            1.05,
            0.1,
            1.05,
            0.0,
            1.0,
        );
    for k in 0..32 {
        assert!((v_bsd.en_escala(k) - v_dirac.en_escala(k)).abs() < 1e-12);
    }
}
