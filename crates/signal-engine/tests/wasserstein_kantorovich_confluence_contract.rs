//! Formal Contract Tests for Ola Ω143: Wasserstein-Kantorovich W_2 Optimal Transport
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
fn contract_wasserstein_confluence_fracture_attenuation_vs_geodesic_amplification() {
    let reg_fracture = build_base_registry();
    reg_fracture.set("wasserstein_conviction_modulation", 0.30);
    let mut eng_frac = FlowExcitationConfluenceEngine::new();
    eng_frac.init(reg_fracture).unwrap();
    let vote_fracture = eng_frac.evaluate_for_coin(0, "");

    let reg_neutral = build_base_registry();
    reg_neutral.set("wasserstein_conviction_modulation", 1.00);
    let mut eng_neut = FlowExcitationConfluenceEngine::new();
    eng_neut.init(reg_neutral).unwrap();
    let vote_neutral = eng_neut.evaluate_for_coin(0, "");

    let reg_geodesic = build_base_registry();
    reg_geodesic.set("wasserstein_conviction_modulation", 1.15);
    let mut eng_geo = FlowExcitationConfluenceEngine::new();
    eng_geo.init(reg_geodesic).unwrap();
    let vote_geodesic = eng_geo.evaluate_for_coin(0, "");

    assert!(vote_fracture > 0.0);
    assert!(
        vote_fracture < vote_neutral * 0.35,
        "Regime fracture Wasserstein conviction (0.30) must strongly attenuate vote: frac={} vs neutral={}",
        vote_fracture,
        vote_neutral
    );
    assert!(
        vote_geodesic > vote_neutral * 1.10,
        "Geodesic Wasserstein conviction (1.15) must amplify vote: geo={} vs neutral={}",
        vote_geodesic,
        vote_neutral
    );
}

#[test]
fn contract_wasserstein_barycenter_velocity_directional_alignment() {
    let build_mod_reg = |wk_vel: f64| {
        let reg = build_base_registry();
        reg.set("wasserstein_conviction_modulation", 1.0);
        reg.set("wasserstein_barycenter_velocity", wk_vel);
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
        "Aligned Benamou-Brenier barycenter velocity must boost confluence: aligned={} vs zero={}",
        v_aligned,
        v_zero
    );
    assert!(
        v_opposed < v_zero * 0.50,
        "Opposed Benamou-Brenier barycenter velocity must smoothly attenuate confluence: opposed={} vs zero={}",
        v_opposed,
        v_zero
    );
}

#[test]
fn contract_wasserstein_voto_espectral_multiescala() {
    let mut desplazamientos = [0.0f64; 32];
    for k in 0..32 {
        desplazamientos[k] = 0.35;
    }

    let ve_fracture =
        FlowExcitationConfluenceEngine::voto_espectral_con_wasserstein_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            2.0,
            0.2,
            1.0,
            0.1,
            0.1,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            -0.5,
            0.30,
        );
    let ve_geodesic =
        FlowExcitationConfluenceEngine::voto_espectral_con_wasserstein_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            2.0,
            0.2,
            1.0,
            0.1,
            0.1,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.5,
            1.15,
        );

    for k in [8usize, 16, 24, 30] {
        let f_val = ve_fracture.en_escala(k);
        let g_val = ve_geodesic.en_escala(k);
        assert!(f_val.is_finite() && g_val.is_finite());
        assert!(
            g_val > f_val * 2.0,
            "Scale {} geodesic Wasserstein vote ({}) must exceed fracture Wasserstein vote ({})",
            k,
            g_val,
            f_val
        );
    }
}

#[test]
fn contract_wasserstein_confluence_nan_immunity() {
    let reg = build_base_registry();
    reg.set("wasserstein_conviction_modulation", f64::NAN);
    reg.set("wasserstein_barycenter_velocity", f64::INFINITY);
    let mut eng = FlowExcitationConfluenceEngine::new();
    eng.init(reg).unwrap();
    let v = eng.evaluate_for_coin(0, "");
    assert!(v.is_finite() && v > 0.0);
}

#[test]
fn contract_wasserstein_backward_compatibility_delegation() {
    let mut desplazamientos = [0.25f64; 32];
    desplazamientos[16] = 0.45;
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
            0.1,
            1.05,
        );
    let v_wk =
        FlowExcitationConfluenceEngine::voto_espectral_con_wasserstein_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            1.8,
            0.15,
            1.0,
            0.2,
            0.1,
            1.05,
            0.1,
            1.05,
            0.1,
            1.05,
            0.0,
            1.0,
        );
    for k in 0..32 {
        assert!((v_dirac.en_escala(k) - v_wk.en_escala(k)).abs() < 1e-12);
    }
}
