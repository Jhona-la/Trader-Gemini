//! Contract Tests for Helmholtz-Hodge Decomposition Coupling into FlowExcitationConfluenceEngine
//!
//! # Ola Ω137 — Certificación de Acoplamiento Confluente Helmholtz-Hodge (L² = ∇φ ⊕ ∇×A ⊕ h)
//!
//! ## QUÉ
//! Suite de pruebas de contrato matemático que valida el acoplamiento entre la descomposición
//! hidrodinámica de Helmholtz-Hodge en el espacio de Hilbert L² y el motor de confluencia de flujo
//! (`FlowExcitationConfluenceEngine`).
//!
//! ## POR QUÉ
//! En microestructura real, los libros L2 sufren de spoofing y vorticidad rotacional alta donde el
//! volumen se mueve en churn sin generar impulso potencial de precio (curl ≠ 0). Al desacoplar
//! el flujo laminar (momentum) de la vorticidad (spoofing), el sistema bonifica entradas en flujos
//! laminares y amortigua la convicción ante turbulencia rotacional.
//!
//! ## PARA QUÉ
//! 1. Certificar amplificación continua C¹ cuando el flujo es laminar y seguro (Λ > 0.35, conviction > 1.0).
//! 2. Certificar atenuación suave cuando domina la enstrofía rotacional (spoofing / turbulencia, conviction < 1.0).
//! 3. Validar modulación coherente en la proyección espectral multiescala (voto_espectral_con_helmholtz).
//! 4. Garantizar inmunidad total ante NaNs e infinitos.

use omniscient_registry::OmniscientRegistry;
use signal_engine::flow_excitation_confluence::FlowExcitationConfluenceEngine;
use strategy_core::QuantumStrategy;
use std::sync::Arc;

fn build_base_long_registry() -> Arc<OmniscientRegistry> {
    let reg = Arc::new(OmniscientRegistry::new());
    reg.set("hawkes_intensity", 2.0);
    reg.set("order_book_imbalance", 0.5);
    reg.set("ml_prob_motor", 0.85);
    reg.set("ml_model_base", 0.30);
    reg.set("ml_model_lift", 0.05);
    reg.set("hawkes_excitation_gene", 0.55);
    reg.set("obi_zscore_gene", 1.0);
    reg.set("obi_p80_medido", 0.15);
    reg
}

#[test]
fn test_helmholtz_backward_compatibility_uninitialized() {
    let reg_uninit = build_base_long_registry();
    let mut eng_uninit = FlowExcitationConfluenceEngine::new();
    assert!(eng_uninit.init(reg_uninit).is_ok());
    let v_uninit = eng_uninit.evaluate();

    let reg_neutral = build_base_long_registry();
    reg_neutral.set("helmholtz_conviction", 1.0);
    let mut eng_neutral = FlowExcitationConfluenceEngine::new();
    assert!(eng_neutral.init(reg_neutral).is_ok());
    let v_neutral = eng_neutral.evaluate();

    assert!(
        (v_uninit - v_neutral).abs() < 1e-9,
        "Ausencia de helmholtz_conviction debe ser bit-a-bit idéntica a conviction neutral (1.0): {} vs {}",
        v_uninit,
        v_neutral
    );
}

#[test]
fn test_helmholtz_laminar_amplification() {
    let reg_neutral = build_base_long_registry();
    reg_neutral.set("helmholtz_conviction", 1.0);
    let mut eng_neutral = FlowExcitationConfluenceEngine::new();
    assert!(eng_neutral.init(reg_neutral).is_ok());
    let v_neutral = eng_neutral.evaluate();

    let reg_laminar = build_base_long_registry();
    reg_laminar.set("helmholtz_conviction", 1.25); // +25% por flujo laminar puro
    let mut eng_laminar = FlowExcitationConfluenceEngine::new();
    assert!(eng_laminar.init(reg_laminar).is_ok());
    let v_laminar = eng_laminar.evaluate();

    assert!(
        v_laminar > v_neutral,
        "Flujo laminar institucional debe amplificar el voto: {} > {}",
        v_laminar,
        v_neutral
    );
    let expected = (v_neutral * 1.25).min(1.0);
    assert!(
        (v_laminar - expected).abs() < 1e-6,
        "La amplificación laminar debe coincidir con el factor de convicción: {} vs {}",
        v_laminar,
        expected
    );
}

#[test]
fn test_helmholtz_turbulent_attenuation() {
    let reg_neutral = build_base_long_registry();
    reg_neutral.set("helmholtz_conviction", 1.0);
    let mut eng_neutral = FlowExcitationConfluenceEngine::new();
    assert!(eng_neutral.init(reg_neutral).is_ok());
    let v_neutral = eng_neutral.evaluate();

    let reg_turbulent = build_base_long_registry();
    reg_turbulent.set("helmholtz_conviction", 0.30); // Atenuación por alta enstrofía / spoofing
    let mut eng_turbulent = FlowExcitationConfluenceEngine::new();
    assert!(eng_turbulent.init(reg_turbulent).is_ok());
    let v_turbulent = eng_turbulent.evaluate();

    assert!(
        v_turbulent < v_neutral,
        "Alta enstrofía/spoofing debe atenuar la señal: {} < {}",
        v_turbulent,
        v_neutral
    );
    let expected = v_neutral * 0.30;
    assert!(
        (v_turbulent - expected).abs() < 1e-6,
        "La atenuación turbulenta debe coincidir exactamente con el factor: {} vs {}",
        v_turbulent,
        expected
    );
}

#[test]
fn test_spectral_vote_with_helmholtz() {
    let displacements = [0.5f64; 32];
    let lambda_ratio = 2.0;

    let v_neutral = FlowExcitationConfluenceEngine::voto_espectral_con_helmholtz(
        &displacements,
        lambda_ratio,
        0.0,
        1.0,
    );
    let v_high_laminar = FlowExcitationConfluenceEngine::voto_espectral_con_helmholtz(
        &displacements,
        lambda_ratio,
        0.0,
        1.44, // sqrt(1.44) = 1.20
    );

    let m_neutral = v_neutral.media_banda(0, 31).unwrap();
    let m_high = v_high_laminar.media_banda(0, 31).unwrap();

    assert!(
        m_high > m_neutral,
        "Mayor pureza laminar debe incrementar el consenso espectral: {} > {}",
        m_high,
        m_neutral
    );
}

#[test]
fn test_helmholtz_nan_inf_immunity() {
    let reg_nan = build_base_long_registry();
    reg_nan.set("helmholtz_conviction", f64::NAN);
    let mut eng_nan = FlowExcitationConfluenceEngine::new();
    assert!(eng_nan.init(reg_nan).is_ok());
    let v_nan = eng_nan.evaluate();
    assert!(v_nan.is_finite(), "NaN en helmholtz_conviction no debe propagarse");

    let reg_inf = build_base_long_registry();
    reg_inf.set("helmholtz_conviction", f64::INFINITY);
    let mut eng_inf = FlowExcitationConfluenceEngine::new();
    assert!(eng_inf.init(reg_inf).is_ok());
    let v_inf = eng_inf.evaluate();
    assert!(v_inf.is_finite(), "INFINITY en helmholtz_conviction no debe propagarse");
}
