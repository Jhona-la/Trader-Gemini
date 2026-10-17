//! Contract Tests for Kyle-Hasbrouck Hidden Liquidity Tensorial Coupling into FlowExcitationConfluenceEngine
//!
//! # Ola Ω133 — Certificación de Acoplamiento Tensorial de Liquidez Oculta
//!
//! ## QUÉ
//! Suite de pruebas de contrato matemático que valida el acoplamiento tensorial entre la absorción
//! de bloques institucionales (órdenes iceberg) y el motor de confluencia de flujo y excitación
//! (`FlowExcitationConfluenceEngine`).
//!
//! ## POR QUÉ
//! En microestructura real, los libros L2 visibles son objeto de spoofing y no revelan las órdenes iceberg
//! institucionales. Si el sistema ignora las ballenas acumulando o distribuyendo, incurre en colisiones
//! frontales letales (vender contra ballena acumulando o comprar contra ballena distribuyendo).
//!
//! ## PARA QUÉ
//! 1. Certificar invariancia y compatibilidad estricta cuando la liquidez oculta es nula o no está presente.
//! 2. Demostrar amplificación continua de confianza cuando la orden coincide con el flujo institucional.
//! 3. Demostrar atenuación suave C¹ (smoothstep) y veto conformal completo a |Π| >= 0.70 sin discontinuidades de salto.
//! 4. Certificar que ballenas ocultas activan detección en libros planos (stealth accumulation).
//! 5. Probar paridad espectral y resiliencia estricta ante NaNs e infinitos.

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

fn build_base_short_registry() -> Arc<OmniscientRegistry> {
    let reg = Arc::new(OmniscientRegistry::new());
    reg.set("hawkes_intensity", 2.0);
    reg.set("order_book_imbalance", -0.5);
    reg.set("ml_prob_motor", 0.15);
    reg.set("ml_model_base", 0.50);
    reg.set("ml_model_lift", 0.05);
    reg.set("hawkes_excitation_gene", 0.55);
    reg.set("obi_zscore_gene", 1.0);
    reg.set("obi_p80_medido", 0.15);
    reg
}

#[test]
fn test_backward_compatibility_zero_iceberg_pressure() {
    let reg_uninit = build_base_long_registry();
    let mut eng_uninit = FlowExcitationConfluenceEngine::new();
    assert!(eng_uninit.init(reg_uninit).is_ok());
    let v_uninit = eng_uninit.evaluate();

    let reg_zero = build_base_long_registry();
    reg_zero.set("hidden_liquidity_pressure", 0.0);
    reg_zero.set("is_iceberg_active", 0.0);
    reg_zero.set("hidden_liquidity_weight", 0.30);
    let mut eng_zero = FlowExcitationConfluenceEngine::new();
    assert!(eng_zero.init(reg_zero).is_ok());
    let v_zero = eng_zero.evaluate();

    assert!(v_uninit.is_finite());
    assert!(v_uninit > 0.0);
    assert!(
        (v_uninit - v_zero).abs() < 1e-12,
        "Sin datos de iceberg o con presión 0.0 el voto debe ser bit a bit idéntico: {v_uninit} vs {v_zero}"
    );
}

#[test]
fn test_aligned_iceberg_amplifies_long_confidence() {
    let reg_base = build_base_long_registry();
    let mut eng_base = FlowExcitationConfluenceEngine::new();
    assert!(eng_base.init(reg_base).is_ok());
    let v_base = eng_base.evaluate();

    let reg_aligned = build_base_long_registry();
    reg_aligned.set("hidden_liquidity_pressure", 0.60);
    reg_aligned.set("is_iceberg_active", 1.0);
    reg_aligned.set("hidden_liquidity_weight", 0.30);
    let mut eng_aligned = FlowExcitationConfluenceEngine::new();
    assert!(eng_aligned.init(reg_aligned).is_ok());
    let v_aligned = eng_aligned.evaluate();

    assert!(
        v_aligned > v_base,
        "La acumulación institucional alineada debe amplificar la convicción: {v_aligned} > {v_base}"
    );
    assert!(v_aligned <= 1.0, "El voto amplificado no debe violar la cota [0, 1]: {v_aligned}");
}

#[test]
fn test_opposing_iceberg_c1_smooth_damping_and_conformal_veto() {
    let sweep_pressures = [
        0.0,
        -0.10,
        -0.20,
        -0.30,
        -0.45,
        -0.60,
        -0.70,
        -0.85,
        -1.0,
    ];

    let mut votes = Vec::new();
    for &p in &sweep_pressures {
        let reg = build_base_long_registry();
        reg.set("hidden_liquidity_pressure", p);
        reg.set("is_iceberg_active", if p.abs() > 0.30 { 1.0 } else { 0.0 });
        reg.set("hidden_liquidity_weight", 0.30);

        let mut eng = FlowExcitationConfluenceEngine::new();
        assert!(eng.init(reg).is_ok());
        votes.push(eng.evaluate());
    }

    // 1. Presión 0 y leve no atenuada
    assert!(votes[0] > 0.0);
    assert!(votes[1] > 0.0);

    // 2. Monotonía decreciente ante presión opuesta creciente
    for i in 1..votes.len() {
        assert!(
            votes[i] <= votes[i - 1] + 1e-9,
            "El voto debe decrecer monótonamente conforme la absorción opuesta aumenta: votes[{i}]={} vs {}",
            votes[i],
            votes[i - 1]
        );
    }

    // 3. Veto conformal completo en |Π| >= 0.70
    assert_eq!(
        votes[6], 0.0,
        "A presión opuesta -0.70 el voto debe estar completamente extinguido (veto conformal C¹): {}",
        votes[6]
    );
    assert_eq!(
        votes[7], 0.0,
        "A presión opuesta -0.85 el voto debe permanecer en 0.0: {}",
        votes[7]
    );
    assert_eq!(
        votes[8], 0.0,
        "A presión opuesta -1.0 el voto debe permanecer en 0.0: {}",
        votes[8]
    );

    // 4. Verificación de suavidad C¹ (sin saltos discontinuos mayores al 40% en un delta de 0.15)
    for i in 1..votes.len() {
        let delta_v = (votes[i - 1] - votes[i]).abs();
        assert!(
            delta_v < 0.40,
            "Transición C¹ suave sin saltos cliff: salto medido={delta_v} en paso {i}"
        );
    }
}

#[test]
fn test_aligned_and_opposing_short_behavior() {
    // Caso Short base
    let reg_short_base = build_base_short_registry();
    let mut eng_short_base = FlowExcitationConfluenceEngine::new();
    assert!(eng_short_base.init(reg_short_base).is_ok());
    let v_short_base = eng_short_base.evaluate();
    assert!(v_short_base < 0.0, "Base short debe votar negativo: {v_short_base}");

    // Short con iceberg alineado (distribución institucional en Bid: Π = -0.60)
    let reg_short_aligned = build_base_short_registry();
    reg_short_aligned.set("hidden_liquidity_pressure", -0.60);
    reg_short_aligned.set("is_iceberg_active", 1.0);
    reg_short_aligned.set("hidden_liquidity_weight", 0.30);
    let mut eng_short_aligned = FlowExcitationConfluenceEngine::new();
    assert!(eng_short_aligned.init(reg_short_aligned).is_ok());
    let v_short_aligned = eng_short_aligned.evaluate();

    assert!(
        v_short_aligned < v_short_base,
        "La distribución institucional alineada debe magnificar la convicción vendedora: {v_short_aligned} < {v_short_base}"
    );

    // Short con iceberg opuesto severo (acumulación institucional en Ask: Π = +0.75)
    let reg_short_opposed = build_base_short_registry();
    reg_short_opposed.set("hidden_liquidity_pressure", 0.75);
    reg_short_opposed.set("is_iceberg_active", 1.0);
    reg_short_opposed.set("hidden_liquidity_weight", 0.30);
    let mut eng_short_opposed = FlowExcitationConfluenceEngine::new();
    assert!(eng_short_opposed.init(reg_short_opposed).is_ok());
    let v_short_opposed = eng_short_opposed.evaluate();

    assert_eq!(
        v_short_opposed, 0.0,
        "Short atacando muro institucional de acumulación debe ser vetado conformalmente a 0.0: {v_short_opposed}"
    );
}

#[test]
fn test_stealth_iceberg_awakens_flat_orderbook() {
    // Con OBI = 0.0, el motor sin liquidez oculta retorna 0.0
    let reg_flat = Arc::new(OmniscientRegistry::new());
    reg_flat.set("hawkes_intensity", 2.2);
    reg_flat.set("order_book_imbalance", 0.0);
    reg_flat.set("ml_prob_motor", 0.85);
    reg_flat.set("ml_model_base", 0.30);
    reg_flat.set("ml_model_lift", 0.05);

    let mut eng_flat = FlowExcitationConfluenceEngine::new();
    assert!(eng_flat.init(Arc::clone(&reg_flat)).is_ok());
    assert_eq!(eng_flat.evaluate(), 0.0);

    // Con iceberg activo de acumulación institucional (+0.80), el OBI efectivo se activa
    reg_flat.set("hidden_liquidity_pressure", 0.80);
    reg_flat.set("is_iceberg_active", 1.0);
    reg_flat.set("hidden_liquidity_weight", 0.40);

    let mut eng_active = FlowExcitationConfluenceEngine::new();
    assert!(eng_active.init(reg_flat).is_ok());
    let v_active = eng_active.evaluate();

    assert!(
        v_active > 0.0,
        "Acumulación stealth activa con libro plano debe producir señal direccional válida: {v_active}"
    );
}

#[test]
fn test_spectral_vote_with_liquidity_modulation() {
    let mut displacements = [0.0f64; 32];
    for i in 0..32 {
        displacements[i] = (i as f64 - 16.0) * 0.1;
    }
    let lambda_ratio = 2.5;

    // Con presión 0.0 debe ser idéntico al voto espectral clásico
    let v_classic = FlowExcitationConfluenceEngine::voto_espectral(&displacements, lambda_ratio);
    let v_liq_zero = FlowExcitationConfluenceEngine::voto_espectral_con_liquidez(&displacements, lambda_ratio, 0.0);

    for k in 0..32 {
        assert_eq!(
            v_classic.en_escala(k), v_liq_zero.en_escala(k),
            "Escala {k} debe ser idéntica con presión 0.0"
        );
    }

    // Con presión positiva, las microescalas (k < 12) deben recibir mayor impulso que las macroescalas
    let v_liq_pos = FlowExcitationConfluenceEngine::voto_espectral_con_liquidez(&displacements, lambda_ratio, 0.8);
    let delta_micro = v_liq_pos.en_escala(0) - v_classic.en_escala(0);
    let delta_macro = v_liq_pos.en_escala(31) - v_classic.en_escala(31);

    assert!(
        delta_micro.abs() >= delta_macro.abs(),
        "El impacto de liquidez oculta debe concentrarse en microescalas: micro={delta_micro}, macro={delta_macro}"
    );
}

#[test]
fn test_resilience_to_nans_and_infinities() {
    let reg_corrupted = build_base_long_registry();
    reg_corrupted.set("hidden_liquidity_pressure", f64::NAN);
    reg_corrupted.set("hidden_liquidity_weight", f64::INFINITY);

    let mut eng = FlowExcitationConfluenceEngine::new();
    assert!(eng.init(reg_corrupted).is_ok());
    let v = eng.evaluate();

    assert!(v.is_finite(), "El motor debe blindarse contra NaNs e infinitos: {v}");
    assert!((-1.0..=1.0).contains(&v), "El voto debe mantenerse acotado en [-1, 1]: {v}");
}
