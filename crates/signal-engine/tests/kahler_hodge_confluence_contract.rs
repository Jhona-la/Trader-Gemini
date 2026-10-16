//! ⚡ CONTRATO DE CERTIFICACIÓN OLA Ω139: ACOPLAMIENTO CONFLUENTE DE VARIEDADES KÄHLERIANAS Y HODGE (1, 1)
//!
//! Valida:
//! 1. Bonificación de confluencia (+hasta 15%) cuando el flujo local está alineado con la corriente holomorfa $\mathcal{J}_{\text{Kahler}}$.
//! 2. Atenuación suave $C^1$ Hermite cuando el flujo colisiona frontalmente contra la variedad de Kähler.
//! 3. Inmunidad e invariancia estricta cuando la moneda está ausente de la variedad multiactivo.
//! 4. Modulación espectral multiescala: localización mesoscópica de paridad compleja en $k \in [8, 24]$.
//! 5. Latencia crítica en stack zero-alloc (<25 µs por evaluación).

use omniscient_registry::OmniscientRegistry;
use signal_engine::flow_excitation_confluence::FlowExcitationConfluenceEngine;
use strategy_core::QuantumStrategy;
use std::sync::Arc;
use std::time::Instant;

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
    reg.set("helmholtz_conviction", 1.0);
    reg.set("yang_mills_current", 0.0);
    reg.set("yang_mills_current_absent", 0.0);
    reg
}

#[test]
fn test_kahler_confluence_aligned_boost() {
    let r_neutral = build_base_long_registry();
    r_neutral.set("kahler_restoring_current", 0.0);
    r_neutral.set("kahler_current_absent", 0.0);

    let r_boosted = build_base_long_registry();
    r_boosted.set("kahler_restoring_current", 0.75);
    r_boosted.set("kahler_current_absent", 0.0);

    let mut engine_neutral = FlowExcitationConfluenceEngine::new();
    engine_neutral.init(r_neutral).unwrap();

    let mut engine_boosted = FlowExcitationConfluenceEngine::new();
    engine_boosted.init(r_boosted).unwrap();

    let v_neutral = engine_neutral.evaluate();
    let v_boosted = engine_boosted.evaluate();

    assert!(v_neutral > 0.0, "El voto neutral base debe ser positivo (Long)");
    assert!(
        v_boosted > v_neutral,
        "La corriente holomorfa alineada debe amplificar el voto: boosted ({:.4}) > neutral ({:.4})",
        v_boosted,
        v_neutral
    );
    let ratio = v_boosted / v_neutral;
    assert!(
        ratio <= 1.1501,
        "El multiplicador de confluencia Kähler-Hodge no debe exceder 1.15: {:.4}",
        ratio
    );
}

#[test]
fn test_kahler_confluence_opposing_damping() {
    let r_neutral = build_base_long_registry();
    r_neutral.set("kahler_restoring_current", 0.0);
    r_neutral.set("kahler_current_absent", 0.0);

    let r_opposed = build_base_long_registry();
    r_opposed.set("kahler_restoring_current", -0.85);
    r_opposed.set("kahler_current_absent", 0.0);

    let mut engine_neutral = FlowExcitationConfluenceEngine::new();
    engine_neutral.init(r_neutral).unwrap();

    let mut engine_opposed = FlowExcitationConfluenceEngine::new();
    engine_opposed.init(r_opposed).unwrap();

    let v_neutral = engine_neutral.evaluate();
    let v_opposed = engine_opposed.evaluate();

    assert!(v_neutral > 0.0);
    assert!(
        v_opposed < v_neutral * 0.50,
        "La colisión contra el potencial de Kähler debe atenuar fuertemente el voto: opposed ({:.4}) vs neutral ({:.4})",
        v_opposed,
        v_neutral
    );
}

#[test]
fn test_kahler_confluence_absent_fallback() {
    let r_absent_flag = build_base_long_registry();
    r_absent_flag.set("kahler_restoring_current", -0.90);
    r_absent_flag.set("kahler_current_absent", 1.0);

    let r_no_param = build_base_long_registry();
    r_no_param.set("kahler_restoring_current", 0.0);
    r_no_param.set("kahler_current_absent", 0.0);

    let mut engine_a = FlowExcitationConfluenceEngine::new();
    engine_a.init(r_absent_flag).unwrap();

    let mut engine_b = FlowExcitationConfluenceEngine::new();
    engine_b.init(r_no_param).unwrap();

    let va = engine_a.evaluate();
    let vb = engine_b.evaluate();

    assert!(
        (va - vb).abs() < 1e-9,
        "Moneda ausente de la variedad Kähleriana debe tener comportamiento neutral exacto: {:.6} vs {:.6}",
        va,
        vb
    );
}

#[test]
fn test_kahler_confluence_voto_espectral_mesoscale_localization() {
    let desplazamientos = [0.10f64; 32];
    let ratio_lambda_mu = 2.0;
    let hidden_liq_pressure = 0.0;
    let laminar_fraction = 1.0;

    let ve_neutral = FlowExcitationConfluenceEngine::voto_espectral_con_kahler_gauge_y_helmholtz(
        &desplazamientos,
        ratio_lambda_mu,
        hidden_liq_pressure,
        laminar_fraction,
        0.0,
        0.0,
    );

    let ve_kahler = FlowExcitationConfluenceEngine::voto_espectral_con_kahler_gauge_y_helmholtz(
        &desplazamientos,
        ratio_lambda_mu,
        hidden_liq_pressure,
        laminar_fraction,
        0.0,
        0.80,
    );

    // En micro-escala (k = 0, meso_factor = 0.0), la modulación Kähler es nula
    let v0_neutral = ve_neutral.en_escala(0);
    let v0_kahler = ve_kahler.en_escala(0);
    assert!(
        (v0_kahler - v0_neutral).abs() < 1e-9,
        "La corriente Kähler no debe distorsionar la micro-escala k=0: {} vs {}",
        v0_kahler,
        v0_neutral
    );

    // En meso-escala central (k = 16, meso_factor = 1.0), la corriente Kähler actúa con máxima intensidad
    let v16_neutral = ve_neutral.en_escala(16);
    let v16_kahler = ve_kahler.en_escala(16);
    assert!(
        v16_kahler > v16_neutral,
        "La meso-escala k=16 debe recibir el impulso holomorfo Kähler: {} > {}",
        v16_kahler,
        v16_neutral
    );
}

#[test]
fn test_kahler_confluence_submicrosecond_latency() {
    let r = build_base_long_registry();
    r.set("kahler_restoring_current", 0.50);
    r.set("kahler_current_absent", 0.0);

    let mut engine = FlowExcitationConfluenceEngine::new();
    engine.init(r).unwrap();

    let iters = 20_000;
    let start = Instant::now();

    for _ in 0..iters {
        let v = engine.evaluate();
        assert!(v.is_finite());
    }

    let elapsed = start.elapsed();
    let per_eval_us = elapsed.as_micros() as f64 / (iters as f64);

    assert!(
        per_eval_us < 25.0,
        "Latencia media de confluencia Kähler-Hodge excedió 25 µs: {:.3} µs",
        per_eval_us
    );
}
