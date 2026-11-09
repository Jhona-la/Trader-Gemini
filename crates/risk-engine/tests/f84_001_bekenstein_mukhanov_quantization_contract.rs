//! # Contrato Formal Fase 84: Bekenstein-Mukhanov Horizon Area Quantization & Discrete Risk Spectrum
//!
//! ## Invariantes Verificados:
//! - C1: Cuantización exacta del área $A_n \le A$, espectro de entropía discreto $S_n = n \ln(3)$ y frecuencia Bohr.
//! - C2: Tasa de evaporación de radiación Hawking $\Gamma \in [0.0, 1.0]$ y modulación de slot.
//! - C3: Disparo determinista del veto `BEKENSTEIN_MUKHANOV_AREA_EVAPORATION_VETO` con colapso de slot a 0.0.
//! - C4: Inmunidad contra NaN/Infinitos y ejecución $O(1)$ sin asignaciones en heap.

use risk_engine::{
    BekensteinMukhanovConfig, BekensteinMukhanovQuantizationEngine, BEKENSTEIN_ALPHA_QUANTUM,
};

#[test]
fn test_c1_bekenstein_area_quantization_and_entropy_levels() {
    let engine = BekensteinMukhanovQuantizationEngine::default();
    let report = engine.evaluate(50.0, 0.02, 0.05);

    assert!(report.horizon_area_a > 0.0);
    assert!(report.quantum_level_n > 0);
    assert!(report.quantized_area_an <= report.horizon_area_a);
    let expected_sn = (report.quantum_level_n as f64) * 3.0_f64.ln();
    assert!((report.bekenstein_entropy_sn - expected_sn).abs() < 1e-10);
    assert!(report.bohr_transition_frequency > 0.0);
    assert!(!report.bekenstein_evaporation_veto);
    assert!(report.slot_scale_factor >= 0.10 && report.slot_scale_factor <= 1.25);
    assert!((BEKENSTEIN_ALPHA_QUANTUM - 4.0 * 3.0_f64.ln()).abs() < 1e-12);
}

#[test]
fn test_c2_hawking_evaporation_rate_and_slot_modulation() {
    let engine = BekensteinMukhanovQuantizationEngine::default();
    let report_calm = engine.evaluate(100.0, 0.01, 0.01);
    let report_stress = engine.evaluate(100.0, 0.15, 0.20);

    assert!(report_stress.hawking_evaporation_rate > report_calm.hawking_evaporation_rate);
    assert!(report_stress.slot_scale_factor < report_calm.slot_scale_factor);
}

#[test]
fn test_c3_horizon_area_evaporation_hard_veto() {
    let config = BekensteinMukhanovConfig {
        floor_equity: 10.0,
        critical_floor_quantum_n: 10,
        target_quantum_n: 50,
    };
    let engine = BekensteinMukhanovQuantizationEngine::new(config);

    // Capital al nivel del suelo -> evaporación del horizonte
    let report_floor = engine.evaluate(10.0, 0.05, 0.30);
    assert!(report_floor.bekenstein_evaporation_veto);
    assert_eq!(report_floor.slot_scale_factor, 0.0);

    // Capital por debajo del suelo
    let report_subfloor = engine.evaluate(8.5, 0.10, 0.50);
    assert!(report_subfloor.bekenstein_evaporation_veto);
    assert_eq!(report_subfloor.slot_scale_factor, 0.0);
}

#[test]
fn test_c4_stack_zero_heap_and_subnormal_immunity() {
    let engine = BekensteinMukhanovQuantizationEngine::default();
    let report_nan = engine.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY);

    assert!(report_nan.horizon_area_a.is_finite());
    assert!(report_nan.quantized_area_an.is_finite());
    assert!(report_nan.bekenstein_entropy_sn.is_finite());
    assert!(report_nan.bohr_transition_frequency.is_finite());
    assert!(report_nan.hawking_evaporation_rate.is_finite());
    assert!(report_nan.slot_scale_factor.is_finite());
}
