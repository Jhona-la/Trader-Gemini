//! CONTRATO DE CERTIFICACIÓN OLA Ω126: DESACOPLAMIENTO DINÁMICO Y CONTINUIDAD C1
//! EN PROYECCIÓN ESPECTRAL, COMPUERTA PERCEPTRÓN Y PROCESO DE HAWKES-BESSEL.

use omniscient_registry::OmniscientRegistry;
use signal_engine::hawkes_bessel::{
    excitacion_hawkes_norm_with_ss, HawkesBesselEngine, HawkesBesselParams, DEFAULT_ALPHA,
    DEFAULT_BETA, DEFAULT_MU,
};
use signal_engine::perceptron_gate::{
    PerceptronGateEngine, PerceptronGateParams,
};
use signal_engine::proyeccion_espectral::{
    ProyeccionEspectralEngine, ProyeccionEspectralParams,
};
use std::sync::Arc;
use strategy_core::QuantumStrategy;

#[test]
fn test_spectral_projection_c1_continuity_and_smoothness() {
    let params = ProyeccionEspectralParams::default();
    assert!(params.validate().is_ok());

    // 1. Cotas extremas
    assert_eq!(params.compute_weight(0.0), 0.0);
    assert_eq!(params.compute_weight(params.masa_minima), 0.0);
    assert_eq!(params.compute_weight(params.masa_minima - 0.05), 0.0);
    assert_eq!(params.compute_weight(params.masa_plena), 1.0);
    assert_eq!(params.compute_weight(params.masa_plena + 0.10), 1.0);

    // 2. Continuidad C1 y monotonía estricta en el intervalo (masa_minima, masa_plena)
    let steps = 1000;
    let delta = (params.masa_plena - params.masa_minima) / steps as f64;
    let mut prev_w = 0.0;
    for i in 0..=steps {
        let masa = params.masa_minima + i as f64 * delta;
        let w = params.compute_weight(masa);
        assert!(w >= prev_w, "Monotonía rota en masa {}: {} < {}", masa, w, prev_w);
        assert!((w - prev_w) <= 0.01, "Discontinuidad de salto detectada en masa {}", masa);
        prev_w = w;
    }
    assert!((prev_w - 1.0).abs() < 1e-12);

    // 3. Inmunidad matemática ante NaN / Infinito
    assert_eq!(params.compute_weight(f64::NAN), 0.0);
    assert_eq!(params.compute_weight(-10.0), 0.0);
    assert_eq!(params.compute_weight(f64::NEG_INFINITY), 0.0);
}

#[test]
fn test_spectral_projection_engine_dynamic_registry_override() {
    let registry = Arc::new(OmniscientRegistry::new());
    let mut engine = ProyeccionEspectralEngine::new();
    assert!(engine.init(registry.clone()).is_ok());

    // Con parámetros canónicos: masa_minima = 0.20, masa_plena = 0.35
    registry.set("espectral_senal_proyectada", 0.8);
    registry.set("espectral_concentracion", 0.7);
    registry.set("espectral_masa_resuelta", 0.50); // masa plena -> peso = 1.0

    let out_plena = engine.evaluate_for_coin(0, "BTCUSDT");
    // 0.8 * 0.7 * 1.0 = 0.56
    assert!((out_plena - 0.56).abs() < 1e-6);

    // Ahora cambiamos la masa a un punto intermedio
    registry.set("espectral_masa_resuelta", 0.275); // punto medio entre 0.20 y 0.35 -> t = 0.5, smoothstep = 0.5
    let out_media = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!((out_media - 0.56 * 0.5).abs() < 1e-6);

    // Override dinámico: F1-124 erradicó las sondas de masa_minima/plena del
    // registro (claves sin escritor en producción); el canal sancionado son los
    // parámetros del motor (el host los sobrescribe ANTES del arranque).
    let params_alt = ProyeccionEspectralParams {
        masa_minima: 0.30,
        masa_plena: 0.60,
    };
    let mut engine_alt = ProyeccionEspectralEngine::with_params(params_alt);
    assert!(engine_alt.init(registry.clone()).is_ok());
    // Ahora con masa 0.275 está por debajo del nuevo mínimo (0.30) -> debe abstenerse (0.0)
    let out_vetada = engine_alt.evaluate_for_coin(0, "BTCUSDT");
    assert_eq!(out_vetada, 0.0);
}

#[test]
fn test_perceptron_gate_params_validation_and_dynamic_infer() {
    let mut params = PerceptronGateParams::default();
    assert!(params.validate().is_ok());

    // Validar rechazo de cotas inválidas
    params.ganancia = -1.0;
    assert!(params.validate().is_err());
    params.ganancia = 2.5;

    params.min_exploration_floor = 1.5;
    assert!(params.validate().is_err());
    params.min_exploration_floor = 0.15;

    // Inferencia con ganancia estándar (2.5) vs ganancia amplificada (5.0)
    let score = 0.4;
    let weight = 1.0;
    let out_std = PerceptronGateEngine::infer_with_params(score, weight, &params);

    let mut high_gain_params = params;
    high_gain_params.ganancia = 5.0;
    let out_high = PerceptronGateEngine::infer_with_params(score, weight, &high_gain_params);
    assert!(out_high > out_std, "Mayor ganancia debe aumentar la respuesta del perceptrón");

    // Dynamic evaluation via registry
    let registry = Arc::new(OmniscientRegistry::new());
    let mut engine = PerceptronGateEngine::new();
    assert!(engine.init(registry.clone()).is_ok());

    // F1-124/F1-098: canal vivo ('order_flow_imbalance' + 'hebbian_weight' estricto)
    registry.set("order_flow_imbalance", 0.5);
    registry.set("hebbian_weight", 1.0);
    let out_reg_default = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(out_reg_default > 0.0);

    registry.set("perceptron_ganancia", 6.0);
    let out_reg_boosted = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(out_reg_boosted > out_reg_default);
}

#[test]
fn test_hawkes_bessel_params_validation_and_branching_ratio() {
    let default_params = HawkesBesselParams::default();
    assert!(default_params.validate().is_ok());

    // Estado estacionario: 1 + 0.3 / 0.5 = 1.6
    assert!((default_params.steady_state_ratio() - (1.0 + DEFAULT_ALPHA / DEFAULT_BETA)).abs() < 1e-12);

    let custom_params = HawkesBesselParams {
        default_mu: DEFAULT_MU,
        alpha: 0.4,
        beta: 0.8,
        mu_tau_s: 30.0,
        mu_floor: 0.02,
    };
    assert!(custom_params.validate().is_ok());
    // 1 + 0.4 / 0.8 = 1.5
    assert!((custom_params.steady_state_ratio() - 1.5).abs() < 1e-12);

    let mut engine = HawkesBesselEngine::with_params(custom_params);
    assert_eq!(engine.params().alpha, 0.4);
    assert_eq!(engine.params().beta, 0.8);
    assert!((engine.branching_ratio() - 0.5).abs() < 1e-12);
    assert!(engine.is_stationary());

    // Excitación con ratio dinámico
    let ss = custom_params.steady_state_ratio();
    assert_eq!(excitacion_hawkes_norm_with_ss(ss, ss), 0.0);
    let exc_cascada = excitacion_hawkes_norm_with_ss(3.0 * ss, ss);
    assert!(exc_cascada > 0.0 && exc_cascada < 1.0);
    assert_eq!(excitacion_hawkes_norm_with_ss(f64::NAN, ss), 0.0);

    // Event recording & dynamic adaptation
    engine.record_event(100.0);
    engine.record_event(100.5);
    assert!(engine.base_rate() > 0.0);
    assert!(engine.intensity(100.5) >= engine.base_rate());
}
