//! B15-14: el dominio admitido debe conservar la recurrencia de momentos
//! y los mismos umbrales Hermite en parámetros locales y en el registry.

use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::vecm_arbitrage::{JohansenVecmEngine, VecmArbitrageParams};
use strategy_core::QuantumStrategy;

fn engine_with_thresholds(
    entry: f64,
    saturation: f64,
) -> (JohansenVecmEngine, Arc<OmniscientRegistry>) {
    let params = VecmArbitrageParams {
        z_entry_threshold: entry,
        z_saturation: saturation,
        ..Default::default()
    };
    assert!(params.validate().is_ok());
    let mut engine = JohansenVecmEngine::new_with_params(0.15, 1.0, params);
    let registry = Arc::new(OmniscientRegistry::new());
    engine.init(registry.clone()).unwrap();
    (engine, registry)
}

#[test]
fn accepted_small_gains_preserve_convex_mean_and_weighted_variance() {
    // Dos observaciones: x0=0, x1=ln(1.1), v0=1e-6.
    // Para un peso w, m1=w*x1 y v1=(1-w)*(v0+w*x1^2).
    // El caso por defecto (0.15 -> w=0.075) conserva su comportamiento.
    for (speed, expected_weight) in [(1e-5, 1e-5), (0.15, 0.075), (1.0, 0.5)] {
        let params = VecmArbitrageParams {
            speed_clamp_min: 1e-5,
            ..Default::default()
        };
        assert!(params.validate().is_ok());
        let mut engine = JohansenVecmEngine::new_with_params(speed, 1.0, params);
        assert_eq!(engine.update(100.0, 100.0), 0.0);
        let z = engine.update(110.0, 100.0);
        let x = 110.0_f64.ln() - 100.0_f64.ln();
        let expected_mean = expected_weight * x;
        let expected_variance = (1.0 - expected_weight) * (1e-6 + expected_weight * x * x);
        assert!(z.is_finite());
        assert!(engine.spread_mean >= 0.0 && engine.spread_mean <= x);
        assert!((engine.spread_mean - expected_mean).abs() < 1e-14);
        assert!((engine.spread_std.powi(2) - expected_variance).abs() < 1e-14);
        assert!(engine.spread_std > 1e-5, "la varianza no colapsa al suelo");
    }
}

#[test]
fn accepted_large_gains_preserve_convex_mean_and_variance() {
    // El dominio necesario es el del PESO w, no el de alpha: al segundo
    // evento alpha=1.5 da w=.75 y alpha=2 da w=1 sin extrapolar.
    // alpha=10 sigue siendo configurable, con su peso efectivo acotado a 1.
    for (speed, expected_weight) in [(1.5, 0.75), (2.0, 1.0), (10.0, 1.0)] {
        let params = VecmArbitrageParams {
            speed_clamp_max: 10.0,
            ..Default::default()
        };
        assert!(params.validate().is_ok());
        let mut engine = JohansenVecmEngine::new_with_params(speed, 1.0, params);
        assert_eq!(
            *engine.params(),
            params,
            "no descartar una configuración válida"
        );
        engine.update(100.0, 100.0);
        let z = engine.update(110.0, 100.0);
        let x = 110.0_f64.ln() - 100.0_f64.ln();
        let expected_mean = expected_weight * x;
        let raw_variance = (1.0 - expected_weight) * (1e-6 + expected_weight * x * x);
        assert!(z.is_finite());
        assert!(engine.spread_mean >= 0.0 && engine.spread_mean <= x);
        assert!((engine.spread_mean - expected_mean).abs() < 1e-14);
        if expected_weight < 1.0 {
            assert!((engine.spread_std.powi(2) - raw_variance).abs() < 1e-14);
        } else {
            // w=1 borra la dispersión de dos observaciones; el suelo
            // numérico de std, ya existente, es 1e-5 y el z queda en cero.
            assert_eq!(engine.spread_std, 1e-5);
            assert_eq!(z, 0.0);
        }
    }
}

#[test]
fn constructor_rejects_the_same_invalid_parameters_as_updates() {
    let defaults = VecmArbitrageParams::default();
    let invalid_cases = [
        VecmArbitrageParams {
            speed_clamp_max: f64::NAN,
            ..defaults
        },
        VecmArbitrageParams {
            speed_clamp_max: 0.001,
            ..defaults
        },
        VecmArbitrageParams {
            speed_clamp_min: f64::NAN,
            ..defaults
        },
        VecmArbitrageParams {
            z_saturation: 1.0,
            ..defaults
        },
    ];
    for invalid in invalid_cases {
        let mut engine = JohansenVecmEngine::new_with_params(10.0, 1.0, invalid);
        assert_eq!(*engine.params(), defaults);
        engine.update(100.0, 100.0);
        engine.update(110.0, 100.0);
        let x = 110.0_f64.ln() - 100.0_f64.ln();
        assert!(engine.spread_mean > 0.0 && engine.spread_mean < x);
        assert!(engine.spread_std > 1e-5);
        engine.update_params(invalid);
        assert_eq!(*engine.params(), defaults);
        assert_eq!(*engine.with_params(invalid).params(), defaults);
    }
}

fn learned_state(engine: &JohansenVecmEngine) -> (u64, u64, u64, u64, String) {
    (
        engine.spread_mean.to_bits(),
        engine.spread_std.to_bits(),
        engine.window_count.to_bits(),
        engine.beta_hedge_ratio.to_bits(),
        format!("{:?}", engine.sde),
    )
}

#[test]
fn invalid_public_parameters_do_not_mutate_any_update_path() {
    for invalid_speed in [false, true] {
        let mut engine = JohansenVecmEngine::default();
        for i in 0..12 {
            engine.update_with_timestamp(100.0 + i as f64, 100.0, 1000 + i * 1000);
        }
        if invalid_speed {
            engine.alpha_speed = f64::NAN;
        } else {
            engine.params.speed_clamp_max = f64::NAN;
        }
        let before = learned_state(&engine);
        assert_eq!(engine.update(150.0, 100.0), 0.0);
        assert_eq!(learned_state(&engine), before);
        assert_eq!(engine.update_with_timestamp(150.0, 100.0, 20_000), 0.0);
        assert_eq!(learned_state(&engine), before);
        assert_eq!(engine.update_with_adaptive_beta(150.0, 100.0), 0.0);
        assert_eq!(learned_state(&engine), before);
    }
}

#[test]
fn lower_entry_boundary_is_honored_locally_and_through_registry() {
    for registry_override in [false, true] {
        let (engine, registry) = if registry_override {
            engine_with_thresholds(1.5, 3.0)
        } else {
            engine_with_thresholds(0.1, 0.5)
        };
        if registry_override {
            registry.set("vecm_z_entry", 0.1);
            registry.set("vecm_z_saturation", 0.5);
        }
        // z está en el punto medio: u=1/2, 3u^2-2u^3=1/2.
        registry.set("vecm_zscore", 0.3);
        assert!((engine.evaluate() + 0.5).abs() < 1e-12);
        registry.set("vecm_zscore", -0.3);
        assert!((engine.evaluate() - 0.5).abs() < 1e-12);
    }
}

#[test]
fn invalid_registry_threshold_pair_preserves_configured_pair() {
    for (entry, saturation) in [
        (f64::NAN, 3.0),
        (-1.0, 3.0),
        (0.05, 3.0),
        (10.1, 15.0),
        (0.5, f64::NAN),
        (0.5, 0.5),
        (0.5, 20.1),
        (2.0, 1.0),
    ] {
        let (engine, registry) = engine_with_thresholds(0.5, 2.5);
        registry.set("vecm_zscore", 1.5);
        registry.set("vecm_z_entry", entry);
        registry.set("vecm_z_saturation", saturation);
        assert!(
            (engine.evaluate() + 0.5).abs() < 1e-12,
            "override inválido ({entry}, {saturation}) conserva (0.5,2.5)"
        );
    }
}
