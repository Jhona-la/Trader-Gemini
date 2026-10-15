//! 🌊 CONTRATO DE ADAPTACIÓN HIDRODINÁMICA Y MICROESTRUCTURA (OLA Ω121)
//!
//! Verifica matemáticamente que:
//! 1. `NavierStokesParams::validate` rechaza límites no físicos, invertidos o no finitos.
//! 2. `NavierStokesReynoldsEngine` se adapta dinámicamente a nuevos umbrales laminares y turbulentos en O(1).
//! 3. `compute_laminar_share_with_crit` es suave C^∞, monótona decreciente y acotada en [0.0, 1.0].
//! 4. `OFIModel` permite configurar `ewma_alpha` dinámicamente sin perder inmunidad a NaN.
//! 5. `OrderFlowTracker` adapta `short_alpha` y `long_alpha` de forma estable.
//! 6. `InstitutionalVolumeTracker` adapta `min_sample_count` evitando ráfagas con conteos insuficientes.

use feature_engine::navier_stokes::{HydrodynamicRegime, NavierStokesParams, NavierStokesReynoldsEngine};
use feature_engine::microstructure::{InstitutionalVolumeTracker, OFIModel, OrderFlowTracker};

#[test]
fn test_navier_stokes_params_validation_and_fail_closed() {
    let default_params = NavierStokesParams::default();
    assert!(default_params.validate().is_ok());

    // Invertir umbrales laminar >= turbulento debe fallar
    let mut invalid_crit = default_params;
    invalid_crit.critical_reynolds_laminar = 6.0;
    invalid_crit.critical_reynolds_turbulent = 5.0;
    assert!(invalid_crit.validate().is_err());

    // NaN o Inf debe fallar
    let mut nan_params = default_params;
    nan_params.relaxation_time_s = f64::NAN;
    assert!(nan_params.validate().is_err());

    let mut inf_params = default_params;
    inf_params.ewma_alpha = f64::INFINITY;
    assert!(inf_params.validate().is_err());

    // Parámetros fuera de rango
    let mut out_of_bounds = default_params;
    out_of_bounds.ewma_alpha = 0.99; // Límite superior es 0.50
    assert!(out_of_bounds.validate().is_err());
}

#[test]
fn test_navier_stokes_engine_dynamic_threshold_adaptation() {
    let mut engine = NavierStokesReynoldsEngine::new();
    assert_eq!(engine.params.critical_reynolds_laminar, 1.0);
    assert_eq!(engine.params.critical_reynolds_turbulent, 5.0);

    // Configurar nuevos parámetros dinámicos con umbrales extendidos
    let custom_params = NavierStokesParams {
        critical_reynolds_laminar: 2.5,
        critical_reynolds_turbulent: 10.0,
        relaxation_time_s: 0.150,
        ewma_alpha: 0.10,
    };
    engine.update_params(custom_params);

    assert_eq!(engine.params.critical_reynolds_laminar, 2.5);
    assert_eq!(engine.params.critical_reynolds_turbulent, 10.0);
    assert_eq!(engine.relaxation_time_s, 0.150);

    // Simular mercado con Reynolds intermedio (ej. Re = 2.0)
    // Con umbral por defecto (1.0), Re = 2.0 sería Transitional.
    // Con umbral adaptado (2.5), Re = 2.0 debe ser clasificado como Laminar!
    engine.ewma_reynolds = 2.0;
    assert_eq!(engine.regime(), HydrodynamicRegime::Laminar);
    assert!(engine.is_laminar());
    assert!(!engine.is_turbulent());

    // Elevar a Re = 6.0: Con turbulento = 10.0, es Transitional (no Turbulent aún)
    engine.ewma_reynolds = 6.0;
    assert_eq!(engine.regime(), HydrodynamicRegime::Transitional);
    assert!(!engine.is_laminar());
    assert!(!engine.is_turbulent());

    // Elevar a Re = 12.0: Es Turbulent
    engine.ewma_reynolds = 12.0;
    assert_eq!(engine.regime(), HydrodynamicRegime::Turbulent);
    assert!(engine.is_turbulent());
}

#[test]
fn test_laminar_share_monotonic_c_infinity_and_bounds() {
    // Para cualquier Re >= 0 y Re_crit > 0:
    // laminar_share(0) == 1.0
    // laminar_share(crit) == 0.5
    // lim_{Re -> ∞} laminar_share(Re) == 0.0
    let crit = 3.0;
    let s0 = NavierStokesReynoldsEngine::compute_laminar_share_with_crit(0.0, crit);
    assert!((s0 - 1.0).abs() < 1e-12);

    let scrit = NavierStokesReynoldsEngine::compute_laminar_share_with_crit(crit, crit);
    assert!((scrit - 0.50).abs() < 1e-12);

    let s_large = NavierStokesReynoldsEngine::compute_laminar_share_with_crit(100.0, crit);
    assert!(s_large > 0.0 && s_large < 0.002);

    // Monotonía estricta
    let mut prev = 1.0;
    for i in 1..=50 {
        let re = i as f64 * 0.2;
        let s = NavierStokesReynoldsEngine::compute_laminar_share_with_crit(re, crit);
        assert!(s < prev, "Debe decrecer monótonamente en {re}: s={s}, prev={prev}");
        assert!(s >= 0.0 && s <= 1.0);
        prev = s;
    }

    // Inmunidad a NaN
    let s_nan = NavierStokesReynoldsEngine::compute_laminar_share_with_crit(f64::NAN, crit);
    assert_eq!(s_nan, 1.0);
}

#[test]
fn test_ofi_model_dynamic_alpha_adaptation_and_robustness() {
    let mut ofi = OFIModel::new();
    assert_eq!(ofi.ewma_alpha, OFIModel::DEFAULT_EWMA_ALPHA);

    // Adaptar alfa a un valor más reactivo
    ofi.set_alpha(0.25);
    assert_eq!(ofi.ewma_alpha, 0.25);

    // Validar clamping seguro de set_alpha
    ofi.set_alpha(1.50); // Fuera de rango, no debe mutar
    assert_eq!(ofi.ewma_alpha, 0.25);
    ofi.set_alpha(f64::NAN); // No finito, no debe mutar
    assert_eq!(ofi.ewma_alpha, 0.25);

    // Actualizaciones con eventos normales
    ofi.update(100.0, 100.1, 10.0, 10.0);
    let val1 = ofi.update(100.05, 100.15, 15.0, 5.0);
    assert!(val1.is_finite());

    // Inmunidad ante precios NaN sin envenenamiento
    let val_nan = ofi.update(f64::NAN, 100.15, 15.0, 5.0);
    assert!(val_nan.is_finite());
}

#[test]
fn test_order_flow_tracker_adaptive_alphas() {
    let mut tracker = OrderFlowTracker::new().with_alphas(0.20, 0.02);
    assert_eq!(tracker.short_alpha, 0.20);
    assert_eq!(tracker.long_alpha, 0.02);

    // Mutador seguro
    tracker.set_alphas(0.05, 0.005);
    assert_eq!(tracker.short_alpha, 0.05);
    assert_eq!(tracker.long_alpha, 0.005);

    // Validación rechaza valores no finitos
    tracker.set_alphas(f64::NAN, 0.005);
    assert_eq!(tracker.short_alpha, 0.05);

    // Simular trades taker buy y sell
    let imb1 = tracker.update(5.0, false); // Taker buy
    assert!(imb1 > 0.0);
    assert!(tracker.short_ema_obi > 0.0);

    let imb2 = tracker.update(10.0, true); // Taker sell más fuerte
    assert!(imb2 < 0.0);
    assert!(tracker.velocity_obi.is_finite());
}

#[test]
fn test_institutional_volume_tracker_adaptive_samples() {
    let mut tracker = InstitutionalVolumeTracker::new(2.5).with_min_samples(10.0);
    assert_eq!(tracker.min_sample_count, 10.0);

    // Alimentar 8 muestras (menor que min_sample_count = 10): is_burst debe ser falso
    for i in 1..=8 {
        let (z, _, is_burst) = tracker.update(100.0 + i as f64);
        assert!(!is_burst, "No debe disparar ráfaga antes de min_samples, z={z}");
    }

    // Llegar a 10 muestras con un pico de 10x volumen
    tracker.update(100.0); // Muestra 9
    let (z_burst, _, is_burst) = tracker.update(2000.0); // Muestra 10 (disparo masivo)
    assert!(z_burst > 2.5);
    assert!(is_burst, "Debe disparar ráfaga al alcanzar min_samples y z_score > 2.5");
}
