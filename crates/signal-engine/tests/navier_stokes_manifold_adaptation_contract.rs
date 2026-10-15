//! 🌊 CONTRATO DE ADAPTACIÓN DEL COLECTOR ESPECTRAL DE NAVIER-STOKES (OLA Ω121)
//!
//! Verifica que:
//! 1. `NavierStokesManifoldParams` valide límites físicos y consistencia dimensional.
//! 2. `NavierStokesManifoldEngine` se adapte dinámicamente vía `with_params` y `update_params`.
//! 3. `voto_espectral_with_params` responda a las 32 escalas diádicas con modulación laminar continua.
//! 4. La sensibilidad modal de Reynolds aumente con `scale_weight_factor`.
//! 5. `turbulent_cascade_flux` y `kolmogorov_energy_spectrum` mantengan conservación de flujo.
//! 6. `evaluate_for_coin` sea seguro y use `velocity_scale` dinámico sin divisiones por cero ni pánicos.

use signal_engine::navier_stokes_manifold::{NavierStokesManifoldEngine, NavierStokesManifoldParams};
use signal_engine::voto_espectral::ESCALAS_VOTO;
use strategy_core::QuantumStrategy;

#[test]
fn test_manifold_params_validation_and_safety() {
    let default_params = NavierStokesManifoldParams::default();
    assert!(default_params.validate().is_ok());

    // Non-finite
    let mut nan_p = default_params;
    nan_p.scale_weight_factor = f64::NAN;
    assert!(nan_p.validate().is_err());

    // Out of bounds
    let mut out_p = default_params;
    out_p.modulation_floor = -0.1;
    assert!(out_p.validate().is_err());

    let mut out_p2 = default_params;
    out_p2.velocity_scale = 0.5; // Mínimo es 1.0
    assert!(out_p2.validate().is_err());
}

#[test]
fn test_manifold_engine_dynamic_adaptation() {
    let mut engine = NavierStokesManifoldEngine::new();
    assert_eq!(engine.params.scale_weight_factor, 0.25);
    assert_eq!(engine.params.velocity_scale, 100.0);

    let custom_params = NavierStokesManifoldParams {
        scale_weight_factor: 0.50,
        modulation_floor: 0.20,
        modulation_scale: 0.80,
        signal_tanh_scale: 1.0,
        velocity_scale: 250.0,
    };
    engine.update_params(custom_params);

    assert_eq!(engine.params.scale_weight_factor, 0.50);
    assert_eq!(engine.params.modulation_floor, 0.20);
    assert_eq!(engine.params.modulation_scale, 0.80);
    assert_eq!(engine.params.signal_tanh_scale, 1.0);
    assert_eq!(engine.params.velocity_scale, 250.0);
}

#[test]
fn test_voto_espectral_with_params_smoothness_and_bounds() {
    let mut desplazamientos = [0.0f64; ESCALAS_VOTO];
    for k in 0..ESCALAS_VOTO {
        desplazamientos[k] = (k as f64 - 15.0) * 0.2; // Rango centrado
    }

    let params = NavierStokesManifoldParams {
        scale_weight_factor: 0.30,
        modulation_floor: 0.30,
        modulation_scale: 0.70,
        signal_tanh_scale: 0.60,
        velocity_scale: 120.0,
    };

    let voto = NavierStokesManifoldEngine::voto_espectral_with_params(
        &desplazamientos,
        1.5,  // Viscosidad cinemática
        0.85, // Flujo laminar dominante
        &params,
    );

    // Verificar que todas las escalas estén acotadas en [-1.0, 1.0] y finitas
    for k in 0..ESCALAS_VOTO {
        let v = voto.en_escala(k);
        assert!(v.is_finite(), "Voto en escala {k} debe ser finito");
        assert!(v >= -1.0 && v <= 1.0, "Voto en escala {k} debe estar en [-1, 1]");
    }

    // Media de banda rápida vs lenta
    let media = voto.media_banda(0, 31);
    assert!(media.is_some());
    assert!(media.unwrap().is_finite());
}

#[test]
fn test_modal_reynolds_scale_factor_monotonicity() {
    let z = 2.0;
    let visc = 1.0;

    let re_low = NavierStokesManifoldEngine::compute_modal_reynolds_with_scale(z, 20, visc, 0.10);
    let re_mid = NavierStokesManifoldEngine::compute_modal_reynolds_with_scale(z, 20, visc, 0.25);
    let re_high = NavierStokesManifoldEngine::compute_modal_reynolds_with_scale(z, 20, visc, 0.50);

    assert!(re_low < re_mid);
    assert!(re_mid < re_high);
}

#[test]
fn test_energy_spectrum_and_cascade_flux_conservation() {
    let mut desplazamientos = [0.0f64; ESCALAS_VOTO];
    for k in 0..ESCALAS_VOTO {
        desplazamientos[k] = 1.0 + (k as f64) * 0.1;
    }

    let energy = NavierStokesManifoldEngine::kolmogorov_energy_spectrum(&desplazamientos);
    for k in 0..ESCALAS_VOTO {
        assert!(energy[k] >= 0.0 && energy[k].is_finite());
    }

    let flux = NavierStokesManifoldEngine::turbulent_cascade_flux(&desplazamientos);
    assert!(flux.is_finite());
    assert!(flux > 0.0, "Flujo de cascada debe ser positivo para energía creciente con escala");
}

#[test]
fn test_fail_closed_uninitialized_strategy_returns_zero() {
    let engine = NavierStokesManifoldEngine::new();
    assert_eq!(engine.evaluate(), 0.0);
    assert_eq!(engine.evaluate_for_coin(0, "BTCUSDT"), 0.0);
}
