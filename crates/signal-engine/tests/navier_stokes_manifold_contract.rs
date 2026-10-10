use signal_engine::navier_stokes_manifold::NavierStokesManifoldEngine;
use signal_engine::voto_espectral::ESCALAS_VOTO;
use strategy_core::{QuantumStrategy, TradeHorizon};

#[test]
fn modal_reynolds_monotonicity_across_dyadic_scales() {
    let momentum_z = 2.0;
    let viscosity = 0.5;

    let mut prev_re = -1.0;
    for k in 0..ESCALAS_VOTO {
        let re = NavierStokesManifoldEngine::compute_modal_reynolds(momentum_z, k, viscosity);
        assert!(re.is_finite(), "Re modal en escala {k} debe ser finito");
        assert!(
            re > prev_re,
            "Re modal debe crecer estrictamente con la escala diádica k: k={k}, re={re}, prev={prev_re}"
        );
        prev_re = re;
    }
}

#[test]
fn kolmogorov_inertial_efficiency_smooth_c_infinity() {
    // Prueba de asíntotas y monotonía en [0, 100]
    let re_values = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0];
    let mut prev_eta = -1.0;

    for &re in &re_values {
        let eta = NavierStokesManifoldEngine::compute_kolmogorov_inertial_efficiency(re);
        assert!(eta.is_finite(), "eta debe ser finito para re={re}");
        assert!(
            (0.0..=1.0).contains(&eta),
            "eta debe estar en [0.0, 1.0], obtenido {eta} para re={re}"
        );
        assert!(
            eta >= prev_eta,
            "eta debe ser monótonamente no-decreciente con re: re={re}, eta={eta}, prev={prev_eta}"
        );
        prev_eta = eta;
    }

    // Asíntota en 0
    assert_eq!(
        NavierStokesManifoldEngine::compute_kolmogorov_inertial_efficiency(0.0),
        0.0
    );
    // Para Re >> 1, eta tiende a 1
    let eta_large = NavierStokesManifoldEngine::compute_kolmogorov_inertial_efficiency(100.0);
    assert!(
        eta_large > 0.999,
        "Para Re=100, eta debe ser > 0.999, obtenido {eta_large}"
    );
}

#[test]
fn kolmogorov_energy_spectrum_and_cascade_flux_conservation() {
    // Caso 1: Desplazamientos planos -> Espectro plano -> Flujo cero
    let uniform_z = [1.5; ESCALAS_VOTO];
    let energy_uniform = NavierStokesManifoldEngine::kolmogorov_energy_spectrum(&uniform_z);
    for k in 0..ESCALAS_VOTO {
        assert!(
            (energy_uniform[k] - 0.5 * 1.5 * 1.5).abs() < 1e-12,
            "Energía uniforme incorrecta en escala {k}"
        );
    }
    let flux_uniform = NavierStokesManifoldEngine::turbulent_cascade_flux(&uniform_z);
    assert!(
        flux_uniform.abs() < 1e-12,
        "Flujo neto debe ser 0 en espectro plano, obtenido {flux_uniform}"
    );

    // Caso 2: Energía decrece de macro a micro (cascada directa hacia disipación)
    let mut direct_cascade_z = [0.0; ESCALAS_VOTO];
    for k in 0..ESCALAS_VOTO {
        // En k=31 (macro) mayor amplitud, en k=0 (micro) menor amplitud
        direct_cascade_z[k] = (k as f64) * 0.1;
    }
    let flux_direct = NavierStokesManifoldEngine::turbulent_cascade_flux(&direct_cascade_z);
    assert!(
        flux_direct > 0.0,
        "Gradiente positivo de macro a micro debe dar flujo positivo acumulado: {flux_direct}"
    );
}

#[test]
fn spectral_vote_produces_consistent_signals() {
    let z_bullish = [1.2; ESCALAS_VOTO];
    let z_bearish = [-1.2; ESCALAS_VOTO];

    let voto_bull = NavierStokesManifoldEngine::voto_espectral(&z_bullish, 0.5, 0.9);
    let voto_bear = NavierStokesManifoldEngine::voto_espectral(&z_bearish, 0.5, 0.9);

    let media_bull = voto_bull.media_banda(0, ESCALAS_VOTO - 1).unwrap();
    let media_bear = voto_bear.media_banda(0, ESCALAS_VOTO - 1).unwrap();

    assert!(
        media_bull > 0.0,
        "Voto alcista debe tener media > 0, obtenido {}",
        media_bull
    );
    assert!(
        media_bear < 0.0,
        "Voto bajista debe tener media < 0, obtenido {}",
        media_bear
    );
    // Simetría hidrodinámica
    assert!(
        (media_bull + media_bear).abs() < 1e-10,
        "Debe haber simetría antisimétrica en voto hidrodinámico"
    );

    // Cada escala debe tener el signo correcto
    for k in 0..ESCALAS_VOTO {
        assert!(voto_bull.en_escala(k) > 0.0);
        assert!(voto_bear.en_escala(k) < 0.0);
        assert!((voto_bull.en_escala(k) + voto_bear.en_escala(k)).abs() < 1e-10);
    }
}

#[test]
fn fail_closed_nan_inf_immunity() {
    let mut corrupted = [f64::NAN; ESCALAS_VOTO];
    corrupted[5] = f64::INFINITY;
    corrupted[10] = f64::NEG_INFINITY;

    let voto = NavierStokesManifoldEngine::voto_espectral(&corrupted, f64::NAN, f64::INFINITY);
    let media = voto.media_banda(0, ESCALAS_VOTO - 1).unwrap();
    assert!(
        media.is_finite(),
        "Voto medio debe ser finito ante corrupción NaN/Inf"
    );
    assert_eq!(media, 0.0, "Voto debe colapsar a 0 ante corrupción total");
    for k in 0..ESCALAS_VOTO {
        assert_eq!(voto.en_escala(k), 0.0);
    }

    let re = NavierStokesManifoldEngine::compute_modal_reynolds(f64::NAN, 10, f64::NEG_INFINITY);
    assert!(re.is_finite());
    assert_eq!(re, 0.0);

    let eta = NavierStokesManifoldEngine::compute_kolmogorov_inertial_efficiency(f64::NAN);
    assert_eq!(eta, 0.0);

    let flux = NavierStokesManifoldEngine::turbulent_cascade_flux(&corrupted);
    assert_eq!(flux, 0.0);
}

#[test]
fn quantum_strategy_contract_compliance() {
    let engine = NavierStokesManifoldEngine::new();
    assert_eq!(engine.name(), "NavierStokesManifoldEngine");
    assert_eq!(engine.horizon(), TradeHorizon::Continuous);

    // Sin registro conectado debe retornar 0.0 seguro sin pánico
    let eval = engine.evaluate();
    assert_eq!(eval, 0.0);
    let eval_coin = engine.evaluate_for_coin(0, "BTCUSDT");
    assert_eq!(eval_coin, 0.0);
}
