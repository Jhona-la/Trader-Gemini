//! Contrato de simetría direccional en modulación espectral del colchón de margen libre
//! y verificación de publicación de métricas de información sobre el símplex continuo (Ola Ω59 / Ficha #694).

use risk_engine::regime::SpectralMarketRegime;
use omniscient_registry::OmniscientRegistry;

/// Verifica la simetría especular exacta en la modulación del capital libre
/// ante mareas y flujos adversos tanto para posiciones Long como Short.
#[test]
fn cushion_directional_symmetry_exact_mirror() {
    let initial_free_cap = 10.0f64;

    let modulate = |is_long: bool, tide: f64, flux: f64| -> f64 {
        let mut free_cap = initial_free_cap;
        let directional_flux = flux.clamp(0.0, 1.0);
        if directional_flux > 0.0 {
            let adverse_tide = if is_long { tide < 0.0 } else { tide > 0.0 };
            if adverse_tide {
                free_cap *= (1.0 - 0.95 * directional_flux).clamp(0.05, 1.0);
            }
        }
        free_cap
    };

    let test_fluxes = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    let test_tides = [0.0, 0.05, 0.2, 0.6, 0.95];

    for &flux in &test_fluxes {
        for &tide in &test_tides {
            // Caso 1: Marea contraria (bajista para Long, alcista para Short)
            let cap_long_adverse = modulate(true, -tide, flux);
            let cap_short_adverse = modulate(false, tide, flux);
            assert!(
                (cap_long_adverse - cap_short_adverse).abs() < 1e-12,
                "Simetría rota en marea adversa: long({:.2}) = {:.4}, short({:.2}) = {:.4} para flux {:.2}",
                -tide, cap_long_adverse, tide, cap_short_adverse, flux
            );

            // Caso 2: Marea favorable (alcista para Long, bajista para Short)
            let cap_long_favorable = modulate(true, tide, flux);
            let cap_short_favorable = modulate(false, -tide, flux);
            assert!(
                (cap_long_favorable - cap_short_favorable).abs() < 1e-12,
                "Simetría rota en marea favorable: long({:.2}) = {:.4}, short({:.2}) = {:.4} para flux {:.2}",
                tide, cap_long_favorable, -tide, cap_short_favorable, flux
            );
            assert_eq!(
                cap_long_favorable, initial_free_cap,
                "Marea favorable no debe penalizar el margen libre"
            );

            // En marea no nula y flujo positivo, la marea adversa debe contraer el capital
            if tide > 0.0 && flux > 0.0 {
                assert!(
                    cap_long_adverse < initial_free_cap,
                    "Flujo adverso debe contraer margen para Long"
                );
                assert!(
                    cap_short_adverse < initial_free_cap,
                    "Flujo adverso debe contraer margen para Short"
                );
            }
        }
    }
}

/// Verifica que la cota inferior de saturación del colchón (0.05) se respete
/// y nunca colapse a cero ni genere valores negativos o no finitos.
#[test]
fn cushion_saturation_floor_and_nonfinite_safety() {
    let initial_free_cap = 13.0f64; // Invariante de microcapital

    let modulate = |is_long: bool, tide: f64, flux: f64| -> f64 {
        let mut free_cap = initial_free_cap;
        let directional_flux = flux.clamp(0.0, 1.0);
        if directional_flux > 0.0 {
            let adverse_tide = if is_long { tide < 0.0 } else { tide > 0.0 };
            if adverse_tide {
                free_cap *= (1.0 - 0.95 * directional_flux).clamp(0.05, 1.0);
            }
        }
        free_cap
    };

    // Saturación máxima (flux = 1.0)
    let min_cap_long = modulate(true, -1.0, 1.0);
    let min_cap_short = modulate(false, 1.0, 1.0);
    let expected_min = initial_free_cap * 0.05;

    assert!((min_cap_long - expected_min).abs() < 1e-12);
    assert!((min_cap_short - expected_min).abs() < 1e-12);

    // Protección ante flujos anómalos / fuera de rango
    let clamped_over = modulate(true, -1.0, 50.0);
    assert!((clamped_over - expected_min).abs() < 1e-12);

    let negative_flux = modulate(true, -1.0, -2.0);
    assert_eq!(negative_flux, initial_free_cap);
}

/// Verifica la publicación de métricas de teoría de información espectral continua en OmniscientRegistry
#[test]
fn continuous_spectral_information_registry_publication() {
    let registry = OmniscientRegistry::new();

    // Símplex de prueba: p_range=0.35, p_bull=0.45, p_crash=0.10, p_chaos=0.10
    let spectral_regime = SpectralMarketRegime::new(0.35, 0.45, 0.10, 0.10);

    let shannon = spectral_regime.shannon_entropy();
    let renyi_collision = spectral_regime.renyi_entropy(2.0);
    let dir_bias = spectral_regime.directional_bias();
    let turb = spectral_regime.turbulence_index();
    let map_u8: u8 = spectral_regime.map_discrete().into();

    registry.set("market_regime_p_range", spectral_regime.p_range);
    registry.set("market_regime_p_bull", spectral_regime.p_bull);
    registry.set("market_regime_p_crash", spectral_regime.p_crash);
    registry.set("market_regime_p_chaos", spectral_regime.p_chaos);
    registry.set("market_regime_shannon_entropy", shannon);
    registry.set("market_regime_renyi_entropy", renyi_collision);
    registry.set("market_regime_directional_bias", dir_bias);
    registry.set("market_regime_turbulence_index", turb);

    // Comprobaciones
    assert_eq!(map_u8, 1u8); // BullRun es el modo MAP (0.45 > 0.35)
    assert_eq!(registry.get_value_fast("market_regime_p_bull"), Some(0.45));
    assert_eq!(registry.get_value_fast("market_regime_p_range"), Some(0.35));
    assert_eq!(registry.get_value_fast("market_regime_p_crash"), Some(0.10));
    assert_eq!(registry.get_value_fast("market_regime_p_chaos"), Some(0.10));

    // Teorema fundamental de Rényi: H_2 <= H_1 (la entropía de colisión acota inferiormente a Shannon)
    assert!(renyi_collision <= shannon + 1e-9);
    assert!(shannon > 0.0);
    assert!(renyi_collision > 0.0);

    // Polarización direccional: p_bull - p_crash = 0.45 - 0.10 = +0.35
    assert!((dir_bias - 0.35).abs() < 1e-12);
    assert_eq!(registry.get_value_fast("market_regime_directional_bias"), Some(0.35));

    // Turbulencia: p_crash + p_chaos = 0.10 + 0.10 = 0.20
    assert!((turb - 0.20).abs() < 1e-12);
    assert_eq!(registry.get_value_fast("market_regime_turbulence_index"), Some(0.20));
}
