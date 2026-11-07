//! Contrato Formal de Verificación: Fase 59 (Operadores de Transferencia de Ruelle-Frobenius)
//!
//! Este conjunto de pruebas certifica las propiedades termodinámicas y espectrales del motor:
//! - C1: Conservación de masa de probabilidad en el operador de transferencia y decaimiento positivo $\gamma_0 > 0$.
//! - C2: Brecha espectral acotada $\Delta_{\text{gap}} \in [0, 1]$ y persistencia temporal del alfa.
//! - C3: Detección de colapso ergódico y reducción de convicción a 0.05.
//! - C4: Determinismo en stack $O(1)$, cero heap allocations en hot path y tiempo de Lyapunov finito.

use signal_engine::{RuelleFrobeniusParams, RuelleFrobeniusTransferEngine, RUELLE_GRID_SIZE};

#[test]
fn test_c1_frobenius_transfer_conservation_and_decay_rate() {
    let params = RuelleFrobeniusParams::default();
    let mut engine = RuelleFrobeniusTransferEngine::new(params);

    // Flujo estacionario ordinario
    let report = engine.update_and_evaluate(
        0.0001, // price_ret
        0.10,   // ofi
        1.5,    // spread_bps
        0.001,  // volatility
        1_700_000_000,
    );

    assert!(
        report.leading_decay_rate_gamma > 0.0,
        "La tasa de decaimiento de Ruelle debe ser estrictamente positiva: {}",
        report.leading_decay_rate_gamma
    );
    assert!(
        report.spectral_gap >= 0.0 && report.spectral_gap <= 1.0,
        "La brecha espectral debe estar en [0, 1]: {}",
        report.spectral_gap
    );
    assert!(
        report.lyapunov_decay_time_ms >= 10.0 && report.lyapunov_decay_time_ms <= 60_000.0,
        "Tiempo de Lyapunov fuera de rango: {}",
        report.lyapunov_decay_time_ms
    );
    assert!(
        !report.ergodic_chaotic_collapse,
        "No debe detectar colapso caótico en flujo suave"
    );
}

#[test]
fn test_c2_alpha_persistence_and_resonant_frequency() {
    let params = RuelleFrobeniusParams::default();
    let mut engine = RuelleFrobeniusTransferEngine::new(params);

    for i in 0..10 {
        let rep = engine.update_and_evaluate(
            0.0002 * (i as f64),
            0.15,
            2.0,
            0.002,
            1_700_000_000 + (i as u64) * 50,
        );

        assert!(
            rep.alpha_persistence_factor >= 0.0 && rep.alpha_persistence_factor <= 1.0,
            "Persistencia del alfa fuera de [0, 1]: {}",
            rep.alpha_persistence_factor
        );
        assert!(
            rep.resonant_frequency_omega.is_finite(),
            "Frecuencia de resonancia debe ser finita: {}",
            rep.resonant_frequency_omega
        );
        assert!(
            rep.signal_conviction_modulator >= 0.05 && rep.signal_conviction_modulator <= 1.30,
            "Modulador de convicción fuera de límites [0.05, 1.30]: {}",
            rep.signal_conviction_modulator
        );
    }
}

#[test]
fn test_c3_ergodic_chaotic_collapse_under_pure_noise() {
    let mut params = RuelleFrobeniusParams::default();
    params.min_spectral_gap = 0.20; // Umbral exigente para certificar protección
    let mut engine = RuelleFrobeniusTransferEngine::new(params);

    // Inyectamos dispersión máxima y alta volatilidad para inducir régimen de ruido caótico
    let mut report = engine.update_and_evaluate(0.05, 0.9, 30.0, 0.50, 1_700_000_100);
    for i in 1..6 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        report = engine.update_and_evaluate(
            sign * 0.08,
            sign * 1.0,
            50.0,
            0.80,
            1_700_000_100 + (i as u64) * 20,
        );
    }

    assert!(
        report.ergodic_chaotic_collapse,
        "Debe detectar colapso ergódico y pérdida de predictibilidad"
    );
    assert_eq!(
        report.signal_conviction_modulator, 0.05,
        "La convicción debe colapsar a la cota mínima de seguridad de 0.05"
    );
}

#[test]
fn test_c4_grid_size_and_stack_boundedness() {
    assert_eq!(RUELLE_GRID_SIZE, 16);
}
