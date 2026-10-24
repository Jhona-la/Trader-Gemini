//! Contrato F1-095 & F1-096 (Ola Ω165): Continuidad física en Hawkes / TrendRunner
//! e Incertidumbre Analítica en la Puerta de Persistencia Hurst DFA.
//!
//! TEORÍA Y DOCTRINA:
//!
//! F1-095: Erradicación de escalones discretos en votantes continuos.
//! - HawkesBesselEngine: Antes leía el escritor binario `order_flow_direction` {-1, 0, 1},
//!   lo que fijaba tanh(±1.0) = ±0.7616 constante independientemente de la magnitud real del flujo.
//!   Ahora consume la jerarquía continua Zero-Copy:
//!   `order_book_imbalance` (L1 O(1) en [-1, 1]) -> `order_flow_imbalance` -> `order_flow_direction`.
//!   Pequeños desbalances producen activaciones suaves y proporcionales.
//! - HighPayoffTrendRunner: Antes leía `trend_direction` {-1, 0, 1}. Ahora consume
//!   `ema_trend_swing_z` (z-score canónico tipificado O(1)), modulando tanh(z) de forma suave.
//!
//! F1-096: Calibración de la puerta de persistencia con error típico analítico \sigma_H.
//! - En un proceso puramente Browniano (H = 0.50), las fluctuaciones muestrales generan
//!   estimaciones de Hurst por encima de 0.50 por mero ruido. Si el ancho de la puerta es
//!   fijo y estrecho (0.04), el ruido activa señales espurias de seguimiento de tendencia.
//! - La solución rigurosa computa el error típico analítico de la regresión log-log en DFA:
//!   SE(\hat{H}) = \sqrt{ \frac{s_{yy}(1 - r^2)}{(K-2) s_{xx}} }
//!   y modula el ancho de activación: ancho = (2 * \sigma_H).max(0.04).
//! - Bajo hipótesis nula Browniana con ruido muestral, la expectativa del valor absoluto
//!   del voto espectral es despreciable (< 0.02).

use omniscient_registry::OmniscientRegistry;
use signal_engine::hawkes_bessel::HawkesBesselEngine;
use signal_engine::trend_runner::HighPayoffTrendRunner;
use strategy_core::QuantumStrategy;
use std::sync::Arc;

#[test]
fn f1_095_hawkes_bessel_graduates_continuously_with_order_book_imbalance() {
    let registry = Arc::new(OmniscientRegistry::new());

    // Configurar intensidad de hawkes > 0 para que la excitación sea positiva
    registry.set_scoped("BTCUSDT", "hawkes_intensity", 2.5);

    let mut engine = HawkesBesselEngine::default();
    let _ = engine.init(registry.clone());

    // 1. Con desbalance neutro (0.0), el voto debe ser exactamente 0.0
    registry.set_scoped("BTCUSDT", "order_book_imbalance", 0.0);
    let v_cero = engine.evaluate_for_coin(0, "BTCUSDT");
    assert_eq!(v_cero, 0.0, "Desbalance cero debe producir voto exactamente nulo");

    // 2. Con desbalances crecientes pequeños y medianos, la respuesta debe ser estrictamente monótona
    // y NO saltar a tanh(1.0) = 0.7616
    registry.set_scoped("BTCUSDT", "order_book_imbalance", 0.10);
    let v_pequeno = engine.evaluate_for_coin(0, "BTCUSDT");

    registry.set_scoped("BTCUSDT", "order_book_imbalance", 0.35);
    let v_mediano = engine.evaluate_for_coin(0, "BTCUSDT");

    registry.set_scoped("BTCUSDT", "order_book_imbalance", 0.85);
    let v_grande = engine.evaluate_for_coin(0, "BTCUSDT");

    assert!(v_pequeno > 0.0, "Voto pequeño debe ser positivo: {}", v_pequeno);
    assert!(v_mediano > v_pequeno, "Voto debe crecer monótonamente: mediano {} > pequeno {}", v_mediano, v_pequeno);
    assert!(v_grande > v_mediano, "Voto debe crecer monótonamente: grande {} > mediano {}", v_grande, v_mediano);

    // Certificar que no existe el escalón rígido de ±0.7616 en desbalance pequeño
    // La ratio v_pequeno / v_grande debe aproximar tanh(0.10) / tanh(0.85) ≈ 0.0996 / 0.691 ≈ 0.144
    let ratio_medida = v_pequeno / v_grande;
    let ratio_teorica = (0.10_f64).tanh() / (0.85_f64).tanh();
    assert!(
        (ratio_medida - ratio_teorica).abs() < 1e-4,
        "La graduación continua debe reflejar tanh(imbalance) analíticamente: medida={}, teorica={}",
        ratio_medida,
        ratio_teorica
    );

    // 3. Verificar fallback continuo a order_flow_imbalance si order_book_imbalance es nulo
    registry.set_scoped("BTCUSDT", "order_book_imbalance", 0.0);
    registry.set_scoped("BTCUSDT", "order_flow_imbalance", -0.45);
    let v_fallback_ofi = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(v_fallback_ofi < 0.0, "Debe caer en OFI bajista cuando OBI es nulo: {}", v_fallback_ofi);
}

#[test]
fn f1_095_trend_runner_graduates_continuously_with_ema_trend_swing_z() {
    let registry = Arc::new(OmniscientRegistry::new());

    // Configurar régimen fuertemente persistente (Hurst = 0.75, baja incertidumbre)
    registry.set_scoped("BTCUSDT", "hurst_exponent", 0.75);
    registry.set_scoped("BTCUSDT", "hurst_std_error", 0.02);
    registry.set_scoped("BTCUSDT", "cvpin", 0.30);

    let mut engine = HighPayoffTrendRunner::default();
    let _ = engine.init(registry.clone());

    // 1. Con z = 0, el voto debe ser 0.0
    registry.set_scoped("BTCUSDT", "ema_trend_swing_z", 0.0);
    let v_cero = engine.evaluate_for_coin(0, "BTCUSDT");
    assert_eq!(v_cero, 0.0, "z=0 debe producir voto nulo");

    // 2. Con z creciente, el voto debe graduarse suavemente según tanh(z)
    registry.set_scoped("BTCUSDT", "ema_trend_swing_z", 0.20);
    let v_suave = engine.evaluate_for_coin(0, "BTCUSDT");

    registry.set_scoped("BTCUSDT", "ema_trend_swing_z", 1.50);
    let v_fuerte = engine.evaluate_for_coin(0, "BTCUSDT");

    assert!(v_suave > 0.0, "Voto suave debe ser positivo: {}", v_suave);
    assert!(v_fuerte > v_suave, "Voto fuerte debe superar a voto suave: {} > {}", v_fuerte, v_suave);

    let ratio_medida = v_suave / v_fuerte;
    let ratio_teorica = (0.20_f64).tanh() / (1.50_f64).tanh();
    assert!(
        (ratio_medida - ratio_teorica).abs() < 1e-4,
        "La graduación continua debe reflejar tanh(z) analíticamente: medida={}, teorica={}",
        ratio_medida,
        ratio_teorica
    );
}

#[test]
fn f1_096_trend_runner_persistence_gate_modulated_by_analytical_uncertainty() {
    // Escenario 1: Proceso Browniano ideal (H = 0.50)
    // Cualquiera sea el desplazamiento o incertidumbre, debe abstenerse (todo cero).
    let mut desplazamientos = [0.0; signal_engine::voto_espectral::ESCALAS_VOTO];
    for (i, d) in desplazamientos.iter_mut().enumerate() {
        *d = (i as f64 + 1.0) * 0.1; // Desplazamiento alcista no nulo
    }

    let voto_browniano = HighPayoffTrendRunner::voto_espectral_con_sigma(
        &desplazamientos,
        0.50,
        0.05,
        0.30,
    );
    assert_eq!(
        voto_browniano.media_banda(0, 31).unwrap_or(0.0),
        0.0,
        "En H=0.50 la puerta de persistencia debe cerrarse estrictamente a 0.0"
    );

    // Escenario 2: H = 0.54 (ligero exceso sobre 0.50)
    // Con alta incertidumbre (\sigma_H = 0.12, regresión ruidosa), el peso de persistencia
    // debe ser mucho más atenuado que con alta precisión (\sigma_H = 0.02).
    let voto_ruidoso = HighPayoffTrendRunner::voto_espectral_con_sigma(
        &desplazamientos,
        0.54,
        0.12,
        0.30,
    );
    let voto_preciso = HighPayoffTrendRunner::voto_espectral_con_sigma(
        &desplazamientos,
        0.54,
        0.02,
        0.30,
    );

    let media_ruidosa = voto_ruidoso.media_banda(0, 31).unwrap_or(0.0);
    let media_precisa = voto_preciso.media_banda(0, 31).unwrap_or(0.0);

    assert!(media_precisa > media_ruidosa,
        "Mayor certidumbre en Hurst debe otorgar mayor convicción: precisa={} > ruidosa={}",
        media_precisa, media_ruidosa
    );

    // Escenario 3: Monte Carlo determinista bajo ruido browniano
    // Simulamos 200 realizaciones de fluctuación muestral de Hurst alrededor de 0.50
    // con error típico analítico \sigma_H = 0.06.
    // Verificamos que la expectativa matemática del voto promedio es menor a 0.02.
    let mut suma_abs_votos = 0.0;
    let n_simulaciones = 200;

    for i in 0..n_simulaciones {
        // Pseudo-ruido determinista sin crates externos
        let phase = (i as f64 * 1.618033988749895).sin();
        let hurst_simulado = 0.50 + 0.06 * phase.abs(); // Excursiones de ruido hacia persistencia aparente
        let sigma_simulada = 0.06;

        let mut despl_sim = [0.0; signal_engine::voto_espectral::ESCALAS_VOTO];
        for (k, val) in despl_sim.iter_mut().enumerate() {
            *val = ((i * 32 + k) as f64 * 0.71828).sin();
        }

        let voto = HighPayoffTrendRunner::voto_espectral_con_sigma(
            &despl_sim,
            hurst_simulado,
            sigma_simulada,
            0.40,
        );
        let media = voto.media_banda(0, 31).unwrap_or(0.0);
        suma_abs_votos += media.abs();
    }

    let esperanza_abs_voto = suma_abs_votos / (n_simulaciones as f64);
    assert!(
        esperanza_abs_voto < 0.02,
        "La expectativa de voto espurio bajo ruido browniano H~N(0.5, sigma^2) debe ser < 0.02: medida={}",
        esperanza_abs_voto
    );
}
