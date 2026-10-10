//! CERT-M5-H03 — RIESGO DE RUINA CENTRALIZADO.
//!
//! Antes de este módulo existían DOS protecciones de ruina desconectadas:
//! la exponencial QO-M1.2 (sólo en `kelly.rs`) y el streak-bound FIX #593
//! (sólo en `kelly_envelope.rs`). Los demás productores de fracción de
//! sizing (`kelly_bootstrap_cold`, `micro_kelly`, `leverage_matrix`,
//! margin bumps) no pasaban por NINGUNA.
//!
//! ## Teoría
//!
//! La exponencial de QO-M1.2, P(f) = ((1−f)/(1+f))^(1/f), es la clásica
//! aproximación de gambler's-ruin para juego par arriesgando fracción f
//! por ronda. Propiedad clave (y razón por la que NO sirve como tope
//! superior): es monótona DECRECIENTE en f, con piso asintótico
//! P(f→0) → e⁻² ≈ 13.5%. El bloque que intentaba usarla como tope en
//! `kelly.rs` (f_cap = −ln(P)/2 ≈ 1.50) jamás vinculaba — código muerto.
//! Se conserva aquí como DIAGNÓSTICO documentado.
//!
//! El tope VIGENTE es el streak-bound (FIX #593, generalizado): si se
//! esperan `streak(q)` pérdidas consecutivas dentro del horizonte de
//! trading, la fracción máxima tal que el capital sobreviviente conserva
//! ≥ SURVIVAL_FLOOR es
//!
//!   f_cap = 1 − SURVIVAL_FLOOR^(1/streak),  streak = ln(H)/ln(q)
//!
//! con H = TRADE_HORIZON (200 trades) y q = probabilidad de pérdida
//! (acotada inferiormente por su LCB cuando hay evidencia). Este bound es
//! monótono CRECIENTE en calidad (menos q ⇒ menos streak ⇒ f_cap menor) y
//! vinculante en la zona peligrosa (q=0.6 ⇒ f_cap≈0.25; q=0.75 ⇒ ≈0.10).
//! Encima actúa el axioma absoluto del sistema: riesgo por trade ≤ 25%.

pub use crate::kelly_envelope::{SURVIVAL_FLOOR, TRADE_HORIZON};

/// P(ruina) analítica QO-M1.2 — SOLO diagnóstico (monótona decreciente:
/// no es un tope superior válido; ver docstring del módulo).
pub fn ruin_probability(f: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    if f >= 1.0 {
        return 1.0;
    }
    ((1.0 - f) / (1.0 + f)).powf(1.0 / f)
}

/// Racha de pérdidas consecutivas esperada dentro del horizonte con
/// probabilidad de pérdida `q` (usar el LCB de q cuando exista evidencia).
pub fn expected_loss_streak(q: f64) -> f64 {
    let q_c = q.clamp(0.01, 0.99);
    let raw = (TRADE_HORIZON.ln() / q_c.ln()).abs();
    if raw.is_finite() {
        raw.min(TRADE_HORIZON).max(3.0)
    } else {
        TRADE_HORIZON
    }
}

/// Tope de fracción por racha de pérdidas (FIX #593 generalizado):
/// f tal que (1−f)^streak ≥ SURVIVAL_FLOOR.
pub fn streak_ruin_cap(q: f64) -> f64 {
    (1.0 - SURVIVAL_FLOOR.powf(1.0 / expected_loss_streak(q))).clamp(0.001, 0.50)
}

/// Aplicación uniforme del control de ruina en CUALQUIER productor de
/// fracción de sizing: streak-bound + axioma 25%. `q` = probabilidad de
/// pérdida estimada (LCB si hay evidencia; 0.60 conservador si no).
pub fn clamp_ruin(f: f64, q: f64) -> f64 {
    if !f.is_finite() {
        return 0.0;
    }
    if f <= 0.0 {
        return f;
    }
    f.min(streak_ruin_cap(q)).min(0.25)
}

/// q conservador por defecto cuando no hay evidencia de win-rate: 60%
/// de pérdidas esperadas (streak ≈ 10 ⇒ f_cap ≈ 0.25 — coincide con el
/// axioma: sin evidencia, el tope es el propio axioma).
pub const CONSERVATIVE_Q: f64 = 0.60;

/// Calcula la probabilidad conservadora de pérdida q utilizando la cota inferior
/// de confianza (LCB) del win rate (Jeffreys Beta posterior vía [`crate::evidence::win_rate_lcb`])
/// cuando se dispone de operaciones cerradas (`trades >= 1.0`).
///
/// Si no hay trades o trades < 1.0, retorna [`CONSERVATIVE_Q`] (0.60).
/// Si win_rate_lcb retorna Some(p_lcb), entonces q = (1.0 - p_lcb).clamp(0.01, 0.99),
/// garantizando que micro-muestras no reclamen rachas cortas de pérdida ficticias (R7-R2-A-2).
#[inline]
pub fn conservative_loss_q(win_rate: f64, trades: f64) -> f64 {
    if trades.is_finite() && trades >= 1.0 {
        if let Some(wr_lcb) = crate::evidence::win_rate_lcb(win_rate, trades) {
            return (1.0 - wr_lcb).clamp(0.01, 0.99);
        } else if win_rate.is_finite() && win_rate > 0.0 && win_rate < 1.0 {
            return (1.0 - win_rate).clamp(0.01, 0.99);
        }
    }
    CONSERVATIVE_Q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_is_diagnostic_only_and_documented_monotone() {
        // Piso asintótico e⁻²: la fórmula NO puede usarse como tope superior.
        assert!(ruin_probability(0.001) > 0.13);
        // Decreciente en f — la propiedad que invalida su uso como cap.
        assert!(ruin_probability(0.5) < ruin_probability(0.05));
    }

    #[test]
    fn streak_cap_binds_harder_with_worse_quality() {
        let cap_good = streak_ruin_cap(0.45); // q<0.5: edge alto
        let cap_bad = streak_ruin_cap(0.75); // q>0.5: edge negativo
        assert!(cap_good > cap_bad, "mejor calidad ⇒ tope más permisivo");
        // Sin evidencia (q=0.60): el cap coincide con el axioma ~0.25.
        assert!((streak_ruin_cap(CONSERVATIVE_Q) - 0.25).abs() < 0.02);
    }

    #[test]
    fn clamp_applies_axiom_everywhere() {
        // Fracción absurda de un path sin evidencia queda acotada por axioma.
        assert!(clamp_ruin(0.9, CONSERVATIVE_Q) <= 0.25);
        // f≤0 pasa intacto (los productores lo usan como "sin señal").
        assert_eq!(clamp_ruin(0.0, 0.5), 0.0);
        assert_eq!(clamp_ruin(-0.1, 0.5), -0.1);
        // F1-B4: NaN se neutraliza a 0.0
        assert_eq!(clamp_ruin(f64::NAN, 0.5), 0.0);
    }

    #[test]
    fn test_conservative_loss_q_accounts_for_small_samples() {
        // Sin trades, retorna CONSERVATIVE_Q
        assert_eq!(conservative_loss_q(0.80, 0.0), CONSERVATIVE_Q);

        // Con 2 trades y 100% win-rate, el raw q sería 0.0 (peligro de sobre-apuesta),
        // pero con la LCB de win-rate el q conservador es significativamente alto (> 0.40).
        let q_micro = conservative_loss_q(1.0, 2.0);
        assert!(q_micro > 0.40, "q_micro debe ser conservador ante N=2: {}", q_micro);

        // Con 100 trades y 60% win-rate, q converge hacia ~0.48
        let q_large = conservative_loss_q(0.60, 100.0);
        assert!(q_large > 0.40 && q_large < 0.60, "q_large debe ser razonable: {}", q_large);
    }
}
