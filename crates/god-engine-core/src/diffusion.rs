//! ESTADÍSTICA DE DIFUSIÓN DEL NÚCLEO DE DECISIÓN (D-621 / D-624 / D-685 — DÉCIMA OLA).
//!
//! # El problema
//!
//! El núcleo comparaba magnitudes de mercado contra literales en unidades
//! incompatibles. El escudo macro exigía −3, −8 y −6 puntos básicos a tres
//! tendencias de horizontes distintos: con un ATR del 0,10 % eso equivale a
//! 0,04 σ, 0,24 σ y 0,87 σ, tres niveles de exigencia sin relación entre sí.
//! Las guardas anti-persecución permitían 0,20 ATR a favor y 1,50 en contra,
//! una asimetría de 7,5× que nadie derivó.
//!
//! # El principio
//!
//! Toda tendencia del motor es una distancia relativa del precio a una EMA de
//! velas de 1 minuto, o la diferencia entre dos de esas EMAs. Para un paseo
//! aleatorio con desviación σ por vela, ambas tienen desviación típica
//! estacionaria exacta:
//!
//! ```text
//! e_t = p_t − m_t = (1−α)·(e_{t−1} + Δp_t)   ⟹   Var(e) = (1−α)²·σ² / (α·(2−α))
//! Cov(e_a, e_b)   = (1−a)(1−b)·σ² / (1 − (1−a)(1−b))
//! Var(m_f − m_s)  = Var(e_f) + Var(e_s) − 2·Cov(e_f, e_s)
//! ```
//!
//! La σ de 1 minuto sale del ATR de 1 minuto con el factor de Parkinson: el
//! rango medio de una vela browniana es `√(8/π)·σ`. Dividir por esa desviación
//! convierte cada magnitud en un z comparable entre horizontes, y un único
//! umbral —el z bilateral del 95 %, el mismo del prior del win rate (D-680)—
//! sustituye a todos los literales.
//!
//! Durante el calentamiento, antes de que existan las EMAs de velas, el motor
//! recurre a EMAs de ticks: ahí el z es aproximado. Tras el calentamiento, las
//! barras de este módulo son exactamente las del `StatefulEngine`.

/// Rango medio de una vela browniana en desviaciones típicas: `√(8/π)`.
pub const PARKINSON_RANGE_FACTOR: f64 = 1.595_769_121_605_730_8;

/// z bilateral del 95 %.
pub const Z95: f64 = 1.959_963_984_540_054;

/// Barras de `kline_ema_fast`.
pub const EMA_FAST_BARS: f64 = 9.0;
/// Barras de `kline_ema_slow`.
pub const EMA_SLOW_BARS: f64 = 21.0;
/// Barras de `kline_ema_trend` (2 h).
pub const EMA_TREND_BARS: f64 = 120.0;
/// Barras de `kline_ema_macro` (12 h).
pub const EMA_MACRO_BARS: f64 = 720.0;

#[inline]
fn alpha(bars: f64) -> f64 {
    2.0 / (bars.max(1.0) + 1.0)
}

/// σ de 1 minuto, como fracción del precio, a partir del ATR relativo de 1 minuto.
#[inline]
pub fn sigma_from_atr(atr_ratio: f64) -> f64 {
    if atr_ratio.is_finite() && atr_ratio > 0.0 {
        atr_ratio / PARKINSON_RANGE_FACTOR
    } else {
        0.0
    }
}

/// Desviación típica estacionaria de `p − EMA_n`, en unidades de σ.
#[inline]
pub fn ema_distance_sd(bars: f64) -> f64 {
    let a = alpha(bars);
    (1.0 - a) / (a * (2.0 - a)).sqrt()
}

/// Desviación típica estacionaria de `EMA_fast − EMA_slow`, en unidades de σ.
#[inline]
pub fn ema_spread_sd(fast_bars: f64, slow_bars: f64) -> f64 {
    let a = alpha(fast_bars);
    let b = alpha(slow_bars);
    let va = (1.0 - a).powi(2) / (a * (2.0 - a));
    let vb = (1.0 - b).powi(2) / (b * (2.0 - b));
    let q = (1.0 - a) * (1.0 - b);
    let cov = q / (1.0 - q);
    (va + vb - 2.0 * cov).max(0.0).sqrt()
}

/// z con signo de una distancia relativa `(p − EMA)/EMA`.
pub fn ema_distance_z(rel_distance: f64, atr_ratio: f64, bars: f64) -> f64 {
    let sd = sigma_from_atr(atr_ratio) * ema_distance_sd(bars);
    if !rel_distance.is_finite() || sd <= 0.0 {
        0.0
    } else {
        rel_distance / sd
    }
}

/// z con signo de un diferencial relativo `(EMA_f − EMA_s)/EMA_s`.
pub fn ema_spread_z(rel_spread: f64, atr_ratio: f64, fast_bars: f64, slow_bars: f64) -> f64 {
    let sd = sigma_from_atr(atr_ratio) * ema_spread_sd(fast_bars, slow_bars);
    if !rel_spread.is_finite() || sd <= 0.0 {
        0.0
    } else {
        rel_spread / sd
    }
}

/// z con signo de una distancia a la EMA expresada en ATR (`price_stretch`).
#[inline]
pub fn atr_stretch_z(stretch_atr: f64, bars: f64) -> f64 {
    if !stretch_atr.is_finite() {
        return 0.0;
    }
    stretch_atr * PARKINSON_RANGE_FACTOR / ema_distance_sd(bars)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Desviación típica simulada de `p − EMA_fast` o de `EMA_fast − EMA_slow`
    /// sobre un paseo aleatorio gaussiano de σ = 1 (semilla fija).
    fn simulated_sd(fast: f64, slow: Option<f64>, steps: usize) -> f64 {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let af = 2.0 / (fast + 1.0);
        let asl = slow.map(|s| 2.0 / (s + 1.0));
        let (mut p, mut mf, mut ms) = (0.0f64, 0.0f64, 0.0f64);
        let burn_in = 50_000;
        let (mut n, mut sum, mut sum2) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..steps {
            let (u1, u2) = (uniform(), uniform());
            p += (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mf = (1.0 - af) * mf + af * p;
            if let Some(a) = asl {
                ms = (1.0 - a) * ms + a * p;
            }
            if i >= burn_in {
                let x = if asl.is_some() { mf - ms } else { p - mf };
                n += 1.0;
                sum += x;
                sum2 += x * x;
            }
        }
        let mean = sum / n;
        (sum2 / n - mean * mean).sqrt()
    }

    #[test]
    fn la_distancia_a_la_ema_coincide_con_la_simulacion() {
        for &(bars, tol) in &[(21.0, 0.03), (120.0, 0.05)] {
            let teorica = ema_distance_sd(bars);
            let simulada = simulated_sd(bars, None, 1_500_000);
            assert!(
                (simulada / teorica - 1.0).abs() < tol,
                "EMA {bars}: teórica {teorica:.4} vs simulada {simulada:.4}"
            );
        }
    }

    #[test]
    fn el_diferencial_entre_emas_coincide_con_la_simulacion() {
        let teorica = ema_spread_sd(EMA_FAST_BARS, EMA_SLOW_BARS);
        let simulada = simulated_sd(EMA_FAST_BARS, Some(EMA_SLOW_BARS), 1_500_000);
        assert!(
            (simulada / teorica - 1.0).abs() < 0.03,
            "EMA 9/21: teórica {teorica:.4} vs simulada {simulada:.4}"
        );
    }

    /// Los tres literales del escudo macro expresados en σ, con ATR del 0,10 %:
    /// la incoherencia que D-624 documenta.
    #[test]
    fn los_umbrales_del_escudo_antiguo_eran_incoherentes() {
        let atr = 0.0010;
        let secular = ema_distance_z(-0.0003, atr, EMA_MACRO_BARS);
        let higher = ema_distance_z(-0.0008, atr, EMA_TREND_BARS);
        let macro_ = ema_spread_z(-0.0006, atr, EMA_FAST_BARS, EMA_SLOW_BARS);
        assert!((secular + 0.036).abs() < 0.005, "secular {secular}");
        assert!((higher + 0.235).abs() < 0.01, "superior {higher}");
        assert!((macro_ + 0.87).abs() < 0.03, "macro {macro_}");
    }

    #[test]
    fn el_stretch_en_atr_equivale_a_la_distancia_relativa() {
        let atr = 0.0015;
        let stretch = -1.2;
        let a = atr_stretch_z(stretch, EMA_SLOW_BARS);
        let b = ema_distance_z(stretch * atr, atr, EMA_SLOW_BARS);
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn sin_volatilidad_medida_el_z_es_neutro() {
        assert_eq!(ema_distance_z(0.05, 0.0, EMA_MACRO_BARS), 0.0);
        assert_eq!(ema_spread_z(f64::NAN, 0.001, 9.0, 21.0), 0.0);
        assert_eq!(atr_stretch_z(f64::INFINITY, 21.0), 0.0);
    }
}
