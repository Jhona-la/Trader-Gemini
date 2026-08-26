use crate::{SignalIntent, SignalType};
use feature_engine::Ewma;

/// Motor de Swing (Axioma V: Aislamiento)
/// Operaciones macro basadas en tendencias y cruce de promedios exponenciales (O(1)).
pub struct SwingEngine {
    fast_ewma: Ewma,
    slow_ewma: Ewma,
    variance_ewma: Ewma, // Para calcular el StdDev (BBands)
}

impl SwingEngine {
    pub fn new(fast_period: f64, slow_period: f64) -> Self {
        Self {
            fast_ewma: Ewma::from_period(fast_period),
            slow_ewma: Ewma::from_period(slow_period),
            variance_ewma: Ewma::from_period(fast_period),
        }
    }

    /// Evalúa la macroestructura (tendencia) y retorna una intención.
    /// Actualiza su estado interno matemático.
    #[inline(always)]
    pub fn evaluate_trend(
        &mut self,
        price: f64,
        hurst: f64,
        trend_threshold: f64,
        ml_pred: f64,
        arena: &quantum_arena::GlobalArena,
    ) -> SignalIntent {
        telemetry_server::profile_node!("SwingEngine::evaluate_trend", {
            if price <= 0.0 || !price.is_finite() || !hurst.is_finite() || !trend_threshold.is_finite() || !ml_pred.is_finite() {
                return SignalIntent::flat();
            }
            // Adaptación de umbrales dinámicos (Cero Valores Fijos)
            // Hurst va de 0 a 1. 0.5 es ruido. > 0.6 es tendencia.
            // Si hay una tendencia extremadamente fuerte (Hurst > 0.8), reducimos el periodo para acelerar el cruce.

            use std::sync::atomic::Ordering;

            let base_hurst = arena.config.hurst_swing_threshold.load(Ordering::Relaxed);
            let regime_period_adj = if hurst > base_hurst {
                1.0 - (hurst - base_hurst) * 0.5
            } else {
                1.0 + (base_hurst - hurst) * 0.5
            };

            // Mutar el alpha dinámicamente en O(1) leyendo el genoma en tiempo real (Cero Sesgo Humano)
            let fast_period = arena.config.ema_fast_period.load(Ordering::Relaxed);
            let slow_period = arena.config.ema_slow_period.load(Ordering::Relaxed);

            self.fast_ewma.alpha = 2.0 / ((fast_period * regime_period_adj).clamp(3.0, 100.0) + 1.0);
            self.slow_ewma.alpha = 2.0 / ((slow_period * regime_period_adj).clamp(10.0, 300.0) + 1.0);
            self.variance_ewma.alpha = self.fast_ewma.alpha; // Mismo alpha para la varianza

            let fast_val = self.fast_ewma.update(price);
            let slow_val = self.slow_ewma.update(price);

            let diff = price - fast_val;
            let variance = self.variance_ewma.update(diff * diff);

            // Necesitamos esperar a que ambas estén inicializadas
            if !self.slow_ewma.is_initialized {
                return SignalIntent::flat();
            }

            // Mean Reversion con Bandas de Bollinger (Alta probabilidad en 1m)
            let _ma = fast_val; // Reemplazamos variable unused ma = fast_val por _ma

            // FASE 2.5: Welford-like variance (simplified EWMA)
            // FIX #661: Sanitizar varianza para evitar raíces cuadradas negativas o NaNs
            let safe_var = if variance.is_finite() && variance > 0.0 { variance } else { 0.0 };
            let std_dev = safe_var.sqrt().max(price * 0.0002); // 2 bps floor
            let z_score = if std_dev > 0.0 { diff / std_dev } else { 0.0 };
            if !z_score.is_finite() {
                return SignalIntent::flat();
            }
            let macd_diff = (fast_val - slow_val) / slow_val;

            // FASE 3: Generación de Señales de Alta Confianza (Axioma II)
            let z_thresh = arena.config.turbo_z_score_stdev.load(Ordering::Relaxed).clamp(1.2, 3.0);

            // ML Integration (DarkAlpha)
            let ml_long = arena.config.ml_threshold_long.load(Ordering::Relaxed);
            let ml_short = arena.config.ml_threshold_short.load(Ordering::Relaxed);

            // FIX #1512: Sanitización de duración de señal de Swing (mínimo 30 minutos)
            let raw_base = arena.config.base_duration_ms.load(Ordering::Relaxed);
            let swing_duration_ms = if raw_base.is_finite() && raw_base > 0.0 {
                (raw_base * 60.0) as u64
            } else {
                3_600_000
            }.max(1_800_000);

            if hurst < trend_threshold {
                let conf_fallback = arena
                    .config
                    .explosive_confidence_threshold
                    .load(Ordering::Relaxed);

                // Mean Reversion en Rango (Hurst < threshold):
                if z_score > z_thresh && price > fast_val && macd_diff <= 0.0005 && ml_pred <= ml_long {
                    let conf = if ml_pred <= ml_short {
                        ((z_score / z_thresh) * 0.5 + (0.5 - ml_pred) * 2.0 * 0.5).clamp(0.1, 1.0)
                    } else {
                        conf_fallback
                    };
                    return SignalIntent {
                        signal: SignalType::Short, // Sobrecomprado en rango, venta a media
                        confidence: conf,
                        expected_duration_ms: swing_duration_ms,
                        horizon: crate::TradeHorizon::Swing,
                        ..Default::default()
                    };
                } else if z_score < -z_thresh && price >= slow_val && macd_diff >= -0.0005 && ml_pred >= ml_short {
                    let conf = if ml_pred >= ml_long {
                        ((z_score.abs() / z_thresh) * 0.5 + (ml_pred - 0.5) * 2.0 * 0.5).clamp(0.1, 1.0)
                    } else {
                        conf_fallback
                    };
                    return SignalIntent {
                        signal: SignalType::Long, // Sobrevendido en rango, compra a media
                        confidence: conf,
                        expected_duration_ms: swing_duration_ms,
                        horizon: crate::TradeHorizon::Swing,
                        ..Default::default()
                    };
                }
            } else {
                // Tendencia Fuerte (Hurst >= threshold):
                let swing_tp = arena.config.swing_tp_base.load(Ordering::Relaxed);
                let threshold = (swing_tp * 0.003).max(0.0001) * (1.0 / hurst.max(0.1));

                // Escalar convicción MACD de forma continua y suave
                if macd_diff > threshold && ml_pred >= ml_short {
                    let raw_conf = (macd_diff.abs() * hurst * 50.0).max((ml_pred - 0.5).max(0.0) * 2.0);
                    let confidence = if raw_conf.is_finite() { raw_conf.tanh().clamp(0.1, 1.0) } else { 0.5 };
                    return SignalIntent {
                        signal: SignalType::Long,
                        confidence,
                        expected_duration_ms: swing_duration_ms,
                        horizon: crate::TradeHorizon::Swing,
                        ..Default::default()
                    };
                } else if macd_diff < -threshold && ml_pred <= ml_long {
                    let raw_conf = (macd_diff.abs() * hurst * 50.0).max((0.5 - ml_pred).max(0.0) * 2.0);
                    let confidence = if raw_conf.is_finite() { raw_conf.tanh().clamp(0.1, 1.0) } else { 0.5 };
                    return SignalIntent {
                        signal: SignalType::Short,
                        confidence,
                        expected_duration_ms: swing_duration_ms,
                        horizon: crate::TradeHorizon::Swing,
                        ..Default::default()
                    };
                }
            }

            SignalIntent {
                signal: SignalType::Flat,
                horizon: crate::TradeHorizon::Swing,
                ..Default::default()
            }
        })
    }
}

impl Default for SwingEngine {
    fn default() -> Self {
        // Valores default para swing: Fast 12, Slow 26 (Standard MACD settings)
        Self::new(12.0, 26.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swing_engine_trend_evaluation() {
        let mut engine = SwingEngine::default();
        let arena = quantum_arena::GlobalArena::new(13.0);

        let mut p = 100.0;
        for i in 0..40 {
            p += 0.5;
            let intent = engine.evaluate_trend(p, 0.75, 0.60, 0.50, &arena);
            if i >= 35 {
                assert_eq!(intent.horizon, crate::TradeHorizon::Swing);
            }
        }
    }
}
