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
            // Adaptación de umbrales dinámicos (Cero Valores Fijos)
            // Hurst va de 0 a 1. 0.5 es ruido. > 0.6 es tendencia.
            // Si hay una tendencia extremadamente fuerte (Hurst > 0.8), reducimos el periodo para acelerar el cruce.

            use std::sync::atomic::Ordering;

            let base_hurst = arena.config.hurst_swing_threshold.load(Ordering::Relaxed); // Eliminar clamp(0.4, 0.7)
            let dynamic_multiplier = if hurst > base_hurst {
                1.0 - (hurst - base_hurst).max(0.0)
            } else {
                1.0
            };

            // Mutar el alpha dinámicamente en O(1) leyendo el genoma en tiempo real (Cero Sesgo Humano)
            let fast_period = arena.config.ema_fast_period.load(Ordering::Relaxed);
            let slow_period = arena.config.ema_slow_period.load(Ordering::Relaxed);

            self.fast_ewma.alpha = 2.0 / ((fast_period * dynamic_multiplier).max(2.0) + 1.0);
            self.slow_ewma.alpha = 2.0 / ((slow_period * dynamic_multiplier).max(5.0) + 1.0);
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
            let std_dev = variance.sqrt().max(price * 0.0002); // 2 bps floor
            let z_score = diff / std_dev;
            let macd_diff = (fast_val - slow_val) / slow_val;

            // FASE 3: Generación de Señales de Alta Confianza (Axioma II)
            let hurst = arena.coins[0].hurst_exponent.load(Ordering::Relaxed);
            let trend_threshold = arena.config.trend_threshold.load(Ordering::Relaxed);
            let z_thresh = arena.config.turbo_z_score_stdev.load(Ordering::Relaxed);

            // ML Integration (DarkAlpha)
            let ml_pred = ml_pred;
            let ml_long = arena.config.ml_threshold_long.load(Ordering::Relaxed);
            let ml_short = arena.config.ml_threshold_short.load(Ordering::Relaxed);

            if hurst < trend_threshold {
                let conf_fallback = arena
                    .config
                    .explosive_confidence_threshold
                    .load(Ordering::Relaxed);
                if z_score > z_thresh || ml_pred < (0.5 - ml_short) {
                    let conf = if ml_pred < (0.5 - ml_short) {
                        (0.5 - ml_pred) * 2.0
                    } else {
                        conf_fallback
                    };
                    return SignalIntent {
                        signal: SignalType::Short, // Sobrecomprado, vender
                        confidence: conf,
                        expected_duration_ms: (arena
                            .config
                            .base_duration_ms
                            .load(Ordering::Relaxed)
                            * 60.0) as u64,
                        horizon: crate::TradeHorizon::Swing,
                        ..Default::default()
                    };
                } else if z_score < -z_thresh || ml_pred > (0.5 + ml_long) {
                    let conf = if ml_pred > (0.5 + ml_long) {
                        (ml_pred - 0.5) * 2.0
                    } else {
                        conf_fallback
                    };
                    return SignalIntent {
                        signal: SignalType::Long, // Sobrevendido, comprar
                        confidence: conf,
                        expected_duration_ms: (arena
                            .config
                            .base_duration_ms
                            .load(Ordering::Relaxed)
                            * 60.0) as u64,
                        horizon: crate::TradeHorizon::Swing,
                        ..Default::default()
                    };
                }
            } else {
                // Si hay tendencia fuerte (hurst > threshold), seguimos la tendencia
                let swing_tp = arena.config.swing_tp_base.load(Ordering::Relaxed);
                let threshold = (swing_tp * 0.25) * (1.0 / hurst.max(0.1));

                if macd_diff > threshold || ml_pred > (0.5 + ml_long) {
                    return SignalIntent {
                        signal: SignalType::Long,
                        confidence: ((macd_diff.abs() * hurst * 100.0).max((ml_pred - 0.5) * 2.0))
                            .tanh(),
                        expected_duration_ms: (arena
                            .config
                            .base_duration_ms
                            .load(Ordering::Relaxed)
                            * 60.0) as u64,
                        horizon: crate::TradeHorizon::Swing,
                        ..Default::default()
                    };
                } else if macd_diff < -threshold || ml_pred < (0.5 - ml_short) {
                    return SignalIntent {
                        signal: SignalType::Short,
                        confidence: ((macd_diff.abs() * hurst * 100.0).max((0.5 - ml_pred) * 2.0))
                            .tanh(),
                        expected_duration_ms: (arena
                            .config
                            .base_duration_ms
                            .load(Ordering::Relaxed)
                            * 60.0) as u64,
                        horizon: crate::TradeHorizon::Swing,
                        ..Default::default()
                    };
                }
            }

            SignalIntent::flat()
        })
    }
}

impl Default for SwingEngine {
    fn default() -> Self {
        // Valores default para swing: Fast 12, Slow 26 (Standard MACD settings)
        Self::new(12.0, 26.0)
    }
}
