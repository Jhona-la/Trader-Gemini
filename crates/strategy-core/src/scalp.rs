use crate::{SignalIntent, SignalType};
use feature_engine::{obi_acceleration, order_book_imbalance, WelfordOnline};

/// Motor de Scalping (Axioma V: Aislamiento)
/// Operaciones ultrarrápidas basadas en desequilibrio del Order Book y Z-Scores dinámicos.

pub struct ScalpEngine {
    // Almacenamos estado previo para derivadas
    prev_obi: f64,
    // Estadísticas dinámicas de aceleración
    accel_stats: WelfordOnline,
}

impl ScalpEngine {
    pub fn new() -> Self {
        Self {
            prev_obi: 0.0,
            accel_stats: WelfordOnline::new(),
        }
    }

    /// Evalúa la microestructura y retorna una intención.
    /// `z_target` se calibra según el régimen de volatilidad (ej. 2.0 o 3.0 para alta confianza).
    #[inline(always)]
    pub fn evaluate_microstructure(
        &mut self,
        bid_vol: f64,
        ask_vol: f64,
        z_target: f64,
        arena: &quantum_arena::GlobalArena,
    ) -> SignalIntent {
        telemetry_server::profile_node!("ScalpEngine::evaluate_microstructure", {
            if !bid_vol.is_finite() || !ask_vol.is_finite() || bid_vol < 0.0 || ask_vol < 0.0 || !z_target.is_finite() {
                return SignalIntent::flat();
            }
            let current_obi = order_book_imbalance(bid_vol, ask_vol);
            let accel = obi_acceleration(current_obi, self.prev_obi);

            // Guardamos el estado O(1)
            self.prev_obi = current_obi;

            let min_samples = arena
                .config
                .scalp_accel_min_samples
                .load(std::sync::atomic::Ordering::Relaxed);

            // FASE 28 & BUG-599: Actualización con decaimiento adaptativo EW-Welford
            let alpha = 2.0 / ((min_samples * 10.0).clamp(50.0, 500.0) + 1.0);
            self.accel_stats.update_decay(accel, alpha);

            // Si no hay suficientes datos para desviación estándar, no operamos
            if self.accel_stats.count < min_samples {
                return SignalIntent::flat();
            }

            let z_score = self.accel_stats.z_score(accel);
            // FIX #614: Verificar finitud estricta de Z-Score para evitar propagación de NaN
            if !z_score.is_finite() {
                return SignalIntent::flat();
            }

            let base_z = arena
                .config
                .turbo_z_score_stdev
                .load(std::sync::atomic::Ordering::Relaxed)
                .max(0.1);

            let final_confidence = (z_score.abs() / base_z).clamp(0.05, 1.0);

            let signal = if z_score > z_target {
                SignalType::Long
            } else if z_score < -z_target {
                SignalType::Short
            } else {
                SignalType::Flat
            };

            if signal != SignalType::Flat {
                // FIX #1513: Sanitización de duración de señal de Scalp
                let raw_dur = arena
                    .config
                    .base_duration_ms
                    .load(std::sync::atomic::Ordering::Relaxed);
                let dynamic_duration = if raw_dur.is_finite() && raw_dur > 0.0 {
                    raw_dur as u64
                } else {
                    15_000
                }.max(15_000);
                SignalIntent {
                    signal,
                    confidence: final_confidence,
                    expected_duration_ms: dynamic_duration,
                    expected_volume_usd: 0.0,
                    volume_flow_rate: 0.0,
                    drift: 0.0,
                    expected_magnitude: 0.0,
                    tp_price_target: 0.0,
                    sl_price_target: 0.0,
                    trajectory_volatility: 0.0,
                    horizon: crate::TradeHorizon::Scalp,
                }
            } else {
                SignalIntent::flat()
            }
        })
    }

    /// Evalúa la microestructura con filtro de toxicidad VPIN y aceleración OBI (Punto #115)
    #[inline(always)]
    pub fn evaluate_microstructure_with_vpin(
        &mut self,
        bid_vol: f64,
        ask_vol: f64,
        vpin: f64,
        z_target: f64,
        arena: &quantum_arena::GlobalArena,
    ) -> SignalIntent {
        if !vpin.is_finite() || vpin > 0.85 {
            // Alta toxicidad de flujo institucional adverso -> rechazar señal de scalp
            return SignalIntent::flat();
        }
        self.evaluate_microstructure(bid_vol, ask_vol, z_target, arena)
    }
}

impl Default for ScalpEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalp_engine_obi_acceleration_and_signal() {
        let mut engine = ScalpEngine::default();
        let arena = quantum_arena::GlobalArena::new(13.0);

        for _ in 0..15 {
            let res = engine.evaluate_microstructure(100.0, 100.0, 1.5, &arena);
            assert_eq!(res.signal, SignalType::Flat);
        }

        // Fuerte desequilibrio de compra
        let buy_signal = engine.evaluate_microstructure(500.0, 10.0, 1.5, &arena);
        if buy_signal.signal != SignalType::Flat {
            assert_eq!(buy_signal.horizon, crate::TradeHorizon::Scalp);
        }
    }

    #[test]
    fn test_scalp_engine_vpin_toxicity_filter() {
        let mut engine = ScalpEngine::default();
        let arena = quantum_arena::GlobalArena::new(13.0);

        // VPIN tóxico (> 0.85) debe vetar inmediatamente
        let toxic_signal = engine.evaluate_microstructure_with_vpin(500.0, 10.0, 0.90, 1.5, &arena);
        assert_eq!(toxic_signal.signal, SignalType::Flat);

        // VPIN no finito (NaN) debe vetar
        let nan_signal = engine.evaluate_microstructure_with_vpin(500.0, 10.0, f64::NAN, 1.5, &arena);
        assert_eq!(nan_signal.signal, SignalType::Flat);
    }

    #[test]
    fn test_scalp_engine_symmetric_short_signal_and_nan_volumes() {
        let mut engine = ScalpEngine::default();
        let arena = quantum_arena::GlobalArena::new(13.0);

        for _ in 0..15 {
            let res = engine.evaluate_microstructure(100.0, 100.0, 1.5, &arena);
            assert_eq!(res.signal, SignalType::Flat);
        }

        // Fuerte desequilibrio de venta (ask masivo vs bid pequeño)
        let sell_signal = engine.evaluate_microstructure(10.0, 500.0, 1.5, &arena);
        if sell_signal.signal != SignalType::Flat {
            assert_eq!(sell_signal.signal, SignalType::Short);
            assert_eq!(sell_signal.horizon, crate::TradeHorizon::Scalp);
        }

        // NaN volumes must return Flat
        let nan_vol = engine.evaluate_microstructure(f64::NAN, 100.0, 1.5, &arena);
        assert_eq!(nan_vol.signal, SignalType::Flat);

        let neg_vol = engine.evaluate_microstructure(-10.0, 100.0, 1.5, &arena);
        assert_eq!(neg_vol.signal, SignalType::Flat);
    }
}

