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
    pub fn evaluate_microstructure(&mut self, bid_vol: f64, ask_vol: f64, z_target: f64, arena: &quantum_arena::GlobalArena) -> SignalIntent {
        telemetry_server::profile_node!("ScalpEngine::evaluate_microstructure", {
            let current_obi = order_book_imbalance(bid_vol, ask_vol);
            let accel = obi_acceleration(current_obi, self.prev_obi);
            
            // Guardamos el estado O(1)
            self.prev_obi = current_obi;
            
            // Actualizamos estadísticas para umbral adaptativo
            self.accel_stats.update(accel);
            
            // Si no hay suficientes datos para desviación estándar, no operamos
            let min_samples = arena.config.scalp_accel_min_samples.load(std::sync::atomic::Ordering::Relaxed);
            if self.accel_stats.count < min_samples {
                return SignalIntent::flat();
            }
            
            let z_score = self.accel_stats.z_score(accel);
            
            // Eliminamos ml_model, dependemos puramente de Z-Score OBI por ahora
            let base_z = arena.config.turbo_z_score_stdev.load(std::sync::atomic::Ordering::Relaxed);
            let conf_clamp = arena.config.explosive_confidence_threshold.load(std::sync::atomic::Ordering::Relaxed); // Removed .min(0.9) clamp
            
            let final_confidence = (z_score.abs() / base_z).max(conf_clamp);
            
            let signal = if z_score > z_target {
                SignalType::Long
            } else if z_score < -z_target {
                SignalType::Short
            } else {
                SignalType::Flat
            };

            if signal != SignalType::Flat {
                let dynamic_duration = arena.config.base_duration_ms.load(std::sync::atomic::Ordering::Relaxed) as u64;
                SignalIntent {
                    signal,
                    confidence: final_confidence.tanh(),
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
}

impl Default for ScalpEngine {
    fn default() -> Self {
        Self::new()
    }
}
