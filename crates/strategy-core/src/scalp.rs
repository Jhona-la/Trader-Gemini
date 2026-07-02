use crate::{SignalIntent, SignalType, ScalpML};
use feature_engine::{obi_acceleration, order_book_imbalance, WelfordOnline};

/// Motor de Scalping (Axioma V: Aislamiento)
/// Operaciones ultrarrápidas basadas en desequilibrio del Order Book y Z-Scores dinámicos.

pub struct ScalpEngine {
    // Almacenamos estado previo para derivadas
    prev_obi: f64,
    // Estadísticas dinámicas de aceleración
    accel_stats: WelfordOnline,
    // Integración de Inteligencia Artificial Nativa
    ml_model: ScalpML,
}

impl ScalpEngine {
    pub fn new() -> Self {
        Self { 
            prev_obi: 0.0,
            accel_stats: WelfordOnline::new(),
            ml_model: ScalpML::new(),
        }
    }

    /// Evalúa la microestructura y retorna una intención.
    /// `z_target` se calibra según el régimen de volatilidad (ej. 2.0 o 3.0 para alta confianza).
    #[inline(always)]
    pub fn evaluate_microstructure(&mut self, bid_vol: f64, ask_vol: f64, z_target: f64) -> SignalIntent {
        let current_obi = order_book_imbalance(bid_vol, ask_vol);
        let accel = obi_acceleration(current_obi, self.prev_obi);
        
        // Guardamos el estado O(1)
        self.prev_obi = current_obi;
        
        // Actualizamos estadísticas para umbral adaptativo
        self.accel_stats.update(accel);
        
        // Si no hay suficientes datos para desviación estándar, no operamos
        if self.accel_stats.count < 30.0 {
            return SignalIntent::flat();
        }
        
        let z_score = self.accel_stats.z_score(accel);
        
        // Inferencia del modelo de Machine Learning (si está activo/entrenado)
        // Usamos un spread proxy fijo de 1.0 por ahora si no tenemos data L2
        let ml_pred = self.ml_model.infer(current_obi, accel, 1.0);
        
        let mut final_confidence = (z_score.abs() / 3.0).clamp(0.5, 1.0);
        
        // Fusión Cuántica: Z-Score + Random Forest
        let signal = if z_score > z_target {
            if ml_pred > 0.0 { final_confidence += 0.2; }
            SignalType::Long
        } else if z_score < -z_target {
            if ml_pred < 0.0 { final_confidence += 0.2; }
            SignalType::Short
        } else {
            // Si el modelo predictivo tiene mucha fuerza pero el Z-Score no llegó al target
            if ml_pred > 0.8 {
                final_confidence = ml_pred;
                SignalType::Long
            } else if ml_pred < -0.8 {
                final_confidence = ml_pred.abs();
                SignalType::Short
            } else {
                SignalType::Flat
            }
        };

        if signal != SignalType::Flat {
            SignalIntent {
                signal,
                confidence: final_confidence.clamp(0.0, 1.0),
            }
        } else {
            SignalIntent::flat()
        }
    }
}

impl Default for ScalpEngine {
    fn default() -> Self {
        Self::new()
    }
}
