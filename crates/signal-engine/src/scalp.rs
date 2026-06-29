use crate::{SignalIntent, SignalType};
use feature_engine::{obi_acceleration, order_book_imbalance};

/// Motor de Scalping (Axioma V: Aislamiento)
/// Operaciones ultrarrápidas basadas en desequilibrio del Order Book.

pub struct ScalpEngine {
    // Almacenamos estado previo para derivadas
    prev_obi: f64,
}

impl ScalpEngine {
    pub fn new() -> Self {
        Self { prev_obi: 0.0 }
    }

    /// Evalúa la microestructura y retorna una intención.
    /// `obi_threshold` se leerá del QuantumArena.
    #[inline(always)]
    pub fn evaluate_microstructure(&mut self, bid_vol: f64, ask_vol: f64, obi_threshold: f64) -> SignalIntent {
        let current_obi = order_book_imbalance(bid_vol, ask_vol);
        let accel = obi_acceleration(current_obi, self.prev_obi);
        
        // Guardamos el estado O(1)
        self.prev_obi = current_obi;

        // Lógica de desbalance extremo: Si el OBI supera el umbral y está acelerando
        if current_obi > obi_threshold && accel > 0.01 {
            return SignalIntent {
                signal: SignalType::Long,
                confidence: current_obi.abs(),
            };
        } else if current_obi < -obi_threshold && accel < -0.01 {
            return SignalIntent {
                signal: SignalType::Short,
                confidence: current_obi.abs(),
            };
        }

        SignalIntent::flat()
    }
}

impl Default for ScalpEngine {
    fn default() -> Self {
        Self::new()
    }
}
