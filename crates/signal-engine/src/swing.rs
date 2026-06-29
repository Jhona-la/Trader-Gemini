use crate::{SignalIntent, SignalType};
use feature_engine::Ewma;

/// Motor de Swing (Axioma V: Aislamiento)
/// Operaciones macro basadas en tendencias y cruce de promedios exponenciales (O(1)).

pub struct SwingEngine {
    fast_ewma: Ewma,
    slow_ewma: Ewma,
}

impl SwingEngine {
    pub fn new(fast_period: f64, slow_period: f64) -> Self {
        Self {
            fast_ewma: Ewma::from_period(fast_period),
            slow_ewma: Ewma::from_period(slow_period),
        }
    }

    /// Evalúa la macroestructura (tendencia) y retorna una intención.
    /// Actualiza su estado interno matemático.
    #[inline(always)]
    pub fn evaluate_trend(&mut self, price: f64) -> SignalIntent {
        let fast_val = self.fast_ewma.update(price);
        let slow_val = self.slow_ewma.update(price);

        // Necesitamos esperar a que ambas estén inicializadas
        if !self.slow_ewma.is_initialized {
            return SignalIntent::flat();
        }

        // Lógica Swing Básica: Cruce de Medias. 
        // TODO: Evolver modificará los periodos atómicamente.
        let diff = (fast_val - slow_val) / slow_val;

        // Umbral de 0.1% de diferencia para considerar tendencia
        if diff > 0.001 {
            SignalIntent {
                signal: SignalType::Long,
                confidence: diff.abs() * 100.0,
            }
        } else if diff < -0.001 {
            SignalIntent {
                signal: SignalType::Short,
                confidence: diff.abs() * 100.0,
            }
        } else {
            SignalIntent::flat()
        }
    }
}

impl Default for SwingEngine {
    fn default() -> Self {
        // Valores default para swing: Fast 12, Slow 26 (Standard MACD settings)
        Self::new(12.0, 26.0)
    }
}
