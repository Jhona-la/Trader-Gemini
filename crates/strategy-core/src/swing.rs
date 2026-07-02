use crate::{SignalIntent, SignalType, SwingML};
use feature_engine::Ewma;

/// Motor de Swing (Axioma V: Aislamiento)
/// Operaciones macro basadas en tendencias y cruce de promedios exponenciales (O(1)).

pub struct SwingEngine {
    fast_ewma: Ewma,
    slow_ewma: Ewma,
    variance_ewma: Ewma, // Para calcular el StdDev (BBands)
    base_fast: f64,
    base_slow: f64,
    ml_model: SwingML,
}

impl SwingEngine {
    pub fn new(fast_period: f64, slow_period: f64) -> Self {
        Self {
            fast_ewma: Ewma::from_period(fast_period),
            slow_ewma: Ewma::from_period(slow_period),
            variance_ewma: Ewma::from_period(fast_period),
            base_fast: fast_period,
            base_slow: slow_period,
            ml_model: SwingML::new(),
        }
    }

    /// Evalúa la macroestructura (tendencia) y retorna una intención.
    /// Actualiza su estado interno matemático.
    #[inline(always)]
    pub fn evaluate_trend(&mut self, price: f64, hurst: f64, trend_threshold: f64) -> SignalIntent {
        // Adaptación de umbrales dinámicos (Cero Valores Fijos)
        // Hurst va de 0 a 1. 0.5 es ruido. > 0.6 es tendencia.
        // Si hay una tendencia extremadamente fuerte (Hurst > 0.8), reducimos el periodo para acelerar el cruce.
        
        let dynamic_multiplier = if hurst > 0.5 {
            1.0 - (hurst - 0.5).max(0.0)
        } else {
            1.0
        };

        // Mutar el alpha dinámicamente en O(1)
        self.fast_ewma.alpha = 2.0 / ((self.base_fast * dynamic_multiplier).max(2.0) + 1.0);
        self.slow_ewma.alpha = 2.0 / ((self.base_slow * dynamic_multiplier).max(5.0) + 1.0);
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
        
        // Simulación de Desviación Estándar Real
        let std_dev = variance.sqrt().max(fast_val * 0.0005); // Mínimo 0.05% de volatilidad
        let z_score = diff / std_dev;
        let macd_diff = (fast_val - slow_val) / slow_val;

        // Inferencia del modelo de Machine Learning nativo para Macro
        let ml_pred = self.ml_model.infer(macd_diff, z_score, hurst);

        // Si hurst < trend_threshold, estamos en rango (mean-reverting).
        
        if hurst < trend_threshold {
            if z_score > 2.0 || ml_pred < -0.6 {
                let conf = if ml_pred < -0.6 { ml_pred.abs() } else { 0.90 };
                return SignalIntent {
                    signal: SignalType::Short, // Sobrecomprado, vender
                    confidence: conf,
                };
            } else if z_score < -2.0 || ml_pred > 0.6 {
                let conf = if ml_pred > 0.6 { ml_pred } else { 0.90 };
                return SignalIntent {
                    signal: SignalType::Long, // Sobrevendido, comprar
                    confidence: conf,
                };
            }
        } else {
            // Si hay tendencia fuerte (hurst > threshold), seguimos la tendencia
            let threshold = 0.0005 * (1.0 / hurst.max(0.1));
            
            if macd_diff > threshold || ml_pred > 0.7 {
                return SignalIntent {
                    signal: SignalType::Long,
                    confidence: (macd_diff.abs() * hurst * 100.0).max(ml_pred).clamp(0.1, 1.0),
                };
            } else if macd_diff < -threshold || ml_pred < -0.7 {
                return SignalIntent {
                    signal: SignalType::Short,
                    confidence: (macd_diff.abs() * hurst * 100.0).max(ml_pred.abs()).clamp(0.1, 1.0),
                };
            }
        }

        SignalIntent::flat()
    }
}

impl Default for SwingEngine {
    fn default() -> Self {
        // Valores default para swing: Fast 12, Slow 26 (Standard MACD settings)
        Self::new(12.0, 26.0)
    }
}
