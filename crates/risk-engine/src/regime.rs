/// El régimen de mercado global, calculado basándose en la correlación de los N activos y la tendencia media.
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum MarketRegime {
    BullRun,
    Crash,
    #[default]
    Range,
    Chaotic,
}


pub struct RegimeDetector {
    correlation_threshold: f64,
    trend_threshold: f64,
    pub current_regime: MarketRegime,
}

impl RegimeDetector {
    pub fn new(correlation_threshold: f64, trend_threshold: f64) -> Self {
        Self {
            correlation_threshold,
            trend_threshold,
            current_regime: MarketRegime::Range,
        }
    }

    /// Actualiza el estado del régimen dado un valor de correlación media y el retorno medio (tendencia).
    #[inline(always)]
    pub fn update(&mut self, average_correlation: f64, average_trend: f64) -> MarketRegime {
        if average_correlation > self.correlation_threshold {
            if average_trend > self.trend_threshold {
                self.current_regime = MarketRegime::BullRun;
            } else if average_trend < -self.trend_threshold {
                self.current_regime = MarketRegime::Crash;
            } else {
                // Highly correlated but not moving much => Range tightening
                self.current_regime = MarketRegime::Range;
            }
        } else if average_correlation < 0.2 {
            self.current_regime = MarketRegime::Chaotic; // Baja correlación = Caos (Alts yendo por su lado)
        } else {
            self.current_regime = MarketRegime::Range;
        }

        self.current_regime
    }
}
