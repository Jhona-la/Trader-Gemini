/// El régimen de mercado global, calculado basándose en la correlación de los N activos y la tendencia media.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarketRegime {
    BullRun,
    Crash,
    #[default]
    Range,
    Chaotic,
}

impl From<u8> for MarketRegime {
    fn from(val: u8) -> Self {
        match val {
            1 => MarketRegime::BullRun,
            2 => MarketRegime::Crash,
            3 => MarketRegime::Chaotic,
            _ => MarketRegime::Range,
        }
    }
}

impl Into<u8> for MarketRegime {
    fn into(self) -> u8 {
        match self {
            MarketRegime::Range => 0,
            MarketRegime::BullRun => 1,
            MarketRegime::Crash => 2,
            MarketRegime::Chaotic => 3,
        }
    }
}

/// Símplex espectral continuo de régimen de mercado: cuatro componentes continuas en el símplex Δ³:
/// `p_range + p_bull + p_crash + p_chaos = 1.0`, con `p_i >= 0.0`.
///
/// Erradica las fronteras discretas rígidas y cuantifica la incertidumbre
/// del régimen como un continuo de probabilidad de Markov/Gibbs y mecánica estadística.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralMarketRegime {
    pub p_range: f64,
    pub p_bull: f64,
    pub p_crash: f64,
    pub p_chaos: f64,
}

impl Default for SpectralMarketRegime {
    #[inline(always)]
    fn default() -> Self {
        Self {
            p_range: 1.0,
            p_bull: 0.0,
            p_crash: 0.0,
            p_chaos: 0.0,
        }
    }
}

impl SpectralMarketRegime {
    /// Construye y normaliza un símplex continuo a partir de 4 probabilidades o pesos brutos.
    /// Sanitiza rigurosamente valores NaN, infinitos y negativos.
    #[inline]
    pub fn new(p_range: f64, p_bull: f64, p_crash: f64, p_chaos: f64) -> Self {
        let r = if p_range.is_finite() && p_range >= 0.0 { p_range } else { 0.0 };
        let b = if p_bull.is_finite() && p_bull >= 0.0 { p_bull } else { 0.0 };
        let c = if p_crash.is_finite() && p_crash >= 0.0 { p_crash } else { 0.0 };
        let ch = if p_chaos.is_finite() && p_chaos >= 0.0 { p_chaos } else { 0.0 };

        let sum = r + b + c + ch;
        if sum > 1e-12 {
            Self {
                p_range: r / sum,
                p_bull: b / sum,
                p_crash: c / sum,
                p_chaos: ch / sum,
            }
        } else {
            Self::default()
        }
    }

    /// Entropía espectral de Shannon continua: H = - sum_i p_i ln(p_i + eps).
    /// Mide la incertidumbre intrínseca del régimen de mercado (0 = estado puro determinado, ln(4) = máxima incertidumbre).
    #[inline]
    pub fn shannon_entropy(&self) -> f64 {
        let eps = 1e-15;
        let mut h = 0.0;
        for &p in &[self.p_range, self.p_bull, self.p_crash, self.p_chaos] {
            if p > eps {
                h -= p * (p + eps).ln();
            }
        }
        h.max(0.0)
    }

    /// Entropía cuántica/generalizada de Rényi de orden alpha:
    /// H_alpha = 1/(1 - alpha) * ln(sum_i p_i^alpha).
    /// Generaliza Shannon (alpha -> 1), colisión (alpha = 2) y min-entropía (alpha -> infty).
    #[inline]
    pub fn renyi_entropy(&self, alpha: f64) -> f64 {
        if !alpha.is_finite() || alpha <= 0.0 || (alpha - 1.0).abs() < 1e-4 {
            return self.shannon_entropy();
        }
        let mut sum_alpha = 0.0;
        for &p in &[self.p_range, self.p_bull, self.p_crash, self.p_chaos] {
            if p > 0.0 {
                sum_alpha += p.powf(alpha);
            }
        }
        if sum_alpha <= 0.0 || !sum_alpha.is_finite() {
            return 0.0;
        }
        (1.0 / (1.0 - alpha) * sum_alpha.ln()).max(0.0)
    }

    /// Polarización direccional continua en [-1.0, 1.0]:
    /// Pi_dir = p_bull - p_crash.
    /// +1.0 = BullRun pleno, -1.0 = Crash sistémico pleno, 0.0 = simétrico/neutral.
    #[inline(always)]
    pub fn directional_bias(&self) -> f64 {
        (self.p_bull - self.p_crash).clamp(-1.0, 1.0)
    }

    /// Índice de turbulencia/inestabilidad continua en [0.0, 1.0]:
    /// tau_turb = p_crash + p_chaos.
    #[inline(always)]
    pub fn turbulence_index(&self) -> f64 {
        (self.p_crash + self.p_chaos).clamp(0.0, 1.0)
    }

    /// Estimador MAP (Maximum A Posteriori) discreto para compatibilidad con código legado.
    #[inline]
    pub fn map_discrete(&self) -> MarketRegime {
        let max_p = self.p_range.max(self.p_bull).max(self.p_crash).max(self.p_chaos);
        if (self.p_bull - max_p).abs() < 1e-9 && self.p_bull > self.p_range {
            MarketRegime::BullRun
        } else if (self.p_crash - max_p).abs() < 1e-9 && self.p_crash > self.p_range {
            MarketRegime::Crash
        } else if (self.p_chaos - max_p).abs() < 1e-9 && self.p_chaos > self.p_range {
            MarketRegime::Chaotic
        } else {
            MarketRegime::Range
        }
    }

    /// Cómputo continuo suave C^inf sobre el símplex a partir de correlación media y tendencia.
    #[inline]
    pub fn compute_continuous_simplex(
        average_correlation: f64,
        average_trend: f64,
        correlation_threshold: f64,
        trend_threshold: f64,
    ) -> Self {
        if !average_correlation.is_finite() || !average_trend.is_finite() {
            return Self::default();
        }
        let safe_corr = if correlation_threshold.is_finite() && correlation_threshold > 0.0 {
            correlation_threshold
        } else {
            0.6
        };
        let safe_trend = if trend_threshold.is_finite() && trend_threshold > 0.0 {
            trend_threshold
        } else {
            0.02
        };

        // Activaciones continuas C^inf suaves sin escalones rígidos
        let s_corr = 1.0 / (1.0 + (-(average_correlation - safe_corr) / 0.08).clamp(-50.0, 50.0).exp());
        let s_chaos = 1.0 / (1.0 + (-(0.25 - average_correlation) / 0.05).clamp(-50.0, 50.0).exp());
        let s_trend_pos = 1.0 / (1.0 + (-(average_trend - safe_trend) / 0.005).clamp(-50.0, 50.0).exp());
        let s_trend_neg = 1.0 / (1.0 + (-(-average_trend - safe_trend) / 0.005).clamp(-50.0, 50.0).exp());

        let w_bull = s_corr * s_trend_pos;
        let w_crash = s_corr * s_trend_neg;
        let w_chaos = (1.0 - s_corr) * s_chaos;
        let w_range = (1.0 - w_bull - w_crash - w_chaos).max(0.05);

        Self::new(w_range, w_bull, w_crash, w_chaos)
    }
}

pub struct RegimeDetector {
    correlation_threshold: f64,
    trend_threshold: f64,
    pub current_regime: MarketRegime,
    pub current_spectral: SpectralMarketRegime,
}

impl RegimeDetector {
    pub fn new(correlation_threshold: f64, trend_threshold: f64) -> Self {
        let safe_corr = if correlation_threshold.is_finite() && correlation_threshold > 0.0 {
            correlation_threshold
        } else {
            0.6
        };
        let safe_trend = if trend_threshold.is_finite() && trend_threshold > 0.0 {
            trend_threshold
        } else {
            0.02
        };
        Self {
            correlation_threshold: safe_corr,
            trend_threshold: safe_trend,
            current_regime: MarketRegime::Range,
            current_spectral: SpectralMarketRegime::default(),
        }
    }

    /// Actualiza el estado del régimen dado un valor de correlación media y el retorno medio (tendencia).
    #[inline(always)]
    pub fn update(&mut self, average_correlation: f64, average_trend: f64) -> MarketRegime {
        if !average_correlation.is_finite() || !average_trend.is_finite() {
            return self.current_regime;
        }

        self.current_spectral = SpectralMarketRegime::compute_continuous_simplex(
            average_correlation,
            average_trend,
            self.correlation_threshold,
            self.trend_threshold,
        );

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

    /// Actualiza y retorna el símplex espectral continuo.
    #[inline(always)]
    pub fn update_spectral(&mut self, average_correlation: f64, average_trend: f64) -> SpectralMarketRegime {
        let _ = self.update(average_correlation, average_trend);
        self.current_spectral
    }
}
