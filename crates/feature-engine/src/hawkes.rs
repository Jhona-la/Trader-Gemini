use std::f64;

/// ⚡ PROCESO ESTOCÁSTICO DE HAWKES (SELF-EXCITING ORDER FLOW INTENSITY)
/// Modelado probabilístico de ráfagas institucionales (Order Arrival Clusters).
/// Detecta aceleraciones auto-excitadas del mercado en nanosegundos.
#[derive(Debug, Clone)]
pub struct HawkesProcessEngine {
    pub mu: f64,               // Tasa base de fondo (baseline arrival rate)
    pub alpha: f64,            // Coeficiente de excitación por impulso
    pub beta: f64,             // Tasa de decaimiento temporal (decay rate)
    pub intensity_bull: f64,   // Intensidad acumulada alcista λ_bull(t)
    pub intensity_bear: f64,   // Intensidad acumulada bajista λ_bear(t)
    pub last_update_ms: u64,   // Timestamp del último tick procesado
}

impl HawkesProcessEngine {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Self {
        Self {
            mu: mu.max(0.01),
            alpha: alpha.clamp(0.05, 0.95),
            beta: beta.max(0.1),
            intensity_bull: mu,
            intensity_bear: mu,
            last_update_ms: 0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, timestamp_ms: u64, delta_ofi: f64, volume_usd: f64, volume_norm: f64) -> (f64, f64, f64) {
        if self.last_update_ms > 0 && timestamp_ms > self.last_update_ms {
            let dt_sec = (timestamp_ms - self.last_update_ms) as f64 / 1000.0;
            let decay = (-self.beta * dt_sec).exp();
            
            // Decaimiento exponencial del estado anterior
            self.intensity_bull = self.mu + (self.intensity_bull - self.mu) * decay;
            self.intensity_bear = self.mu + (self.intensity_bear - self.mu) * decay;
        }
        self.last_update_ms = timestamp_ms;

        // Magnitud de excitación escalada por volumen
        let impulse = self.alpha * (1.0 + (volume_usd / volume_norm.max(1.0)).ln_1p().min(3.0));

        if delta_ofi > 0.0 {
            self.intensity_bull += impulse * delta_ofi.abs();
        } else if delta_ofi < 0.0 {
            self.intensity_bear += impulse * delta_ofi.abs();
        }

        let total_intensity = self.intensity_bull + self.intensity_bear;
        let hawkes_ratio = if total_intensity > 0.0 {
            (self.intensity_bull - self.intensity_bear) / total_intensity
        } else {
            0.0
        };

        (self.intensity_bull, self.intensity_bear, hawkes_ratio)
    }
}

impl Default for HawkesProcessEngine {
    fn default() -> Self {
        Self::new(0.05, 0.35, 1.5)
    }
}
