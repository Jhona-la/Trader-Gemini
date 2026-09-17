use std::f64;

/// ⚡ PROCESO ESTOCÁSTICO DE HAWKES (SELF-EXCITING ORDER FLOW INTENSITY)
/// Modelado probabilístico de ráfagas institucionales (Order Arrival Clusters).
/// Detecta aceleraciones auto-excitadas del mercado en nanosegundos.
#[derive(Debug, Clone)]
pub struct HawkesProcessEngine {
    pub mu: f64,             // Tasa base de fondo (baseline arrival rate)
    pub alpha: f64,          // Coeficiente de excitación por impulso
    pub beta: f64,           // Tasa de decaimiento temporal (decay rate)
    pub intensity_bull: f64, // Intensidad acumulada alcista λ_bull(t)
    pub intensity_bear: f64, // Intensidad acumulada bajista λ_bear(t)
    pub last_update_ms: u64, // Timestamp del último tick procesado
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
    pub fn update(
        &mut self,
        timestamp_ms: u64,
        delta_ofi: f64,
        volume_usd: f64,
        volume_norm: f64,
    ) -> (f64, f64, f64) {
        if !delta_ofi.is_finite() || !volume_usd.is_finite() || !volume_norm.is_finite() {
            let total = self.intensity_bull + self.intensity_bear;
            let ratio = if total > 0.0 {
                (self.intensity_bull - self.intensity_bear) / total
            } else {
                0.0
            };
            return (self.intensity_bull, self.intensity_bear, ratio);
        }

        if self.last_update_ms > 0 && timestamp_ms > self.last_update_ms {
            let dt_sec = ((timestamp_ms - self.last_update_ms) as f64 / 1000.0).min(300.0);
            let decay = (-self.beta * dt_sec).exp();

            // Decaimiento exponencial del estado anterior acotado a tasa base mu
            self.intensity_bull = (self.mu + (self.intensity_bull - self.mu) * decay).max(self.mu);
            self.intensity_bear = (self.mu + (self.intensity_bear - self.mu) * decay).max(self.mu);
            self.last_update_ms = timestamp_ms;
        } else if self.last_update_ms == 0 {
            self.last_update_ms = timestamp_ms;
        }
        // FIX #904: Ticks intra-milisegundo (timestamp_ms == last_update_ms) comparten el mismo instante (dt=0),
        // por lo que no se aplica decaimiento temporal y solo se acumulan los impulsos de auto-excitación.

        // Magnitud de excitación escalada por volumen con protecciones numéricas
        let vol_ratio = (volume_usd.max(0.0) / volume_norm.max(1e-8)).min(100.0);
        let impulse = self.alpha * (1.0 + vol_ratio.ln_1p().min(3.0));

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

    /// Procesa una ráfaga por lotes de ticks para acelerar la ingestión L2 en nanosegundos (Punto #031)
    #[inline(always)]
    pub fn update_batch(&mut self, events: &[(u64, f64, f64, f64)]) -> (f64, f64, f64) {
        let mut last_res = (self.intensity_bull, self.intensity_bear, 0.0);
        for &(ts, delta_ofi, vol_usd, vol_norm) in events {
            last_res = self.update(ts, delta_ofi, vol_usd, vol_norm);
        }
        last_res
    }
}

impl Default for HawkesProcessEngine {
    fn default() -> Self {
        Self::new(0.05, 0.35, 1.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hawkes_process_self_excitation_and_decay() {
        let mut engine = HawkesProcessEngine::new(0.05, 0.35, 1.5);
        let (bull_init, bear_init, ratio_init) = engine.update(1000, 0.0, 1000.0, 1000.0);
        assert_eq!(bull_init, 0.05);
        assert_eq!(bear_init, 0.05);
        assert_eq!(ratio_init, 0.0);

        // Impulso alcista
        let (bull_impulse, _, ratio_impulse) = engine.update(1100, 1.5, 5000.0, 1000.0);
        assert!(bull_impulse > 0.05);
        assert!(ratio_impulse > 0.0);

        // Decaimiento temporal
        let (bull_decay, _, _) = engine.update(5000, 0.0, 1000.0, 1000.0);
        assert!(bull_decay < bull_impulse);
    }

    #[test]
    fn test_hawkes_process_batch_and_nan_immunity() {
        let mut engine = HawkesProcessEngine::new(0.05, 0.35, 1.5);
        let events = vec![
            (1000, 1.0, 2000.0, 1000.0),
            (1100, f64::NAN, 1000.0, 1000.0),
            (1200, -1.5, 3000.0, 1000.0),
        ];
        let (bull, bear, ratio) = engine.update_batch(&events);
        assert!(bull.is_finite());
        assert!(bear.is_finite());
        assert!(ratio.is_finite());
        assert!(ratio >= -1.0 && ratio <= 1.0);
    }
}
