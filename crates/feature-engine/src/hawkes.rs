use std::f64;

/// ⚡ PROCESO ESTOCÁSTICO DE HAWKES (SELF-EXCITING ORDER FLOW INTENSITY)
/// Modelado probabilístico de ráfagas institucionales (Order Arrival Clusters).
/// Detecta aceleraciones auto-excitadas del mercado en nanosegundos.
#[derive(Debug, Clone)]
pub struct HawkesProcessEngine {
    pub mu: f64,             // Tasa base de fondo / mu_hat empírico adaptativo
    pub alpha: f64,          // Coeficiente de excitación por impulso
    pub beta: f64,           // Tasa de decaimiento temporal (decay rate)
    pub intensity_bull: f64, // Intensidad acumulada alcista λ_bull(t)
    pub intensity_bear: f64, // Intensidad acumulada bajista λ_bear(t)
    pub last_update_ms: u64, // Timestamp del último tick procesado
    pub n_seen: u64,         // Muestras observadas para calibración empírica de μ̂ (R7-R2-E-1)
}

impl HawkesProcessEngine {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Self {
        let mu_init = mu.max(0.01);
        Self {
            mu: mu_init,
            alpha: alpha.clamp(0.05, 0.95),
            beta: beta.max(0.1),
            intensity_bull: mu_init,
            intensity_bear: mu_init,
            last_update_ms: 0,
            n_seen: 0,
        }
    }

    /// Tasa base empírica estimada μ̂ del flujo exógeno (R7-R2-E-1)
    #[inline(always)]
    pub fn mu_hat(&self) -> f64 {
        self.mu
    }

    /// Ratio de ramificación estocástico n = α / β (estacionario si n < 1)
    #[inline(always)]
    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Ratio de intensidad en estado estacionario λ_ss / μ̂ = 1 + α / β
    #[inline(always)]
    pub fn steady_state_ratio(&self) -> f64 {
        1.0 + self.alpha / self.beta
    }

    /// Exceso de intensidad sobre la tasa base empírica (λ_bull + λ_bear) / μ̂
    #[inline(always)]
    pub fn intensity_ratio(&self) -> f64 {
        if self.mu > 0.0 {
            (self.intensity_bull + self.intensity_bear) / self.mu
        } else {
            1.0
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

        // #660 (F2-B6): evento RETRÓGRADO (ts < last) — no excita: antes
        // saltaba el decaimiento pero seguía excitando con historia
        // desalineada. Reloj monotónico.
        if self.last_update_ms > 0 && timestamp_ms < self.last_update_ms {
            let total = self.intensity_bull + self.intensity_bear;
            let ratio = if total > 0.0 {
                (self.intensity_bull - self.intensity_bear) / total
            } else {
                0.0
            };
            return (self.intensity_bull, self.intensity_bear, ratio);
        }
        let dt_sec = if self.last_update_ms > 0 {
            ((timestamp_ms - self.last_update_ms) as f64 / 1000.0).min(300.0)
        } else {
            0.0
        };
        if dt_sec > 0.0 {
            // R7-R2-E-1 (Ola Ω50): Estimación empírica adaptativa de μ̂ con constante de tiempo τ = 60s
            // y siembra en el segundo tick (idéntica formulación que signal_engine::hawkes_bessel #535).
            // Garantiza que la tasa de fondo converja al ritmo de llegada real del activo.
            let inst_rate = (1.0 / dt_sec).clamp(0.01, 200.0);
            if self.n_seen == 1 {
                self.mu = inst_rate.clamp(0.01, 50.0);
            } else if self.n_seen > 1 {
                let gain = 1.0 - (-dt_sec / 60.0).exp();
                self.mu = (self.mu + gain * (inst_rate - self.mu)).clamp(0.01, 200.0);
            }
            self.n_seen = self.n_seen.saturating_add(1);

            let decay = (-self.beta * dt_sec).exp();
            // Decaimiento exponencial del estado anterior acotado a tasa base empírica mu
            self.intensity_bull = (self.mu + (self.intensity_bull - self.mu) * decay).max(self.mu);
            self.intensity_bear = (self.mu + (self.intensity_bear - self.mu) * decay).max(self.mu);
        } else if self.last_update_ms == 0 {
            self.n_seen = 1;
        }
        self.last_update_ms = timestamp_ms;

        // #660 (F2-B6) — INVARIANZA TEMPORAL: el impulso del evento se
        // integra sobre el Δt que representa (kernel α·β·dt del proceso
        // Hawkes): con feeds densos cada evento aporta proporcionalmente
        // menos y λ deja de inflarse con la TASA de eventos del feed.
        // Piso de 1 ms (#904): los ticks intra-milisegundo conservan su
        // quantum de excitación.
        let dt_efectivo = dt_sec.max(0.001);
        let vol_ratio = (volume_usd.max(0.0) / volume_norm.max(1e-8)).min(100.0);
        let impulse = self.alpha * self.beta * dt_efectivo * (1.0 + vol_ratio.ln_1p().min(3.0));

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

    #[test]
    fn test_r7_r2_e1_hawkes_mu_hat_adaptacion_empirica() {
        let mut engine = HawkesProcessEngine::new(0.05, 0.35, 1.5);
        assert_eq!(engine.mu_hat(), 0.05);
        assert!((engine.branching_ratio() - 0.35 / 1.5).abs() < 1e-12);
        assert!((engine.steady_state_ratio() - (1.0 + 0.35 / 1.5)).abs() < 1e-12);

        // Primer tick a t=1000 ms
        engine.update(1000, 0.0, 1000.0, 1000.0);
        assert_eq!(engine.n_seen, 1);
        assert_eq!(engine.mu_hat(), 0.05);

        // Segundo tick a t=1050 ms (Δt = 50 ms = 0.05 s => tasa instantánea = 20.0 events/s)
        // La siembra rápida ajusta mu_hat instantáneamente a 20.0
        engine.update(1050, 0.0, 1000.0, 1000.0);
        assert_eq!(engine.n_seen, 2);
        assert!((engine.mu_hat() - 20.0).abs() < 1e-6, "Siembra rápida a 20.0 Hz");

        // Alimentar 10 ticks a cadencia rápida constante de 50 ms
        for i in 3..=12 {
            engine.update(1050 + (i - 2) * 50, 0.0, 1000.0, 1000.0);
        }
        // mu_hat debe permanecer en el entorno de 20.0 Hz
        assert!((engine.mu_hat() - 20.0).abs() < 0.5);

        // En ausencia de impulsos netos (delta_ofi = 0), intensity_ratio debe estar cercano a 2.0 (bull + bear = 2*mu)
        let ir = engine.intensity_ratio();
        assert!((ir - 2.0).abs() < 0.1, "Intensity ratio neutral debe ser ~2.0 (bull+bear)/mu");
    }
}
