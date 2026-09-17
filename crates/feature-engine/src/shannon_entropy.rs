use std::f64;

/// 🔬 MOTOR DE ENTROPÍA DE INFORMACIÓN DE SHANNON (SHANNON ENTROPY MICROSTRUCTURE ENGINE)
/// Mide el nivel de desorden probabilístico del flujo de órdenes en nanosegundos.
/// Bloquea trades en fases de alta entropía (ruido/caos) y amplifica la convicción en baja entropía.
#[derive(Debug, Clone)]
pub struct ShannonEntropyEngine {
    pub num_bins: usize,
    pub counts: [u64; 32],
    pub total_samples: u64,
}

impl ShannonEntropyEngine {
    pub fn new(num_bins: usize) -> Self {
        let bins = num_bins.clamp(4, 32);
        Self {
            num_bins: bins,
            counts: [0; 32],
            total_samples: 0,
        }
    }

    /// Registra una nueva observación de flujo (-1.0 a +1.0) y calcula la entropía de Shannon normalizada (0.0 a 1.0)
    #[inline(always)]
    pub fn update(&mut self, val: f64) -> Option<f64> {
        let safe_val = if val.is_finite() { val } else { 0.0 };
        let clamped = safe_val.clamp(-1.0, 1.0);
        let bin_idx = (((clamped + 1.0) / 2.0) * (self.num_bins as f64 - 1e-5)) as usize;

        self.counts[bin_idx.min(self.num_bins - 1)] += 1;
        self.total_samples += 1;

        // Decaimiento periódico para evitar petrificación de entropía tras horas de trading
        if self.total_samples > 1000 {
            let mut new_total = 0;
            for i in 0..self.num_bins {
                self.counts[i] = self.counts[i].div_ceil(2);
                new_total += self.counts[i];
            }
            self.total_samples = new_total;
        }

        if self.total_samples < 20 {
            return None; // Fallo explícito por calentamiento incompleto (previene sesgos)
        }

        let mut entropy = 0.0;
        let total = self.total_samples as f64;

        for i in 0..self.num_bins {
            let c = self.counts[i];
            if c > 0 {
                let p = c as f64 / total;
                entropy -= p * p.log2();
            }
        }

        let max_entropy = (self.num_bins as f64).log2();
        let normalized_entropy = (entropy / max_entropy).clamp(0.0, 1.0);

        Some(normalized_entropy)
    }
}

impl Default for ShannonEntropyEngine {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_decay_and_bounds() {
        let mut engine = ShannonEntropyEngine::new(10);
        for _ in 0..19 {
            assert!(engine.update(0.0).is_none());
        }

        // Flujo idéntico -> baja entropía (orden perfecto)
        for _ in 0..50 {
            let e = engine.update(0.0);
            assert!(e.is_some());
        }
        let low_ent = engine.update(0.0).unwrap();
        assert!(low_ent < 0.30);

        // Dispersión uniforme -> alta entropía
        let mut uniform_engine = ShannonEntropyEngine::new(10);
        for i in 0..500 {
            let val = ((i % 10) as f64 / 4.5) - 1.0;
            uniform_engine.update(val);
        }
        let high_ent = uniform_engine.update(0.5).unwrap();
        assert!(high_ent > 0.80);
    }

    #[test]
    fn test_shannon_entropy_nan_input_immunity() {
        let mut engine = ShannonEntropyEngine::new(8);
        for _ in 0..25 {
            let _ = engine.update(f64::NAN);
        }
        let e = engine.update(f64::INFINITY);
        assert!(e.is_some());
        let val = e.unwrap();
        assert!(val.is_finite() && val >= 0.0 && val <= 1.0);
    }
}
