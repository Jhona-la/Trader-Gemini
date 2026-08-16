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
        let clamped = val.clamp(-1.0, 1.0);
        let bin_idx = (((clamped + 1.0) / 2.0) * (self.num_bins as f64 - 1e-5)) as usize;

        self.counts[bin_idx.min(self.num_bins - 1)] += 1;
        self.total_samples += 1;

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
