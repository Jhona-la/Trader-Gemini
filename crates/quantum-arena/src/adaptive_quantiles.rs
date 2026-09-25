use std::f64;

/// Estimador heurístico P² del cuantil de todas las observaciones aceptadas.
/// Mantiene cinco marcadores: memoria y trabajo O(1) por observación.
/// No incorpora olvido temporal, ventana móvil ni intervalos de confianza.
/// Actualizar un cuantil no calibra por sí solo un umbral predictivo.
#[derive(Debug, Clone)]
pub struct P2Quantile {
    pub p: f64,       // Percentil objetivo (e.g., 0.80 para P80)
    pub count: u64,   // Número total de observaciones
    pub q: [f64; 5],  // Alturas de los 5 marcadores
    pub n: [i64; 5],  // Posiciones actuales de los 5 marcadores
    pub np: [f64; 5], // Posiciones deseadas de los 5 marcadores
    pub dn: [f64; 5], // Incrementos deseados por cada observación
    pub initialized: bool,
}

impl P2Quantile {
    /// Adaptador histórico: recorta p a [0.01,0.99], incluidos infinitos.
    /// NaN se rechaza al construir; antes congelaba silenciosamente el ajuste.
    /// Para validar una probabilidad sin modificarla, usar `try_new`.
    ///
    /// # Panics
    /// Si p es NaN. Los consumidores configurables deben usar `try_new`.
    pub fn new(p: f64) -> Self {
        Self::try_new(p.clamp(0.01, 0.99)).expect("quantile probability must not be NaN")
    }

    /// Construye sin recortar p: el dominio de P² es 0 < p < 1.
    /// Los extremos requieren estimadores de mínimo/máximo, no este marcador.
    pub fn try_new(p: f64) -> Result<Self, &'static str> {
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err("quantile probability must be finite and strictly between 0 and 1");
        }
        Ok(Self {
            p,
            count: 0,
            q: [0.0; 5],
            n: [1, 2, 3, 4, 5],
            np: [
                1.0,
                1.0 + 2.0 * p,
                1.0 + 4.0 * p,
                3.0 + 2.0 * p,
                5.0,
            ],
            dn: [
                0.0,
                p / 2.0,
                p,
                (1.0 + p) / 2.0,
                1.0,
            ],
            initialized: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }

        if !self.initialized {
            if self.count < 5 {
                self.q[self.count as usize] = x;
                self.count += 1;
                if self.count == 5 {
                    self.q
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    self.initialized = true;
                }
                return;
            }
        }

        self.count += 1;

        // Encontrar k tal que q[k] <= x < q[k+1]
        let k = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x >= self.q[4] {
            self.q[4] = x;
            3
        } else {
            let mut found = 0;
            for i in 0..4 {
                if self.q[i] <= x && x < self.q[i + 1] {
                    found = i;
                    break;
                }
            }
            found
        };

        // Incrementar posiciones a la derecha de k
        for i in (k + 1)..5 {
            self.n[i] += 1;
        }

        // Actualizar posiciones deseadas
        for i in 0..5 {
            self.np[i] += self.dn[i];
        }

        // Ajustar alturas de marcadores 1, 2 y 3 si es necesario
        for i in 1..4 {
            let d = self.np[i] - self.n[i] as f64;

            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1)
                || (d <= -1.0 && self.n[i - 1] - self.n[i] < -1)
            {
                let d_sign = if d > 0.0 { 1 } else { -1 };
                let q_est = self.parabolic_interpolation(i, d_sign as f64);

                if self.q[i - 1] < q_est && q_est < self.q[i + 1] {
                    self.q[i] = q_est;
                } else {
                    self.q[i] = self.linear_interpolation(i, d_sign as f64);
                }

                self.n[i] += d_sign;
            }
        }
    }

    #[inline(always)]
    fn parabolic_interpolation(&self, i: usize, d: f64) -> f64 {
        let n_i = self.n[i] as f64;
        let n_ip1 = self.n[i + 1] as f64;
        let n_im1 = self.n[i - 1] as f64;

        let q_i = self.q[i];
        let q_ip1 = self.q[i + 1];
        let q_im1 = self.q[i - 1];

        let predict = |lower: f64, center: f64, upper: f64| {
            center + (d / (n_ip1 - n_im1))
                * ((n_i - n_im1 + d) * (upper - center) / (n_ip1 - n_i)
                    + (n_ip1 - n_i - d) * (center - lower) / (n_i - n_im1))
        };
        let estimate = predict(q_im1, q_i, q_ip1);
        if estimate.is_finite() {
            return estimate;
        }
        // P² is homogeneous in marker heights. Normalize only if intermediate
        // arithmetic overflowed, retaining ordinary-range numerical behavior.
        // A true overshoot is still rejected by the neighboring-marker check.
        let scale = q_im1.abs().max(q_i.abs()).max(q_ip1.abs());
        predict(q_im1 / scale, q_i / scale, q_ip1 / scale) * scale
    }

    #[inline(always)]
    fn linear_interpolation(&self, i: usize, d: f64) -> f64 {
        let idx_next = if d > 0.0 { i + 1 } else { i - 1 };
        let n_diff = (self.n[idx_next] - self.n[i]) as f64;
        let q_diff = self.q[idx_next] - self.q[i];

        if q_diff.is_finite() {
            self.q[i] + d * (q_diff / n_diff)
        } else {
            // Opposite-sign finite endpoints can have an infinite difference.
            // The move is a convex combination (0 < weight < 1), so the
            // result remains between those endpoints without that subtraction.
            let weight = d / n_diff;
            (1.0 - weight) * self.q[i] + weight * self.q[idx_next]
        }
    }

    #[inline(always)]
    pub fn value(&self) -> f64 {
        if !self.initialized {
            if self.count == 0 {
                return 0.0;
            }
            let mut temp = self.q;
            let temp = &mut temp[..self.count as usize];
            temp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((self.count as f64 - 1.0) * self.p) as usize;
            return temp[idx.min(temp.len() - 1)];
        }
        self.q[2]
    }
}

/// Cuantiles acumulativos con percentiles, pisos y fallbacks de política fijos.
/// `initialized` sólo indica cinco observaciones, no evidencia suficiente para
/// un percentil extremo, adaptación a un régimen nuevo o precisión garantizada.
#[derive(Debug, Clone)]
pub struct AdaptiveQuantileEngine {
    pub ofi_p80: P2Quantile,
    pub obi_p80: P2Quantile,
    pub holistic_p85: P2Quantile,
    pub atr_p50: P2Quantile,
}

impl AdaptiveQuantileEngine {
    pub fn new() -> Self {
        Self {
            ofi_p80: P2Quantile::new(0.80),
            obi_p80: P2Quantile::new(0.80),
            holistic_p85: P2Quantile::new(0.85),
            atr_p50: P2Quantile::new(0.50),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, ofi: f64, obi: f64, holistic: f64, atr_pct: f64) {
        self.ofi_p80.update(ofi.abs());
        self.obi_p80.update(obi.abs());
        self.holistic_p85.update(holistic.abs());
        self.atr_p50.update(atr_pct);
    }

    #[inline(always)]
    pub fn dynamic_ofi_threshold(&self) -> f64 {
        if self.ofi_p80.initialized {
            self.ofi_p80.value().max(0.02)
        } else {
            0.15
        }
    }

    #[inline(always)]
    pub fn dynamic_obi_threshold(&self) -> f64 {
        if self.obi_p80.initialized {
            self.obi_p80.value().max(0.02)
        } else {
            0.15
        }
    }

    #[inline(always)]
    pub fn dynamic_holistic_threshold(&self) -> f64 {
        if self.holistic_p85.initialized {
            self.holistic_p85.value().max(0.10)
        } else {
            0.40
        }
    }
}

impl Default for AdaptiveQuantileEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p2_quantile_convergence() {
        let mut p50 = P2Quantile::new(0.50);
        for i in 1..=100 {
            p50.update(i as f64);
        }
        assert!(p50.initialized);
        let val = p50.value();
        // El percentil 50 de 1..100 debe estar en el rango [40, 60]
        assert!(
            val >= 40.0 && val <= 60.0,
            "P50 converge cerca de 50: {}",
            val
        );
    }

    #[test]
    fn test_p2_quantile_nan_and_uninitialized_value() {
        let mut p80 = P2Quantile::new(0.80);
        assert_eq!(p80.value(), 0.0);
        p80.update(f64::NAN);
        p80.update(f64::INFINITY);
        assert!(!p80.initialized);
        assert_eq!(p80.count, 0);

        // 3 samples (still uninitialized)
        p80.update(10.0);
        p80.update(20.0);
        p80.update(30.0);
        assert!(!p80.initialized);
        assert!(p80.value() >= 10.0 && p80.value() <= 30.0);
    }

    #[test]
    fn test_adaptive_quantile_engine_initialization_and_update() {
        let mut engine = AdaptiveQuantileEngine::new();
        assert_eq!(engine.dynamic_ofi_threshold(), 0.15);
        assert_eq!(engine.dynamic_obi_threshold(), 0.15);
        assert_eq!(engine.dynamic_holistic_threshold(), 0.40);

        for i in 0..10 {
            engine.update(i as f64 * 0.1, 0.2, 0.5, 0.01);
        }

        assert!(engine.ofi_p80.initialized);
        assert!(engine.dynamic_ofi_threshold() >= 0.02);
        assert!(engine.dynamic_obi_threshold() >= 0.02);
        assert!(engine.dynamic_holistic_threshold() >= 0.10);
    }

    #[test]
    fn test_adaptive_quantile_engine_nan_immunity() {
        let mut engine = AdaptiveQuantileEngine::default();
        for _ in 0..20 {
            engine.update(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
        }
        assert_eq!(engine.dynamic_ofi_threshold(), 0.15);
        assert_eq!(engine.dynamic_obi_threshold(), 0.15);
        assert_eq!(engine.dynamic_holistic_threshold(), 0.40);
    }
}
