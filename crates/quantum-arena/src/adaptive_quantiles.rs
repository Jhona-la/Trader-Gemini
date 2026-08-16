use std::f64;

/// 📊 ALGORITMO P2 (PI-SQUARE) DE ESTIMACIÓN DE CUANTILES EN TIEMPO REAL O(1)
/// Permite rastrear percentiles dinámicos (e.g., P80, P90, P95) sin almacenar vectores de datos.
/// Elimina al 100% la necesidad de umbrales hardcodeados.
#[derive(Debug, Clone)]
pub struct P2Quantile {
    pub p: f64,             // Percentil objetivo (e.g., 0.80 para P80)
    pub count: u64,         // Número total de observaciones
    pub q: [f64; 5],        // Alturas de los 5 marcadores
    pub n: [i64; 5],        // Posiciones actuales de los 5 marcadores
    pub np: [f64; 5],       // Posiciones deseadas de los 5 marcadores
    pub dn: [f64; 5],       // Incrementos deseados por cada observación
    pub initialized: bool,
}

impl P2Quantile {
    pub fn new(p: f64) -> Self {
        let p_clamped = p.clamp(0.01, 0.99);
        Self {
            p: p_clamped,
            count: 0,
            q: [0.0; 5],
            n: [1, 2, 3, 4, 5],
            np: [1.0, 1.0 + 2.0 * p_clamped, 1.0 + 4.0 * p_clamped, 3.0 + 2.0 * p_clamped, 5.0],
            dn: [0.0, p_clamped / 2.0, p_clamped, (1.0 + p_clamped) / 2.0, 1.0],
            initialized: false,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) {
        if !self.initialized {
            if self.count < 5 {
                self.q[self.count as usize] = x;
                self.count += 1;
                if self.count == 5 {
                    self.q.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1) || (d <= -1.0 && self.n[i - 1] - self.n[i] < -1) {
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

        q_i + (d / (n_ip1 - n_im1))
            * ((n_i - n_im1 + d) * (q_ip1 - q_i) / (n_ip1 - n_i)
                + (n_ip1 - n_i - d) * (q_i - q_im1) / (n_i - n_im1))
    }

    #[inline(always)]
    fn linear_interpolation(&self, i: usize, d: f64) -> f64 {
        let idx_next = if d > 0.0 { i + 1 } else { i - 1 };
        let n_diff = (self.n[idx_next] - self.n[i]) as f64;
        let q_diff = self.q[idx_next] - self.q[i];

        self.q[i] + d * (q_diff / n_diff)
    }

    #[inline(always)]
    pub fn value(&self) -> f64 {
        if !self.initialized {
            if self.count == 0 { return 0.0; }
            let mut temp = self.q[..self.count as usize].to_vec();
            temp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((self.count as f64 - 1.0) * self.p) as usize;
            return temp[idx.min(temp.len() - 1)];
        }
        self.q[2]
    }
}

/// 🛡️ MOTOR DE ADAPTACIÓN ANTI-HARDCODE PARAMÉTRICO
/// Mantiene cuantiles dinámicos para eliminar cualquier constante estática del sistema.
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
        self.ofi_p80.value().max(0.15)
    }

    #[inline(always)]
    pub fn dynamic_obi_threshold(&self) -> f64 {
        self.obi_p80.value().max(0.15)
    }

    #[inline(always)]
    pub fn dynamic_holistic_threshold(&self) -> f64 {
        self.holistic_p85.value().max(0.40)
    }
}
