/// Algoritmo de Welford Online para Varianza
/// Permite calcular la media, la varianza y la desviación estándar en una sola pasada (O(1) por update)
/// sin sufrir cancelación catastrófica (IEEE-754 precision issues).

#[derive(Debug, Clone, Copy)]
pub struct WelfordOnline {
    pub count: f64,
    pub mean: f64,
    pub m2: f64,
    pub is_decay: bool,
}

impl WelfordOnline {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
            is_decay: false,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, val: f64) {
        // FIX #1429: Inmunidad ante valores no finitos o corrompidos
        if !val.is_finite() {
            return;
        }

        if self.count >= 2000.0 {
            // FIX #389: Transición suave a EWMA sin discontinuidad de varianza
            if !self.is_decay {
                self.m2 = (self.m2 / (self.count - 1.0)).max(0.0);
                self.is_decay = true;
            }
            let alpha = 2.0 / (self.count.min(2000.0) + 1.0);
            let delta = val - self.mean;
            self.mean += alpha * delta;
            let delta2 = val - self.mean;
            self.m2 = (1.0 - alpha) * self.m2 + alpha * delta * delta2;
        } else {
            self.count += 1.0;
            let delta = val - self.mean;
            self.mean += delta / self.count;
            let delta2 = val - self.mean;
            self.m2 += delta * delta2;
        }
    }

    /// Actualización con factor de decaimiento exponencial explícito (EW-Welford)
    #[inline(always)]
    pub fn update_decay(&mut self, val: f64, alpha: f64) {
        // FIX #1429: Inmunidad ante valores no finitos
        if !val.is_finite() || !alpha.is_finite() || alpha <= 0.0 {
            return;
        }

        if !self.is_decay {
            if self.count >= 2.0 {
                self.m2 = (self.m2 / (self.count - 1.0)).max(0.0);
            }
            self.is_decay = true;
        }
        let delta = val - self.mean;
        self.mean += alpha * delta;
        let delta2 = val - self.mean;
        self.m2 = (1.0 - alpha) * self.m2 + alpha * delta * delta2;
        self.count = (self.count * (1.0 - alpha) + 1.0).min(1.0 / alpha);
    }

    #[inline(always)]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    #[inline(always)]
    pub fn variance(&self) -> f64 {
        if self.is_decay {
            self.m2.max(0.0)
        } else if self.count < 2.0 {
            0.0
        } else {
            (self.m2 / (self.count - 1.0)).max(0.0)
        }
    }

    #[inline(always)]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Calcula el Z-Score actual basado en la estadística acumulada.
    #[inline(always)]
    pub fn z_score(&self, val: f64) -> f64 {
        let std = self.std_dev();
        if std <= 1e-9 {
            0.0
        } else {
            (val - self.mean) / std
        }
    }
}

impl Default for WelfordOnline {
    fn default() -> Self {
        Self::new()
    }
}
