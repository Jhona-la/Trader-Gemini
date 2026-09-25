/// Estadísticas por evento: Welford acumulativo, seguido de varianza exponencial.
/// Reduce cancelación frente a E[x²]-E[x]²; no evita todo overflow de f64.
/// El umbral histórico de 2000 observaciones es política de memoria, no tiempo físico.

#[derive(Debug, Clone, Copy)]
pub struct WelfordOnline {
    /// Conteo acumulativo antes del cambio de modo. En update_decay conserva
    /// el indicador de masa histórico; NO es tamaño muestral efectivo ni reloj.
    pub count: f64,
    pub mean: f64,
    /// Suma de cuadrados centrados en modo acumulativo; varianza en modo decay.
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

        if self.is_decay || self.count >= 2000.0 {
            // FIX #389: Transición suave a EWMA sin discontinuidad de varianza
            if !self.is_decay {
                self.m2 = (self.m2 / (self.count - 1.0)).max(0.0);
                self.is_decay = true;
            }
            // Una vez convertido, m2 nunca vuelve a ser suma de cuadrados.
            // update usa siempre la política histórica; para otro alpha se
            // debe llamar explícitamente a update_decay en cada observación.
            let alpha = 2.0 / 2001.0;
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

    /// Actualización exponencial con 0 < alpha <= 1; entradas inválidas no mutan.
    /// La primera muestra inicializa la media sin inventar un prior en cero.
    /// Al convertir un estado acumulativo se conserva su varianza muestral
    /// como estado inicial del filtro, no como momento empírico ponderado exacto.
    #[inline(always)]
    pub fn update_decay(&mut self, val: f64, alpha: f64) {
        // FIX #1429: Inmunidad ante valores no finitos
        if !val.is_finite() || !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
            return;
        }

        if self.count == 0.0 {
            self.count = 1.0;
            self.mean = val;
            self.m2 = 0.0;
            self.is_decay = true;
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
