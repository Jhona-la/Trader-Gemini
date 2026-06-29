/// Axioma II: Complejidad O(1) Estricta
/// Exponentially Weighted Moving Average (EWMA).
/// Mantiene un estado interno mínimo y se actualiza en O(1) sin arrays dinámicos.

#[derive(Debug, Clone, Copy)]
pub struct Ewma {
    pub value: f64,
    pub alpha: f64,
    pub is_initialized: bool,
}

impl Ewma {
    /// Crea un nuevo EWMA. 
    /// `alpha` determina la velocidad de decaimiento (0 < alpha <= 1).
    #[inline(always)]
    pub fn new(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha,
            is_initialized: false,
        }
    }

    /// Crea un EWMA basado en el periodo `N`.
    /// La fórmula estándar es: alpha = 2 / (N + 1).
    #[inline(always)]
    pub fn from_period(period: f64) -> Self {
        let alpha = 2.0 / (period + 1.0);
        Self::new(alpha)
    }

    /// Actualiza el valor con la nueva observación en O(1)
    #[inline(always)]
    pub fn update(&mut self, new_val: f64) -> f64 {
        if !self.is_initialized {
            self.value = new_val;
            self.is_initialized = true;
        } else {
            // S_t = (alpha * X_t) + ((1 - alpha) * S_{t-1})
            self.value = (self.alpha * new_val) + ((1.0 - self.alpha) * self.value);
        }
        self.value
    }

    #[inline(always)]
    pub fn get(&self) -> f64 {
        self.value
    }
}
