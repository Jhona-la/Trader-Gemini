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
        // FIX #1455: Sanitización de alpha
        let safe_alpha = if alpha.is_finite() && alpha > 0.0 && alpha <= 1.0 { alpha } else { 0.1 };
        Self {
            value: 0.0,
            alpha: safe_alpha,
            is_initialized: false,
        }
    }

    /// Crea un EWMA basado en el periodo `N`.
    /// La fórmula estándar es: alpha = 2 / (N + 1).
    #[inline(always)]
    pub fn from_period(period: f64) -> Self {
        // FIX #1455: Sanitización de periodo
        let safe_period = if period.is_finite() && period >= 1.0 { period } else { 14.0 };
        let alpha = 2.0 / (safe_period + 1.0);
        Self::new(alpha)
    }

    /// Actualiza el valor con la nueva observación en O(1)
    #[inline(always)]
    pub fn update(&mut self, new_val: f64) -> f64 {
        // FIX #1455: Inmunidad ante NaNs
        if !new_val.is_finite() {
            return self.value;
        }
        if !self.is_initialized {
            self.value = new_val;
            self.is_initialized = true;
        } else {
            // S_t = (alpha * X_t) + ((1 - alpha) * S_{t-1})
            let updated = (self.alpha * new_val) + ((1.0 - self.alpha) * self.value);
            self.value = if updated.is_finite() { updated } else { self.value };
        }
        self.value
    }

    #[inline(always)]
    pub fn get(&self) -> f64 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_initialization_and_update() {
        let mut ewma = Ewma::from_period(9.0);
        assert!((ewma.alpha - 0.2).abs() < 1e-10);
        assert!(!ewma.is_initialized);

        assert_eq!(ewma.update(100.0), 100.0);
        assert!(ewma.is_initialized);

        let next = ewma.update(110.0);
        // 0.2 * 110 + 0.8 * 100 = 22 + 80 = 102
        assert!((next - 102.0).abs() < 1e-10);
    }

    #[test]
    fn test_ewma_nan_immunity() {
        let mut ewma = Ewma::new(0.5);
        ewma.update(50.0);
        assert_eq!(ewma.update(f64::NAN), 50.0);
        assert_eq!(ewma.update(f64::INFINITY), 50.0);
    }
}

