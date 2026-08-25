/// Fase A: Filtro de Kalman Cuántico
/// Un filtro de estado ultra-rápido para la predictibilidad de micro-tendencias (Scalping).
/// Modela la incertidumbre y suaviza señales con varianza adaptativa.

#[derive(Debug, Clone)]
pub struct KalmanFilter1D {
    // Estado estimado (ej. precio o velocidad)
    pub x: f64,
    // Covarianza del error de estimación
    pub p: f64,
    // Ruido del proceso (varianza del modelo, qué tan rápido cambia la realidad)
    pub q: f64,
    // Ruido de la medición (varianza del sensor, qué tan ruidoso es el tick)
    pub r: f64,
}

impl KalmanFilter1D {
    pub fn new(initial_x: f64, initial_p: f64, q: f64, r: f64) -> Self {
        Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
        }
    }

    /// Actualiza el filtro con una nueva medición O(1)
    #[inline(always)]
    pub fn update(&mut self, measurement: f64) -> f64 {
        if !measurement.is_finite() {
            return self.x;
        }

        // Predicción
        self.p = (self.p + self.q).max(1e-12);

        // Actualización
        let denom = self.p + self.r;
        let k = if denom.abs() > 1e-12 {
            self.p / denom
        } else {
            0.0
        };
        self.x += k * (measurement - self.x);
        self.p = ((1.0 - k) * self.p).max(1e-12);

        self.x
    }

    /// Actualiza dinámicamente el ruido de medición basado en la volatilidad reciente
    #[inline(always)]
    pub fn update_with_dynamic_r(&mut self, measurement: f64, dynamic_r: f64) -> f64 {
        if dynamic_r.is_finite() && dynamic_r > 0.0 {
            self.r = dynamic_r.max(1e-6);
        }
        self.update(measurement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter_noise_smoothing() {
        let mut kf = KalmanFilter1D::new(100.0, 1.0, 1e-4, 1.0);
        let mut estimated = 100.0;
        for i in 0..20 {
            let noise = if i % 2 == 0 { 2.0 } else { -2.0 };
            estimated = kf.update(100.0 + noise);
        }
        // El estimado debe estar muy cerca de 100.0, absorbiendo el ruido
        assert!((estimated - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_kalman_filter_nan_measurement_and_dynamic_r_immunity() {
        let mut kf = KalmanFilter1D::new(50.0, 1.0, 1e-4, 1.0);
        let est_nan = kf.update(f64::NAN);
        assert_eq!(est_nan, 50.0);

        let est_dyn_nan = kf.update_with_dynamic_r(52.0, f64::NAN);
        assert!(est_dyn_nan.is_finite());
        assert!((est_dyn_nan - 50.0).abs() < 2.0);
    }
}
