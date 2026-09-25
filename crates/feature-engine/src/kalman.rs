/// Filtro escalar clásico de paseo aleatorio y observación directa (H = 1).
/// x tiene unidades de la observación; p, q y r tienen sus unidades al cuadrado.
/// q es varianza POR ACTUALIZACIÓN: esta API no conoce timestamps ni estima q/r.
/// No representa un cálculo cuántico ni una taxonomía de horizontes de trading.

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
    /// # Panics
    /// Ante estado/covarianzas inválidos; usar try_new para datos externos.
    pub fn new(initial_x: f64, initial_p: f64, q: f64, r: f64) -> Self {
        Self::try_new(initial_x, initial_p, q, r)
            .expect("Kalman requires finite state and nonnegative finite variances")
    }

    /// Construcción fallible para configuración externa; no inventa covarianzas.
    pub fn try_new(initial_x: f64, initial_p: f64, q: f64, r: f64) -> Result<Self, &'static str> {
        if !initial_x.is_finite()
            || !Self::valid_variance(initial_p)
            || !Self::valid_variance(q)
            || !Self::valid_variance(r)
        {
            return Err("Kalman requires finite state and nonnegative finite variances");
        }
        Ok(Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
        })
    }

    fn valid_variance(value: f64) -> bool {
        value.is_finite() && value >= 0.0
    }

    /// Actualiza el filtro con una nueva medición O(1)
    #[inline(always)]
    pub fn update(&mut self, measurement: f64) -> f64 {
        self.try_update(measurement).unwrap_or(self.x)
    }

    /// Rechaza una entrada/estado inválido o una predicción no representable sin
    /// cambiar x/p/r. Los campos públicos obligan a validar también cada update.
    /// Un resultado Err no significa incertidumbre cero ni observación neutra.
    pub fn try_update(&mut self, measurement: f64) -> Result<f64, &'static str> {
        self.try_update_with_r(measurement, self.r)
    }

    fn try_update_with_r(&mut self, measurement: f64, r: f64) -> Result<f64, &'static str> {
        if !measurement.is_finite()
            || !self.x.is_finite()
            || !Self::valid_variance(self.p)
            || !Self::valid_variance(self.q)
            || !Self::valid_variance(r)
        {
            return Err("invalid Kalman observation or covariance state");
        }
        let predicted_p = self.p + self.q;
        if !predicted_p.is_finite() || (predicted_p == 0.0 && r == 0.0) {
            return Err("Kalman innovation variance is singular or not representable");
        }
        // K=P-/(P-+R), evaluado como razones <=1: evita overflow de la suma.
        // P+=P-*R/(P-+R), sin producto desbordado ni cancelación de (1-K).
        let (k, next_p) = if predicted_p >= r {
            let ratio = r / predicted_p;
            (1.0 / (1.0 + ratio), r / (1.0 + ratio))
        } else {
            let ratio = predicted_p / r;
            (ratio / (1.0 + ratio), predicted_p / (1.0 + ratio))
        };
        // Una combinación convexa evita overflow de measurement-x con signos opuestos.
        let next_x = (1.0 - k) * self.x + k * measurement;
        if !next_x.is_finite() || !next_p.is_finite() {
            return Err("Kalman posterior is not representable");
        }
        self.x = next_x;
        self.p = next_p;
        self.r = r;
        Ok(next_x)
    }

    /// Actualiza dinámicamente el ruido de medición basado en la volatilidad reciente
    #[inline(always)]
    pub fn update_with_dynamic_r(&mut self, measurement: f64, dynamic_r: f64) -> f64 {
        // Compatibilidad: un R dinámico inválido conserva el R anterior. La
        // publicación del nuevo R se realiza sólo si toda la actualización pasa.
        let r = if dynamic_r.is_finite() && dynamic_r > 0.0 {
            dynamic_r
        } else {
            self.r
        };
        self.try_update_with_r(measurement, r).unwrap_or(self.x)
    }

    /// Modula el ruido de medición R en función de la volatilidad instantánea (ej. ATR o Garman-Klass)
    /// para evitar sobreajuste a ruidos de microestructura.
    #[inline(always)]
    pub fn update_with_instantaneous_volatility(
        &mut self,
        measurement: f64,
        inst_vol: f64,
        base_r: f64,
    ) -> f64 {
        let safe_vol = if inst_vol.is_finite() && inst_vol > 0.0 {
            inst_vol
        } else {
            0.001
        };
        let safe_base = if base_r.is_finite() && base_r > 0.0 {
            base_r
        } else {
            self.r
        };
        let modulated_r = safe_base * (1.0 + (safe_vol * 100.0).clamp(0.0, 100.0));
        self.update_with_dynamic_r(measurement, modulated_r)
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

    #[test]
    fn test_kalman_filter_instantaneous_volatility_modulation() {
        let mut kf = KalmanFilter1D::new(100.0, 1.0, 1e-4, 0.1);
        let est_calm = kf.update_with_instantaneous_volatility(105.0, 0.001, 0.1);

        let mut kf_noisy = KalmanFilter1D::new(100.0, 1.0, 1e-4, 0.1);
        let est_noisy = kf_noisy.update_with_instantaneous_volatility(105.0, 0.50, 0.1);

        // In high volatility, R is larger, so the update towards 105.0 is dampened
        assert!((est_noisy - 100.0) < (est_calm - 100.0));
        assert!(est_noisy > 100.0);
    }
}
