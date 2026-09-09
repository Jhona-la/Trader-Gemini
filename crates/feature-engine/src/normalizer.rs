pub struct StatisticalNormalizer {
    mad_multiplier: f64,
}

impl StatisticalNormalizer {
    pub fn new(mad_multiplier: f64) -> Self {
        Self { mad_multiplier }
    }

    /// Detecta outliers usando Median Absolute Deviation (MAD)
    /// Retorna (es_outlier, valor_suavizado_si_es_outlier)
    pub fn filter_outlier(&self, value: f64, rolling_median: f64, rolling_mad: f64) -> (bool, f64) {
        if !value.is_finite() {
            let safe_med = if rolling_median.is_finite() {
                rolling_median
            } else {
                0.0
            };
            return (false, safe_med);
        }
        if !rolling_median.is_finite() || !rolling_mad.is_finite() || rolling_mad <= 1e-9 {
            return (false, value); // No hay suficiente varianza para juzgar
        }

        let deviation = (value - rolling_median).abs();
        let is_outlier = deviation > self.mad_multiplier * rolling_mad;

        if is_outlier {
            // Suavizamos el valor (clipping) al borde del límite aceptable
            let sign = if value > rolling_median { 1.0 } else { -1.0 };
            let clipped = rolling_median + sign * self.mad_multiplier * rolling_mad;
            (true, clipped)
        } else {
            (false, value)
        }
    }
}

/// Estimador de Volatilidad Extrema de Garman-Klass (Punto #035)
/// 8 veces más eficiente estadísticamente que el estimador estándar close-to-close.
/// $\sigma_{\text{GK}}^2 = 0.5 \cdot \left(\ln\frac{H}{L}\right)^2 - (2\ln 2 - 1) \cdot \left(\ln\frac{C}{O}\right)^2$
#[derive(Debug, Clone, Copy)]
pub struct GarmanKlassVolatilityEstimator {
    pub rolling_variance_ema: f64,
    pub alpha: f64,
}

impl Default for GarmanKlassVolatilityEstimator {
    fn default() -> Self {
        Self::new(0.05) // ventana efectiva de ~20 velas
    }
}

impl GarmanKlassVolatilityEstimator {
    pub fn new(alpha: f64) -> Self {
        Self {
            rolling_variance_ema: 0.0001,
            alpha: alpha.clamp(0.001, 1.0),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> f64 {
        if open <= 0.0
            || high <= 0.0
            || low <= 0.0
            || close <= 0.0
            || !open.is_finite()
            || !high.is_finite()
            || !low.is_finite()
            || !close.is_finite()
            || low > high
        {
            return self.rolling_variance_ema.sqrt();
        }

        let hl_ratio = (high / low).max(1.0);
        let co_ratio = (close / open).max(1e-6);

        let log_hl = hl_ratio.ln();
        let log_co = co_ratio.ln();

        let term1 = 0.5 * log_hl * log_hl;
        let term2 = (2.0 * std::f64::consts::LN_2 - 1.0) * log_co * log_co;

        let sample_variance = (term1 - term2).max(1e-12);

        // Actualización EWMA online O(1)
        self.rolling_variance_ema =
            (1.0 - self.alpha) * self.rolling_variance_ema + self.alpha * sample_variance;

        self.rolling_variance_ema.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_normalizer_mad() {
        let norm = StatisticalNormalizer::new(3.0);
        let (is_outlier, val) = norm.filter_outlier(100.0, 10.0, 2.0);
        assert!(is_outlier);
        assert!((val - 16.0).abs() < 1e-6);

        let (is_normal, val_norm) = norm.filter_outlier(12.0, 10.0, 2.0);
        assert!(!is_normal);
        assert_eq!(val_norm, 12.0);
    }

    #[test]
    fn test_garman_klass_volatility_estimator() {
        let mut gk = GarmanKlassVolatilityEstimator::new(0.1);
        let vol = gk.update(60000.0, 60500.0, 59800.0, 60200.0);
        assert!(vol > 0.0 && vol < 0.10);

        let vol_nan = gk.update(f64::NAN, 60500.0, 59800.0, 60200.0);
        assert!(vol_nan.is_finite());
    }

    #[test]
    fn test_statistical_normalizer_nan_and_zero_mad() {
        let norm = StatisticalNormalizer::new(3.0);
        let (is_outlier_nan, val_nan) = norm.filter_outlier(f64::NAN, 10.0, 2.0);
        assert!(!is_outlier_nan);
        assert_eq!(val_nan, 10.0);

        let (is_outlier_zero_mad, val_zero) = norm.filter_outlier(50.0, 10.0, 0.0);
        assert_eq!(val_zero, 50.0);
        assert!(!is_outlier_zero_mad);
    }
}
