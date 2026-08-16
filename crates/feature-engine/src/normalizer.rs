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
        if rolling_mad <= 1e-9 {
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
