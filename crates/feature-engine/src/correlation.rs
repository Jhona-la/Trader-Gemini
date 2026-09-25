use crate::ewma::Ewma;

/// Media de correlaciones EWMA de cada activo con la cesta equiponderada que
/// lo incluye. NO es beta ni una matriz de correlaciones por pares. O(N) por
/// actualización, memoria O(N), sin asignaciones tras construir el estimador.
/// La API exige cortes completos sincronizados por el llamador; no infiere reloj.
pub struct MarketCorrelationHeatmap {
    num_assets: usize,
    returns_ewma: Vec<Ewma>,
    variance_ewma: Vec<Ewma>,
    market_return_ewma: Ewma,
    market_variance_ewma: Ewma,
    covariances: Vec<Ewma>,

    last_prices: Vec<f64>,
    returns_buffer: Vec<f64>,
    staged_moments: Vec<(Ewma, Ewma, Ewma)>,
}

impl MarketCorrelationHeatmap {
    /// # Panics
    /// Si num_assets es cero o period no es finito >=1. Para configuración
    /// externa usar try_new y manejar el error antes de iniciar el consumidor.
    pub fn new(num_assets: usize, period: f64) -> Self {
        Self::try_new(num_assets, period).expect("invalid basket size or EWMA period")
    }

    pub fn try_new(num_assets: usize, period: f64) -> Result<Self, &'static str> {
        if num_assets == 0 {
            return Err("correlation requires at least one asset");
        }
        let seed = Ewma::try_from_period(period)?;
        Ok(Self {
            num_assets,
            returns_ewma: (0..num_assets).map(|_| Ewma::from_period(period)).collect(),
            variance_ewma: (0..num_assets).map(|_| Ewma::from_period(period)).collect(),
            market_return_ewma: Ewma::from_period(period),
            market_variance_ewma: Ewma::from_period(period),
            covariances: (0..num_assets).map(|_| Ewma::from_period(period)).collect(),
            last_prices: vec![0.0; num_assets],
            returns_buffer: vec![0.0; num_assets],
            staged_moments: vec![(seed, seed, seed); num_assets],
        })
    }

    /// Ingresa un vector de precios en tiempo real para todos los activos simultáneamente.
    /// Retorna la Correlación Media del Mercado [-1.0 a 1.0].
    #[inline(always)]
    pub fn update(&mut self, current_prices: &[f64]) -> f64 {
        self.try_update(current_prices)
            .ok()
            .flatten()
            .unwrap_or(0.0)
    }

    /// None = todavía sin correlación identificable (baseline o varianza cero).
    /// Err = corte inválido; precios y momentos aceptados permanecen intactos.
    /// El wrapper legacy devuelve cero en ambos casos: para calidad usar esta API.
    pub fn try_update(&mut self, current_prices: &[f64]) -> Result<Option<f64>, &'static str> {
        if current_prices.len() != self.num_assets {
            return Err("incomplete price cross-section");
        }
        if current_prices.iter().any(|p| !p.is_finite() || *p <= 0.0) {
            return Err("prices must be finite and positive");
        }
        if self.last_prices[0] == 0.0 {
            self.last_prices.copy_from_slice(current_prices);
            return Ok(None);
        }

        let mut market_return = 0.0;

        for (i, (&current_price, last_price)) in current_prices
            .iter()
            .zip(self.last_prices.iter())
            .enumerate()
        {
            let ret = (current_price - *last_price) / *last_price;
            if !ret.is_finite() {
                return Err("return is not representable");
            }
            self.returns_buffer[i] = ret;
            market_return += ret / self.num_assets as f64;
        }

        let mut next_market_mean = self.market_return_ewma;
        let mut next_market_var = self.market_variance_ewma;
        let market_mean = next_market_mean.update(market_return);

        let market_dev = market_return - market_mean;
        let market_square = market_dev * market_dev;
        if !market_return.is_finite() || !market_mean.is_finite() || !market_square.is_finite() {
            return Err("market moment is not representable");
        }
        let market_var = next_market_var.update(market_square);

        let mut sum_correlation = 0.0;
        let mut valid_assets = 0.0;

        for (i, &ret) in self.returns_buffer.iter().enumerate().take(self.num_assets) {
            let mut next_mean = self.returns_ewma[i];
            let mut next_var = self.variance_ewma[i];
            let mut next_cov = self.covariances[i];
            let mean_i = next_mean.update(ret);
            let dev_i = ret - mean_i;
            let square = dev_i * dev_i;
            let product = dev_i * market_dev;
            if !mean_i.is_finite() || !square.is_finite() || !product.is_finite() {
                return Err("asset moment is not representable");
            }
            let var_i = next_var.update(square);
            let cov_i = next_cov.update(product);
            if !var_i.is_finite() || !cov_i.is_finite() || !market_var.is_finite() {
                return Err("EWMA moment is not representable");
            }
            self.staged_moments[i] = (next_mean, next_var, next_cov);

            if var_i > 0.0 && market_var > 0.0 {
                let correlation = (cov_i / var_i.sqrt()) / market_var.sqrt();
                if !correlation.is_finite() {
                    return Err("correlation is not representable");
                }
                // Clamp correlation entre -1 y 1 para evitar errores de coma flotante
                sum_correlation += correlation.clamp(-1.0, 1.0);
                valid_assets += 1.0;
            }
        }

        self.last_prices.copy_from_slice(current_prices);
        self.market_return_ewma = next_market_mean;
        self.market_variance_ewma = next_market_var;
        for (i, &(mean, variance, covariance)) in self.staged_moments.iter().enumerate() {
            self.returns_ewma[i] = mean;
            self.variance_ewma[i] = variance;
            self.covariances[i] = covariance;
        }
        Ok((valid_assets > 0.0).then(|| sum_correlation / valid_assets))
    }
}
