use crate::ewma::Ewma;

/// Heatmap de Correlación O(1) usando Welford / EWMA recursivo.
/// Mide qué tanto se mueven las monedas en conjunto (Market Beta).
pub struct MarketCorrelationHeatmap {
    num_assets: usize,
    returns_ewma: Vec<Ewma>,
    variance_ewma: Vec<Ewma>,
    market_return_ewma: Ewma,
    market_variance_ewma: Ewma,
    covariances: Vec<Ewma>,
    
    last_prices: Vec<f64>,
}

impl MarketCorrelationHeatmap {
    pub fn new(num_assets: usize, period: f64) -> Self {
        Self {
            num_assets,
            returns_ewma: (0..num_assets).map(|_| Ewma::from_period(period)).collect(),
            variance_ewma: (0..num_assets).map(|_| Ewma::from_period(period)).collect(),
            market_return_ewma: Ewma::from_period(period),
            market_variance_ewma: Ewma::from_period(period),
            covariances: (0..num_assets).map(|_| Ewma::from_period(period)).collect(),
            last_prices: vec![0.0; num_assets],
        }
    }

    /// Ingresa un vector de precios en tiempo real para todos los activos simultáneamente.
    /// Retorna la Correlación Media del Mercado [-1.0 a 1.0].
    #[inline(always)]
    pub fn update(&mut self, current_prices: &[f64]) -> f64 {
        if current_prices.len() != self.num_assets {
            return 0.0; // Fail-safe
        }

        let mut current_returns = vec![0.0; self.num_assets];
        let mut market_return = 0.0;

        for i in 0..self.num_assets {
            let last_price = self.last_prices[i];
            let current_price = current_prices[i];
            
            if last_price > 0.0 {
                let ret = (current_price - last_price) / last_price;
                current_returns[i] = ret;
                market_return += ret;
            }
            self.last_prices[i] = current_price;
        }

        market_return /= self.num_assets as f64;
        let market_mean = self.market_return_ewma.update(market_return);
        
        let market_dev = market_return - market_mean;
        let market_var = self.market_variance_ewma.update(market_dev * market_dev);

        let mut sum_correlation = 0.0;
        let mut valid_assets = 0.0;

        for i in 0..self.num_assets {
            let ret = current_returns[i];
            if ret == 0.0 && self.last_prices[i] == 0.0 {
                continue; // No data yet
            }

            let mean_i = self.returns_ewma[i].update(ret);
            let dev_i = ret - mean_i;
            let var_i = self.variance_ewma[i].update(dev_i * dev_i);
            
            let cov_i = self.covariances[i].update(dev_i * market_dev);

            if var_i > 0.0 && market_var > 0.0 {
                let correlation = cov_i / (var_i.sqrt() * market_var.sqrt());
                // Clamp correlation entre -1 y 1 para evitar errores de coma flotante
                sum_correlation += correlation.clamp(-1.0, 1.0);
                valid_assets += 1.0;
            }
        }

        if valid_assets > 0.0 {
            sum_correlation / valid_assets
        } else {
            0.0
        }
    }
}
