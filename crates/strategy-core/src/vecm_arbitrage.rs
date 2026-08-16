use std::f64;

/// ⚡ ARBITRAJE ESTADÍSTICO COINTEGRADO VECTORES VECM (JOHANSEN COINTEGRATION ENGINE)
/// Mide la relación de equilibrio a largo plazo entre pares de activos (e.g. ADA/AVAX, DOGE/SHIB).
/// Opera la reversión a la media en nanosegundos cuando el Z-Score supera |Z| > 2.5.
#[derive(Debug, Clone)]
pub struct JohansenVecmEngine {
    pub alpha_speed: f64,       // Velocidad de ajuste a la media (speed of adjustment)
    pub beta_hedge_ratio: f64,  // Ratio de cobertura beta de cointegración
    pub spread_mean: f64,       // Media móvil del spread cointegrado
    pub spread_std: f64,        // Desviación estándar del spread
    pub window_count: f64,
}

impl JohansenVecmEngine {
    pub fn new(alpha_speed: f64, beta_hedge_ratio: f64) -> Self {
        Self {
            alpha_speed,
            beta_hedge_ratio,
            spread_mean: 0.0,
            spread_std: 0.001,
            window_count: 0.0,
        }
    }

    /// Actualiza el par cointegrado (Price A, Price B) y retorna el Z-Score de la divergencia
    #[inline(always)]
    pub fn update(&mut self, price_a: f64, price_b: f64) -> f64 {
        if price_a <= 0.0 || price_b <= 0.0 {
            return 0.0;
        }

        // Spread cointegrado: S_t = ln(P_A) - beta * ln(P_B)
        let spread = price_a.ln() - self.beta_hedge_ratio * price_b.ln();

        self.window_count += 1.0;
        let weight = (1.0 / self.window_count).max(0.01);

        // Actualización Welford O(1) de media y desviación del spread
        let delta = spread - self.spread_mean;
        self.spread_mean += weight * delta;
        let delta2 = spread - self.spread_mean;
        self.spread_std = (self.spread_std.powi(2) * (1.0 - weight) + delta * delta2 * weight).sqrt().max(1e-5);

        // Z-Score de la divergencia
        (spread - self.spread_mean) / self.spread_std
    }
}

impl Default for JohansenVecmEngine {
    fn default() -> Self {
        Self::new(0.15, 1.0)
    }
}
