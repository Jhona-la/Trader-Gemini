/// Algoritmo de Welford Online para Varianza
/// Permite calcular la media, la varianza y la desviación estándar en una sola pasada (O(1) por update)
/// sin sufrir cancelación catastrófica (IEEE-754 precision issues).

#[derive(Debug, Clone, Copy)]
pub struct WelfordOnline {
    pub count: f64,
    pub mean: f64,
    pub m2: f64,
}

impl WelfordOnline {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, val: f64) {
        self.count += 1.0;
        let delta = val - self.mean;
        self.mean += delta / self.count;
        let delta2 = val - self.mean;
        self.m2 += delta * delta2;
    }

    #[inline(always)]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    #[inline(always)]
    pub fn variance(&self) -> f64 {
        if self.count < 2.0 {
            0.0
        } else {
            self.m2 / (self.count - 1.0)
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
        if std == 0.0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_online_mean_variance_and_zscore() {
        let mut welford = WelfordOnline::new();
        let samples = [10.0, 20.0, 30.0, 40.0, 50.0];
        for &s in &samples {
            welford.update(s);
        }

        assert_eq!(welford.count, 5.0);
        assert_eq!(welford.mean(), 30.0);
        assert_eq!(welford.variance(), 250.0);
        let std = (250.0f64).sqrt();
        assert!((welford.std_dev() - std).abs() < 1e-10);

        let z = welford.z_score(30.0);
        assert_eq!(z, 0.0);
        let z_high = welford.z_score(30.0 + std);
        assert!((z_high - 1.0).abs() < 1e-10);
    }
}

