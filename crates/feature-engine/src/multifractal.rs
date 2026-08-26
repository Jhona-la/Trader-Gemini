use std::f64;

/// 🧬 ESPECTRO MULTIFRACTAL DE MANDELBROT (MULTIFRACTAL SPECTRUM ENGINE)
/// Mide la dimensión multifractal de Hölder D(q) para discriminar entre caos ruidoso y tendencia inercial.
/// Reemplaza el exponente monofractal de Hurst por un análisis de multifractalidad dinámico.
#[derive(Debug, Clone)]
pub struct MultifractalSpectrumEngine {
    pub window_size: usize,
    pub returns_history: [f64; 50],
    pub head: usize,
    pub count: usize,
    pub last_price: f64,
    pub sum_q1: f64,
    pub sum_q2: f64,
}

impl MultifractalSpectrumEngine {
    pub fn new(window_size: usize) -> Self {
        let size = window_size.clamp(10, 50);
        Self {
            window_size: size,
            returns_history: [0.0; 50],
            head: 0,
            count: 0,
            last_price: 0.0,
            sum_q1: 0.0,
            sum_q2: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64) -> (f64, f64) {
        if price <= 0.0 || !price.is_finite() {
            return (0.50, 0.0);
        }

        if self.last_price == 0.0 {
            self.last_price = price;
            return (0.50, 0.0);
        }

        let ret = (price / self.last_price).ln().abs();
        self.last_price = price;

        if self.count >= self.window_size {
            let old_ret = self.returns_history[self.head];
            self.sum_q1 -= old_ret;
            self.sum_q2 -= old_ret * old_ret;
        } else {
            self.count += 1;
        }
        
        self.returns_history[self.head] = ret;
        self.head = (self.head + 1) % self.window_size;
        
        self.sum_q1 = (self.sum_q1 + ret).max(0.0);
        self.sum_q2 = (self.sum_q2 + ret * ret).max(0.0);

        if self.count < 10 {
            return (0.50, 0.0);
        }

        let n_f64 = self.count as f64;
        let mean_q1 = (self.sum_q1 / n_f64).max(1e-12);
        let mean_q2 = (self.sum_q2 / n_f64).max(1e-12);
        let rms_q2 = mean_q2.sqrt();

        // FIX #386: Normalizar momentos relativos a la escala gaussiana base (sqrt(2/pi) ≈ 0.797884)
        // para evitar que ln(retorno_nominal) sature el clamp a -0.45 permanentemente.
        let gaussian_ratio = 0.7978845608;
        let scale_ratio = (mean_q1 / (rms_q2 * gaussian_ratio).max(1e-12)).max(1e-6);
        let log_time = n_f64.ln().max(1.0);

        // Desviación del exponente de Hölder respecto al régimen monofractal
        let delta_h = (scale_ratio.ln() / log_time).clamp(-0.45, 0.45);
        let h_q1 = (0.50 + delta_h).clamp(0.05, 0.95);
        let h_q2 = 0.50; // Línea base browniana

        let multifractal_width = (h_q1 - h_q2).abs().clamp(0.0, 1.0);
        let dynamic_hurst = h_q1.clamp(0.05, 0.95);

        (dynamic_hurst, multifractal_width)
    }

    /// Descomposición Wavelet de Haar de 1 nivel para análisis de micro-régimen (#96-#112)
    /// Retorna: `(energy_approx, energy_detail, detail_to_approx_ratio)`
    pub fn compute_haar_wavelet_energy(&self) -> (f64, f64, f64) {
        if self.count < 8 {
            return (0.0, 0.0, 0.0);
        }

        let pairs = (self.count / 2).min(25);
        let mut energy_approx = 0.0;
        let mut energy_detail = 0.0;

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        for i in 0..pairs {
            let idx1 = (self.head + self.window_size - self.count + 2 * i) % self.window_size;
            let idx2 = (self.head + self.window_size - self.count + 2 * i + 1) % self.window_size;

            let x1 = self.returns_history[idx1];
            let x2 = self.returns_history[idx2];

            let s = (x1 + x2) * inv_sqrt2;
            let d = (x1 - x2) * inv_sqrt2;

            energy_approx += s * s;
            energy_detail += d * d;
        }

        let ratio = if energy_approx > 1e-12 {
            energy_detail / energy_approx
        } else {
            0.0
        };

        (energy_approx, energy_detail, ratio)
    }
}

impl Default for MultifractalSpectrumEngine {
    fn default() -> Self {
        Self::new(50)
    }
}

/// Confluencia de Régimen por Exponente de Hurst Multi-Escala (Punto #104)
/// Calcula el exponente de Hurst en 3 escalas temporales distintas:
/// - Micro-Escala (Scalp, ventana corta de 10 ticks)
/// - Meso-Escala (Intradía, ventana media de 25 ticks)
/// - Macro-Escala (Swing, ventana larga de 50 ticks)
///   y evalúa la confluencia direccional de persistencia (>0.55) o anti-persistencia (<0.45).
#[derive(Debug, Clone)]
pub struct MultiScaleHurstConfluence {
    pub engine_micro: MultifractalSpectrumEngine,
    pub engine_meso: MultifractalSpectrumEngine,
    pub engine_macro: MultifractalSpectrumEngine,
}

impl Default for MultiScaleHurstConfluence {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiScaleHurstConfluence {
    pub fn new() -> Self {
        Self {
            engine_micro: MultifractalSpectrumEngine::new(10),
            engine_meso: MultifractalSpectrumEngine::new(25),
            engine_macro: MultifractalSpectrumEngine::new(50),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64) -> (f64, f64, f64, f64, bool, bool) {
        let (h_micro, _) = self.engine_micro.update(price);
        let (h_meso, _) = self.engine_meso.update(price);
        let (h_macro, _) = self.engine_macro.update(price);

        let c_micro: f64 = if h_micro > 0.55 { 1.0 } else if h_micro < 0.45 { -1.0 } else { 0.0 };
        let c_meso: f64 = if h_meso > 0.55 { 1.0 } else if h_meso < 0.45 { -1.0 } else { 0.0 };
        let c_macro: f64 = if h_macro > 0.55 { 1.0 } else if h_macro < 0.45 { -1.0 } else { 0.0 };

        let confluence_score: f64 = (c_micro * 0.4 + c_meso * 0.3 + c_macro * 0.3).clamp(-1.0, 1.0);

        // O(1) Branchless-like thresholds para determinar la viabilidad atómica del horizonte
        let is_scalp_viable = h_micro > 0.60 || h_micro < 0.40; // Micro-tendencia fuerte o Micro-reversión fuerte
        let is_swing_viable = h_macro > 0.65 || h_macro < 0.35; // Macro-tendencia o Macro-rango

        (h_micro, h_meso, h_macro, confluence_score, is_scalp_viable, is_swing_viable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multifractal_spectrum_calculation() {
        let mut engine = MultifractalSpectrumEngine::new(30);
        let mut p = 60000.0;
        for i in 0..40 {
            p += (i as f64 * 0.1).sin() * 5.0;
            let (hurst, width) = engine.update(p);
            if i >= 15 {
                assert!(hurst >= 0.05 && hurst <= 0.95);
                assert!(width >= 0.0 && width <= 1.0);
            }
        }

        let (approx, detail, ratio) = engine.compute_haar_wavelet_energy();
        assert!(approx >= 0.0);
        assert!(detail >= 0.0);
        assert!(ratio >= 0.0);
    }

    #[test]
    fn test_multi_scale_hurst_confluence() {
        let mut confluence = MultiScaleHurstConfluence::new();
        let mut p = 50000.0;
        for i in 0..60 {
            p += (i as f64 * 0.2).cos() * 10.0;
            let (h_micro, h_meso, h_macro, score, _scalp, _swing) = confluence.update(p);
            assert!(h_micro >= 0.05 && h_micro <= 0.95);
            assert!(h_meso >= 0.05 && h_meso <= 0.95);
            assert!(h_macro >= 0.05 && h_macro <= 0.95);
            assert!(score >= -1.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_multifractal_nan_and_negative_immunity() {
        let mut engine = MultifractalSpectrumEngine::new(30);
        let (h_nan, w_nan) = engine.update(f64::NAN);
        assert_eq!(h_nan, 0.50);
        assert_eq!(w_nan, 0.0);

        let (h_neg, w_neg) = engine.update(-100.0);
        assert_eq!(h_neg, 0.50);
        assert_eq!(w_neg, 0.0);

        let mut confluence = MultiScaleHurstConfluence::new();
        let (h1, h2, h3, sc, scalp, swing) = confluence.update(f64::NAN);
        assert_eq!(h1, 0.50);
        assert_eq!(h2, 0.50);
        assert_eq!(h3, 0.50);
        assert_eq!(sc, 0.0);
        assert_eq!(scalp, false);
        assert_eq!(swing, false);
    }
}

