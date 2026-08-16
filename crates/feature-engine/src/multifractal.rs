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
        let size = window_size.min(50).max(10);
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
        if price <= 0.0 {
            return (0.50, 0.0);
        }

        if self.last_price == 0.0 {
            self.last_price = price;
            return (0.50, 0.0);
        }

        let ret = (price - self.last_price).abs();
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
        
        self.sum_q1 += ret;
        self.sum_q2 += ret * ret;

        let n = self.count + 1;
        if n < 30 {
            return (0.50, 0.0);
        }

        // Estimar dimensión de Hölder h(q) para q=1 y q=2
        let n_f64 = n as f64;
        let h_q1 = (self.sum_q1 / n_f64).max(1e-9).ln() / (1.0 / n_f64).ln();
        let h_q2 = (self.sum_q2 / n_f64).max(1e-9).sqrt().ln() / (1.0 / n_f64).ln();

        let multifractal_width = (h_q1 - h_q2).abs().clamp(0.0, 1.0);
        let dynamic_hurst = h_q1.clamp(0.1, 0.9);

        (dynamic_hurst, multifractal_width)
    }
}

impl Default for MultifractalSpectrumEngine {
    fn default() -> Self {
        Self::new(100)
    }
}
