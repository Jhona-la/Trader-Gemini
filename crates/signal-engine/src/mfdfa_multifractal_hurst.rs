//! Fase 71: Exponentes de Hurst Multifractales y Análisis de Fluctuación Sin Tendencia (MF-DFA).
//!
//! Este módulo implementa la descomposición multifractal continua del flujo de retornos:
//! $$F_q(s) = \left( \frac{1}{2 N_s} \sum_{\nu=1}^{2 N_s} [F^2(\nu, s)]^{q/2} \right)^{1/q} \sim s^{h(q)}$$
//! extrayendo el espectro de singularidad multifractal $f(\alpha)$ vía transformación de Legendre:
//! $$\tau(q) = q h(q) - 1, \quad \alpha = \frac{d\tau}{dq}, \quad f(\alpha) = q \alpha - \tau(q)$$
//! calculando el ancho del espectro $\Delta \alpha = \alpha_{\max} - \alpha_{\min}$ y la asimetría espectral $A$.
//!
//! # Invariantes y Leyes Físico-Matemáticas:
//! 1. Monotonía del exponente generalizado de Hurst: $h(q)$ decrece o es constante con $q$.
//! 2. Ancho multifractal $\Delta \alpha > 0$: $\Delta \alpha \approx 0$ representa un mono-fractal gaussiano;
//!    $\Delta \alpha > 0.30$ captura intermitencia multifractal pesada y turbulencia de liquidez.
//! 3. Invariante `rb_p1`: Cero pánicos, 0 unwrap, 0 expect, 0 panic! y memoria acotada en stack $O(1)$.

/// Capacidad del buffer circular de retornos temporales en stack $O(1)$.
pub const MFDFA_WINDOW_CAPACITY: usize = 32;

/// Escalas diádicas de análisis $s \in \{4, 8, 16\}$.
pub const MFDFA_SCALES: [usize; 3] = [4, 8, 16];
pub const NUM_SCALES: usize = 3;

/// Órdenes de momentos multifractales $q \in \{-2.0, 0.0, 2.0\}$.
pub const MFDFA_ORDERS: [f64; 3] = [-2.0, 0.0, 2.0];
pub const NUM_ORDERS: usize = 3;

/// Parámetros de configuración del motor de MF-DFA multifractal.
#[derive(Debug, Clone, Copy)]
pub struct MfDfaParams {
    /// Umbral mínimo de ancho de espectro $\Delta \alpha$ para clasificar régimen turbulento.
    pub turbulent_width_threshold: f64,
    /// Epsilon para evitar singularidades y divisiones por cero.
    pub epsilon: f64,
    /// Factor mínimo de convicción de señal.
    pub min_conviction: f64,
    /// Factor máximo de convicción de señal.
    pub max_conviction: f64,
}

impl Default for MfDfaParams {
    fn default() -> Self {
        Self {
            turbulent_width_threshold: 0.40,
            epsilon: 1e-12,
            min_conviction: 0.50,
            max_conviction: 1.35,
        }
    }
}

/// Informe omnisciente del análisis multifractal MF-DFA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MfDfaHurstReport {
    /// Exponente de Hurst estándar / de segundo momento $h(2.0) \in [0.01, 1.50]$.
    pub generalized_hurst_h2: f64,
    /// Exponente de Hurst para fluctuaciones pequeñas $h(-2.0) \in [0.01, 1.50]$.
    pub generalized_hurst_h_minus2: f64,
    /// Exponente de Hurst del orden neutro $h(0.0) \in [0.01, 1.50]$.
    pub generalized_hurst_h0: f64,
    /// Ancho del espectro de singularidad $\Delta \alpha = \alpha_{\max} - \alpha_{\min} \ge 0$.
    pub multifractal_spectrum_width: f64,
    /// Asimetría espectral de singularidad $A \in [-1.0, 1.0]$.
    pub spectral_asymmetry: f64,
    /// Modulador continuo de convicción de señal $\in [0.50, 1.35]$.
    pub signal_conviction_modulator: f64,
    /// Detección de turbulencia intermitente de liquidez multifractal.
    pub is_turbulent_multifractal: bool,
    /// Régimen de persistencia: 1 = persistente ($h(2) > 0.55$), -1 = antipersistente ($h(2) < 0.45$), 0 = browniano.
    pub persistence_regime: i32,
}

impl Default for MfDfaHurstReport {
    fn default() -> Self {
        Self {
            generalized_hurst_h2: 0.50,
            generalized_hurst_h_minus2: 0.50,
            generalized_hurst_h0: 0.50,
            multifractal_spectrum_width: 0.0,
            spectral_asymmetry: 0.0,
            signal_conviction_modulator: 1.0,
            is_turbulent_multifractal: false,
            persistence_regime: 0,
        }
    }
}

/// Motor de MF-DFA para Análisis Multifractal de Microestructura.
#[derive(Debug, Clone)]
pub struct MfDfaMultifractalHurstEngine {
    params: MfDfaParams,
    returns_buffer: [f64; MFDFA_WINDOW_CAPACITY],
    count: usize,
    write_idx: usize,
}

impl MfDfaMultifractalHurstEngine {
    /// Construye una nueva instancia en stack $O(1)$.
    pub fn new(params: MfDfaParams) -> Self {
        Self {
            params,
            returns_buffer: [0.0; MFDFA_WINDOW_CAPACITY],
            count: 0,
            write_idx: 0,
        }
    }

    /// Actualiza el buffer circular con un nuevo retorno de precio y evalúa MF-DFA.
    pub fn update(&mut self, price_return: f64) -> MfDfaHurstReport {
        let safe_ret = if price_return.is_finite() {
            price_return.clamp(-0.5, 0.5)
        } else {
            0.0
        };

        self.returns_buffer[self.write_idx] = safe_ret;
        self.write_idx = (self.write_idx + 1) % MFDFA_WINDOW_CAPACITY;
        if self.count < MFDFA_WINDOW_CAPACITY {
            self.count += 1;
        }

        // Si aún no tenemos suficientes puntos para la escala máxima (16 puntos mínimo),
        // devolvemos reporte por defecto neutro
        if self.count < MFDFA_SCALES[NUM_SCALES - 1] {
            return MfDfaHurstReport::default();
        }

        let n = self.count;

        // 1. Desenrollar retornos ordenados temporalmente en stack
        let mut ordered_returns = [0.0; MFDFA_WINDOW_CAPACITY];
        let start_idx = if self.count < MFDFA_WINDOW_CAPACITY {
            0
        } else {
            self.write_idx
        };

        let mut sum_ret = 0.0;
        for i in 0..n {
            let buf_i = (start_idx + i) % MFDFA_WINDOW_CAPACITY;
            let val = self.returns_buffer[buf_i];
            ordered_returns[i] = val;
            sum_ret += val;
        }

        let mean_ret = sum_ret / (n as f64);

        // 2. Construcción del perfil integrado Y(k) = \sum (X_i - \bar{X})
        let mut profile = [0.0; MFDFA_WINDOW_CAPACITY];
        let mut accum = 0.0;
        for i in 0..n {
            accum += ordered_returns[i] - mean_ret;
            profile[i] = accum;
        }

        // 3. Cálculo de fluctuaciones detrended F_q(s) para cada escala y orden
        // f_q_matrix[order_idx][scale_idx]
        let mut ln_f_q = [[0.0; NUM_SCALES]; NUM_ORDERS];

        for (s_idx, &scale_s) in MFDFA_SCALES.iter().enumerate() {
            let n_s = n / scale_s;
            if n_s == 0 {
                continue;
            }
            let total_segments = 2 * n_s;

            // Almacenar varianzas de cada segmento
            let mut var_segments = [0.0; 2 * (MFDFA_WINDOW_CAPACITY / 4)];

            // Segmentos forward
            for nu in 0..n_s {
                let offset = nu * scale_s;
                let var = Self::calc_segment_detrended_variance(&profile[offset..offset + scale_s]);
                var_segments[nu] = var.max(self.params.epsilon);
            }

            // Segmentos backward
            for nu in 0..n_s {
                let offset = n - (nu + 1) * scale_s;
                let var = Self::calc_segment_detrended_variance(&profile[offset..offset + scale_s]);
                var_segments[n_s + nu] = var.max(self.params.epsilon);
            }

            // Evaluar F_q(s) para q = -2.0, 0.0, 2.0
            for (q_idx, &q_val) in MFDFA_ORDERS.iter().enumerate() {
                if q_val.abs() < 1e-6 {
                    // q = 0: F_0(s) = exp( 1/(2 total_segments) \sum ln(F^2) )
                    let mut sum_ln = 0.0;
                    for k in 0..total_segments {
                        sum_ln += var_segments[k].ln();
                    }
                    let ln_val = 0.5 * (sum_ln / (total_segments as f64));
                    ln_f_q[q_idx][s_idx] = ln_val;
                } else {
                    let mut sum_pow = 0.0;
                    let p = q_val / 2.0;
                    for k in 0..total_segments {
                        sum_pow += var_segments[k].powf(p);
                    }
                    let mean_pow = (sum_pow / (total_segments as f64)).max(self.params.epsilon);
                    let f_q_val = mean_pow.powf(1.0 / q_val);
                    ln_f_q[q_idx][s_idx] = f_q_val.max(self.params.epsilon).ln();
                }
            }
        }

        // 4. Regresión lineal de ln(F_q(s)) vs ln(s) para obtener h(q)
        let ln_scales: [f64; NUM_SCALES] = [
            (MFDFA_SCALES[0] as f64).ln(),
            (MFDFA_SCALES[1] as f64).ln(),
            (MFDFA_SCALES[2] as f64).ln(),
        ];

        let mut h_q = [0.5; NUM_ORDERS];
        for q_idx in 0..NUM_ORDERS {
            h_q[q_idx] = Self::linear_regression_slope(&ln_scales, &ln_f_q[q_idx])
                .clamp(0.01, 1.50);
        }

        let h_minus2 = h_q[0];
        let h_0 = h_q[1];
        let h_2 = h_q[2];

        // 5. Espectro de singularidad multifractal vía transformación de Legendre
        // \tau(q) = q * h(q) - 1
        // \alpha(q) = h(q) + q * h'(q)
        // Aproximación de derivadas de h(q):
        let dh_dq_minus2 = (h_0 - h_minus2) / 2.0;
        let _dh_dq_0 = (h_2 - h_minus2) / 4.0;
        let dh_dq_2 = (h_2 - h_0) / 2.0;

        let alpha_minus2 = (h_minus2 + (-2.0) * dh_dq_minus2).clamp(0.0, 2.0);
        let alpha_0 = h_0.clamp(0.0, 2.0);
        let alpha_2 = (h_2 + 2.0 * dh_dq_2).clamp(0.0, 2.0);

        // Ancho del espectro \Delta \alpha = \alpha_{\max} - \alpha_{\min}
        let alpha_max = alpha_minus2.max(alpha_0).max(alpha_2);
        let alpha_min = alpha_minus2.min(alpha_0).min(alpha_2);
        let multifractal_spectrum_width = (alpha_max - alpha_min).clamp(0.0, 2.0);

        // Asimetría espectral A = (\Delta \alpha_L - \Delta \alpha_R) / (\Delta \alpha_L + \Delta \alpha_R)
        let delta_alpha_l = (alpha_0 - alpha_min).max(0.0);
        let delta_alpha_r = (alpha_max - alpha_0).max(0.0);
        let sum_delta = delta_alpha_l + delta_alpha_r;
        let spectral_asymmetry = if sum_delta > self.params.epsilon {
            ((delta_alpha_l - delta_alpha_r) / sum_delta).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // 6. Detección de turbulencia y modulación de convicción de señal
        let is_turbulent_multifractal = multifractal_spectrum_width >= self.params.turbulent_width_threshold;

        let persistence_regime = if h_2 > 0.55 {
            1
        } else if h_2 < 0.45 {
            -1
        } else {
            0
        };

        // Modulación constructiva en persistencia y bajo ruido multifractal
        let conviction = 1.0 + (h_2 - 0.50) * 0.45 - (multifractal_spectrum_width - 0.20) * 0.30;
        let signal_conviction_modulator = conviction.clamp(self.params.min_conviction, self.params.max_conviction);

        MfDfaHurstReport {
            generalized_hurst_h2: h_2,
            generalized_hurst_h_minus2: h_minus2,
            generalized_hurst_h0: h_0,
            multifractal_spectrum_width,
            spectral_asymmetry,
            signal_conviction_modulator,
            is_turbulent_multifractal,
            persistence_regime,
        }
    }

    /// Calcula la varianza de residuos respecto a una tendencia lineal en un segmento.
    fn calc_segment_detrended_variance(segment: &[f64]) -> f64 {
        let s = segment.len();
        if s < 2 {
            return 1e-12;
        }

        let s_f64 = s as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;

        for (i, &y) in segment.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let denom = s_f64 * sum_xx - sum_x * sum_x;
        let (m, c) = if denom.abs() > 1e-12 {
            let slope = (s_f64 * sum_xy - sum_x * sum_y) / denom;
            let intercept = (sum_y - slope * sum_x) / s_f64;
            (slope, intercept)
        } else {
            (0.0, sum_y / s_f64)
        };

        let mut sum_sq_err = 0.0;
        for (i, &y) in segment.iter().enumerate() {
            let pred = m * (i as f64) + c;
            let err = y - pred;
            sum_sq_err += err * err;
        }

        sum_sq_err / s_f64
    }

    /// Regresión lineal OLS simple para estimar la pendiente (Hurst exponent).
    fn linear_regression_slope(x: &[f64; NUM_SCALES], y: &[f64; NUM_SCALES]) -> f64 {
        let n = NUM_SCALES as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;

        for i in 0..NUM_SCALES {
            sum_x += x[i];
            sum_y += y[i];
            sum_xx += x[i] * x[i];
            sum_xy += x[i] * y[i];
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() > 1e-12 {
            (n * sum_xy - sum_x * sum_y) / denom
        } else {
            0.50
        }
    }
}
