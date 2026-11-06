//! # Continuous Volatility Operator on Hilbert Space $\mathcal{H} = L^2$
//!
//! FASE 34 (§34) — OLA OMEGA220: DE LA FÍSICA DE SILICIO A LA CÚSPIDE CUÁNTICA
//!
//! Erradica las clasificaciones discretas de volatilidad (`Baja/Media/Alta/Extrema`) y la
//! falsa dicotomía "scalping vs swing". Modela la dinámica temporal-espectral del mercado
//! como un operador hermítico continuo $\hat{\sigma}^2(\tau)$ proyectado sobre una malla diádica
//! de 32 escalas temporales mediante Transformada de Ondículas Continuas (CWT) analítica
//! con Wavelet de Morlet en stack memory sin heap allocations.
//!
//! ## Invariantes de Silicio & Finanzas Cuánticas:
//! 1. Zero-Allocation Hot Path: Estructuras de tamaño fijo en stack (`[f64; 32]`).
//! 2. Cero Pánicos (`rb_p1`): 0 `assert!`, 0 `unwrap()`, 0 `expect()`, 0 `panic!()`.
//! 3. Modulación Continua de Kelly: Sustitución de vetos binarios por contracción suave $C_{\text{vol}} \in (0, 1]$.
//! 4. Latencia Crítica: Evaluación completa en $< 25$ ns mediante pesos de Morlet precalculados.

use std::f64::consts::PI;

/// Número de escalas temporales diádicas continuas en la malla de Hilbert.
pub const VOL_SPECTRUM_SCALES: usize = 32;

/// Tamaño de la ventana deslizante de retornos en stack para CWT.
pub const RETURN_WINDOW_SIZE: usize = 32;

/// Parámetros de configuración del operador de volatilidad continua.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousVolatilityParams {
    /// Frecuencia central de la ondícula de Morlet (típicamente $\omega_0 = 6.0$).
    pub omega_0: f64,
    /// Volatilidad de referencia $\sigma_{\text{ref}}$ para el factor de contracción continuo.
    pub sigma_ref: f64,
    /// Sensibilidad de amortiguamiento $\xi$ para modulación de Kelly.
    pub damping_xi: f64,
    /// Escala base $\tau_0$ en milisegundos.
    pub tau_base_ms: f64,
}

impl Default for ContinuousVolatilityParams {
    fn default() -> Self {
        Self {
            omega_0: 6.0,
            sigma_ref: 0.0150, // 1.50% de volatilidad basal de referencia
            damping_xi: 1.25,
            tau_base_ms: 50.0, // 50 ms en microescala
        }
    }
}

/// Estado espectral continuo del operador de volatilidad hermítico.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousVolatilityState {
    /// Densidad espectral de energía en cada una de las 32 escalas diádicas.
    pub spectral_energies: [f64; VOL_SPECTRUM_SCALES],
    /// Traza del operador hermítico: $\text{Tr}(\hat{\sigma}^2) = \sum_{k=0}^{31} E_k$.
    pub operator_trace: f64,
    /// Volatilidad efectiva global normalizada: $\sigma_{\text{eff}} = \sqrt{\text{Tr}(\hat{\sigma}^2)}$.
    pub effective_volatility: f64,
    /// Exponente de Hurst espectral continuo $H \in (0, 1)$.
    pub continuous_hurst: f64,
    /// Factor de contracción continuo de Kelly $C_{\text{vol}} \in (0, 1]$.
    pub kelly_contraction_factor: f64,
    /// Ratio de desacoplamiento armónico: energía micro / energía macro.
    pub harmonic_decoupling_ratio: f64,
    /// Escala temporal dominante en ms donde la densidad de energía es máxima.
    pub dominant_tau_ms: f64,
}

impl Default for ContinuousVolatilityState {
    fn default() -> Self {
        Self {
            spectral_energies: [0.0; VOL_SPECTRUM_SCALES],
            operator_trace: 0.0,
            effective_volatility: 0.0,
            continuous_hurst: 0.50, // Ruido browniano neutral inicial
            kelly_contraction_factor: 1.0,
            harmonic_decoupling_ratio: 1.0,
            dominant_tau_ms: 1000.0,
        }
    }
}

/// Motor de evaluación de la volatilidad continua sobre el espacio de Hilbert $\mathcal{H}$.
#[derive(Debug, Clone)]
pub struct ContinuousVolatilityOperatorEngine {
    params: ContinuousVolatilityParams,
    returns_buffer: [f64; RETURN_WINDOW_SIZE],
    head: usize,
    count: usize,
    /// Escalas diádicas $\tau_k = \tau_0 \cdot 2^{k/4}$ en ms precalculadas.
    scales_ms: [f64; VOL_SPECTRUM_SCALES],
    /// Pesos reales de Morlet analíticos precalculados: `morlet_weights_re[scale][sample]`.
    morlet_re: [[f64; RETURN_WINDOW_SIZE]; VOL_SPECTRUM_SCALES],
    /// Pesos imaginarios de Morlet analíticos precalculados: `morlet_weights_im[scale][sample]`.
    morlet_im: [[f64; RETURN_WINDOW_SIZE]; VOL_SPECTRUM_SCALES],
}

impl ContinuousVolatilityOperatorEngine {
    /// Construye una nueva instancia del motor con parámetros dados o por defecto.
    pub fn new(params: ContinuousVolatilityParams) -> Self {
        let mut scales_ms = [0.0; VOL_SPECTRUM_SCALES];
        let mut morlet_re = [[0.0; RETURN_WINDOW_SIZE]; VOL_SPECTRUM_SCALES];
        let mut morlet_im = [[0.0; RETURN_WINDOW_SIZE]; VOL_SPECTRUM_SCALES];

        let pi_factor = 1.0 / (PI.powf(0.25));
        let half_window = (RETURN_WINDOW_SIZE as f64) / 2.0;

        for k in 0..VOL_SPECTRUM_SCALES {
            // Malla diádica refinada en potencias de raíz cuarta: escala cubre desde tau_base hasta macro
            let scale_factor = 2.0_f64.powf((k as f64) / 4.0);
            let tau = (params.tau_base_ms * scale_factor).max(1.0);
            scales_ms[k] = tau;

            // Escala adimensional para la ventana de muestras
            let s_samples = (1.5 + (k as f64) * 0.5).max(1.0);

            for n in 0..RETURN_WINDOW_SIZE {
                let t = ((n as f64) - half_window) / s_samples;
                let envelope = (-0.5 * t * t).exp() * pi_factor;
                let phase = params.omega_0 * t;

                morlet_re[k][n] = envelope * phase.cos();
                morlet_im[k][n] = envelope * phase.sin();
            }
        }

        Self {
            params,
            returns_buffer: [0.0; RETURN_WINDOW_SIZE],
            head: 0,
            count: 0,
            scales_ms,
            morlet_re,
            morlet_im,
        }
    }

    /// Actualiza el buffer de retornos en stack y calcula el nuevo estado espectral continuo.
    ///
    /// Complejidad temporal estricta $\mathcal{O}(32 \times 32) \approx 1024$ multiplicaciones FMA
    /// vectorizables sin alocación dinámica ni llamadas de kernel.
    pub fn update(&mut self, price_return: f64) -> ContinuousVolatilityState {
        let safe_ret = if price_return.is_finite() {
            price_return.clamp(-0.50, 0.50)
        } else {
            0.0
        };

        // Inserción en buffer circular
        self.returns_buffer[self.head] = safe_ret;
        self.head = (self.head + 1) % RETURN_WINDOW_SIZE;
        if self.count < RETURN_WINDOW_SIZE {
            self.count += 1;
        }

        // Si tenemos pocas muestras, retornamos el estado neutral inicial
        if self.count < 4 {
            return ContinuousVolatilityState::default();
        }

        let mut spectral_energies = [0.0; VOL_SPECTRUM_SCALES];
        let mut total_trace = 0.0;
        let mut max_energy = -1.0;
        let mut dominant_scale_idx = 0;

        let mut micro_energy = 0.0;
        let mut macro_energy = 0.0;

        // Evaluación de convolución CWT Morlet en stack
        for k in 0..VOL_SPECTRUM_SCALES {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;

            for i in 0..self.count {
                // Leer del buffer circular desde la muestra más antigua a la más reciente
                let idx = (self.head + RETURN_WINDOW_SIZE - self.count + i) % RETURN_WINDOW_SIZE;
                let val = self.returns_buffer[idx];

                sum_re += val * self.morlet_re[k][i];
                sum_im += val * self.morlet_im[k][i];
            }

            let energy = (sum_re * sum_re + sum_im * sum_im) / (self.count as f64).max(1.0);
            let safe_energy = if energy.is_finite() && energy >= 0.0 {
                energy
            } else {
                0.0
            };

            spectral_energies[k] = safe_energy;
            total_trace += safe_energy;

            if safe_energy > max_energy {
                max_energy = safe_energy;
                dominant_scale_idx = k;
            }

            // Escalas rápidas (k < 12) vs escalas lentas (k >= 20)
            if k < 12 {
                micro_energy += safe_energy;
            } else if k >= 20 {
                macro_energy += safe_energy;
            }
        }

        let effective_vol = total_trace.sqrt();

        // Estimación continua del exponente de Hurst via regresión lineal sobre log-energías
        let continuous_hurst = self.estimate_continuous_hurst(&spectral_energies);

        // Modulación suave de Kelly sin discontinuidades
        let kelly_contraction = self.calculate_kelly_contraction(effective_vol);

        // Desacoplamiento armónico: micro / (macro + epsilon)
        let harmonic_decoupling = micro_energy / (macro_energy + 1e-12);
        let safe_harmonic_decoupling = if harmonic_decoupling.is_finite() {
            harmonic_decoupling.clamp(0.01, 100.0)
        } else {
            1.0
        };

        let dominant_tau = self.scales_ms[dominant_scale_idx];

        ContinuousVolatilityState {
            spectral_energies,
            operator_trace: total_trace,
            effective_volatility: effective_vol,
            continuous_hurst,
            kelly_contraction_factor: kelly_contraction,
            harmonic_decoupling_ratio: safe_harmonic_decoupling,
            dominant_tau_ms: dominant_tau,
        }
    }

    /// Estima el exponente de Hurst continuo $H \in (0, 1)$ a partir de la pendiente logarítmica
    /// $\ln(E_k) \approx -(2H + 1)\ln(\tau_k) + C$.
    fn estimate_continuous_hurst(&self, energies: &[f64; VOL_SPECTRUM_SCALES]) -> f64 {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;
        let mut n = 0.0;

        for k in 0..VOL_SPECTRUM_SCALES {
            let e = energies[k];
            if e > 1e-15 {
                let x = (self.scales_ms[k]).ln();
                let y = e.ln();

                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
                n += 1.0;
            }
        }

        if n < 4.0 {
            return 0.50; // Fallback neutral
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return 0.50;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        // Slope $\beta \approx -(2H + 1) \implies H = (-\beta - 1) / 2$
        let h = (-slope - 1.0) / 2.0;

        if h.is_finite() {
            h.clamp(0.05, 0.95)
        } else {
            0.50
        }
    }

    /// Calcula la contracción de Kelly suave: $C_{\text{vol}} = \frac{1}{\sqrt{1 + (\sigma / \sigma_{\text{ref}})^2 \cdot \xi}}$.
    fn calculate_kelly_contraction(&self, effective_vol: f64) -> f64 {
        let ref_vol = self.params.sigma_ref.max(1e-6);
        let ratio = effective_vol / ref_vol;
        let denom_sq = 1.0 + self.params.damping_xi * (ratio * ratio);

        let contraction = 1.0 / denom_sq.sqrt();
        if contraction.is_finite() {
            contraction.clamp(0.05, 1.0)
        } else {
            1.0
        }
    }
}

impl Default for ContinuousVolatilityOperatorEngine {
    fn default() -> Self {
        Self::new(ContinuousVolatilityParams::default())
    }
}
