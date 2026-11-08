//! Fase 68: Ecuación Dispersiva No-Lineal de Schrödinger Cúbica (NLS) y Solitones Envolventes de Peregrine.
//!
//! Este módulo implementa la solución analítica de la ecuación de Schrödinger no lineal 1D focalizante:
//! $$i \partial_t \psi + \frac{1}{2} \partial_{xx} \psi + |\psi|^2 \psi = 0$$
//! detectando ondas rebeldes (rogue waves) de Peregrine en el flujo de órdenes institucional:
//! $$\psi_P(x, t) = \left( 1 - \frac{4 (1 + 2 i t)}{1 + 4 x^2 + 4 t^2} \right) e^{i t}$$
//! cuya amplitud pico $|\psi_P(0, 0)| = 3.0$ triplica el fondo continuo de liquidez.
//!
//! # Invariantes y Leyes Físico-Matemáticas:
//! 1. Amplitud Teórica de Cresta: $|\psi_P(0, 0)| = 3.0$.
//! 2. Inestabilidad Modulacional de Benjamin-Feir: $\text{BFI} > 1.0$ induce la emergencia de ondas solitarias gigantes.
//! 3. Transformación Hidrodinámica de Madelung: Densidad $\rho = |\psi|^2$, velocidad cuántica $u = \partial_x \arg(\psi)$.
//! 4. Invariante `rb_p1`: Cero pánicos, 0 unwrap, 0 expect, 0 panic! y memoria acotada en stack $O(1)$.

/// Dimensión de la rejilla espacial discreta en stack.
pub const NLS_GRID_POINTS: usize = 16;

/// Parámetros de configuración del motor de solitones de Peregrine y NLS.
#[derive(Debug, Clone, Copy)]
pub struct NlsPeregrineParams {
    /// Umbral de ratio de amplitud para clasificar un solitón de Peregrine activo (típicamente 2.20).
    pub rogue_wave_threshold: f64,
    /// Umbral mínimo del índice de Benjamin-Feir para inestabilidad modulacional.
    pub min_bfi_threshold: f64,
    /// Epsilon para divisiones numéricas seguras.
    pub epsilon: f64,
    /// Coeficiente de impulso direccional de grupo.
    pub group_velocity_gain: f64,
}

impl Default for NlsPeregrineParams {
    fn default() -> Self {
        Self {
            rogue_wave_threshold: 2.20,
            min_bfi_threshold: 0.90,
            epsilon: 1e-12,
            group_velocity_gain: 1.0,
        }
    }
}

/// Informe omnisciente de la dinámica no lineal de Schrödinger y ondas de Peregrine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NlsPeregrineReport {
    /// Amplitud compleja envolvente normalizada $|\psi_P(x, t)| \ge 0$.
    pub envelope_amplitude: f64,
    /// Fase cuántica local $\arg(\psi_P) \in [-\pi, \pi]$.
    pub quantum_phase: f64,
    /// Índice de Inestabilidad Modulacional de Benjamin-Feir (BFI).
    pub benjamin_feir_index: f64,
    /// Velocidad de grupo hidrodinámica de Madelung $u = \partial_x \phi$.
    pub madelung_group_velocity: f64,
    /// Señal direccional de impulso del solitón sintetizada $\in [-1.0, 1.0]$.
    pub soliton_impulse_signal: f64,
    /// Detección de onda solitaria gigante (Rogue Wave) de Peregrine activa.
    pub peregrine_rogue_wave_detected: bool,
    /// Modulador de convicción para la asignación de slots $\in [0.20, 1.40]$.
    pub signal_conviction_modulator: f64,
}

impl Default for NlsPeregrineReport {
    fn default() -> Self {
        Self {
            envelope_amplitude: 1.0,
            quantum_phase: 0.0,
            benjamin_feir_index: 0.50,
            madelung_group_velocity: 0.0,
            soliton_impulse_signal: 0.0,
            peregrine_rogue_wave_detected: false,
            signal_conviction_modulator: 1.0,
        }
    }
}

/// Motor de Solitones de Peregrine y Ecuación de Schrödinger No Lineal Cúbica.
#[derive(Debug, Clone)]
pub struct NlsPeregrineSolitonEngine {
    params: NlsPeregrineParams,
    prev_phase: f64,
}

impl NlsPeregrineSolitonEngine {
    /// Construye una nueva instancia en stack $O(1)$ con parámetros dados.
    pub fn new(params: NlsPeregrineParams) -> Self {
        Self {
            params,
            prev_phase: 0.0,
        }
    }

    /// Evalúa la ecuación NLS y sintetiza la envolvente de Peregrine.
    ///
    /// # Argumentos:
    /// - `price_ret`: Retorno continuo instantáneo de precio.
    /// - `order_flow_imbalance`: OFI normalizado $\in [-1, 1]$.
    /// - `spread_bps`: Spread actual en puntos básicos $\ge 0$.
    /// - `instantaneous_vol`: Volatilidad continua actual $\ge 0$.
    pub fn update(
        &mut self,
        price_ret: f64,
        order_flow_imbalance: f64,
        spread_bps: f64,
        instantaneous_vol: f64,
    ) -> NlsPeregrineReport {
        let safe_ofi = if order_flow_imbalance.is_finite() {
            order_flow_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let safe_ret = if price_ret.is_finite() {
            price_ret.clamp(-0.5, 0.5)
        } else {
            0.0
        };

        let safe_spread = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.1, 100.0)
        } else {
            1.0
        };

        let safe_vol = if instantaneous_vol.is_finite() && instantaneous_vol > 0.0 {
            instantaneous_vol.clamp(0.0001, 1.0)
        } else {
            0.01
        };

        // 1. Mapeo a coordenadas espaciotemporales normalizadas (x, t) del paquete de ondas
        // x \in [-1.5, 1.5]: coordenada espacial dictada por el OFI y retorno relativo
        let x = (safe_ofi * 1.2 + safe_ret * 20.0).clamp(-2.0, 2.0);
        // t \in [-1.0, 1.0]: coordenada temporal dictada por la fricción de spread y volatilidad
        let t = ((safe_spread * 0.05 - 0.20) / (safe_vol * 10.0 + 0.1)).clamp(-2.0, 2.0);

        // 2. Solución Analítica de Peregrine:
        // \psi_P(x, t) = \left( 1 - \frac{4 (1 + 2 i t)}{1 + 4 x^2 + 4 t^2} \right) e^{i t}
        let denom = 1.0 + 4.0 * x * x + 4.0 * t * t;
        let real_factor = 1.0 - 4.0 / denom;
        let imag_factor = -8.0 * t / denom;

        // Multiplicación por e^{i t} = \cos(t) + i \sin(t)
        let cos_t = t.cos();
        let sin_t = t.sin();

        let psi_real = real_factor * cos_t - imag_factor * sin_t;
        let psi_imag = real_factor * sin_t + imag_factor * cos_t;

        // Amplitud envolvente |\psi_P| y fase
        let envelope_amplitude = (psi_real * psi_real + psi_imag * psi_imag).sqrt();
        let quantum_phase = psi_imag.atan2(psi_real);

        // 3. Índice de Benjamin-Feir (BFI):
        // BFI = \frac{\sqrt{2} \cdot a_0 \cdot k_0}{\Delta k}
        let carrier_amp = (safe_vol * 10.0).clamp(0.1, 3.0);
        let dispersion_width = (safe_spread * 0.1).clamp(0.1, 5.0);
        let benjamin_feir_index = (1.41421356 * carrier_amp / dispersion_width).clamp(0.0, 5.0);

        // 4. Velocidad de Grupo Hidrodinámica de Madelung:
        // u = \partial_x \phi \approx \frac{\Delta \phi}{\Delta x}
        let dphi = quantum_phase - self.prev_phase;
        let madelung_group_velocity = (dphi * self.params.group_velocity_gain).clamp(-5.0, 5.0);
        self.prev_phase = quantum_phase;

        // 5. Detección de Onda Rebelde de Peregrine
        let peregrine_rogue_wave_detected = envelope_amplitude >= self.params.rogue_wave_threshold
            && benjamin_feir_index >= self.params.min_bfi_threshold;

        // 6. Señal de Impulso Direccional y Modulación de Convicción:
        let directional_polarity = if safe_ofi >= 0.0 { 1.0 } else { -1.0 };
        let soliton_intensity = ((envelope_amplitude - 1.0) / 2.0).clamp(0.0, 1.0);
        let soliton_impulse_signal = directional_polarity * soliton_intensity;

        let signal_conviction_modulator = if peregrine_rogue_wave_detected {
            // Boost constructivo de alta convicción ante onda gigante coherente
            1.15 + 0.20 * (envelope_amplitude - 2.20).clamp(0.0, 1.0)
        } else {
            let base = 0.85 + 0.15 * envelope_amplitude.clamp(0.5, 1.5);
            base.clamp(0.20, 1.25)
        };

        NlsPeregrineReport {
            envelope_amplitude,
            quantum_phase,
            benjamin_feir_index,
            madelung_group_velocity,
            soliton_impulse_signal,
            peregrine_rogue_wave_detected,
            signal_conviction_modulator,
        }
    }
}
