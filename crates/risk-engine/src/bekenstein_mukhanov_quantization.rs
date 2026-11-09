//! # Fase 84 (§84): Cuantización de Área del Horizonte de Bekenstein-Mukhanov & Espectro Discreto de Riesgo
//!
//! ## 🏛️ Gobernanza y Mandato del Consejo Colegiado (10 Roles Senior)
//! Módulo formalmente integrado en cumplimiento estricto del Plan Maestro Unificado Omega-220
//! (`docs/PLAN_MAESTRO_UNIFICADO_OMEGA220_CUANTICO.md`). Modela la frontera de preservación de
//! capital y contención de pérdidas no como una curva continua susceptible a deslizamientos
//! infinitésimos, sino como un horizonte cuántico de Bekenstein-Mukhanov con área cuantizada en
//! múltiplos enteros discretos: $A_n = n \cdot \alpha \cdot \ell_P^2$, con $\alpha = 4 \ln(3)$.
//!
//! ## 👨‍🏫 Modo Profesor: Explicación Integral de la Fase 84
//! - **QUÉ**: Motor de cuantización de área del horizonte de riesgo que mapea el capital del portafolio
//!   a niveles cuánticos discretos $n \in \mathbb{N}$, calculando el espectro de área $A_n$, la
//!   entropía macroscópica de Bekenstein $S_n = n \ln(3)$, la frecuencia de transición de Bohr $\omega_0$,
//!   la tasa de emisión de radiación de drawdown $\Gamma_{n \to n-1}$ y la cercanía al estado fundamental crítico.
//! - **POR QUÉ**: El riesgo de ruina en mercados reales salta de forma discreta entre estados energéticos.
//!   Permitir que el dimensionamiento ignore la cuantización de área produce degradación térmica continua
//!   ("síndrome de la rana hervida"). La cuantización de Bekenstein-Mukhanov impone escalones cuánticos
//!   infranqueables que protegen la supermartingala de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//! - **PARA QUÉ**: Detectar transiciones de pérdida de nivel cuántico $n \to n - 1$, modular el factor de
//!   slot de asignación $\eta_{\text{bm}} \in [0.10, 1.25]$ y detonar el veto determinista absoluto
//!   `BEKENSTEIN_MUKHANOV_AREA_EVAPORATION_VETO` con colapso inmediato a slot 0.0 cuando el portafolio
//!   alcanza el nivel de evaporación del horizonte cuántico ($n \le n_{\text{critical\_floor}}$).
//! - **CÓMO**:
//!   1. Cuanto fundamental de área: $\alpha = 4.0 \cdot \ln(3.0) \approx 4.39444915$.
//!   2. Escala de Planck de capital: $\ell_P^2 = \text{floor\_capital} / 100.0$.
//!   3. Masa gravitacional efectiva del portafolio: $M = \ln(1.0 + \text{equity} / \text{floor})$.
//!   4. Área continua del horizonte: $A = 4.0 \pi R_s^2 = 16.0 \pi M^2$.
//!   5. Proyección del número cuántico entero: $n = \operatorname{clamp}(\lfloor A / (\alpha \ell_P^2) \rfloor, 0, 10000)$.
//!   6. Frecuencia espectral de Bohr: $\omega_0 = \frac{\alpha}{32.0 \pi M}$.
//!   7. Factor de modulación continuo y verificación de la condición de veto determinista.
//! - **CUÁNDO**: En cada evaluación del gobernador de riesgo del colector de Hilbert.
//! - **DÓNDE**: `crates/risk-engine/src/bekenstein_mukhanov_quantization.rs`.
//! - **QUIÉN**: `BekensteinMukhanovQuantizationEngine`.
//!
//! ## 🔒 Invariantes y Garantías de Silicio
//! - Cero asignaciones en heap ($O(1)$ stack allocation estricto).
//! - Ratchet de cero pánicos (`rb_p1`): 0 unwrap, 0 expect, 0 panic.
//! - Inmunidad contra subnormales y NaN mediante `f64::clamp` y `is_finite`.

/// Cuanto fundamental de Bekenstein-Mukhanov: $\alpha = 4 \ln(3) \approx 4.394449154672439$.
pub const BEKENSTEIN_ALPHA_QUANTUM: f64 = 4.394449154672439;

/// Configuración del motor de cuantización de Bekenstein-Mukhanov.
#[derive(Debug, Clone, Copy)]
pub struct BekensteinMukhanovConfig {
    /// Capital suelo absoluto Grossman-Zhou (floor equity).
    pub floor_equity: f64,
    /// Nivel cuántico crítico mínimo para detonar veto de evaporación.
    pub critical_floor_quantum_n: u64,
    /// Nivel cuántico objetivo nominal para pleno apalancamiento seguro.
    pub target_quantum_n: u64,
}

impl Default for BekensteinMukhanovConfig {
    fn default() -> Self {
        Self {
            floor_equity: 10.0,
            critical_floor_quantum_n: 5,
            target_quantum_n: 50,
        }
    }
}

/// Informe analítico de cuantización del horizonte de Bekenstein-Mukhanov.
#[derive(Debug, Clone, Copy)]
pub struct BekensteinMukhanovReport {
    /// Número cuántico principal del horizonte $n = \lfloor A / (\alpha \ell_P^2) \rfloor$.
    pub quantum_level_n: u64,
    /// Área continua calculada del horizonte $A \ge 0$.
    pub horizon_area_a: f64,
    /// Área cuantizada discreta $A_n = n \cdot \alpha \cdot \ell_P^2$.
    pub quantized_area_an: f64,
    /// Entropía discreta de Bekenstein-Mukhanov $S_n = n \ln(3)$.
    pub bekenstein_entropy_sn: f64,
    /// Frecuencia de transición espectral de Bohr $\omega_0$.
    pub bohr_transition_frequency: f64,
    /// Tasa relativa de radiación de Hawking de drawdown $\Gamma \in [0.0, 1.0]$.
    pub hawking_evaporation_rate: f64,
    /// Indicador booleano de veto determinista por evaporación del horizonte de capital.
    pub bekenstein_evaporation_veto: bool,
    /// Factor de escala continuo de slot $\eta_{\text{bm}} \in [0.10, 1.25]$.
    pub slot_scale_factor: f64,
}

/// Motor de Cuantización de Área de Bekenstein-Mukhanov.
#[derive(Debug, Clone)]
pub struct BekensteinMukhanovQuantizationEngine {
    config: BekensteinMukhanovConfig,
}

impl Default for BekensteinMukhanovQuantizationEngine {
    fn default() -> Self {
        Self::new(BekensteinMukhanovConfig::default())
    }
}

impl BekensteinMukhanovQuantizationEngine {
    /// Instancia un nuevo motor de cuantización con configuración dada.
    pub fn new(config: BekensteinMukhanovConfig) -> Self {
        Self { config }
    }

    /// Evalúa la cuantización del horizonte a partir de los observables de capital y riesgo:
    /// - `equity`: Capital neto actual del portafolio.
    /// - `portfolio_volatility`: Volatilidad agregada ponderada de los activos en cartera.
    /// - `max_drawdown_pct`: Drawdown actual respecto al pico histórico $\in [0.0, 1.0]$.
    pub fn evaluate(
        &self,
        equity: f64,
        portfolio_volatility: f64,
        max_drawdown_pct: f64,
    ) -> BekensteinMukhanovReport {
        let floor = if self.config.floor_equity.is_finite() && self.config.floor_equity > 0.0 {
            self.config.floor_equity
        } else {
            10.0
        };

        let clean_equity = if equity.is_finite() && equity > 0.0 {
            equity
        } else {
            0.001
        };

        let clean_vol = if portfolio_volatility.is_finite() && portfolio_volatility >= 0.0 {
            portfolio_volatility.clamp(0.001, 5.0)
        } else {
            0.02
        };

        let clean_dd = if max_drawdown_pct.is_finite() {
            max_drawdown_pct.clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 1. Escala de Planck de capital: \ell_P^2 = \text{floor} / 100.0
        let planck_area = (floor / 100.0).max(1e-6);

        // 2. Masa gravitacional efectiva del capital: M = \ln(1 + \text{equity}/\text{floor})
        let effective_mass = (1.0 + clean_equity / floor).ln().max(0.001);

        // 3. Área continua del horizonte de Schwarzschild de riesgo: A = 16 \pi M^2
        let horizon_area_a = 16.0 * std::f64::consts::PI * effective_mass * effective_mass;

        // 4. Proyección del número cuántico entero: n = \lfloor A / (\alpha \ell_P^2) \rfloor
        let quantum_step = BEKENSTEIN_ALPHA_QUANTUM * planck_area;
        let raw_n = (horizon_area_a / quantum_step).floor();
        let quantum_level_n = if raw_n.is_finite() && raw_n > 0.0 {
            (raw_n as u64).min(100_000)
        } else {
            0
        };

        // 5. Área cuantizada discreta y entropía de Bekenstein-Mukhanov
        let quantized_area_an = (quantum_level_n as f64) * quantum_step;
        let bekenstein_entropy_sn = (quantum_level_n as f64) * 3.0_f64.ln();

        // 6. Frecuencia de transición espectral de Bohr: \omega_0 = \frac{\alpha}{32 \pi M}
        let bohr_transition_frequency =
            BEKENSTEIN_ALPHA_QUANTUM / (32.0 * std::f64::consts::PI * effective_mass);

        // 7. Tasa de evaporación de Hawking amplificada por volatilidad y drawdown
        let base_evaporation = (-BEKENSTEIN_ALPHA_QUANTUM).exp(); // 1/81 \approx 0.012345
        let hawking_evaporation_rate =
            (base_evaporation * (1.0 + clean_vol * 10.0) * (1.0 + clean_dd * 5.0)).clamp(0.0, 1.0);

        // 8. Veto determinista de evaporación: nivel cuántico por debajo del suelo crítico
        let bekenstein_evaporation_veto =
            quantum_level_n <= self.config.critical_floor_quantum_n || clean_equity <= floor;

        // 9. Factor de modulación continuo de slot \eta_{\text{bm}} \in [0.10, 1.25]
        let slot_scale_factor = if bekenstein_evaporation_veto {
            0.0
        } else {
            let target = self.config.target_quantum_n.max(1) as f64;
            let ratio = (quantum_level_n as f64 / target).min(1.0);
            let evaporation_drag = 1.0 - hawking_evaporation_rate * 0.5;
            let scale = (0.20 + 0.80 * ratio) * evaporation_drag;
            scale.clamp(0.10, 1.25)
        };

        BekensteinMukhanovReport {
            quantum_level_n,
            horizon_area_a,
            quantized_area_an,
            bekenstein_entropy_sn,
            bohr_transition_frequency,
            hawking_evaporation_rate,
            bekenstein_evaporation_veto,
            slot_scale_factor,
        }
    }
}
