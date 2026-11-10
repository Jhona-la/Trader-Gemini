//! # Fase 105 (§105) — Métrica de Agujero Negro Kerr-Newman & Termodinámica de Bardeen-Carter-Hawking
//!
//! ## Modo Profesor: 7 Pilares de la Relatividad General y Termodinámica de Agujeros Negros
//!
//! ### 1. QUÉ (Definición Teórica y Matemática)
//! Solución exacta electro-vacía de Einstein-Maxwell que describe un agujero negro con masa $M$,
//! momento angular de espín $a = J/M$ y carga eléctrica neta $Q$:
//! $$ds^2 = -\frac{\Delta}{\rho^2}\left(dt - a\sin^2\theta d\phi\right)^2 + \frac{\sin^2\theta}{\rho^2}\left((r^2+a^2)d\phi - a dt\right)^2 + \frac{\rho^2}{\Delta}dr^2 + \rho^2 d\theta^2$$
//! donde $\rho^2 = r^2 + a^2\cos^2\theta$ y $\Delta = r^2 - 2Mr + a^2 + Q^2$.
//!
//! ### 2. POR QUÉ (Fundamentación de Mercado y Dinámica de Riesgo)
//! En el micro-espacio de liquidez, la masa $M$ modela la volatilidad y riesgo total, el espín $a$ modela
//! el momentum direccional o persistencia de tendencia, y la carga $Q$ cuantifica el desbalance neto
//! acumulado del flujo de órdenes (Order Flow Imbalance).
//! Si $M^2 < a^2 + Q^2$, la función $\Delta$ carece de raíces reales; el horizonte de eventos desaparece
//! y se expone una singularidad desnuda violando el Principio de Censura Cósmica de Penrose (colapso de
//! liquidez o flash crash inminente sin protección causal).
//!
//! ### 3. PARA QUÉ (Impacto en PnL, Ville y Crecimiento Compuesto)
//! Blindar el portafolio ante singularidades desnudas y evaporación térmica descontrolada de margen:
//! - Detonar `KERR_NEWMAN_NAKED_SINGULARITY_VETO` cuando $M^2 \le (a^2 + Q^2) \times \text{margin}$.
//! - Detonar `HAWKING_THERMAL_EVAPORATION_VETO` cuando la temperatura de radiación $T_H > T_{\text{crit}}$.
//! - Escalar la asignación de capital $\eta_{\text{KN}} \in [0.10, 1.25]$ mediante la eficiencia del Proceso de Penrose
//!   $\eta_{\text{Penrose}} = 1 - M_{\text{irr}} / M$.
//!
//! ### 4. CÓMO (Implementación en Silicio y Algorítmica)
//! - Evaluación analítica en $O(1)$ sin heap ni llamadas al sistema.
//! - Cálculo de horizontes de Killing: $r_\pm = M \pm \sqrt{\max(0, M^2 - a^2 - Q^2)}$.
//! - Ergosfera ecuatorial ($\theta = \pi/2$): $r_E = M + \sqrt{\max(0, M^2 - Q^2)}$.
//! - Masa irreducible: $M_{\text{irr}} = \sqrt{(r_+^2 + a^2)/4}$.
//! - Temperatura de Hawking: $T_H = \frac{r_+ - M}{2\pi(r_+^2 + a^2)}$.
//! - Entropía de Bekenstein-Hawking: $S_{BH} = \pi(r_+^2 + a^2)$.
//!
//! ### 5. CUÁNDO (Condiciones de Activación)
//! En cada tick de microestructura durante la síntesis holística del grafo omnisciente de riesgo.
//!
//! ### 6. DÓNDE (Ubicación en la Arquitectura)
//! `crates/risk-engine/src/kerr_newman_black_hole.rs`, exportado en `crates/risk-engine/src/lib.rs`.
//!
//! ### 7. QUIÉN (Componente y Rol)
//! `KerrNewmanBlackHoleEngine`, bajo la custodia del Rol 3 (Risk Manager) y Rol 8 (Físico Relativista).

use core::f64::consts::PI;

/// Parámetros de configuración para el análisis de agujero negro Kerr-Newman.
#[derive(Debug, Clone, Copy)]
pub struct KerrNewmanConfig {
    /// Masa mínima para considerar formación de horizonte gravitatorio.
    pub min_mass_horizon: f64,
    /// Temperatura crítica de Hawking máxima admisible antes de veto de evaporación.
    pub max_hawking_temp: f64,
    /// Factor de margen para el Principio de Censura Cósmica de Penrose (e.g. 1.05 = 5% buffer).
    pub cosmic_censorship_margin: f64,
    /// Escala de arrastre inercial de la ergosfera.
    pub ergosphere_drag_scale: f64,
}

impl Default for KerrNewmanConfig {
    fn default() -> Self {
        Self {
            min_mass_horizon: 0.001,
            max_hawking_temp: 2.50,
            cosmic_censorship_margin: 1.05,
            ergosphere_drag_scale: 1.0,
        }
    }
}

/// Reporte diagnóstico emitido por el motor Kerr-Newman.
#[derive(Debug, Clone, Copy)]
pub struct KerrNewmanReport {
    /// Radio del horizonte exterior de eventos $r_+$.
    pub event_horizon_r_plus: f64,
    /// Radio del horizonte interior de Cauchy $r_-$.
    pub inner_cauchy_horizon_r_minus: f64,
    /// Radio de la ergosfera en el plano ecuatorial ($\theta = \pi/2$).
    pub ergosphere_radius_equator: f64,
    /// Temperatura termodinámica de Hawking $T_H$.
    pub hawking_temperature: f64,
    /// Entropía geométrica de Bekenstein-Hawking $S_{BH} = A / 4$.
    pub bekenstein_hawking_entropy: f64,
    /// Masa irreducible de Christodoulou $M_{\text{irr}}$.
    pub irreducible_mass: f64,
    /// Eficiencia máxima teórica de extracción de energía de Penrose $\eta_{\text{Penrose}}$.
    pub penrose_efficiency: f64,
    /// Factor continuo de escala de slot $\eta_{\text{KN}} \in [0.10, 1.25]$.
    pub slot_scaling_factor: f64,
    /// Veto estricto por violación de censura cósmica (singularidad desnuda super-extremal).
    pub naked_singularity_veto: bool,
    /// Veto estricto por evaporación térmica descontrolada de Hawking.
    pub hawking_thermal_evaporation_veto: bool,
}

impl Default for KerrNewmanReport {
    fn default() -> Self {
        Self {
            event_horizon_r_plus: 1.0,
            inner_cauchy_horizon_r_minus: 0.5,
            ergosphere_radius_equator: 1.5,
            hawking_temperature: 0.10,
            bekenstein_hawking_entropy: PI,
            irreducible_mass: 0.80,
            penrose_efficiency: 0.20,
            slot_scaling_factor: 1.0,
            naked_singularity_veto: false,
            hawking_thermal_evaporation_veto: false,
        }
    }
}

/// Motor determinista de cálculo de geometría espacio-temporal Kerr-Newman y termodinámica de agujeros negros.
#[derive(Debug, Clone)]
pub struct KerrNewmanBlackHoleEngine {
    config: KerrNewmanConfig,
}

impl KerrNewmanBlackHoleEngine {
    /// Inicializa una nueva instancia con los parámetros de configuración.
    #[inline]
    pub fn new(config: KerrNewmanConfig) -> Self {
        Self { config }
    }

    /// Evalúa la métrica Kerr-Newman a partir de las magnitudes dinámicas de microestructura.
    ///
    /// # Parámetros
    /// - `mass`: Volatilidad realizada o masa efectiva $M > 0$.
    /// - `angular_momentum_spin`: Momentum direccional o parámetro de espín $a = J/M$.
    /// - `charge`: Desbalance acumulado o carga electrostática efectiva $Q$.
    /// - `theta`: Ángulo colatitud en el espacio de fase $\theta \in [0, \pi]$.
    pub fn update(
        &mut self,
        mass: f64,
        angular_momentum_spin: f64,
        charge: f64,
        theta: f64,
    ) -> KerrNewmanReport {
        // 1. Sanitización de entradas bajo el estándar RB-P1
        let m = if mass.is_finite() && mass > self.config.min_mass_horizon {
            mass.clamp(self.config.min_mass_horizon, 1e6)
        } else {
            self.config.min_mass_horizon
        };

        let a = if angular_momentum_spin.is_finite() {
            angular_momentum_spin.clamp(-m * 2.0, m * 2.0)
        } else {
            0.0
        };

        let q = if charge.is_finite() {
            charge.clamp(-m * 2.0, m * 2.0)
        } else {
            0.0
        };

        let th = if theta.is_finite() {
            theta.clamp(0.0, PI)
        } else {
            PI * 0.50
        };

        // 2. Discriminante del horizonte: \Delta = r^2 - 2Mr + a^2 + Q^2 = 0
        // Solución: r_\pm = M \pm \sqrt{M^2 - a^2 - Q^2}
        let spin_sq = a * a;
        let charge_sq = q * q;
        let mass_sq = m * m;
        let horizon_radicand = mass_sq - spin_sq - charge_sq;

        // Censura cósmica de Penrose: el radicando debe ser >= 0 con margen de seguridad
        let cosmic_threshold = (spin_sq + charge_sq) * self.config.cosmic_censorship_margin;
        let naked_singularity_veto = mass_sq < cosmic_threshold || horizon_radicand < 0.0;

        let sqrt_disc = if horizon_radicand > 0.0 {
            horizon_radicand.sqrt()
        } else {
            0.0
        };

        let event_horizon_r_plus = m + sqrt_disc;
        let inner_cauchy_horizon_r_minus = (m - sqrt_disc).max(0.0);

        // 3. Límite estático de la ergosfera: g_{tt} = 0 => r_E(\theta) = M + \sqrt{M^2 - a^2 \cos^2\theta - Q^2}
        let cos_th = th.cos();
        let ergo_radicand = (mass_sq - spin_sq * cos_th * cos_th - charge_sq).max(0.0);
        let ergosphere_radius_equator = m + ergo_radicand.sqrt();

        // 4. Área del horizonte y Entropía de Bekenstein-Hawking: S_{BH} = \pi(r_+^2 + a^2)
        let r_plus_sq = event_horizon_r_plus * event_horizon_r_plus;
        let horizon_area = 4.0 * PI * (r_plus_sq + spin_sq);
        let bekenstein_hawking_entropy = horizon_area * 0.25;

        // 5. Temperatura de Hawking: T_H = \frac{r_+ - M}{2\pi(r_+^2 + a^2)}
        let denom_temp = 2.0 * PI * (r_plus_sq + spin_sq);
        let hawking_temperature = if denom_temp > 1e-12 {
            ((event_horizon_r_plus - m) / denom_temp).clamp(0.0, 10.0)
        } else {
            0.0
        };

        let hawking_thermal_evaporation_veto =
            hawking_temperature > self.config.max_hawking_temp || !hawking_temperature.is_finite();

        // 6. Masa irreducible de Christodoulou y Eficiencia de Penrose
        // M_{\text{irr}} = \sqrt{\frac{A}{16\pi}} = \sqrt{\frac{r_+^2 + a^2}{4}}
        let irreducible_mass = ((r_plus_sq + spin_sq) * 0.25).sqrt();
        let penrose_efficiency = if m > 1e-12 {
            ((1.0 - irreducible_mass / m) * self.config.ergosphere_drag_scale).clamp(0.0, 0.50)
        } else {
            0.0
        };

        // 7. Factor continuo de escala de slot
        let slot_scaling_factor = if naked_singularity_veto || hawking_thermal_evaporation_veto {
            0.10
        } else {
            let stability_boost =
                (1.0 - hawking_temperature / self.config.max_hawking_temp).clamp(0.0, 1.0);
            (0.30 + 0.70 * stability_boost + penrose_efficiency * 0.50).clamp(0.10, 1.25)
        };

        KerrNewmanReport {
            event_horizon_r_plus,
            inner_cauchy_horizon_r_minus,
            ergosphere_radius_equator,
            hawking_temperature,
            bekenstein_hawking_entropy,
            irreducible_mass,
            penrose_efficiency,
            slot_scaling_factor,
            naked_singularity_veto,
            hawking_thermal_evaporation_veto,
        }
    }
}

impl Default for KerrNewmanBlackHoleEngine {
    fn default() -> Self {
        Self::new(KerrNewmanConfig::default())
    }
}
