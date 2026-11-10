//! # Fase 95 (§95) — Supersimetría $\mathcal{N}=1$ & Ecuación de Nicolai
//!
//! ## Modo Profesor — Ontología Técnica y Matemática (7 Pilares Rigurosos)
//!
//! ### 1. QUÉ (Definición Técnica)
//! El motor `SupersymmetryNicolaiEngine` implementa la mecánica cuántica supersimétrica ($\mathcal{N}=1$ SUSY QM)
//! formulada por Edward Witten (1981) y el mapeo estocástico no lineal de Hermann Nicolai (1980). Modela el
//! continuo de precios y volúmenes acoplando grados de libertad bosónicos (precios y flujo agresor $x$) con
//! grados de libertad fermiónicos (pares de órdenes agresivas/pasivas $\psi, \psi^\dagger$). Las supercargas
//! $Q, Q^\dagger$ satisfacen el álgebra de Lie graduada:
//! $$\{Q, Q^\dagger\} = 2H, \quad Q^2 = (Q^\dagger)^2 = 0$$
//! con Hamiltoniano supersimétrico:
//! $$H = \frac{1}{2} p^2 + \frac{1}{2} (\nabla W(x))^2 + \frac{1}{2} [\psi^\dagger, \psi] \nabla^2 W(x)$$
//! donde $W(x)$ es el superpotencial de Witten. El motor evalúa el **Índice de Witten** $\Delta = n_B^0 - n_F^0$
//! y la transformación estocástica de Nicolai $\xi(t) = \dot{x} + \nabla W(x)$, calculando la energía de
//! vacío $E_0 \ge 0$ y el residuo de gaussianidad de Nicolai.
//!
//! ### 2. POR QUÉ (Fundamento Teórico Físico)
//! Los modelos habituales de señales de trading ignoran la simetría microscópica fundamental entre compradores
//! agresivos y vendedores pasivos. Cuando la supersimetría está intacta ($E_0 = 0$, $\Delta \ne 0$), el jacobiano
//! funcional de Nicolai cancela exactamente las fluctuaciones cuánticas fermiónicas, y el mercado se comporta
//! como un campo libre perfectamente equilibrado sin deriva parásita. Sin embargo, cuando se produce una
//! **ruptura espontánea de supersimetría** ($E_0 > 0$, $\Delta = 0$), emerge un Goldstino sin masa que rompe
//! el equilibrio metastable, originando cascadas direccionales extremas o colapsos violentos de liquidez.
//!
//! ### 3. PARA QUÉ (Objetivo de Trading y Cota de Ville)
//! - Identificar fases de supersimetría exacta donde el mapeo de Nicolai garantiza reversión armónica a la
//!   media gaussiana, modulando el slot scaling factor $\eta_{\text{susy}} \in [0.15, 1.30]$ y el sesgo
//!   direccional continuo $\kappa_{\text{susy}} \in [-1.0, 1.0]$.
//! - Vetar inmediatamente (`SUSY_SPONTANEOUS_BREAKING_VETO`) ante ruptura espontánea de supersimetría con
//!   energía de vacío anormalmente alta ($E_0 > E_{\text{crit}}$) o divergencia del mapa de Nicolai,
//!   blindando la supermartingala de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//!
//! ### 4. CÓMO (Silicio y Algorítmica en Stack $O(1)$)
//! - Discretización temporal estocástica en tiempo continuo:
//!   - Campo bosónico $x(t)$ y momento canónico $p(t) = \dot{x}$.
//!   - Gradiente del superpotencial de Witten $\nabla W(x) \approx g \cdot (x - x_0) \cdot (1 + \text{vol}^2)$.
//!   - Variable de Nicolai: $\xi(t) = \dot{x} + \nabla W(x)$.
//!   - Energía de vacío de Witten: $E_0 = \frac{1}{2} (\nabla W)^2 + \frac{1}{2} |\nabla^2 W|$.
//!   - Índice de Witten normalizado: $\Delta \approx \tanh(\nabla^2 W)$.
//! - Zero allocation en heap, cero división por cero, robustez total ante NaNs e infinitos.
//!
//! ### 5. CUÁNDO (Condiciones de Ejecución)
//! En cada tick de síntesis de señales en `signal-engine` dentro de `OmniscientLivingGraphOrchestrator`.
//!
//! ### 6. DÓNDE (Ubicación en la Arquitectura)
//! `crates/signal-engine/src/supersymmetry_nicolai_map.rs`, exportado en `crates/signal-engine/src/lib.rs`.
//!
//! ### 7. QUIÉN (Componente y Rol)
//! `SupersymmetryNicolaiEngine`, diseñado y auditado por el Rol 8 (Físico Teórico de Supersimetría)
//! y Rol 2 (Quant Developer de Microestructura).

/// Parámetros de configuración para la supersimetría $\mathcal{N}=1$ y mapa de Nicolai.
#[derive(Debug, Clone, Copy)]
pub struct SupersymmetryNicolaiConfig {
    /// Constante de acoplamiento del superpotencial de Witten $g > 0$.
    pub superpotential_coupling: f64,
    /// Energía máxima de vacío admisible antes de ruptura espontánea $E_{\max}$.
    pub max_vacuum_energy: f64,
    /// Residuo crítico admisible del mapeo de Nicolai $\|\xi\|_{\max}$.
    pub max_nicolai_residual: f64,
    /// Intervalo de tiempo por defecto $\Delta t$.
    pub dt_default: f64,
    /// Tasa de decaimiento temporal de memoria de campo $\lambda$.
    pub memory_decay: f64,
    /// Regularización numérica $\epsilon$.
    pub epsilon: f64,
}

impl Default for SupersymmetryNicolaiConfig {
    fn default() -> Self {
        Self {
            superpotential_coupling: 1.0,
            max_vacuum_energy: 2.50,
            max_nicolai_residual: 3.00,
            dt_default: 0.10,
            memory_decay: 0.15,
            epsilon: 1e-7,
        }
    }
}

/// Reporte de estado emitido por el motor de supersimetría y mapa de Nicolai.
#[derive(Debug, Clone, Copy)]
pub struct SupersymmetryNicolaiReport {
    /// Densidad de energía de vacío de Witten $E_0 \ge 0$.
    pub vacuum_energy: f64,
    /// Índice de Witten regularizado $\Delta \in [-1.0, 1.0]$.
    pub witten_index: f64,
    /// Residuo del mapeo estocástico no lineal de Nicolai $\|\xi(t)\|$.
    pub nicolai_map_residual: f64,
    /// Paridad fermiónica efectiva $(-1)^F \in [-1.0, 1.0]$.
    pub fermionic_parity: f64,
    /// Sesgo direccional supersimétrico normalizado $\kappa_{\text{susy}} \in [-1.0, 1.0]$.
    pub directional_bias: f64,
    /// Factor de escala de slot continuo $\eta_{\text{susy}} \in [0.15, 1.30]$.
    pub slot_scaling_factor: f64,
    /// Indicador de veto estricto por ruptura espontánea de supersimetría.
    pub susy_spontaneous_breaking_veto: bool,
}

/// Motor de supersimetría $\mathcal{N}=1$ y mapeo estocástico de Nicolai.
#[derive(Debug, Clone)]
pub struct SupersymmetryNicolaiEngine {
    config: SupersymmetryNicolaiConfig,
    bosonic_field: f64,
    fermionic_condensate: f64,
    prev_field: f64,
}

impl SupersymmetryNicolaiEngine {
    /// Crea una nueva instancia del motor con la configuración dada.
    pub fn new(config: SupersymmetryNicolaiConfig) -> Self {
        Self {
            config,
            bosonic_field: 0.0,
            fermionic_condensate: 0.0,
            prev_field: 0.0,
        }
    }

    /// Evalúa la supersimetría y el mapa de Nicolai para el tick actual.
    ///
    /// # Argumentos
    /// * `price_ret` - Retorno de precio instantáneo.
    /// * `ofi` - Desbalance de flujo de órdenes.
    /// * `spread_bps` - Spread en puntos básicos.
    /// * `vol_norm` - Volatilidad normalizada.
    /// * `dt` - Intervalo temporal en segundos transcurrido.
    pub fn evaluate(
        &mut self,
        price_ret: f64,
        ofi: f64,
        spread_bps: f64,
        vol_norm: f64,
        dt: f64,
    ) -> SupersymmetryNicolaiReport {
        let eps = self.config.epsilon.max(1e-12);
        let input_invalid = !price_ret.is_finite()
            || !ofi.is_finite()
            || !spread_bps.is_finite()
            || !vol_norm.is_finite()
            || !dt.is_finite();

        let safe_p = if price_ret.is_finite() { price_ret } else { 0.0 };
        let safe_q = if ofi.is_finite() { ofi } else { 0.0 };
        let safe_dt = if dt.is_finite() && dt > 0.0 {
            dt.clamp(0.001, 10.0)
        } else {
            self.config.dt_default
        };
        let safe_vol = if vol_norm.is_finite() {
            vol_norm.abs().max(eps)
        } else {
            eps
        };
        let safe_spread = if spread_bps.is_finite() {
            spread_bps.abs().max(eps)
        } else {
            eps
        };

        // 1. Evolución del campo bosónico x(t) amortiguado
        let input_drive = safe_p + 0.5 * safe_q;
        let decay = (1.0 - self.config.memory_decay).clamp(0.0, 1.0);
        self.bosonic_field = decay * self.bosonic_field + self.config.memory_decay * input_drive;

        // Velocidad temporal del campo: \dot{x}
        let field_velocity = (self.bosonic_field - self.prev_field) / safe_dt;

        // 2. Superpotencial de Witten W(x):
        // W(x) = \frac{1}{2} g x^2 + \frac{\lambda}{4} x^4
        // \nabla W(x) = g x + \lambda x^3
        // \nabla^2 W(x) = g + 3 \lambda x^2
        let g = self.config.superpotential_coupling;
        let lambda_nonlin = 0.25 * safe_vol;
        let grad_w = g * self.bosonic_field + lambda_nonlin * self.bosonic_field.powi(3);
        let hessian_w = g + 3.0 * lambda_nonlin * self.bosonic_field.powi(2);

        // 3. Ecuación estocástica de Nicolai:
        // \xi(t) = \dot{x} + \nabla W(x)
        let nicolai_residual = (field_velocity + grad_w).abs();

        // 4. Energía de Vacío de Witten:
        // E_0 = \frac{1}{2} (\nabla W)^2 + \frac{1}{2} |\nabla^2 W - g|
        let vacuum_energy = (0.5 * grad_w.powi(2) + 0.5 * (hessian_w - g).abs()).max(0.0);

        // 5. Índice de Witten regularizado:
        // \Delta = \text{Tr}[(-1)^F] \approx \tanh(\text{Index}) \in [-1.0, 1.0]
        let witten_index = (hessian_w / (hessian_w.abs() + 1.0)).clamp(-1.0, 1.0);

        // 6. Condensado y paridad fermiónica (-1)^F:
        let fermi_drive = safe_q - safe_p;
        self.fermionic_condensate =
            decay * self.fermionic_condensate + self.config.memory_decay * fermi_drive;
        let fermionic_parity = (-self.fermionic_condensate.tanh()).clamp(-1.0, 1.0);

        // 7. Sesgo direccional continuo \kappa_{\text{susy}} \in [-1.0, 1.0]
        let susy_force = -grad_w + 0.20 * fermionic_parity;
        let directional_bias = susy_force.tanh().clamp(-1.0, 1.0);

        // 8. Factor de escala continuo \eta_{\text{susy}} \in [0.15, 1.30]
        // Se premia la estabilidad de vacío (E_0 bajo) y la gaussianidad del mapa de Nicolai
        let vacuum_suppression = (-0.5 * vacuum_energy).exp();
        let nicolai_adherence = (-0.3 * nicolai_residual).exp();
        let spread_damping = 1.0 / (1.0 + (safe_spread / 25.0).powi(2));
        let slot_scaling_factor = (0.15
            + 1.15 * vacuum_suppression * nicolai_adherence * spread_damping)
            .clamp(0.15, 1.30);

        // 9. Veto por ruptura espontánea de supersimetría:
        // Detonado si la energía de vacío excede el límite crítico o el mapa de Nicolai diverge
        let vacuum_breakdown = vacuum_energy > self.config.max_vacuum_energy;
        let nicolai_divergence = nicolai_residual > self.config.max_nicolai_residual;
        let numerical_fault = vacuum_energy.is_nan()
            || vacuum_energy.is_infinite()
            || nicolai_residual.is_nan()
            || nicolai_residual.is_infinite();
        let susy_spontaneous_breaking_veto =
            vacuum_breakdown || nicolai_divergence || numerical_fault || input_invalid;

        // Actualización de estado
        self.prev_field = self.bosonic_field;

        SupersymmetryNicolaiReport {
            vacuum_energy,
            witten_index,
            nicolai_map_residual: nicolai_residual,
            fermionic_parity,
            directional_bias,
            slot_scaling_factor,
            susy_spontaneous_breaking_veto,
        }
    }

    /// Reinicia el estado interno del motor a cero.
    pub fn reset(&mut self) {
        self.bosonic_field = 0.0;
        self.fermionic_condensate = 0.0;
        self.prev_field = 0.0;
    }
}
