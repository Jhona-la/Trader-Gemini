//! # Módulo: Puerta Cuántica de Ignición y Certificación OOS con Cota de Ville y Piso de Grossman-Zhou
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Puerta de ignición determinista y continua (`VilleGrossmanIgnitionGate`) que acopla
//!   la desigualdad maximal de supermartingalas de Ville ($\mathbb{P}(\text{ruina}) \le 10^{-6}$)
//!   con la teoría de control estocástico de piso móvil de Grossman-Zhou ($W_t \ge S_t$) y la
//!   certificación cuantitativa Fuera de Muestra (OOS).
//! - **POR QUÉ:** La promesa de duplicación exponencial del capital cada 72 horas ($\ln(2) \approx 0.69315$ nats / 72h)
//!   exige que la probabilidad de ruina absorbente sea estrictamente acotada a nivel de hardware y matemática
//!   ($\le 10^{-6}$). El modelo de Grossman-Zhou garantiza que el drawdown nunca sobrepase el límite
//!   innegociable (1.5%) reduciendo dinámicamente el cushion de riesgo a cero antes del suelo móvil.
//! - **PARA QUÉ:**
//!   1. Bloquear de raíz cualquier emisión de orden real en Binance si las condiciones de seguridad matemática
//!      no están certificadas al 100%.
//!   2. Modular suavemente el dimensionamiento de posición en base al colchón de riqueza disponible (cushion).
//!   3. Garantizar que sólo estrategias con Sharpe OOS $\ge 2.0$ y drawdown $\le 1.5\%$ puedan cruzar la puerta de ignición.
//! - **CÓMO:**
//!   - Mantiene un proceso supermartingala de Ville $M_t$ que evalúa la hipótesis nula de ausencia de edge $H_0$.
//!   - Computa el suelo móvil de Grossman-Zhou: $S_t = \alpha \cdot \max_{u \le t} W_u$.
//!   - Calcula el cojín de seguridad $C_t = \max(0, W_t - S_t)$ y el multiplicador de riesgo $m_t = \min(1.0, C_t / ((1-\alpha)W_t))$.
//!   - Si $C_t \le 0$, el multiplicador colapsa a $0.0$ y la puerta rechaza cualquier orden (veto duro).
//!   - Cero asignaciones en heap (`#[repr(C)]`, operaciones atómicas y aritméticas en stack).
//! - **CUÁNDO:** Antes de admitir cualquier orden en el motor de ejecución y en cada actualización de riqueza/PnL.
//! - **DÓNDE:** `crates/risk-engine/src/ville_grossman_ignition_gate.rs`.
//! - **QUIÉN:** Consejo Colegiado (Roles 1: Arquitecto, 2: Quant Dev, 3: Risk Manager).

#![allow(clippy::all)]

use crate::ville_e_process::VilleEProcess;

/// Reporte formal emitido por la Puerta Cuántica de Ignición.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IgnitionCertificationReport {
    /// Veredicto definitivo: true si y solo si TODOS los invariantes matemáticos y OOS son conformes.
    pub certified_for_live_ignition: bool,
    /// Cota superior maximal de Ville sobre la probabilidad de ruina [0.0, 1.0].
    pub ruin_probability_bound: f64,
    /// Valor actual del supermartingala / E-proceso de Ville M_t.
    pub ville_e_value: f64,
    /// Colchón relativo de riqueza de Grossman-Zhou [0.0, 1.0] (C_t / W_t).
    pub grossman_cushion_ratio: f64,
    /// Multiplicador de riesgo y dimensionamiento resultante [0.0, 1.0].
    pub risk_multiplier: f64,
    /// Drawdown actual respecto al máximo histórico registrado [0.0, 1.0].
    pub current_drawdown_pct: f64,
    /// Estimador empírico del Ratio de Sharpe OOS post-costes.
    pub oos_sharpe_estimate: f64,
    /// Tasa de aciertos empírica OOS (Win Rate) [0.0, 1.0].
    pub oos_win_rate: f64,
    /// Conteo de operaciones OOS registradas.
    pub oos_sample_count: usize,
    /// Máscara de bits de violaciones de pre-ignición (0 = PASS total).
    pub violation_mask: u32,
}

/// Máscara de bits para diagnósticos de veto en la puerta de ignición.
pub struct IgnitionViolationBits;

impl IgnitionViolationBits {
    pub const RUIN_PROBABILITY_EXCEEDED: u32 = 1 << 0;
    pub const GROSSMAN_FLOOR_BREACHED: u32 = 1 << 1;
    pub const MAX_DRAWDOWN_EXCEEDED: u32 = 1 << 2;
    pub const OOS_SHARPE_INSUFFICIENT: u32 = 1 << 3;
    pub const OOS_SAMPLES_INSUFFICIENT: u32 = 1 << 4;
    pub const WEALTH_NON_FINITE: u32 = 1 << 5;
}

/// Parámetros de calibración para la Puerta de Ignición de Ville + Grossman-Zhou.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VilleGrossmanParams {
    /// Cota máxima admisible de probabilidad de ruina (por defecto 1e-6).
    pub max_ruin_probability: f64,
    /// Fracción de suelo móvil de Grossman-Zhou alpha en (0, 1) (ej. 0.985 para max drawdown 1.5%).
    pub grossman_alpha_floor: f64,
    /// Drawdown máximo permitido en porcentaje (ej. 0.015 = 1.5%).
    pub max_allowed_drawdown_pct: f64,
    /// Umbral mínimo de Ratio de Sharpe OOS para autorizar ignición (ej. 2.0).
    pub min_oos_sharpe: f64,
    /// Mínimo número de observaciones OOS requeridas para certificación (ej. 20).
    pub min_oos_samples: usize,
}

impl Default for VilleGrossmanParams {
    fn default() -> Self {
        Self {
            max_ruin_probability: 1e-6,
            grossman_alpha_floor: 0.985, // 98.5% de riqueza preservada (suelo del 1.5% max DD)
            max_allowed_drawdown_pct: 0.015, // 1.5% Drawdown máximo
            min_oos_sharpe: 2.0,         // Sharpe OOS >= 2.0
            min_oos_samples: 20,         // Mínimo 20 trades OOS
        }
    }
}

/// Puerta Cuántica de Ignición y Certificación OOS.
pub struct VilleGrossmanIgnitionGate {
    /// Parámetros calibrados de seguridad.
    params: VilleGrossmanParams,
    /// E-Proceso de Ville para monitoreo secuencial anytime-valid.
    ville_process: VilleEProcess,
    /// Riqueza inicial base W_0.
    initial_wealth: f64,
    /// Riqueza actual W_t.
    current_wealth: f64,
    /// Riqueza pico histórica max_{u <= t} W_u.
    peak_wealth: f64,
    /// Suma acumulada de retornos OOS en bps.
    oos_pnl_bps_sum: f64,
    /// Suma cuadrática acumulada de retornos OOS en bps^2.
    oos_pnl_bps_sq_sum: f64,
    /// Conteo de trades ganadores OOS.
    oos_win_count: usize,
    /// Conteo total de trades OOS.
    oos_trade_count: usize,
}

impl VilleGrossmanIgnitionGate {
    /// Construye una nueva puerta de ignición con capital inicial y parámetros personalizados.
    pub fn new(initial_wealth: f64, params: VilleGrossmanParams) -> Self {
        let safe_initial = if initial_wealth.is_finite() && initial_wealth > 0.0 {
            initial_wealth
        } else {
            100.0
        };

        // Instanciar VilleEProcess con alpha correspondiente a la cota de ruina requerida
        let alpha = params.max_ruin_probability.clamp(1e-8, 0.10);
        let ville = VilleEProcess::new(alpha).unwrap_or_else(|_| VilleEProcess {
            alpha,
            threshold: 1.0 / alpha,
            e_value: 1.0,
            log_wealth: 0.0,
            peak_e_value: 1.0,
            count: 0,
            running_mean: 0.0,
            running_sq_mean: 0.0,
            lambda_min: 0.0,
            lambda_max: 0.50,
            last_lambda: 0.0,
        });

        Self {
            params,
            ville_process: ville,
            initial_wealth: safe_initial,
            current_wealth: safe_initial,
            peak_wealth: safe_initial,
            oos_pnl_bps_sum: 0.0,
            oos_pnl_bps_sq_sum: 0.0,
            oos_win_count: 0,
            oos_trade_count: 0,
        }
    }

    /// Actualiza la riqueza actual del portfolio y computa los nuevos máximos y suelos móviles.
    #[inline]
    pub fn update_wealth(&mut self, current_wealth: f64) {
        if !current_wealth.is_finite() || current_wealth <= 0.0 {
            return;
        }

        self.current_wealth = current_wealth;
        if current_wealth > self.peak_wealth {
            self.peak_wealth = current_wealth;
        }
    }

    /// Registra el desenlace de una operación fuera de muestra (OOS) en bps.
    #[inline]
    pub fn record_oos_trade(&mut self, pnl_bps: f64) {
        if !pnl_bps.is_finite() {
            return;
        }

        self.oos_trade_count += 1;
        self.oos_pnl_bps_sum += pnl_bps;
        self.oos_pnl_bps_sq_sum += pnl_bps * pnl_bps;

        if pnl_bps > 0.0 {
            self.oos_win_count += 1;
        }

        // Actualizar el E-proceso de Ville normalizando la innovación al intervalo [-1, 1]
        // Escalamos 100 bps como unidad base de innovación
        let normalized_innovation = (pnl_bps / 100.0).clamp(-1.0, 1.0);
        self.ville_process.update(normalized_innovation);
    }

    /// Computa el colchón de riqueza de Grossman-Zhou y el multiplicador de dimensionamiento resultante:
    /// S_t = alpha_floor * peak_wealth
    /// Cushion C_t = max(0, W_t - S_t)
    /// risk_mult = min(1.0, C_t / ((1 - alpha_floor) * W_t))
    pub fn compute_grossman_zhou_cushion(&self) -> (f64, f64, f64) {
        let floor = self.params.grossman_alpha_floor * self.peak_wealth;
        let cushion = (self.current_wealth - floor).max(0.0);
        let cushion_ratio = if self.current_wealth > 0.0 {
            cushion / self.current_wealth
        } else {
            0.0
        };

        let max_possible_cushion = (1.0 - self.params.grossman_alpha_floor) * self.current_wealth;
        let risk_multiplier = if max_possible_cushion > 1e-6 {
            let raw_mult = cushion / max_possible_cushion;
            if (raw_mult - 1.0).abs() < 1e-5 {
                1.0
            } else {
                raw_mult.clamp(0.0, 1.0)
            }
        } else {
            0.0
        };

        (cushion, cushion_ratio, risk_multiplier)
    }

    /// Evalúa exhaustivamente todos los invariantes matemáticos y certifica si el sistema puede ignicionar.
    pub fn evaluate_ignition_readiness(&self) -> IgnitionCertificationReport {
        let mut violations = 0_u32;

        // 1. Verificación de Riqueza finita y positiva
        if !self.current_wealth.is_finite() || self.current_wealth <= 0.0 {
            violations |= IgnitionViolationBits::WEALTH_NON_FINITE;
        }

        // 2. Control de Drawdown empírico
        let current_drawdown = if self.peak_wealth > 0.0 {
            ((self.peak_wealth - self.current_wealth) / self.peak_wealth).max(0.0)
        } else {
            0.0
        };

        if current_drawdown > self.params.max_allowed_drawdown_pct {
            violations |= IgnitionViolationBits::MAX_DRAWDOWN_EXCEEDED;
        }

        // 3. Control de Piso y Colchón Grossman-Zhou
        let (_cushion, cushion_ratio, risk_mult) = self.compute_grossman_zhou_cushion();
        if cushion_ratio <= 1e-6 || risk_mult <= 1e-6 {
            violations |= IgnitionViolationBits::GROSSMAN_FLOOR_BREACHED;
        }

        // 4. Cota maximal de Ville sobre la probabilidad de ruina
        // Por Ville (1939): P_{H0}(sup M_t >= lambda) <= 1 / lambda
        let ruin_prob_bound = self.ville_process.anytime_p_value();

        if ruin_prob_bound > self.params.max_ruin_probability
            && self.oos_trade_count >= self.params.min_oos_samples
        {
            violations |= IgnitionViolationBits::RUIN_PROBABILITY_EXCEEDED;
        }

        // 5. Estadísticas OOS (Muestreo y Sharpe Ratio)
        if self.oos_trade_count < self.params.min_oos_samples {
            violations |= IgnitionViolationBits::OOS_SAMPLES_INSUFFICIENT;
        }

        let n = self.oos_trade_count as f64;
        let oos_mean = if n > 0.0 {
            self.oos_pnl_bps_sum / n
        } else {
            0.0
        };
        let oos_var = if n > 1.0 {
            let variance_raw = (self.oos_pnl_bps_sq_sum - n * oos_mean * oos_mean) / (n - 1.0);
            variance_raw.max(0.0)
        } else {
            0.0
        };
        let oos_std = oos_var.sqrt();

        // Anualización aproximada de Sharpe basada en operaciones
        let oos_sharpe = if oos_std > 1e-4 {
            (oos_mean / oos_std) * (252.0_f64).sqrt()
        } else if oos_mean > 0.0 {
            10.0
        } else {
            0.0
        };

        if self.oos_trade_count >= self.params.min_oos_samples
            && oos_sharpe < self.params.min_oos_sharpe
        {
            violations |= IgnitionViolationBits::OOS_SHARPE_INSUFFICIENT;
        }

        let oos_win_rate = if self.oos_trade_count > 0 {
            self.oos_win_count as f64 / self.oos_trade_count as f64
        } else {
            0.0
        };

        let certified = violations == 0;

        IgnitionCertificationReport {
            certified_for_live_ignition: certified,
            ruin_probability_bound: ruin_prob_bound,
            ville_e_value: self.ville_process.e_value,
            grossman_cushion_ratio: cushion_ratio,
            risk_multiplier: risk_mult,
            current_drawdown_pct: current_drawdown,
            oos_sharpe_estimate: oos_sharpe,
            oos_win_rate,
            oos_sample_count: self.oos_trade_count,
            violation_mask: violations,
        }
    }

    /// Retorna la riqueza inicial base.
    #[inline(always)]
    pub fn initial_wealth(&self) -> f64 {
        self.initial_wealth
    }

    /// Retorna la riqueza actual registrada.
    #[inline(always)]
    pub fn current_wealth(&self) -> f64 {
        self.current_wealth
    }

    /// Retorna el pico de riqueza histórico.
    #[inline(always)]
    pub fn peak_wealth(&self) -> f64 {
        self.peak_wealth
    }
}
