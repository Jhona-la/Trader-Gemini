//! # Módulo: Motor de Realimentación Causal y Aprendizaje Continuo Online sin Paradas
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Motor de aprendizaje continuo autónomo en tiempo real (`OnlineCausalFeedbackEngine`)
//!   que incorpora retroalimentación post-ejecución inmediata mediante estimación causal AIPW
//!   (Augmented Inverse Probability Weighting) doblemente robusta, cerrando la brecha entre
//!   ejecución, resultado observado y adaptación de política sin reiniciar el sistema.
//! - **POR QUÉ:** Los mercados de criptoactivos sufren regímenes no estacionarios y selección
//!   adversa variable. Si un modelo asume estáticamente que cotizar Maker es siempre superior a
//!   Taker debido al menor fee, incurrirá en pérdidas catastróficas cuando el flujo tóxico agresor
//!   barra el libro pasivo. El estimador AIPW computa el Efecto Promedio del Tratamiento (ATE) causal
//!   aislando el sesgo de selección y permitiendo al sistema transicionar fluidamente entre Maker y
//!   Taker según la evidencia empírica continua.
//! - **PARA QUÉ:**
//!   1. Adaptar dinámicamente la política de enrutamiento Maker/Taker en caliente.
//!   2. Garantizar que cada céntimo ganado o perdido refine las probabilidades operativas.
//!   3. Proteger la CPU de 16 GB laptop modulando la carga operativa ante estrangulamiento térmico.
//!   4. Sostener la tasa de crecimiento compuesto exponencial sin pausas operativas.
//! - **CÓMO:**
//!   - Mantiene una instancia interna de `AipwCausalGovernorEngine` en stack.
//!   - Procesa eventos de cierre de posición (`record_closed_trade`) extrayendo el PnL neto en bps,
//!     el propensity score $e(X)$ de la microestructura y las predicciones contrafactuales $m_1(X), m_0(X)$.
//!   - Evalúa la significancia causal con intervalos de confianza al 95%: si el límite inferior
//!     $\hat{\Gamma}_{\text{lower}} > 0$, la cotización Maker tiene ventaja causal demostrada;
//!     si $\hat{\Gamma}_{\text{upper}} < 0$, la selección adversa exige ejecución Taker inmediata.
//!   - Cero asignaciones en heap (`#[repr(C)]`, memoria plana, aritmética acotada).
//! - **CUÁNDO:** Tras el cierre de cada posición y en cada tick de decisión de orden.
//! - **DÓNDE:** `crates/telemetry-server/src/online_causal_feedback.rs`.
//! - **QUIÉN:** Consejo Colegiado (Roles 2: Quant Dev, 3: Risk Manager, 10: Auditor Forense Causal).

#![allow(clippy::all)]

use crate::aipw_causal_governor::{
    AipwCausalGovernorEngine, AipwGovernorReport, SiliconThermalState,
};
use core::sync::atomic::{AtomicU64, Ordering};

/// Decisión cuantitativa de enrutamiento emitida por el motor de realimentación causal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalRoutingDecision {
    /// Indica si se prefiere enrutamiento pasivo (Maker) o agresivo (Taker).
    pub prefer_maker: bool,
    /// Grado de confianza causal en la recomendación $[0.0, 1.0]$.
    pub confidence: f64,
    /// Efecto Promedio del Tratamiento (ATE) estimado en puntos básicos (bps).
    pub ate_bps: f64,
    /// Límite inferior del intervalo de confianza al 95% del ATE.
    pub ci_lower_95: f64,
    /// Límite superior del intervalo de confianza al 95% del ATE.
    pub ci_upper_95: f64,
    /// Estado del silicio y control térmico del hardware.
    pub thermal_state: SiliconThermalState,
    /// Concurrencia máxima recomendada de slots simultáneos [1, 3].
    pub recommended_max_slots: usize,
    /// Modificador de tamaño de posición derivado del edge causal $[0.5, 1.5]$.
    pub edge_size_multiplier: f64,
}

impl Default for CausalRoutingDecision {
    fn default() -> Self {
        Self {
            prefer_maker: true,
            confidence: 0.50,
            ate_bps: 0.0,
            ci_lower_95: 0.0,
            ci_upper_95: 0.0,
            thermal_state: SiliconThermalState::Nominal,
            recommended_max_slots: 3,
            edge_size_multiplier: 1.0,
        }
    }
}

/// Motor de Realimentación Causal y Aprendizaje Continuo Online.
pub struct OnlineCausalFeedbackEngine {
    /// Gobernador causal AIPW integrado.
    governor: AipwCausalGovernorEngine,
    /// Contador total de trades analizados causalmente.
    total_trades_analyzed: AtomicU64,
    /// Contador de adaptaciones de política hacia Maker.
    maker_promotions: AtomicU64,
    /// Contador de adaptaciones de política hacia Taker (por fuga o selección adversa).
    taker_switches: AtomicU64,
    /// Factor de atenuación de tamaño bajo estrés térmico.
    thermal_attenuation_factor: f64,
}

impl Default for OnlineCausalFeedbackEngine {
    fn default() -> Self {
        Self::new(75.0, 85.0)
    }
}

impl OnlineCausalFeedbackEngine {
    /// Construye una nueva instancia del motor de aprendizaje causal online.
    pub fn new(temp_warn_celsius: f64, temp_crit_celsius: f64) -> Self {
        Self {
            governor: AipwCausalGovernorEngine::new(temp_warn_celsius, temp_crit_celsius),
            total_trades_analyzed: AtomicU64::new(0),
            maker_promotions: AtomicU64::new(0),
            taker_switches: AtomicU64::new(0),
            thermal_attenuation_factor: 0.50,
        }
    }

    /// Registra el desenlace de un trade cerrado y actualiza la ventana causal AIPW sin interrupción.
    #[inline]
    pub fn record_closed_trade(
        &mut self,
        is_maker: bool,
        pnl_bps: f64,
        propensity_score: f64,
        expected_m1: f64,
        expected_m0: f64,
    ) {
        if !pnl_bps.is_finite() {
            return;
        }

        let treatment = if is_maker { 1 } else { 0 };
        self.governor.record_causal_event(
            treatment,
            pnl_bps,
            propensity_score,
            expected_m1,
            expected_m0,
        );

        self.total_trades_analyzed.fetch_add(1, Ordering::Relaxed);
    }

    /// Actualiza la lectura térmica del silicio de la CPU.
    #[inline]
    pub fn update_cpu_temperature(&self, temp_celsius: f64) {
        self.governor.update_temperature(temp_celsius);
    }

    /// Evalúa la política óptima de enrutamiento y dimensionamiento causal en tiempo de ejecución.
    pub fn evaluate_optimal_routing(&self, aperf: u64, mperf: u64) -> CausalRoutingDecision {
        let report: AipwGovernorReport = self.governor.analyze(aperf, mperf);

        // Si tenemos pocas muestras (< 4), mantenemos la política prudente base Maker
        if report.sample_count < 4 {
            let (max_slots, mult) = match report.thermal_state {
                SiliconThermalState::Nominal => (3, 1.0),
                SiliconThermalState::ThrottlingWarning => (2, 0.75),
                SiliconThermalState::CriticalThermalBrake => (1, self.thermal_attenuation_factor),
            };

            return CausalRoutingDecision {
                prefer_maker: true,
                confidence: 0.50,
                ate_bps: 0.0,
                ci_lower_95: 0.0,
                ci_upper_95: 0.0,
                thermal_state: report.thermal_state,
                recommended_max_slots: max_slots,
                edge_size_multiplier: mult,
            };
        }

        // Análisis de significancia causal con doble robustez:
        // Si CI 95% inferior > 0: Maker tiene edge estadísticamente comprobado
        // Si CI 95% superior < 0: Maker sufre selección adversa sistemática -> usar Taker agresivo
        let prefer_maker = if report.ci_upper_95 < 0.0 {
            self.taker_switches.fetch_add(1, Ordering::Relaxed);
            false
        } else if report.ci_lower_95 > 0.0 {
            self.maker_promotions.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            // Indeterminación estadística: default al ordenamiento pasivo Maker para ahorrar comisiones
            true
        };

        // Confianza acotada en [0.5, 0.99] en base a la distancia al cero del CI
        let se_safe = if report.standard_error > 1e-4 {
            report.standard_error
        } else {
            1.0
        };
        let t_stat = (report.average_treatment_effect / se_safe).abs();
        let confidence = (0.50 + 0.10 * t_stat).clamp(0.50, 0.99);

        // Modulación por estado térmico del procesador
        let (max_slots, thermal_mult) = match report.thermal_state {
            SiliconThermalState::Nominal => (3, 1.0),
            SiliconThermalState::ThrottlingWarning => (2, 0.75),
            SiliconThermalState::CriticalThermalBrake => (1, self.thermal_attenuation_factor),
        };

        // Multiplicador de sizing por edge causal: expande si el ATE es fuertemente positivo,
        // atenúa si el edge es débil o negativo
        let edge_mult = if prefer_maker && report.ci_lower_95 > 1.0 {
            (1.0 + 0.05 * report.average_treatment_effect.min(10.0)).clamp(1.0, 1.50)
        } else if !prefer_maker && report.ci_upper_95 < -1.0 {
            (1.0 + 0.05 * (-report.average_treatment_effect).min(10.0)).clamp(1.0, 1.30)
        } else {
            1.0
        };

        let combined_multiplier = match report.thermal_state {
            SiliconThermalState::CriticalThermalBrake => (edge_mult * thermal_mult)
                .min(self.thermal_attenuation_factor)
                .clamp(0.25, 1.50),
            _ => (edge_mult * thermal_mult).clamp(0.25, 1.50),
        };

        CausalRoutingDecision {
            prefer_maker,
            confidence,
            ate_bps: report.average_treatment_effect,
            ci_lower_95: report.ci_lower_95,
            ci_upper_95: report.ci_upper_95,
            thermal_state: report.thermal_state,
            recommended_max_slots: max_slots,
            edge_size_multiplier: combined_multiplier,
        }
    }

    /// Retorna el número acumulado de operaciones evaluadas causalmente.
    #[inline(always)]
    pub fn total_trades_analyzed(&self) -> u64 {
        self.total_trades_analyzed.load(Ordering::Relaxed)
    }

    /// Retorna el número de promociones a política Maker confirmadas.
    #[inline(always)]
    pub fn maker_promotions_count(&self) -> u64 {
        self.maker_promotions.load(Ordering::Relaxed)
    }

    /// Retorna el número de conmutaciones a Taker por selección adversa.
    #[inline(always)]
    pub fn taker_switches_count(&self) -> u64 {
        self.taker_switches.load(Ordering::Relaxed)
    }
}
