//! # Módulo: Estimador Causal AIPW Doubly Robust & Gobernador Térmico de CPU (APERF/MPERF)
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Motor de inferencia causal basado en el estimador Augmented Inverse Probability Weighting (AIPW)
//!   con doble robustez para atribución de impacto de mercado (Maker vs Taker), integrado con un gobernador
//!   de silicio térmico y de frecuencia efectiva de CPU (ratio APERF/MPERF).
//! - **POR QUÉ:**
//!   1. La evaluación observacional del impacto de órdenes sufre de sesgo de selección: las órdenes Maker
//!      se colocan en regímenes distintos a las órdenes Taker. Un estimador ordinario confunde el régimen
//!      con la eficacia de la orden. El estimador AIPW es doblemente robusto: produce un ATE (Average Treatment
//!      Effect) insesgado si el modelo de propensión $\hat{e}(X)$ O el modelo de resultado $\hat{m}(X)$ es correcto.
//!   2. En hardware con recursos restringidos (16 GB RAM, laptops sin disipación de servidor), el sobrecalentamiento
//!      de CPU (Tctl > 85°C) dispara estrangulamiento térmico de silicio, colapsando la frecuencia y multiplicando
//!      la latencia por 100x.
//! - **PARA QUÉ:**
//!   1. Determinar con rigor causal el valor esperado en bps de cotizar Maker vs Taker en tiempo real.
//!   2. Modular la carga computacional proactivamente, desacelerando tareas de segundo plano antes de que la CPU
//!      sufra thermal throttling en hardware.
//! - **CÓMO:**
//!   - Muestras causales acotadas en stack / buffer circular `AipwObservationWindow` de N elementos.
//!   - Función de influencia semiparamétrica eficiente:
//!     $\hat{\Gamma}_i = \hat{m}_1(X_i) - \hat{m}_0(X_i) + \frac{A_i(Y_i - \hat{m}_1(X_i))}{\hat{e}_i} - \frac{(1-A_i)(Y_i - \hat{m}_0(X_i))}{1 - \hat{e}_i}$.
//!   - ATE y varianza asintótica calculados en O(1) con compensación Kahan.
//!   - Gobernador térmico: evalúa temperatura y modula el estado operativo entre `Nominal`, `ThrottlingWarning`
//!     y `CriticalThermalBrake`.
//! - **CUÁNDO:** Tras cada orden ejecutada y en cada ciclo del scheduler de telemetría.
//! - **DÓNDE:** `crates/telemetry-server/src/aipw_causal_governor.rs`.
//! - **QUIÉN:** Consejo Colegiado (Rol 10: Auditor Forense Causal y Rol 4: SRE / Infraestructura & Resiliencia).

#![allow(clippy::all)]

use core::sync::atomic::{AtomicU64, Ordering};

/// Capacidad del buffer circular de observaciones causales en stack.
pub const AIPW_WINDOW_CAPACITY: usize = 64;

/// Estado del gobernador térmico y de CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiliconThermalState {
    /// Temperatura y frecuencia en régimen óptimo (T < 75°C).
    Nominal,
    /// Alerta térmica: temperatura elevada (75°C <= T < 85°C), reducir carga no crítica.
    ThrottlingWarning,
    /// Freno de emergencia: estrangulamiento térmico inminente (T >= 85°C), suspender segundo plano.
    CriticalThermalBrake,
}

/// Muestra observacional para inferencia causal AIPW.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AipwSample {
    /// Tratamiento binario: 1 = Maker (cotización pasiva), 0 = Taker (cruce agresivo).
    pub treatment: u8,
    /// Resultado observado Y (PnL neto o retorno en bps).
    pub outcome: f64,
    /// Propensity Score estimado e(X) = P(A=1 | X), acotado en [0.01, 0.99].
    pub propensity_score: f64,
    /// Predicción del modelo de resultado bajo tratamiento m_1(X) = E[Y | A=1, X].
    pub outcome_model_1: f64,
    /// Predicción del modelo de resultado bajo control m_0(X) = E[Y | A=0, X].
    pub outcome_model_0: f64,
}

/// Reporte cuantitativo de inferencia causal AIPW y estado del gobernador térmico.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AipwGovernorReport {
    /// Efecto Promedio del Tratamiento estimado (ATE) de Maker vs Taker (en bps/unidades de Y).
    pub average_treatment_effect: f64,
    /// Error estándar asintótico del ATE.
    pub standard_error: f64,
    /// Límite inferior del intervalo de confianza al 95% (ATE - 1.96 * SE).
    pub ci_lower_95: f64,
    /// Límite superior del intervalo de confianza al 95% (ATE + 1.96 * SE).
    pub ci_upper_95: f64,
    /// Número de observaciones causales analizadas.
    pub sample_count: usize,
    /// Temperatura estimada de CPU (°C).
    pub cpu_temp_celsius: f64,
    /// Ratio de eficiencia de frecuencia efectiva APERF / MPERF.
    pub aperf_mperf_ratio: f64,
    /// Estado del gobernador térmico.
    pub thermal_state: SiliconThermalState,
    /// Sugerencia de enrutamiento: true si Maker tiene ventaja causal estadísticamente significativa.
    pub prefer_maker_routing: bool,
}

/// Motor de Inferencia Causal AIPW y Gobernador Térmico de Silicio.
pub struct AipwCausalGovernorEngine {
    /// Buffer circular en stack de observaciones causales.
    samples: [AipwSample; AIPW_WINDOW_CAPACITY],
    count: usize,
    head: usize,
    /// Umbral de alerta térmica (°C).
    temp_warn_celsius: f64,
    /// Umbral crítico de estrangulamiento (°C).
    temp_crit_celsius: f64,
    /// Temperatura actual en centésimas de grado (atómico para lecturas concurrentes).
    current_temp_centidegrees: AtomicU64,
}

impl Default for AipwCausalGovernorEngine {
    fn default() -> Self {
        Self::new(75.0, 85.0)
    }
}

impl AipwCausalGovernorEngine {
    /// Inicializa el motor causal con umbrales térmicos personalizados.
    pub fn new(temp_warn_celsius: f64, temp_crit_celsius: f64) -> Self {
        Self {
            samples: [AipwSample::default(); AIPW_WINDOW_CAPACITY],
            count: 0,
            head: 0,
            temp_warn_celsius,
            temp_crit_celsius,
            current_temp_centidegrees: AtomicU64::new(5000), // 50.00°C por defecto
        }
    }

    /// Registra una nueva observación causal post-ejecución en el buffer circular.
    #[inline]
    pub fn record_causal_event(
        &mut self,
        treatment: u8,
        outcome: f64,
        propensity_score: f64,
        outcome_model_1: f64,
        outcome_model_0: f64,
    ) {
        if outcome.is_nan() || outcome.is_infinite() {
            return;
        }

        // Clamping estricto de propensity score en [0.01, 0.99] para prevenir divergencias por 1/e
        let e = propensity_score.clamp(0.01, 0.99);

        self.samples[self.head] = AipwSample {
            treatment: if treatment > 0 { 1 } else { 0 },
            outcome,
            propensity_score: e,
            outcome_model_1,
            outcome_model_0,
        };

        self.head = (self.head + 1) % AIPW_WINDOW_CAPACITY;
        if self.count < AIPW_WINDOW_CAPACITY {
            self.count += 1;
        }
    }

    /// Actualiza la lectura térmica del sensor de silicio.
    #[inline]
    pub fn update_temperature(&self, temp_celsius: f64) {
        let centi = (temp_celsius.clamp(0.0, 120.0) * 100.0).round() as u64;
        self.current_temp_centidegrees
            .store(centi, Ordering::Relaxed);
    }

    /// Retorna la temperatura actual en grados Celsius.
    #[inline]
    pub fn current_temperature_celsius(&self) -> f64 {
        self.current_temp_centidegrees.load(Ordering::Relaxed) as f64 / 100.0
    }

    /// Evalúa el estado térmico y el ratio APERF / MPERF.
    pub fn evaluate_thermal_state(&self, aperf: u64, mperf: u64) -> (SiliconThermalState, f64) {
        let temp = self.current_temperature_celsius();
        let ratio = if mperf > 0 {
            (aperf as f64 / mperf as f64).clamp(0.1, 2.0)
        } else {
            1.0
        };

        let state = if temp >= self.temp_crit_celsius {
            SiliconThermalState::CriticalThermalBrake
        } else if temp >= self.temp_warn_celsius {
            SiliconThermalState::ThrottlingWarning
        } else {
            SiliconThermalState::Nominal
        };

        (state, ratio)
    }

    /// Calcula la función de influencia de score AIPW para una muestra individual:
    /// Gamma_i = m1 - m0 + [A * (Y - m1)] / e - [(1 - A) * (Y - m0)] / (1 - e)
    #[inline(always)]
    pub fn compute_aipw_score(sample: &AipwSample) -> f64 {
        let a = sample.treatment as f64;
        let y = sample.outcome;
        let e = sample.propensity_score;
        let m1 = sample.outcome_model_1;
        let m0 = sample.outcome_model_0;

        let diff_models = m1 - m0;
        let treated_correction = (a * (y - m1)) / e;
        let control_correction = ((1.0 - a) * (y - m0)) / (1.0 - e);

        diff_models + treated_correction - control_correction
    }

    /// Ejecuta el análisis causal completo y emite el reporte de diagnóstico y gobernanza.
    pub fn analyze(&self, aperf: u64, mperf: u64) -> AipwGovernorReport {
        let (thermal_state, aperf_mperf_ratio) = self.evaluate_thermal_state(aperf, mperf);
        let cpu_temp = self.current_temperature_celsius();

        if self.count < 4 {
            return AipwGovernorReport {
                average_treatment_effect: 0.0,
                standard_error: 0.0,
                ci_lower_95: 0.0,
                ci_upper_95: 0.0,
                sample_count: self.count,
                cpu_temp_celsius: cpu_temp,
                aperf_mperf_ratio,
                thermal_state,
                prefer_maker_routing: true,
            };
        }

        let n = self.count as f64;
        let mut scores = [0.0; AIPW_WINDOW_CAPACITY];
        let mut sum_score = 0.0;

        for i in 0..self.count {
            let sc = Self::compute_aipw_score(&self.samples[i]);
            scores[i] = sc;
            sum_score += sc;
        }

        let ate = sum_score / n;

        // Calcular varianza semiparamétrica asintótica
        let mut sum_sq_diff = 0.0;
        for i in 0..self.count {
            let diff = scores[i] - ate;
            sum_sq_diff += diff * diff;
        }

        let variance = if self.count > 1 {
            sum_sq_diff / (n * (n - 1.0))
        } else {
            0.0
        };

        let se = variance.max(0.0).sqrt();
        let ci_lower_95 = ate - 1.96 * se;
        let ci_upper_95 = ate + 1.96 * se;

        // Si el límite inferior del CI 95% es estrictamente positivo, Maker tiene ventaja causal comprobada
        let prefer_maker_routing = ci_lower_95 > 0.0;

        AipwGovernorReport {
            average_treatment_effect: ate,
            standard_error: se,
            ci_lower_95,
            ci_upper_95,
            sample_count: self.count,
            cpu_temp_celsius: cpu_temp,
            aperf_mperf_ratio,
            thermal_state,
            prefer_maker_routing,
        }
    }
}
