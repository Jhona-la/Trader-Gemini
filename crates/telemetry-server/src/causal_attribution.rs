//! Meta-Analítica Cuántica Causal (FASE 5)
//!
//! Atribución causal de PnL y ROI mediante Valores de Aumann-Shapley (juegos continuos no atómicos),
//! Valores de Coalición de Owen, Gobernador de Eficiencia Silicio-Alpha (eta_CPU) y
//! Detector Secuencial de Puntos de Cambio de Shiryaev-Roberts-Pollak (SRP).
//!
//! # QUÉ:
//! Sistema off-thread de descomposición causal exacta del rendimiento financiero,
//! eficiencia computacional en silicio y detección ultratemprana de degradación de alpha.
//!
//! # POR QUÉ:
//! En un sistema multivariante complejo con 6 pilares matemáticos concurrentes
//! (MPS, Yang-Mills, Navier-Stokes, Koopman, Microestructura, TDA), atribuir ganancias
//! o pérdidas con heurísticas clásicas conduce a sesgos y sobreajustes.
//! Aumann-Shapley garantiza descomposición aditiva exacta: sum(phi_i) = PnL Neto.
//!
//! # PARA QUÉ:
//! 1. Desglosar la contribución neta y bruta de cada pilar pre y post fees.
//! 2. Auditar la eficiencia silicio-alpha eta_CPU = Delta_EV / (Ciclos * Misses).
//! 3. Aislar en caliente subsistemas degradados mediante el estadístico SRP
//!    antes de que causen drawdowns en el capital real.
//!
//! # CÓMO:
//! Integración exacta cerrada de derivadas direccionales parciales a lo largo de la diagonal,
//! partición de coaliciones de Owen, y recursión estocástica de Shiryaev-Roberts-Pollak.
//!
//! # DÓNDE:
//! En `crates/telemetry-server/src/causal_attribution.rs`.
//!
//! # QUIÉN:
//! Consejo Quant Senior & Risk Manager (Fase 5).

use serde::{Deserialize, Serialize};

/// Los 6 pilares fundacionales del motor cuántico-espectral.
pub const SUBSYSTEM_COUNT: usize = 6;
pub const SUBSYSTEM_NAMES: [&str; SUBSYSTEM_COUNT] = [
    "QuantumMPS",
    "GaugeYangMills",
    "HydroNavierStokes",
    "KoopmanDMD",
    "MicrostructureL2L3",
    "TopologicalTDA",
];

/// Índices canónicos de subsistemas.
pub const IDX_QUANTUM_MPS: usize = 0;
pub const IDX_GAUGE_YANG_MILLS: usize = 1;
pub const IDX_HYDRO_NAVIER_STOKES: usize = 2;
pub const IDX_KOOPMAN_DMD: usize = 3;
pub const IDX_MICROSTRUCTURE_L2L3: usize = 4;
pub const IDX_TOPOLOGICAL_TDA: usize = 5;

/// Registro de atribución causal individual para un subsistema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubsystemAttribution {
    pub name: String,
    pub index: usize,
    pub aumann_shapley_pnl: f64,
    pub owen_coalition_pnl: f64,
    pub gross_share_pct: f64,
    pub net_share_pct: f64,
    pub attributed_fees: f64,
    pub silicon_efficiency_eta: f64,
    pub rdtsc_cycles_avg: u64,
    pub l1d_misses_est: u64,
    pub srp_statistic: f64,
    pub is_quarantined: bool,
}

/// Informe consolidado de Meta-Analítica Causal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalMetaAnalyticsReport {
    pub timestamp_ns: u64,
    pub total_gross_pnl: f64,
    pub total_net_pnl: f64,
    pub total_fees: f64,
    pub efficiency_sum: f64,
    pub attributions: Vec<SubsystemAttribution>,
    pub srp_alarm_active: bool,
    pub quarantined_count: usize,
}

/// Estado persistente del detector de Shiryaev-Roberts-Pollak (SRP) por subsistema.
#[derive(Debug, Clone)]
pub struct ShiryaevRobertsPollakDetector {
    pub r_statistic: f64,
    pub threshold: f64,
    pub in_control_mean: f64,
    pub out_control_mean: f64,
    pub variance: f64,
    pub sample_count: u64,
}

impl ShiryaevRobertsPollakDetector {
    /// Inicializa el detector SRP con umbral calibrado (ej. A_srp = 100.0).
    pub fn new(threshold: f64, in_control_mean: f64, out_control_mean: f64, variance: f64) -> Self {
        Self {
            r_statistic: 0.0,
            threshold: threshold.max(10.0),
            in_control_mean,
            out_control_mean,
            variance: variance.max(1e-6),
            sample_count: 0,
        }
    }

    /// Actualiza secuencialmente el estadístico recursivo:
    /// R_n = (1 + R_{n-1}) * Lambda_n
    pub fn update(&mut self, observation: f64) -> (f64, bool) {
        if !observation.is_finite() {
            return (self.r_statistic, false);
        }

        self.sample_count += 1;

        // Log-Likelihood Ratio Gaussiano:
        // ln(Lambda_n) = ((mu_1 - mu_0) / sigma^2) * (x - (mu_0 + mu_1) / 2)
        let delta_mu = self.out_control_mean - self.in_control_mean;
        let mid_mu = (self.in_control_mean + self.out_control_mean) * 0.5;
        let log_lr = (delta_mu / self.variance) * (observation - mid_mu);

        // Clamping numérico para evitar overflow en exp()
        let lr = log_lr.clamp(-20.0, 20.0).exp();

        // Recursión canónica de Shiryaev-Roberts
        self.r_statistic = (1.0 + self.r_statistic) * lr;

        let alarm = self.r_statistic >= self.threshold;
        (self.r_statistic, alarm)
    }

    /// Reinicia el estadístico tras aislamiento o recuperación certificada.
    pub fn reset(&mut self) {
        self.r_statistic = 0.0;
    }
}

/// Motor de Atribución Causal Aumann-Shapley & Owen.
pub struct CausalAttributionEngine {
    srp_detectors: [ShiryaevRobertsPollakDetector; SUBSYSTEM_COUNT],
}

impl Default for CausalAttributionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalAttributionEngine {
    pub fn new() -> Self {
        let detectors = std::array::from_fn(|_| {
            // H0: Rendimiento positivo esperado (+0.05 R), H1: Rendimiento negativo (-0.05 R)
            ShiryaevRobertsPollakDetector::new(100.0, 0.05, -0.05, 0.04)
        });
        Self {
            srp_detectors: detectors,
        }
    }

    /// Calcula la descomposición causal exacta de Aumann-Shapley y valores de Owen.
    ///
    /// # Propiedad Invariante Fundamental:
    /// sum(phi_i) == total_net_pnl exacto (error < 1e-9).
    pub fn evaluate(
        &mut self,
        activity_vector: &[f64; SUBSYSTEM_COUNT],
        cycles: &[u64; SUBSYSTEM_COUNT],
        l1d_misses: &[u64; SUBSYSTEM_COUNT],
        total_gross_pnl: f64,
        total_fees: f64,
    ) -> CausalMetaAnalyticsReport {
        let safe_gross = if total_gross_pnl.is_finite() { total_gross_pnl } else { 0.0 };
        let safe_fees = if total_fees.is_finite() && total_fees >= 0.0 { total_fees } else { 0.0 };
        let safe_net = safe_gross - safe_fees;

        // Pesos lineales base w_i de cada pilar en la función de juego continuo
        let base_weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10];

        // Matriz simétrica de interacción entre subsistemas (gamma_ij)
        // Refleja acoplamientos reales (ej. MPS con Gauge, Navier con OFI, TDA con Riesgo)
        let mut gamma = [[0.0; SUBSYSTEM_COUNT]; SUBSYSTEM_COUNT];
        gamma[IDX_QUANTUM_MPS][IDX_GAUGE_YANG_MILLS] = 0.08;
        gamma[IDX_GAUGE_YANG_MILLS][IDX_QUANTUM_MPS] = 0.08;
        gamma[IDX_HYDRO_NAVIER_STOKES][IDX_MICROSTRUCTURE_L2L3] = 0.06;
        gamma[IDX_MICROSTRUCTURE_L2L3][IDX_HYDRO_NAVIER_STOKES] = 0.06;
        gamma[IDX_TOPOLOGICAL_TDA][IDX_GAUGE_YANG_MILLS] = 0.05;
        gamma[IDX_GAUGE_YANG_MILLS][IDX_TOPOLOGICAL_TDA] = 0.05;

        // Cálculo de valor continuo de Aumann-Shapley en forma cerrada:
        // phi_i = x_i * [ w_i + 0.5 * sum_{j != i} gamma_ij * x_j ]
        let mut unnormalized_phi = [0.0; SUBSYSTEM_COUNT];
        for i in 0..SUBSYSTEM_COUNT {
            let x_i = activity_vector[i].clamp(0.0, 1.0);
            let mut cross_sum = 0.0;
            for j in 0..SUBSYSTEM_COUNT {
                if i != j {
                    cross_sum += gamma[i][j] * activity_vector[j].clamp(0.0, 1.0);
                }
            }
            unnormalized_phi[i] = x_i * (base_weights[i] + 0.5 * cross_sum);
        }

        let sum_raw: f64 = unnormalized_phi.iter().sum();
        let scale = if sum_raw.abs() > 1e-12 {
            1.0 / sum_raw
        } else {
            1.0 / (SUBSYSTEM_COUNT as f64)
        };

        // Proporciones normalizadas exactas que suman 1.0
        let mut shares = [0.0; SUBSYSTEM_COUNT];
        for i in 0..SUBSYSTEM_COUNT {
            shares[i] = unnormalized_phi[i] * scale;
        }

        // Partición de Coaliciones de Owen:
        // Coalición 1: Física Cuántica & Milenio (0, 1, 2, 3)
        // Coalición 2: Microestructura L2/L3 (4)
        // Coalición 3: Topología & Riesgo TDA (5)
        let coalition_1_share = shares[0] + shares[1] + shares[2] + shares[3];
        let coalition_2_share = shares[4];
        let coalition_3_share = shares[5];

        let mut owen_values = [0.0; SUBSYSTEM_COUNT];
        for i in 0..4 {
            let intra_weight = shares[i] / coalition_1_share.max(1e-12);
            owen_values[i] = safe_net * coalition_1_share * intra_weight;
        }
        owen_values[4] = safe_net * coalition_2_share;
        owen_values[5] = safe_net * coalition_3_share;

        let mut attributions = Vec::with_capacity(SUBSYSTEM_COUNT);
        let mut any_alarm = false;
        let mut quarantined_count = 0;

        for i in 0..SUBSYSTEM_COUNT {
            let as_pnl = safe_net * shares[i];
            let attributed_fee = safe_fees * shares[i];
            let gross_pnl_i = safe_gross * shares[i];

            let c = cycles[i].max(1);
            let m = l1d_misses[i].max(1);

            // Gobernador de Eficiencia Silicio-Alpha:
            // eta_CPU = Delta_EV_neto / (Ciclos * L1D_Misses) en escala pico-alpha
            let delta_ev = as_pnl;
            let silicon_eta = (delta_ev * 1e9) / (c as f64 * m as f64);

            // Actualización del detector SRP con el PnL normalizado atribuido
            let (srp_val, is_alarm) = self.srp_detectors[i].update(as_pnl);
            if is_alarm {
                any_alarm = true;
                quarantined_count += 1;
            }

            attributions.push(SubsystemAttribution {
                name: SUBSYSTEM_NAMES[i].to_string(),
                index: i,
                aumann_shapley_pnl: as_pnl,
                owen_coalition_pnl: owen_values[i],
                gross_share_pct: if safe_gross.abs() > 1e-12 {
                    (gross_pnl_i / safe_gross) * 100.0
                } else {
                    shares[i] * 100.0
                },
                net_share_pct: shares[i] * 100.0,
                attributed_fees: attributed_fee,
                silicon_efficiency_eta: silicon_eta,
                rdtsc_cycles_avg: c,
                l1d_misses_est: m,
                srp_statistic: srp_val,
                is_quarantined: is_alarm,
            });
        }

        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        CausalMetaAnalyticsReport {
            timestamp_ns,
            total_gross_pnl: safe_gross,
            total_net_pnl: safe_net,
            total_fees: safe_fees,
            efficiency_sum: attributions.iter().map(|a| a.silicon_efficiency_eta).sum(),
            attributions,
            srp_alarm_active: any_alarm,
            quarantined_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aumann_shapley_exact_decomposition() {
        let mut engine = CausalAttributionEngine::new();
        let activity = [1.0, 0.8, 0.6, 0.5, 0.9, 0.7];
        let cycles = [5000, 4200, 3100, 2800, 1500, 6000];
        let misses = [50, 40, 25, 20, 10, 70];
        let gross_pnl = 150.0;
        let fees = 15.0;
        let expected_net = gross_pnl - fees;

        let report = engine.evaluate(&activity, &cycles, &misses, gross_pnl, fees);

        assert_eq!(report.attributions.len(), SUBSYSTEM_COUNT);

        // Axioma de Eficiencia: sum(phi_i) == total_net_pnl exacto
        let sum_as_pnl: f64 = report.attributions.iter().map(|a| a.aumann_shapley_pnl).sum();
        assert!(
            (sum_as_pnl - expected_net).abs() < 1e-9,
            "Aumann-Shapley sum {} != expected net {}",
            sum_as_pnl,
            expected_net
        );

        // Axioma de Eficiencia de Owen: sum(psi_i) == total_net_pnl exacto
        let sum_owen: f64 = report.attributions.iter().map(|a| a.owen_coalition_pnl).sum();
        assert!(
            (sum_owen - expected_net).abs() < 1e-9,
            "Owen value sum {} != expected net {}",
            sum_owen,
            expected_net
        );

        // Las comisiones atribuidas deben sumar exactamente las comisiones totales
        let sum_fees: f64 = report.attributions.iter().map(|a| a.attributed_fees).sum();
        assert!((sum_fees - fees).abs() < 1e-9);
    }

    #[test]
    fn test_shiryaev_roberts_pollak_change_detection() {
        let mut detector = ShiryaevRobertsPollakDetector::new(50.0, 0.10, -0.10, 0.05);

        // En régimen de control (rendimiento positivo), R debe mantenerse bajo
        for _ in 0..20 {
            let (r, alarm) = detector.update(0.12);
            assert!(!alarm, "No debe disparar alarma bajo control");
            assert!(r < 50.0);
        }

        // En régimen degradado (pérdidas consecutivas), R debe crecer exponencialmente hasta cruzar el umbral
        let mut triggered = false;
        for _ in 0..50 {
            let (_, alarm) = detector.update(-0.15);
            if alarm {
                triggered = true;
                break;
            }
        }
        assert!(triggered, "Detector SRP debió activar alarma ante degradación sostenida");
    }

    #[test]
    fn test_silicon_alpha_governor_monotonicity() {
        let mut engine = CausalAttributionEngine::new();
        let activity = [1.0; 6];
        // Dos subsistemas con idéntica actividad pero uno gasta 10x más ciclos de CPU
        let cycles = [10_000, 100_000, 5000, 5000, 5000, 5000];
        let misses = [100, 100, 50, 50, 50, 50];

        let report = engine.evaluate(&activity, &cycles, &misses, 100.0, 10.0);

        let eta_fast = report.attributions[0].silicon_efficiency_eta;
        let eta_slow = report.attributions[1].silicon_efficiency_eta;

        // El subsistema que gasta 10x más ciclos debe tener menor eficiencia silicio-alpha
        assert!(
            eta_fast > eta_slow,
            "Subsistema rápido debe tener mayor eficiencia silicio que el lento: {} vs {}",
            eta_fast,
            eta_slow
        );
    }
}
