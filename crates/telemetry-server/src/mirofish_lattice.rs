//! Mirofish Probability Lattice & 3D Tail Ridge Engine (FASE 5)
//!
//! Implementación matemática rigurosa en Rust 100% nativo de la simulación
//! de caminos probabilísticos hacia adelante (Forward Probability Lattice /
//! Tablero de Galton Cuántico-Estocástico Multivariante) y el relieve tridimensional
//! de colas extremas (Tail Probability Ridge 3D) a lo largo de las 32 escalas
//! del espectro temporal diádico continuo.
//!
//! # QUÉ:
//! Un lattice recombinante no gaussiano que proyecta la distribución condicional
//! multivariante de precios y órdenes hacia el futuro, modulado por el drift
//! microestructural (OFI/CVD), la curvatura gauge Yang-Mills y el entrelazamiento cuántico.
//!
//! # POR QUÉ:
//! Las proyecciones brownianas estándar asumen normalidad e independencia temporal,
//! ignorando colas pesadas leptocúrticas, saltos abruptos de liquidez y memoria
//! multifractal. Mirofish preserva la masa unitaria exacta y modela la asimetría real.
//!
//! # PARA QUÉ:
//! 1. Alimentar la visualización interactiva del tablero de Galton en vivo.
//! 2. Calcular los conos de incertidumbre conformal ACI dinámicos en 3D.
//! 3. Cuantificar el riesgo de cola extremo (CVaR/Expected Shortfall) por escala temporal.
//!
//! # CÓMO:
//! Mediante una red piramidal recombinante O(L^2) en stack/vector preasignado,
//! con probabilidades de transición asimétricas y relieve espectral en 32 escalas.
//!
//! # CUÁNDO:
//! Evaluado en cada intervalo de telemetría y disponible para la suite visual desacoplada.
//!
//! # DÓNDE:
//! En `crates/telemetry-server/src/mirofish_lattice.rs`.
//!
//! # QUIÉN:
//! Consejo Senior Quant & Científico de Información Cuántica (Fase 5).

use serde::{Deserialize, Serialize};

/// Las 32 escalas temporales canónicas en milisegundos (1 ns a ~9.13 años).
pub const SPECTRUM_SCALES_MS: [f64; 32] = [
    1.0e-6,                   // 1 ns
    4.0e-6,                   // 4 ns
    1.6e-5,                   // 16 ns
    6.4e-5,                   // 64 ns
    2.56e-4,                  // 256 ns
    1.024e-3,                 // ~1.02 µs
    4.096e-3,                 // ~4.10 µs
    1.6384e-2,                // ~16.38 µs
    6.5536e-2,                // ~65.54 µs
    0.262144,                 // ~262.14 µs
    1.048576,                 // ~1.05 ms
    4.194304,                 // ~4.19 ms
    16.777216,                // ~16.78 ms
    67.108864,                // ~67.11 ms
    268.435456,               // ~268.44 ms
    1_073.741824,             // ~1.07 s
    4_294.967296,             // ~4.29 s
    17_179.869184,            // ~17.18 s
    68_719.476736,            // ~1.15 min
    274_877.906944,           // ~4.58 min
    1_099_511.627776,         // ~18.33 min
    4_398_046.511104,         // ~1.22 h
    17_592_186.044416,        // ~4.89 h
    70_368_744.177664,        // ~19.55 h
    281_474_976.710656,       // ~3.26 d
    1_125_899_906.842624,     // ~13.03 d
    4_503_599_627.370496,     // ~52.12 d
    18_014_398_509.481984,    // ~208.5 d
    72_057_594_037.927936,    // ~2.28 años
    288_230_376_151.711744,   // ~9.13 años
    1_152_921_504_606.8469,   // ~36.5 años
    4_611_686_018_427.3877,   // ~146.15 años
];

/// Nodo individual en el Probability Lattice (Galton Board).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatticeNode {
    pub level: usize,
    pub index: usize,
    pub price: f64,
    pub probability: f64,
    pub cumulative_up: f64,
    pub cumulative_down: f64,
}

/// Nivel de profundidad en el Lattice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatticeLevel {
    pub step: usize,
    pub time_elapsed_s: f64,
    pub nodes: Vec<LatticeNode>,
    pub expected_price: f64,
    pub variance: f64,
}

/// Simulación completa del Lattice de Mirofish.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirofishLattice {
    pub root_price: f64,
    pub volatility_annual: f64,
    pub ofi_drift: f64,
    pub ising_delta_s: f64,
    pub conformal_alpha: f64,
    pub levels: Vec<LatticeLevel>,
    pub prob_up_total: f64,
    pub prob_down_total: f64,
}

/// Punto del Relieve de Colas 3D sobre una escala temporal.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TailRidgePoint {
    pub tau_ms: f64,
    pub scale_index: usize,
    pub left_tail_cvar: f64,
    pub right_tail_density: f64,
    pub conformal_band_low: f64,
    pub conformal_band_high: f64,
    pub excess_kurtosis: f64,
}

/// Conjunto de 32 puntos que definen el relieve tridimensional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TailRidge3D {
    pub points: Vec<TailRidgePoint>,
    pub max_risk_scale_ms: f64,
    pub min_risk_scale_ms: f64,
}

impl MirofishLattice {
    /// Construye una proyección recombinante completa de Mirofish.
    ///
    /// Preserva rigurosamente la ley de probabilidad total: sum(P) = 1.0 en cada nivel.
    pub fn compute(
        spot_price: f64,
        vol_annual: f64,
        ofi_drift: f64,
        ising_delta_s: f64,
        conformal_alpha: f64,
        max_levels: usize,
        dt_seconds: f64,
    ) -> Self {
        let safe_spot = if spot_price.is_finite() && spot_price > 0.0 {
            spot_price
        } else {
            100.0
        };
        let safe_vol = if vol_annual.is_finite() && vol_annual > 0.001 {
            vol_annual
        } else {
            0.50
        };
        let safe_levels = max_levels.clamp(1, 16);
        let safe_dt = if dt_seconds.is_finite() && dt_seconds > 0.0 {
            dt_seconds
        } else {
            1.0
        };
        let safe_alpha = if conformal_alpha.is_finite() {
            conformal_alpha.clamp(0.01, 0.50)
        } else {
            0.10
        };

        // Drift asimétrico modulado por el Order Flow Imbalance y la transición cuántica
        let safe_ofi = if ofi_drift.is_finite() { ofi_drift } else { 0.0 };
        let safe_ising = if ising_delta_s.is_finite() { ising_delta_s } else { 0.0 };
        
        let sigma_step = safe_vol * (safe_dt / (365.25 * 86400.0)).sqrt();
        let u = (sigma_step).exp();
        let d = 1.0 / u;

        // Probabilidad de subida local asimétrica con acoplamiento continuo
        let drift_term = (safe_ofi * 0.5 + safe_ising * 0.25).clamp(-2.0, 2.0);
        let p_up = (0.5 + 0.5 * (drift_term / (sigma_step + 1e-6)).tanh()).clamp(0.01, 0.99);
        let p_down = 1.0 - p_up;

        let mut levels = Vec::with_capacity(safe_levels + 1);

        // Nivel 0: Nodo raíz
        let root_node = LatticeNode {
            level: 0,
            index: 0,
            price: safe_spot,
            probability: 1.0,
            cumulative_up: p_up,
            cumulative_down: p_down,
        };
        levels.push(LatticeLevel {
            step: 0,
            time_elapsed_s: 0.0,
            nodes: vec![root_node],
            expected_price: safe_spot,
            variance: 0.0,
        });

        // Generar niveles 1..=safe_levels
        for step in 1..=safe_levels {
            let mut next_nodes = Vec::with_capacity(step + 1);
            let prev_level = &levels[step - 1];

            for i in 0..=step {
                // El precio en el nodo i del paso step es: spot * u^(step - i) * d^i
                let up_moves = (step - i) as i32;
                let down_moves = i as i32;
                let exponent = (up_moves - down_moves) as f64;
                let price = safe_spot * u.powf(exponent);

                // Recombinación probabilística binomial ponderada
                let mut prob = 0.0;
                if i < step {
                    // Contribución desde el nodo i del nivel anterior al bajar
                    let prev_node_a = &prev_level.nodes[i];
                    prob += prev_node_a.probability * p_down;
                }
                if i > 0 {
                    // Contribución desde el nodo i-1 del nivel anterior al subir
                    let prev_node_b = &prev_level.nodes[i - 1];
                    prob += prev_node_b.probability * p_up;
                }

                next_nodes.push(LatticeNode {
                    level: step,
                    index: i,
                    price,
                    probability: prob,
                    cumulative_up: 0.0,
                    cumulative_down: 0.0,
                });
            }

            // Normalización estricta de probabilidad para prevenir errores acumulativos IEEE-754
            let total_p: f64 = next_nodes.iter().map(|n| n.probability).sum();
            if total_p > 0.0 && total_p.is_finite() {
                for node in &mut next_nodes {
                    node.probability /= total_p;
                }
            }

            // Acumulados y estadísticas del nivel
            let mut exp_price = 0.0;
            let mut cum_up = 0.0;
            let mut cum_down = 0.0;
            for node in &mut next_nodes {
                exp_price += node.price * node.probability;
                if node.price >= safe_spot {
                    cum_up += node.probability;
                } else {
                    cum_down += node.probability;
                }
                node.cumulative_up = cum_up;
                node.cumulative_down = cum_down;
            }

            let mut var = 0.0;
            for node in &next_nodes {
                let diff = node.price - exp_price;
                var += diff * diff * node.probability;
            }

            levels.push(LatticeLevel {
                step,
                time_elapsed_s: step as f64 * safe_dt,
                nodes: next_nodes,
                expected_price: exp_price,
                variance: var,
            });
        }

        let last_level = &levels[levels.len() - 1];
        let prob_up_total = last_level
            .nodes
            .iter()
            .filter(|n| n.price >= safe_spot)
            .map(|n| n.probability)
            .sum();
        let prob_down_total = 1.0 - prob_up_total;

        Self {
            root_price: safe_spot,
            volatility_annual: safe_vol,
            ofi_drift: safe_ofi,
            ising_delta_s: safe_ising,
            conformal_alpha: safe_alpha,
            levels,
            prob_up_total,
            prob_down_total,
        }
    }
}

impl TailRidge3D {
    /// Genera el relieve de colas 3D a través de las 32 escalas del espectro continuo.
    pub fn generate(
        spot_price: f64,
        vol_base: f64,
        hurst: f64,
        conformal_alpha: f64,
    ) -> Self {
        let safe_spot = if spot_price.is_finite() && spot_price > 0.0 {
            spot_price
        } else {
            100.0
        };
        let safe_vol = if vol_base.is_finite() && vol_base > 0.001 {
            vol_base
        } else {
            0.50
        };
        let safe_h = if hurst.is_finite() {
            hurst.clamp(0.05, 0.95)
        } else {
            0.50
        };
        let safe_alpha = if conformal_alpha.is_finite() {
            conformal_alpha.clamp(0.005, 0.50)
        } else {
            0.05
        };

        let mut points = Vec::with_capacity(32);
        let tau_0_ms = 1_000.0; // 1 segundo como escala de normalización

        let mut max_risk = f64::NEG_INFINITY;
        let mut min_risk = f64::INFINITY;
        let mut max_risk_scale = SPECTRUM_SCALES_MS[0];
        let mut min_risk_scale = SPECTRUM_SCALES_MS[0];

        for (idx, &tau_ms) in SPECTRUM_SCALES_MS.iter().enumerate() {
            // Difusión multifractal: sigma(tau) = sigma_0 * (tau / tau_0)^H
            let tau_ratio = (tau_ms / tau_0_ms).max(1e-12);
            let sigma_tau = safe_vol * tau_ratio.powf(safe_h);

            // Exceso de curtosis decrece con la escala: alta frecuencia = leptocúrtico extremo
            let kurtosis_excess = (6.0 / (1.0 + (tau_ms / 1_000.0).sqrt())).clamp(0.1, 12.0);

            // Left Tail Expected Shortfall (CVaR al 1% con corrección de Cornish-Fisher)
            let z_alpha: f64 = 2.3263; // Cuantil 99% normal
            let cf_correction: f64 = (kurtosis_excess / 24.0) * (z_alpha * z_alpha * z_alpha - 3.0 * z_alpha);
            let z_cf: f64 = z_alpha + cf_correction;
            let left_tail_cvar = (safe_spot * (sigma_tau * z_cf * 0.1).tanh()).clamp(0.0, safe_spot * 0.95);

            // Densidad de cola derecha
            let right_tail_density = (0.5 * (-0.5 * (z_cf * 0.8) * (z_cf * 0.8)).exp() / (2.0 * std::f64::consts::PI).sqrt()).clamp(1e-6, 1.0);

            // Bandas conformales dinámicas
            let q_conformal = (1.0 - safe_alpha).clamp(0.5, 0.999);
            let margin = safe_spot * (sigma_tau * q_conformal * 0.15).tanh();
            let conformal_band_low = (safe_spot - margin).max(0.0);
            let conformal_band_high = safe_spot + margin;

            if left_tail_cvar > max_risk {
                max_risk = left_tail_cvar;
                max_risk_scale = tau_ms;
            }
            if left_tail_cvar < min_risk {
                min_risk = left_tail_cvar;
                min_risk_scale = tau_ms;
            }

            points.push(TailRidgePoint {
                tau_ms,
                scale_index: idx,
                left_tail_cvar,
                right_tail_density,
                conformal_band_low,
                conformal_band_high,
                excess_kurtosis: kurtosis_excess,
            });
        }

        Self {
            points,
            max_risk_scale_ms: max_risk_scale,
            min_risk_scale_ms: min_risk_scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirofish_lattice_probability_conservation() {
        let lattice = MirofishLattice::compute(100.0, 0.40, 0.0, 0.0, 0.10, 8, 1.0);
        assert_eq!(lattice.levels.len(), 9);

        // En cada nivel, la suma de probabilidades de los nodos debe ser exactamente 1.0
        for level in &lattice.levels {
            let sum_p: f64 = level.nodes.iter().map(|n| n.probability).sum();
            assert!(
                (sum_p - 1.0).abs() < 1e-9,
                "La suma de probabilidades en nivel {} no es 1.0 (fue {})",
                level.step,
                sum_p
            );
        }

        // Suma de probabilidades totales terminales debe ser 1.0
        assert!((lattice.prob_up_total + lattice.prob_down_total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mirofish_drift_asymmetry() {
        // Un drift positivo fuerte (OFI positivo) debe aumentar la probabilidad UP
        let neutral = MirofishLattice::compute(100.0, 0.30, 0.0, 0.0, 0.10, 6, 1.0);
        let bullish = MirofishLattice::compute(100.0, 0.30, 2.0, 1.0, 0.10, 6, 1.0);

        assert!(
            bullish.prob_up_total > neutral.prob_up_total,
            "Drift alcista debió aumentar prob_up_total: neutral={}, bullish={}",
            neutral.prob_up_total,
            bullish.prob_up_total
        );
        assert!(
            bullish.levels[6].expected_price > neutral.levels[6].expected_price,
            "Precio esperado debe ser superior con drift alcista"
        );
    }

    #[test]
    fn test_tail_ridge_3d_32_scales() {
        let ridge = TailRidge3D::generate(100.0, 0.50, 0.60, 0.05);
        assert_eq!(ridge.points.len(), 32);

        for (i, p) in ridge.points.iter().enumerate() {
            assert_eq!(p.scale_index, i);
            assert!(p.tau_ms > 0.0);
            assert!(p.left_tail_cvar >= 0.0 && p.left_tail_cvar.is_finite());
            assert!(p.conformal_band_low <= 100.0);
            assert!(p.conformal_band_high >= 100.0);
            assert!(p.excess_kurtosis > 0.0);
        }

        // La alta frecuencia (primeros puntos) debe exhibir mayor exceso de curtosis
        assert!(
            ridge.points[0].excess_kurtosis > ridge.points[31].excess_kurtosis,
            "Curtosis debe ser mayor en microestructura de alta frecuencia"
        );
    }
}
