use crate::anti_bias_governor::AntiBiasGovernor;
use crate::entropy_fitness::EntropyFitness;
use arc_swap::ArcSwap;
use dark_alpha_engine::MoENeatEngine;
use data_pipeline::telemetry_bus::ZeroCopyTelemetryBus;
use quantum_arena::GlobalArena;
use quantum_arena::config::QuantumConfig;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::time::sleep;

/// Daemon Autoevolutivo Cuántico (Online Learning & NEAT)
///
/// V9 FORENSIC FIX: Ahora usa datos REALES de la Arena atómica,
/// evoluciona AMBOS motores (Scalp + Swing), y calcula estadísticas
/// reales (Kurtosis, Skewness) de los trades para el AntiBiasGovernor.
/// U-5 (MOTOR UNIVERSAL CONTINUO): el daemon gemelo de evolución MoE
/// (scalp_moe/swing_moe con dos campeones y dos bucles) fue EXTIRPADO —
/// no tenía NINGÚN caller: era infraestructura muerta que además
/// propagaba el binario. Sobreviven las utilidades Pareto vivas.

pub struct ParetoCandidate {
    pub id: usize,
    pub sharpe: f64,       // Maximizar
    pub win_rate: f64,     // Maximizar
    pub max_drawdown: f64, // Minimizar
    pub pnl_realized: f64, // Maximizar
}

impl ParetoCandidate {
    pub fn new(
        id: usize,
        sharpe: f64,
        win_rate: f64,
        max_drawdown: f64,
        pnl_realized: f64,
    ) -> Self {
        // FIX #671: Sanitizar métricas de Pareto
        let s = if sharpe.is_finite() { sharpe } else { 0.0 };
        let wr = if win_rate.is_finite() {
            win_rate.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let dd = if max_drawdown.is_finite() {
            max_drawdown.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let pnl = if pnl_realized.is_finite() {
            pnl_realized
        } else {
            0.0
        };

        Self {
            id,
            sharpe: s,
            win_rate: wr,
            max_drawdown: dd,
            pnl_realized: pnl,
        }
    }

    /// Determina si `self` domina a `other` en sentido estricto de Pareto (NSGA-II)
    #[inline(always)]
    pub fn dominates(&self, other: &ParetoCandidate) -> bool {
        let not_worse = self.sharpe >= other.sharpe
            && self.win_rate >= other.win_rate
            && self.max_drawdown <= other.max_drawdown
            && self.pnl_realized >= other.pnl_realized;

        let strictly_better = self.sharpe > other.sharpe
            || self.win_rate > other.win_rate
            || self.max_drawdown < other.max_drawdown
            || self.pnl_realized > other.pnl_realized;

        not_worse && strictly_better
    }
}

/// Clasificación no dominada rápida (Fast Non-Dominated Sorting NSGA-II)
pub fn fast_non_dominated_sort(candidates: &[ParetoCandidate]) -> Vec<Vec<usize>> {
    let n = candidates.len();
    if n == 0 {
        return Vec::new();
    }

    let mut domination_counts = vec![0_usize; n];
    let mut dominated_sets: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut first_front = Vec::new();

    for p in 0..n {
        for q in 0..n {
            if p == q {
                continue;
            }
            if candidates[p].dominates(&candidates[q]) {
                dominated_sets[p].push(q);
            } else if candidates[q].dominates(&candidates[p]) {
                domination_counts[p] += 1;
            }
        }
        if domination_counts[p] == 0 {
            first_front.push(p);
        }
    }

    fronts.push(first_front);
    let mut i = 0;
    while i < fronts.len() && !fronts[i].is_empty() {
        let mut next_front = Vec::new();
        for &p in &fronts[i] {
            for &q in &dominated_sets[p] {
                domination_counts[q] = domination_counts[q].saturating_sub(1);
                if domination_counts[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        if !next_front.is_empty() {
            fronts.push(next_front);
        }
        i += 1;
    }

    fronts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pareto_dominance_and_nsga2_sorting() {
        let c1 = ParetoCandidate::new(1, 2.5, 0.65, 0.03, 100.0); // Dominante
        let c2 = ParetoCandidate::new(2, 1.5, 0.55, 0.08, 50.0); // Dominado por c1
        let c3 = ParetoCandidate::new(3, 3.0, 0.50, 0.02, 120.0); // No dominado (mejor Sharpe/DD)

        assert!(c1.dominates(&c2));
        assert!(!c2.dominates(&c1));
        assert!(!c1.dominates(&c3));
        assert!(!c3.dominates(&c1));

        let candidates = vec![c1, c2, c3];
        let fronts = fast_non_dominated_sort(&candidates);

        assert!(!fronts.is_empty());
        // Frente 1 debe contener c1 (idx 0) y c3 (idx 2)
        assert!(fronts[0].contains(&0));
        assert!(fronts[0].contains(&2));
        // Frente 2 debe contener c2 (idx 1)
        if fronts.len() > 1 {
            assert!(fronts[1].contains(&1));
        }
    }
}
