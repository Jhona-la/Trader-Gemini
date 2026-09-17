//! # Autonomous Phase Machine Engine
//!
//! Living 10-phase state machine governing organism life cycles:
//! Fase 0: Genesis
//! Fase 1: Ingesta
//! Fase 2: Exploración
//! Fase 3: Consolidación
//! Fase 4: Operación
//! Fase 5: Mutación
//! Fase 6: Validación
//! Fase 7: Crisis
//! Fase 8: Hibernación
//! Fase 9: Reproducción

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaseAutonomous {
    Fase0Genesis,
    Fase1Ingesta,
    Fase2Exploracion,
    Fase3Consolidacion,
    Fase4Operacion,
    Fase5Mutacion,
    Fase6Validacion,
    Fase7Crisis,
    Fase8Hibernacion,
    Fase9Reproduccion,
    Fase10CrisisEpistemica,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub concept_drift_score: f64,
    pub real_drawdown_pct: f64,
    pub sharpe_30d: f64,
    pub execution_latency_us: u64,
    pub data_checksum_ok: bool,
    pub compilation_success: bool,
    pub immune_tests_pass: bool,
    pub auditor_discrepancy_pct: f64,
    pub self_deception_detected: bool,
}

pub struct FaseAutonomousManager {
    pub current_phase: FaseAutonomous,
    pub phase_history: Vec<(FaseAutonomous, u64)>,
}

impl Default for FaseAutonomousManager {
    fn default() -> Self {
        Self {
            current_phase: FaseAutonomous::Fase0Genesis,
            phase_history: Vec::new(),
        }
    }
}

impl FaseAutonomousManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates epistemic conditions and executes phase transitions
    pub fn evaluate_transition(&mut self, tick: u64, metrics: &HealthMetrics) -> FaseAutonomous {
        let is_discrepancy_critical =
            !metrics.auditor_discrepancy_pct.is_finite() || metrics.auditor_discrepancy_pct > 0.10;
        if is_discrepancy_critical || metrics.self_deception_detected {
            if self.current_phase != FaseAutonomous::Fase10CrisisEpistemica {
                self.phase_history.push((self.current_phase, tick));
                self.current_phase = FaseAutonomous::Fase10CrisisEpistemica;
            }
            return FaseAutonomous::Fase10CrisisEpistemica;
        }

        let next_phase = match self.current_phase {
            FaseAutonomous::Fase0Genesis => {
                if metrics.data_checksum_ok {
                    FaseAutonomous::Fase2Exploracion
                } else {
                    FaseAutonomous::Fase1Ingesta
                }
            }
            FaseAutonomous::Fase1Ingesta => {
                if metrics.data_checksum_ok {
                    FaseAutonomous::Fase2Exploracion
                } else {
                    FaseAutonomous::Fase1Ingesta
                }
            }
            FaseAutonomous::Fase2Exploracion => {
                if metrics.compilation_success {
                    FaseAutonomous::Fase3Consolidacion
                } else {
                    FaseAutonomous::Fase2Exploracion
                }
            }
            FaseAutonomous::Fase3Consolidacion => {
                if metrics.immune_tests_pass {
                    FaseAutonomous::Fase4Operacion
                } else {
                    FaseAutonomous::Fase5Mutacion
                }
            }
            FaseAutonomous::Fase4Operacion => {
                if metrics.real_drawdown_pct > 0.08
                    || metrics.execution_latency_us > 500_000
                    || !metrics.data_checksum_ok
                {
                    FaseAutonomous::Fase7Crisis
                } else if metrics.concept_drift_score > 0.70 {
                    FaseAutonomous::Fase5Mutacion
                } else if metrics.sharpe_30d > 2.0 {
                    FaseAutonomous::Fase9Reproduccion
                } else {
                    FaseAutonomous::Fase4Operacion
                }
            }
            FaseAutonomous::Fase5Mutacion => {
                if metrics.compilation_success && metrics.immune_tests_pass {
                    FaseAutonomous::Fase6Validacion
                } else {
                    FaseAutonomous::Fase5Mutacion
                }
            }
            FaseAutonomous::Fase6Validacion => {
                if metrics.sharpe_30d >= 1.0 && metrics.real_drawdown_pct < 0.05 {
                    FaseAutonomous::Fase4Operacion
                } else {
                    FaseAutonomous::Fase2Exploracion
                }
            }
            FaseAutonomous::Fase7Crisis => {
                if metrics.immune_tests_pass && metrics.real_drawdown_pct < 0.02 {
                    FaseAutonomous::Fase2Exploracion
                } else {
                    FaseAutonomous::Fase7Crisis
                }
            }
            FaseAutonomous::Fase8Hibernacion => {
                if metrics.data_checksum_ok {
                    FaseAutonomous::Fase4Operacion
                } else {
                    FaseAutonomous::Fase8Hibernacion
                }
            }
            FaseAutonomous::Fase9Reproduccion => FaseAutonomous::Fase4Operacion,
            FaseAutonomous::Fase10CrisisEpistemica => {
                if metrics.auditor_discrepancy_pct <= 0.05 && !metrics.self_deception_detected {
                    FaseAutonomous::Fase2Exploracion
                } else {
                    FaseAutonomous::Fase10CrisisEpistemica
                }
            }
        };

        if next_phase != self.current_phase {
            self.phase_history.push((self.current_phase, tick));
            self.current_phase = next_phase;
        }

        self.current_phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fase_autonomous_transition_lifecycle() {
        let mut manager = FaseAutonomousManager::new();
        assert_eq!(manager.current_phase, FaseAutonomous::Fase0Genesis);

        let metrics = HealthMetrics {
            concept_drift_score: 0.1,
            real_drawdown_pct: 0.01,
            sharpe_30d: 2.5,
            execution_latency_us: 1000,
            data_checksum_ok: true,
            compilation_success: true,
            immune_tests_pass: true,
            auditor_discrepancy_pct: 0.01,
            self_deception_detected: false,
        };

        // Fase 0 -> Fase 2 (data_checksum_ok)
        let p = manager.evaluate_transition(1, &metrics);
        assert_eq!(p, FaseAutonomous::Fase2Exploracion);

        // Fase 2 -> Fase 3 (compilation_success)
        let p = manager.evaluate_transition(2, &metrics);
        assert_eq!(p, FaseAutonomous::Fase3Consolidacion);

        // Fase 3 -> Fase 4 (immune_tests_pass)
        let p = manager.evaluate_transition(3, &metrics);
        assert_eq!(p, FaseAutonomous::Fase4Operacion);

        // Fase 4 -> Fase 9 (sharpe > 2.0)
        let p = manager.evaluate_transition(4, &metrics);
        assert_eq!(p, FaseAutonomous::Fase9Reproduccion);
    }

    #[test]
    fn test_fase_autonomous_epistemic_crisis_override() {
        let mut manager = FaseAutonomousManager::new();
        let crisis_metrics = HealthMetrics {
            concept_drift_score: 0.1,
            real_drawdown_pct: 0.01,
            sharpe_30d: 2.5,
            execution_latency_us: 1000,
            data_checksum_ok: true,
            compilation_success: true,
            immune_tests_pass: true,
            auditor_discrepancy_pct: 0.25, // Discrepancia crítica
            self_deception_detected: true,
        };

        let p = manager.evaluate_transition(10, &crisis_metrics);
        assert_eq!(p, FaseAutonomous::Fase10CrisisEpistemica);
        assert_eq!(
            manager.current_phase,
            FaseAutonomous::Fase10CrisisEpistemica
        );
    }

    #[test]
    fn test_fase_autonomous_nan_discrepancy_triggers_crisis() {
        let mut manager = FaseAutonomousManager::new();
        let nan_metrics = HealthMetrics {
            concept_drift_score: 0.1,
            real_drawdown_pct: 0.01,
            sharpe_30d: 2.5,
            execution_latency_us: 1000,
            data_checksum_ok: true,
            compilation_success: true,
            immune_tests_pass: true,
            auditor_discrepancy_pct: f64::NAN, // Discrepancia corrupta
            self_deception_detected: false,
        };

        let p = manager.evaluate_transition(10, &nan_metrics);
        assert_eq!(p, FaseAutonomous::Fase10CrisisEpistemica);
    }
}
