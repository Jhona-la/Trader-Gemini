use crate::SignalType;
use strategy_core::{QuantumStrategy, TradeHorizon};

#[derive(Debug, Clone, Copy)]
pub struct TensorDecision {
    pub signal: SignalType,
    pub net_confidence: f64,
    pub expected_volatility: f64, // To be used by QuantumOrderRouter for Slippage Defense
    pub expected_lifetime_ms: u64, // Time-in-force heuristic
    pub horizon: TradeHorizon,
}

pub struct TensorVoteOrchestrator {
    strategies: Vec<Box<dyn QuantumStrategy>>,
    arena: std::sync::Arc<quantum_arena::GlobalArena>,
}

impl TensorVoteOrchestrator {
    pub fn new(arena: std::sync::Arc<quantum_arena::GlobalArena>) -> Self {
        Self {
            strategies: Vec::new(),
            arena,
        }
    }

    pub fn add_strategy(&mut self, mut strategy: Box<dyn QuantumStrategy>) {
        let _ = strategy.init(std::sync::Arc::clone(&self.arena.registry));
        self.strategies.push(strategy);
    }

    /// FIX #407: Evalúa el consenso bayesiano para un horizonte temporal específico (Scalp vs Swing),
    /// evitando la aniquilación mutua de estrategias con diferentes frecuencias operativas.
    pub fn evaluate_horizon_consensus(&self, target_horizon: TradeHorizon) -> TensorDecision {
        let horizon_strategies: Vec<&Box<dyn QuantumStrategy>> = self
            .strategies
            .iter()
            .filter(|s| s.horizon() == target_horizon)
            .collect();

        if horizon_strategies.is_empty() {
            return TensorDecision {
                signal: SignalType::Flat,
                net_confidence: 0.0,
                expected_volatility: 0.0,
                expected_lifetime_ms: 0,
                horizon: target_horizon,
            };
        }

        let mut long_votes = 0.0;
        let mut short_votes = 0.0;
        let mut active_weight = 0.0;
        let mut max_volatility = 0.0;

        for strategy in &horizon_strategies {
            let output = strategy.evaluate();
            if !output.is_finite() {
                continue;
            }
            let abs_weight = output.abs();

            if output > 0.0 {
                long_votes += abs_weight;
            } else if output < 0.0 {
                short_votes += abs_weight;
            }

            if abs_weight > max_volatility {
                max_volatility = abs_weight;
            }

            active_weight += abs_weight;
        }

        if active_weight == 0.0 {
            return TensorDecision {
                signal: SignalType::Flat,
                net_confidence: 0.0,
                expected_volatility: 0.0,
                expected_lifetime_ms: 0,
                horizon: target_horizon,
            };
        }

        let prob_long = long_votes / active_weight;
        let prob_short = short_votes / active_weight;

        let avg_conviction = active_weight / horizon_strategies.len().max(1) as f64;
        let ensemble_boost = 1.0 + (horizon_strategies.len().min(5) as f64 - 1.0) * 0.1;
        // FIX #1511: Sanitización estricta de convicción y volatilidad de ensamble
        let effective_conviction = if max_volatility.is_finite() && avg_conviction.is_finite() {
            max_volatility.max(avg_conviction * ensemble_boost).clamp(0.0, 1.0)
        } else {
            0.5
        };
        // FIX #592: Blindaje de finitud numérica para evitar propagación de NaN
        let raw_confidence = (prob_long - prob_short) * effective_conviction;
        let net_confidence = if raw_confidence.is_finite() { raw_confidence } else { 0.0 };
        let expected_volatility = if max_volatility.is_finite() { max_volatility.max(0.0) } else { 0.0 };

        let confidence_cutoff = self
            .arena
            .config
            .explosive_confidence_threshold
            .load(std::sync::atomic::Ordering::Relaxed);
        // FIX #1510: Sanitización de base_duration_ms antes del cálculo de lifetime
        let raw_base = self
            .arena
            .config
            .base_duration_ms
            .load(std::sync::atomic::Ordering::Relaxed);
        let base_duration = if raw_base.is_finite() && raw_base > 0.0 {
            raw_base as u64
        } else {
            30_000
        };

        let expected_lifetime_ms = match target_horizon {
            TradeHorizon::Scalp => {
                if net_confidence.abs() > confidence_cutoff {
                    (base_duration / 2).max(15_000)
                } else {
                    base_duration.max(30_000)
                }
            }
            TradeHorizon::Swing => {
                // Swing horizon operates on multi-hour time-in-force (minimum 1 hour)
                (base_duration * 10).max(3_600_000)
            }
        };

        // FIX #622 & #1403: Mapeo simétrico bayesiano de umbrales probabilísticos a espacio de convicción delta [-1, 1]
        let raw_long = self
            .arena
            .config
            .ml_threshold_long
            .load(std::sync::atomic::Ordering::Relaxed);
        let raw_short = self
            .arena
            .config
            .ml_threshold_short
            .load(std::sync::atomic::Ordering::Relaxed);

        let safe_long = if raw_long.is_finite() { raw_long.clamp(0.50, 0.99) } else { 0.55 };
        let safe_short = if raw_short.is_finite() { raw_short.clamp(0.01, 0.50) } else { 0.45 };

        let long_cutoff = ((safe_long - 0.50) * 2.0).clamp(0.01, 0.95);
        let short_cutoff = ((0.50 - safe_short) * 2.0).clamp(0.01, 0.95);

        if net_confidence > long_cutoff {
            TensorDecision {
                signal: SignalType::Long,
                net_confidence,
                expected_volatility,
                expected_lifetime_ms,
                horizon: target_horizon,
            }
        } else if net_confidence < -short_cutoff {
            TensorDecision {
                signal: SignalType::Short,
                net_confidence: net_confidence.abs(),
                expected_volatility,
                expected_lifetime_ms,
                horizon: target_horizon,
            }
        } else {
            TensorDecision {
                signal: SignalType::Flat,
                net_confidence: net_confidence.abs(),
                expected_volatility: 0.0,
                expected_lifetime_ms: 0,
                horizon: target_horizon,
            }
        }
    }

    pub fn evaluate_scalp_consensus(&self) -> TensorDecision {
        self.evaluate_horizon_consensus(TradeHorizon::Scalp)
    }

    pub fn evaluate_swing_consensus(&self) -> TensorDecision {
        self.evaluate_horizon_consensus(TradeHorizon::Swing)
    }

    /// Evalúa de forma desacoplada ambos horizontes simultáneamente (Scalp y Swing) sin supresión mutua (BUG-643)
    pub fn evaluate_dual_consensus(&self) -> (TensorDecision, TensorDecision) {
        let scalp_decision = self.evaluate_scalp_consensus();
        let swing_decision = self.evaluate_swing_consensus();
        (scalp_decision, swing_decision)
    }

    /// Evalúa todas las estrategias preservando la señal de mayor convicción según su horizonte
    pub fn evaluate_consensus(&self) -> TensorDecision {
        let scalp_decision = self.evaluate_scalp_consensus();
        let swing_decision = self.evaluate_swing_consensus();

        if scalp_decision.signal != SignalType::Flat && swing_decision.signal != SignalType::Flat {
            // FIX #599 & #1542: Ponderar convicción Swing (1.2x) por persistencia temporal macro con finitud estricta
            let weighted_swing_conf = if swing_decision.net_confidence.is_finite() {
                swing_decision.net_confidence * 1.20
            } else {
                0.0
            };
            if scalp_decision.net_confidence >= weighted_swing_conf {
                scalp_decision
            } else {
                swing_decision
            }
        } else if scalp_decision.signal != SignalType::Flat {
            scalp_decision
        } else {
            swing_decision
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omniscient_registry::OmniscientRegistry;
    use std::sync::Arc;

    struct MockStrategy {
        name: &'static str,
        value: f64,
    }

    impl QuantumStrategy for MockStrategy {
        fn name(&self) -> &str {
            self.name
        }

        fn init(&mut self, _registry: Arc<OmniscientRegistry>) -> Result<(), String> {
            Ok(())
        }

        fn evaluate(&self) -> f64 {
            self.value
        }
    }

    #[test]
    fn test_tensor_vote_orchestrator_consensus() {
        let arena = Arc::new(quantum_arena::GlobalArena::new(13.0));
        arena.config.ml_threshold_long.store(0.2, std::sync::atomic::Ordering::Relaxed);
        arena.config.ml_threshold_short.store(0.2, std::sync::atomic::Ordering::Relaxed);

        let mut orch = TensorVoteOrchestrator::new(arena);
        orch.add_strategy(Box::new(MockStrategy { name: "Bullish1", value: 0.9 }));
        orch.add_strategy(Box::new(MockStrategy { name: "Bullish2", value: 0.8 }));
        orch.add_strategy(Box::new(MockStrategy { name: "Bearish1", value: -0.1 }));

        let decision = orch.evaluate_consensus();
        assert_eq!(decision.signal, SignalType::Long);
        assert!(decision.net_confidence > 0.2);
    }

    #[test]
    fn test_tensor_vote_orchestrator_nan_and_flat_immunity() {
        let arena = Arc::new(quantum_arena::GlobalArena::new(13.0));
        let mut orch = TensorVoteOrchestrator::new(arena);
        orch.add_strategy(Box::new(MockStrategy { name: "NaN_Strat", value: f64::NAN }));
        orch.add_strategy(Box::new(MockStrategy { name: "Inf_Strat", value: f64::INFINITY }));

        let decision = orch.evaluate_consensus();
        assert_eq!(decision.signal, SignalType::Flat);
        assert_eq!(decision.net_confidence, 0.0);
    }

    struct HorizonMockStrategy {
        name: &'static str,
        value: f64,
        horizon: TradeHorizon,
    }

    impl QuantumStrategy for HorizonMockStrategy {
        fn name(&self) -> &str {
            self.name
        }

        fn init(&mut self, _registry: Arc<OmniscientRegistry>) -> Result<(), String> {
            Ok(())
        }

        fn evaluate(&self) -> f64 {
            self.value
        }

        fn horizon(&self) -> TradeHorizon {
            self.horizon
        }
    }

    #[test]
    fn test_tensor_vote_orchestrator_dual_horizon_consensus() {
        let arena = Arc::new(quantum_arena::GlobalArena::new(13.0));
        arena.config.ml_threshold_long.store(0.2, std::sync::atomic::Ordering::Relaxed);
        arena.config.ml_threshold_short.store(0.2, std::sync::atomic::Ordering::Relaxed);

        let mut orch = TensorVoteOrchestrator::new(arena);
        // Scalp es Bullish (+0.8), Swing es Bearish (-0.8)
        orch.add_strategy(Box::new(HorizonMockStrategy { name: "ScalpBull", value: 0.8, horizon: TradeHorizon::Scalp }));
        orch.add_strategy(Box::new(HorizonMockStrategy { name: "SwingBear", value: -0.8, horizon: TradeHorizon::Swing }));

        let (scalp_dec, swing_dec) = orch.evaluate_dual_consensus();
        assert_eq!(scalp_dec.signal, SignalType::Long, "Scalp debe resolver Long");
        assert_eq!(swing_dec.signal, SignalType::Short, "Swing debe resolver Short simultáneamente");
    }
}
