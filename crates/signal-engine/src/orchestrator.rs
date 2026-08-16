use crate::SignalType;
use strategy_core::QuantumStrategy;

#[derive(Debug, Clone, Copy)]
pub struct TensorDecision {
    pub signal: SignalType,
    pub net_confidence: f64,
    pub expected_volatility: f64, // To be used by QuantumOrderRouter for Slippage Defense
    pub expected_lifetime_ms: u64, // Time-in-force heuristic
}

pub struct TensorVoteOrchestrator {
    strategies: Vec<Box<dyn QuantumStrategy>>,
    config: std::sync::Arc<quantum_arena::config::QuantumConfig>,
}

impl TensorVoteOrchestrator {
    pub fn new(config: std::sync::Arc<quantum_arena::config::QuantumConfig>) -> Self {
        Self {
            strategies: Vec::new(),
            config,
        }
    }

    pub fn add_strategy(&mut self, strategy: Box<dyn QuantumStrategy>) {
        self.strategies.push(strategy);
    }

    /// Evalúa todas las estrategias y usa un voto de consenso bayesiano para emitir la señal final.
    /// Resuelve el problema de "Ping-Pong" donde estrategias opuestas cancelan fees innecesariamente.
    pub fn evaluate_consensus(&self) -> TensorDecision {
        if self.strategies.is_empty() {
            return TensorDecision {
                signal: SignalType::Flat,
                net_confidence: 0.0,
                expected_volatility: 0.0,
                expected_lifetime_ms: 0,
            };
        }

        let mut long_votes = 0.0;
        let mut short_votes = 0.0;
        let mut total_weight = 0.0;
        let mut max_volatility = 0.0;

        for strategy in &self.strategies {
            let output = strategy.evaluate();
            // output está normalizado típicamente en [-1.0, 1.0] gracias al .tanh() de la red
            let abs_weight = output.abs();

            if output > 0.0 {
                long_votes += abs_weight;
            } else if output < 0.0 {
                short_votes += abs_weight;
            }

            // Si la estrategia es muy agresiva, subimos la volatilidad esperada
            if abs_weight > max_volatility {
                max_volatility = abs_weight;
            }

            total_weight += 1.0; // The theoretical max weight per strategy is 1.0
        }

        if total_weight == 0.0 {
            return TensorDecision {
                signal: SignalType::Flat,
                net_confidence: 0.0,
                expected_volatility: 0.0,
                expected_lifetime_ms: 0,
            };
        }

        // Probabilidades relativas
        let prob_long = long_votes / total_weight;
        let prob_short = short_votes / total_weight;

        // Si la diferencia (Net Confidence) supera el umbral estadístico (Conformal Prediction threshold)
        let net_confidence = prob_long - prob_short;

        let vol_scaler = self
            .config
            .tensor_dropout_rate
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(0.01);
        let expected_volatility = max_volatility * vol_scaler;

        let confidence_cutoff = self
            .config
            .explosive_confidence_threshold
            .load(std::sync::atomic::Ordering::Relaxed);
        let base_duration = self
            .config
            .base_duration_ms
            .load(std::sync::atomic::Ordering::Relaxed) as u64;
        let expected_lifetime_ms = if net_confidence.abs() > confidence_cutoff {
            base_duration / 4
        } else {
            base_duration
        };

        let ml_long_thresh = self
            .config
            .ml_threshold_long
            .load(std::sync::atomic::Ordering::Relaxed);
        let ml_short_thresh = self
            .config
            .ml_threshold_short
            .load(std::sync::atomic::Ordering::Relaxed);

        // FASE 13: HARDWARE-ASSISTED OUT-OF-BAND TELEMETRY
        // Removida la instrumentación explícita (push_tensor_record) del Hot-Path
        // para garantizar "Zero-Jitter" real. Ahora la telemetría se hace pasivamente
        // desde el Kernel (eBPF) y el PMU del CPU (Performance Monitoring Unit).

        if net_confidence > ml_long_thresh {
            TensorDecision {
                signal: SignalType::Long,
                net_confidence,
                expected_volatility,
                expected_lifetime_ms,
            }
        } else if net_confidence < -ml_short_thresh {
            TensorDecision {
                signal: SignalType::Short,
                net_confidence: net_confidence.abs(),
                expected_volatility,
                expected_lifetime_ms,
            }
        } else {
            // Estasis: El sistema no tiene la convicción matemática para arriesgar el capital
            TensorDecision {
                signal: SignalType::Flat,
                net_confidence: net_confidence.abs(),
                expected_volatility: 0.0,
                expected_lifetime_ms: 0,
            }
        }
    }
}
