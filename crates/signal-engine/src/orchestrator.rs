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
        if target_horizon == TradeHorizon::Continuous {
            return self.evaluate_continuous_consensus();
        }
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
        // FASE 2 (calibración): la convicción efectiva ya NO toma
        // `max_volatility` como término — mezclar volatilidad con convicción
        // inflaba la confianza de forma estructural (toda señal "sonaba" a
        // >0.9 sin relación con su frecuencia empírica de acierto). La
        // convicción es acuerdo del ensamble, no ruido del mercado.
        let effective_conviction = if avg_conviction.is_finite() {
            (avg_conviction * ensemble_boost).clamp(0.0, 1.0)
        } else {
            0.5
        };
        // FIX #592: Blindaje de finitud numérica para evitar propagación de NaN
        let raw_confidence = (prob_long - prob_short) * effective_conviction;
        let net_confidence = if raw_confidence.is_finite() {
            raw_confidence
        } else {
            0.0
        };
        // R1.6 — `expected_volatility` vuelve a ser lo que su nombre promete:
        // VOLATILIDAD DE PRECIO ESPERADA (ATR% del feature engine), no el
        // máximo |peso| de las salidas de estrategia (adimensional 0..1).
        // El consumidor crítico es el gate breakout del router, que la compara
        // contra scalp_sl_base/2 (una fracción de precio): con la versión
        // anterior el gate era SIEMPRE verdadero y la defensa anti-slippage
        // por volatilidad no existía. `max_volatility` queda como valor de
        // colas (clamp acotado) solo si el ATR no está disponible.
        let atr_pct = self.arena.registry.get_value_or("atr_pct", f64::NAN);
        let expected_volatility = if atr_pct.is_finite() && atr_pct > 0.0 {
            atr_pct
        } else if max_volatility.is_finite() {
            max_volatility.max(0.0).min(0.10)
        } else {
            0.0
        };

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
            TradeHorizon::Continuous => {
                // Sistema continuo universal: la vida esperada de la posición
                // se interpola por confianza entre el horizonte corto (1x base)
                // y el extendido (10x base), en vez de fijarse por modo binario.
                let conf = net_confidence.abs().clamp(0.0, 1.0);
                let scale = 1.0 + 9.0 * conf;
                ((base_duration as f64) * scale).max(30_000.0) as u64
            }
            TradeHorizon::Scalp => {
                if net_confidence.abs() > confidence_cutoff {
                    (base_duration / 2).max(15_000)
                } else {
                    base_duration.max(30_000)
                }
            }
            TradeHorizon::Swing => {
                (base_duration * 10).max(3_600_000)
            }
        };

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

        let long_dist = if raw_long >= 0.50 {
            raw_long - 0.50
        } else {
            0.50 - raw_long
        };
        let short_dist = if raw_short >= 0.50 {
            raw_short - 0.50
        } else {
            0.50 - raw_short
        };

        // FASE 2: el piso del cutoff ya no es el literal 0.08 (que dejaba
        // pasar casi cualquier señal cuando el umbral del genoma rondaba
        // 0.5). Se deriva del gen propio de confianza mínima
        // (`min_confidence_btc`): el edge mínimo operable es coherente con la
        // confianza mínima que el genoma exige en su gen más conservador.
        let min_conf_gene = self
            .arena
            .config
            .min_confidence_btc
            .load(std::sync::atomic::Ordering::Relaxed);
        let cutoff_floor = ((min_conf_gene - 0.50) * 2.0).clamp(0.0, 0.90);
        let long_cutoff = (long_dist * 2.0).clamp(cutoff_floor, 0.95);
        let short_cutoff = (short_dist * 2.0).clamp(cutoff_floor, 0.95);

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
        self.evaluate_continuous_consensus()
    }

    pub fn evaluate_swing_consensus(&self) -> TensorDecision {
        self.evaluate_continuous_consensus()
    }

    /// D-117: Consenso de Scalp escopado por activo real
    pub fn evaluate_scalp_consensus_for_coin(
        &self,
        coin_id: usize,
        symbol: &str,
    ) -> TensorDecision {
        self.evaluate_continuous_consensus_for_coin(coin_id, symbol)
    }

    /// D-117: Consenso de Swing escopado por activo real
    pub fn evaluate_swing_consensus_for_coin(
        &self,
        coin_id: usize,
        symbol: &str,
    ) -> TensorDecision {
        self.evaluate_continuous_consensus_for_coin(coin_id, symbol)
    }

    /// U-F2 — CONSENSO DEL MOTOR TEMPORAL UNIVERSAL: TODO el ensamble
    /// participa (sin particiones por etiqueta) y el lifetime resultante es
    /// el del continuo (interpolado por confianza, ya existente en la rama
    /// Continuous de evaluate_horizon_consensus). El motor universal tiene
    /// UNA opinión del mercado por tick; las etiquetas de estrategia son
    /// herencia de las fuentes, no del consenso.
    pub fn evaluate_continuous_consensus(&self) -> TensorDecision {
        self.evaluate_continuous_consensus_for_coin(0, "BTCUSDT")
    }

    /// D-101 & D-111: Consenso continuo multiactivo escopado por símbolo y moneda.
    /// Evita contaminación cruzada y colisiones de estado en el ensamble cuántico.
    pub fn evaluate_continuous_consensus_for_coin(
        &self,
        coin_id: usize,
        symbol: &str,
    ) -> TensorDecision {
        let all: Vec<&Box<dyn QuantumStrategy>> = self.strategies.iter().collect();
        if all.is_empty() {
            return TensorDecision {
                signal: SignalType::Flat,
                net_confidence: 0.0,
                expected_volatility: 0.0,
                expected_lifetime_ms: 0,
                horizon: TradeHorizon::Continuous,
            };
        }
        let mut long_votes = 0.0;
        let mut short_votes = 0.0;
        let mut active_weight = 0.0;
        let mut max_volatility = 0.0f64;
        for s in &all {
            let output = s.evaluate_for_coin(coin_id, symbol);
            if !output.is_finite() {
                continue;
            }
            let abs_w = output.abs();
            if output > 0.0 {
                long_votes += abs_w;
            } else if output < 0.0 {
                short_votes += abs_w;
            }
            active_weight += abs_w;
            if abs_w > max_volatility {
                max_volatility = abs_w;
            }
        }
        if active_weight == 0.0 {
            return TensorDecision {
                signal: SignalType::Flat,
                net_confidence: 0.0,
                expected_volatility: 0.0,
                expected_lifetime_ms: 0,
                horizon: TradeHorizon::Continuous,
            };
        }
        let prob_long = long_votes / active_weight;
        let prob_short = short_votes / active_weight;
        let avg_conviction = active_weight / all.len().max(1) as f64;
        let ensemble_boost = 1.0 + (all.len().min(5) as f64 - 1.0) * 0.1;
        let effective_conviction = if avg_conviction.is_finite() {
            (avg_conviction * ensemble_boost).clamp(0.0, 1.0)
        } else {
            0.5
        };
        // D-101 & D-425: Normalización continua sin double-squashing cuadrático y alineada con el quórum bayesiano
        let raw_net = prob_long - prob_short;
        let net_confidence = raw_net * (0.70 + 0.30 * effective_conviction);
        let net_confidence = if net_confidence.is_finite() {
            net_confidence
        } else {
            0.0
        };

        let coin_atr_key = format!("{}_atr_pct", symbol);
        let atr_pct = self.arena.registry.get_value_or(&coin_atr_key, f64::NAN);
        let atr_pct = if atr_pct.is_finite() && atr_pct > 0.0 {
            atr_pct
        } else {
            self.arena.registry.get_value_or("atr_pct", f64::NAN)
        };
        let expected_volatility = if atr_pct.is_finite() && atr_pct > 0.0 {
            atr_pct
        } else if max_volatility.is_finite() {
            max_volatility.max(0.0).min(0.10)
        } else {
            0.0
        };
        let base_min_conf = self
            .arena
            .config
            .min_confidence_btc
            .load(std::sync::atomic::Ordering::Relaxed);
        let min_conf_gene = self
            .arena
            .registry
            .get_value_or(&format!("{}_min_confidence", symbol), base_min_conf);
        let cutoff_floor = ((min_conf_gene - 0.50) * 2.0).clamp(0.0, 0.90);
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
        let conf = net_confidence.abs().clamp(0.0, 1.0);
        let scale = 1.0 + 9.0 * conf;
        let expected_lifetime_ms = ((base_duration as f64) * scale).max(30_000.0) as u64;
        if net_confidence.abs() > cutoff_floor {
            if net_confidence > 0.0 {
                TensorDecision {
                    signal: SignalType::Long,
                    net_confidence: net_confidence.abs(),
                    expected_volatility,
                    expected_lifetime_ms,
                    horizon: TradeHorizon::Continuous,
                }
            } else {
                TensorDecision {
                    signal: SignalType::Short,
                    net_confidence: net_confidence.abs(),
                    expected_volatility,
                    expected_lifetime_ms,
                    horizon: TradeHorizon::Continuous,
                }
            }
        } else {
            TensorDecision {
                signal: SignalType::Flat,
                net_confidence: 0.0,
                expected_volatility,
                expected_lifetime_ms,
                horizon: TradeHorizon::Continuous,
            }
        }
    }

    /// Evalúa de forma desacoplada ambos horizontes simultáneamente (Scalp y Swing) sin supresión mutua (BUG-643)
    pub fn evaluate_dual_consensus(&self) -> (TensorDecision, TensorDecision) {
        self.evaluate_dual_consensus_for_coin(0, "BTCUSDT")
    }

    /// D-424: Consenso dual desacoplado escopado por activo real
    pub fn evaluate_dual_consensus_for_coin(
        &self,
        coin_id: usize,
        symbol: &str,
    ) -> (TensorDecision, TensorDecision) {
        let scalp_decision = self.evaluate_scalp_consensus_for_coin(coin_id, symbol);
        let swing_decision = self.evaluate_swing_consensus_for_coin(coin_id, symbol);
        (scalp_decision, swing_decision)
    }

    /// Evalúa todas las estrategias preservando la señal de mayor convicción según su horizonte
    pub fn evaluate_consensus(&self) -> TensorDecision {
        self.evaluate_consensus_for_coin(0, "BTCUSDT")
    }

    /// D-424: Consenso global preservando mayor convicción escopado por activo real
    pub fn evaluate_consensus_for_coin(&self, coin_id: usize, symbol: &str) -> TensorDecision {
        let scalp_decision = self.evaluate_scalp_consensus_for_coin(coin_id, symbol);
        let swing_decision = self.evaluate_swing_consensus_for_coin(coin_id, symbol);

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
        arena
            .config
            .ml_threshold_long
            .store(0.2, std::sync::atomic::Ordering::Relaxed);
        arena
            .config
            .ml_threshold_short
            .store(0.2, std::sync::atomic::Ordering::Relaxed);

        let mut orch = TensorVoteOrchestrator::new(arena);
        orch.add_strategy(Box::new(MockStrategy {
            name: "Bullish1",
            value: 0.9,
        }));
        orch.add_strategy(Box::new(MockStrategy {
            name: "Bullish2",
            value: 0.8,
        }));
        orch.add_strategy(Box::new(MockStrategy {
            name: "Bearish1",
            value: -0.1,
        }));

        let decision = orch.evaluate_consensus();
        assert_eq!(decision.signal, SignalType::Long);
        assert!(decision.net_confidence > 0.2);
    }

    #[test]
    fn test_tensor_vote_orchestrator_nan_and_flat_immunity() {
        let arena = Arc::new(quantum_arena::GlobalArena::new(13.0));
        let mut orch = TensorVoteOrchestrator::new(arena);
        orch.add_strategy(Box::new(MockStrategy {
            name: "NaN_Strat",
            value: f64::NAN,
        }));
        orch.add_strategy(Box::new(MockStrategy {
            name: "Inf_Strat",
            value: f64::INFINITY,
        }));

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
    fn test_tensor_vote_orchestrator_continuous_consensus() {
        let arena = Arc::new(quantum_arena::GlobalArena::new(13.0));
        arena
            .config
            .min_confidence_btc
            .store(0.51, std::sync::atomic::Ordering::Relaxed);

        let mut orch = TensorVoteOrchestrator::new(arena);
        orch.add_strategy(Box::new(HorizonMockStrategy {
            name: "Bull1",
            value: 0.8,
            horizon: TradeHorizon::Continuous,
        }));
        orch.add_strategy(Box::new(HorizonMockStrategy {
            name: "Bull2",
            value: 0.6,
            horizon: TradeHorizon::Continuous,
        }));

        let dec = orch.evaluate_continuous_consensus();
        assert_eq!(
            dec.signal,
            SignalType::Long,
            "Continuous debe resolver Long"
        );
        assert_eq!(dec.horizon, TradeHorizon::Continuous);
    }
}
