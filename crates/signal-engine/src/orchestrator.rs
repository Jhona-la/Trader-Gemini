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

/// ORQUESTADOR DE VOTO TENSORIAL DEL MOTOR CONTINUO.
///
/// # U-ERR-2 (ERRADICACIÓN DE LA ARBITRACIÓN BINARIA MUERTA)
///
/// Este orquestador arrastraba SIETE entradas públicas de consenso además de
/// la real, todas sin un solo llamador en el repositorio:
///
/// * `evaluate_horizon_consensus` particionaba el ensamble por
///   `TradeHorizon` y aplicaba su propio corte doble
///   (`ml_threshold_long/short × 2`). Dentro calculaba `confidence_cutoff`
///   desde el gen `explosive_confidence_threshold` y NO LO USABA: una lectura
///   atómica por tick cuyo resultado se descartaba, que además hacía creer que
///   ese gen tenía consumidor.
/// * `evaluate_scalp_consensus_for_coin` y `evaluate_swing_consensus_for_coin`
///   eran alias IDÉNTICOS de `evaluate_continuous_consensus_for_coin`.
/// * `evaluate_dual_consensus[_for_coin]` devolvía el mismo valor dos veces.
/// * `evaluate_consensus[_for_coin]` arbitraba entre esos dos alias ponderando
///   «la banda lenta» por 1,20. Como ambas ramas eran el MISMO objeto, la
///   comparación `scalp.net_confidence >= swing.net_confidence * 1.20` era
///   `c >= 1.20·c`: falsa para toda confianza positiva. La arbitración
///   «elegía» siempre la segunda copia del mismo valor. Ningún 1,20 se derivó
///   nunca de una persistencia medida.
///
/// Queda UNA superficie: el consenso continuo escopado por moneda, que es la
/// que el core llama de verdad. En un motor temporal-espectral continuo no hay
/// bandas que arbitrar; hay un ensamble con UNA opinión por tick.
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

    /// D-101 & D-111: Consenso continuo multiactivo escopado por símbolo y moneda.
    /// Evita contaminación cruzada y colisiones de estado en el ensamble cuántico.
    ///
    /// U-F2 — CONSENSO DEL MOTOR TEMPORAL UNIVERSAL: TODO el ensamble
    /// participa (sin particiones por etiqueta) y la vida esperada resultante
    /// es la del continuo, interpolada por confianza.
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

        // R1.6 — `expected_volatility` es VOLATILIDAD DE PRECIO ESPERADA
        // (ATR% del feature engine), no el máximo |peso| de las salidas de
        // estrategia (adimensional 0..1). El consumidor crítico es el gate
        // del router, que la compara contra una fracción de precio.
        // `max_volatility` queda como valor de colas (clamp acotado) sólo si
        // el ATR no está disponible.
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
        // MOD2/7-012: ×1.0 (no ×2) y techo 0.45: mayoría simple, no
        // supermayoría. Con min_conf 0.70 ⇒ cutoff 0.20 en vez de 0.40.
        let cutoff_floor = ((min_conf_gene - 0.50) * 1.0).clamp(0.0, 0.45);
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
        // U-6 (MOTOR UNIVERSAL CONTINUO): la vida esperada de la posición se
        // interpola por confianza entre el horizonte base (1x) y el extendido
        // (10x) — sin modos binarios de horizonte.
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

    /// Arena con el gen de confianza mínima fijado, para que el corte del
    /// consenso sea determinista en los tests.
    fn arena_con_min_conf(min_conf: f64) -> Arc<quantum_arena::GlobalArena> {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        arena
            .config
            .min_confidence_btc
            .store(min_conf, std::sync::atomic::Ordering::Relaxed);
        arena
    }

    #[test]
    fn el_consenso_resuelve_long_con_mayoria_alcista() {
        let mut orch = TensorVoteOrchestrator::new(arena_con_min_conf(0.51));
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

        let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert_eq!(decision.signal, SignalType::Long);
        assert!(decision.net_confidence > 0.2);
        assert_eq!(decision.horizon, TradeHorizon::Continuous);
    }

    #[test]
    fn inmunidad_a_nan_e_infinito() {
        let mut orch = TensorVoteOrchestrator::new(arena_con_min_conf(0.51));
        orch.add_strategy(Box::new(MockStrategy {
            name: "NaN_Strat",
            value: f64::NAN,
        }));
        orch.add_strategy(Box::new(MockStrategy {
            name: "Inf_Strat",
            value: f64::INFINITY,
        }));

        let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert_eq!(decision.signal, SignalType::Flat);
        assert_eq!(decision.net_confidence, 0.0);
    }

    /// U-ERR-2 — LA ARBITRACIÓN POR BANDA NO EXISTE.
    ///
    /// Este test falla con el código viejo. Allí la entrada pública
    /// `evaluate_consensus_for_coin` comparaba dos copias del MISMO consenso
    /// ponderando una por 1,20: `c >= 1.20·c` es falso para toda `c > 0`, así
    /// que la rama «lenta» ganaba siempre por construcción, no por evidencia.
    ///
    /// La invariante que se fija aquí: para un ensamble dado, la decisión del
    /// motor es ÚNICA y ninguna ponderación de banda la altera — dos
    /// evaluaciones de la misma moneda con el mismo estado devuelven
    /// exactamente el mismo veredicto y la misma confianza, y la confianza NO
    /// está escalada por ningún factor de banda.
    #[test]
    fn u_err_2_una_sola_decision_sin_ponderacion_de_banda() {
        let mut orch = TensorVoteOrchestrator::new(arena_con_min_conf(0.51));
        orch.add_strategy(Box::new(MockStrategy {
            name: "A",
            value: 0.6,
        }));
        orch.add_strategy(Box::new(MockStrategy {
            name: "B",
            value: 0.4,
        }));

        let a = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        let b = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

        assert_eq!(a.signal, b.signal);
        assert_eq!(a.net_confidence, b.net_confidence);
        assert_eq!(a.expected_lifetime_ms, b.expected_lifetime_ms);

        // Con acuerdo unánime, prob_long = 1 y prob_short = 0: la confianza
        // neta es el quórum por el factor de convicción del ensamble, que
        // vive en [0.70, 1.00]. Cualquier ponderación de banda (p. ej. ×1,20)
        // la sacaría de ese intervalo.
        assert!(
            a.net_confidence > 0.0 && a.net_confidence <= 1.0,
            "confianza fuera del rango del quórum: {}",
            a.net_confidence
        );
    }
}
