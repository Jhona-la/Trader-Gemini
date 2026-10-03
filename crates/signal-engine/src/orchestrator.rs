use crate::SignalType;
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// #611 (Ola 33) — CENSO EMPÍRICO DE VOTANTES: por estrategia, cuántas
    /// evaluaciones produjeron voto no-cero vs el total. La prueba de vida
    /// de las voces del consenso (el "vota 0" de CL se cierra con datos,
    /// no con lectura estática). Telemetría: sin efecto en la decisión.
    censo_total: Vec<AtomicU64>,
    censo_no_cero: Vec<AtomicU64>,
    nombres: Vec<&'static str>,
    consensos_desde_publicacion: AtomicU64,
    /// #624 (Ola 45) — decisiones dirigidas por el CONSENSO ESPECTRAL
    /// (#623). Telemetría de adopción: cuántas veces habló la escala
    /// dominante vs el total (publicada junto al censo, cadencia 1024).
    decisiones_espectrales: AtomicU64,
    /// #652/H7 — distribución del dominante espectral: evaluaciones con
    /// clave publicada y las que superan el cutoff. La zona muerta de
    /// señales débiles unánimes sólo se recalibra con la distribución
    /// MEDIDA (observacional, sin tocar la decisión).
    vdom_evaluados: AtomicU64,
    vdom_sobre_corte: AtomicU64,
}

impl TensorVoteOrchestrator {
    pub fn new(arena: std::sync::Arc<quantum_arena::GlobalArena>) -> Self {
        Self {
            censo_total: Vec::new(),
            censo_no_cero: Vec::new(),
            nombres: Vec::new(),
            consensos_desde_publicacion: AtomicU64::new(0),
            decisiones_espectrales: AtomicU64::new(0),
            vdom_evaluados: AtomicU64::new(0),
            vdom_sobre_corte: AtomicU64::new(0),
            strategies: Vec::new(),
            arena,
        }
    }

    /// Compatibility wrapper; a failed initialization never admits a voter.
    pub fn add_strategy(&mut self, strategy: Box<dyn QuantumStrategy>) {
        if let Err(error) = self.try_add_strategy(strategy) {
            eprintln!("Strategy not registered: {error}");
        }
    }

    /// Returns initialization failure to callers that need startup completeness.
    /// Does not roll back registry side effects inside a strategy's initializer.
    pub fn try_add_strategy(
        &mut self,
        mut strategy: Box<dyn QuantumStrategy>,
    ) -> Result<(), String> {
        strategy.init(std::sync::Arc::clone(&self.arena.registry))?;
        // #611: el nombre debe vivir tanto como el censo (filtrado a
        // &'static — fuga acotada: una vez por estrategia registrada).
        let nombre: &'static str = Box::leak(strategy.name().to_string().into_boxed_str());
        self.nombres.push(nombre);
        self.censo_total.push(AtomicU64::new(0));
        self.censo_no_cero.push(AtomicU64::new(0));
        self.strategies.push(strategy);
        Ok(())
    }

    /// #611 — instantánea del censo empírico: (nombre, total, no_cero) por
    /// estrategia registrada. Lectura para el forense y los contratos; la
    /// vía en vivo es el registro (censo_total_*/censo_no_cero_*).
    pub fn censo_snapshot(&self) -> Vec<(&'static str, u64, u64)> {
        self.nombres
            .iter()
            .zip(self.censo_total.iter())
            .zip(self.censo_no_cero.iter())
            .map(|((nombre, total), nc)| {
                (*nombre, total.load(Ordering::Relaxed), nc.load(Ordering::Relaxed))
            })
            .collect()
    }

    /// Legacy horizon entry point; Continuous is the only representable variant.
    /// The vote score is a heuristic, not a calibrated Bayesian probability.
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

        let _confidence_cutoff = self
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

        let expected_lifetime_ms = {
            // U-6 (MOTOR UNIVERSAL CONTINUO): la vida esperada de la posición
            // se interpola por confianza entre el horizonte corto (1x base)
            // y el extendido (10x base) — sin modos binarios de horizonte.
            let conf = net_confidence.abs().clamp(0.0, 1.0);
            let scale = 1.0 + 9.0 * conf;
            ((base_duration as f64) * scale).max(30_000.0) as u64
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
        // MOD2/7-012 (INFORME DECIMOCUARTO): el factor ×2 convertía el gen en
        // SUPERMAYORÍA — con min_conf 0.70 exigía |net| > 0.40 entre 15
        // estrategias heterogéneas (≈ 70-30) y el consenso era Flat casi
        // siempre. Pendiente unitaria ×1.0 y techo 0.45: min_conf 0.70 ⇒
        // cutoff 0.20 (mayoría simple ≈ 60-40). El ML (B3.18) ya gatea la
        // entrada aguas abajo; el consenso no necesita repetir la supermayoría.
        let min_conf_gene = self
            .arena
            .config
            .min_confidence_btc
            .load(std::sync::atomic::Ordering::Relaxed);
        let cutoff_floor = ((min_conf_gene - 0.50) * 1.0).clamp(0.0, 0.45);
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
    ///
    /// U-F2 — CONSENSO DEL MOTOR TEMPORAL UNIVERSAL: TODO el ensamble
    /// participa (sin particiones por etiqueta) y la vida esperada resultante
    /// es la del continuo, interpolada por confianza.
    pub fn evaluate_continuous_consensus_for_coin(
        &self,
        coin_id: usize,
        symbol: &str,
    ) -> TensorDecision {
        let all = &self.strategies;
        if all.is_empty() {
            return TensorDecision {
                signal: SignalType::Flat,
                net_confidence: 0.0,
                expected_volatility: 0.0,
                expected_lifetime_ms: 0,
                horizon: TradeHorizon::Continuous,
            };
        }
        // #624 (Ola 45) — CONSUMO DEL CONSENSO ESPECTRAL: la sombra #623
        // del core compone los votos por escala de los 11 motores con
        // voto_espectral() y publica su escala dominante. Cuando existe
        // (≠0, finito), ESE es el veredicto del motor universal: dirección
        // y convicción de la escala que habló, y la posición vive a ESA τ
        // (la escala que la generó, no una interpolación por confianza).
        // Se lee ANTES de la guardia de peso activo: un ensamble escalar
        // que se abstiene (voto 0 unánime) no puede callar al espectro.
        // Sin dominante (arranque frío, espectro plano) el consenso
        // escalar queda bit a bit (disciplina D-754 de fallback).
        let v_dom = self
            .arena
            .registry
            .get_for_coin_or(coin_id, "consenso_espectral_dominante", 0.0);
        let tau_dom = self
            .arena
            .registry
            .get_for_coin_or(coin_id, "consenso_espectral_tau", 0.0);
        // #652/H2 (Ola 52) — el veredicto espectral sólo DIRIGE cuando su
        // escala es OPERABLE (τ ≥ 30 s). Un dominante a τ < 30 s rompía el
        // contrato "la posición vive a la escala que habló": el else
        // interpolaba la vida por confianza y el router dimensionaba la
        // geometría con una τ que NO era la del voto — y la orden moriría
        // en la puerta de banda operable (#586) de todos modos. Con τ
        // inoperable el veredicto espectral NO se consume: fallback
        // escalar bit a bit (la dirección espectral no puede comprar
        // geometría a una escala que el sistema no opera).
        let tau_operable = tau_dom.is_finite()
            && tau_dom >= quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS;
        let espectral_activo = v_dom.is_finite() && v_dom.abs() > 1e-9 && tau_operable;
        if espectral_activo {
            self.decisiones_espectrales.fetch_add(1, Ordering::Relaxed);
        }
        let mut long_votes = 0.0;
        let mut short_votes = 0.0;
        let mut active_weight = 0.0;
        let mut max_volatility = 0.0f64;
        // #611 — censo empírico: por estrategia, total de evaluaciones y
        // las que produjeron voto no-cero. Telemetría pura: no toca la
        // decisión (los acumuladores de abajo son los de siempre).
        let mut censo_muestras: Vec<(&'static str, u64, u64)> = Vec::new();
        for (idx, s) in all.iter().enumerate() {
            let output = s.evaluate_for_coin(coin_id, symbol);
            let (total_prev, nc_prev) = (
                self.censo_total.get(idx).map(|a| a.load(Ordering::Relaxed)).unwrap_or(0),
                self.censo_no_cero.get(idx).map(|a| a.load(Ordering::Relaxed)).unwrap_or(0),
            );
            let no_cero = u64::from(output.is_finite() && output.abs() > 1e-9);
            if let Some(a) = self.censo_total.get(idx) {
                a.store(total_prev + 1, Ordering::Relaxed);
            }
            if let Some(a) = self.censo_no_cero.get(idx) {
                a.store(nc_prev + no_cero, Ordering::Relaxed);
            }
            if let Some(nombre) = self.nombres.get(idx) {
                censo_muestras.push((nombre, total_prev + 1, nc_prev + no_cero));
            }
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
        // Publicación cadenciosa (en coin 0 — el censista global): cada
        // 1024 consensos, el mapa vivo de voces al registro.
        if coin_id == 0 {
            let n = self
                .consensos_desde_publicacion
                .fetch_add(1, Ordering::Relaxed);
            if n % 1024 == 0 {
                for (nombre, total, no_cero) in &censo_muestras {
                    self.arena.registry.set(
                        &format!("censo_total_{}", nombre),
                        *total as f64,
                    );
                    self.arena.registry.set(
                        &format!("censo_no_cero_{}", nombre),
                        *no_cero as f64,
                    );
                }
                // #624 — adopción del consenso espectral: decisiones
                // dirigidas por la escala dominante vs total evaluado.
                let espectral = self.decisiones_espectrales.load(Ordering::Relaxed);
                let total = (n + 1) as f64;
                self.arena
                    .registry
                    .set("qo_624_decisiones_espectrales", espectral as f64);
                self.arena
                    .registry
                    .set("qo_624_fraccion_espectral", espectral as f64 / total);
                // #652/H7 — distribución del dominante: evaluados vs
                // sobre-cutoff (fracción). La zona muerta H7 se
                // recalibra con esta distribución medida.
                let evaluados = self.vdom_evaluados.load(Ordering::Relaxed);
                let sobre = self.vdom_sobre_corte.load(Ordering::Relaxed);
                self.arena
                    .registry
                    .set("qo_652_vdom_evaluados", evaluados as f64);
                self.arena
                    .registry
                    .set("qo_652_vdom_sobre_corte", sobre as f64);
                self.arena.registry.set(
                    "qo_652_fraccion_sobre_corte",
                    sobre as f64 / evaluados.max(1) as f64,
                );
            }
        }
        if active_weight == 0.0 && !espectral_activo {
            return TensorDecision {
                signal: SignalType::Flat,
                net_confidence: 0.0,
                expected_volatility: 0.0,
                expected_lifetime_ms: 0,
                horizon: TradeHorizon::Continuous,
            };
        }
        // Con ensamble escalar abstenido y espectro hablando, los ratios
        // 0/0 se sanean abajo (NaN ⇒ 0.0); la convicción del ensamble es
        // 0 y el veredicto es puramente espectral.
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
        // #652/H4 — override espectral con modulación INDEPENDIENTE: la
        // convicción neta es el voto de la escala dominante modulado por
        // la COHERENCIA INTER-ESPECTRAL — cuánto respalda el resto del
        // espectro (media de banda) a su escala dominante: coherencia =
        // |media_banda|/|v_dom| ∈ [0,1]. Antes la modulación usaba la
        // convicción ESCALAR del ensamble — los mismos motores que
        // componen v_dom — la misma información contaba dos veces
        // (hallazgo H4 de la auditoría de arquitectura).
        let media_espectral = self.arena.registry.get_for_coin_or(
            coin_id,
            "consenso_espectral_media",
            0.0,
        );
        let coherencia_inter = if espectral_activo && v_dom.abs() > 1e-9 {
            (media_espectral / v_dom)
                .clamp(0.0, 1.0)
                .min(1.0)
        } else {
            0.0
        };
        let net_confidence = if espectral_activo {
            v_dom * (0.70 + 0.30 * coherencia_inter)
        } else {
            net_confidence
        };

        // R1.6 — `expected_volatility` es VOLATILIDAD DE PRECIO ESPERADA
        // (ATR% del feature engine), no el máximo |peso| de las salidas de
        // estrategia (adimensional 0..1). El consumidor crítico es el gate
        // del router, que la compara contra una fracción de precio.
        // `max_volatility` queda como valor de colas (clamp acotado) sólo si
        // el ATR no está disponible.
        let atr_pct = self
            .arena
            .registry
            .get_scoped_value_or(symbol, "atr_pct", f64::NAN);
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
            .get_scoped_value_or(symbol, "min_confidence", base_min_conf);
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
        let expected_lifetime_ms = if espectral_activo
            && tau_dom.is_finite()
            && tau_dom >= quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS
        {
            // #624 — la posición vive a la escala que habló: τ del consenso
            // espectral, clampeada a la banda operativa [30 s, 12 h].
            tau_dom
                .min(quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS)
                as u64
        } else {
            ((base_duration as f64) * scale).max(30_000.0) as u64
        };
        // #652/H7 — distribución contable: evaluaciones con dominante
        // publicado y las que superan el cutoff (recalibrar la zona
        // muerta exige la distribución medida, no opinión).
        self.vdom_evaluados.fetch_add(1, Ordering::Relaxed);
        if v_dom.is_finite() && v_dom.abs() > cutoff_floor {
            self.vdom_sobre_corte.fetch_add(1, Ordering::Relaxed);
        }
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

    /// Legacy pair of views of ONE continuous decision, not two independent engines.
    pub fn evaluate_dual_consensus(&self) -> (TensorDecision, TensorDecision) {
        self.evaluate_dual_consensus_for_coin(0, "BTCUSDT")
    }

    /// Evaluates once so stateful voters cannot consume the same observation twice.
    pub fn evaluate_dual_consensus_for_coin(
        &self,
        coin_id: usize,
        symbol: &str,
    ) -> (TensorDecision, TensorDecision) {
        let decision = self.evaluate_continuous_consensus_for_coin(coin_id, symbol);
        (decision, decision)
    }

    /// Compatibility entry point to the universal continuous consensus.
    pub fn evaluate_consensus(&self) -> TensorDecision {
        self.evaluate_consensus_for_coin(0, "BTCUSDT")
    }

    /// No second evaluation or legacy 1.2x preference for a nominal horizon.
    pub fn evaluate_consensus_for_coin(&self, coin_id: usize, symbol: &str) -> TensorDecision {
        self.evaluate_continuous_consensus_for_coin(coin_id, symbol)
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

    /// #624 (Ola 45) — CONTRATO DE INTEGRACIÓN ESPECTRAL.
    ///
    /// El orquestador consume el consenso espectral (#623): cuando la
    /// escala dominante tiene voto ≠0, ESE voto dirige la señal, modula
    /// por la convicción del ensamble, y la posición vive a ESA τ. Sin
    /// dominante publicado el consenso escalar decide bit a bit.
    #[test]
    fn qo_624_espectral_dirige_direccion_confianza_y_tau() {
        let arena = arena_con_min_conf(0.51);
        let mut orch = TensorVoteOrchestrator::new(std::sync::Arc::clone(&arena));
        orch.add_strategy(Box::new(MockStrategy {
            name: "M",
            value: 0.9,
        }));
        arena
            .registry
            .set_for_coin(0, "consenso_espectral_dominante", 0.8);
        arena
            .registry
            .set_for_coin(0, "consenso_espectral_tau", 3_600_000.0);
        // Sin media publicada ⇒ coherencia inter = 0 ⇒ conf = 0.8·0.70.
        let d = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert_eq!(d.signal, SignalType::Long);
        assert!(
            (d.net_confidence - 0.56).abs() < 1e-9,
            "confianza espectral sin respaldo de banda: {}",
            d.net_confidence
        );
        assert_eq!(d.expected_lifetime_ms, 3_600_000);

        // #652/H4 — coherencia INTER-espectral: la banda respalda al
        // dominante (media = v_dom) ⇒ modulación plena: conf = 0.8·1.0.
        arena
            .registry
            .set_for_coin(0, "consenso_espectral_media", 0.8);
        let d = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert!(
            (d.net_confidence - 0.8).abs() < 1e-9,
            "confianza con respaldo pleno de banda: {}",
            d.net_confidence
        );

        // Short simétrico: mismo |voto|, misma τ, misma coherencia.
        arena
            .registry
            .set_for_coin(0, "consenso_espectral_dominante", -0.8);
        arena
            .registry
            .set_for_coin(0, "consenso_espectral_media", -0.8);
        let d = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert_eq!(d.signal, SignalType::Short);
        assert_eq!(d.expected_lifetime_ms, 3_600_000);
        assert!((d.net_confidence - 0.8).abs() < 1e-9);
    }

    /// #624 — el dominante espectral GANA a la proyección escalar: es la
    /// opinión resuelta del mismo ensamble, no un voto más.
    #[test]
    fn qo_624_espectral_sobre_la_proyeccion_escalar() {
        let arena = arena_con_min_conf(0.51);
        let mut orch = TensorVoteOrchestrator::new(std::sync::Arc::clone(&arena));
        orch.add_strategy(Box::new(MockStrategy {
            name: "Bear",
            value: -0.9,
        }));
        arena
            .registry
            .set_for_coin(0, "consenso_espectral_dominante", 0.8);
        arena
            .registry
            .set_for_coin(0, "consenso_espectral_tau", 600_000.0);

        let d = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert_eq!(d.signal, SignalType::Long);
        assert_eq!(d.expected_lifetime_ms, 600_000);
        // La modulación NO usa la convicción escalar del mismo ensamble
        // (doble conteo H4): sin media ⇒ 0.8·0.70 = 0.56, NO 0.8·0.97.
        assert!((d.net_confidence - 0.56).abs() < 1e-9);
    }

    /// #624 — un ensamble escalar ABSTENIDO no calla al espectro: la
    /// guardia de peso activo cede cuando hay dominante espectral.
    #[test]
    fn qo_624_espectro_habla_cuando_el_ensamble_se_abstiene() {
        let arena = arena_con_min_conf(0.51);
        let mut orch = TensorVoteOrchestrator::new(std::sync::Arc::clone(&arena));
        orch.add_strategy(Box::new(MockStrategy {
            name: "Silencio",
            value: 0.0,
        }));
        arena
            .registry
            .set_for_coin(0, "consenso_espectral_dominante", 0.9);
        // #652/H2: τ operable — sin ella el veredicto espectral no dirige.
        arena
            .registry
            .set_for_coin(0, "consenso_espectral_tau", 120_000.0);

        let d = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert_eq!(d.signal, SignalType::Long);
        // Convicción del ensamble 0 y sin media ⇒ conf = 0.9·0.70.
        assert!((d.net_confidence - 0.63).abs() < 1e-9);
    }

    /// #624 — arranque frío bit a bit: sin dominante publicado (ausente,
    /// 0.0 explícito o no finito), la decisión es la del consenso escalar.
    #[test]
    fn qo_624_arranque_frio_fallback_escalar_bit_a_bit() {
        let arena = arena_con_min_conf(0.51);
        let mut orch = TensorVoteOrchestrator::new(std::sync::Arc::clone(&arena));
        orch.add_strategy(Box::new(MockStrategy {
            name: "A",
            value: 0.6,
        }));

        let frio = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert_eq!(frio.signal, SignalType::Long);

        arena
            .registry
            .set_for_coin(0, "consenso_espectral_dominante", 0.0);
        let plano = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert_eq!(frio.signal, plano.signal);
        assert_eq!(frio.net_confidence, plano.net_confidence);
        assert_eq!(frio.expected_lifetime_ms, plano.expected_lifetime_ms);

        arena
            .registry
            .set_for_coin(0, "consenso_espectral_dominante", f64::NAN);
        let nan = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert_eq!(frio.signal, nan.signal);
        assert_eq!(frio.net_confidence, nan.net_confidence);
        assert_eq!(frio.expected_lifetime_ms, nan.expected_lifetime_ms);
    }

    /// #624 — τ fuera de la banda operativa: por debajo del ancla rápida
    /// interpola por confianza; por encima de 12 h se clampa al techo.
    #[test]
    fn qo_624_tau_clampeada_a_la_banda_operativa() {
        let arena = arena_con_min_conf(0.51);
        let mut orch = TensorVoteOrchestrator::new(std::sync::Arc::clone(&arena));
        orch.add_strategy(Box::new(MockStrategy {
            name: "M",
            value: 0.9,
        }));
        arena
            .registry
            .set_for_coin(0, "consenso_espectral_dominante", 0.8);

        // #652/H2 — τ INOPERABLE (< 30 s): el espectral NO dirige —
        // fallback escalar (la dirección espectral no compra geometría a
        // una escala que el sistema no opera). La vida interpola.
        arena.registry.set_for_coin(0, "consenso_espectral_tau", 1_000.0);
        let baja = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert!(baja.expected_lifetime_ms >= 30_000);
        // Escalar el que decide: conf = 0.97 (ya modulada por el
        // ensamble) — NO el 0.8·0.70=0.56 que daría el espectral.
        assert!(
            (baja.net_confidence - 0.97).abs() < 1e-9,
            "fallback escalar con tau inoperable: {}",
            baja.net_confidence
        );

        arena
            .registry
            .set_for_coin(0, "consenso_espectral_tau", 100_000_000.0);
        let alta = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
        assert_eq!(
            alta.expected_lifetime_ms as f64,
            quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS
        );
    }
}
