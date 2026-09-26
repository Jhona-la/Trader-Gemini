pub mod capital_compounder;
pub mod capital_regime;
pub mod correlation_guard;
pub mod epigenetic_capital_alloc;
pub mod epigenetic_fitness_landscape;
pub mod evidence;
pub mod guard;
pub mod kelly;
pub mod kelly_envelope;
pub mod leverage_matrix;
pub mod orchestrator;
pub mod random_matrix;
pub mod regime;
pub mod drawdown;
pub mod ruin;
pub mod tp_sl;

pub use kelly_envelope::{EdgePosterior, RiskEnvelope, SURVIVAL_FLOOR, TRADE_HORIZON};

use quantum_arena::GlobalArena;

use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct ValidatedOrder {
    pub signal: SignalType,
    pub volume_usd: f64,
    pub leverage: f64,
    pub maker_only: bool,
    pub tp_target: f64,
    pub sl_target: f64,
    pub fee_buffer_multiplier: f64,
    /// D-745 — HORIZONTE CON EL QUE SE DIMENSIONÓ ESTA ORDEN (ms). El núcleo
    /// lo guarda tal cual en `entry_tau_ms`, de modo que la geometría, el
    /// trailing, la caducidad, el Kelly del cierre y el apalancamiento del
    /// host razonan sobre el MISMO horizonte con el que se calculó el tamaño.
    /// Antes, el núcleo lo recalculaba desde la τ dominante del espectro y la
    /// misma posición se dimensionaba a un horizonte y se gestionaba en otro.
    pub tau_ms: f64,
}

impl ValidatedOrder {
    /// Exact integer representation, never a truncating conversion. Exchange
    /// adapters must additionally enforce their instrument/API-specific cap.
    pub fn integer_leverage(&self) -> Option<u32> {
        if !self.leverage.is_finite()
            || self.leverage < 1.0
            || self.leverage > u32::MAX as f64
            || self.leverage.fract() != 0.0
        {
            return None;
        }
        Some(self.leverage as u32)
    }

    pub fn rejected() -> Self {
        Self {
            signal: SignalType::Flat,
            volume_usd: 0.0,
            leverage: 1.0,
            maker_only: false,
            tp_target: 0.0,
            sl_target: 0.0,
            fee_buffer_multiplier: 1.01,
            tau_ms: 0.0,
        }
    }
}

// Symbol constraints and size limits have been removed to allow purely dynamic and infinite asset discovery.
// The engine now strictly relies on mathematical limits derived from Kelly and margin constraints.

/// DIAGNÓSTICO SIGNAL-PATH: contadores de rechazo por compuerta de
/// evaluate_single_intent. Índices:
/// 0=flat/coin 1=exposure0 2=correlación 3=spec 4=EV 5=fee_impact
/// 6=min_notional 7=margen_insuf 8=orchestrator 9=otros
/// 10=drawdown 11=suelo TP/SL 12=confianza.
/// 13=viabilidad (D-750: la orden mínima ya arriesga más de lo que el control
/// de ruina permite) 14=sin evidencia (D-751) 15=entrada inválida
/// 16=geometría inválida (fusión PR #5: ambas familias de códigos, slots
/// expandidos 15→17). Antes el drawdown compartía el índice 2 con la
/// correlación, y el suelo TP/SL y la confianza el 4 con el EV: la telemetría
/// no podía decir qué compuerta rechazaba.
use std::sync::atomic::AtomicU64;
pub const REJECT_SLOTS: usize = 17;
pub const REJ_DRAWDOWN: usize = 10;
pub const REJ_TP_SL_FLOOR: usize = 11;
pub const REJ_CONFIDENCE: usize = 12;
pub const REJ_INVALID_INPUT: usize = 15;
pub const REJ_TARGET_GEOMETRY: usize = 16;
/// D-750 — la orden más pequeña que el símbolo acepta ya arriesga más de lo que
/// el control de ruina permite: la operación es INVIABLE, no «pequeña».
pub const REJ_VIABILIDAD: usize = 13;
/// D-751 — no hay probabilidad de ganar (ni calibrada ni observada) con la que
/// evaluar el valor esperado: se rechaza por falta de evidencia.
pub const REJ_SIN_EVIDENCIA: usize = 14;

#[allow(clippy::declare_interior_mutable_const)]
const REJECT_ZERO: AtomicU64 = AtomicU64::new(0);
#[allow(clippy::declare_interior_mutable_const)]
const REJECT_ZERO_ROW: [AtomicU64; REJECT_SLOTS] = [REJECT_ZERO; REJECT_SLOTS];

pub static REJECT_COUNTERS: [AtomicU64; REJECT_SLOTS] = [REJECT_ZERO; REJECT_SLOTS];

/// Rechazos atribuidos a la dirección de la intención evaluada: [largo, corto].
pub static REJECT_COUNTERS_DIR: [[AtomicU64; REJECT_SLOTS]; 2] = [REJECT_ZERO_ROW; 2];

thread_local! {
    /// Dirección de la intención en evaluación: 0 largo, 1 corto, 2 sin dirección.
    static REJECT_DIR: std::cell::Cell<usize> = const { std::cell::Cell::new(2) };
}

/// Fija la dirección con la que se atribuyen los rechazos siguientes del hilo.
pub fn set_reject_direction(signal: SignalType) {
    let d = match signal {
        SignalType::Long => 0,
        SignalType::Short => 1,
        SignalType::Flat => 2,
    };
    REJECT_DIR.with(|cell| cell.set(d));
}

fn rej(i: usize) -> ValidatedOrder {
    REJECT_COUNTERS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let d = REJECT_DIR.with(|cell| cell.get());
    if d < 2 {
        REJECT_COUNTERS_DIR[d][i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    ValidatedOrder::rejected()
}

/// Nombre de cada compuerta de rechazo, por índice.
pub const REJECT_NAMES: [&str; REJECT_SLOTS] = [
    "flat/coin",
    "exposure0",
    "correlacion",
    "spec",
    "EV",
    "fee_impact",
    "min_notional",
    "margen_insuf",
    "orchestrator",
    "otros",
    "drawdown",
    "suelo_tp_sl",
    "confianza",
    "viabilidad",
    "sin_evidencia",
    "entrada_invalida",
    "geometria_invalida",
];

fn format_reject_counters(counters: &[AtomicU64; REJECT_SLOTS]) -> String {
    let v: Vec<String> = counters
        .iter()
        .enumerate()
        .filter(|(_i, c)| c.load(std::sync::atomic::Ordering::Relaxed) > 0)
        .map(|(i, c)| {
            format!(
                "{}={}",
                REJECT_NAMES[i],
                c.load(std::sync::atomic::Ordering::Relaxed)
            )
        })
        .collect();
    if v.is_empty() {
        "sin rechazos".into()
    } else {
        v.join(" ")
    }
}

/// Rechazos globales y por dirección de la intención evaluada.
pub fn reject_report() -> String {
    format!(
        "{} · largo: {} · corto: {}",
        format_reject_counters(&REJECT_COUNTERS),
        format_reject_counters(&REJECT_COUNTERS_DIR[0]),
        format_reject_counters(&REJECT_COUNTERS_DIR[1]),
    )
}

pub struct RiskEngine {
    pub peak_capital: f64,
}

impl RiskEngine {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            peak_capital: initial_capital,
        }
    }

    pub fn reset(&mut self, initial_capital: f64) {
        self.peak_capital = initial_capital;
    }

    /// U-3 (MOTOR UNIVERSAL CONTINUO): la API dual `evaluate_order`
    /// (scalp+swing → dos órdenes, split de capital por horizonte, picos y
    /// Kelly envelopes gemelos) fue EXTIRPADA — el camino de producción es
    /// y era esta función: una intención, un pico, Kelly con
    /// `temporal_scale` s∈[0,1] continuo (D-427).
    /// Evalúa la intención unificada de señal cuántica continua sobre el 100% del capital disponible.
    pub fn evaluate_quantum_order(
        &mut self,
        coin_id: usize,
        intent: &SignalIntent,
        arena: &GlobalArena,
    ) -> ValidatedOrder {
        set_reject_direction(intent.signal);
        if coin_id >= arena.coins.len() || intent.signal == SignalType::Flat {
            return rej(0);
        }

        let current_capital = arena.unified_capital.load(Ordering::Relaxed);
        // FMT-212: validate before clamps/comparisons can turn an unknown
        // probability, peak or configuration into permission (or a panic).
        let clamp_min = arena.config.kelly_clamp_min.load(Ordering::Relaxed);
        let clamp_max = arena.config.kelly_clamp_max.load(Ordering::Relaxed);
        let configured_dd = arena.config.global_max_drawdown.load(Ordering::Relaxed);
        if !current_capital.is_finite()
            || current_capital <= 0.0
            || !self.peak_capital.is_finite()
            || self.peak_capital <= 0.0
            || !intent.confidence.is_finite()
            || !(0.0..=1.0).contains(&intent.confidence)
            || !intent.win_probability.is_finite()
            || !(0.0..=1.0).contains(&intent.win_probability)
            || !(0.0..=1.0).contains(&clamp_min)
            || !(clamp_min..=1.0).contains(&clamp_max)
            || !(0.0..=1.0).contains(&configured_dd)
        {
            return rej(REJ_INVALID_INPUT);
        }
        // Zero alone means "derive this target". Malformed explicit targets
        // must not silently fall back to a different trade.
        if !intent.tp_price_target.is_finite()
            || intent.tp_price_target < 0.0
            || !intent.sl_price_target.is_finite()
            || intent.sl_price_target < 0.0
        {
            return rej(REJ_TARGET_GEOMETRY);
        }

        if current_capital > self.peak_capital {
            self.peak_capital = current_capital;
        }
        // U-3: los picos espejo scalp/swing (sin lectores) extirpados.

        // O-04 — CIRCUIT BREAKER DE DRAWDOWN en el path cuántico: el gen
        // global_max_drawdown era funcionalmente MUERTO (se convertía en cap
        // de margen constante). Ahora es un VETO REAL: si el drawdown desde
        // el pico supera el gen, no se abre nueva posición hasta recuperación.
        // D-641 (completo): el cortacircuitos de drawdown deja de saltar en $15.
        // En régimen micro pleno (≤3 operaciones mínimas, p. ej. $13) conserva
        // la tolerancia de 0,85 diseñada para permitir la recuperación del
        // crecimiento compuesto; en régimen estándar rige el gen; entre ambos,
        // transición continua. Antes el gen quedaba anulado en producción.
        let micro_w = crate::capital_regime::micro_weight(
            current_capital,
            arena.config.min_notional.load(Ordering::Relaxed),
        );
        let max_dd = crate::capital_regime::lerp(configured_dd, 0.85, micro_w);
        if self.peak_capital > 0.0 && max_dd > 0.0 && max_dd < 1.0 {
            let dd = (self.peak_capital - current_capital) / self.peak_capital;
            let q_perdida = 1.0
                - arena.coins[coin_id]
                    .metrics
                    .win_rate
                    .load(Ordering::Relaxed)
                    .clamp(0.0, 1.0);
            // D-744b: sin riesgo medido rige el gen; el veto nunca se salta.
            let max_dd = crate::drawdown::drawdown_maximo(
                arena.riesgo_por_operacion.load(Ordering::Relaxed),
                q_perdida,
                arena.config.global_max_drawdown.load(Ordering::Relaxed),
            );
            if dd >= max_dd {
                return rej(REJ_DRAWDOWN);
            }
        }

        let base_capital = arena.config.base_capital.load(Ordering::Relaxed);
        let trades_n = arena.coins[coin_id]
            .metrics
            .trade_count
            .load(Ordering::Relaxed);
        // D-749 — EL PROFIT FACTOR QUE ENTRA AL DIMENSIONADO ES UNA COTA
        // INFERIOR, NO EL LITERAL QUE PUBLICA EL PRODUCTOR.
        //
        // `coin.metrics.profit_factor` vale 5,0 en cuanto hay ganancias y
        // ninguna pérdida registrada, y 1,50 sin historial: dos números
        // inventados. Con 5,0 una ÚNICA operación ganadora sacaba a la moneda
        // de la rama de exploración (`pf <= 1.0`) y ponía
        // `kelly_from_pf = p·(1 − 1/5) = 0,8·p` — el Kelly prácticamente en su
        // techo con una observación. Aquí se recalcula desde los estadísticos
        // suficientes (sumas de ganancias y pérdidas + número de operaciones)
        // con corrección de continuidad y descuento por tamaño de muestra.
        let pf = crate::evidence::profit_factor_lcb(
            arena.coins[coin_id]
                .metrics
                .gross_wins
                .load(Ordering::Relaxed),
            arena.coins[coin_id]
                .metrics
                .gross_losses
                .load(Ordering::Relaxed),
            arena.coins[coin_id]
                .metrics
                .trade_count
                .load(Ordering::Relaxed) as f64,
        );
        let clamp_min = arena
            .config
            .kelly_clamp_min
            .load(Ordering::Relaxed)
            .max(0.0);
        let clamp_max = arena
            .config
            .kelly_clamp_max
            .load(Ordering::Relaxed)
            .clamp(clamp_min, 1.0);
        let raw_kelly = arena.coins[coin_id]
            .metrics
            .kelly_fraction
            .load(Ordering::Relaxed);

        // CERT-M5-C01: el bootstrap anterior clampeaba a [0.05, 0.35] con
        // baseline 0.5 → 0.35. Con PF ≤ 1 (probado SIN edge), una moneda
        // tradearía al 35% Kelly — bypassando la protección de kelly.rs
        // (PF≤1 → exploración ≤ ¼ del piso). Ahora: PF ≤ 1 ⇒ ≤ ¼ del
        // clamp_min del genoma (la MISMA regla que kelly.rs aplica una
        // capa abajo); PF > 1 sin historial ⇒ bootstrap del genoma.
        // D-749: la segunda lectura de `profit_factor` —idéntica a la de arriba,
        // que la sombreaba sin cambiarla— se elimina: el PF de esta evaluación
        // es UNO y es la cota inferior calculada más arriba.
        let kelly_cold_raw = arena
            .config
            .kelly_bootstrap_cold
            .load(Ordering::Relaxed);
        let kelly_cold = if pf <= 1.0 {
            // Sin edge probado: exploración ultra-conservadora (¼ del piso)
            (clamp_min.max(0.0) * 0.25).clamp(0.0, 0.05)
        } else {
            kelly_cold_raw.clamp(0.05, 0.35)
        };
        let kelly_frac = if trades_n == 0 || raw_kelly <= 0.0 {
            kelly_cold
        } else {
            raw_kelly.clamp(clamp_min, clamp_max)
        };
        // CERT-M5-H03: el camino bootstrap/micro YA NO bypassa el tope de
        // ruina — mismo streak-bound + axioma 25% que kelly/envelope. q del
        // win-rate del coin si existe; conservador si no.
        let wr_coin = arena.coins[coin_id]
            .metrics
            .win_rate
            .load(Ordering::Relaxed);
        let q = if wr_coin > 0.0 && wr_coin < 1.0 {
            1.0 - wr_coin
        } else {
            crate::ruin::CONSERVATIVE_Q
        };
        let kelly_frac = crate::ruin::clamp_ruin(kelly_frac, q);
        let temporal_scale = arena
            .config
            .temporal_scale
            .load(Ordering::Relaxed)
            .clamp(0.0, 1.0);
        // D-427: Preservar variedad continua sin colapso booleano s in [0, 1]
        self.evaluate_single_intent(
            coin_id,
            intent,
            kelly_frac,
            current_capital,
            base_capital,
            pf,
            temporal_scale,
            arena,
        )
    }

    /// Evalúa la intención cuántica continua (alias para compatibilidad)
    pub fn evaluate_quantum_order_by_horizon(
        &mut self,
        coin_id: usize,
        intent: &SignalIntent,
        _is_scalp: bool,
        arena: &GlobalArena,
    ) -> ValidatedOrder {
        self.evaluate_quantum_order(coin_id, intent, arena)
    }

    fn evaluate_single_intent(
        &self,
        coin_id: usize,
        intent: &SignalIntent,
        kelly_fraction: f64,
        allocated_capital: f64,
        base_allocated: f64,
        profit_factor: f64,
        temporal_scale: f64,
        arena: &GlobalArena,
    ) -> ValidatedOrder {
        set_reject_direction(intent.signal);
        if intent.signal == SignalType::Flat || allocated_capital <= 0.0 {
            return ValidatedOrder::rejected();
        }
        // D-750 — EL NOCIONAL MÍNIMO ES EL DEL SÍMBOLO, NO UN LITERAL DE
        // CONFIGURACIÓN.
        //
        // `arena.config.min_notional` nace con el literal 5,0 y NADIE lo
        // escribe nunca (no existe un solo `.store` sobre él en todo el
        // repositorio): es un número congelado que gobernaba el régimen de
        // capital, el guard de correlación y el piso de viabilidad del sizing.
        // El mínimo REAL lo publica el exchange por símbolo
        // (`symbol_registry` ← `exchangeInfo`) y `capital_regime::
        // effective_min_notional` ya sabía leerlo — pero sólo lo usaba el
        // chequeo de nocional, al final de la cadena. Con un símbolo cuyo
        // mínimo sea 20 $ en lugar de 5 $, el régimen de capital situaba una
        // cuenta de 13 $ en «2,6 operaciones mínimas» cuando en realidad no
        // cabe ni una. La consulta del spec sube AQUÍ para que TODA la
        // evaluación razone con el mismo mínimo.
        let spec = match quantum_arena::symbol_registry::try_spec(coin_id) {
            Some(s) => s,
            None => return rej(3),
        };
        let max_exchange_leverage = spec.max_leverage as f64;
        let dynamic_min_notional = crate::capital_regime::effective_min_notional(spec.min_notional);

        // D-641 (completo): peso del régimen de capital micro para TODA la
        // evaluación. Un único valor, calculado una vez, del que derivan todas
        // las transiciones que antes eran escalones en $15 y $20.
        let micro_w_alloc =
            crate::capital_regime::micro_weight(allocated_capital, dynamic_min_notional);

        let dir = match intent.signal {
            SignalType::Long => 1.0,
            SignalType::Short => -1.0,
            _ => 0.0,
        };

        // O-03 — KELLY ESCALADO POR RIESGO DEL STOP: la fracción Kelly es
        // asintóticamente proporcional al edge/riesgo; si el stop es K× más
        // ancho que el extremo corto, la MISMA fracción arriesga K× más
        // capital por trade. Escalamos por temporal_scale para que el RIESGO
        // POR TRADE sea constante en el continuo (sin el salto invisible
        // de 4-8x en riesgo entre extremos).
        // O-03 — KELLY ESCALADO POR RIESGO DEL STOP CONTINUO:
        // En el espectro continuo universal (1 ns a 100 años), evaluamos el SL directamente
        // sobre la curva analítica del genoma sl_at_tau(tau_ms) para normalizar el riesgo
        // de forma suave y C^inf sin buckets discretos ni saltos artificiales.
        // D-638b: mapeo τ ÚNICO del sistema. Antes este bloque interpolaba entre
        // 10 s y 24 h mientras el gate de TP/SL lo hacía sobre los extremos del
        // espectro y la matriz de apalancamiento con otra fórmula: tres
        // horizontes distintos para la misma intención.
        let tau_ms = horizon_tau_ms_coin(intent, arena, coin_id);
        let continuous_sl = arena.config.sl_at_tau(tau_ms).max(1e-6);
        let fast_anchor_sl = arena
            .config
            .sl_at_tau(quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS)
            .max(1e-6);
        let risk_normalizer = (fast_anchor_sl / continuous_sl).clamp(0.15, 1.0);
        let kelly_adjusted = kelly_fraction * risk_normalizer;

        // D-494: Micro-Account Kelly Scaler. En micro-cuentas ($13 USD), la fracción base (0.10)
        // produce $0.91 de margen (subcrítico, por debajo de Binance $5 min notional a 5x).
        // Escalamos adaptativamente con la convicción Bayesiana para operar entre $1.15 y $1.80 de margen,
        // dentro del límite seguro del 25% del capital ($2.60).
        // D-641 (completo): el escalador micro de Kelly deja de saltar en $20.
        // D-750 — EL PISO DE VIABILIDAD NO ES UNA FRACCIÓN DE APUESTA.
        //
        // CERT-M5-C02 declaró cerrado el modo de fallo «fracción mínima sin
        // edge», pero lo reintrodujo con otro nombre:
        //
        // ```text
        //   micro_min_viable = (min_notional · 5 / capital).clamp(0, 0,10)
        //   micro_kelly      = (kelly.max(micro_min_viable) · conv)
        //                        .clamp(micro_min_viable.min(0,10), 0,20)
        // ```
        //
        // Con 13 $ y `min_notional = 5`, `5·5/13 = 1,92` se recortaba a 0,10 y
        // ese 0,10 pasaba a ser el LÍMITE INFERIOR del clamp: el 10 % del
        // capital apostado aunque el Kelly medido valiera cero. El `· 5` era
        // además un apalancamiento implícito escrito a mano, y el techo 0,20
        // un literal por encima del tope de ruina del propio sistema.
        //
        // La viabilidad es otra cosa: es el MARGEN mínimo con el que la orden
        // alcanza el nocional mínimo DEL SÍMBOLO al apalancamiento que se va a
        // usar, `margen_min = min_notional / L`. Eso se comprueba más abajo,
        // una vez conocidos `L` y el stop, y si ese mínimo excede lo que el
        // control de ruina permite la orden se RECHAZA (`REJ_VIABILIDAD`): no
        // se infla la apuesta para que quepa.
        //
        // Aquí sólo queda la modulación por convicción, acotada por el tope de
        // ruina —streak-bound + axioma del 25 %—, que es un límite derivado y
        // no el literal 0,20. La convicción tampoco se mide ya contra el par
        // de literales `(confianza − 0,65) · 1,5`: el exceso relevante es el
        // que hay SOBRE LA PUERTA DE CONFIANZA del genoma, que es el punto a
        // partir del cual el sistema considera la señal accionable.
        let conviccion = 1.0
            + (intent.confidence - arena.config.min_confidence_btc.load(Ordering::Relaxed))
                .max(0.0);
        let micro_kelly =
            crate::ruin::clamp_ruin(kelly_adjusted * conviccion, crate::ruin::CONSERVATIVE_Q)
                .max(0.0);
        let kelly_for_scale =
            crate::capital_regime::lerp(kelly_adjusted, micro_kelly, micro_w_alloc);

        let epi_bias = if coin_id < arena.coins.len() {
            arena.coins[coin_id].epigenetic_bias.load(Ordering::Relaxed)
        } else {
            1.0
        };
        let effective_confidence =
            (intent.confidence * epi_bias.clamp(0.60, 1.40)).clamp(0.05, 0.98);
        let raw_exposure = dir * effective_confidence * kelly_for_scale * allocated_capital;
        if !raw_exposure.is_finite() {
            return rej(REJ_INVALID_INPUT);
        }
        if raw_exposure == 0.0 {
            return rej(1);
        }

        // D-748 — GUARD DE CORRELACIÓN QUE MIDE CORRELACIÓN.
        //
        // Antes: se contaban posiciones en la misma dirección y se comparaban
        // con `(global_correlation_threshold · 5).round()`. Multiplicar un
        // coeficiente de correlación por cinco para obtener un número de
        // posiciones es un cambio de unidades inventado, y NINGUNA correlación
        // se medía: dos ALTCOINs gemelas contaban igual que BTC contra un
        // activo descorrelacionado.
        //
        // Ahora se mide de verdad —Pearson sobre los retornos logarítmicos del
        // mid, llevados a una rejilla temporal común desde los anillos de ticks
        // del arena—, el gen recupera su significado literal (umbral de
        // correlación a partir del cual dos posiciones son la MISMA apuesta) y
        // el límite de exposición sale del riesgo medido contra el tope de
        // ruina del sistema, no de un múltiplo.
        let is_long = intent.signal == SignalType::Long;
        let current_cap = arena.unified_capital.load(Ordering::Relaxed);
        let corr_thresh = arena
            .config
            .global_correlation_threshold
            .load(Ordering::Relaxed);
        let mut misma_apuesta = 0usize;
        {
            let ticks_candidata = arena.coins[coin_id]
                .tick_ring
                .snapshot_recent(correlation_guard::MAX_TICKS_MUESTRA);
            // (Ola XLI·C1) MARCHENKO-PASTUR: se recolecta la matriz de
            // correlación del GRUPO (candidata + mismas-dirección abiertas).
            // Si TODO el espectro cabe en la banda de ruido MP (γ = T/N), los
            // pares Pearson altos son RUIDO que parece correlación: el ruido
            // NO VETA y el grupo se cuenta como apuestas independientes. Con
            // modo sistemático real (o muestra insuficiente para afirmar),
            // rige el pairwise D-748 sin cambios.
            let mut grupo_ids = vec![coin_id];
            let mut pares_r: Vec<(usize, usize, f64)> = Vec::new();
            for (otro_id, c) in arena.coins.iter().enumerate() {
                if otro_id == coin_id {
                    continue;
                }
                let pos = &c.positions.position;
                if !pos.is_open() || pos.is_long.load(Ordering::Relaxed) != is_long {
                    continue;
                }
                grupo_ids.push(otro_id);
                let ticks_otro = c
                    .tick_ring
                    .snapshot_recent(correlation_guard::MAX_TICKS_MUESTRA);
                // (Ola XLIII·B) HAYASHI-YOSHIDA primero: los ticks de monedas
                // distintas no comparten reloj y el Pearson en rejilla sufre
                // Epps effect (la correlación decae con la desincronía). HY
                // usa TODOS los solapes sin rejilla; si no hay evidencia HY,
                // fallback al estimador en rejilla existente.
                let r = correlation_guard::hayashi_yoshida_correlation(
                    &ticks_candidata,
                    &ticks_otro,
                )
                .or_else(|| {
                    correlation_guard::correlacion_de_retornos(
                        &ticks_candidata,
                        &ticks_otro,
                        corr_thresh,
                    )
                });
                pares_r.push((0, grupo_ids.len() - 1, r.unwrap_or(f64::NAN)));
                if correlation_guard::CorrelationGuardEngine::es_la_misma_apuesta(r, corr_thresh)
                {
                    misma_apuesta += 1;
                }
            }
            // Una posición ya abierta en la PROPIA moneda es, por definición, la
            // misma apuesta: correlación 1 sin necesidad de medirla.
            let propia = &arena.coins[coin_id].positions.position;
            let propia_abierta =
                propia.is_open() && propia.is_long.load(Ordering::Relaxed) == is_long;
            if propia_abierta {
                misma_apuesta += 1;
            }
            // Veredicto MP sobre el grupo (sólo si hay pares que denoisingar).
            if !pares_r.is_empty() {
                let n = grupo_ids.len();
                let mut corr = vec![vec![0.0f64; n]; n];
                for i in 0..n {
                    corr[i][i] = 1.0;
                }
                for &(i, j, r) in &pares_r {
                    if r.is_finite() {
                        corr[i][j] = r;
                        corr[j][i] = r;
                    }
                }
                if let Some(random_matrix::MppVerdict::AllNoise) =
                    random_matrix::systematic_mode(&corr, correlation_guard::MAX_TICKS_MUESTRA)
                {
                    // Toda la correlación del grupo es compatible con ruido:
                    // sólo la PROPIA moneda (correlación 1 por definición)
                    // sigue contando como misma apuesta.
                    misma_apuesta = if propia_abierta { 1 } else { 0 };
                }
            }
        }
        if correlation_guard::CorrelationGuardEngine::veto_por_exposicion_direccional(
            misma_apuesta,
            arena.riesgo_por_operacion.load(Ordering::Relaxed),
            1.0 - arena.coins[coin_id]
                .metrics
                .win_rate
                .load(Ordering::Relaxed)
                .clamp(0.0, 1.0),
        ) {
            return rej(2);
        }

        // D-750: el spec y el nocional mínimo del símbolo se resolvieron al
        // principio de la evaluación, porque de ellos depende también el
        // régimen de capital. Aquí sólo se usa lo ya resuelto.
        let coin = &arena.coins[coin_id];

        let current_atr = coin.current_atr.load(Ordering::Relaxed);
        let current_price = coin.current_price.load(Ordering::Relaxed);
        if !current_price.is_finite() || current_price <= 0.0 {
            return rej(REJ_INVALID_INPUT);
        }
        let atr_pct = current_atr / current_price;

        let hurst_exponent = coin.hurst_exponent.load(Ordering::Relaxed);
        let vol_mult = if coin_id == 0 {
            arena
                .config
                .btc_volatility_multiplier
                .load(Ordering::Relaxed)
        } else if coin_id == 1 {
            arena
                .config
                .eth_volatility_multiplier
                .load(Ordering::Relaxed)
        } else {
            let eth_mult = arena
                .config
                .eth_volatility_multiplier
                .load(Ordering::Relaxed);
            let btc_price = arena.coins[0]
                .current_price
                .load(Ordering::Relaxed)
                .max(1e-8);
            let btc_atr = arena.coins[0].current_atr.load(Ordering::Relaxed);
            let btc_atr_pct = if btc_atr > 0.0 && btc_price > 1e-6 {
                (btc_atr / btc_price).clamp(0.0005, 0.50)
            } else {
                0.005 // 50 bps baseline default
            };
            let effective_atr_pct = atr_pct.clamp(0.0001, 1.0);
            let relative_vol = (effective_atr_pct / btc_atr_pct.max(0.0005)).clamp(0.5, 3.0);
            eth_mult * relative_vol
        };
        let genomic_leverage_cap = arena.config.global_leverage.load(Ordering::Relaxed);
        if !genomic_leverage_cap.is_finite()
            || genomic_leverage_cap < 1.0
            || max_exchange_leverage < 1.0
        {
            return rej(REJ_INVALID_INPUT);
        }
        let genome_max_leverage = genomic_leverage_cap.min(max_exchange_leverage);

        // U-C — FIX ZOMBIE: coin.scalp/swing ya nadie los escribe (motor
        // unificado -> coin.metrics). Antes el leverage leía un win-rate
        // congelado en el valor inicial para SIEMPRE: el sizing dinámico
        // jamás aprendía de resultados (never-start). Ahora lee la fuente
        // viva única.
        let real_win_rate = coin.metrics.win_rate.load(Ordering::Relaxed);

        let mut dynamic_leverage =
            leverage_matrix::QuantumLeverageMatrix::calculate_dynamic_leverage(
                intent,
                temporal_scale, // D-509: Variedad temporal continua sin colapso discreto
                allocated_capital,
                base_allocated,
                atr_pct,
                vol_mult,
                hurst_exponent,
                profit_factor,
                real_win_rate,
                genome_max_leverage,
                // D-750: el mínimo del símbolo, resuelto una vez al principio
                // de la evaluación, gobierna también el techo de leverage.
                dynamic_min_notional,
                arena,
            );
        // D-730 (DÉCIMA OLA · auditoría integral): EL APALANCAMIENTO SE CUANTIZA
        // DONDE SE DECIDE.
        //
        // `POST /fapi/v1/leverage` sólo acepta ENTEROS, y el ejecutor envía
        // `order.leverage as u32`; el nocional, en cambio, se dimensionaba con el
        // f64 continuo. Con L = 2,9 y 2,40 USD de margen se enviaba un nocional de
        // 6,96 USD que la cuenta, ya a 2x, exige respaldar con 3,48 USD: un 45 %
        // más de margen del presupuestado, rechazo -2019 o margen bloqueado que
        // `used_margin` no registra. El sesgo es sistemático y crece cuanto menor
        // es el apalancamiento, es decir en régimen micro. Cuantizar aquí, ANTES
        // de las comprobaciones de nocional mínimo y de margen, hace que toda la
        // cadena razone con el mismo entero que verá el exchange. La
        // discretización no es una constante arbitraria: la impone el contrato del
        // endpoint.
        if !dynamic_leverage.is_finite() || dynamic_leverage < 1.0 {
            return rej(REJ_INVALID_INPUT);
        }
        dynamic_leverage = dynamic_leverage.min(genome_max_leverage).floor();

        let _maker_fee = arena.config.live_maker_fee.load(Ordering::Relaxed);
        let taker_fee = arena.config.live_taker_fee.load(Ordering::Relaxed);
        // D-01 — FRICCIÓN REAL EN EL EV GATE: antes el gate comparaba contra
        // solo maker+taker (~7bps), pero el motor EJECUTA 2×(taker + slippage)
        // por roundtrip bajo HyperRealistic. El gate subestimaba la fricción
        // 40-70% y certificaba como EV-positivos trades que la física del
        // propio motor volvía negativos. Ahora incluimos la fricción de
        // física que el fill path aplica (entrada taker + salida mayormente
        // taker, cada una con max(impacto_cuadrático+latency, floor)).
        let slip_floor = arena
            .config
            .base_slippage_floor
            .load(Ordering::Relaxed)
            .max(0.00001);
        let lat_ms = arena
            .config
            .latency_penalty_ms
            .load(Ordering::Relaxed)
            .max(0.0);
        // D-635: el umbral de maker-only sale del gen; se elimina el segundo
        // literal (`>= 1000.0`) que lo anulaba y dejaba muerta la ruta maker
        // en toda cuenta pequeña, con independencia de lo que evolucionara.
        let maker_capital_threshold = arena
            .config
            .maker_only_capital_threshold
            .load(Ordering::Relaxed);
        let maker_only = maker_capital_threshold.is_finite()
            && maker_capital_threshold > 0.0
            && allocated_capital >= maker_capital_threshold;

        // D-645 (DÉCIMA OLA) — EL MODELO DE FRICCIÓN COINCIDE CON LA FÍSICA.
        //
        // El modelo asumía UNA pierna maker y UNA taker, mientras el comentario
        // inmediatamente superior reconocía que «el motor EJECUTA 2×(taker +
        // slippage) por roundtrip». El defecto estaba documentado en el propio
        // código y sin corregir: subestimaba (taker − maker) ≈ 3 bps por
        // operación, un 6 % del margen bruto sobre un edge objetivo de 50 bps.
        //
        // La salida es taker salvo que la entrada fuera maker Y el cierre sea
        // por objetivo; como las salidas por stop, trailing, zombi y timeout
        // son TODAS taker, el caso conservador —y el que la física aplica— es
        // taker en ambas piernas.
        // D-645 (revisado): la ENTRADA también es taker. El binario de producción
        // envía la entrada como orden MARKET —la ruta maker está desactivada con
        // `force_maker = false` en god_engine.rs y el iceberg sólo actúa por
        // encima del umbral genómico de nocional—, sin consultar `maker_only`.
        // Modelar la entrada como maker cuando `maker_only` subestimaba la
        // fricción en (taker − maker) precisamente en las cuentas que superan el
        // umbral maker.
        let entry_fee_rate = taker_fee;
        let exit_fee_rate = taker_fee;

        // D-747 — UNA SOLA LEY PARA EL DESLIZAMIENTO POR LATENCIA.
        //
        // Este gate cobraba `atr_pct · (lat / umbral_de_pánico)`: LINEAL en el
        // tiempo y normalizado contra un gen que no es una escala de
        // volatilidad sino el umbral a partir del cual el enlace se considera
        // roto. La física de ejecución cobra, en cambio, la ley de difusión
        // `σ · √(t/τ_ref)`. Dos fórmulas incompatibles para el mismo evento:
        // el gate certificaba como rentables operaciones que la ejecución
        // volvía negativas.
        //
        // Ahora ambos lados hablan de difusión y la fuente es única:
        // `tp_sl::latency_slippage_pct`, con la referencia temporal en la
        // escala a la que se MIDE el ATR (la vela interna de 1 minuto), la
        // misma que ya gobierna la dispersión de TP/SL.
        let latency_slip = crate::tp_sl::latency_slippage_pct(atr_pct, lat_ms);
        let per_side_slip = (slip_floor + latency_slip).clamp(0.0, 0.05);
        let roundtrip_fee = entry_fee_rate + exit_fee_rate + 2.0 * per_side_slip;
        if !roundtrip_fee.is_finite() || roundtrip_fee < 0.0 {
            return rej(REJ_INVALID_INPUT);
        }

        // D-637 (DÉCIMA OLA) — EL GATE EVALÚA EL TRADE QUE SE VA A EJECUTAR.
        //
        // Antes existían DOS cálculos independientes de la misma magnitud: el
        // gate estimaba `expected_win` como `max(tp_base, atr·1,5)` —sin techo—
        // mientras la orden se construía con un TP acotado a 115 bps. Con ATR
        // del 2 % el EV se sobreestimaba 2,61× y con ATR del 5 %, 6,52×. Como
        // la barrera de comisiones sí era real, la condición efectiva sobre el
        // trade real era `EV_real > hurdle/k`: la fricción se desactivaba justo
        // en los regímenes volátiles, donde el deslizamiento es mayor.
        //
        // Ahora ambos caminos llaman a la MISMA función pura con las MISMAS
        // entradas: la identidad es estructural, no disciplinaria.
        let tau_for_sizing = horizon_tau_ms_coin(intent, arena, coin_id);
        // S-7: Hurst DE LA ESCALA OPERADA — hurst_scale_matched es el H(τ)
        // multifractal que el core selecciona por τ dominante; fallback al
        // escalar global si aún no fue escrito (0.0).
        let hurst_for_geometry = {
            let h_scale = arena.coins[coin_id]
                .hurst_scale_matched
                .load(Ordering::Relaxed);
            if h_scale.is_finite() && (0.05..=0.95).contains(&h_scale) {
                h_scale
            } else {
                hurst_exponent
            }
        };
        // D-682 (DÉCIMA OLA): el gate evaluaba `compute_tp_sl` (TP = SL·RR_req)
        // mientras la orden usaba `compute_tp_sl_with_target_rr` (TP = SL·RR
        // genómico, mayor): la identidad que D-637 prometía seguía rota. Ahora
        // se construye UNA vez y la orden reutiliza exactamente lo evaluado.
        let tpsl_gate = crate::tp_sl::compute_tp_sl_with_target_rr(
            crate::tp_sl::TpSlInputs {
                tau_ms: tau_for_sizing,
                atr_ratio: atr_pct,
                hurst: hurst_for_geometry,
                roundtrip_fee,
                sl_atr_multiplier: if coin_id == 0 {
                    arena.config.sl_atr_mult_btc.load(Ordering::Relaxed)
                } else {
                    arena.config.sl_atr_multiplier.load(Ordering::Relaxed)
                },
                // D-754: la σ que el espectro predictivo pronostica PARA ESTE
                // horizonte, si ha demostrado habilidad fuera de muestra. El
                // arena publica ceros mientras no la tenga, y entonces la
                // geometría sigue con la ley de escala sobre el ATR medido.
                sigma_forecast: arena.coins[coin_id].sigma_forecast_at(tau_for_sizing),
            },
            arena.config.tp_rr_ratio_btc.load(Ordering::Relaxed),
        );
        // D-636b & #585: Rechazo Físico Invariante de Suelo Operable (below_tradeable_floor).
        // Política de presupuesto fricción/stop del modelo (FMT-041): no es
        // un teorema universal de EV negativo. Se conserva esta protección.
        // Se rechaza limpiamente con REJ_TP_SL_FLOOR en lugar de inflar artificialmente el stop.
        if tpsl_gate.below_tradeable_floor {
            return rej(REJ_TP_SL_FLOOR);
        }
        // Blindaje Cuántico Micro-Cuenta ($13 USD):
        // Dado el suelo de Binance de $5.00 min notional, el tamaño no puede comprimirse por debajo de ~$5.10.
        // Si el stop difusivo sigma(tau)*k excede 55 bps en régimen micro, la pérdida en dólares violaría el presupuesto
        // de ruina ($0.0280 USD max). Se acota el stop a 55 bps y se preserva el ratio RR >= 2.25 de diseño.
        let (expected_win, expected_loss) = if micro_w_alloc > 0.5 && tpsl_gate.sl_pct > 0.0055 {
            let sl = 0.0055;
            let tp = (sl * tpsl_gate.rr_applied).max(sl * 2.25);
            (tp, sl)
        } else {
            (tpsl_gate.tp_pct, tpsl_gate.sl_pct)
        };

        // FMT-211: resolve once BEFORE EV and reuse these exact prices below.
        // Payouts are signed fractions of entry price, not leveraged returns.
        let final_tp = if intent.tp_price_target > 0.0 {
            intent.tp_price_target
        } else {
            current_price * (1.0 + dir * expected_win)
        };
        let final_sl = if intent.sl_price_target > 0.0 {
            intent.sl_price_target
        } else {
            current_price * (1.0 - dir * expected_loss)
        };
        let expected_win = dir * ((final_tp - current_price) / current_price);
        let expected_loss = dir * ((current_price - final_sl) / current_price);
        if !final_tp.is_finite()
            || final_tp <= 0.0
            || !final_sl.is_finite()
            || final_sl <= 0.0
            || !expected_win.is_finite()
            || expected_win <= 0.0
            || !expected_loss.is_finite()
            || expected_loss <= 0.0
        {
            return rej(REJ_TARGET_GEOMETRY);
        }

        // D-642 (DÉCIMA OLA): la confianza entra tal cual. El suelo `.max(0.51)`
        // falseaba la probabilidad que alimenta a Kelly y al EV, inflando el
        // tamaño de posición precisamente en las señales más débiles.
        let confidence = intent.confidence.clamp(0.0, 1.0);

        // D-641 (DÉCIMA OLA) — SE ELIMINA EL ACANTILADO EN capital = $15.
        //
        // Antes había TRES discontinuidades en el mismo flujo con el mismo
        // umbral literal: confianza (0,66 / 0,62), barrera de comisiones
        // (1,25 / 1,05) y colchón de margen (0,98 / gen). Con $15,00 el sistema
        // se comportaba de un modo y con $15,01 de otro.
        //
        // Lo grave no era la discontinuidad sino que los dos entornos vivían en
        // LADOS OPUESTOS de la frontera: la evolución corre con capital de
        // backtest (rama estándar) y producción con ~$13 (rama micro). Ningún
        // genoma fue jamás evaluado contra los umbrales que lo gobiernan en vivo.
        //
        // La magnitud que de verdad importa no es el capital absoluto sino
        // cuántas operaciones de tamaño mínimo caben en la cuenta.
        // D-641 (completo): `scarcity` pasa a ser el peso del régimen de capital
        // compartido. La versión anterior, 1/(1+N/3), aplicaba sólo el 54 % del
        // régimen micro a una cuenta de $13 y cambiaba el comportamiento que se
        // había diseñado para ella; el peso compartido vale exactamente 1 ahí,
        // de modo que la barrera de comisiones vuelve a 1,25 y el colchón a 0,98.
        let scarcity = micro_w_alloc;
        let base_conf_gate = arena
            .config
            .min_confidence_btc
            .load(Ordering::Relaxed)
            .clamp(0.05, 0.95);

        // Heuristic spectral modulation of an admission threshold. Coherence
        // and entropy are not calibrated win probabilities or evidence of
        // physical/quantum certainty. The .05/.85 branches remain policy
        // discontinuities pending out-of-sample validation (audit XXV).
        let is_long = intent.signal == SignalType::Long;
        let raw_coh = if coin_id < arena.coins.len() {
            arena.coins[coin_id]
                .spectral_coherence
                .load(Ordering::Relaxed)
        } else {
            0.0
        };
        let spec_coh = if is_long { raw_coh } else { -raw_coh };
        let spec_ent = if coin_id < arena.coins.len() {
            arena.coins[coin_id]
                .spectral_entropy
                .load(Ordering::Relaxed)
        } else {
            1.0
        };
        let coh_benefit = if spec_coh > 0.05 && spec_ent < 0.85 {
            (spec_coh * (1.0 - spec_ent * 0.5)).clamp(0.0, 0.80)
        } else {
            0.0
        };
        let base_micro_ratio = 0.66 / 0.62;
        let effective_micro_ratio = base_micro_ratio - (base_micro_ratio - 1.0) * coh_benefit;

        let now_ts = if coin_id < arena.coins.len() {
            arena.coins[coin_id]
                .last_tick_timestamp_ms
                .load(Ordering::Relaxed)
        } else {
            0
        };
        let epi_thresh = if coin_id < arena.coins.len() {
            arena.coins[coin_id].get_active_epigenetic_threshold(now_ts)
        } else {
            1.0
        };
        let min_required_confidence = (crate::capital_regime::lerp(
            base_conf_gate,
            base_conf_gate * effective_micro_ratio,
            scarcity,
        ) * epi_thresh.clamp(0.75, 1.35))
        .clamp(0.05, 0.98);
        if confidence < min_required_confidence {
            return rej(REJ_CONFIDENCE);
        }

        // D-751 — LA CONFIANZA DE UNA RAMA NO ES P(GANAR).
        //
        // El valor esperado se calculaba como
        // `ev = confianza · TP − (1 − confianza) · SL`, tratando la puntuación
        // de convicción de la señal como si fuera una probabilidad calibrada.
        // No lo es: es una puntuación heurística cuya escala ni siquiera está
        // acotada a la frecuencia con la que esas señales ganan —el propio
        // núcleo lo documenta en D-619/D-690, y por eso mantiene un calibrador
        // aparte—. Con una puntuación típica de 0,70 el EV salía positivo por
        // construcción aunque la rama ganase el 40 % de las veces.
        //
        // Orden de preferencia, de más a menos informativo:
        //   1. la probabilidad CALIBRADA con resultados reales que el núcleo
        //      adjunta a la intención (`win_probability`, D-690);
        //   2. la frecuencia OBSERVADA de la moneda con su cota inferior
        //      bayesiana (posterior de Jeffreys, el mismo z del sistema);
        //   3. ninguna: entonces el EV no puede evaluarse y la entrada se
        //      rechaza POR FALTA DE EVIDENCIA, no con un número inventado.
        // D-751b (fusión PR #5 + FMT): ARRANQUE FRÍO TOTAL. D-750 financia la
        // sonda mínima pero D-751 rechazaba el EV «sin evidencia»: sin
        // trades jamás hay evidencia — deadlock que mató el oráculo genético
        // (T-1: 0/144 tras la fusión). Unificación doctrinal: en arranque
        // frío TOTAL (cero operaciones del símbolo y sin probabilidad
        // calibrada adjunta) el EV no se inventa ni bloquea: la orden sigue
        // como SONDA D-750, cuyo sizing mínimo ya está acotado por el control
        // de ruina. Con evidencia (aunque sea una) rige D-751 íntegro.
        let arranque_frio_total = arena.coins[coin_id]
            .metrics
            .trade_count
            .load(Ordering::Relaxed)
            == 0;
        let p_ganar = if intent.win_probability.is_finite()
            && intent.win_probability > 0.0
            && intent.win_probability < 1.0
        {
            Some(intent.win_probability)
        } else {
            match crate::evidence::win_rate_lcb(
                arena.coins[coin_id]
                    .metrics
                    .win_rate
                    .load(Ordering::Relaxed),
                arena.coins[coin_id]
                    .metrics
                    .trade_count
                    .load(Ordering::Relaxed) as f64,
            ) {
                Some(p) => Some(p),
                None if arranque_frio_total => None,
                None => return rej(REJ_SIN_EVIDENCIA),
            }
        };
        let expected_value_pct = match p_ganar {
            Some(p) => (p * expected_win) - ((1.0 - p) * expected_loss),
            // Sonda en arranque frío: EV neutral-conservador al 50 %, sólo
            // para atravesar los literales de abajo; el veto EV real no
            // aplica a la sonda (ver abajo).
            None => 0.5 * expected_win - 0.5 * expected_loss,
        };

        // D-641: misma transición continua para la barrera de comisiones.
        // Con la cuenta al límite se exige hasta un 25 % de margen sobre la
        // fricción; con holgura, un 5 %. Sin escalón.
        let min_ev_mult = 1.05 + 0.20 * scarcity;
        let ev_fee_multiplier = arena
            .config
            .ev_fee_multiplier
            .load(Ordering::Relaxed)
            .clamp(min_ev_mult, 1.80);
        if !expected_value_pct.is_finite() || !ev_fee_multiplier.is_finite() {
            return rej(REJ_INVALID_INPUT);
        }
        if p_ganar.is_some() && expected_value_pct <= (roundtrip_fee * ev_fee_multiplier) {
            return rej(4);
        }

        let max_acceptable_fee_pct = arena.config.max_fee_pct.load(Ordering::Relaxed);
        if !max_acceptable_fee_pct.is_finite() || max_acceptable_fee_pct < 0.0 {
            return rej(REJ_INVALID_INPUT);
        }
        // Cost per notional times L is cost per unit of allocated margin.
        // Preserve the existing micro policy, but enforce it on every final
        // order, not just branches that rescue a minimum notional (FMT-217).
        let max_fee_limit =
            crate::capital_regime::lerp(max_acceptable_fee_pct, 0.035, micro_w_alloc);
        let dynamic_min_notional = crate::capital_regime::effective_min_notional(spec.min_notional);
        let _max_safe_leverage = if roundtrip_fee > 0.0 {
            max_acceptable_fee_pct / roundtrip_fee
        } else {
            100.0
        };

        // D-750 — PISO DE VIABILIDAD DERIVADO, Y RECHAZO SI NO CABE.
        //
        // La orden más pequeña que el símbolo acepta tiene nocional
        // `min_notional` y, con el stop de esta geometría, arriesga
        // `min_notional · SL` dólares, es decir una fracción
        // `min_notional · SL / capital` de la cuenta. Nótese que el
        // apalancamiento NO aparece: reparte el mismo nocional entre margen y
        // préstamo, pero no cambia lo que se pierde si el stop se toca — por eso
        // «subir el apalancamiento para que quepa» no hace viable nada.
        //
        // Ese riesgo mínimo se compara con el ÚNICO tope de riesgo por evento
        // del sistema (`ruin::clamp_ruin`: streak-bound + axioma del 25 %). Si
        // lo excede, ni la orden mínima del exchange es compatible con la
        // supervivencia de esta cuenta: se RECHAZA. Antes, en cambio, el sizing
        // inflaba la fracción apostada hasta el 10 % del capital para alcanzar
        // el mínimo, que es exactamente el modo de fallo que CERT-M5-C02
        // decía haber cerrado.
        let q_para_ruina = {
            let wr = arena.coins[coin_id]
                .metrics
                .win_rate
                .load(Ordering::Relaxed);
            if wr > 0.0 && wr < 1.0 {
                1.0 - wr
            } else {
                crate::ruin::CONSERVATIVE_Q
            }
        };
        let tope_riesgo_evento = crate::ruin::clamp_ruin(1.0, q_para_ruina);
        if !crate::capital_regime::orden_viable(
            dynamic_min_notional,
            expected_loss,
            allocated_capital,
            tope_riesgo_evento,
        ) {
            return rej(REJ_VIABILIDAD);
        }

        let bounded_exposure = raw_exposure.clamp(-allocated_capital, allocated_capital);
        let mut final_margin = bounded_exposure.abs();
        let margin_cushion_pct = arena.config.margin_cushion_pct.load(Ordering::Relaxed);
        // D-641: el colchón de margen sale SIEMPRE del gen; la escasez sólo lo
        // relaja de forma continua hacia el máximo operativo. Antes el literal
        // 0,98 anulaba el gen precisamente en el entorno de capital real, de
        // modo que un gen evolucionado para la prudencia quedaba inerte en
        // producción.
        // D-634/D-635: la fórmula vive en `capital_regime::margin_cushion`,
        // compartida con la comprobación de margen libre del núcleo.
        let safe_cushion = crate::capital_regime::margin_cushion(margin_cushion_pct, scarcity);

        // D-130: Evaluar si el notional real de la orden (final_margin * dynamic_leverage) cumple con el mínimo
        if final_margin > 0.0 && final_margin * dynamic_leverage < dynamic_min_notional {
            // FIX min_notional (diag R4): el leverage necesario para alcanzar
            // el notional mínimo se calcula sobre el MARGEN DEL TRADE, no
            // sobre el capital total. La fórmula anterior
            // (min_notional/(allocated*cushion)) producía leverage < 1 en
            // micro-cuenta y garantizaba el rechazo posterior: era la causa
            // de 0 trades en la certificación 30d (min_notional ~500-640
            // rechazos/día con señales sanas de conf 0.7+). El fee_impact
            // check de abajo sigue limitando el costo.
            let candidate_leverage = (dynamic_min_notional / final_margin.max(0.01)) * 1.02;
            // D-641 (completo): el techo micro de apalancamiento es continuo en
            // la confianza (antes escalones en 0,70 y 0,75) y en el capital
            // (antes escalón de ~5× a 50× en $20). Interpolación geométrica: el
            // punto medio natural entre 5× y 50× es ~16×, no 27,5×.
            let conf_t = ((intent.confidence - 0.65) / 0.10).clamp(0.0, 1.0);
            let micro_lev_cap = 5.0 + 1.5 * conf_t * conf_t * (3.0 - 2.0 * conf_t);
            let max_lev_cap = crate::capital_regime::log_lerp(50.0, micro_lev_cap, micro_w_alloc);
            dynamic_leverage = candidate_leverage
                .min(genome_max_leverage)
                .min(max_lev_cap)
                .floor();
        }

        // FASE 3 FIX: Micro-Account Notional Safety
        // Sumamos diez centavos de dólar (+0.1) al min notional para evitar rechazos
        // por pérdida de precisión IEEE-754 en multiplicaciones de apalancamiento
        let safe_min_notional = dynamic_min_notional + 0.1;
        // D-750: el margen mínimo viable sale de la MISMA función que la
        // comprobación de viabilidad de más arriba: `min_notional / L`.
        let required_margin_for_min_notional =
            crate::capital_regime::margen_minimo_viable(safe_min_notional, dynamic_leverage);
        if final_margin < required_margin_for_min_notional {
            // FIX FP (diag R4): lev x (min_notional/lev) puede dar
            // 5.0999... < 5.1 en punto flotante y rechazar la orden en el
            // borde EXACTO — la causa terminal de los 0 trades. Épsilon
            // relativo de 5 bps de margen cierra la frontera.
            final_margin = required_margin_for_min_notional * 1.0005;
        }

        let (meets_min_notional, _) =
            guard::enforce_minimum_notional(final_margin, safe_min_notional, dynamic_leverage);
        if !meets_min_notional {
            if REJECT_COUNTERS[6].load(std::sync::atomic::Ordering::Relaxed) % 200 == 0 {
                println!(
                    "🔍 [REJ6] margin={:.4} lev={:.4} min_notional={:.4} allocated={:.4} kelly={:.4}",
                    final_margin, dynamic_leverage, safe_min_notional, allocated_capital, kelly_fraction
                );
            }
            return rej(6);
        }

        // D-641 (completo): límite de margen por operación continuo. En régimen
        // micro pleno conserva el 20–25 % diseñado para cubrir el notional mínimo
        // con apalancamiento prudente; en estándar, el colchón genómico.
        let micro_safe_limit = (allocated_capital * 0.25).clamp(1.20, 2.60);
        let standard_safe_limit = (allocated_capital * safe_cushion).min(current_cap * 0.90);
        let safe_limit =
            crate::capital_regime::lerp(standard_safe_limit, micro_safe_limit, micro_w_alloc);
        if final_margin > safe_limit {
            final_margin = safe_limit;
            if final_margin > 0.0 && final_margin * dynamic_leverage < safe_min_notional {
                let re_lev = (safe_min_notional / final_margin) * 1.01;
                // Quantize BEFORE checking feasibility: all downstream
                // notional/margin calculations must see the exchange integer.
                let capped_leverage = re_lev.min(genome_max_leverage).min(50.0).floor();
                let fee_impact = roundtrip_fee * capped_leverage;
                if fee_impact <= max_fee_limit {
                    dynamic_leverage = capped_leverage;
                }
            }
        }
        // Terminal invariants after every size/leverage adjustment. No
        // fallback to 1x can silently change the validated action here.
        if !dynamic_leverage.is_finite()
            || dynamic_leverage < 1.0
            || dynamic_leverage.fract() != 0.0
            || dynamic_leverage > genome_max_leverage
        {
            return rej(REJ_INVALID_INPUT);
        }
        let final_fee_impact = roundtrip_fee * dynamic_leverage;
        if !final_fee_impact.is_finite() || final_fee_impact > max_fee_limit {
            return rej(5);
        }
        // (fusión PR #5: la viabilidad de margen mínimo ya se exigió arriba con
        // rej(6); el duplicado del hunk se elimina)

        let orchestrator = orchestrator::PortfolioOrchestrator::new(arena);
        let raw_regime = arena.market_regime.load(Ordering::Relaxed);
        let regime = crate::regime::MarketRegime::from(raw_regime);

        if !orchestrator.allow_trade(
            bounded_exposure > 0.0,
            final_margin,
            regime,
            dynamic_min_notional,
        ) {
            return rej(8);
        }

        // Reuse the prices evaluated above, including explicit intent targets.

        let safe_tp = if final_tp.is_finite() && final_tp > 0.0 {
            final_tp
        } else {
            0.0
        };
        let safe_sl = if final_sl.is_finite() && final_sl > 0.0 {
            final_sl
        } else {
            0.0
        };
        let safe_vol = if final_margin.is_finite() && final_margin > 0.0 {
            final_margin
        } else {
            0.0
        };
        let safe_lev = dynamic_leverage;

        if safe_vol <= 0.0
            || (intent.signal != SignalType::Flat && (safe_tp <= 0.0 || safe_sl <= 0.0))
        {
            return rej(REJ_INVALID_INPUT);
        }

        // D-744: el RIESGO REALMENTE TOMADO por esta orden —lo que se pierde
        // si su stop se toca, en fracción del capital— alimenta la media móvil
        // que convierte una caída observada en evidencia. Sin esta medida, el
        // cortacircuitos de drawdown es una opinión sobre un número inventado.
        // D-744c (auditoría PR #5): se registra DESPUÉS del último rechazo de
        // esta función; antes también entraban órdenes que aquí mismo se
        // rechazaban por geometría inválida.
        let sl_pct = tpsl_gate.sl_pct;
        if sl_pct > 0.0 && current_cap > 0.0 {
            let riesgo = (safe_vol * safe_lev * sl_pct) / current_cap;
            let previo = arena.riesgo_por_operacion.load(Ordering::Relaxed);
            arena.riesgo_por_operacion.store(
                crate::drawdown::actualizar_riesgo_ewma(
                    previo,
                    riesgo.clamp(0.0, 1.0),
                    crate::drawdown::TRADE_HORIZON / 10.0,
                ),
                Ordering::Relaxed,
            );
        }

        ValidatedOrder {
            signal: intent.signal,
            volume_usd: safe_vol,
            leverage: safe_lev,
            maker_only,
            tp_target: safe_tp,
            sl_target: safe_sl,
            fee_buffer_multiplier: ev_fee_multiplier,
            // D-745: la orden se lleva el horizonte con el que fue dimensionada.
            tau_ms: tau_for_sizing,
        }
    }
}

/// D-637 — HORIZONTE OPERATIVO EN MILISEGUNDOS.
///
/// D-638b (DÉCIMA OLA): la versión anterior interpolaba sobre los EXTREMOS
/// del espectro. Al ampliarse éste a 1 ns–146 años, `temporal_scale = 0,05`
/// producía ~9 ns y 0,95 ~17 años: el gate de TP/SL rechazaba por «no
/// operable» o dimensionaba stops de décadas. Ahora delega en
/// `temporal_spectrum::operating_tau_ms`, la fuente única.
///
/// Puente temporal mientras `SignalIntent` conserva el enum `TradeHorizon`
/// (D-602). El orden de preferencia respeta la jerarquía correcta:
///   1. la duración esperada que la señal declara — información real;
///   2. el eje temporal continuo del arena, mapeado log-linealmente sobre el
///      espectro, cuando la señal no declara duración.
///
/// En ningún caso se consulta la etiqueta discreta para elegir parámetros:
/// ésta sólo desempata el extremo del continuo cuando no hay nada mejor.
fn horizon_tau_ms(intent: &SignalIntent, arena: &GlobalArena) -> f64 {
    horizon_tau_ms_coin(intent, arena, usize::MAX)
}

/// D-745 — EL HORIZONTE DE LA ORDEN ES EL QUE EL MERCADO MUESTRA, NO UN GEN.
///
/// Si la señal declara una duración, manda ella: es información de la rama que
/// la produjo. Si no, el respaldo era el gen estático `temporal_scale` —el
/// mismo para las 30 monedas y para todo el mes—, mientras el núcleo abría la
/// posición con la τ DOMINANTE medida del espectro de ESA moneda. Dimensionar
/// a 19 minutos y gestionar a 30 segundos es el defecto, no el gen. Ahora el
/// respaldo es esa misma τ medida, y el gen sólo entra mientras el espectro no
/// ha arrancado (arranque en frío).
fn horizon_tau_ms_coin(intent: &SignalIntent, arena: &GlobalArena, coin_id: usize) -> f64 {
    let _ = intent.horizon;
    if intent.expected_duration_ms > 0 {
        return intent.expected_duration_ms as f64;
    }
    if coin_id < arena.coins.len() {
        let medida = arena.coins[coin_id].dominant_tau_ms.load(Ordering::Relaxed);
        if medida.is_finite() && medida > 0.0 {
            return medida.clamp(
                quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
            );
        }
    }
    let s = arena
        .config
        .temporal_scale
        .load(Ordering::Relaxed)
        .clamp(0.0, 1.0);
    quantum_arena::temporal_spectrum::operating_tau_ms(intent.expected_duration_ms, s)
}

#[cfg(test)]
mod reject_direction_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn rechazos_se_atribuyen_a_la_direccion_evaluada() {
        let long0 = REJECT_COUNTERS_DIR[0][REJ_CONFIDENCE].load(Ordering::Relaxed);
        let short0 = REJECT_COUNTERS_DIR[1][REJ_CONFIDENCE].load(Ordering::Relaxed);
        set_reject_direction(SignalType::Long);
        let _ = rej(REJ_CONFIDENCE);
        set_reject_direction(SignalType::Short);
        let _ = rej(REJ_CONFIDENCE);
        let _ = rej(REJ_CONFIDENCE);
        set_reject_direction(SignalType::Flat);
        let _ = rej(REJ_CONFIDENCE);
        assert!(REJECT_COUNTERS_DIR[0][REJ_CONFIDENCE].load(Ordering::Relaxed) >= long0 + 1);
        assert!(REJECT_COUNTERS_DIR[1][REJ_CONFIDENCE].load(Ordering::Relaxed) >= short0 + 2);
        assert!(reject_report().contains("confianza="));
    }
}
