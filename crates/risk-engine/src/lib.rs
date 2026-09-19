pub mod capital_compounder;
pub mod capital_regime;
pub mod correlation_guard;
pub mod epigenetic_capital_alloc;
pub mod epigenetic_fitness_landscape;
pub mod guard;
pub mod kelly;
pub mod kelly_envelope;
pub mod leverage_matrix;
pub mod orchestrator;
pub mod regime;
pub mod ruin;
pub mod tp_sl;

pub use kelly_envelope::{EdgePosterior, RiskEnvelope, SURVIVAL_FLOOR, TRADE_HORIZON};

use quantum_arena::GlobalArena;

use signal_engine::{SignalIntent, SignalType, TradeHorizon};
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
}

impl ValidatedOrder {
    pub fn rejected() -> Self {
        Self {
            signal: SignalType::Flat,
            volume_usd: 0.0,
            leverage: 1.0,
            maker_only: false,
            tp_target: 0.0,
            sl_target: 0.0,
            fee_buffer_multiplier: 1.01,
        }
    }
}

// Symbol constraints and size limits have been removed to allow purely dynamic and infinite asset discovery.
// The engine now strictly relies on mathematical limits derived from Kelly and margin constraints.

/// DIAGNÓSTICO SIGNAL-PATH: contadores de rechazo por compuerta de
/// evaluate_single_intent. Índices:
/// 0=flat/coin 1=exposure0 2=correlación 3=spec 4=EV 5=fee_impact
/// 6=min_notional 7=margen_insuf 8=orchestrator 9=otros
/// 10=drawdown 11=suelo TP/SL 12=confianza. Antes el drawdown compartía el
/// índice 2 con la correlación, y el suelo TP/SL y la confianza el 4 con el EV:
/// la telemetría no podía decir qué compuerta rechazaba.
use std::sync::atomic::AtomicU64;
pub const REJECT_SLOTS: usize = 13;
pub const REJ_DRAWDOWN: usize = 10;
pub const REJ_TP_SL_FLOOR: usize = 11;
pub const REJ_CONFIDENCE: usize = 12;

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
        if !current_capital.is_finite() || current_capital <= 0.0 {
            return ValidatedOrder::rejected();
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
        let max_dd = crate::capital_regime::lerp(
            arena.config.global_max_drawdown.load(Ordering::Relaxed),
            0.85,
            micro_w,
        );
        if self.peak_capital > 0.0 && max_dd > 0.0 && max_dd < 1.0 {
            let dd = (self.peak_capital - current_capital) / self.peak_capital;
            if dd >= max_dd {
                return rej(REJ_DRAWDOWN);
            }
        }

        let base_capital = arena.config.base_capital.load(Ordering::Relaxed);
        let pf = arena.coins[coin_id]
            .metrics
            .profit_factor
            .load(Ordering::Relaxed);
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
        let pf = arena.coins[coin_id]
            .metrics
            .profit_factor
            .load(Ordering::Relaxed);
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
        let kelly_frac = if raw_kelly <= 0.0 {
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
        // D-641 (completo): peso del régimen de capital micro para TODA la
        // evaluación. Un único valor, calculado una vez, del que derivan todas
        // las transiciones que antes eran escalones en $15 y $20.
        let micro_w_alloc = crate::capital_regime::micro_weight(
            allocated_capital,
            arena.config.min_notional.load(Ordering::Relaxed),
        );

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
        let tau_ms = horizon_tau_ms(intent, arena);
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
        // CERT-M5-C02: el PISO anterior `.max(0.12)` y `.clamp(0.10, 0.20)`
        // forzaba 10-20% de fracción Kelly SIN EDGE en la cuenta demo ($13).
        // Ahora el piso sólo existe para VIABILIDAD de notional (no como
        // fracción de apuesta): el mínimo es el necesario para alcanzar
        // el min_notional, jamás un % fijo del capital.
        let micro_min_viable = if allocated_capital > 0.0 {
            // Fracción mínima para que margin = notional/leverage ≥
            // min_notional en micro-cuenta: ~(min_notional × lev) / capital
            let mn = arena.config.min_notional.load(Ordering::Relaxed).max(1.0);
            (mn * 5.0 / allocated_capital).clamp(0.0, 0.10)
        } else {
            0.0
        };
        let micro_kelly = (kelly_adjusted.max(micro_min_viable)
            * (1.0 + (intent.confidence - 0.65).max(0.0) * 1.5))
            .clamp(micro_min_viable.min(0.10), 0.20);
        // CERT-M5-H03: el escalador micro tampoco escapa al tope de ruina.
        // El piso de viabilidad (micro_min_viable) puede EXCEDER el cap cuando
        // la cuenta es diminuta vs min_notional: en ese caso la orden es
        // inviable-by-design (el host la vetará por margen) — preservamos el
        // cap y no la fracción inflada.
        let micro_kelly = crate::ruin::clamp_ruin(micro_kelly, crate::ruin::CONSERVATIVE_Q).max(0.0);
        let kelly_for_scale =
            crate::capital_regime::lerp(kelly_adjusted, micro_kelly, micro_w_alloc);

        let raw_exposure = dir * intent.confidence * kelly_for_scale * allocated_capital;
        if raw_exposure == 0.0 {
            return rej(1);
        }

        // FASE 16 & BUG-578: Correlation Guard (Continuous Universal)
        let is_long = intent.signal == SignalType::Long;
        let mut same_dir_count = 0;
        for c in arena.coins.iter() {
            let pos = &c.positions.position;
            if pos.is_open() && (pos.is_long.load(Ordering::Relaxed) == is_long) {
                same_dir_count += 1;
            }
        }
        let corr_thresh = arena
            .config
            .global_correlation_threshold
            .load(Ordering::Relaxed);
        let max_allowed_cluster = (corr_thresh * 5.0).round() as usize;
        let current_cap = arena.unified_capital.load(Ordering::Relaxed);
        // D-401: Desasfixia multiactivo para micro-cuentas ($13 USD) - permite hasta 2 micro-posiciones continuas
        if correlation_guard::CorrelationGuardEngine::is_continuous_correlation_vetoed(
            same_dir_count,
            current_cap,
            arena.config.min_notional.load(Ordering::Relaxed),
            max_allowed_cluster.max(2),
        ) {
            return rej(2);
        }

        let coin = &arena.coins[coin_id];
        let spec = match quantum_arena::symbol_registry::try_spec(coin_id) {
            Some(s) => s,
            None => return rej(3),
        };
        let max_exchange_leverage = spec.max_leverage as f64;

        let current_atr = coin.current_atr.load(Ordering::Relaxed);
        let current_price = coin.current_price.load(Ordering::Relaxed).max(1e-8);
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
        let genome_max_leverage = arena
            .config
            .global_leverage
            .load(Ordering::Relaxed)
            .min(max_exchange_leverage);

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
                arena,
            );

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

        // La normalización de la latencia deja de ser un literal: se compara
        // contra el umbral de pánico de latencia, que es el gen que define
        // qué cuenta como «lento» para este sistema.
        let latency_ref_ms = arena
            .config
            .latency_ms_panic_threshold
            .load(Ordering::Relaxed)
            .clamp(10.0, 5_000.0);
        let latency_slip = atr_pct * (lat_ms / latency_ref_ms);
        let per_side_slip = (slip_floor + latency_slip).clamp(0.0, 0.05);
        let roundtrip_fee = entry_fee_rate + exit_fee_rate + 2.0 * per_side_slip;

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
        let tau_for_sizing = horizon_tau_ms(intent, arena);
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
                sl_atr_multiplier: arena
                    .config
                    .sl_atr_multiplier
                    .load(Ordering::Relaxed),
            },
            // El RR genómico puede ser MÁS ambicioso que el mínimo exigido por
            // la fricción, nunca menor.
            arena.config.tp_rr_ratio_btc.load(Ordering::Relaxed),
        );
        // Horizonte no operable: la dispersión esperada a esa tau no cubre la
        // fricción. Se RECHAZA en lugar de acotar y fingir que es viable.
        if tpsl_gate.below_tradeable_floor {
            return rej(REJ_TP_SL_FLOOR);
        }
        let expected_win = tpsl_gate.tp_pct;
        let expected_loss = tpsl_gate.sl_pct;

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
        // Endurecimiento micro del gate de confianza en la proporción que fijaba
        // la calibración original (0,66 frente a 0,62), aplicada sobre el gen.
        let min_required_confidence =
            crate::capital_regime::lerp(base_conf_gate, base_conf_gate * (0.66 / 0.62), scarcity)
                .clamp(0.05, 0.98);
        if confidence < min_required_confidence {
            return rej(REJ_CONFIDENCE);
        }

        let expected_value_pct = (confidence * expected_win) - ((1.0 - confidence) * expected_loss);

        // D-641: misma transición continua para la barrera de comisiones.
        // Con la cuenta al límite se exige hasta un 25 % de margen sobre la
        // fricción; con holgura, un 5 %. Sin escalón.
        let min_ev_mult = 1.05 + 0.20 * scarcity;
        let ev_fee_multiplier = arena
            .config
            .ev_fee_multiplier
            .load(Ordering::Relaxed)
            .clamp(min_ev_mult, 1.80);
        if expected_value_pct <= (roundtrip_fee * ev_fee_multiplier) {
            return rej(4);
        }

        let max_acceptable_fee_pct = arena.config.max_fee_pct.load(Ordering::Relaxed);
        let _max_safe_leverage = if roundtrip_fee > 0.0 {
            max_acceptable_fee_pct / roundtrip_fee
        } else {
            100.0
        };
        let dynamic_min_notional = crate::capital_regime::effective_min_notional(spec.min_notional);

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
            // D-641 (completo): tolerancia de impacto de comisión continua.
            let max_fee_limit =
                crate::capital_regime::lerp(max_acceptable_fee_pct, 0.035, micro_w_alloc);
            let fee_impact_pct = roundtrip_fee * candidate_leverage;
            if fee_impact_pct > max_fee_limit {
                if arena.tick_counter.load(Ordering::Relaxed) % 100_000 == 0 {
                    println!(
                        "🔍 [RISK REJECT] FEE_IMPACT: fee_impact={:.6} > limit={:.6}",
                        fee_impact_pct, max_fee_limit
                    );
                }
                return rej(5);
            }
            // D-641 (completo): el techo micro de apalancamiento es continuo en
            // la confianza (antes escalones en 0,70 y 0,75) y en el capital
            // (antes escalón de ~5× a 50× en $20). Interpolación geométrica: el
            // punto medio natural entre 5× y 50× es ~16×, no 27,5×.
            let conf_t = ((intent.confidence - 0.65) / 0.10).clamp(0.0, 1.0);
            let micro_lev_cap = 5.0 + 1.5 * conf_t * conf_t * (3.0 - 2.0 * conf_t);
            let max_lev_cap =
                crate::capital_regime::log_lerp(50.0, micro_lev_cap, micro_w_alloc);
            dynamic_leverage = candidate_leverage
                .min(max_exchange_leverage)
                .min(max_lev_cap);
        }

        // FASE 3 FIX: Micro-Account Notional Safety
        // Sumamos un centavo de dólar (+0.1) al min notional para evitar rechazos
        // por pérdida de precisión IEEE-754 en multiplicaciones de apalancamiento
        let safe_min_notional = dynamic_min_notional + 0.1;
        let required_margin_for_min_notional = safe_min_notional / dynamic_leverage;
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
                let fee_impact = roundtrip_fee * re_lev;
                let max_fee_lim =
                    crate::capital_regime::lerp(max_acceptable_fee_pct, 0.035, micro_w_alloc);
                if fee_impact <= max_fee_lim {
                    dynamic_leverage = re_lev.min(max_exchange_leverage).min(50.0);
                }
            }
        }
        let required_margin_for_min_notional = safe_min_notional / dynamic_leverage;
        if final_margin < required_margin_for_min_notional {
            return rej(7);
        }

        let orchestrator = orchestrator::PortfolioOrchestrator::new(arena);
        let raw_regime = arena.market_regime.load(Ordering::Relaxed);
        let regime = crate::regime::MarketRegime::from(raw_regime);

        if !orchestrator.allow_trade(bounded_exposure > 0.0, final_margin, regime) {
            return rej(8);
        }

        // D-637/D-638/D-639/D-640 — LA ORDEN USA LA MISMA FUENTE QUE EL GATE.
        //
        // Sustituye a dos `match intent.horizon` encadenados (uno de los
        // cuales descartaba su propio SL con `_sl_base`), al piso difusivo que
        // se anulaba con el `clamp` que le seguía, y a la banda literal que
        // confinaba el stop entre 40 y 60 bps con independencia de la
        // volatilidad, del horizonte y de los genes.
        // D-682: la orden usa exactamente el TP/SL que el gate evaluó.
        let tpsl = tpsl_gate;
        let sl_pct = tpsl.sl_pct;
        let tp_pct = tpsl.tp_pct;

        let final_sl = if intent.sl_price_target > 0.0 {
            intent.sl_price_target
        } else if dir > 0.0 {
            current_price * (1.0 - sl_pct)
        } else {
            current_price * (1.0 + sl_pct)
        };

        let final_tp = if intent.tp_price_target > 0.0 {
            intent.tp_price_target
        } else if dir > 0.0 {
            current_price * (1.0 + tp_pct)
        } else {
            current_price * (1.0 - tp_pct)
        };

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
        let safe_lev = if dynamic_leverage.is_finite() && dynamic_leverage >= 1.0 {
            dynamic_leverage
        } else {
            1.0
        };

        if safe_vol <= 0.0
            || (intent.signal != SignalType::Flat && (safe_tp <= 0.0 || safe_sl <= 0.0))
        {
            return ValidatedOrder::rejected();
        }

        ValidatedOrder {
            signal: intent.signal,
            volume_usd: safe_vol,
            leverage: safe_lev,
            maker_only,
            tp_target: safe_tp,
            sl_target: safe_sl,
            fee_buffer_multiplier: ev_fee_multiplier,
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
    // U-6: el motor continuo sólo produce TradeHorizon::Continuous — el eje
    // temporal es el `temporal_scale` del arena (log-lineal sobre el espectro).
    let _ = intent.horizon;
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
