pub mod copulas_store;
pub mod capital_regime;
pub mod cramer_lundberg;
pub mod correlation_guard;
pub mod evidence;
pub mod freno_estrategia;
pub mod guard;
pub mod hodge;
pub mod kelly;
pub mod kelly_envelope;
pub mod leverage_matrix;
pub mod orchestrator;
pub mod random_matrix;
pub mod regime;
pub mod riemann_zeta_gue;
pub mod drawdown;
pub mod envio;
pub mod ruin;
pub mod sonda;
pub mod tp_sl;
pub mod veto_registry;
pub mod selection_stats;
pub mod ville_e_process;

pub use capital_regime::{resolve_micro_max_fee_pct, DEFAULT_MICRO_MAX_FEE_PCT};
pub use riemann_zeta_gue::{
    hermite_smoothstep_c1, montgomery_odlyzko_pair_correlation, wigner_gue_density,
    RiemannZetaGueEngine, RiemannZetaGueState, GUE_RIEMANN_GAP_RATIO_MEAN,
    MAX_RIEMANN_GUE_LEVELS, POISSON_GAP_RATIO_MEAN,
};
pub use evidence::{
    profit_factor_lcb, resolve_bayes_prior, win_rate_hierarchical_lcb, win_rate_lcb,
    DEFAULT_BAYES_PRIOR_MASS, DEFAULT_BAYES_PRIOR_WIN_RATE,
};
pub use kelly_envelope::{EdgePosterior, RiskEnvelope, SURVIVAL_FLOOR, TRADE_HORIZON};
pub use sonda::{
    admite_sonda, admite_sonda_con_fase, en_fase_de_sonda, resolve_max_cierres_sonda,
    sprt_bernoulli, SprtDecision, CIERRES_DE_SONDA,
};
pub use tp_sl::{
    compute_tp_sl, compute_tp_sl_with_target_rr, latency_slippage_pct, latency_slippage_pct_hurst,
    roundtrip_friction, roundtrip_friction_hurst, try_compute_tp_sl, TpSl, TpSlInputs,
};
pub use ville_e_process::VilleEProcess;

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
pub const REJECT_SLOTS: usize = 19;
pub const REJ_FLAT_COIN: usize = 0;
pub const REJ_EXPOSURE_ZERO: usize = 1;
pub const REJ_CORRELATION: usize = 2;
pub const REJ_SPEC: usize = 3;
pub const REJ_EV: usize = 4;
pub const REJ_FEE_IMPACT: usize = 5;
pub const REJ_MIN_NOTIONAL: usize = 6;
pub const REJ_MARGIN_INSUFFICIENT: usize = 7;
pub const REJ_ORCHESTRATOR: usize = 8;
pub const REJ_OTROS: usize = 9;
pub const REJ_DRAWDOWN: usize = 10;
pub const REJ_TP_SL_FLOOR: usize = 11;
pub const REJ_CONFIDENCE: usize = 12;
/// D-750 — la orden más pequeña que el símbolo acepta ya arriesga más de lo que
/// el control de ruina permite: la operación es INVIABLE, no «pequeña».
pub const REJ_VIABILIDAD: usize = 13;
/// D-751 — no hay probabilidad de ganar (ni calibrada ni observada) con la que
/// evaluar el valor esperado: se rechaza por falta de evidencia.
pub const REJ_SIN_EVIDENCIA: usize = 14;
pub const REJ_INVALID_INPUT: usize = 15;
pub const REJ_TARGET_GEOMETRY: usize = 16;
/// D4 (ADR-0016, CL-98) — la moneda está en fase de sonda y ya hay una sonda
/// abierta en la cartera: las sondas se abren de una en una (`sonda.rs`).
pub const REJ_SONDA_OCUPADA: usize = 17;
/// CL-107 — el freno de la estrategia rechazó H0 (proceso e de Ville sobre R,
/// `freno_estrategia.rs`): no se admiten entradas nuevas.
pub const REJ_ESTRATEGIA_FALSADA: usize = 18;

/// CL-100 — umbral del veto de primer toque R-15, que ahora sólo se publica
/// (`r15_vetaria` por moneda): P(SL antes que TP) por encima de él habría
/// vetado la orden. Es el literal de R7-R2-A-1 (Ola Ω73), sin derivar.
pub const UMBRAL_R15_SOMBRA: f64 = 0.88;

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
    /// QS-R4b: razón (`REJ_*`) del último rechazo de la evaluación en curso
    /// en este hilo; `usize::MAX` si no la hubo.
    static ULTIMO_RECHAZO: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Fija la dirección con la que se atribuyen los rechazos siguientes del hilo.
/// Marca además el inicio de una evaluación: borra la razón del rechazo
/// anterior (ver [`ultimo_rechazo`]).
pub fn set_reject_direction(signal: SignalType) {
    let d = match signal {
        SignalType::Long => 0,
        SignalType::Short => 1,
        SignalType::Flat => 2,
    };
    REJECT_DIR.with(|cell| cell.set(d));
    ULTIMO_RECHAZO.with(|cell| cell.set(usize::MAX));
}

/// QS-R4b — razón (`REJ_*`) por la que la última evaluación de este hilo
/// devolvió una orden rechazada. `None` si la admitió, y también en el único
/// rechazo que no pasa por los contadores (intención plana o capital
/// asignado ≤ 0 en `evaluate_single_intent`): el llamador mira primero la
/// señal de la orden. El libro
/// contrafactual (`audit_engine::shadow_ledger`) la usa como fuente del
/// veto sin cambiar la forma de `ValidatedOrder`. Léase en el mismo hilo,
/// justo después de `evaluate_quantum_order`.
pub fn ultimo_rechazo() -> Option<usize> {
    let v = ULTIMO_RECHAZO.with(|cell| cell.get());
    (v < REJECT_SLOTS).then_some(v)
}

fn rej(i: usize) -> ValidatedOrder {
    REJECT_COUNTERS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let d = REJECT_DIR.with(|cell| cell.get());
    if d < 2 {
        REJECT_COUNTERS_DIR[d][i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    ULTIMO_RECHAZO.with(|cell| cell.set(i));
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
    "sonda_ocupada",
    "estrategia_falsada",
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
    /// CL-107 — freno de la estrategia (proceso e de Ville sobre la R de cada
    /// cierre, H0 = +0,30 R). Echado, no se admiten entradas nuevas.
    pub freno_estrategia: freno_estrategia::FrenoEstrategia,
    /// CL-131 — lo medido de la última orden validada, hasta que el núcleo
    /// la tome al abrirla (`tomar_medida_validada`).
    medida_pendiente: Option<MedidaOrden>,
}

/// CL-131 (R2 de la revisión del PR #62) — lo que una orden validada aporta
/// a las medias de la cuenta: su riesgo al stop (en fracción del capital al
/// dimensionar y en USD) y su geometría en stops, `w = (TP − f)/SL` y
/// `l = (SL + f)/SL`. CL-135: el núcleo la guarda con la posición y la
/// registra al cerrarla (`registrar_medida`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MedidaOrden {
    coin_id: usize,
    riesgo: f64,
    riesgo_usd: f64,
    geometria: Option<(f64, f64)>,
}

impl RiskEngine {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            peak_capital: initial_capital,
            freno_estrategia: freno_estrategia::FrenoEstrategia::new(),
            medida_pendiente: None,
        }
    }

    /// CL-134 — el motor empieza de cero con `initial_capital`: el pico, el
    /// freno de la estrategia y la medida pendiente.
    pub fn reset(&mut self, initial_capital: f64) {
        self.peak_capital = initial_capital;
        self.freno_estrategia = freno_estrategia::FrenoEstrategia::new();
        self.medida_pendiente = None;
    }

    /// CL-131/CL-135 — la medida de la última orden validada de `coin_id`,
    /// que deja de estar pendiente. El núcleo la guarda con la posición que
    /// abre y la registra al cerrarla (`registrar_medida`), sólo si ese
    /// cierre puede enseñar.
    ///
    /// Antes de CL-131 las medias se actualizaban al final de
    /// `evaluate_single_intent`, antes de los vetos del núcleo, y la misma
    /// intención vetada se medía en cada tick. CL-131 las movió a la
    /// apertura del núcleo, que es LOCAL y anterior a los vetos del host: una
    /// intención que el host revierte y el núcleo reintenta en el evento
    /// siguiente se seguía midiendo cada vez (PR70-INT-04 de la Línea G).
    pub fn tomar_medida_validada(&mut self, coin_id: usize) -> Option<MedidaOrden> {
        let m = self.medida_pendiente.filter(|m| m.coin_id == coin_id)?;
        self.medida_pendiente = None;
        Some(m)
    }

    /// CL-135 — lo medido de una operación entra en las medias de la cuenta
    /// (riesgo al stop en fracción y en USD, ganancia y pérdida en stops).
    pub fn registrar_medida(arena: &GlobalArena, m: &MedidaOrden) {
        let actualizar = |celda: &quantum_arena::AtomicF64, nuevo: f64| {
            let previo = celda.load(Ordering::Relaxed);
            celda.store(
                crate::drawdown::actualizar_riesgo_ewma(
                    previo,
                    nuevo,
                    crate::drawdown::TRADE_HORIZON / 10.0,
                ),
                Ordering::Relaxed,
            );
        };
        actualizar(&arena.riesgo_por_operacion, m.riesgo);
        actualizar(&arena.riesgo_por_operacion_usd, m.riesgo_usd);
        if let Some((ganancia, perdida)) = m.geometria {
            actualizar(&arena.ganancia_en_stops, ganancia);
            actualizar(&arena.perdida_en_stops, perdida);
        }
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
        // CL-131: la medida pendiente es la de la última orden validada.
        self.medida_pendiente = None;
        if coin_id >= arena.coins.len() || intent.signal == SignalType::Flat {
            return rej(REJ_FLAT_COIN);
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

        // CL-99 (LC-F4-10): el pico es el del arena, compartido con el
        // sistema inmune del host y guardado por él entre reinicios.
        // `peak_capital` queda como su copia para los lectores de este motor.
        // CL-132 (R3 de la revisión del PR #62): se registra y se mide con el
        // MISMO capital que el host (`drawdown::capital_del_freno`, el más
        // conservador entre el arena y el exchange). Antes el riesgo subía el
        // pico con el arena: en papel sobre mainnet, ganar en papel se veía en
        // el host como una caída del saldo real.
        let capital_freno = crate::drawdown::capital_del_freno(arena);
        self.peak_capital = crate::drawdown::registrar_pico(
            &arena.pico_capital,
            capital_freno.max(self.peak_capital),
        );
        // U-3: los picos espejo scalp/swing (sin lectores) extirpados.

        // CL-133 (B2): con el pico guardado ilegible no se sabe cuánto se ha
        // caído; no entran posiciones nuevas (fail-closed, como CL-108f).
        if arena.pico_ilegible.load(Ordering::Relaxed) {
            return rej(REJ_DRAWDOWN);
        }
        // O-04 → CL-106 → CL-130 (LC-F4-10, decisión del dueño «Ambos»;
        // R1 de la revisión del PR #62): FRENO DURO DE LA CUENTA, la MISMA
        // función del sistema inmune del host. No se abre posición nueva
        // mientras la caída de los cierres, contada en stops de SUS
        // posiciones, llegue al cuantil 95 % de la caída máxima en 200
        // operaciones con H0 = +0,05 R, o la caída del capital desde el pico
        // llegue a d* (ADR-0016 D3). Sustituye a la prueba de racha
        // (`drawdown_compatible`), que con 8-12 R daba un 36-49 % de falsa
        // alarma en 200 operaciones aun con +0,30 R de ventaja, y a la
        // tolerancia micro (D-641). Hasta CL-130 la caída se medía en USD y
        // se dividía por la EWMA del riesgo de las últimas órdenes, que
        // encoge en una racha perdedora (LH-04): disparaba tras 13 SL.
        if crate::drawdown::freno_duro(arena, self.peak_capital, capital_freno).echado() {
            return rej(REJ_DRAWDOWN);
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
        if !raw_kelly.is_finite() {
            return rej(REJ_INVALID_INPUT);
        }

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
            // F1-087 (Ola Ω191): No re-clampear a clamp_min. Si PF <= 1.0, calculate_kelly_fraction
            // produce exploración en [0, 0.25·clamp_min]. Re-clampear a clamp_min destruía
            // la exploración fraccional de kelly.rs (CERT-M5-C01) inflando el riesgo 4x-15x
            // en monedas sin edge probado. Si PF > 1.0, kelly.rs ya aplicó clamp_min internamente.
            raw_kelly.min(clamp_max)
        };
        // CERT-M5-H03: el camino bootstrap/micro YA NO bypassa el tope de
        // ruina — mismo streak-bound + axioma 25% que kelly/envelope. q del
        // win-rate del coin si existe; conservador si no.
        let wr_coin = arena.coins[coin_id]
            .metrics
            .win_rate
            .load(Ordering::Relaxed);
        let trades_coin = arena.coins[coin_id]
            .metrics
            .trade_count
            .load(Ordering::Relaxed) as f64;
        let q = crate::ruin::conservative_loss_q(wr_coin, trades_coin);
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
        &mut self,
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
        // CL-146: un capital NaN pasaba esta puerta y llegaba como límite NaN
        // a `f64::clamp` (pánico; con `panic = "abort"` el proceso vivo muere).
        if intent.signal == SignalType::Flat
            || allocated_capital.is_nan()
            || allocated_capital <= 0.0
        {
            return ValidatedOrder::rejected();
        }
        // D4 (ADR-0016, LC-F4-19, CL-98): una moneda con menos de
        // `sonda::CIERRES_DE_SONDA` cierres sólo puede abrir una SONDA, y la
        // cartera tiene una sola sonda abierta a la vez. Es la misma puerta
        // que la sonda del ML-GATE del núcleo (`arranque_frio_sin_roster`).
        // CL-107: con el freno de la estrategia echado no entra nada nuevo
        // (las posiciones abiertas se gestionan fuera de aquí).
        if self.freno_estrategia.rechazada() {
            return rej(REJ_ESTRATEGIA_FALSADA);
        }
        let es_sonda = crate::sonda::en_fase_de_sonda(arena, coin_id);
        if !crate::sonda::admite_sonda_con_fase(arena, es_sonda) {
            return rej(REJ_SONDA_OCUPADA);
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
            None => return rej(REJ_SPEC),
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
        let signal_prob = if intent.win_probability > 0.0 {
            intent.win_probability
        } else {
            intent.confidence
        };
        let conviccion = 1.0
            + (signal_prob - arena.config.min_confidence_btc.load(Ordering::Relaxed))
                .max(0.0);
        let micro_kelly_input = kelly_adjusted * conviccion;
        // A zero returned for invalid size must not be blended with the
        // positive standard candidate, which would reopen this rejected input.
        if !micro_kelly_input.is_finite() {
            return rej(REJ_INVALID_INPUT);
        }
        let micro_kelly =
            crate::ruin::clamp_ruin(micro_kelly_input, crate::ruin::CONSERVATIVE_Q).max(0.0);
        let kelly_for_scale =
            crate::capital_regime::lerp(kelly_adjusted, micro_kelly, micro_w_alloc);

        // OLA Ω134: Modulación Conformal de Microestructura en Dimensionamiento Kelly
        // Leemos la escala conformal de incertidumbre de absorción de bloques (Kyle-Hasbrouck).
        // Si hay compresión de absorción alineada (conformal_band_scale = 0.70 < 1.0), la certidumbre
        // local permite aceleración controlada (+25%). Si hay choque/expansión (conformal_band_scale = 1.50 > 1.0),
        // se reduce la fracción de Kelly (-33%) para blindar el micro-capital ($13 USD).
        let sym_opt = Some(spec.symbol.as_str());
        let conformal_band_scale = arena
            .registry
            .get_scoped_parameter(
                sym_opt,
                Some(coin_id),
                "hidden_liquidity_conformal_scale",
                "RiskEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0)
            .clamp(0.50, 2.0);
        let conformal_kelly_mult = (1.0 / conformal_band_scale).clamp(0.50, 1.25);
        let kelly_for_scale = kelly_for_scale * conformal_kelly_mult;

        let epi_bias = if coin_id < arena.coins.len() {
            let now_ts = arena.coins[coin_id]
                .last_tick_timestamp_ms
                .load(Ordering::Relaxed);
            arena.coins[coin_id].get_active_epigenetic_bias(now_ts)
        } else {
            1.0
        };
        // Ola Ω100 / Ola Ω109 (F3.4-07): Relajación continua Ornstein-Uhlenbeck del sesgo
        // epigenético evaluada al reloj causal `now_ts`. Erradica la petrificación no-ergódica.
        // F1-087 (Ola Ω191): Asimetría anti-hot-hand unificada con la doctrina canónica de LH-04 (:1503-1505).
        // Racha perdedora (bias < 1.0) encoge confianza y tamaño (información honesta de quiebre de régimen).
        // Racha ganadora (bias > 1.0) NO infla (techo 1.00): la cota de vida ya es el techo honesto (anti-hot-hand).
        let effective_confidence =
            (signal_prob * epi_bias.clamp(0.50, 1.00)).clamp(0.05, 0.98);
        let raw_exposure = dir * effective_confidence * kelly_for_scale * allocated_capital;
        if !raw_exposure.is_finite() {
            return rej(REJ_INVALID_INPUT);
        }
        if raw_exposure == 0.0 {
            return rej(REJ_EXPOSURE_ZERO);
        }

        // D-748 / SPECTRAL-003/004: medir dependencia de PnL en TODOS los
        // slots, sin separar motores/horizontes. Unknown sigue siendo Unknown;
        // una matriz estrella incompleta y T=capacidad no pueden certificar
        // independencia ni borrar el conteo conservador.
        let is_long = intent.signal == SignalType::Long;
        let current_cap = arena.unified_capital.load(Ordering::Relaxed);
        let corr_thresh = arena
            .config
            .global_correlation_threshold
            .load(Ordering::Relaxed);
        // (Ola XLV/LXVII) N_eff como telemetría: dimensión efectiva del
        // grupo para el sizing por factor. El TODO original ("wire when
        // matrix exposed") quedó huérfano desde D-748 — cerrado así: la
        // matriz no se materializa (dependency_exposure trabaja por
        // pares), así que el N_eff se publica DESPUÉS de dependency_exposure
        // usando su ρ efectiva medida del grupo (XLVI·D) con la forma
        // equicorrelada de Grinold-Kahn: N_eff = k/(1+(k−1)·ρ̄).
        // Telemetría de diagnóstico, NO gate.
        let Some(dependence) = correlation_guard::dependency_exposure(
            arena,
            coin_id,
            is_long,
            corr_thresh,
        ) else {
            return rej(REJ_INVALID_INPUT);
        };
        {
            let k = dependence.same_bet_positions as f64;
            if let Some(rho) = dependence.same_bet_rho_efectivo() {
                if k >= 1.0 && rho < 1.0 {
                    let neff = k / (1.0 + (k - 1.0) * rho);
                    arena.registry.set("n_eff_grupo", neff);
                }
            }
        }
        // Las APIs MP/XLIV se conservan para diagnóstico e investigación.
        // (Ola XLVI·D/E, SPECTRAL-010) El veto agrega el RIESGO REAL de cada
        // miembro misma-apuesta al stop (qty·|entry−sl|/capital, medido del
        // snapshot), el riesgo EWMA de la candidata, y la ρ_PnL medida del
        // grupo (D-748). Continuidad: miembro no medido ⇒ proxy tope/8 (el
        // presupuesto lineal legado es el caso todos-no-medidos); riesgos
        // uniformes reducen bit a bit a r·√(k+k(k−1)ρ̄).
        let wr_coin = arena.coins[coin_id]
            .metrics
            .win_rate
            .load(Ordering::Relaxed);
        let trades_coin = arena.coins[coin_id]
            .metrics
            .trade_count
            .load(Ordering::Relaxed) as f64;
        let q_perdida = crate::ruin::conservative_loss_q(wr_coin, trades_coin);
        let tope = crate::ruin::clamp_ruin(1.0, q_perdida);
        let riesgo_ewma = arena.riesgo_por_operacion.load(Ordering::Relaxed);
        let riesgo_candidata = if riesgo_ewma.is_finite() && riesgo_ewma > 0.0 {
            riesgo_ewma
        } else {
            tope / 8.0 // arranque frío: mismo proxy que el veto legado
        };
        // OLA Ω194 (Zero-Alloc Hot Path / F1-089): Copia de riesgos híbridos en buffer de stack
        let mut riesgos_buf = [0.0f64; quantum_arena::state::MAX_COINS * 3 + 1];
        let num_riesgos = dependence.same_bet_riesgos_hibridos_into(
            tope / 8.0,
            riesgo_candidata,
            &mut riesgos_buf,
        );
        let riesgos = &riesgos_buf[..num_riesgos];
        // #602 (Ola 24): el tope del grupo se APRIETA con la cota actuarial
        // de Cramér-Lundberg cuando existe R medido para esta moneda
        // (`c{id}:lundberg_r_nocional`, publicado por el estimador #600 en
        // cada cierre). Clave ausente o R ≤ 0 → None → tope_streak intacto
        // (bit a bit el comportamiento anterior: disciplina de arranque
        // frío D-754 — la cota no significa nada sin edge medido).
        // ε = 0.05 es POLÍTICA del dueño (ψ(m) ≤ 5%, la misma convención
        // del `lundberg_margen_5pct` que publica el estimador).
        // #651 (Ola 51) — CONVERSIÓN DE UNIDADES: el estimador observa
        // retornos POR NOCIONAL ⇒ R_nocional; los `riesgos[]` y el tope
        // son fracciones de CAPITAL. Los siniestros de capital son los de
        // nocional × apalancamiento L ⇒ R_capital = R_nocional / L y el
        // margen en unidades de capital = L · (ln(1/ε)/R_nocional). Sin
        // la conversión, el margen quedaba en unidades de nocional
        // mezcladas contra fracciones de capital. Se usa el apalancamiento
        // F1-014 / LC-F4-01 / F1-015 (Ola Ω152): Corrección de unidades de Cramér-Lundberg.
        // El estimador observa retornos por nocional => R_nocional.
        // La cota de capital (cramer_lundberg.rs:21-22) exige R_capital = R_nocional / f_exposure,
        // donde f_exposure = nocional / capital es la fracción de exposición de capital de la orden.
        // Dividir erróneamente por `max_exchange_leverage` (hasta 125x) colapsaba R_capital entre 25x y 320x,
        // presumiendo un apalancamiento ficticio del 100% de margen a L_max y vetando candidatos con edge real.
        // La exposición real f_exposure está acotada por [dynamic_min_notional/capital, margin_cap * L_max_envio].
        let r_raw = arena
            .registry
            .get_for_coin_or(coin_id, "lundberg_r_nocional", 0.0);
        // CL-146: con 13 $ y un símbolo de mínimo 100 $ (BTCUSDT), el suelo
        // (7,7) supera al techo (0,25·20 = 5) y `f64::clamp` entra en pánico:
        // con `panic = "abort"`, el proceso vivo muere. El suelo es la
        // exposición REAL de la orden más pequeña que el exchange admite, así
        // que manda él; si la orden cabe o no lo deciden las puertas de abajo
        // (`orden_viable`), como antes. Con suelo ≤ techo es el mismo clamp.
        let f_suelo = dynamic_min_notional / allocated_capital.max(1.0);
        let f_techo =
            (0.25 * (spec.max_leverage as f64).min(crate::envio::TOPE_ENVIO as f64)).max(1.0);
        let f_exposure = (riesgo_candidata / continuous_sl.max(1e-4))
            .min(f_techo)
            .max(f_suelo);
        let r_lundberg = (r_raw.is_finite() && r_raw > 0.0)
            .then(|| r_raw / f_exposure);
        // #613 (Ola 35): el ρ̄ del grupo se APRIETA con la coherencia
        // espectral a la escala de τ* (`qo_613_rho_tau`, publicado por el
        // core desde el módulo multiactivo #607). Solo TIGHTEN: si el IC
        // espectral supera al escalar medido, la orden opera en una escala
        // donde el grupo está MÁS acoplado de lo que el PnL agregado ve.
        // Clave ausente (sin pares maduros a esa escala, fixture mono-
        // activo) o IC ≤ escalar ⇒ ρ de siempre, bit a bit. D-750b: el IC
        // es SIGNED — la anticorrelación (cobertura) NUNCA afloja.
        let rho_escalar = dependence.same_bet_rho_efectivo();
        let rho_ic = arena
            .registry
            .get_for_coin_or(coin_id, "qo_613_rho_tau", f64::NAN);
        let rho_final = match rho_escalar {
            Some(base) => {
                if rho_ic.is_finite() && rho_ic > base {
                    if rho_ic > 1.0 {
                        return rej(REJ_CORRELATION); // IC clampeado a 1 en el módulo; defensa
                    }
                    let previo = arena
                        .registry
                        .get_for_coin_or(coin_id, "qo_613_aprietes", 0.0);
                    arena
                        .registry
                        .set_for_coin(coin_id, "qo_613_aprietes", previo + 1.0);
                    Some(rho_ic)
                } else {
                    Some(base)
                }
            }
            None => rho_escalar,
        };
        if correlation_guard::veto_por_riesgo_cramer_lundberg(
            riesgos,
            rho_final,
            tope,
            r_lundberg,
            0.05,
        ) {
            // Contable para el consejo: el veto disparó CON la cota
            // disponible (la misma función pura decide — paridad #586).
            if r_lundberg.is_some() {
                let previo = arena
                    .registry
                    .get_for_coin_or(coin_id, "qo_602_veto_lundberg", 0.0);
                arena
                    .registry
                    .set_for_coin(coin_id, "qo_602_veto_lundberg", previo + 1.0);
            }
            return rej(REJ_CORRELATION);
        }

        // D-750: el spec y el nocional mínimo del símbolo se resolvieron al
        // principio de la evaluación, porque de ellos depende también el
        // régimen de capital. Aquí sólo se usa lo ya resuelto.
        let coin = &arena.coins[coin_id];

        let current_atr = coin.current_atr.load(Ordering::Relaxed);
        let current_price = coin.current_price.load(Ordering::Relaxed);
        // F1-118: Fail-closed ante precio o ATR no finito o <= 0.0.
        // Evita geometrías ficticias y fricciones sin término de latencia.
        if !current_price.is_finite()
            || current_price <= 0.0
            || !current_atr.is_finite()
            || current_atr <= 0.0
        {
            return rej(REJ_INVALID_INPUT);
        }
        let atr_pct = current_atr / current_price;
        if !atr_pct.is_finite() || atr_pct <= 0.0 {
            return rej(REJ_INVALID_INPUT);
        }

        let hurst_exponent = coin.hurst_exponent.load(Ordering::Relaxed);
        // ADR-0011 / LC-F1-03: Multiplicador de volatilidad resuelto por IDENTIDAD de símbolo en el arena
        let vol_mult = arena.volatility_multiplier_for(coin_id, atr_pct);
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
        // XLIV-8: la fórmula completa vive en `tp_sl::roundtrip_friction`,
        // la MISMA que usan los brackets del host y el daemon.
        debug_assert_eq!(entry_fee_rate, exit_fee_rate);
        let roundtrip_fee =
            crate::tp_sl::roundtrip_friction(entry_fee_rate, slip_floor, atr_pct, lat_ms);
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
        // F-A1 (Fable, 2026-10-17) — COSTE DE CARRY ESPERADO del funding.
        // Un perpetuo paga funding en cada asentamiento (T_F = 8 h): una
        // posición mantenida τ cruza en media τ/T_F asentamientos y en cada
        // uno paga d·f·nocional (d = +1 largo, −1 corto; f = tasa prevista
        // del PRÓXIMO asentamiento, que el poller del host publica por
        // símbolo en `CoinArena::funding_rate_next`). E[carry] = d·f·τ/T_F
        // es exacto bajo fase uniforme, causal (f se conoce en t) y por
        // símbolo. Sólo se cobra el COSTE; el ingreso esperado se acota en 0
        // (un pronóstico de cobro no es evidencia para abrir). Sin dato o
        // caducado (TTL) ⇒ 0: idéntico al comportamiento anterior. Entra en
        // la barrera de EV (D-751) y en el impacto de comisión (LH-02); la
        // geometría TP/SL (D-637) sigue sobre la fricción de ejecución, que
        // es la que el host reproduce al construir los brackets.
        let reloj_ms = arena.registry.get_event_time().max(0) as u64;
        let carry_esperado = arena.coins[coin_id].expected_funding_cost(
            intent.signal == SignalType::Long,
            tau_for_sizing,
            reloj_ms,
        );
        let friccion_total = roundtrip_fee + carry_esperado;
        if !friccion_total.is_finite() || friccion_total < 0.0 {
            return rej(REJ_INVALID_INPUT);
        }
        // CL-26: la ley de escala del TP/SL usa el Hurst MUESTREADO POR RELOJ
        // (DFA al cierre de la vela de 1 minuto, D-615b), no
        // `hurst_scale_matched`. Ése (S-7) sale de ventanas de 10/25/50
        // EVENTOS del motor multifractal: mide la microestructura del feed
        // (con muchos ticks repetidos da H ≈ 0,2 en un paseo browniano) y,
        // como τ dominante se queda en el ancla de 30 s, siempre era la
        // ventana de 10 eventos. D-615b retiró el Hurst por evento
        // precisamente porque gobierna esta ley; S-7 lo había reintroducido.
        let hurst_for_geometry = hurst_exponent;
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
                // CL-62: el gen de BTC va con su identidad, no con el slot 0.
                sl_atr_multiplier: arena.sl_atr_multiplier_for(coin_id),
                // D-754: la σ que el espectro predictivo pronostica PARA ESTE
                // horizonte, si ha demostrado habilidad fuera de muestra. El
                // arena publica ceros mientras no la tenga, y entonces la
                // geometría sigue con la ley de escala sobre el ATR medido.
                sigma_forecast: arena.coins[coin_id].sigma_forecast_at(tau_for_sizing),
            },
            arena.config.tp_rr_ratio_btc.load(Ordering::Relaxed),
        );
        // D-636b & #585 + QS-D2: un horizonte cuyo stop difusivo no paga la
        // fricción se RECHAZA, en cualquier régimen de capital. Para informar,
        // `compute_tp_sl` eleva el stop al suelo de viabilidad, pero esa orden
        // llevaría un stop más ancho que la dispersión de su τ: durante la
        // vida de la posición casi nunca toca TP ni SL y sale por tiempo. Lo
        // que corresponde es otro τ (la puerta QO-586 del núcleo ya lo filtra
        // antes), no otro stop. QS-D2 retira el atajo micro de Ω47/F4-M1, que
        // admitía esa orden cuando el suelo cabía en 55 pb.
        if tpsl_gate.below_tradeable_floor {
            return rej(REJ_TP_SL_FLOOR);
        }
        // QS-D2: el stop y el TP de la orden son los de su horizonte. Se
        // retira el tope micro de 55 pb (que interpolaba el stop hacia 55 pb
        // sin tocar τ). Con barreras a y b de un movimiento browniano con
        // deriva μ y volatilidad σ, la ganancia esperada por operación es
        // μ·E[T], con E[T] ≈ a·b/σ² (tiempo medio de salida sin deriva).
        // Acortar las dos barreras por un factor k reduce lo que la deriva
        // cobra en k², mientras la comisión de ida y vuelta no cambia. Lo que
        // se pierde en dólares lo acotan el dimensionado y la viabilidad
        // (R-19: `orden_viable` con el stop real), no la geometría.
        let (expected_win, expected_loss) = (tpsl_gate.tp_pct, tpsl_gate.sl_pct);

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

        // R7-R2-A-1 / Ola Ω73: Activación analítica de la probabilidad en forma cerrada de primer toque (Ω3 / CL-34)
        // Calibración R-15: La coherencia espectral es una correlación instantánea en [-1, 1],
        // cuya persistencia decorrelaciona con el horizonte tau (Hurst / half-life).
        // Se escala dimensionalmente con (30s / tau)^{1/2} para evitar que horizontes extendidos
        // sobre-acumulen deriva constante un-física y veten señales válidas con desacuerdos mínimos.
        let spectral_tide = arena.coins[coin_id].spectral_coherence.load(Ordering::Relaxed);
        // CL-83 (F3.2-03): sin suelo. El recorte a 0,15 se activaba con
        // τ > 30 s/0,15² = 1 333 s y por encima la deriva volvía a acumularse
        // ∝ √τ: con brackets brownianos (∝ √τ) R-15 vetaba las τ largas que
        // decía dejar de castigar (P = 0,883 a 4 h y 0,953 a 12 h con una
        // coherencia de −0,3, frente a 0,762 hasta 1 333 s). Sin el suelo,
        // θ·tp no depende de τ. El modelo de fondo (la coherencia como
        // deriva, el ATR de 1 min como σ y el ancla de 30 s) sigue abierto
        // en C-F4 (LC-F4-04).
        let tau_ratio = (30_000.0 / tau_for_sizing.max(30_000.0)).sqrt();
        let directional_drift = if spectral_tide.is_finite() {
            dir * spectral_tide * atr_pct * tau_ratio
        } else {
            0.0
        };
        let p_hit_sl_first = crate::tp_sl::probabilidad_tocar_sl_antes_de_tp(
            expected_win,
            expected_loss,
            directional_drift,
            atr_pct,
        );
        // CL-100 (R-15 EN SOMBRA, decisión del dueño 2026-10-12): el primer
        // toque ya no veta. Sin deriva, P(SL antes que TP) = TP/(TP + SL) =
        // RR/(1 + RR): toda geometría con TP/SL > 7,33 se vetaba por su
        // forma, no por la marea, y la deriva sale de la coherencia espectral
        // por el ATR de 1 min, un modelo sin validar (LC-F4-04). Queda como
        // telemetría por moneda: la probabilidad y si habría vetado con el
        // umbral de antes, para medir el veto antes de devolverle poder.
        arena.registry.set_for_coin(coin_id, "p_hit_sl_first", p_hit_sl_first);
        arena.registry.set_for_coin(
            coin_id,
            "r15_vetaria",
            if p_hit_sl_first > UMBRAL_R15_SOMBRA { 1.0 } else { 0.0 },
        );

        // CL-2 — EL APALANCAMIENTO SE DECIDE CON LA GEOMETRÍA YA RESUELTA.
        // Antes se calculaba ANTES de construir TP/SL: el freno de
        // volatilidad (D-746) tenía que adivinar el stop con la τ del gen
        // `temporal_scale` y la curva `sl_at_tau`, mientras la orden usa la τ
        // medida de la moneda (D-745) y el stop de `compute_tp_sl`. Nada de
        // la geometría depende del apalancamiento, así que se decide aquí.
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
                // CL-2: el freno de volatilidad mide contra el stop que la
                // orden USARÁ (τ medida, geometría real, cap micro y precio
                // objetivo de la señal), no contra la curva del gen.
                expected_loss,
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
        // F1-088 (Ola Ω192): Tasa de margen de mantenimiento del símbolo (MMR)
        // y distancia física total al stop con fricción y colchón de liquidación.
        let mmr_spec = arena
            .registry
            .get_for_coin_or(coin_id, "maintenance_margin_rate", 0.01)
            .clamp(0.001, 0.05);
        let dist_stop_full = expected_loss + friccion_total + mmr_spec;
        if dist_stop_full >= 1.0 {
            return rej(REJ_TARGET_GEOMETRY);
        }
        let max_safe_lev_liq = ((1.0 - 1e-4) / dist_stop_full).floor().max(1.0);

        if !dynamic_leverage.is_finite() || dynamic_leverage < 1.0 {
            return rej(REJ_INVALID_INPUT);
        }
        dynamic_leverage = dynamic_leverage
            .min(genome_max_leverage)
            .min(max_safe_lev_liq)
            .floor();


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
        // Transición suave C^1 Hermite cúbico (smoothstep) erradicando la discontinuidad discreta.
        let spec_coh_scale = arena.registry.get_for_coin_or(coin_id, "spec_coh_scale", 0.10).max(1e-4);
        let spec_ent_scale = arena.registry.get_for_coin_or(coin_id, "spec_ent_scale", 0.30).max(1e-4);
        let u_coh = (spec_coh / spec_coh_scale).clamp(0.0, 1.0);
        let s_coh = u_coh * u_coh * (3.0 - 2.0 * u_coh);
        let u_ent = ((1.0 - spec_ent) / spec_ent_scale).clamp(0.0, 1.0);
        let s_ent = u_ent * u_ent * (3.0 - 2.0 * u_ent);
        let coh_benefit = (spec_coh.max(0.0) * (1.0 - spec_ent * 0.5) * s_coh * s_ent).clamp(0.0, 0.80);
        let base_micro_ratio = arena.registry.get_for_coin_or(coin_id, "base_micro_ratio", 0.66 / 0.62);
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
        let trade_count = arena.coins[coin_id]
            .metrics
            .trade_count
            .load(Ordering::Relaxed);
        let arranque_frio_total = trade_count == 0;
        let p_ganar = if intent.win_probability.is_finite()
            && intent.win_probability > 0.0
            && intent.win_probability < 1.0
        {
            Some(intent.win_probability)
        } else if arranque_frio_total {
            None
        } else {
            // Ω46.1 / Ola Ω108 (F3.4-06): Contracción jerárquica bayesiana (Empirical Bayes Shrinkage).
            // Para muestras pequeñas (n < 20), el estimador no colapsa a 0.0 ni crea
            // un estado absorbente de muerte en n=1. Contrae suavemente hacia el prior empírico
            // resuelto dinámicamente desde el ensamble/portafolio (resolve_bayes_prior) con fallback a 0.55 y k0=10.0.
            let (p_prior, k_prior) = crate::evidence::resolve_bayes_prior(arena, coin_id);
            Some(crate::evidence::win_rate_hierarchical_lcb(
                arena.coins[coin_id]
                    .metrics
                    .win_rate
                    .load(Ordering::Relaxed),
                trade_count as f64,
                p_prior,
                k_prior,
            ))
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
        // Ω46.2 / D4 (CL-98): en fase de sonda (menos de `CIERRES_DE_SONDA`
        // cierres) la orden es una sonda: se admite porque está presupuestada
        // (una abierta en la cartera, orden mínima, pérdidas dentro del
        // drawdown de D3), no por un EV que nadie ha medido.
        //
        // Antes se le pedía que el EV con un prior de 0,55 cubriera la
        // fricción: 0,55·TP − 0,45·SL > f. El 0,55 no tenía origen (ADR-0016
        // D4). Para CUALQUIER prior p esa condición exige TP > f/p > f, así
        // que lo que se conserva sin inventar p es su parte común: una sonda
        // que, si gana, no cubre la fricción de ida y vuelta no puede medir
        // nada rentable. Con la geometría de `compute_tp_sl` (TP ≥ 1,5·SL +
        // 2,5·f) siempre se cumple; muerde con objetivos explícitos de la
        // intención (TP de 1 pb frente a 1 % de stop).
        //
        // Fuera de la fase de sonda rige D-751 íntegro.
        if es_sonda {
            if expected_win <= friccion_total {
                return rej(REJ_EV);
            }
        } else if p_ganar.is_some() {
            let ev_minimo_requerido = friccion_total * ev_fee_multiplier;
            if expected_value_pct <= ev_minimo_requerido {
                return rej(REJ_EV);
            }
        }

        let max_acceptable_fee_pct = arena.config.max_fee_pct.load(Ordering::Relaxed);
        if !max_acceptable_fee_pct.is_finite() || max_acceptable_fee_pct < 0.0 {
            return rej(REJ_INVALID_INPUT);
        }
        // Cost per notional times L is cost per unit of allocated margin.
        // Preserve the existing micro policy, but enforce it on every final
        // order, not just branches that rescue a minimum notional (FMT-217).
        let micro_fee_ceiling =
            crate::capital_regime::resolve_micro_max_fee_pct(arena, coin_id);
        let max_fee_limit =
            crate::capital_regime::lerp(max_acceptable_fee_pct, micro_fee_ceiling, micro_w_alloc);
        let dynamic_min_notional = crate::capital_regime::effective_min_notional(spec.min_notional);
        // Revisión del PR #61 (F1): la orden mínima que de verdad sale es la
        // del LOTE, el menor múltiplo de `stepSize` con `qty ≥ minQty` y un
        // nocional de al menos el mínimo más su margen de 0,10 USD (FASE 3),
        // al precio actual. Con lote grueso es mayor que el nocional mínimo
        // (BNB a 600 con paso 0,01: 6 USD, no 5,10): la viabilidad, el rescate
        // y el tope de la sonda razonan con ella. Sin paso o precio
        // utilizables queda el nocional mínimo de antes.
        let nocional_minimo = crate::capital_regime::nocional_minimo_ejecutable(
            dynamic_min_notional + 0.1,
            spec.step_size,
            spec.min_qty,
            current_price,
        )
        .unwrap_or(dynamic_min_notional + 0.1);

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
        let wr = arena.coins[coin_id]
            .metrics
            .win_rate
            .load(Ordering::Relaxed);
        let trades_coin = arena.coins[coin_id]
            .metrics
            .trade_count
            .load(Ordering::Relaxed) as f64;
        let q_para_ruina = crate::ruin::conservative_loss_q(wr, trades_coin);
        let tope_riesgo_evento = crate::ruin::clamp_ruin(1.0, q_para_ruina);
        if !crate::capital_regime::orden_viable(
            nocional_minimo,
            expected_loss,
            allocated_capital,
            tope_riesgo_evento,
        ) {
            return rej(REJ_VIABILIDAD);
        }

        let bounded_exposure = raw_exposure.clamp(-allocated_capital, allocated_capital);
        let mut final_margin = bounded_exposure.abs();

        // LH-04 (Línea H ola 1, Claude — GLM 117): DIMENSIONADO EN ESPACIO
        // DE RIESGO cuando hay evidencia medida. La medición T-3: con
        // oráculo +0.94 R/op, acierto 0.65 y pago b=1.40, el Kelly pleno
        // es 40% del capital por operación — el motor arriesgaba 0.2%
        // (todas las órdenes al nocional mínimo; 0.049/día vs 0.231/día
        // de la meta). El tamaño ES la palanca.
        //
        // f = ½ · max(0, p_LCB − (1 − p_LCB)/b)   [Plan V2 §1.1.4]
        //
        // - p_LCB: cota INFERIOR de Wilson (z=1.64, 90% uniladal — misma
        //   familia que `conviccion_de_rama` D-752) del win-rate MEDIDO de
        //   la moneda. Bajo incertidumbre (n chico) la cota cae sola y f
        //   → 0: honesto por construcción, no por gate.
        // - b: pago realizado = PF·(1−p̂)/p̂ (de PF = p̂·b/(1−p̂)).
        // - ½: medio-Kelly amortiguando la varianza de trayectoria
        //   (doctrina canónica del Plan V2 §1.1.4, no un literal libre).
        // - f se acota por el ÚNICO tope de riesgo por evento del sistema
        //   (`clamp_ruin`: streak-bound + axioma 25%).
        //
        // El margen se deriva del nocional objetivo al apalancamiento YA
        // decidido por la matriz (L no depende del margen — :981), de
        // modo que el riesgo al stop de la orden final ≈ f·C. TODOS los
        // floors downstream (viabilidad D-750, rescate de nocional
        // mínimo, fee_impact LH-02, invariantes) se aplican IGUAL sobre
        // este margen. Sin evidencia (n < mínimo, p̂ degenerado, PF ≤ 0):
        // f = 0 y el camino sizing/micro/exploración existente queda
        // INTACTO — fail-closed al estado actual.
        // OLA Ω127: Desacoplamiento de constantes estáticas hacia el registro omnisciente con fallbacks canónicos.
        let z_lcb_lh04 = arena.registry.get_value_or("lh04_z_lcb", 1.64);
        let half_kelly_lh04 = arena.registry.get_value_or("lh04_kelly_fraction", 0.5);
        // F1-116: Muestras mínimas continuas (n >= 1.0) con decaimiento natural de Wilson LCB.
        let muestras_min_lh04 = arena.registry.get_value_or("lh04_muestras_min", 1.0);
        let lh04_wr = arena.coins[coin_id]
            .metrics
            .win_rate
            .load(Ordering::Relaxed);
        let lh04_n = arena.coins[coin_id]
            .metrics
            .trade_count
            .load(Ordering::Relaxed) as f64;
        let lh04_pf = arena.coins[coin_id]
            .metrics
            .profit_factor
            .load(Ordering::Relaxed);
        let mut lh04_f = 0.0f64;
        let mut lh04_p_lcb = f64::NAN;
        let mut lh04_b = f64::NAN;
        if lh04_wr.is_finite()
            && (0.0..=1.0).contains(&lh04_wr)
            && lh04_n >= muestras_min_lh04
            && lh04_pf.is_finite()
            && lh04_pf > 0.0
            && lh04_wr > 0.0
            && lh04_wr < 1.0
        {
            let n = lh04_n;
            let z2 = z_lcb_lh04 * z_lcb_lh04;
            let denom = 1.0 + z2 / n;
            let centro = (lh04_wr + z2 / (2.0 * n)) / denom;
            let radio = z_lcb_lh04
                * ((lh04_wr * (1.0 - lh04_wr) / n + z2 / (4.0 * n * n)).sqrt() / denom);
            let p_lcb = (centro - radio).clamp(0.0, 1.0);
            let b = lh04_pf * (1.0 - lh04_wr) / lh04_wr;
            let f_raw = half_kelly_lh04 * (p_lcb - (1.0 - p_lcb) / b).max(0.0);
            let q_lh04 = crate::ruin::conservative_loss_q(lh04_wr, lh04_n);
            let tope_lh04 = crate::ruin::clamp_ruin(1.0, q_lh04);
            lh04_f = if f_raw.is_finite() { f_raw.min(tope_lh04) } else { 0.0 };
            lh04_p_lcb = p_lcb;
            lh04_b = b;
        }

        // F1-087 (Ola Ω191): Detección de ausencia de edge con evidencia madura.
        // Si la moneda tiene muestras suficientes (lh04_n >= muestras_min_lh04) y la cota inferior
        // de Wilson demuestra que no hay ventaja (lh04_f == 0.0), o sus métricas son degradadas
        // (PF <= 1.0 o WR <= 0.0), se activa `lh04_sin_edge`. Esto cierra el fail-open impidiendo
        // que una moneda sin edge reciba una apuesta de Kelly legado completo, dimensionándola
        // estrictamente a la orden mínima de exploración (sonda).
        let lh04_tiene_evidencia = lh04_wr.is_finite()
            && (0.0..=1.0).contains(&lh04_wr)
            && lh04_n >= muestras_min_lh04
            && lh04_pf.is_finite();
        let lh04_sin_edge = lh04_tiene_evidencia
            && (lh04_f == 0.0 || lh04_pf <= 1.0 || lh04_wr <= 0.0);
        // Streak-adaptación (GLM 118): la síntesis bias×LH-04. El bias
        // epigenético (racha, OU-relajado Ω109, rango [0.5, 2.0], leído
        // al RELOJ CAUSAL del último tick — mismo acceso que el camino de
        // convicción en :586) modula el tamaño de forma ASIMÉTRICA:
        // - Racha PERDEDORA (bias < 1): encoge f — información honesta de
        //   quiebre de régimen; la racha ES la señal (el enemigo de toda
        //   la saga L: reaccionar lento al quiebre).
        // - Racha GANADORA (bias > 1): NO infla — min(bias, 1.0). La cota
        //   inferior de Wilson de vida ya es el techo honesto; inflar por
        //   racha sería doble-cuento de suerte (cautela anti-hot-hand).
        // - SIN ESTADO ABSORBENTE (lección Ω110): el bias OU-relaja a 1.0
        //   en ~30 min (τ_bias de Ω109) — nada de (1−DD/d*) que muere en
        //   d*; y si f encogido deja el margen bajo el mínimo, el rescate
        //   de nocional mínimo existente sostiene la orden mínima (piso
        //   tipo sonda): encogimiento, jamás veto eterno.
        // p_LCB misma NO cambia: la estadística de la cota queda limpia;
        // la adaptación es un multiplicador de régimen sobre el tamaño.
        let lh04_bias_factor = if lh04_f > 0.0 && coin_id < arena.coins.len() {
            let now_ts = arena.coins[coin_id]
                .last_tick_timestamp_ms
                .load(Ordering::Relaxed);
            arena.coins[coin_id]
                .get_active_epigenetic_bias(now_ts)
                .clamp(0.0, 1.0)
        } else {
            1.0
        };
        let lh04_f_efectiva = lh04_f * lh04_bias_factor;
        // F-A5 (Fable, 2026-10-17; hallazgo F1-013, confirmado por dos
        // escépticos): el nocional en espacio de riesgo se obtiene con el
        // stop QUE LA ORDEN LLEVA (`expected_loss`, el de CL-2/CL-7 y el que
        // mide D-744/CL-131), no con `sl_at_tau(τ)`, la curva genómica. Con la
        // curva, el riesgo realizado al stop era f·C·(stop real/stop genómico):
        // con ATR alto se arriesgaba varias veces f y con ATR bajo una fracción,
        // y la medida D-744 (que sí usa el stop real) no reconciliaba con el
        // objetivo. `expected_loss` ya pasó la validación de geometría (finito
        // y > 0, si no la intención fue rechazada arriba); el respaldo a la
        // curva sólo cubre una reordenación futura de ese bloque.
        let stop_real = if expected_loss.is_finite() && expected_loss > 0.0 {
            expected_loss
        } else {
            continuous_sl
        };
        let mut lh04_f_realizado = 0.0_f64;
        if lh04_f_efectiva > 0.0 && dynamic_leverage >= 1.0 {
            let nocional_obj = lh04_f_efectiva * allocated_capital / stop_real;
            let margen_obj = nocional_obj / dynamic_leverage;
            if margen_obj.is_finite() && margen_obj > 0.0 {
                // F1-086 (Ola Ω190): Desacoplamiento del recorte erróneo de margen P1.
                // En LH-04, el riesgo al stop objetivo (f·C) YA está acotado por el axioma
                // de ruina (clamp_ruin <= 0.25 sobre f_raw en :1491).
                // El recorte P1 anterior imponía `final_margin = margen_obj.min(C · tope_ruina)`
                // aplicando un tope de RIESGO (25%) directamente a MARGEN (colateral).
                // Esto equivalía a riesgo <= tope sólo si L·SL == 1; con L·SL < 1 (el caso
                // normal de trading apalancado) restringía el riesgo efectivo a
                // f <= 0.25·L·SL (apenas ~1% para stops ceñidos), impidiendo que una orden
                // con evidencia estadística sólida tomara el riesgo calibrado por Kelly
                // e imposibilitando el crecimiento exponencial (+100%/72h).
                // Se asigna `final_margin = margen_obj` directamente: el colateral queda
                // acotado por capacidad y liquidez real downstream vía `resolve_margin_cap_limits`
                // y `available_margin_headroom` (Sol A2).
                final_margin = margen_obj;
                // F-A5 / F1-086: riesgo al stop que este dimensionado realiza, en
                // fracción del capital asignado, antes de rescates de nocional mínimo y
                // de techos de liquidez downstream.
                lh04_f_realizado = if allocated_capital > 0.0 {
                    final_margin * dynamic_leverage * stop_real / allocated_capital
                } else {
                    0.0
                };
            }
        }
        // Telemetría LH-04 (auditable por moneda): f EFECTIVA (post-racha).
        arena
            .registry
            .set_for_coin(coin_id, "lh04_f_riesgo", lh04_f_efectiva);
        arena
            .registry
            .set_for_coin(coin_id, "lh04_f_realizado", lh04_f_realizado);
        arena
            .registry
            .set_for_coin(coin_id, "lh04_capital_asignado", allocated_capital);
        arena
            .registry
            .set_for_coin(coin_id, "lh04_stop_real", stop_real);
        arena
            .registry
            .set_for_coin(coin_id, "lh04_p_lcb", lh04_p_lcb);
        arena
            .registry
            .set_for_coin(coin_id, "lh04_b_realizado", lh04_b);
        arena
            .registry
            .set_for_coin(coin_id, "lh04_bias_factor", lh04_bias_factor);

        let margin_cushion_pct = arena.config.margin_cushion_pct.load(Ordering::Relaxed);
        // D-641: el colchón de margen sale SIEMPRE del gen; la escasez sólo lo
        // relaja de forma continua hacia el máximo operativo. Antes el literal
        // 0,98 anulaba el gen precisamente en el entorno de capital real, de
        // modo que un gen evolucionado para la prudencia quedaba inerte en
        // producción.
        // D-634/D-635: la fórmula vive en `capital_regime::margin_cushion`,
        // compartida con la comprobación de margen libre del núcleo.
        let safe_cushion = crate::capital_regime::margin_cushion(margin_cushion_pct, scarcity);

        // D-641 (completo): el techo micro de apalancamiento es continuo en
        // la confianza (antes escalones en 0,70 y 0,75) y en el capital
        // (antes escalón de ~5× a 50× en $20). Interpolación geométrica: el
        // punto medio natural entre 5× y 50× es ~16×, no 27,5×.
        // CL-6: se calcula una vez y lo respetan LOS DOS rescates del nocional
        // mínimo (el segundo usaba el literal 50).
        let lev_base = arena.registry.get_for_coin_or(coin_id, "micro_lev_base", 5.0);
        let lev_bonus = arena.registry.get_for_coin_or(coin_id, "micro_lev_bonus", 1.5);
        let conf_center = arena.registry.get_for_coin_or(coin_id, "micro_lev_conf_center", 0.65);
        let conf_width = arena.registry.get_for_coin_or(coin_id, "micro_lev_conf_width", 0.10).max(1e-4);
        let macro_cap = arena.registry.get_for_coin_or(coin_id, "macro_lev_cap", 50.0);

        let conf_t = ((intent.confidence - conf_center) / conf_width).clamp(0.0, 1.0);
        let micro_lev_cap = lev_base + lev_bonus * conf_t * conf_t * (3.0 - 2.0 * conf_t);
        let mut max_lev_cap = crate::capital_regime::log_lerp(macro_cap, micro_lev_cap, micro_w_alloc);

        // F1-037: Acotación actuarial por cota de ruina de Cramér-Lundberg ψ ≤ e^{-R_capital · m} ≤ ε (ε = 0.05).
        // Si la moneda posee estimación madura de R_nocional > 0, el apalancamiento máximo seguro
        // no excede L_lundberg = (R_nocional · cushion) / ln(1/ε).
        let r_nocional = arena.registry.get_for_coin_or(coin_id, "lundberg_r_nocional", 0.0);
        if r_nocional.is_finite() && r_nocional > 0.0 {
            let ln_inv_eps = 2.995732273553991; // ln(1 / 0.05)
            let lundberg_lev_bound = (r_nocional * safe_cushion / ln_inv_eps).clamp(1.0, macro_cap);
            max_lev_cap = max_lev_cap.min(lundberg_lev_bound);
        }

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
            // CL-6 / AGY-AUD: usar .ceil() antes del cap para que el apalancamiento
            // alcance el mínimo sin forzar margen adicional innecesario.
            // F1-088 (Ola Ω192): El apalancamiento candidato queda estrictamente
            // acotado por el techo físico de liquidación `max_safe_lev_liq`
            // (`L · (SL + f + mmr) < 1.0`), garantizando que el stop loss ejecute
            // antes del precio de liquidación del exchange en futuros perpetuos.
            let candidate_leverage = (dynamic_min_notional / final_margin.max(0.01)) * 1.02;
            dynamic_leverage = candidate_leverage
                .ceil()
                .min(genome_max_leverage)
                .min(max_lev_cap)
                .min(max_safe_lev_liq)
                .floor();
        }

        // FASE 3 FIX: Micro-Account Notional Safety
        // Sumamos diez centavos de dólar (+0.1) al min notional para evitar rechazos
        // por pérdida de precisión IEEE-754 en multiplicaciones de apalancamiento.
        // Revisión del PR #61 (F1): y se redondea al lote (`nocional_minimo`).
        let safe_min_notional = nocional_minimo;
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
        // D4 (ADR-0016, CL-98, F1-087): la sonda o la orden en moneda sin edge demostrado
        // es la orden mínima del símbolo, con el apalancamiento ya decidido.
        // Es información para adquirir muestras, no alfa: su tamaño se acota al nocional mínimo viable.
        if es_sonda || lh04_sin_edge {
            final_margin = final_margin.min(required_margin_for_min_notional * 1.0005);
        }

        let (meets_min_notional, _) =
            guard::enforce_minimum_notional(final_margin, safe_min_notional, dynamic_leverage);
        if !meets_min_notional {
            if REJECT_COUNTERS[REJ_MIN_NOTIONAL].load(std::sync::atomic::Ordering::Relaxed) % 200 == 0 {
                println!(
                    "🔍 [REJ6] margin={:.4} lev={:.4} min_notional={:.4} allocated={:.4} kelly={:.4}",
                    final_margin, dynamic_leverage, safe_min_notional, allocated_capital, kelly_fraction
                );
            }
            return rej(REJ_MIN_NOTIONAL);
        }

        // D-641 / Ω94 / Ω112: límite de margen por operación continuo sin anclaje rígido
        // en dólares ni literales fijos (0.25 / 0.90). En régimen micro pleno
        // conserva por defecto el 25% del capital como límite seguro (configurable/adaptable
        // vía micro_margin_cap_ratio en el registro omnisciente), garantizando como piso
        // el margen necesario para cubrir el nocional mínimo ejecutable. Al crecer
        // la cuenta (interés compuesto), el límite escala proporcionalmente sin
        // estrangular el crecimiento.
        let orchestrator = orchestrator::PortfolioOrchestrator::new(arena);
        let portfolio_headroom = orchestrator
            .available_margin_headroom(bounded_exposure > 0.0, dynamic_min_notional);
        if !portfolio_headroom.is_finite() || portfolio_headroom <= 0.0 {
            return rej(REJ_ORCHESTRATOR);
        }
        let single_asset_cap = crate::capital_regime::resolve_margin_cap_limits(
            arena,
            coin_id,
            allocated_capital,
            required_margin_for_min_notional,
            safe_cushion,
            current_cap,
            micro_w_alloc,
        );
        let safe_limit = single_asset_cap.min(portfolio_headroom);
        if final_margin > safe_limit {
            // SOL-A2 — el "rescate" de margen era SILENCIOSO: reducía el
            // tamaño evaluado y, si hacia falta, SUBÍA el apalancamiento para
            // seguir alcanzando el nocional mínimo, sin dejar rastro. El
            // invariante terminal de abajo lo re-verifica, pero el consejo no
            // podía medir cuántas órdenes se redimensionan ni por qué. No
            // cambio la matemática: la hago AUDITABLE por moneda.
            // El cociente se mide ANTES de mutar: el margen pedido frente al
            // techo que se le impone (1,0 = sin recorte; <1 = recorte real).
            let cociente_recorte = (safe_limit / final_margin.max(1e-12)).min(1e6);
            let previo = arena
                .registry
                .get_for_coin_or(coin_id, "sol_a2_margen_reducido", 0.0);
            arena
                .registry
                .set_for_coin(coin_id, "sol_a2_margen_reducido", previo + 1.0);
            arena.registry.set_for_coin(
                coin_id,
                "sol_a2_margen_reducido_cociente",
                cociente_recorte,
            );
            final_margin = safe_limit;
            if final_margin > 0.0 && final_margin * dynamic_leverage < safe_min_notional {
                // CL-6 / F1-088: el apalancamiento ENTERO que alcanza el mínimo es el
                // techo del cociente, no su suelo, acotado por los MISMOS techos
                // que el primer rescate incluyendo max_safe_lev_liq.
                let needed_leverage = (safe_min_notional / final_margin).ceil();
                dynamic_leverage = needed_leverage
                    .min(genome_max_leverage)
                    .min(max_lev_cap)
                    .min(max_safe_lev_liq)
                    .floor()
                    .max(dynamic_leverage);
            }
        }

        // F1-092 (Ola Ω195): Axioma de Ruina sobre el Riesgo Realizado al Stop.
        // En cualquier camino (LH-04, camino legado o tras rescates de nocional mínimo),
        // el riesgo total realizado en dólares al tocar el stop:
        //   riesgo_usd = final_margin * dynamic_leverage * expected_loss
        // NUNCA debe exceder el tope actuarial de ruina `tope_riesgo_evento` (<= 0.25 * capital).
        // Si el riesgo excede este límite, se recorta `final_margin` a la cota máxima segura:
        //   margin_cap_ruin = (tope_riesgo_evento * allocated_capital) / (dynamic_leverage * expected_loss)
        // Y si el margen recortado no satisface el nocional mínimo de Binance, la comprobación
        // terminal de REJ_MIN_NOTIONAL rechaza la orden de forma fail-closed.
        if dynamic_leverage > 0.0 && expected_loss > 0.0 && allocated_capital > 0.0 {
            let max_riesgo_usd = tope_riesgo_evento * allocated_capital;
            let riesgo_propuesto_usd = final_margin * dynamic_leverage * expected_loss;
            if riesgo_propuesto_usd > max_riesgo_usd {
                let margin_cap = max_riesgo_usd / (dynamic_leverage * expected_loss);
                if margin_cap < final_margin {
                    final_margin = margin_cap;
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

        // F1-088: Invariante terminal de liquidación física en futuros:
        // L · (expected_loss + friccion_total + mmr_spec) < 1.0.
        // Si el stop loss más la fricción queda a o más allá de la distancia de liquidación,
        // el exchange liquida la posición antes de ejecutar el SL, causando pérdida del 100%
        // de colateral más comisión de liquidación mientras el registro modelaría un SL ficticio.
        if dynamic_leverage * dist_stop_full >= 1.0 {
            return rej(REJ_TARGET_GEOMETRY);
        }
        let final_fee_impact = friccion_total * dynamic_leverage;
        // LH-02 (Línea H / AGY): En el valle de capital ($35–$65 USD), allocated_capital
        // sitúa a la cuenta en régimen estándar (micro_w_alloc -> 0), haciendo que
        // max_fee_limit caiga al 1.0% de margen. Sin embargo, el dimensionamiento de
        // posición prudente asigna un margen pequeño ($0.60–$1.50 USD) que OBLIGA a un
        // apalancamiento L ≈ 5–10 para satisfacer el nocional mínimo de Binance ($5 USD).
        // La comisión roundtrip multiplicada por L (0.0012 · 8 = 0.0096..0.0120) superaba
        // el 1.0% de margen y rechazaba el 100% de las órdenes válidas con REJ_FEE_IMPACT.
        //
        // Solución continua: si el margen individual de la posición está en escala micro
        // (final_margin < 3 · safe_min_notional), la posición requiere apalancamiento
        // para alcanzar el nocional mínimo del exchange, por lo que la tolerancia sobre
        // margen conserva el techo micro (0.035 = 3.5% de margen), garantizando la
        // viabilidad sin relajar los frenos ante fricciones auténticamente excesivas.
        let rescued_for_min_notional = dynamic_leverage > 1.0
            && final_margin <= safe_min_notional * 1.05
            && max_acceptable_fee_pct >= 0.005;
        let effective_max_fee_limit = if rescued_for_min_notional {
            max_fee_limit.max(micro_fee_ceiling)
        } else {
            max_fee_limit
        };
        if !final_fee_impact.is_finite() || final_fee_impact > effective_max_fee_limit {
            return rej(REJ_FEE_IMPACT);
        }
        // CL-6: ninguna orden validada queda bajo el nocional mínimo del
        // símbolo. Antes nada lo re-verificaba tras el segundo rescate y la
        // orden salía con un nocional que el exchange rechaza.
        if !(final_margin * dynamic_leverage >= safe_min_notional * (1.0 - 1e-9)) {
            return rej(REJ_MIN_NOTIONAL);
        }
        // (fusión PR #5: la viabilidad de margen mínimo ya se exigió arriba con
        // rej(6); el duplicado del hunk se elimina)

        let raw_regime = arena.market_regime.load(Ordering::Relaxed);
        let regime = crate::regime::MarketRegime::from(raw_regime);

        if !orchestrator.allow_trade(
            bounded_exposure > 0.0,
            final_margin,
            regime,
            dynamic_min_notional,
        ) {
            return rej(REJ_ORCHESTRATOR);
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

        // F1-086 (Ola Ω190): Publicar el riesgo realmente realizado por la orden
        // emitida (`safe_vol · safe_lev · stop_real / allocated_capital`) en el
        // registro omnisciente `lh04_f_realizado`. Si la orden fue recortada
        // downstream por límite de margen o colchón del portfolio (Sol A2),
        // `lh04_f_realizado` refleja la fracción real de capital que se arriesga al
        // stop, sin mentir publicando f_objetivo cuando el colateral ejecutado fue menor.
        if lh04_f_efectiva > 0.0 && allocated_capital > 0.0 {
            let f_realizado_final = (safe_vol * safe_lev * stop_real / allocated_capital)
                .min(lh04_f_efectiva);
            arena
                .registry
                .set_for_coin(coin_id, "lh04_f_realizado", f_realizado_final);
        }

        // D-744: el RIESGO REALMENTE TOMADO por esta orden —lo que se pierde
        // si su stop se toca, en fracción del capital— alimenta la media móvil
        // que convierte una caída observada en evidencia. Sin esta medida, el
        // cortacircuitos de drawdown es una opinión sobre un número inventado.
        // D-744c (auditoría PR #5): se mide DESPUÉS del último rechazo de
        // esta función; antes también entraban órdenes que aquí mismo se
        // rechazaban por geometría inválida.
        // CL-7: el stop es el QUE LA ORDEN LLEVA (`expected_loss`, ya con el
        // tope micro de 55 pb o el objetivo explícito de la intención), no el
        // difusivo sin acotar de `tpsl_gate`: éste registraba más riesgo del
        // que se tomaba y aflojaba el cortacircuitos en la misma proporción.
        // CL-131/CL-135: la medida queda PENDIENTE; el núcleo la toma al
        // abrir la orden y la registra al cerrarla, una vez por operación.
        let sl_pct = expected_loss;
        if sl_pct > 0.0 && current_cap > 0.0 {
            // CL-106b: el riesgo también en USD, que no depende del capital
            // al dimensionar. CL-106: la geometría de la orden en unidades de
            // su stop, con la fricción, para el umbral del freno duro.
            let riesgo_usd = safe_vol * safe_lev * sl_pct;
            let ganancia = (expected_win - roundtrip_fee) / sl_pct;
            let perdida = (sl_pct + roundtrip_fee) / sl_pct;
            self.medida_pendiente = Some(MedidaOrden {
                coin_id,
                riesgo: (riesgo_usd / current_cap).clamp(0.0, 1.0),
                riesgo_usd,
                geometria: (ganancia.is_finite() && perdida.is_finite() && perdida > 0.0)
                    .then_some((ganancia, perdida)),
            });
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
#[allow(dead_code)]
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
    /// #651 — CONVERSIÓN DE UNIDADES Lundberg: R_nocional/L contra R
    /// crudo. La cota en unidades de capital es L·(ln(1/eps)/R_nocional);
    /// con L=10 el margen capital es 10x el nocional (equivalentemente
    /// R_capital = R_nocional/10).
    #[test]
    fn qo_651_lundberg_conversion_de_unidades() {
        let m_nocional =
            crate::cramer_lundberg::EstimadorSiniestros::margen_de_cota(0.02, 0.05).unwrap();
        let m_capital_l10 =
            crate::cramer_lundberg::EstimadorSiniestros::margen_de_cota(0.02 / 10.0, 0.05)
                .unwrap();
        assert!(
            (m_capital_l10 / m_nocional - 10.0).abs() < 1e-9,
            "margen capital = L x margen nocional: {} vs {}",
            m_capital_l10,
            m_nocional
        );
        // Sin apalancamiento (L=1) la conversión es identidad.
        let m_l1 =
            crate::cramer_lundberg::EstimadorSiniestros::margen_de_cota(0.02 / 1.0, 0.05)
                .unwrap();
        assert!((m_l1 - m_nocional).abs() < 1e-12);
    }
    /// #653 — el lerp micro compone con la cota MEDIDA (D-641 × D-744b):
    /// umbral = lerp(dd_max_medido, 0.85, micro_w). En régimen estándar la
    /// cota medida rige; en micro pleno la tolerancia 0.85 la relaja — pero
    /// el veto NUNCA desaparece (antes el lerp era código muerto: sólo vivía
    /// en el if-guard y el veto comparaba contra el gen crudo).
    #[test]
    fn qo_653_dd_lerp_micro_compone_con_cota_medida() {
        // Estándar (w=0): la cota medida rige intacta.
        assert_eq!(crate::capital_regime::lerp(0.20, 0.85, 0.0), 0.20);
        // Micro pleno (w=1): tolerancia 0.85 — relajada pero PRESENTE.
        assert_eq!(crate::capital_regime::lerp(0.20, 0.85, 1.0), 0.85);
        // Transición continua y monótona.
        let medio = crate::capital_regime::lerp(0.20, 0.85, 0.5);
        assert!((medio - 0.525).abs() < 1e-12);
        // La relajación micro nunca BAJA el umbral estándar (sólo sube).
        assert!(crate::capital_regime::lerp(0.20, 0.85, 0.5) > 0.20);
    }
}
