use crate::atomic_float::AtomicF64;
use crate::config::QuantumConfig;
pub use omniscient_registry::OmniscientRegistry;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

/// Axioma V: Cohesión Celular Absoluta.
/// Los motores Scalp y Swing no se pisan porque operan en structs aislados,
/// pero unidos dentro del mismo bloque contiguo de RAM (GlobalArena).
/// D-605 (DÉCIMA OLA): capacidad de monedas del arena. El arena es un bloque
/// contiguo de tamaño fijo (`Box<[CoinArena; MAX_COINS]>`, bloqueado en RAM), y
/// el núcleo dimensionaba sus vectores por moneda con su propio literal 30. Una
/// sola constante evita que ambos diverjan.
pub const MAX_COINS: usize = 30;

#[repr(C, align(64))]
pub struct ScalpState {
    pub pnl_realized: AtomicF64,
    pub pnl_unrealized: AtomicF64,
    pub pnl_gross: AtomicF64,
    pub gross_wins: AtomicF64,
    pub gross_losses: AtomicF64,
    pub active_positions: AtomicUsize,
    pub win_rate: AtomicF64,
    pub profit_factor: AtomicF64,
    pub kelly_fraction: AtomicF64,
    pub roi_pre_fee: AtomicF64,
    pub roi_post_fee: AtomicF64,
    pub trade_count: AtomicUsize,
    pub zombie_promotions: AtomicUsize,
}

impl ScalpState {
    pub fn new(w_base: f64) -> Self {
        Self {
            pnl_realized: AtomicF64::new(0.0),
            pnl_unrealized: AtomicF64::new(0.0),
            pnl_gross: AtomicF64::new(0.0),
            gross_wins: AtomicF64::new(0.0),
            gross_losses: AtomicF64::new(0.0),
            active_positions: AtomicUsize::new(0),
            win_rate: AtomicF64::new(w_base), // Derived from genome
            profit_factor: AtomicF64::new(1.50), // Prior Bayesiano neutro optimista (R:R >= 1.5:1)
            kelly_fraction: AtomicF64::new(0.25), // Prior Bayesiano (Quarter-Kelly)
            roi_pre_fee: AtomicF64::new(0.0),
            roi_post_fee: AtomicF64::new(0.0),
            trade_count: AtomicUsize::new(0),
            zombie_promotions: AtomicUsize::new(0),
        }
    }
}

/// U-1 (MOTOR UNIVERSAL): `SwingState` EXTIRPADO — gemelo de `ScalpState`
/// cuyo único slot vivía en el campo zombi `coin.swing` (sin escritores
/// desde F-014). La métrica del motor continuo es `coin.metrics`.

/// Pre-allocated ring buffer size. Power of 2 for branchless modulo via bitmask.
pub const TICK_RING_SIZE: usize = 32768; // 2^15 = 32K ticks (~1MB per coin)
const TICK_RING_MASK: usize = TICK_RING_SIZE - 1;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CompactTick {
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

/// Lock-free ring buffer for tick data.
/// Single-producer (WS thread), multi-reader (Darwin, Telemetry).
/// Uses UnsafeCell + AtomicUsize for zero-lock writes in the hot path.
/// Safety: Single producer guaranteed by architecture (one WS thread per coin).
pub struct LockFreeTickRing {
    buffer: UnsafeCell<[CompactTick; TICK_RING_SIZE]>,
    head: AtomicUsize,
    len: AtomicUsize,
}

// Safety: Single-producer architecture. Reads are best-effort snapshots.
unsafe impl Sync for LockFreeTickRing {}
unsafe impl Send for LockFreeTickRing {}

impl LockFreeTickRing {
    pub fn new() -> Self {
        Self {
            buffer: UnsafeCell::new([CompactTick::default(); TICK_RING_SIZE]),
            head: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
        }
    }

    /// O(1) lock-free push. ~5ns on modern x86.
    #[inline(always)]
    pub fn push(&self, tick: CompactTick) {
        let current_head = self.head.load(Ordering::Relaxed);
        let idx = current_head & TICK_RING_MASK;
        // Safety: single-producer guaranteed by architecture
        unsafe {
            (*self.buffer.get())[idx] = tick;
        }
        // Publish the new head with Release ordering so readers acquire the writes
        self.head
            .store(current_head.wrapping_add(1), Ordering::Release);
        // Track fill level atómicamente sin condiciones de carrera (cap at TICK_RING_SIZE)
        let _ = self
            .len
            .fetch_update(Ordering::Release, Ordering::Relaxed, |len| {
                if len < TICK_RING_SIZE {
                    Some(len + 1)
                } else {
                    None
                }
            });
    }

    /// Returns current length of valid data
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot read for Darwin daemon: copies recent ticks to output array.
    /// Zero-allocation lock-free snapshot.
    pub fn snapshot_recent_into(&self, max_ticks: usize, out: &mut [CompactTick]) -> usize {
        let current_len = self.len.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        let count = max_ticks.min(current_len).min(out.len());

        let buf = unsafe { &*self.buffer.get() };

        // Read from oldest to newest
        let start = if current_len >= TICK_RING_SIZE {
            head.wrapping_sub(count) & TICK_RING_MASK
        } else {
            head.saturating_sub(count)
        };

        for (i, out_item) in out.iter_mut().enumerate().take(count) {
            let idx = (start + i) & TICK_RING_MASK;
            *out_item = buf[idx];
        }
        count
    }

    /// Snapshot read for Darwin daemon: copies recent ticks to output Vec.
    /// Not lock-free from the reader side (memcpy), but never blocks the writer.
    pub fn snapshot_recent(&self, max_ticks: usize) -> Vec<CompactTick> {
        let current_len = self.len.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        let count = max_ticks.min(current_len);

        let mut out = Vec::with_capacity(count);
        let buf = unsafe { &*self.buffer.get() };

        // Read from oldest to newest
        let start = if current_len >= TICK_RING_SIZE {
            head.wrapping_sub(count) & TICK_RING_MASK
        } else {
            head.saturating_sub(count)
        };

        for i in 0..count {
            let idx = (start + i) & TICK_RING_MASK;
            out.push(buf[idx]);
        }
        out
    }
}

impl Default for LockFreeTickRing {
    fn default() -> Self {
        Self::new()
    }
}

/// Estado aislado por moneda
#[repr(C, align(64))]
pub struct CoinArena {
    /// U-1 (MOTOR UNIVERSAL): métrica ÚNICA por moneda. Los slots gemelos
    /// `scalp`/`swing` eran ZOMBIES desde F-014 (sin escritores) mientras
    /// 52 lectores (dashboards, shadow_certifier, mmap) seguían sumando
    /// contadores muertos — win rates y PnL mostrados estructuralmente en
    /// cero. Extirpados: el compilador es el inventario.
    pub metrics: ScalpState,
    pub positions: crate::position::PositionManager,
    pub current_price: AtomicF64,
    pub ml_prob: AtomicF64,
    pub current_atr: AtomicF64,
    /// D-754 — σ PRONOSTICADA por el espectro predictivo en cada ancla
    /// (fracción de precio, `spectral_tape::ANCLAS_PRONOSTICO`). El núcleo
    /// sólo la publica cuando el pronosticador ha DEMOSTRADO habilidad fuera
    /// de muestra frente a la climatología: un 0 significa «no hay pronóstico
    /// con evidencia», y el consumidor cae al ATR hacia atrás de siempre.
    pub sigma_forecast: [AtomicF64; 7],
    /// D-745 — τ DOMINANTE MEDIDA del espectro temporal de esta moneda (ms).
    /// El núcleo la publica en cada tick; el risk-engine la usa como horizonte
    /// de la orden cuando la señal no declara duración propia. Antes ese
    /// respaldo era el gen estático `temporal_scale`, de modo que el
    /// dimensionado y la gestión de la MISMA posición vivían en horizontes
    /// distintos. 0 = el espectro aún no ha arrancado.
    pub dominant_tau_ms: AtomicF64,
    /// #594 — IC prequential MEDIDO de la escala τ* publicada (habilidad
    /// direccional señal→retorno del bloque propio). τ* sólo fue elegida
    /// por habilidad si este valor es > 0; 0.0 o negativo = respaldo de
    /// energía o escala sin evidencia — contable para el consejo.
    pub tau_habilidad: AtomicF64,
    pub hurst_exponent: AtomicF64,
    /// S-7 (ESPECTRALIZACIÓN): Hurst multifractal SELECCIONADO POR τ — el
    /// H de la escala que el motor opera AHORA (micro si τ<2min, meso si
    /// τ<1h, macro después). `hurst_exponent` sigue siendo el escalar
    /// global; `compute_tp_sl` consume ESTE: la geometría del trade usa la
    /// persistencia del horizonte que realmente tradea.
    pub hurst_scale_matched: AtomicF64,
    /// Universo Multivariante Continuo Temporal Espectral: Coherencia armónica de fase global [-1.0, 1.0]
    pub spectral_coherence: AtomicF64,
    /// Entropía espectral de Shannon normalizada [0.0, 1.0] (0 = orden láser, 1 = ruido térmico)
    pub spectral_entropy: AtomicF64,
    /// Longitud de onda o centro de masa espectral armónico continuo tau* (ms)
    pub spectral_resonant_tau: AtomicF64,
    /// (Ola XLI·B1) Crash-ness continua [0,1] del campo espectral — ver
    /// spectral_regime::SpectralRegimeField. 0 = calma.
    pub spectral_crash_flux: AtomicF64,
    /// (Ola XLI·C2) Intermitencia χ = (1−ζ3)⁺ de las funciones de estructura
    /// de Kolmogorov: colas más gruesas que la autosimilaridad K41.
    pub spectral_intermittency: AtomicF64,
    /// (Ola XLI·C3) Información de Fisher de la masa espectral respecto a
    /// ln(τ): identificabilidad del régimen (None→negativo = frío).
    pub spectral_fisher: AtomicF64,
    /// P-3b — OPEN INTEREST per-símbolo (normalizado log contra $100M de
    /// contratos abiertos, [0,1]): el dinero apalancado DENTRO de esta
    /// moneda. Alimenta el asiento Ente del Mercado. Escrito por el poller
    /// REST (120s), leído en la deliberación — fuera del hot path.
    pub open_interest_norm: AtomicF64,
    /// F-A1 (Fable, 2026-10-17) — FUNDING PREVISTO per-símbolo: tasa que el
    /// exchange aplicará en el PRÓXIMO asentamiento (`lastFundingRate` de
    /// `/fapi/v1/premiumIndex`, fracción del nocional; positiva = los largos
    /// pagan a los cortos). La escribe el poller REST del host (120 s) con su
    /// marca de tiempo; la lee el risk-engine como coste de carry esperado de
    /// la orden sobre su horizonte τ (ver `expected_funding_cost`). NaN =
    /// sin dato: el consumidor cae a 0 (sin carry), nunca propaga el NaN.
    pub funding_rate_next: AtomicF64,
    /// Marca (ms época) de la última publicación de `funding_rate_next`;
    /// 0 = nunca. Con antigüedad > `FUNDING_TTL_MS` el dato cuenta como
    /// AUSENTE (fail-closed a «sin dato», no a un valor viejo).
    pub funding_rate_ts_ms: AtomicU64,
    /// Epigenetic Multipliers (Memory of past success/failure)
    pub epigenetic_bias: AtomicF64,
    pub epigenetic_threshold_modifier: AtomicF64,
    /// Tracking del mercado Spot (Leading Indicator para microestructura)
    pub spot_bid: AtomicF64,
    pub spot_ask: AtomicF64,
    pub spot_bid_qty: AtomicF64,
    pub spot_ask_qty: AtomicF64,
    /// CL-79 (F3.7-03): marca de llegada del último spot, en el reloj de
    /// eventos del núcleo (`registry.get_event_time()`, la marca del último
    /// evento del futuro procesado). El bookTicker de spot no trae la hora
    /// del exchange, y el TTL del núcleo compara con ese mismo reloj, en vivo
    /// y en los replays. 0 = nunca llegó.
    pub spot_ts_ms: AtomicU64,
    /// Tracking de CVD (Cumulative Volume Delta) - Flujo de Capital
    pub agg_buy_vol: AtomicF64,
    pub agg_sell_vol: AtomicF64,
    /// #660 (F2-B5) — reloj del último agg trade: el decaimiento del flujo
    /// agregado es TIEMPO FÍSICO (τ=60 s), no 0.995 por evento (la memoria
    /// variaba ×100 entre feeds de 1 y 100 ev/s).
    pub last_agg_trade_ms: AtomicU64,
    /// (Ola XLII·A3a) Estadística medida del CVD rodante para su
    /// z-tipificación: media EWMA y segundo momento EWMA (de donde sale
    /// σ). Los vetos/confluencias de flujo dejan de leer literales en
    /// puntos básicos y pasan a leer CUÁNTO DE SU PROPIA DISTRIBUCIÓN
    /// se aparta el flujo actual. 0 = aún en calentamiento.
    pub cvd_mean_ewma: AtomicF64,
    pub cvd_sq_ewma: AtomicF64,
    /// Ola XLIV: peso de corrección de sesgo de las EWMAs del CVD
    /// (EWMA de la constante 1: `1 − (1 − α)^t`).
    pub cvd_ewma_peso: AtomicF64,
    /// (Ola XLII·A3b) Estadística del CAMPO espectral para los extremos de
    /// X-016: segundo momento de la marea neutra (swing·0.6+secular·0.4) y
    /// media/segundo momento de la entropía. Los «extremos» pasan de
    /// literales a z-scores contra la historia del propio campo.
    pub tide_sq_ewma: AtomicF64,
    pub entropy_mean_ewma: AtomicF64,
    pub entropy_sq_ewma: AtomicF64,
    /// Ola XLIV: peso de corrección de sesgo compartido por las EWMAs de
    /// marea y entropía (se actualizan juntas, con la misma α).
    pub campo_ewma_peso: AtomicF64,
    /// Tracking de Liquidez Profunda L2
    pub l2_bid_wall: AtomicF64, // Volumen acumulado en bids
    pub l2_ask_wall: AtomicF64, // Volumen acumulado en asks
    /// Lock-free ring buffer: zero contention in hot path
    pub tick_ring: LockFreeTickRing,
    pub tick_head: AtomicUsize,
    pub last_close_ts: AtomicU64,
    pub last_scalp_close_ts: AtomicU64,
    pub last_swing_close_ts: AtomicU64,
    pub last_close_is_long: AtomicBool,
    pub last_close_was_win: AtomicBool,
    pub last_close_reason: AtomicU8,
    pub last_close_exit_price: AtomicF64,
    pub last_tick_timestamp_ms: AtomicU64,
    /// F1-004: Registro del último trade cerrado en esta moneda con los rasgos
    /// causales medidos en la apertura [obi, accel, spread_bps, atr_pct, hurst, velocity].
    /// Consumido por el online_daemon para el aprendizaje del Shadow Forest sin fuga temporal.
    pub last_closed_trade: std::sync::Mutex<Option<ClosedTradeRecord>>,
}

/// F1-004: Registro de trade cerrado con rasgos fotografiados en la apertura.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosedTradeRecord {
    pub is_long: bool,
    pub net_pnl_pct: f64,
    pub entry_shadow_features: [f64; 6],
    pub close_ts_ms: u64,
    pub generation: u64,
}

impl Default for ClosedTradeRecord {
    fn default() -> Self {
        Self {
            is_long: true,
            net_pnl_pct: 0.0,
            entry_shadow_features: [0.0; 6],
            close_ts_ms: 0,
            generation: 0,
        }
    }
}

impl CoinArena {
    /// F-A1 — intervalo canónico de asentamiento del funding en Binance
    /// USDⓈ-M perpetuos: 8 h. Una posición abierta en fase uniforme respecto
    /// al ciclo cruza en media τ/T_F asentamientos: E[N] = τ/T_F exactamente.
    pub const FUNDING_INTERVAL_MS: f64 = 8.0 * 3_600_000.0;
    /// F-A1 — TTL del dato de funding: tres ciclos del poller (120 s).
    pub const FUNDING_TTL_MS: u64 = 360_000;

    /// F-A1 — publica la tasa prevista del próximo asentamiento (poller del
    /// host, fuera del hot path). Valores no finitos se ignoran; la tasa se
    /// acota a [-1, 1] como en el multiplexer (FIX #1537).
    #[inline]
    pub fn publish_funding_next(&self, rate: f64, now_ms: u64) {
        if rate.is_finite() && now_ms > 0 {
            self.funding_rate_next
                .store(rate.clamp(-1.0, 1.0), Ordering::Relaxed);
            self.funding_rate_ts_ms.store(now_ms, Ordering::Release);
        }
    }

    /// F-A1 — tasa prevista vigente en `now_ms`, o `None` si nunca se
    /// publicó, caducó (TTL) o no es finita.
    #[inline]
    pub fn funding_next_at(&self, now_ms: u64) -> Option<f64> {
        let ts = self.funding_rate_ts_ms.load(Ordering::Acquire);
        if ts == 0 || now_ms < ts || now_ms - ts > Self::FUNDING_TTL_MS {
            return None;
        }
        let f = self.funding_rate_next.load(Ordering::Relaxed);
        if f.is_finite() {
            Some(f)
        } else {
            None
        }
    }

    /// F-A1 — coste de carry ESPERADO (fracción del nocional, ≥ 0) de una
    /// posición con dirección `is_long` mantenida `tau_ms`:
    /// E[carry] = d·f·τ/T_F, d = +1 largo, −1 corto.
    /// Sólo se cobra el COSTE (d·f > 0). El ingreso esperado se acota
    /// inferiormente en 0: `f` es un pronóstico del próximo asentamiento, y
    /// un pronóstico de cobro no es evidencia para abrir una posición
    /// (inmunidad a la ruina antes que EV nominal). Sin dato ⇒ 0.0, que es
    /// exactamente el comportamiento anterior a F-A1.
    #[inline]
    pub fn expected_funding_cost(&self, is_long: bool, tau_ms: f64, now_ms: u64) -> f64 {
        let Some(f) = self.funding_next_at(now_ms) else {
            return 0.0;
        };
        if !tau_ms.is_finite() || tau_ms <= 0.0 {
            return 0.0;
        }
        let d = if is_long { 1.0 } else { -1.0 };
        let carry = d * f * (tau_ms / Self::FUNDING_INTERVAL_MS);
        if carry.is_finite() {
            carry.max(0.0)
        } else {
            0.0
        }
    }

    /// D-754 — σ PRONOSTICADA AL HORIZONTE τ, interpolada en log τ entre las
    /// anclas publicadas. Fuera del rango de anclas se toma la del extremo:
    /// extrapolar una ley de escala más allá de donde se ha medido es
    /// exactamente lo que el espectro temporal hacía cuando opinaba con
    /// escalas que no había observado (D-742).
    ///
    /// `None` si el pronosticador todavía no ha demostrado habilidad (el
    /// núcleo publica ceros): el llamador debe caer a la volatilidad medida
    /// hacia atrás, nunca inventarse un pronóstico.
    pub fn sigma_forecast_at(&self, tau_ms: f64) -> Option<f64> {
        if !tau_ms.is_finite() || tau_ms <= 0.0 {
            return None;
        }
        let anclas = crate::spectral_tape::ANCLAS_PRONOSTICO;
        let mut v = [0.0f64; 7];
        let mut any_positive = false;
        for (i, a) in self.sigma_forecast.iter().enumerate() {
            let val = a.load(std::sync::atomic::Ordering::Relaxed);
            v[i] = val;
            if val.is_finite() && val > 0.0 {
                any_positive = true;
            }
        }
        if !any_positive {
            return None;
        }
        let lt = tau_ms.ln();
        if lt <= anclas[0].ln() {
            return if v[0] > 0.0 { Some(v[0]) } else { None };
        }
        let ultimo = anclas.len() - 1;
        if lt >= anclas[ultimo].ln() {
            return if v[ultimo] > 0.0 { Some(v[ultimo]) } else { None };
        }
        let mut i = 0usize;
        while i + 1 < anclas.len() && anclas[i + 1].ln() < lt {
            i += 1;
        }
        let (l0, l1) = (anclas[i].ln(), anclas[i + 1].ln());
        let (s0, s1) = (v[i], v[i + 1]);
        if !(s0 > 0.0 && s1 > 0.0) {
            return None;
        }
        let f = ((lt - l0) / (l1 - l0)).clamp(0.0, 1.0);
        // Interpolación geométrica: σ escala como potencia de τ, de modo que
        // en log-log la recta es la ley de difusión y no una cuerda arbitraria.
        Some((s0.ln() * (1.0 - f) + s1.ln() * f).exp())
    }

    /// O(1) lock-free tick push. ~5 nanoseconds.
    #[inline(always)]
    pub fn push_tick(&self, tick: CompactTick) {
        self.last_tick_timestamp_ms.store(tick.timestamp, Ordering::Relaxed);
        self.tick_ring.push(tick);
        self.tick_head.fetch_add(1, Ordering::Relaxed);
    }

    /// Retorna el sesgo epigenético activo aplicando decaimiento homeostático continuo Ornstein-Uhlenbeck
    /// hacia 1.0 (equilibrio físico sin histeresis no-ergódica petrificada).
    #[inline]
    pub fn get_active_epigenetic_bias(&self, now_ms: u64) -> f64 {
        let stored = self.epigenetic_bias.load(Ordering::Relaxed);
        let last_close = self.last_close_ts.load(Ordering::Relaxed);
        Self::epigenetic_bias_at(stored, last_close, now_ms)
    }

    #[inline]
    pub fn epigenetic_bias_at(stored: f64, last_close: u64, now_ms: u64) -> f64 {
        if last_close == 0 || now_ms <= last_close {
            return stored;
        }
        let elapsed_ms = (now_ms - last_close) as f64;
        // Relajación homeostática Ornstein-Uhlenbeck hacia 1.0 (tau_homeo = 30 minutos = 1,800,000 ms)
        let decay = (-elapsed_ms / 1_800_000.0).exp();
        let bias = 1.0 + (stored - 1.0) * decay;
        if bias.is_finite() {
            bias.clamp(0.50, 2.00)
        } else {
            1.0
        }
    }

    /// Retorna el modificador epigenético de umbral activo aplicando decaimiento homeostático continuo
    /// hacia 1.0 (equilibrio físico sin estados absorbentes).
    #[inline]
    pub fn get_active_epigenetic_threshold(&self, now_ms: u64) -> f64 {
        let stored = self.epigenetic_threshold_modifier.load(Ordering::Relaxed);
        let last_close = self.last_close_ts.load(Ordering::Relaxed);
        Self::epigenetic_threshold_at(stored, last_close, now_ms)
    }

    #[inline]
    fn epigenetic_threshold_at(stored: f64, last_close: u64, now_ms: u64) -> f64 {
        if last_close == 0 || now_ms <= last_close {
            return stored;
        }
        let elapsed_ms = (now_ms - last_close) as f64;
        // Relajación homeostática hacia 1.0 (tau_homeo = 15 minutos = 900,000 ms)
        let decay = (-elapsed_ms / 900_000.0).exp();
        1.0 + (stored - 1.0) * decay
    }

    /// Quantum Epigenetics: Modifies the genetic bias and thresholds continuously based on trade feedback.
    ///
    /// **QUÉ:** Función de retroalimentación epigenética que ajusta la confianza y los umbrales de entrada
    /// basándose en el resultado de cada operación cerrada, POR MONEDA.
    ///
    /// **POR QUÉ:** Las estrategias estáticas no se adaptan al cambio de régimen. La epigenética permite
    /// que cada moneda "recuerde" su historial reciente de PnL y ajuste su agresividad.
    ///
    /// **CÓMO:**
    /// - `epigenetic_bias` (0.5x → 2.0x): Multiplicador de confianza. Win → sube, Loss → baja.
    /// - `epigenetic_threshold_modifier` (0.8x → 1.25x): Multiplicador del min_confidence_cutoff.
    ///   Loss → sube moderadamente con disipación temporal continua, Win rápido → baja suavemente.
    ///
    /// **CUÁNDO:** Se llama al cerrar cada posición (scalp o swing) en process_tick y process_tick_shadow.
    /// **DÓNDE:** CoinArena (crates/quantum-arena/src/state.rs).
    /// **QUIÉN:** GodEngineCore invoca esto tras cada cierre de posición.
    /// Retroalimentación Epigenética Adaptativa Multivariante Espectral.
    ///
    /// Integra la escala temporal de operación `tau_trade_ms` y el timestamp causal `event_time_ms`.
    pub fn apply_spectral_epigenetic_feedback_with_time(
        &self,
        pnl_pct: f64,
        trade_duration_ms: u64,
        tau_trade_ms: f64,
        event_time_ms: u64,
    ) {
        // Reject invalid evidence before changing either adaptation or its clock.
        if !pnl_pct.is_finite() {
            return;
        }
        // event_time_ms == 0 is the wall-clock sentinel (RET-32 removed the
        // wrappers that passed it, but core callers can still pass 0).
        let close_ts = if event_time_ms > 0 {
            event_time_ms
        } else {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        };
        let last_close = self.last_close_ts.load(Ordering::Relaxed);
        // Distinct slots may close in the same event; equality is valid.
        if close_ts < last_close {
            return;
        }
        let stored_bias = self.epigenetic_bias.load(Ordering::Relaxed);
        let stored_threshold = self.epigenetic_threshold_modifier.load(Ordering::Relaxed);
        if !stored_bias.is_finite() || !stored_threshold.is_finite() {
            return;
        }
        let safe_tau = if tau_trade_ms.is_finite() && tau_trade_ms > 10.0 {
            tau_trade_ms
        } else {
            30_000.0
        };
        // Las ondas rápidas (tau bajo) operan con mayor agilidad; las ondas lentas poseen inercia secular
        let tau_inertia = (safe_tau / 3_600_000.0).clamp(0.10, 2.0);
        let pnl_shift = (pnl_pct * (4.0 / tau_inertia)).clamp(-0.15, 0.15);

        // Advance the old bias to this event before applying its feedback.
        // F3.4-07 (Ola Ω109): Relajación continua Ornstein-Uhlenbeck para erradicar petrificación no-ergódica.
        let old_bias = Self::epigenetic_bias_at(stored_bias, last_close, close_ts);
        let new_bias = (old_bias + pnl_shift).clamp(0.50, 2.00);

        // Modificador de umbral adaptativo continuo con proporcionalidad de pérdida y saturación controlada
        let duration_factor = 1.0 - (trade_duration_ms as f64 / (safe_tau * 3.0)).clamp(0.0, 1.0);
        let pnl_mag = (pnl_pct.abs() / 0.002).clamp(0.15, 2.5);
        let threshold_shift = if pnl_pct > 0.0 {
            (-0.006 - 0.015 * duration_factor) * pnl_mag
        } else {
            (0.010 * (1.0 / tau_inertia.sqrt()).clamp(0.5, 1.5)) * pnl_mag
        };

        // Advance the old state to this event before applying its feedback.
        // Restarting decay from the stored value would resurrect expired memory.
        let old_threshold = Self::epigenetic_threshold_at(stored_threshold, last_close, close_ts);
        let new_threshold = (old_threshold + threshold_shift).clamp(0.80, 1.25);
        self.epigenetic_bias.store(new_bias, Ordering::Relaxed);
        self.epigenetic_threshold_modifier
            .store(new_threshold, Ordering::Relaxed);

        // Actualizar métricas universales de cierre con timestamp del evento causal.
        // As before, these separate atomics require a serialized feedback writer;
        // this preflight is not a transaction between concurrent writers.
        self.last_close_ts.store(close_ts, Ordering::Relaxed);
        self.last_scalp_close_ts.store(close_ts, Ordering::Relaxed);
        self.last_swing_close_ts.store(close_ts, Ordering::Relaxed);
        self.last_close_was_win.store(pnl_pct > 0.0, Ordering::Relaxed);
    }
}

impl CoinArena {
    pub fn new(w_base: f64) -> Self {
        Self {
            metrics: ScalpState::new(w_base),
            positions: crate::position::PositionManager::default(),
            current_price: AtomicF64::new(0.0),
            ml_prob: AtomicF64::new(w_base),
            current_atr: AtomicF64::new(0.0),
            dominant_tau_ms: AtomicF64::new(0.0),
            tau_habilidad: AtomicF64::new(0.0),
            sigma_forecast: [
                AtomicF64::new(0.0),
                AtomicF64::new(0.0),
                AtomicF64::new(0.0),
                AtomicF64::new(0.0),
                AtomicF64::new(0.0),
                AtomicF64::new(0.0),
                AtomicF64::new(0.0),
            ],
            hurst_exponent: AtomicF64::new(0.5),
            hurst_scale_matched: AtomicF64::new(0.5),
            spectral_coherence: AtomicF64::new(0.0),
            spectral_entropy: AtomicF64::new(1.0),
            spectral_resonant_tau: AtomicF64::new(30_000.0),
            spectral_crash_flux: AtomicF64::new(0.0),
            spectral_intermittency: AtomicF64::new(0.0),
            spectral_fisher: AtomicF64::new(-1.0),
            open_interest_norm: AtomicF64::new(0.0),
            funding_rate_next: AtomicF64::new(f64::NAN),
            funding_rate_ts_ms: AtomicU64::new(0),
            epigenetic_bias: AtomicF64::new(1.0),
            epigenetic_threshold_modifier: AtomicF64::new(1.0),
            spot_bid: AtomicF64::new(0.0),
            spot_ask: AtomicF64::new(0.0),
            spot_bid_qty: AtomicF64::new(0.0),
            spot_ask_qty: AtomicF64::new(0.0),
            spot_ts_ms: AtomicU64::new(0),
            agg_buy_vol: AtomicF64::new(0.0),
            last_agg_trade_ms: AtomicU64::new(0),
            cvd_mean_ewma: AtomicF64::new(0.0),
            cvd_sq_ewma: AtomicF64::new(0.0),
            cvd_ewma_peso: AtomicF64::new(0.0),
            tide_sq_ewma: AtomicF64::new(0.0),
            campo_ewma_peso: AtomicF64::new(0.0),
            // Estos son acumuladores SIN normalizar y campo_ewma_peso nace
            // en cero. Sembrarlos en uno inventaría masa no incluida en peso.
            entropy_mean_ewma: AtomicF64::new(0.0),
            entropy_sq_ewma: AtomicF64::new(0.0),
            agg_sell_vol: AtomicF64::new(0.0),
            l2_bid_wall: AtomicF64::new(0.0),
            l2_ask_wall: AtomicF64::new(0.0),
            tick_ring: LockFreeTickRing::new(),
            tick_head: AtomicUsize::new(0),
            last_close_ts: AtomicU64::new(0),
            last_scalp_close_ts: AtomicU64::new(0),
            last_swing_close_ts: AtomicU64::new(0),
            last_close_is_long: AtomicBool::new(false),
            last_close_was_win: AtomicBool::new(false),
            last_close_reason: AtomicU8::new(0),
            last_close_exit_price: AtomicF64::new(0.0),
            last_tick_timestamp_ms: AtomicU64::new(0),
            last_closed_trade: std::sync::Mutex::new(None),
        }
    }

    /// F1-004: Registra el último trade cerrado con sus rasgos causales de entrada.
    #[inline]
    pub fn record_closed_trade(&self, record: ClosedTradeRecord) {
        if let Ok(mut g) = self.last_closed_trade.lock() {
            *g = Some(record);
        }
    }

    /// F1-004: Extrae el último trade cerrado consumiéndolo para el Shadow Forest.
    #[inline]
    pub fn take_last_closed_trade(&self) -> Option<ClosedTradeRecord> {
        self.last_closed_trade.lock().ok().and_then(|mut g| g.take())
    }
}

/// GlobalArena: El hipergrafo en memoria que todos los hilos leen y escriben.
/// Contiene configuración atómica y estado aislado por horizonte de tiempo.
/// 100% lock-free. Zero Mutex, Zero RwLock.
#[repr(C, align(64))]
pub struct GlobalArena {
    pub config: QuantumConfig,
    pub coins: Box<[CoinArena; MAX_COINS]>,

    // Portfolio & Risk
    pub unified_capital: AtomicF64,
    pub used_margin: AtomicF64,
    /// D-744 — RIESGO POR OPERACIÓN REALMENTE TOMADO (fracción del capital
    /// que se pierde si el stop de la orden se toca), como media móvil
    /// exponencial de las OPERACIONES (CL-131/CL-135: una vez por operación,
    /// al cerrarla, si su entrada llegó al exchange o es simulación aislada).
    /// Es la magnitud que convierte una caída observada en evidencia: sin
    /// ella, un tope de drawdown es una opinión. 0 = todavía no se ha cerrado
    /// ninguna operación.
    pub riesgo_por_operacion: AtomicF64,
    /// CL-106b (T1-02 de la revisión de la Línea G) — RIESGO POR OPERACIÓN
    /// EN USD: lo que se pierde si el stop de la orden se toca (nocional ×
    /// SL), media móvil con la misma memoria que `riesgo_por_operacion`.
    /// Desde CL-130 el freno duro ya no lo usa: cuenta la caída en stops de
    /// cada posición (`resultado_en_stops`). Queda como medida de la cuenta.
    /// 0 = todavía no se ha cerrado ninguna operación.
    pub riesgo_por_operacion_usd: AtomicF64,
    /// CL-99 (LC-F4-10) — PICO DEL CAPITAL desde el que miden la caída los
    /// dos cortacircuitos de drawdown (veto de entradas del riesgo y sistema
    /// inmune del host). Sólo sube (`risk_engine::drawdown::registrar_pico`)
    /// y el host lo guarda por entorno para que sobreviva al reinicio.
    pub pico_capital: AtomicF64,
    /// CL-106 (LC-F4-10) — GEOMETRÍA DE LAS ÓRDENES EN UNIDADES DEL STOP,
    /// media móvil con la misma memoria que `riesgo_por_operacion`: lo que
    /// gana una orden que toca el TP, `(TP − f)/SL`, y lo que pierde la que
    /// toca el SL, `(SL + f)/SL`, con `f` la fricción de ida y vuelta. Con
    /// ellas el freno duro de drawdown deriva su umbral en stops
    /// (`risk_engine::drawdown::freno_duro`). 0 = sin operaciones cerradas aún.
    pub ganancia_en_stops: AtomicF64,
    pub perdida_en_stops: AtomicF64,
    /// CL-130 (R1 de la revisión del PR #62) — CAÍDA DEL FRENO DURO EN STOPS
    /// DE CADA POSICIÓN: la suma de los resultados de los cierres, cada uno
    /// en stops de su propia posición (PnL neto / riesgo al stop de esa
    /// posición), y su máximo, que empieza en 0
    /// (`risk_engine::drawdown::registrar_resultado_en_stops`). La caída es
    /// `pico_en_stops − resultado_en_stops`.
    pub resultado_en_stops: AtomicF64,
    pub pico_en_stops: AtomicF64,
    /// F1-011: Anillo de los últimos 200 resultados en stops para ventana deslizante de caída (nivel alpha = 0.05).
    pub ventana_stops_buffer: std::sync::Mutex<([f64; 200], usize, usize)>,
    /// CL-132 (R3) — SALDO DEL EXCHANGE, que el host publica en cada ciclo de
    /// su sistema inmune. Los frenos de drawdown registran el pico y miden la
    /// caída con el mínimo entre éste y `unified_capital`
    /// (`risk_engine::drawdown::capital_del_freno`). 0 = sin saldo del
    /// exchange (backtest, o antes del primer ciclo del host).
    pub capital_exchange: AtomicF64,
    /// CL-133 (B2) — el pico guardado es ilegible: no entran posiciones
    /// nuevas hasta que el operador repare o retire el archivo.
    pub pico_ilegible: AtomicBool,
    pub tick_counter: AtomicU64,
    pub kill_switch_active: AtomicBool,
    pub last_ws_latency_ms: AtomicU64,
    /// I-2b (Línea I): componente red+reloj de la edad del último evento
    /// público (ms, con signo). Incluye el error de la proyección del reloj:
    /// un valor negativo delata que la proyección quedó por detrás de E.
    pub last_ws_red_reloj_ms: std::sync::atomic::AtomicI64,
    /// I-2b (Línea I): espera en la cola local del último evento público (ms).
    pub last_ws_espera_local_ms: AtomicU64,
    /// I-2b (Línea I): deriva D del reloj de pared frente a la proyección
    /// monótona del bucle de eventos (ms, con signo). Entra en
    /// `last_ws_red_reloj_ms` con el signo contrario hasta I-5.
    pub deriva_reloj_pared_ms: std::sync::atomic::AtomicI64,
    pub server_time_offset_ms: std::sync::atomic::AtomicI64,
    pub market_regime: std::sync::atomic::AtomicU8, // 0: Range, 1: BullRun, 2: Crash, 3: Chaotic (MAP discreto)
    /// Distribución espectral continua del régimen de mercado [p_range, p_bull, p_crash, p_chaos]
    pub regime_p_range: AtomicF64,
    pub regime_p_bull: AtomicF64,
    pub regime_p_crash: AtomicF64,
    pub regime_p_chaos: AtomicF64,
    pub panic_memory_dump: AtomicBool,              // Flag de pánico por memoria
    pub registry: Arc<OmniscientRegistry>,
    pub global_covariance_tensor: AtomicF64,
    pub global_momentum_vector: AtomicF64,
    /// Generación monotónica del genoma actualmente aplicado en la arena viva (0 = baseline)
    pub applied_generation: AtomicU64,
}

impl GlobalArena {
    /// D-714 (DÉCIMA OLA · auditoría integral) — CONSTRUCCIÓN CON PILA SUFICIENTE,
    /// EN UN SOLO SITIO.
    ///
    /// `CoinArena` lleva en línea el anillo de ticks de cada moneda y la
    /// construcción materializa temporales de ese tamaño: no cabe en la pila por
    /// defecto (1–2 MiB en un hilo de Windows, de un test o de un worker de
    /// rayon). D-684 lo resolvió a mano en los binarios, pero cada sitio nuevo
    /// que construye un arena vuelve a tropezar: dos tests desbordaban la pila y
    /// mataban su suite entera, y el daemon Darwin lo construye dentro de un
    /// `par_iter` —con `panic = "abort"`, eso es el proceso muerto con posiciones
    /// abiertas—. Aquí vive la única forma segura de construirlo.
    pub fn build_in_own_stack(initial_capital: f64) -> std::sync::Arc<Self> {
        std::thread::Builder::new()
            .name("arena-build".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || std::sync::Arc::new(Self::new(initial_capital)))
            .expect("no se pudo crear el hilo de construcción del arena")
            .join()
            .expect("la construcción del arena entró en pánico")
    }

    pub fn new(initial_capital: f64) -> Self {
        Self::build(initial_capital, QuantumConfig::new(initial_capital))
    }

    /// D-650: arena construida desde un genoma EXPLÍCITO. Permite al test T-2
    /// comparar el arranque en frío con el hot-swap sin depender del genoma
    /// que hubiera en disco.
    pub fn from_genome(initial_capital: f64, genome: &crate::genome::SuperGenotype) -> Self {
        Self::build(
            initial_capital,
            QuantumConfig::from_genome(initial_capital, genome),
        )
    }

    /// D-714: variante con pila suficiente de `from_genome`, para los llamadores
    /// que no corren en un hilo de 32 MiB (tests, workers de rayon).
    pub fn from_genome_in_own_stack(
        initial_capital: f64,
        genome: &crate::genome::SuperGenotype,
    ) -> std::sync::Arc<Self> {
        let config = QuantumConfig::from_genome(initial_capital, genome);
        std::thread::Builder::new()
            .name("arena-build".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || std::sync::Arc::new(Self::build(initial_capital, config)))
            .expect("no se pudo crear el hilo de construcción del arena")
            .join()
            .expect("la construcción del arena entró en pánico")
    }

    fn build(initial_capital: f64, config: QuantumConfig) -> Self {
        // D-680 (DÉCIMA OLA): el prior del win rate era `maker_obi_threshold`
        // (0,95 en el genoma de producción), un umbral de desequilibrio del
        // libro de órdenes, no una tasa de acierto: antes de su primera
        // operación el sistema se creía acertando el 95 %.
        let w_base = crate::genome::SuperGenotype::WORST_TOLERATED_WR;
        let mut coins_vec = Vec::with_capacity(MAX_COINS);
        for _ in 0..MAX_COINS {
            coins_vec.push(CoinArena::new(w_base));
        }
        let coins: Box<[CoinArena; MAX_COINS]> = coins_vec
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("Box conversion failed"));

        Self {
            config,
            coins,
            unified_capital: AtomicF64::new(initial_capital),
            used_margin: AtomicF64::new(0.0),
            riesgo_por_operacion: AtomicF64::new(0.0),
            riesgo_por_operacion_usd: AtomicF64::new(0.0),
            pico_capital: AtomicF64::new(initial_capital),
            ganancia_en_stops: AtomicF64::new(0.0),
            perdida_en_stops: AtomicF64::new(0.0),
            resultado_en_stops: AtomicF64::new(0.0),
            pico_en_stops: AtomicF64::new(0.0),
            ventana_stops_buffer: std::sync::Mutex::new(([0.0; 200], 0, 0)),
            // F2: capital_exchange nace en NAN (no observado). Permite distinguir
            // ausencia de saldo exchange (backtest/offline) de un cero real observado (0.0).
            capital_exchange: AtomicF64::new(f64::NAN),
            pico_ilegible: AtomicBool::new(false),
            tick_counter: AtomicU64::new(0),
            kill_switch_active: AtomicBool::new(false),
            last_ws_latency_ms: AtomicU64::new(0),
            last_ws_red_reloj_ms: std::sync::atomic::AtomicI64::new(0),
            last_ws_espera_local_ms: AtomicU64::new(0),
            deriva_reloj_pared_ms: std::sync::atomic::AtomicI64::new(0),
            server_time_offset_ms: std::sync::atomic::AtomicI64::new(0),
            market_regime: std::sync::atomic::AtomicU8::new(0),
            regime_p_range: AtomicF64::new(1.0),
            regime_p_bull: AtomicF64::new(0.0),
            regime_p_crash: AtomicF64::new(0.0),
            regime_p_chaos: AtomicF64::new(0.0),
            panic_memory_dump: AtomicBool::new(false),
            registry: Arc::new(OmniscientRegistry::new()),
            global_covariance_tensor: AtomicF64::new(1.0), // Base variance multiplier
            global_momentum_vector: AtomicF64::new(0.0),   // Neutral momentum
            applied_generation: AtomicU64::new(0),
        }
    }
}

impl GlobalArena {
    /// F1-011: Registra un resultado en stops en la ventana deslizante de 200 operaciones.
    #[inline]
    pub fn push_resultado_en_stops_ventana(&self, r: f64) {
        if !r.is_finite() {
            return;
        }
        if let Ok(mut g) = self.ventana_stops_buffer.lock() {
            let (ref mut buf, ref mut head, ref mut count) = *g;
            buf[*head] = r;
            *head = (*head + 1) % 200;
            if *count < 200 {
                *count += 1;
            }
        }
    }

    /// F1-011: Caída máxima en stops calculada sobre la ventana deslizante de los últimos <= 200 cierres.
    /// Si la ventana no tiene elementos, devuelve 0.0.
    #[inline]
    pub fn caida_en_stops_ventana(&self) -> f64 {
        if let Ok(g) = self.ventana_stops_buffer.lock() {
            let (ref buf, head, count) = *g;
            if count == 0 {
                return 0.0;
            }
            let start = if count < 200 { 0 } else { head };
            let (mut acum, mut pico, mut max_dd) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..count {
                let idx = (start + i) % 200;
                acum += buf[idx];
                pico = pico.max(acum);
                max_dd = max_dd.max(pico - acum);
            }
            max_dd
        } else {
            0.0
        }
    }

    /// F1-011: Reinicia la ventana deslizante de stops.
    #[inline]
    pub fn reset_ventana_stops(&self) {
        if let Ok(mut g) = self.ventana_stops_buffer.lock() {
            g.0.fill(0.0);
            g.1 = 0;
            g.2 = 0;
        }
    }

    /// CL-62 (LC-F1-03): multiplicador ATR del stop de la moneda del slot.
    /// `sl_atr_mult_btc` es el gen de BTCUSDT por su IDENTIDAD (ADR-0011: el
    /// slot es una posición, el símbolo la identidad); las demás monedas
    /// llevan `sl_atr_multiplier`. Antes el gate de riesgo y la sonda de
    /// banda del núcleo elegían con `coin_id == 0`: el forense y el
    /// entrenador, que ponen en el slot 0 el símbolo que miden, daban a una
    /// alt el stop de BTC, y en vivo BTC lo recibía sólo porque el bootloader
    /// lo lista primero.
    #[inline]
    pub fn sl_atr_multiplier_for(&self, coin_id: usize) -> f64 {
        if crate::symbol_registry::slot_es_simbolo(coin_id, "BTCUSDT") {
            self.config.sl_atr_mult_btc.load(Ordering::Relaxed)
        } else {
            self.config.sl_atr_multiplier.load(Ordering::Relaxed)
        }
    }

    /// ADR-0011 / LC-F1-03: Multiplicador de volatilidad por IDENTIDAD de símbolo,
    /// no por slot físico rígido. BTC y ETH leen sus genes dedicados por símbolo;
    /// las demás monedas miden su volatilidad relativa respecto al ancla de mercado (BTC).
    #[inline]
    pub fn volatility_multiplier_for(&self, coin_id: usize, atr_pct: f64) -> f64 {
        let is_empty_universe = crate::symbols::get_active_universe_size() == 0;
        let is_btc = crate::symbol_registry::slot_es_simbolo(coin_id, "BTCUSDT")
            || (is_empty_universe && coin_id == 0);
        let is_eth = crate::symbol_registry::slot_es_simbolo(coin_id, "ETHUSDT")
            || (is_empty_universe && coin_id == 1);

        if is_btc {
            self.config.btc_volatility_multiplier.load(Ordering::Relaxed)
        } else if is_eth {
            self.config.eth_volatility_multiplier.load(Ordering::Relaxed)
        } else {
            let eth_mult = self.config.eth_volatility_multiplier.load(Ordering::Relaxed);
            let btc_slot = crate::symbols::get_coin_id("BTCUSDT").unwrap_or(0);
            let btc_atr_pct = if btc_slot < self.coins.len() {
                let btc_price = self.coins[btc_slot].current_price.load(Ordering::Relaxed).max(1e-8);
                let btc_atr = self.coins[btc_slot].current_atr.load(Ordering::Relaxed);
                if btc_atr > 0.0 && btc_price > 1e-6 {
                    (btc_atr / btc_price).clamp(0.0005, 0.50)
                } else {
                    0.005
                }
            } else {
                0.005
            };
            let effective_atr_pct = atr_pct.clamp(0.0001, 1.0);
            let relative_vol = (effective_atr_pct / btc_atr_pct.max(0.0005)).clamp(0.5, 3.0);
            eth_mult * relative_vol
        }
    }

    #[inline(always)]
    pub fn increment_tick(&self) -> u64 {
        self.tick_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// CL-44b (ADR-0015) — el aplanado total del sistema inmune cierra en el
    /// exchange TODAS las posiciones sin tocar sus ranuras locales, que siguen
    /// abiertas hasta la reconciliación. Con el kill-switch el núcleo sigue
    /// gestionándolas (CL-43): sin esto, un SL o TP local posterior se
    /// contabilizaba como cierre real, a un precio que no es el del aplanado,
    /// y alimentaba capital, Kelly, calibradores y el sobre de Kelly en disco.
    ///
    /// Se consume la confirmación ANTES de aplanar: lo que el núcleo cierre
    /// después es papel (no aprende) y el host sigue enviando su reduce-only
    /// de respaldo, que cierra lo que el aplanado no pudo. Devuelve cuántas
    /// ranuras abiertas tenían confirmación.
    pub fn consume_exchange_confirmations(&self) -> usize {
        self.coins
            .iter()
            .flat_map(|c| c.positions.slots())
            .filter(|p| p.consume_exchange_confirmation())
            .count()
    }

    /// MOD6/8-010 (INFORME DECIMOCUARTO): lector saturado de `used_margin`.
    /// Con `fetch_sub` atómico, un drift contable (p. ej. doble liberación
    /// de margen tras reconciliación) puede dejar el átomo levemente
    /// NEGATIVO. Un lector que viera el negativo inflamaría `free_margin`
    /// y autorizaría sobre-exposición — todos los lectores de margen
    /// deben saturar a cero por el camino de la orden.
    #[inline(always)]
    pub fn used_margin_saturated(&self) -> f64 {
        self.used_margin.load(Ordering::Relaxed).max(0.0)
    }

    #[inline(always)]
    pub fn update_market_data(
        &self,
        coin_id: usize,
        bid_price: f64,
        ask_price: f64,
        bid_qty: f64,
        ask_qty: f64,
        timestamp: u64,
    ) {
        if coin_id < crate::symbols::get_active_universe_size() {
            let mid_price = (bid_price + ask_price) / 2.0;
            self.coins[coin_id]
                .current_price
                .store(mid_price, Ordering::Relaxed);
            self.coins[coin_id].push_tick(CompactTick {
                timestamp,
                bid_price,
                ask_price,
                bid_qty,
                ask_qty,
            });
        }
    }

    #[inline(always)]
    pub fn update_spot_data(
        &self,
        coin_id: usize,
        bid_price: f64,
        ask_price: f64,
        bid_qty: f64,
        ask_qty: f64,
    ) {
        if coin_id < self.coins.len() {
            self.coins[coin_id]
                .spot_bid
                .store(bid_price, Ordering::Relaxed);
            self.coins[coin_id]
                .spot_ask
                .store(ask_price, Ordering::Relaxed);
            self.coins[coin_id]
                .spot_bid_qty
                .store(bid_qty, Ordering::Relaxed);
            self.coins[coin_id]
                .spot_ask_qty
                .store(ask_qty, Ordering::Relaxed);
            // CL-79 (F3.7-03): la edad del spot se mide con el reloj de
            // eventos del núcleo; ver `CoinArena::spot_ts_ms`.
            self.coins[coin_id]
                .spot_ts_ms
                .store(self.registry.get_event_time().max(0) as u64, Ordering::Relaxed);
        }
    }

    /// #660 (F2-B5) — el decaimiento del CVD/OBI agregado vive en TIEMPO
    /// FÍSICO: factor = exp(−dt/τ) con τ = 60 s (TAU_FLUJO_AGG_S), dt desde
    /// el último trade con reloj monotónico. Antes: 0.995 POR EVENTO — la
    /// memoria efectiva variaba ×100 entre feeds de 1 y 100 ev/s y rompía
    /// la comparabilidad entre monedas (doctrina ADR-0014: tiempo continuo).
    #[inline(always)]
    pub fn update_agg_trade(&self, coin_id: usize, is_buyer_maker: bool, qty: f64, ts_ms: u64) {
        if coin_id < MAX_COINS {
            const TAU_FLUJO_AGG_S: f64 = 60.0;
            let coin = &self.coins[coin_id];
            let prev_ts = coin.last_agg_trade_ms.load(Ordering::Relaxed);
            let decay = if prev_ts > 0 && ts_ms > prev_ts {
                let dt_s = ((ts_ms - prev_ts) as f64 / 1000.0).min(3600.0);
                (-(dt_s / TAU_FLUJO_AGG_S)).exp()
            } else {
                1.0 // primer trade o ticks intra-ms: sin doble decaimiento
            };
            coin.last_agg_trade_ms.store(ts_ms.max(prev_ts), Ordering::Relaxed);
            let current_buy = coin.agg_buy_vol.load(Ordering::Relaxed) * decay;
            let current_sell = coin.agg_sell_vol.load(Ordering::Relaxed) * decay;

            let (new_buy, new_sell) = if is_buyer_maker {
                // El comprador es maker = agresivo VENTA (Sell)
                (current_buy, current_sell + qty)
            } else {
                // El comprador es taker = agresivo COMPRA (Buy)
                (current_buy + qty, current_sell)
            };

            self.coins[coin_id]
                .agg_buy_vol
                .store(new_buy, Ordering::Relaxed);
            self.coins[coin_id]
                .agg_sell_vol
                .store(new_sell, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn update_l2_depth(&self, coin_id: usize, bid_wall: f64, ask_wall: f64) {
        if coin_id < MAX_COINS {
            self.coins[coin_id]
                .l2_bid_wall
                .store(bid_wall, Ordering::Relaxed);
            self.coins[coin_id]
                .l2_ask_wall
                .store(ask_wall, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod qo_660_tests {
    use super::*;

    /// #660 (F2-B5) — la memoria del flujo agregado vive en TIEMPO FÍSICO
    /// (τ=60 s), no en eventos: 60 s de silencio decaen el volumen a
    /// exactamente e^{-1}; el mismo milisegundo no decae dos veces; un
    /// ts retrógrado no altera el reloj.
    #[test]
    fn qo_660_decay_fisico_agg() {
        let arena = GlobalArena::build_in_own_stack(13.0);
        // 100 unidades compradoras en t=0.
        arena.update_agg_trade(0, false, 100.0, 1_000);
        let v0 = arena.coins[0].agg_buy_vol.load(Ordering::Relaxed);
        assert!((v0 - 100.0).abs() < 1e-9);

        // 60 s después otra compra de 0: el fondo decae a e^{-1}.
        arena.update_agg_trade(0, false, 0.0, 61_000);
        let v1 = arena.coins[0].agg_buy_vol.load(Ordering::Relaxed);
        let esperado = 100.0 * (-1.0f64).exp();
        assert!(
            (v1 - esperado).abs() < 1e-6,
            "60 s ⇒ e^-1 del volumen: {v1} vs {esperado}"
        );

        // Intra-ms: sin doble decaimiento.
        arena.update_agg_trade(0, false, 5.0, 61_000);
        let v2 = arena.coins[0].agg_buy_vol.load(Ordering::Relaxed);
        assert!((v2 - (esperado + 5.0)).abs() < 1e-6, "mismo ms no decae: {v2}");

        // Retrógrado: el reloj guarda el máximo; el volumen no cambia por
        // un ts menor (factor 1.0).
        arena.update_agg_trade(0, false, 0.0, 50_000);
        assert_eq!(
            arena.coins[0].last_agg_trade_ms.load(Ordering::Relaxed),
            61_000,
            "reloj monotónico"
        );
        let v3 = arena.coins[0].agg_buy_vol.load(Ordering::Relaxed);
        assert!((v3 - v2).abs() < 1e-9, "retrógrado no decae: {v3} vs {v2}");
    }
}
