use crate::atomic_float::AtomicF64;
use crate::config::QuantumConfig;
use omniscient_registry::OmniscientRegistry;
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

#[repr(C, align(64))]
pub struct SwingState {
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
}

impl SwingState {
    pub fn new(w_base: f64) -> Self {
        Self {
            pnl_realized: AtomicF64::new(0.0),
            pnl_unrealized: AtomicF64::new(0.0),
            pnl_gross: AtomicF64::new(0.0),
            gross_wins: AtomicF64::new(0.0),
            gross_losses: AtomicF64::new(0.0),
            active_positions: AtomicUsize::new(0),
            win_rate: AtomicF64::new(w_base),
            profit_factor: AtomicF64::new(1.50), // Prior Bayesiano Swing
            kelly_fraction: AtomicF64::new(0.25), // Prior Bayesiano (Quarter-Kelly)
            roi_pre_fee: AtomicF64::new(0.0),
            roi_post_fee: AtomicF64::new(0.0),
            trade_count: AtomicUsize::new(0),
        }
    }
}

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
    pub metrics: ScalpState,
    pub scalp: ScalpState,
    pub swing: SwingState,
    pub positions: crate::position::PositionManager,
    pub current_price: AtomicF64,
    pub ml_prob: AtomicF64,
    pub current_atr: AtomicF64,
    pub hurst_exponent: AtomicF64,
    /// Epigenetic Multipliers (Memory of past success/failure)
    pub epigenetic_bias: AtomicF64,
    pub epigenetic_threshold_modifier: AtomicF64,
    /// Tracking del mercado Spot (Leading Indicator para microestructura)
    pub spot_bid: AtomicF64,
    pub spot_ask: AtomicF64,
    pub spot_bid_qty: AtomicF64,
    pub spot_ask_qty: AtomicF64,
    /// Tracking de CVD (Cumulative Volume Delta) - Flujo de Capital
    pub agg_buy_vol: AtomicF64,
    pub agg_sell_vol: AtomicF64,
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
}

impl CoinArena {
    /// O(1) lock-free tick push. ~5 nanoseconds.
    #[inline(always)]
    pub fn push_tick(&self, tick: CompactTick) {
        self.tick_ring.push(tick);
        self.tick_head.fetch_add(1, Ordering::Relaxed);
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
    /// - `epigenetic_threshold_modifier` (0.8x → 1.5x): Multiplicador del min_confidence_cutoff.
    ///   Loss → sube (requiere MÁS confianza para entrar), Win rápido → baja levemente.
    ///
    /// **CUÁNDO:** Se llama al cerrar cada posición (scalp o swing) en process_tick y process_tick_shadow.
    /// **DÓNDE:** CoinArena (crates/quantum-arena/src/state.rs).
    /// **QUIÉN:** GodEngineCore invoca esto tras cada cierre de posición.
    #[inline(always)]
    pub fn apply_epigenetic_feedback(&self, pnl_pct: f64, trade_duration_ms: u64) {
        // Bias shift: PnL drives confidence multiplier.
        // A +2% trade shifts bias by +0.10, a -2% trade shifts by -0.10. Clamped for stability.
        let pnl_shift = (pnl_pct * 5.0).clamp(-0.1, 0.1);

        let old_bias = self.epigenetic_bias.load(Ordering::Relaxed);
        let new_bias = (old_bias + pnl_shift).clamp(0.5, 2.0);
        self.epigenetic_bias.store(new_bias, Ordering::Relaxed);

        // Threshold modifier: Controls how strict entry requirements become.
        // threshold_modifier multiplies min_confidence_cutoff:
        //   > 1.0 = needs MORE confidence to enter (defensive after losses)
        //   < 1.0 = allows lower confidence entries (aggressive after wins)
        //
        // Duration-aware: Fast wins get a stronger relaxation (the strategy is working well
        // in current conditions). Slow wins get less relaxation.
        let duration_factor = 1.0 - (trade_duration_ms as f64 / 3_600_000.0).clamp(0.0, 1.0);

        let threshold_shift = if pnl_pct > 0.0 {
            // Win: Relax threshold. Fast wins relax more (-0.02), slow wins relax less (-0.005).
            -0.005 - 0.015 * duration_factor
        } else {
            // Loss: Tighten threshold. Require MORE confidence next time.
            0.05
        };

        let old_threshold = self.epigenetic_threshold_modifier.load(Ordering::Relaxed);
        let new_threshold = (old_threshold + threshold_shift).clamp(0.8, 1.5);
        self.epigenetic_threshold_modifier
            .store(new_threshold, Ordering::Relaxed);
    }
}

impl CoinArena {
    pub fn new(w_base: f64) -> Self {
        Self {
            metrics: ScalpState::new(w_base),
            scalp: ScalpState::new(w_base),
            swing: SwingState::new(w_base),
            positions: crate::position::PositionManager::default(),
            current_price: AtomicF64::new(0.0),
            ml_prob: AtomicF64::new(w_base),
            current_atr: AtomicF64::new(0.0),
            hurst_exponent: AtomicF64::new(0.5),
            epigenetic_bias: AtomicF64::new(1.0),
            epigenetic_threshold_modifier: AtomicF64::new(1.0),
            spot_bid: AtomicF64::new(0.0),
            spot_ask: AtomicF64::new(0.0),
            spot_bid_qty: AtomicF64::new(0.0),
            spot_ask_qty: AtomicF64::new(0.0),
            agg_buy_vol: AtomicF64::new(0.0),
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
        }
    }
}

/// GlobalArena: El hipergrafo en memoria que todos los hilos leen y escriben.
/// Contiene configuración atómica y estado aislado por horizonte de tiempo.
/// 100% lock-free. Zero Mutex, Zero RwLock.
#[repr(C, align(64))]
pub struct GlobalArena {
    pub config: QuantumConfig,
    pub coins: Box<[CoinArena; MAX_COINS]>,
    pub tensor_arena: Option<Box<CoinTensorArena>>, // FASE 8: SIMD Layer

    // Portfolio & Risk
    pub unified_capital: AtomicF64,
    pub used_margin: AtomicF64,
    pub scalp_used_margin: AtomicF64,
    pub swing_used_margin: AtomicF64,
    pub tick_counter: AtomicU64,
    pub kill_switch_active: AtomicBool,
    pub last_ws_latency_ms: AtomicU64,
    pub server_time_offset_ms: std::sync::atomic::AtomicI64,
    pub market_regime: std::sync::atomic::AtomicU8, // 0: Range, 1: BullRun, 2: Crash, 3: Chaotic
    pub panic_memory_dump: AtomicBool,              // Flag de pánico por memoria
    pub registry: Arc<OmniscientRegistry>,
    pub global_covariance_tensor: AtomicF64,
    pub global_momentum_vector: AtomicF64,
    /// Generación monotónica del genoma actualmente aplicado en la arena viva (0 = baseline)
    pub applied_generation: AtomicU64,
}

impl GlobalArena {
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
            scalp_used_margin: AtomicF64::new(0.0),
            swing_used_margin: AtomicF64::new(0.0),
            tick_counter: AtomicU64::new(0),
            kill_switch_active: AtomicBool::new(false),
            last_ws_latency_ms: AtomicU64::new(0),
            server_time_offset_ms: std::sync::atomic::AtomicI64::new(0),
            market_regime: std::sync::atomic::AtomicU8::new(0),
            panic_memory_dump: AtomicBool::new(false),
            registry: Arc::new(OmniscientRegistry::new()),
            global_covariance_tensor: AtomicF64::new(1.0), // Base variance multiplier
            global_momentum_vector: AtomicF64::new(0.0),   // Neutral momentum
            applied_generation: AtomicU64::new(0),
            tensor_arena: Some(Box::new(CoinTensorArena::new())), // FASE 8 Tensor Allocation
        }
    }
}

impl GlobalArena {
    #[inline(always)]
    pub fn increment_tick(&self) -> u64 {
        self.tick_counter.fetch_add(1, Ordering::Relaxed)
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

            // FASE 8: Propagación paralela al Tensor
            if let Some(tensor) = &self.tensor_arena {
                tensor.current_price[coin_id].store(mid_price.to_bits(), Ordering::Relaxed);
            }
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
        if coin_id < crate::symbols::get_active_universe_size() {
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

            // FASE 8: Propagación paralela al Tensor
            if let Some(tensor) = &self.tensor_arena {
                tensor.spot_bid[coin_id].store(bid_price.to_bits(), Ordering::Relaxed);
                tensor.spot_ask[coin_id].store(ask_price.to_bits(), Ordering::Relaxed);
            }
        }
    }

    #[inline(always)]
    pub fn update_agg_trade(&self, coin_id: usize, is_buyer_maker: bool, qty: f64) {
        if coin_id < MAX_COINS {
            let decay = 0.995; // EWMA decay for real-time microstructural order flow (Axioma II)
            let current_buy = self.coins[coin_id].agg_buy_vol.load(Ordering::Relaxed) * decay;
            let current_sell = self.coins[coin_id].agg_sell_vol.load(Ordering::Relaxed) * decay;

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

            // FASE 8: Propagación paralela al Tensor
            if let Some(tensor) = &self.tensor_arena {
                tensor.agg_buy_vol[coin_id].store(new_buy.to_bits(), Ordering::Relaxed);
                tensor.agg_sell_vol[coin_id].store(new_sell.to_bits(), Ordering::Relaxed);
            }
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

            // FASE 8: Propagación paralela al Tensor
            if let Some(tensor) = &self.tensor_arena {
                tensor.l2_bid_wall[coin_id].store(bid_wall.to_bits(), Ordering::Relaxed);
                tensor.l2_ask_wall[coin_id].store(ask_wall.to_bits(), Ordering::Relaxed);
            }
        }
    }
}

/// FASE 8: REPRESENTACIÓN TENSORIAL (Lock-Free SIMD-Ready)
/// Esta estructura mapea el estado de todos los activos en memoria contigua (Structure of Arrays),
/// alineada a 64-bytes (L1 Cache Line) para evitar False Sharing y permitir cargas matriciales/SIMD paralelas.
#[repr(C, align(64))]
pub struct CoinTensorArena {
    pub current_price: [AtomicU64; 32],
    pub ml_prob: [AtomicU64; 32],
    pub current_atr: [AtomicU64; 32],
    pub hurst_exponent: [AtomicU64; 32],
    pub epigenetic_bias: [AtomicU64; 32],
    pub epigenetic_threshold_modifier: [AtomicU64; 32],
    pub spot_bid: [AtomicU64; 32],
    pub spot_ask: [AtomicU64; 32],
    pub agg_buy_vol: [AtomicU64; 32],
    pub agg_sell_vol: [AtomicU64; 32],
    pub l2_bid_wall: [AtomicU64; 32],
    pub l2_ask_wall: [AtomicU64; 32],
}

impl Default for CoinTensorArena {
    fn default() -> Self {
        Self::new()
    }
}

impl CoinTensorArena {
    pub fn new() -> Self {
        let init_f64 = |val: f64| -> [AtomicU64; 32] {
            let mut arr: [AtomicU64; 32] = Default::default();
            for item in &mut arr {
                *item = AtomicU64::new(val.to_bits());
            }
            arr
        };

        Self {
            current_price: init_f64(0.0),
            ml_prob: init_f64(0.5),
            current_atr: init_f64(0.0),
            hurst_exponent: init_f64(0.5),
            epigenetic_bias: init_f64(1.0),
            epigenetic_threshold_modifier: init_f64(1.0),
            spot_bid: init_f64(0.0),
            spot_ask: init_f64(0.0),
            agg_buy_vol: init_f64(0.0),
            agg_sell_vol: init_f64(0.0),
            l2_bid_wall: init_f64(0.0),
            l2_ask_wall: init_f64(0.0),
        }
    }
}
