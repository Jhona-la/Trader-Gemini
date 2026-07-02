use crate::atomic_float::AtomicF64;
use crate::config::QuantumConfig;
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::cell::UnsafeCell;
use omniscient_registry::OmniscientRegistry;

/// Axioma V: Cohesión Celular Absoluta.
/// Los motores Scalp y Swing no se pisan porque operan en structs aislados, 
/// pero unidos dentro del mismo bloque contiguo de RAM (GlobalArena).

#[repr(C, align(64))]
pub struct ScalpState {
    pub pnl_realized: AtomicF64,
    pub pnl_unrealized: AtomicF64,
    pub active_positions: AtomicUsize,
    pub win_rate: AtomicF64,
    pub profit_factor: AtomicF64,
    pub kelly_fraction: AtomicF64,
    pub trade_count: AtomicUsize,
    pub zombie_promotions: AtomicUsize,
}

impl Default for ScalpState {
    fn default() -> Self {
        Self {
            pnl_realized: AtomicF64::new(0.0),
            pnl_unrealized: AtomicF64::new(0.0),
            active_positions: AtomicUsize::new(0),
            win_rate: AtomicF64::new(0.55),     // Asumimos 55% inicial para que Kelly sea > 0
            profit_factor: AtomicF64::new(1.5), // Profit factor rentable
            kelly_fraction: AtomicF64::new(0.0),
            trade_count: AtomicUsize::new(0),
            zombie_promotions: AtomicUsize::new(0),
        }
    }
}

#[repr(C, align(64))]
pub struct SwingState {
    pub pnl_realized: AtomicF64,
    pub pnl_unrealized: AtomicF64,
    pub active_positions: AtomicUsize,
    pub win_rate: AtomicF64,
    pub profit_factor: AtomicF64,
    pub kelly_fraction: AtomicF64,
    pub trade_count: AtomicUsize,
}

impl Default for SwingState {
    fn default() -> Self {
        Self {
            pnl_realized: AtomicF64::new(0.0),
            pnl_unrealized: AtomicF64::new(0.0),
            active_positions: AtomicUsize::new(0),
            win_rate: AtomicF64::new(0.55),
            profit_factor: AtomicF64::new(1.5),
            kelly_fraction: AtomicF64::new(0.0),
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
        self.head.store(current_head.wrapping_add(1), Ordering::Release);
        // Track fill level (cap at TICK_RING_SIZE)
        let current_len = self.len.load(Ordering::Relaxed);
        if current_len < TICK_RING_SIZE {
            self.len.store(current_len + 1, Ordering::Relaxed);
        }
    }

    /// Returns current length of valid data
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
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
    pub scalp: ScalpState,
    pub swing: SwingState,
    pub positions: crate::position::PositionManager,
    pub current_price: AtomicF64,
    pub ml_prob: AtomicF64,
    pub hurst_exponent: AtomicF64,
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
}

impl CoinArena {
    /// O(1) lock-free tick push. ~5 nanoseconds.
    #[inline(always)]
    pub fn push_tick(&self, tick: CompactTick) {
        self.tick_ring.push(tick);
        self.tick_head.fetch_add(1, Ordering::Relaxed);
    }
}

impl Default for CoinArena {
    fn default() -> Self {
        Self {
            scalp: ScalpState::default(),
            swing: SwingState::default(),
            positions: crate::position::PositionManager::default(),
            current_price: AtomicF64::new(0.0),
            ml_prob: AtomicF64::new(0.5),
            hurst_exponent: AtomicF64::new(0.5),
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
        }
    }
}

/// GlobalArena: El hipergrafo en memoria que todos los hilos leen y escriben.
/// Contiene configuración atómica y estado aislado por horizonte de tiempo.
/// 100% lock-free. Zero Mutex, Zero RwLock.
#[repr(C, align(64))]
pub struct GlobalArena {
    pub config: QuantumConfig,
    pub coins: Box<[CoinArena; 30]>,
    pub unified_capital: AtomicF64,
    pub used_margin: AtomicF64,
    pub tick_counter: AtomicU64,
    pub kill_switch_active: AtomicBool,
    pub last_ws_latency_ms: AtomicU64,
    pub market_regime: std::sync::atomic::AtomicU8, // 0: Range, 1: BullRun, 2: Crash, 3: Chaotic
    pub registry: Arc<OmniscientRegistry>,
}

impl GlobalArena {
    pub fn new(initial_capital: f64) -> Self {
        let mut coins_vec = Vec::with_capacity(30);
        for _ in 0..30 {
            coins_vec.push(CoinArena::default());
        }
        let coins: Box<[CoinArena; 30]> = coins_vec.into_boxed_slice().try_into().unwrap_or_else(|_| panic!("Box conversion failed"));

        Self {
            config: QuantumConfig::default(),
            coins,
            unified_capital: AtomicF64::new(initial_capital),
            used_margin: AtomicF64::new(0.0),
            tick_counter: AtomicU64::new(0),
            kill_switch_active: AtomicBool::new(false),
            last_ws_latency_ms: AtomicU64::new(0),
            market_regime: std::sync::atomic::AtomicU8::new(0),
            registry: Arc::new(OmniscientRegistry::new()),
        }
    }
}

impl Default for GlobalArena {
    fn default() -> Self {
        let config = QuantumConfig::default();
        let initial_capital = config.base_capital.load(Ordering::Relaxed);
        
        let mut coins_vec = Vec::with_capacity(30);
        for _ in 0..30 {
            coins_vec.push(CoinArena::default());
        }
        let coins: Box<[CoinArena; 30]> = coins_vec.into_boxed_slice().try_into().unwrap_or_else(|_| panic!("Box conversion failed"));

        Self {
            config,
            coins,
            unified_capital: AtomicF64::new(initial_capital),
            used_margin: AtomicF64::new(0.0),
            tick_counter: AtomicU64::new(0),
            kill_switch_active: AtomicBool::new(false),
            last_ws_latency_ms: AtomicU64::new(0),
            market_regime: std::sync::atomic::AtomicU8::new(0),
            registry: Arc::new(OmniscientRegistry::new()),
        }
    }
}

impl GlobalArena {

    #[inline(always)]
    pub fn increment_tick(&self) -> u64 {
        self.tick_counter.fetch_add(1, Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn update_market_data(&self, coin_id: usize, bid_price: f64, ask_price: f64, bid_qty: f64, ask_qty: f64) {
        if coin_id < 30 {
            let mid_price = (bid_price + ask_price) / 2.0;
            self.coins[coin_id].current_price.store(mid_price, Ordering::Relaxed);
            self.coins[coin_id].push_tick(CompactTick {
                bid_price,
                ask_price,
                bid_qty,
                ask_qty,
            });
        }
    }

    #[inline(always)]
    pub fn update_spot_data(&self, coin_id: usize, bid_price: f64, ask_price: f64, bid_qty: f64, ask_qty: f64) {
        if coin_id < 30 {
            self.coins[coin_id].spot_bid.store(bid_price, Ordering::Relaxed);
            self.coins[coin_id].spot_ask.store(ask_price, Ordering::Relaxed);
            self.coins[coin_id].spot_bid_qty.store(bid_qty, Ordering::Relaxed);
            self.coins[coin_id].spot_ask_qty.store(ask_qty, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn update_agg_trade(&self, coin_id: usize, is_buyer_maker: bool, qty: f64) {
        if coin_id < 30 {
            if is_buyer_maker {
                // El comprador es maker = agresivo VENTA (Sell)
                let current = self.coins[coin_id].agg_sell_vol.load(Ordering::Relaxed);
                self.coins[coin_id].agg_sell_vol.store(current + qty, Ordering::Relaxed);
            } else {
                // El comprador es taker = agresivo COMPRA (Buy)
                let current = self.coins[coin_id].agg_buy_vol.load(Ordering::Relaxed);
                self.coins[coin_id].agg_buy_vol.store(current + qty, Ordering::Relaxed);
            }
        }
    }

    #[inline(always)]
    pub fn update_l2_depth(&self, coin_id: usize, bid_wall: f64, ask_wall: f64) {
        if coin_id < 30 {
            self.coins[coin_id].l2_bid_wall.store(bid_wall, Ordering::Relaxed);
            self.coins[coin_id].l2_ask_wall.store(ask_wall, Ordering::Relaxed);
        }
    }
}
