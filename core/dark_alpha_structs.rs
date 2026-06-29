// 🌑 DARK ALPHA LAYER - HIGH-PERFORMANCE RUST STRUCTURES
// =========================================================
// These structures are designed for zero-copy memory mapping and
// L1 CPU Cache optimization to process DEX, MEV and Mempool events
// at nanosecond speeds.

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct JupiterSwapEvent {
    pub timestamp_ms: u64,
    pub out_amount_usd: f64,
    pub is_split_route: bool,
    pub is_stable_to_crypto: bool,
    pub slippage_bps: u16,
    // Padding for cache alignment
    pub _pad: [u8; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HyperliquidLiquidation {
    pub timestamp_ms: u64,
    pub side: i8, // 1 for LONG (Sell pressure), -1 for SHORT (Buy pressure)
    pub size_usd: f64,
    // 8 + 1 + 8 = 17 bytes -> pad to 24
    pub _pad: [u8; 7],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MEVBundlePattern {
    pub timestamp_ms: u64,
    pub is_bullish_sandwich: bool,
    pub is_bearish_sandwich: bool,
    pub volume_usd: f64,
    pub execution_latency_ms: u32,
    // 8 + 1 + 1 + 8 + 4 = 22 -> pad to 24
    pub _pad: [u8; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RBFUrgencyEvent {
    pub timestamp_ms: u64,
    pub old_gas_price_gwei: f32,
    pub new_gas_price_gwei: f32,
    pub estimated_fee_delta_usd: f64,
    pub is_dex_router: bool,
    pub is_exchange_hot_wallet: bool,
    // 8 + 4 + 4 + 8 + 1 + 1 = 26 -> pad to 32
    pub _pad: [u8; 6],
}
