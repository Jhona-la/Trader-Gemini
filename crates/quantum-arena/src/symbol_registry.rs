use arc_swap::ArcSwap;
use lazy_static::lazy_static;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SymbolSpec {
    pub symbol: String,
    pub step_size: f64,
    pub tick_size: f64,
    pub min_qty: f64,
    pub min_notional: f64,
    pub max_leverage: u32,
    pub maker_fee: f64,
    pub taker_fee: f64,
    pub is_shadow: bool,
}

impl SymbolSpec {
    #[inline]
    pub fn validate_order(&self, raw_qty: f64, price: f64) -> Result<f64, &'static str> {
        if price <= 0.0 || raw_qty <= 0.0 {
            return Err("Invalid price or quantity");
        }
        let step = if self.step_size > 0.0 {
            self.step_size
        } else {
            1.0
        };
        let qty_steps = (raw_qty / step).floor();
        let mut adjusted_qty = qty_steps * step;

        // Exact decimal rounding based on step_size to eliminate floating point residues (BUG-673)
        if step < 1.0 && step > 0.0 {
            let decimals = (-step.log10()).round() as i32;
            if decimals > 0 && decimals <= 8 {
                let factor = 10_f64.powi(decimals);
                adjusted_qty = (adjusted_qty * factor).round() / factor;
            }
        }

        if adjusted_qty < self.min_qty {
            return Err("Qty below minQty");
        }
        let final_notional = adjusted_qty * price;
        if final_notional < self.min_notional {
            return Err("Notional below minNotional");
        }
        Ok(adjusted_qty)
    }

    #[inline]
    pub fn roundtrip_fee_cost(&self, notional: f64) -> f64 {
        notional * (self.maker_fee + self.taker_fee)
    }
}

lazy_static! {
    static ref DYNAMIC_REGISTRY: ArcSwap<Vec<SymbolSpec>> =
        ArcSwap::from_pointee(get_default_specs());
}

pub fn update_registry(new_specs: Vec<SymbolSpec>) {
    let current = DYNAMIC_REGISTRY.load();
    if current.is_empty() {
        DYNAMIC_REGISTRY.store(Arc::new(new_specs));
        return;
    }

    // FIX #905: Preservar estabilidad de coin_id para que símbolos existentes mantengan su índice
    let mut updated = (**current).clone();
    for spec in new_specs {
        if let Some(pos) = updated
            .iter()
            .position(|s| s.symbol.eq_ignore_ascii_case(&spec.symbol))
        {
            updated[pos] = spec;
        } else {
            updated.push(spec);
        }
    }
    DYNAMIC_REGISTRY.store(Arc::new(updated));
}

#[inline(always)]
pub fn try_spec(coin_id: usize) -> Option<SymbolSpec> {
    // We clone the struct to avoid lifetime issues since it's accessed heavily.
    // The struct is small enough that cloning is practically free.
    let registry = DYNAMIC_REGISTRY.load();
    registry.get(coin_id).cloned()
}

#[inline(always)]
pub fn try_symbol(coin_id: usize) -> Option<String> {
    let registry = DYNAMIC_REGISTRY.load();
    registry.get(coin_id).map(|s| s.symbol.clone())
}

#[inline(always)]
pub fn try_index(symbol: &str) -> Option<usize> {
    let registry = DYNAMIC_REGISTRY.load();
    let sym_upper = symbol.to_uppercase();
    registry
        .iter()
        .position(|s| s.symbol.to_uppercase() == sym_upper)
}

#[inline(always)]
pub fn spec(coin_id: usize) -> SymbolSpec {
    match try_spec(coin_id) {
        Some(s) => s,
        None => panic!("⚠️ [CRITICAL] Intento de acceder a SymbolSpec no inicializado o fuera de rango (ID: {}).", coin_id),
    }
}

pub fn get_official_binance_spec(symbol: &str) -> SymbolSpec {
    let sym_upper = symbol.to_uppercase();
    let (step_size, tick_size, min_qty) = match sym_upper.as_str() {
        "BTCUSDT" => (0.001, 0.1, 0.001),
        "ETHUSDT" => (0.01, 0.01, 0.01),
        "SOLUSDT" => (0.1, 0.01, 0.1),
        "BNBUSDT" => (0.01, 0.01, 0.01),
        "DOGEUSDT" => (1.0, 0.00001, 1.0),
        "XRPUSDT" => (0.1, 0.0001, 0.1),
        "ADAUSDT" => (1.0, 0.0001, 1.0),
        "AVAXUSDT" => (0.01, 0.001, 0.01),
        "LINKUSDT" => (0.01, 0.001, 0.01),
        "SUIUSDT" => (0.1, 0.0001, 0.1),
        "NEARUSDT" => (0.1, 0.001, 0.1),
        "APTUSDT" => (0.1, 0.001, 0.1),
        "ARBUSDT" => (0.1, 0.0001, 0.1),
        "OPUSDT" => (0.1, 0.0001, 0.1),
        "MATICUSDT" | "POLUSDT" => (1.0, 0.0001, 1.0),
        "DOTUSDT" => (0.1, 0.001, 0.1),
        _ => (0.01, 0.001, 0.01),
    };
    SymbolSpec {
        symbol: sym_upper,
        step_size,
        tick_size,
        min_qty,
        min_notional: 5.0, // Binance USDⓈ-M Futures Official Min Notional
        max_leverage: 20,
        maker_fee: 0.0002, // 0.02%
        taker_fee: 0.0005, // 0.05%
        is_shadow: false,
    }
}

fn get_default_specs() -> Vec<SymbolSpec> {
    Vec::new()
}
