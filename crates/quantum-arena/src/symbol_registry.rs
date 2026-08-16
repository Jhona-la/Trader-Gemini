use std::sync::Arc;
use arc_swap::ArcSwap;
use lazy_static::lazy_static;

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
        let notional = raw_qty * price;
        if notional < self.min_notional {
            return Err("Notional below minNotional");
        }
        if raw_qty < self.min_qty {
            return Err("Qty below minQty");
        }
        let qty_steps = (raw_qty / self.step_size).floor();
        let adjusted_qty = qty_steps * self.step_size;
        Ok(adjusted_qty)
    }

    #[inline]
    pub fn roundtrip_fee_cost(&self, notional: f64) -> f64 {
        notional * (self.maker_fee + self.taker_fee)
    }
}

lazy_static! {
    static ref DYNAMIC_REGISTRY: ArcSwap<Vec<SymbolSpec>> = ArcSwap::from_pointee(get_default_specs());
}

pub fn update_registry(new_specs: Vec<SymbolSpec>) {
    DYNAMIC_REGISTRY.store(Arc::new(new_specs));
}

#[inline(always)]
pub fn try_spec(coin_id: usize) -> Option<SymbolSpec> {
    // We clone the struct to avoid lifetime issues since it's accessed heavily.
    // The struct is small enough that cloning is practically free.
    let registry = DYNAMIC_REGISTRY.load();
    registry.get(coin_id).cloned()
}

#[inline(always)]
pub fn spec(coin_id: usize) -> SymbolSpec {
    match try_spec(coin_id) {
        Some(s) => s,
        None => panic!("⚠️ [CRITICAL] Intento de acceder a SymbolSpec no inicializado o fuera de rango (ID: {}).", coin_id),
    }
}

fn get_default_specs() -> Vec<SymbolSpec> {
    Vec::new()
}
