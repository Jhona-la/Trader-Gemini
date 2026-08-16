use arc_swap::ArcSwap;
use std::sync::Arc;

lazy_static::lazy_static! {
    static ref DYNAMIC_UNIVERSE: ArcSwap<Vec<String>> = ArcSwap::from_pointee(get_default_symbols());
}

pub fn update_dynamic_universe(new_universe: Vec<String>) {
    DYNAMIC_UNIVERSE.store(Arc::new(new_universe));
}

pub fn get_active_universe() -> Vec<String> {
    let u = DYNAMIC_UNIVERSE.load();
    (**u).clone()
}

pub fn get_active_universe_size() -> usize {
    DYNAMIC_UNIVERSE.load().len()
}

pub fn get_coin_id(symbol: &str) -> Option<usize> {
    let u = DYNAMIC_UNIVERSE.load();
    u.iter().position(|s| s == symbol)
}

fn get_default_symbols() -> Vec<String> {
    Vec::new()
}
