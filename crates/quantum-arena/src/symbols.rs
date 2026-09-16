use arc_swap::ArcSwap;
use std::sync::Arc;

lazy_static::lazy_static! {
    static ref DYNAMIC_UNIVERSE: ArcSwap<Vec<String>> = ArcSwap::from_pointee(get_default_symbols());
}

pub fn update_dynamic_universe(new_universe: Vec<String>) {
    DYNAMIC_UNIVERSE.store(Arc::new(new_universe));
}

pub fn get_active_universe_ref() -> Arc<Vec<String>> {
    DYNAMIC_UNIVERSE.load_full()
}

pub fn with_active_universe<F, R>(f: F) -> R
where
    F: FnOnce(&[String]) -> R,
{
    let u = DYNAMIC_UNIVERSE.load();
    f(&u)
}

pub fn get_active_universe() -> Vec<String> {
    let u = DYNAMIC_UNIVERSE.load();
    (**u).clone()
}

pub fn get_active_universe_size() -> usize {
    DYNAMIC_UNIVERSE.load().len()
}

/// D-725: el id de una moneda es su posición en el UNIVERSO — el mismo índice
/// con el que el productor de datos escribe en `arena.coins[i]`. Sólo con el
/// universo vacío (arranque, o binarios que sólo registran specs) se recurre al
/// registro, que entonces es el único espacio de índices existente.
pub fn get_coin_id(symbol: &str) -> Option<usize> {
    let u = DYNAMIC_UNIVERSE.load();
    if !u.is_empty() {
        return u.iter().position(|s| s.eq_ignore_ascii_case(symbol));
    }
    crate::symbol_registry::try_index(symbol)
}

fn get_default_symbols() -> Vec<String> {
    Vec::new()
}

#[cfg(test)]
pub static UNIVERSE_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_universe_update_and_lookup() {
        let _guard = UNIVERSE_TEST_MUTEX.lock().unwrap();
        let universe = vec![
            "BTCUSDT".to_string(),
            "ETHUSDT".to_string(),
            "SOLUSDT".to_string(),
        ];
        update_dynamic_universe(universe.clone());

        assert_eq!(get_active_universe_size(), 3);
        assert_eq!(get_active_universe(), universe);

        with_active_universe(|u| {
            assert_eq!(u.len(), 3);
            assert_eq!(u[0], "BTCUSDT");
        });
    }
}
