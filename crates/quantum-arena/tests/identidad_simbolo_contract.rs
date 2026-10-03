//! CL-37 — UN SOLO NOMBRE POR SÍMBOLO. El universo dinámico, el registro de
//! specs y todas las claves que distinguen mayúsculas (`{SYM}_MOTOR`,
//! `{SYM}_VOL`, el funding con scope, el NN sólo-BTC) deben hablar la MISMA
//! grafía. El bootloader publicaba el universo en minúsculas y el registro en
//! MAYÚSCULAS: para el mismo slot `try_symbol` y `try_spec` discrepaban, y
//! ningún bosque `{SYM}_MOTOR` (cargado por el stem del archivo, en
//! MAYÚSCULAS) se encontraba nunca.
use quantum_arena::{symbol_registry, symbols};

// El universo es un estado global del proceso: los tests de este binario
// lo serializan entre sí.
static M: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn cl37_try_symbol_try_spec_y_try_index_son_el_mismo_mapa() {
    let _g = M.lock().unwrap_or_else(|e| e.into_inner());
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("IDSAUSDT"),
        symbol_registry::get_official_binance_spec("IDSBUSDT"),
    ]);
    // Como lo publicaba el host: la lista del bootloader en minúsculas.
    symbols::update_dynamic_universe(vec!["idsbusdt".into(), "idsausdt".into()]);
    for i in 0..2 {
        let s = symbol_registry::try_symbol(i).expect("slot sin símbolo");
        let spec = symbol_registry::try_spec(i).expect("slot sin spec");
        assert_eq!(
            s, spec.symbol,
            "slot {i}: el universo y el registro deben dar la MISMA grafía"
        );
        assert_eq!(symbol_registry::try_index(&s), Some(i));
        assert_eq!(symbols::get_coin_id(&s), Some(i));
    }
    assert_eq!(symbol_registry::try_symbol(0).as_deref(), Some("IDSBUSDT"));
}

#[test]
fn cl37_el_universo_se_publica_canonico_y_conserva_el_orden() {
    let _g = M.lock().unwrap_or_else(|e| e.into_inner());
    symbols::update_dynamic_universe(vec![
        "idscusdt".into(),
        "IdsDUsdt".into(),
        "IDSEUSDT".into(),
    ]);
    assert_eq!(
        symbols::get_active_universe(),
        vec!["IDSCUSDT", "IDSDUSDT", "IDSEUSDT"],
        "el orden de los slots no cambia; sólo la grafía"
    );
    assert_eq!(symbols::simbolo_canonico("btcUSDT"), "BTCUSDT");
}
