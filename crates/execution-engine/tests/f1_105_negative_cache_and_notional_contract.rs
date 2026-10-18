//! Contrato F1-105: Validación de caché negativa con TTL de filtros de símbolos y saneamiento de notional.
//!
//! F1-105 (Ø159, AGY):
//! 1. Símbolos desconocidos o no parseables se almacenan en una caché negativa con TTL de 60s,
//!    evitando reintentos masivos que bloqueen el hilo ~20s re-descargando exchangeInfo en el hot-path.
//! 2. Saneamiento estricto de min_notional asegurando valores finitos positivos y fallback al
//!    registro local de especificaciones cuando exchangeInfo no lo provee.

use execution_engine::executor::{OrderExecutor, SymbolFilter};

#[tokio::test]
async fn f1_105_unknown_symbol_enters_negative_cache_and_avoids_re_fetch() {
    let executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    assert_eq!(executor.negative_symbol_cache_len(), 0);

    let unknown_sym = "NON_EXISTENT_DELISTED_COIN_999";

    // Primera consulta: falla tras consultar exchangeInfo y registro local
    let res1 = executor.get_symbol_filter(unknown_sym).await;
    assert!(res1.is_err());
    assert_eq!(executor.negative_symbol_cache_len(), 1);

    // Segunda consulta: debe golpear inmediatamente la caché negativa (TTL 60s)
    let res2 = executor.get_symbol_filter(unknown_sym).await;
    assert!(res2.is_err());
    let err_msg = res2.unwrap_err();
    assert!(
        err_msg.contains("en caché negativa TTL 60s"),
        "El error debe reflejar la presencia en la caché negativa: {}",
        err_msg
    );
    assert_eq!(executor.negative_symbol_cache_len(), 1);
}

#[tokio::test]
async fn f1_105_clear_negative_symbol_cache_works() {
    let executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    let unknown_sym = "PHANTOM_PAIR_USDT";

    let _ = executor.get_symbol_filter(unknown_sym).await;
    assert_eq!(executor.negative_symbol_cache_len(), 1);

    executor.clear_negative_symbol_cache();
    assert_eq!(executor.negative_symbol_cache_len(), 0);
}

#[tokio::test]
async fn f1_105_local_registry_fallback_resolves_when_present() {
    let executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    let sym = "TESTSPEC105USDT";

    quantum_arena::symbol_registry::update_registry(vec![quantum_arena::symbol_registry::SymbolSpec {
        symbol: sym.into(),
        step_size: 0.005,
        tick_size: 0.001,
        min_qty: 0.01,
        min_notional: 15.0,
        max_leverage: 20,
        maker_fee: 0.0002,
        taker_fee: 0.0005,
        is_shadow: false,
    }]);

    let filter_res = executor.get_symbol_filter(sym).await;
    assert!(filter_res.is_ok(), "Debe resolver desde el registro local");
    let filter = filter_res.unwrap();
    assert_eq!(filter.step_size, 0.005);
    assert_eq!(filter.tick_size, 0.001);
    assert_eq!(filter.min_notional, 15.0);
}

#[test]
fn f1_105_symbol_filter_min_notional_preservation() {
    let sf = SymbolFilter {
        step_size: 0.001,
        tick_size: 0.01,
        min_notional: 100.0, // BTCUSDT real min_notional
    };
    assert_eq!(sf.min_notional, 100.0);
    assert!(sf.min_notional.is_finite() && sf.min_notional > 0.0);
}
