//! CL-11 — la envolvente del host y la del replay razonan con el nocional
//! mínimo del SÍMBOLO (registro de `exchangeInfo`), no con el literal 5,0.
use quantum_arena::symbol_registry::{update_registry, SymbolSpec};
use risk_engine::capital_regime::{min_notional_del_simbolo, DEFAULT_MIN_NOTIONAL};

fn spec(symbol: &str, min_notional: f64) -> SymbolSpec {
    SymbolSpec {
        symbol: symbol.into(),
        step_size: 0.001,
        tick_size: 0.01,
        min_qty: 0.001,
        min_notional,
        max_leverage: 20,
        maker_fee: 0.0002,
        taker_fee: 0.0004,
        is_shadow: false,
    }
}

#[test]
fn cl11_el_minimo_es_el_del_simbolo() {
    update_registry(vec![spec("CLONCEUSDT", 100.0), spec("CLONCEBUSDT", 5.0)]);
    assert_eq!(min_notional_del_simbolo(0), 100.0);
    assert_eq!(min_notional_del_simbolo(1), 5.0);
    assert_eq!(min_notional_del_simbolo(999), DEFAULT_MIN_NOTIONAL);
}

#[test]
fn cl11_host_y_replay_no_llevan_el_literal() {
    for (nombre, codigo) in [
        ("god_engine.rs", include_str!("../../../src/bin/god_engine.rs")),
        ("booktick_replay.rs", include_str!("../../backtest-engine/src/booktick_replay.rs")),
    ] {
        let plano: String = codigo.split_whitespace().collect();
        assert!(!plano.contains("letenv_min_notional=5.0;"), "{nombre}");
        assert!(
            plano.contains("letenv_min_notional=risk_engine::capital_regime::min_notional_del_simbolo(coin_id);"),
            "{nombre}"
        );
    }
}
