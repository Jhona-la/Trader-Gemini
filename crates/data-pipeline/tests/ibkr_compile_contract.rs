// IBKR-I1..I8 (integrador, 2026-11-16): contrato de compilacion del conector
// IBKR de SOLO LECTURA (pedido junie_cli_pedido_conector_ibkr_solo_lectura_v2.md,
// revisado por QODER-CLI: APTO). Ninguna prueba de este archivo usa red ni el
// IB Gateway.

/// I1: los tipos de lectura de ibapi 2.7.0 son accesibles (pin exacto =2.7.0).
#[test]
fn i1_ibapi_dependency_compiles() {
    use ibapi::contracts::Contract;
    use ibapi::market_data::MarketDataType;
    use ibapi::Client;
    let _ = std::any::type_name::<Client>();
    let _ = std::any::type_name::<Contract>();
    let _ = std::any::type_name::<MarketDataType>();
}

/// I2: el modulo ibkr publica sus tipos de solo lectura y las constantes.
#[test]
fn i2_ibkr_module_exposes_read_only_types() {
    use data_pipeline::ibkr::{IbkrBar, IbkrGateway, IBKR_HISTORICAL_MAX_PER_10MIN};
    let _ = std::any::type_name::<IbkrGateway>();
    let _ = std::any::type_name::<IbkrBar>();
    assert_eq!(IBKR_HISTORICAL_MAX_PER_10MIN, 60);
}

/// I2: el modulo importa los tipos exactos de ibapi, no su prelude.
#[test]
fn i2_ibkr_module_no_usa_prelude() {
    let fuente = include_str!("../src/ibkr/mod.rs");
    assert!(!fuente.contains("ibapi::prelude"), "src/ibkr/mod.rs no debe importar ibapi::prelude");
    for tipo in ["use ibapi::Client;", "use ibapi::contracts::Contract;", "use ibapi::market_data::MarketDataType;"] {
        assert!(fuente.contains(tipo), "falta el import exacto {tipo}");
    }
}