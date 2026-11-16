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
