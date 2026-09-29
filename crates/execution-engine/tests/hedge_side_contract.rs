//! CL-12 — en modo hedge el protocolo de emergencia del host actúa sobre el
//! LADO de la entrada, nunca sobre el primer registro abierto del símbolo.
use execution_engine::reconciliation::cantidad_abierta_del_lado;
use execution_engine::PositionRiskEntry;

fn fila(symbol: &str, amt: f64, side: &str) -> PositionRiskEntry {
    PositionRiskEntry {
        symbol: symbol.into(),
        position_amt: amt,
        position_side: side.into(),
        ..Default::default()
    }
}

#[test]
fn cl12_la_cantidad_es_la_del_lado_pedido() {
    // El registro del lado contrario ordena primero, como en el hallazgo.
    let ps = [
        fila("ETHUSDT", 3.0, "LONG"),
        fila("SOLUSDT", -0.5, "SHORT"),
        fila("SOLUSDT", 0.2, "LONG"),
    ];
    assert_eq!(cantidad_abierta_del_lado(&ps, "SOLUSDT", true), Some(0.2));
    assert_eq!(cantidad_abierta_del_lado(&ps, "SOLUSDT", false), Some(0.5));
    // Lado plano (registro vacío primero) con el contrario abierto: no hay nada
    // que proteger ni que cerrar de ESTE lado.
    let plano = [fila("SOLUSDT", 0.0, "LONG"), fila("SOLUSDT", -0.5, "SHORT")];
    assert_eq!(cantidad_abierta_del_lado(&plano, "SOLUSDT", true), None);
    // Modo one-way: el signo decide el lado.
    let one_way = [fila("SOLUSDT", -0.7, "BOTH")];
    assert_eq!(cantidad_abierta_del_lado(&one_way, "SOLUSDT", false), Some(0.7));
    assert_eq!(cantidad_abierta_del_lado(&one_way, "SOLUSDT", true), None);
}

#[test]
fn cl12_el_protocolo_de_emergencia_del_host_respeta_el_lado() {
    let host: String = include_str!("../../../src/bin/god_engine.rs")
        .split_whitespace()
        .collect();
    assert!(!host.contains("p.symbol==parsed_sym_str&&p.position_amt.abs()>0.0"));
    assert!(!host.contains("cancel_all_symbol_orders(&parsed_sym_str)"));
    assert_eq!(host.matches("cantidad_abierta_del_lado(").count(), 3);
}
