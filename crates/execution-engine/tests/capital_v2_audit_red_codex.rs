//! Dictamen de Codex QRA sobre 1757709c4 (P1.1): sus dos RED, TAL COMO ESTÁN
//! ESCRITAS en logs/m2-calidad-datos-20261111/CAPITAL-V2.audit.RED.rs. Sólo se
//! añade esta cabecera y el `#[path]` que apunta al módulo (su arnés lo
//! compilaba suelto, con `mod puente_cuenta;` al lado).

#[path = "../src/puente_cuenta.rs"]
#[allow(dead_code)]
mod puente_cuenta;
use puente_cuenta::{EventoStream, LecturaRest, PuenteCuenta};
use std::collections::HashMap;

fn rest(saldo: f64, tiempo_ms: u64) -> LecturaRest {
    LecturaRest { saldo, upnl: Some(HashMap::new()), upnl_del_activo: 0.0, tiempo_ms: Some(tiempo_ms) }
}

#[test]
fn older_rest_cannot_rewind_an_already_accepted_websocket_balance() -> Result<(), Box<dyn std::error::Error>> {
    let puente = PuenteCuenta::new();
    puente.observar_stream(puente.era(), &EventoStream {
        tiempo_ms: Some(200), saldo: Some(0.0), posiciones: vec![],
    }).map_err(|e| e.to_string())?;
    let ticket = puente.ticket();
    let result = puente.observar_rest(ticket, &rest(100.0, 100));
    assert!(result.is_err(), "stale REST restored spent capital: {result:?}");
    Ok(())
}

#[test]
fn older_balance_does_not_gain_freshness_from_newer_position_timestamp() -> Result<(), Box<dyn std::error::Error>> {
    let puente = PuenteCuenta::new();
    let ticket = puente.ticket();
    puente.observar_stream(puente.era(), &EventoStream {
        tiempo_ms: Some(200), saldo: Some(0.0), posiciones: vec![],
    }).map_err(|e| e.to_string())?;
    let result = puente.observar_rest(ticket, &rest(100.0, 201));
    assert!(result.is_err(), "a single composite timestamp does not prove freshness of the balance: {result:?}");
    Ok(())
}
