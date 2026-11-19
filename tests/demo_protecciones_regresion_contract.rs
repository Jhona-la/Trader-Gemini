//! DEMO (punto 3 del pedido «Demo en Testnet sin ambiguedad»): NINGUNA PROTECCION CONTRA LA RUINA
//! SE DEBILITA. Regresion de las cinco: freno duro, kill-switch, STOP del operador, fee breaker y
//! D-382. Las partes funcionales llaman a las funciones reales; las que viven en linea en
//! `god_engine.rs` se fijan con anclas de fuente (CRLF normalizado). Debe pasar igual en la base
//! (a1d3857fc) y tras DEMO-A/B/C: es una regresion, no una prueba del cambio.

use execution_engine::executor::{ExecutionProvider, OrderExecutor};
use quantum_arena::GlobalArena;
use risk_engine::drawdown::{drawdown_de_falsacion, freno_duro, ALFA_FALSACION, FRACCION_KELLY_MAXIMA};
use risk_engine::envio::{apalancamiento_de_envio, VetoEnvio};
use std::sync::atomic::Ordering;

const HOST: &str = include_str!("../src/bin/god_engine.rs");

fn produccion() -> String {
    HOST.replace("\r\n", "\n").split("#[cfg(test)]").next().unwrap_or("").to_string()
}

#[test]
fn proteccion_freno_duro_dispara_en_su_tope_y_no_antes() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.config.global_max_drawdown.store(0.5, Ordering::Relaxed);
    let tope = 0.5f64.min(drawdown_de_falsacion(FRACCION_KELLY_MAXIMA, ALFA_FALSACION));
    let justo = freno_duro(&arena, 100.0, 100.0 * (1.0 - tope));
    assert!(justo.por_capital() && justo.echado(), "en el tope dispara");
    let antes = freno_duro(&arena, 100.0, 100.0 * (1.0 - tope) + 1e-6);
    assert!(!antes.por_capital(), "por debajo del tope no dispara");
    // Cuenta liquidada: dispara (nunca se desarma).
    assert!(freno_duro(&arena, 100.0, 0.0).echado());
    // El host sigue llamando al MISMO freno en el lazo del inmune.
    assert!(produccion().contains("let freno = risk_engine::drawdown::freno_duro(&arena_imm, peak_capital, cap);"));
}

#[test]
fn proteccion_kill_switch_es_latch_y_cruza_el_relevo() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let previo = OrderExecutor::new(String::new(), "offline".into(), true);
    previo.arena.store(Some(std::sync::Arc::clone(&arena)));
    previo.trigger_kill_switch();
    assert!(previo.is_kill_switch_active());
    let (sucesor, _) = OrderExecutor::sucesor_de(&previo, "k".into(), "s".into(), true, std::sync::Arc::clone(&arena));
    assert!(sucesor.is_kill_switch_active(), "N58: el latch cruza el relevo");
}

#[test]
fn proteccion_stop_fee_breaker_y_d382_siguen_en_el_camino_vivo() {
    let p = produccion();
    // STOP del operador: el lazo del inmune lo lee en cada ciclo.
    assert!(p.contains("let operator_lock = operator_lock_exists();"));
    assert!(p.contains("fn operator_lock_exists() -> bool {"));
    // Fee breaker: veta la entrada del simbolo suspendido antes del envio.
    let envio = p.find("let decision_envio: Option<u32> = 'envio: {").expect("decision de envio");
    let tramo = &p[envio..];
    let fee = tramo.find("if is_symbol_suspended(&parsed_sym_str) {").expect("fee breaker");
    let d382 = tramo.find("risk_engine::envio::apalancamiento_de_envio(").expect("D-382");
    let spawn = tramo.find("rt_handle.spawn(async move {").expect("envio");
    assert!(fee < d382 && d382 < spawn, "fee breaker y D-382 antes del envio");
    // D-382 funcional: subordinado al validado, veto de margen y de la envolvente.
    assert_eq!(apalancamiento_de_envio(10, Some(20), 100.0, 10.0), Ok(13));
    assert_eq!(apalancamiento_de_envio(10, None, 100.0, 10.0), Err(VetoEnvio::SinValidacion));
    assert_eq!(apalancamiento_de_envio(20, Some(20), 100.0, 1.0), Err(VetoEnvio::Margen));
    assert_eq!(apalancamiento_de_envio(0, Some(20), 100.0, 10.0), Err(VetoEnvio::Envolvente));
}
