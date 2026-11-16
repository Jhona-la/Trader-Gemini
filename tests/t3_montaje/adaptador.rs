//! DEMO_TEL T3 — adaptador de montaje de la ENTREGA (qra_t3_criterios.md, recibos :74).
//!
//! Los oráculos T3-1 y T3-5 (`demo_tel_capital_update_sink_contract.rs` y
//! `demo_tel_lines_contract.rs`) son el mismo archivo en la base `bb2db3e4` y en la
//! entrega. Lo único que cambia entre pines es este adaptador: conecta el oráculo con
//! las rutas reales de cada SHA (en la base, `t3_logs/adaptador_base.rs`, que llama a lo
//! que hace el host de bb2db3e4). No contiene aserciones del oráculo.
#![allow(dead_code)]

use execution_engine::puente_cuenta::{LecturaCuenta, Observacion, PuenteCuenta, Rechazo, Ticket};

pub const PIN: &str = "entrega";

/// La REST de arranque: lo mismo que el host (`god_engine.rs`, arranque) hace con la
/// lectura: el puente la observa y, si la acepta, `capital_rest_aceptado` la registra.
pub fn rest_inicial(puente: &PuenteCuenta, ticket: Ticket, sesion: u64, l: &LecturaCuenta) -> Result<Observacion, Rechazo> {
    let obs = puente.observar_rest_de_sesion(ticket, sesion, l);
    if let Ok(aceptada) = &obs {
        quantum_engine::demo_tel::capital_rest_aceptado(&ticket, l, aceptada);
    }
    obs
}

/// La resincronización periódica del host: el GET envuelto por `LecturaRestAnotada`,
/// `resincronizar_una_vez` real y el registro de la pasada aceptada.
pub async fn resincronizar<F>(
    puente: &PuenteCuenta,
    sesion: u64,
    lectura: F,
    publicar: impl FnOnce(&Observacion),
) -> Result<Observacion, String>
where
    F: std::future::Future<Output = Result<LecturaCuenta, String>>,
{
    let rest_anotada = quantum_engine::demo_tel::LecturaRestAnotada::default();
    let r = execution_engine::resincronizar_cuenta::resincronizar_una_vez(
        puente,
        sesion,
        rest_anotada.anotar(puente, lectura),
        std::time::Duration::from_secs(5),
        publicar,
    )
    .await;
    if let Ok(obs) = &r {
        rest_anotada.registrar(obs);
    }
    r
}

fn en_orden(fuente: &str, pasos: &[&str]) -> Result<(), String> {
    let f = fuente.replace("\r\n", "\n");
    let mut resto = f.as_str();
    for p in pasos {
        resto = resto.split_once(p).ok_or_else(|| format!("montaje: ancla ausente o fuera de orden: {p}"))?.1;
    }
    Ok(())
}

/// El host de este pin llama a lo mismo que este adaptador (montaje con rutas reales).
pub fn ruta_rest_del_host(host: &str) -> Result<(), String> {
    en_orden(host, &[
        ".observar_rest_de_sesion(ticket, token_arranque.generation(), &l)",
        "quantum_engine::demo_tel::capital_rest_aceptado(&ticket, &l, aceptada);",
    ])?;
    en_orden(host, &[
        "let rest_anotada = quantum_engine::demo_tel::LecturaRestAnotada::default();",
        "resincronizar_cuenta::resincronizar_una_vez(",
        "rest_anotada.anotar(",
        "executor.fetch_account_lectura(),",
        "rest_anotada.registrar(obs);",
    ])?;
    en_orden(host, &[
        "puente_transicion.observar_rest_de_sesion(ticket, balance_token.generation(), l)",
        "quantum_engine::demo_tel::capital_rest_aceptado(&ticket, l, &aceptada_rest);",
    ])
}

fn disparo(c: &super::CasoInmune) -> quantum_engine::demo_tel::DisparoInmune {
    quantum_engine::demo_tel::DisparoInmune {
        ts_ms: c.ts_ms,
        motivo: c.motivo,
        freno_capital: c.freno_capital,
        peak_capital: c.peak_capital,
        equity_del_freno: c.equity_del_freno.unwrap_or(f64::NAN),
        caida_capital_pct: c.caida_capital_pct,
        tope_capital_pct: c.tope_capital_pct,
        operator_lock: c.operator_lock,
        insolvent: c.insolvent,
        dd_breach: c.dd_breach,
        lat_breach: c.lat_breach,
        latency_strikes: c.latency_strikes,
    }
}

/// El emisor que llama la rama real del disparo.
pub fn emitir_immune(c: &super::CasoInmune) {
    quantum_engine::demo_tel::emitir_immune_trigger(&disparo(c));
}

/// El texto que ese emisor entrega al logger (antes del prefijo).
pub fn linea_immune(c: &super::CasoInmune) -> String {
    let mut s = String::new();
    quantum_engine::demo_tel::escribir_immune_trigger(&mut s, &disparo(c)).expect("escribe");
    s
}

/// Significado de un plano: el código corto según la tabla del lector (o el texto largo de T2).
pub fn significado_plano(campo: &str, valor: &str) -> Option<String> {
    telemetry_server::report_demo_run::PLANOS_IMMUNE
        .iter()
        .find(|(k, codigo, largo)| *k == campo && (valor == *codigo || valor == *largo))
        .map(|(_, _, largo)| largo.to_string())
}

/// La rama real del disparo llama a este emisor tras comprometer el disparo (C3, decision del
/// coordinador D1-D2: `comprometer_disparo` arma el kill-switch bajo el mutex del puente; antes,
/// el emisor iba antes de `arena_imm.kill_switch_active.store` en el host).
pub fn ruta_del_disparo(host: &str) -> Result<(), String> {
    en_orden(host, &[
        "if operator_lock || insolvent || dd_breach || lat_breach {",
        "execution_engine::aplanado_inmune::comprometer_disparo(",
        "🚨 [SISTEMA INMUNE] ACTIVADO",
        "quantum_engine::demo_tel::emitir_immune_trigger(&quantum_engine::demo_tel::DisparoInmune {",
        "autoria_del_latch = Some(autoria);",
    ])
}
