//! Adaptador de M: lo que el host de ESTE commit hace en linea (`src/bin/god_engine.rs`).
#![allow(dead_code)]
use super::*;

pub const PIN: &str = "M";

/// Reintento del inmune (god_engine.rs, bloque `if latched {`): vigencia de la autoria y toma del
/// ejecutor en UN paso bajo el mutex del relevo (`aplanado_inmune::sujeto_del_reintento`). `entre`
/// es lo que ocurre entre la comprobacion del latch del ciclo y la toma del ejecutor.
pub fn sujeto_del_reintento_del_host(
    p: &PuenteCuenta,
    a: &GlobalArena,
    vigente: &ArcSwap<OrderExecutor>,
    autoria: Option<execution_engine::aplanado_inmune::AutoriaDelDisparo>,
    entre: impl FnOnce(),
) -> Option<Arc<OrderExecutor>> {
    if autoria.is_some_and(|x| !x.vigente(a)) {
        return None;
    }
    entre();
    execution_engine::aplanado_inmune::sujeto_del_reintento(p, a, vigente, autoria)
}

/// Callback de income: guarda del profiler con el testigo de la consulta; despues (la ventana) el
/// certificado se publica con `host_oms::certificado_demo_de_era`.
#[allow(clippy::too_many_arguments)]
pub fn certificado_del_host(p: &PuenteCuenta, era: u64, ruta: &std::path::Path, cierres: u64, bruto: f64, fees: f64, wins: u64, entre: impl FnOnce()) -> bool {
    if p.con_era_vigente(era, |_| ()).is_none() {
        return false;
    }
    entre();
    matches!(quantum_engine::host_oms::certificado_demo_de_era(p, era, true, ruta, cierres, bruto, fees, wins), Some(Ok(true)))
}

/// Callback de income: las filas que el host entrega al registro de flujos.
pub fn flujos_de_la_consulta(e: &execution_engine::income_evidence::IngresosAgotados) -> Vec<execution_engine::order_types::IncomeEntry> {
    e.con_sin_identidad()
}
