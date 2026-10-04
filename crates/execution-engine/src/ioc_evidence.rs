//! CL-39 — UNA IOC ACEPTADA NO ES UN LLENADO.
//!
//! AGY-AUD-P32 enruta casi toda entrada pequeña como LIMIT IOC con un tope de
//! deslizamiento. El envío daba `Ok(())` ante cualquier HTTP 2xx sin leer el
//! cuerpo: una IOC que expira sin ejecutar nada (HTTP 200, `EXPIRED`,
//! `executedQty = 0`) se confirmaba como entrada. El host dejaba entonces una
//! posición FANTASMA en el arena (margen retenido, comisión de entrada
//! cobrada, banda τ ocupada) que sólo la reconciliación limpiaba, y nunca si
//! la moneda tenía dos ranuras abiertas. Además, un 5xx o un -1007 salía como
//! rechazo y el host revertía una orden que pudo ejecutarse.
//!
//! Aquí se decide con la evidencia de la PROPIA orden (respuesta `RESULT`):
//! - terminal con ejecución > 0 (total o parcial) ⇒ hubo entrada;
//! - terminal sin ejecución ⇒ `IOC_UNFILLED`, un error NO ambiguo: el host
//!   revierte la reserva y devuelve margen y comisión;
//! - cuerpo sin los campos de evidencia o estado no terminal ⇒ `AMBIGUOUS`:
//!   la orden pudo ejecutarse; el host conserva la reserva y reconcilia.
//!
//! Reutiliza sin cambios los validadores estrictos de `execution_evidence`
//! (un campo ausente no se toma por cero).
use crate::execution_evidence::{parse_query_order_response, terminal_maker_executed_quantity};
use crate::{OrderAck, OrderStatus, TrackedOrder};

/// Prefijo del error no ambiguo de una IOC terminada sin ejecución.
pub const IOC_UNFILLED: &str = "IOC_UNFILLED";

#[derive(Debug, Clone)]
pub enum ResultadoIoc {
    /// Estado terminal con cantidad ejecutada > 0 (total o parcial).
    Llenada { ack: OrderAck, ejecutada: f64 },
    /// Estado terminal (EXPIRED, CANCELED, REJECTED) sin ejecución.
    SinLlenado { ack: OrderAck },
}

impl ResultadoIoc {
    pub fn ack(&self) -> &OrderAck {
        match self {
            ResultadoIoc::Llenada { ack, .. } | ResultadoIoc::SinLlenado { ack } => ack,
        }
    }
}

/// Clasifica el cuerpo de la respuesta `newOrderRespType=RESULT` de una IOC.
pub fn clasificar_respuesta_ioc(
    body: &str,
    symbol: &str,
    client_order_id: &str,
) -> Result<ResultadoIoc, String> {
    let ack = parse_query_order_response(body, symbol, client_order_id)
        .map_err(|e| format!("AMBIGUOUS: IOC_EVIDENCE_INVALID id={client_order_id}: {e}"))?;
    let ejecutada = terminal_maker_executed_quantity(symbol, client_order_id, &ack).map_err(|_| {
        format!(
            "AMBIGUOUS: IOC_TERMINAL_UNVERIFIED id={client_order_id} status={}",
            ack.status
        )
    })?;
    if ejecutada > 0.0 {
        Ok(ResultadoIoc::Llenada { ack, ejecutada })
    } else {
        Ok(ResultadoIoc::SinLlenado { ack })
    }
}

/// CL-39b: un error del envío cierra la intención local sólo si es firme (la
/// orden nunca llegó a existir: rechazo 4xx con código, 429, 418). Un error
/// `AMBIGUOUS` (5xx, 408, -1006/-1007, cuerpo ilegible, red) la deja viva:
/// la orden pudo ejecutarse y la consulta por REST la resuelve.
pub fn error_cierra_la_intencion(error: &str) -> bool {
    !error.starts_with("AMBIGUOUS")
}

/// Lo que ve el host: `Ok(())` sólo si la orden ejecutó algo.
pub fn resultado_para_el_host(r: &ResultadoIoc, client_order_id: &str) -> Result<(), String> {
    match r {
        ResultadoIoc::Llenada { .. } => Ok(()),
        ResultadoIoc::SinLlenado { ack } => Err(format!(
            "{IOC_UNFILLED}: id={client_order_id} status={} sin ejecución",
            ack.status
        )),
    }
}

/// CL-45: cantidad ejecutada de una orden YA TERMINAL según el registro
/// (respuesta `RESULT` fusionada con los fills del WS). `None` si la orden no
/// está, sigue activa, su estado es desconocido o no ejecutó nada: entonces
/// no hay evidencia de cantidad y el host confirma como antes.
pub fn cantidad_ejecutada_terminal(orden: Option<&TrackedOrder>) -> Option<f64> {
    let o = orden?;
    if o.status.is_active() || o.status == OrderStatus::Unknown {
        return None;
    }
    (o.executed_qty.is_finite() && o.executed_qty > 0.0).then_some(o.executed_qty)
}
