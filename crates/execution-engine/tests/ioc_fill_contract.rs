//! CL-39 — una IOC aceptada no es un llenado: el estado terminal y la
//! cantidad ejecutada de la PROPIA orden deciden si hubo entrada. Respuestas
//! sintéticas; sin red.
use execution_engine::ioc_evidence::{
    clasificar_respuesta_ioc, resultado_para_el_host, ResultadoIoc, IOC_UNFILLED,
};

const SYM: &str = "AUDITUSDT";
const ID: &str = "cL_ioc1";

fn cuerpo(status: &str, orig: &str, executed: Option<&str>, avg: &str) -> String {
    let executed = executed
        .map(|e| format!(r#","executedQty":"{e}""#))
        .unwrap_or_default();
    format!(
        r#"{{"symbol":"{SYM}","clientOrderId":"{ID}","orderId":7,"side":"BUY","status":"{status}","origQty":"{orig}"{executed},"avgPrice":"{avg}","type":"LIMIT","timeInForce":"IOC"}}"#
    )
}

#[test]
fn cl39_expirada_sin_ejecucion_no_es_entrada_y_no_es_ambigua() {
    let r = clasificar_respuesta_ioc(&cuerpo("EXPIRED", "1.2", Some("0"), "0"), SYM, ID)
        .expect("evidencia válida");
    assert!(matches!(r, ResultadoIoc::SinLlenado { .. }));
    let e = resultado_para_el_host(&r, ID).unwrap_err();
    assert!(e.starts_with(IOC_UNFILLED), "{e}");
    assert!(
        !e.starts_with("AMBIGUOUS") && !e.starts_with("MAKER_CHASE_UNVERIFIED"),
        "el host debe REVERTIR (rama no ambigua): {e}"
    );
}

#[test]
fn cl39_expirada_con_ejecucion_parcial_es_entrada() {
    let r = clasificar_respuesta_ioc(&cuerpo("EXPIRED", "1.2", Some("0.5"), "4.506"), SYM, ID)
        .expect("evidencia válida");
    match &r {
        ResultadoIoc::Llenada { ejecutada, .. } => assert_eq!(*ejecutada, 0.5),
        otro => panic!("parcial debe ser Llenada: {otro:?}"),
    }
    assert_eq!(resultado_para_el_host(&r, ID), Ok(()));
}

#[test]
fn cl39_llena_es_entrada() {
    let r = clasificar_respuesta_ioc(&cuerpo("FILLED", "1.2", Some("1.2"), "4.507"), SYM, ID)
        .expect("evidencia válida");
    assert!(matches!(r, ResultadoIoc::Llenada { ejecutada, .. } if ejecutada == 1.2));
    assert_eq!(resultado_para_el_host(&r, ID), Ok(()));
}

#[test]
fn cl39_sin_evidencia_terminal_es_ambigua() {
    // Estado no terminal (respuesta ACK), campo ausente (no es cero) e
    // identidad ajena: la orden pudo ejecutarse y no se revierte.
    for (body, id) in [
        (cuerpo("NEW", "1.2", Some("0"), "0"), ID),
        (cuerpo("EXPIRED", "1.2", None, "0"), ID),
        (cuerpo("EXPIRED", "1.2", Some("0"), "0"), "cL_otra"),
        ("{".to_string(), ID),
    ] {
        let e = clasificar_respuesta_ioc(&body, SYM, id).unwrap_err();
        assert!(e.starts_with("AMBIGUOUS"), "{body} -> {e}");
        assert!(!e.contains("MAKER"), "la telemetría no debe decir maker en una IOC: {e}");
    }
}

/// El envío real usa la evidencia: ni POST sin cuerpo ni Ok ante cualquier
/// 2xx. Guardia sobre la fuente (las URL del exchange están fijadas en el
/// código y no hay doble de red).
#[test]
fn cl39_el_envio_de_la_ioc_lee_su_estado_terminal() {
    let src: String = include_str!("../src/executor.rs")
        .split_whitespace()
        .collect();
    let desde = src
        .rfind("asyncfnexecute_ioc_order(")
        .expect("impl de execute_ioc_order");
    let resto = &src[desde + 1..];
    let hasta = resto.find("asyncfn").unwrap_or(resto.len());
    let ioc = &resto[..hasta];
    assert!(
        !ioc.contains(".execute_order_payload("),
        "un HTTP 2xx (EXPIRED, executedQty=0) no es un fill"
    );
    for requerido in [
        "newOrderRespType=RESULT",
        "register_intent(",
        "is_overflow()",
        "execute_order_payload_body(",
        "clasificar_respuesta_ioc(",
        "apply_ack(",
        "handle_rate_limit_error(",
    ] {
        assert!(ioc.contains(requerido), "la IOC debe usar {requerido}");
    }
}
