//! CL-39 — una IOC aceptada no es un llenado: el estado terminal y la
//! cantidad ejecutada de la PROPIA orden deciden si hubo entrada. Respuestas
//! sintéticas; sin red.
use execution_engine::ioc_evidence::{
    cantidad_ejecutada_terminal, clasificar_respuesta_ioc, destino_tras_consulta,
    error_cierra_la_intencion, resultado_para_el_host, DestinoReserva, ResultadoIoc, IOC_UNFILLED,
};
use execution_engine::{OrderRegistry, OrderResolution, TrackedOrder};

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
        "mark_local_reject(",
    ] {
        assert!(ioc.contains(requerido), "la IOC debe usar {requerido}");
    }
    // CL-39b: la intención se registra después de la última salida previa
    // al envío (el buffer desbordado no deja una intención huérfana).
    let registro = ioc.find("register_intent(").unwrap();
    let desborde = ioc.find("is_overflow()").unwrap();
    assert!(desborde < registro, "intención registrada antes de poder abortar sin enviar");
}

/// CL-39c: sólo un error firme cierra la intención local. Con un error
/// ambiguo la orden pudo ejecutarse: cerrarla dejaría de seguir una posición
/// real. La guardia exige además que el ejecutor pase por esta decisión.
#[test]
fn cl39c_solo_un_error_firme_cierra_la_intencion() {
    for ambiguo in [
        "AMBIGUOUS: HTTP 503 body=",
        "AMBIGUOUS: HTTP 408 body=",
        r#"AMBIGUOUS: HTTP 400 body={"code":-1007,"msg":"Timeout"}"#,
        "AMBIGUOUS: HTTP 200 unreadable response body",
        "AMBIGUOUS: Network Error: connection reset",
    ] {
        assert!(!error_cierra_la_intencion(ambiguo), "{ambiguo}");
    }
    for firme in [
        "HTTP_429_RATE_LIMITED retry_after=1",
        "HTTP_418_IP_BANNED",
        "BINANCE_REJECT code=-2019 Margin is insufficient.",
    ] {
        assert!(error_cierra_la_intencion(firme), "{firme}");
    }

    let src: String = include_str!("../src/executor.rs").split_whitespace().collect();
    assert_eq!(src.matches("mark_local_reject(").count(), 1, "un único cierre local");
    assert!(
        src.contains(
            "ifcrate::ioc_evidence::error_cierra_la_intencion(&e){self.order_registry.mark_local_reject("
        ),
        "el cierre local va detrás de la decisión firme/ambigua"
    );
}

/// La orden tal como la deja el registro tras aplicar la respuesta `body`.
fn orden(body: &str) -> TrackedOrder {
    let r = clasificar_respuesta_ioc(body, SYM, ID).expect("evidencia válida");
    let reg = OrderRegistry::new();
    reg.register_intent(ID, SYM, "BUY", "LONG", "LIMIT", 1.2, 1);
    reg.apply_ack(r.ack(), 2);
    reg.get(ID).expect("orden registrada")
}

/// CL-45: sólo una orden terminal con ejecución aporta cantidad.
#[test]
fn cl45_cantidad_ejecutada_sale_de_la_orden_terminal() {
    let parcial = orden(&cuerpo("EXPIRED", "1.2", Some("0.5"), "4.506"));
    assert_eq!(cantidad_ejecutada_terminal(Some(&parcial)), Some(0.5));
    let llena = orden(&cuerpo("FILLED", "1.2", Some("1.2"), "4.507"));
    assert_eq!(cantidad_ejecutada_terminal(Some(&llena)), Some(1.2));
    let vacia = orden(&cuerpo("EXPIRED", "1.2", Some("0"), "0"));
    assert_eq!(cantidad_ejecutada_terminal(Some(&vacia)), None);
    assert_eq!(cantidad_ejecutada_terminal(None), None);
}

/// CL-46: la consulta REST de una entrada ambigua decide la reserva.
#[test]
fn cl46_destino_de_la_reserva_tras_la_consulta() {
    let ambiguo = "AMBIGUOUS: HTTP 503 body=";
    let parcial = orden(&cuerpo("EXPIRED", "1.2", Some("0.5"), "4.506"));
    let llena = orden(&cuerpo("FILLED", "1.2", Some("1.2"), "4.507"));
    let viva = {
        let reg = OrderRegistry::new();
        reg.register_intent(ID, SYM, "BUY", "LONG", "LIMIT", 1.2, 1);
        reg.get(ID).unwrap()
    };
    let casos: [(&str, OrderResolution, Option<&TrackedOrder>, DestinoReserva); 7] = [
        (ambiguo, OrderResolution::Rejected, None, DestinoReserva::Revertir),
        (ambiguo, OrderResolution::Accepted, Some(&parcial), DestinoReserva::Confirmar { ejecutada: 0.5 }),
        (ambiguo, OrderResolution::Accepted, Some(&llena), DestinoReserva::Confirmar { ejecutada: 1.2 }),
        (ambiguo, OrderResolution::Accepted, Some(&viva), DestinoReserva::Conservar),
        (ambiguo, OrderResolution::Accepted, None, DestinoReserva::Conservar),
        (ambiguo, OrderResolution::Timeout, None, DestinoReserva::Conservar),
        ("MAKER_CHASE_UNVERIFIED: hijo mcT_1", OrderResolution::Rejected, None, DestinoReserva::Conservar),
    ];
    for (error, consulta, o, esperado) in casos {
        assert_eq!(destino_tras_consulta(error, consulta, o), esperado, "{error} {consulta:?}");
    }
}
