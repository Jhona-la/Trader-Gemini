//! CL-45/CL-46: lo que el host hace con la reserva de una entrada depende de
//! la evidencia de la PROPIA orden. Guardias sobre la fuente del host (el
//! binario no es testeable por unidad); las decisiones viven en funciones
//! puras de `ioc_evidence`, con su propio contrato en `ioc_fill_contract.rs`.

fn host() -> String {
    include_str!("../../../src/bin/god_engine.rs")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

/// V4M (D1, autorizado por el coordinador 20261115-0050): el brazo del resultado de la entrada se
/// movio del host a `host_oms::resolver_resultado_de_entrada`, que el host llama con el resultado
/// real del envio y su reserva. Las anclas siguen comprobando las mismas propiedades: en la
/// resolucion (fuente `oms()`), y en el host lo que el host sigue haciendo tras ella (diario y bracket).
fn oms() -> String {
    include_str!("../../../src/host_oms.rs")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

fn tramo<'a>(h: &'a str, inicio: &str, desde: &str, hasta: &str) -> &'a str {
    let m = h.find(inicio).unwrap_or_else(|| panic!("ancla «{inicio}» ausente"));
    let i = m + h[m..].find(desde).unwrap_or_else(|| panic!("ancla «{desde}» ausente"));
    let f = i + h[i..].find(hasta).unwrap_or_else(|| panic!("ancla «{hasta}» ausente tras «{desde}»"));
    &h[i..f]
}

const RESOLUCION: &str = "pubasyncfnresolver_resultado_de_entrada(";

/// El host entrega a la resolucion el resultado REAL del envio, su sesion, su reserva y su orden.
fn el_host_usa_la_resolucion(h: &str) {
    assert!(
        h.contains("letdesenlace=quantum_engine::host_oms::resolver_resultado_de_entrada(entry_result,&entry_oms,&entry_session,&arena_clone,&entry_executor,&reservation,&client_id,"),
        "el host no resuelve con la resolucion productiva"
    );
}

/// CL-46: antes el brazo AMBIGUOUS consultaba la orden por REST y sólo
/// registraba el resultado. Una consulta concluyente sin ejecución
/// (EXPIRED/CANCELED/REJECTED con 0) dejaba una posición que sólo existía en
/// el arena (margen retenido, comisión cobrada, ranura ocupada); una
/// consulta con ejecución dejaba la posición real sin confirmar.
#[test]
fn cl46_la_consulta_de_una_entrada_ambigua_decide_la_reserva() {
    el_host_usa_la_resolucion(&host());
    let o = oms();
    let b = tramo(&o, RESOLUCION, "Err(e)ife.starts_with(\"AMBIGUOUS\")", "Err(_)=>(DestinoReserva::Revertir,false),");
    assert!(b.contains("resolve_terminal_entry_until("), "la entrada ambigua se consulta");
    let d = tramo(&o, RESOLUCION, "matchdestino{", "fnnow_ms(");
    assert!(d.contains("DestinoReserva::Revertir"), "sin ejecución se revierte");
    assert!(d.contains("DestinoReserva::Confirmar"), "con ejecución se confirma");
    assert!(d.contains("revertir_entrada(arena,reservation)"), "sin ejecución se revierte");
    assert!(d.contains("confirmar_llenado_con_marca("), "con ejecución se confirma");
}

/// CL-45: antes el brazo Ok confirmaba la reserva con la cantidad, el margen
/// y la comisión de la orden completa aunque la IOC ejecutara sólo una parte.
#[test]
fn cl45_el_host_confirma_la_reserva_con_lo_ejecutado() {
    let h = host();
    el_host_usa_la_resolucion(&h);
    let o = oms();
    let b = tramo(&o, RESOLUCION, "Ok(())=>{", "matchdestino{");
    assert!(b.contains("entry_execution_for_host(executor.registry().get(client_id).as_ref(),"), "la confirmación lleva lo ejecutado");
    let d = tramo(&o, RESOLUCION, "matchdestino{", "fnnow_ms(");
    assert!(d.contains("confirmar_llenado_con_marca("), "la confirmación lleva lo ejecutado");
    assert!(!d.contains("reservation.confirm(") && !h.contains("reservation.confirm(&arena_clone)"), "una sola vía de confirmación");
}

/// CL-45b: el bracket y el diario de la entrada usan lo LLENADO. Antes el
/// bracket partía de la cantidad pedida y sólo se recortaba a la posición del
/// lado en el exchange, que en hedge suma todas las ranuras del lado: un
/// parcial apilado sobre otra ranura ponía TP/SL de esta ranura sobre la
/// ajena. El diario anotaba la cantidad pedida y la τ de la primera ranura
/// abierta del lado.
#[test]
fn cl45b_bracket_y_diario_usan_lo_llenado() {
    let h = host();
    let b = tramo(&h, "letdesenlace=quantum_engine::host_oms::resolver_resultado_de_entrada(", "DesenlaceDeEntrada::Confirmada{ejecutada,envio_ambiguo:false}=>ejecutada,", "if!oco_success{");
    assert!(b.contains("letcantidad_llenada=ejecutada.min(final_qty.abs());"));
    assert!(b.contains("letqty_intent=cantidad_llenada;"));
    assert!(!b.contains("letqty_intent=final_qty.abs();"));
    assert!(b.contains("anotar_diario_de_posicion(&arena_clone,&reservation,&parsed_sym_str,final_is_long,cantidad_llenada,"));
}

/// CL-46b: el brazo AMBIGUOUS que confirma también anota el diario B2.7.
/// Antes sólo lo hacía el brazo Ok: tras un reinicio, la posición confirmada
/// por la consulta se re-protegía con la τ de una entrada anterior del mismo
/// símbolo y lado (o con la τ de respaldo).
#[test]
fn cl46b_la_confirmacion_tras_consulta_anota_el_diario() {
    let h = host();
    let b = tramo(&h, "letdesenlace=quantum_engine::host_oms::resolver_resultado_de_entrada(", "DesenlaceDeEntrada::Confirmada{ejecutada,envio_ambiguo:true}=>{", "DesenlaceDeEntrada::Revertida=>{");
    assert!(b.contains("anotar_diario_de_posicion(&arena_clone,&reservation,&parsed_sym_str,final_is_long,ejecutada.min(final_qty.abs()),"));
    // La confirmación de ese caso (envío ambiguo resuelto) es la de la resolución, con la marca.
    assert!(oms().contains("Ok(())=>DesenlaceDeEntrada::Confirmada{ejecutada,envio_ambiguo},"));
}