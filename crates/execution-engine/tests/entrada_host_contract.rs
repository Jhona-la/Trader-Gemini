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

fn brazo<'a>(h: &'a str, desde: &str, hasta: &str) -> &'a str {
    let m = h.find("matchentry_result{").expect("match del resultado de la entrada");
    let i = m + h[m..].find(desde).unwrap_or_else(|| panic!("ancla «{desde}» ausente"));
    let f = i + h[i..]
        .find(hasta)
        .unwrap_or_else(|| panic!("ancla «{hasta}» ausente tras «{desde}»"));
    &h[i..f]
}

/// CL-46: antes el brazo AMBIGUOUS consultaba la orden por REST y sólo
/// registraba el resultado. Una consulta concluyente sin ejecución
/// (EXPIRED/CANCELED/REJECTED con 0) dejaba una posición que sólo existía en
/// el arena (margen retenido, comisión cobrada, ranura ocupada); una
/// consulta con ejecución dejaba la posición real sin confirmar.
#[test]
fn cl46_la_consulta_de_una_entrada_ambigua_decide_la_reserva() {
    let h = host();
    let b = brazo(&h, "Err(e)ife.starts_with(\"AMBIGUOUS\")", "Err(e)=>{");
    assert!(b.contains("resolve_via_rest("));
    assert!(b.contains("destino_tras_consulta("), "la consulta decide");
    assert!(b.contains("rollback_positions("), "sin ejecución se revierte");
    assert!(b.contains("confirmar_llenado("), "con ejecución se confirma");
}

/// CL-45: antes el brazo Ok confirmaba la reserva con la cantidad, el margen
/// y la comisión de la orden completa aunque la IOC ejecutara sólo una parte.
#[test]
fn cl45_el_host_confirma_la_reserva_con_lo_ejecutado() {
    let h = host();
    let b = brazo(&h, "Ok(())=>{", "iforder_tp_price>0.0&&order_sl_price>0.0{");
    assert!(b.contains("confirmar_llenado("), "la confirmación lleva lo ejecutado");
    assert!(
        !b.contains("reservation.confirm(&arena_clone)"),
        "una sola vía de confirmación"
    );
}
