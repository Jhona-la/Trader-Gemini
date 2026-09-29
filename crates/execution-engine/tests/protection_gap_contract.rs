//! CL-17 — la re-protección del host envía las piernas con gap aunque su
//! nocional quede bajo el mínimo, y el watchdog sólo escala (cierre de
//! emergencia a mercado) cuando falta cobertura de STOP.

fn host_compacto() -> String {
    include_str!("../../../src/bin/god_engine.rs")
        .split_whitespace()
        .collect()
}

/// HOST-024 omitía la pierna bajo minNotional y la contaba como «rechazo
/// determinístico» sin haberla enviado: el watchdog cerraba a mercado toda
/// la posición sin evidencia del exchange.
#[test]
fn cl17_las_piernas_bajo_el_minimo_se_envian_no_se_omiten() {
    let host = host_compacto();
    let cuerpo = host
        .split("asyncfnensure_position_protected(")
        .nth(1)
        .expect("ensure_position_protected");
    let cuerpo = &cuerpo[..cuerpo.find("///B3.6").expect("fin de la función")];
    assert!(!cuerpo.contains("<min_notional"), "no se omite ninguna pierna por nocional");
    assert!(!host.contains("min_notional_skips"), "sin rechazos inventados");
    assert!(cuerpo.contains("\"TAKE_PROFIT_MARKET\""));
    assert!(cuerpo.contains("\"STOP_MARKET\""));
    assert!(cuerpo.matches("note_rejection(symbol,&e)").count() >= 2);
}

/// Un TP ausente con el SL completo no deja la posición desnuda: no puede
/// disparar el cierre de emergencia de TODA la posición.
#[test]
fn cl17_solo_el_gap_de_stop_escala() {
    let host = host_compacto();
    assert!(!host.contains("ifg_tp>0.0||g_sl>0.0{"));
    assert!(host.contains("ifg_sl>0.0{naked_total+=1;"));
    let escalado = host.find("letstreak=naked_streak").expect("streak del watchdog");
    let rama = host[..escalado].rfind("ifg_sl>0.0{").expect("rama del SL");
    assert!(host[rama..escalado].contains("ifrejections_now<=rejections_prev{"));
}
