//! CL-44 (FMT-232, ADR-0015): el kill-switch del host bloquea lo que AUMENTA
//! el riesgo, nunca la defensa de una posición viva.
//!
//! El binario `god_engine` no es testeable por unidad, así que el contrato se
//! fija sobre su fuente (mismo estilo que `genoma_fijado_contract.rs`): se
//! quitan los comentarios de línea completa y se compacta el espacio.

fn host() -> String {
    include_str!("../../../src/bin/god_engine.rs")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

fn tramo<'a>(h: &'a str, desde: &str, hasta: &str) -> &'a str {
    let i = h.find(desde).unwrap_or_else(|| panic!("ancla «{desde}» ausente"));
    let f = i + h[i..]
        .find(hasta)
        .unwrap_or_else(|| panic!("ancla «{hasta}» ausente tras «{desde}»"));
    &h[i..f]
}

/// Antes el vigilante hacía `if executor.is_kill_switch_active() {
/// clear_dirty(); continue; }` antes de auditar: con el latch armado ninguna
/// posición desnuda se re-protegía ni escalaba a cierre, aunque las piernas
/// protectoras y el reduce-only están exentos del kill-switch (CL-3, CL-20).
#[test]
fn cl44_el_vigilante_de_proteccion_corre_bajo_el_kill_switch() {
    let h = host();
    let t = tramo(
        &h,
        "[PROTECTION-WATCHDOG]Vigilantedeposicióndesnudaactivo",
        "ensure_position_protected(",
    );
    assert!(
        !t.contains("is_kill_switch_active()"),
        "el kill-switch no apaga la re-protección de posiciones vivas"
    );
}

/// X-009: tres OCO fallidos y dos cierres de emergencia fallidos dejan la
/// posición viva y sin brackets. Antes sólo se armaban los latches; ahora,
/// además, se despierta al vigilante para que re-proteja o escale el cierre.
#[test]
fn cl44_x009_arma_el_latch_y_despierta_la_re_proteccion() {
    let h = host();
    let t = tramo(&h, "[X-009ESCALADA]", "msg_count+=1;");
    assert!(t.contains("kill_switch_active.store(true"));
    assert!(
        t.contains("protection_health::mark_dirty()"),
        "X-009 arma el latch y además despierta la re-protección"
    );
}

/// Si el aplanado del sistema inmune falla, la posición queda viva con el
/// latch armado: el vigilante debe auditarla ya, no a los 60 s.
/// CL-44b: también cuando devuelve Ok. `flatten_all_positions` sólo devuelve
/// Err si falla la lectura inicial (nada cancelado); si un cierre falla tras
/// purgar las piernas de su símbolo devuelve Ok con la posición desnuda.
/// Antes el aviso sólo estaba en el brazo Err.
#[test]
fn cl44_un_aplanado_fallido_del_sistema_inmune_despierta_al_vigilante() {
    let h = host();
    let t = tramo(&h, "[SISTEMAINMUNE]ACTIVADO", "latched=true;");
    assert!(t.contains("flatten_all_positions()"));
    assert!(
        t.ends_with("}quantum_arena::protection_health::mark_dirty();"),
        "el vigilante se despierta tras el aplanado, en los dos brazos del resultado"
    );
}

/// CL-44b: el aplanado cierra en el exchange sin tocar las ranuras, que el
/// núcleo sigue gestionando bajo el latch (CL-43). Su confirmación se consume
/// ANTES de aplanar, para que un cierre local posterior sea papel.
#[test]
fn cl44b_el_aplanado_inmune_consume_las_confirmaciones_antes_de_aplanar() {
    let h = host();
    let t = tramo(&h, "[SISTEMAINMUNE]ACTIVADO", "latched=true;");
    let consumo = t
        .find("consume_exchange_confirmations()")
        .expect("el aplanado inmune consume las confirmaciones de las ranuras");
    let aplanado = t.find("flatten_all_positions()").unwrap();
    assert!(consumo < aplanado, "se consumen antes de aplanar, no después");
}

/// CL-44b: el aplanado (inmune y de apagado) se anuncia y espera a la
/// auditoría en curso antes de cancelar piernas; el vigilante no empieza una
/// auditoría con un aplanado anunciado. Antes las cancelaciones del aplanado
/// despertaban al vigilante, que re-armaba piernas sobre una posición que se
/// cerraba a continuación.
#[test]
fn cl44b_los_aplanados_y_el_vigilante_no_se_solapan() {
    let h = host();
    let inmune = tramo(&h, "[SISTEMAINMUNE]ACTIVADO", "latched=true;");
    let apagado = tramo(&h, "UnifiedEventLoopsafelyterminated", "std::process::exit(0)");
    for (nombre, t) in [("inmune", inmune), ("apagado", apagado)] {
        let anuncio = t
            .find("protection_health::begin_flatten()")
            .unwrap_or_else(|| panic!("el aplanado {nombre} se anuncia"));
        let espera = t
            .find("esperar_auditoria_en_curso()")
            .unwrap_or_else(|| panic!("el aplanado {nombre} espera la auditoría en curso"));
        let aplanado = t.find("flatten_all_positions()").unwrap();
        assert!(anuncio < espera && espera < aplanado, "orden: anunciar, esperar, aplanar ({nombre})");
    }
    let vigilante = tramo(
        &h,
        "[PROTECTION-WATCHDOG]Vigilantedeposicióndesnudaactivo",
        "fetch_position_risk()",
    );
    assert!(
        vigilante.contains("protection_health::try_begin_audit()"),
        "el vigilante no audita durante un aplanado"
    );
}

/// CL-44b: tras el aplanado de apagado no queda auditoría que purgue las
/// piernas sin posición: el apagado las purga antes de salir.
#[test]
fn cl44b_el_apagado_purga_las_piernas_huerfanas_antes_de_salir() {
    let h = host();
    let apagado = tramo(&h, "UnifiedEventLoopsafelyterminated", "std::process::exit(0)");
    let aplanado = apagado.find("flatten_all_positions()").unwrap();
    let purga = apagado
        .find("purge_orphan_legs(")
        .expect("el apagado purga las piernas huérfanas");
    assert!(aplanado < purga, "la purga va después del aplanado");
}
