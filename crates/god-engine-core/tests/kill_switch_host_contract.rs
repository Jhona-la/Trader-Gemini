//! CL-44 (FMT-232, ADR-0014): el kill-switch del host bloquea lo que AUMENTA
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
#[test]
fn cl44_un_aplanado_fallido_del_sistema_inmune_despierta_al_vigilante() {
    let h = host();
    let t = tramo(&h, "[SISTEMAINMUNE]ACTIVADO", "latched=true;");
    assert!(t.contains("flatten_all_positions()"));
    assert!(
        t.contains("protection_health::mark_dirty()"),
        "tras un aplanado fallido la posición viva se re-protege"
    );
}
