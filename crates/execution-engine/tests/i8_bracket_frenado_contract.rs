//! I-8 (Línea I, fila N18 del censo C-F7b): un bracket que sólo rechaza el
//! freno temporal del exchange no escala a X-009.
//!
//! Con un 429/418 activo, los tres intentos de bracket de una entrada
//! llenada, el cierre de emergencia y su reintento se rechazaban en local y
//! la escalada X-009 armaba el kill-switch permanente. El host ahora junta
//! los rechazos de los intentos y, si todos son del freno, difiere al
//! vigilante (CL-44) en vez de cerrar y armar el kill-switch. El
//! comportamiento del ejecutor frenado lo fijan las pruebas de
//! `execution_engine::freno_temporal`; aquí, la decisión pública y su uso en
//! el host.

use execution_engine::freno_temporal::{
    es_freno_temporal, tras_bracket_fallido, TrasBracketFallido,
};

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");

/// El bloque del bracket de entrada: desde `let mut oco_success = false;`
/// hasta el kill-switch que arma la escalada X-009, incluido.
fn bloque_bracket() -> &'static str {
    const FIN: &str = "kill_switch_active.store(true";
    let Some(i) = HOST.find("let mut oco_success = false;") else {
        return "";
    };
    let Some(e) = HOST[i..].find("[X-009 ESCALADA]") else {
        return "";
    };
    let Some(f) = HOST[i + e..].find(FIN) else {
        return "";
    };
    &HOST[i..i + e + f + FIN.len()]
}

#[test]
fn i8_decision_publica_difiere_solo_con_todos_los_intentos_frenados() {
    let cooldown = "RATE_LIMIT_COOLDOWN: activo hasta 2 (ahora 1)";
    let freno = "RATE_BRAKE: freno temporal activo hasta 2 (ahora 1)";
    let genuino = r#"Binance API Error: {"code":-2021,"msg":"Order would immediately trigger."}"#;
    assert!(es_freno_temporal(cooldown) && es_freno_temporal(freno));
    assert!(!es_freno_temporal(genuino));
    assert_eq!(
        tras_bracket_fallido(&[cooldown, cooldown, freno]),
        TrasBracketFallido::DiferirAlVigilante
    );
    assert_eq!(
        tras_bracket_fallido(&[cooldown, genuino, cooldown]),
        TrasBracketFallido::CierreDeEmergencia
    );
    assert_eq!(
        tras_bracket_fallido::<&str>(&[]),
        TrasBracketFallido::CierreDeEmergencia
    );
}

/// RED antes de I-8: el host no juntaba los rechazos ni consultaba la
/// decisión, y todo `!oco_success` iba a X-009.
#[test]
fn i8_el_host_junta_los_rechazos_y_consulta_la_decision_antes_de_x009() {
    let b = bloque_bracket();
    assert!(!b.is_empty(), "bloque del bracket de entrada en el host");

    let reintento = b.find("[OCO RETRY {}/3]").unwrap_or(usize::MAX);
    let anota = b.find("errores_bracket.push(e);").unwrap_or(usize::MAX);
    assert!(
        reintento < anota && anota != usize::MAX,
        "cada intento fallido entra en errores_bracket en su brazo Err"
    );

    let decide = b
        .find("freno_temporal::tras_bracket_fallido(&errores_bracket)")
        .unwrap_or(usize::MAX);
    let emergencia = b.find("[EMERGENCY CLOSE]").unwrap_or(usize::MAX);
    assert!(
        anota < decide && decide < emergencia && emergencia != usize::MAX,
        "la decisión de I-8 va antes del cierre de emergencia de X-009"
    );
}

/// RED antes de I-8: no había rama de diferimiento. La rama que difiere no
/// cierra, no purga ni arma el kill-switch; anota el diario de la posición
/// y deja la protección sucia para el vigilante. X-009 sigue intacto en la
/// otra rama.
#[test]
fn i8_la_rama_diferida_no_cierra_ni_arma_el_kill_switch() {
    let b = bloque_bracket();
    let Some(i) = b.find("TrasBracketFallido::DiferirAlVigilante") else {
        panic!("rama de diferimiento ausente en el host");
    };
    let Some(f) = b[i..].find("} else if !oco_success {") else {
        panic!("X-009 debe seguir como la otra rama (else if !oco_success)");
    };
    let diferida = &b[i..i + f];
    for prohibido in [
        "trigger_kill_switch",
        "kill_switch_active",
        "execute_reduce_only_market",
        "cancel_position_oco_orders",
        "record_emergency_close",
        "rollback_positions",
    ] {
        assert!(
            !diferida.contains(prohibido),
            "la rama diferida no usa {prohibido}"
        );
    }
    assert!(
        diferida.contains("protection_health::mark_dirty()"),
        "deja sucia la protección"
    );
    assert!(
        diferida.contains("anotar_diario_de_posicion("),
        "anota el diario de la posición viva"
    );

    let x009 = &b[i + f..];
    assert!(
        x009.contains("[EMERGENCY CLOSE]"),
        "X-009 conserva su cierre"
    );
    assert!(
        x009.contains("trigger_kill_switch()"),
        "X-009 conserva su escalada"
    );
}
