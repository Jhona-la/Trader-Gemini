//! I-9 (Línea I, fila N15b del censo C-F7b): una purga de brackets que falla
//! tras un cierre queda pendiente, bloquea las entradas en el símbolo y el
//! vigilante la reintenta.
//!
//! Antes el host sólo reportaba el fallo (N15, CL-176): la pierna seguía viva,
//! una reentrada en el símbolo abría con ella armada encima y el vigilante no
//! la purgaba mientras hubiera posición en el símbolo o el lado. El registro y
//! su reintento los prueban las pruebas de
//! `execution_engine::purga_pendiente`; aquí, su uso en el host. Guarda
//! estática sobre la fuente del host; sin exchange ni red.

use execution_engine::purga_pendiente::{PurgasPendientes, PURGAS_PENDIENTES};

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
const ANOTA: &str = "anotar_purga_pendiente(";

/// El texto entre `desde` y la primera aparición de `hasta` tras él.
fn tramo(desde: &str, hasta: &str) -> &'static str {
    let Some(i) = HOST.find(desde) else {
        return "";
    };
    let Some(f) = HOST[i..].find(hasta) else {
        return "";
    };
    &HOST[i..i + f]
}

#[test]
fn i9_el_registro_publico_bloquea_hasta_resolver() {
    let p = PurgasPendientes::new();
    p.anotar("BTCUSDT", true);
    assert!(p.bloquea_entrada("BTCUSDT"));
    p.resolver("BTCUSDT", true);
    assert!(!p.bloquea_entrada("BTCUSDT"));
    // El global del proceso arranca vacío.
    assert!(!PURGAS_PENDIENTES.bloquea_entrada("I9_NUNCA_ANOTADO"));
}

/// RED antes de I-9: el helper no existía.
#[test]
fn i9_anotar_registra_y_despierta_al_vigilante() {
    let cuerpo = tramo("fn anotar_purga_pendiente(", "\n}\n");
    assert!(
        !cuerpo.is_empty(),
        "helper anotar_purga_pendiente en el host"
    );
    assert!(cuerpo.contains("PURGAS_PENDIENTES.anotar(symbol, is_long)"));
    assert!(
        cuerpo.contains("protection_health::mark_dirty()"),
        "despierta al vigilante para reintentar en la próxima auditoría"
    );
}

/// RED antes de I-9: cada purga que falla tras un cierre sólo se reportaba.
#[test]
fn i9_toda_purga_fallida_tras_un_cierre_queda_pendiente() {
    for (nombre, desde, hasta, llamada) in [
        (
            "PURGE OCO (cierre del núcleo)",
            "[PURGE OCO] {} cerrada pero brackets sin purgar",
            "} else {",
            "anotar_purga_pendiente(&parsed_sym_str, is_long_close)",
        ),
        (
            "EMERGENCY (X-009)",
            "[EMERGENCY] Purga de órdenes de {} falló",
            "match close_res {",
            "anotar_purga_pendiente(&parsed_sym_str, is_long_close)",
        ),
        (
            "NAKED-ESCALATION",
            "[NAKED-ESCALATION] {} cerrada pero brackets sin purgar",
            "naked_streak.remove(",
            "anotar_purga_pendiente(&close_sym, close_long)",
        ),
    ] {
        let t = tramo(desde, hasta);
        assert!(!t.is_empty(), "{nombre}: rama de error ausente");
        assert!(
            t.contains(llamada),
            "{nombre}: la purga fallida queda pendiente"
        );
    }
    // La escalada que encuentra la posición ya cerrada no purga: sus piernas
    // quedan pendientes para el vigilante.
    let ya_cerrada = tramo("// La posición ya no existe", "} else {");
    assert!(
        ya_cerrada.contains("anotar_purga_pendiente(&close_sym, close_long)"),
        "la escalada sin posición deja la purga pendiente"
    );
    assert_eq!(
        HOST.matches(ANOTA).count(),
        5,
        "una definición y cuatro llamadas"
    );
}

/// RED antes de I-9: el vigilante no reintentaba nada.
#[test]
fn i9_el_vigilante_reintenta_antes_de_la_purga_de_huerfanas() {
    let vigilante = tramo(
        "let positions = match executor.fetch_position_risk().await {",
        "ensure_position_protected(",
    );
    let reintento = vigilante
        .find("reintentar_purgas_pendientes(&executor, &positions)")
        .unwrap_or(usize::MAX);
    let huerfanas = vigilante
        .find("purge_orphan_legs(&executor, &piernas, &positions)")
        .unwrap_or(usize::MAX);
    assert!(
        reintento < huerfanas && huerfanas != usize::MAX,
        "el reintento corre con la foto de la auditoría, antes de la purga de huérfanas"
    );
    let cuerpo = tramo("async fn reintentar_purgas_pendientes(", "\n}\n");
    assert!(cuerpo.contains("PURGAS_PENDIENTES.reintentar(executor, positions)"));
}

/// RED antes de I-9: nada bloqueaba la reentrada en el símbolo.
#[test]
fn i9_la_entrada_espera_a_la_purga_pendiente() {
    let envio = tramo(
        "let decision_envio: Option<u32> = 'envio: {",
        "Some(effective_leverage)",
    );
    let Some(i) = envio.find(".bloquea_entrada(&parsed_sym_str)") else {
        panic!("la decisión de envío no consulta las purgas pendientes");
    };
    let rama = &envio[i..];
    let rama = &rama[..rama.find("break 'envio None;").unwrap_or(0)];
    assert!(
        rama.contains("rollback_positions(&engine_real.arena)"),
        "la entrada vetada revierte su reserva"
    );
    let ajuste = envio.find("apalancamiento_de_envio(").unwrap_or(0);
    assert!(i < ajuste, "el veto va antes de decidir el apalancamiento");
}
