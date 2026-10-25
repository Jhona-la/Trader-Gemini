//! N15 (censo C-F7b; CL-176): ninguna purga de brackets del host descarta su
//! error, y el cierre por escalada desnuda sólo dice «brackets purgados» si
//! la purga respondió Ok.
//!
//! Antes la escalada desnuda hacía `let _ = exec_esc.cancel_position_oco_orders(..)`
//! y logueaba «brackets purgados» pasara lo que pasara. Los otros dos cierres
//! del host (PURGE OCO y EMERGENCY) ya reportaban el fallo, como pide D-698:
//! una pierna que sobrevive a la purga dispara sobre la posición siguiente.
//! Guarda estática sobre la fuente del host; sin exchange ni red.

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
const LLAMADA: &str = ".cancel_position_oco_orders(";
const EXITO: &str = "[NAKED-ESCALATION] {} cerrada por emergencia — brackets purgados";

/// Texto de la sentencia que contiene la llamada: desde el último `;`, `{` o
/// `}` anterior hasta la llamada.
fn sentencias_con_la_llamada() -> Vec<&'static str> {
    HOST.match_indices(LLAMADA)
        .map(|(i, _)| {
            let inicio = HOST[..i].rfind([';', '{', '}']).map_or(0, |j| j + 1);
            &HOST[inicio..i]
        })
        .collect()
}

#[test]
fn n15_ninguna_purga_del_host_descarta_su_error() {
    let sentencias = sentencias_con_la_llamada();
    assert_eq!(
        sentencias.len(),
        3,
        "escalada desnuda, PURGE OCO y EMERGENCY: {sentencias:?}"
    );
    for s in sentencias {
        assert!(!s.contains("let _ ="), "la purga descarta su error: {s:?}");
    }
}

#[test]
fn n15_la_escalada_solo_dice_purgados_si_la_purga_respondio_ok() {
    assert_eq!(
        HOST.matches(EXITO).count(),
        1,
        "un solo log de éxito de la purga tras la escalada"
    );
    let tras_ok = HOST.split("if purga.is_ok() {").nth(1).unwrap_or("");
    assert!(
        tras_ok
            .split("} else {")
            .next()
            .is_some_and(|rama| rama.contains(EXITO)),
        "«brackets purgados» sólo en la rama Ok de la purga"
    );
    let tras_err = HOST.split("if let Err(e) = &purga {").nth(1).unwrap_or("");
    assert!(
        tras_err
            .split("naked_streak.remove(")
            .next()
            .is_some_and(|rama| rama.contains("telemetry_err!") && rama.contains("sin purgar")),
        "el fallo de la purga va al log de errores"
    );
}
