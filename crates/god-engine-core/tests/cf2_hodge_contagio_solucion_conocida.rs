//! C-F2.1 (Línea C): curl de Hodge sobre el campo de contagio vivo, contra
//! una solución conocida.
//!
//! Teoría: una cascada con orden de liderazgo total (L -> S1 -> S2, retardos
//! 0, 150 y 300 ms) es un DAG; sus retardos son un potencial y cualquier
//! medida de flujo ADITIVA sobre ella es un gradiente, así que HodgeRank da
//! curl = 0. El campo vivo usa los z-scores de `contagion_matrix` como flujo
//! (no aditivos) y censura las direcciones inversas a 0, de modo que el curl
//! de una jerarquía perfecta sale lejos de 0.
//!
//! Es una prueba de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlo: `cargo test ... -- --include-ignored`.

use feature_engine::hawkes_cross::contagion_matrix;
use risk_engine::hodge::hodge_curl_share;

/// Rejilla de desfases viva del publicador (contagion_publisher.rs:52).
const REJILLA_VIVA_MS: [u64; 6] = [200, 500, 1_000, 5_000, 15_000, 30_000];

/// Valor verdadero: curl = 0 (jerarquía perfecta = gradiente). Tolerancia
/// 0,05: la prueba es determinista, así que no es una banda estadística sino
/// un margen numérico por la discretización de la rejilla de desfases; una
/// medida aditiva daría 0 exacto. Hoy da ≈ 0,289 (z L->S1 = z S1->S2 ≈ 42,3,
/// z L->S2 ≈ 24,4 y las inversas censuradas a 0).
///
/// Nota de los escépticos: ninguna medida de efectos DIRECTOS cumple este
/// oráculo (en una cascada Hawkes real n(L->S2) ≈ 0 y HodgeRank sobre
/// (1, 1, 0) da 2/3). El residuo de HodgeRank con orden consistente y
/// magnitud inconsistente es legítimo; lo que falla es leerlo como «cámara de
/// eco». La prueba de aceptación del sucesor debe medir ciclicidad (p. ej. el
/// radio espectral de la parte cruzada de la matriz de ramificación, 0 en
/// todo DAG por nilpotencia).
///
/// Revisión cruzada (CL-65): aun tratando las inversas censuradas como
/// ausentes, el flujo neto (21; 21; 12) da curl ≈ 0,292; sólo un flujo
/// aditivo (p. ej. el retardo mediano con signo) llega a < 0,05. Así que esta
/// prueba no se activa sola con el arreglo: quien lo haga fija aquí el flujo
/// que mide el sucesor o cambia la aserción a su criterio de ciclicidad.
#[test]
#[ignore = "OPEN F2.1-03 (Línea C, hodge.rs; Qoder/GLM, hawkes_cross.rs), CONTRATO DEL SUCESOR, se reescribe con su flujo: el curl usa z-scores no aditivos y cuenta las direcciones censuradas como flujo cero medido"]
fn cf2_hodge_cascada_jerarquica_del_kernel_vivo() {
    let lider: Vec<u64> = (0..200).map(|i| 1_000 + i as u64 * 2_000).collect();
    let s1: Vec<u64> = lider.iter().map(|&t| t + 150).collect();
    let s2: Vec<u64> = lider.iter().map(|&t| t + 300).collect();
    let series = vec![lider, s1, s2];
    let spans: Vec<u64> = series.iter().map(|s| s.last().unwrap() - s[0]).collect();
    let m = contagion_matrix(&series, &spans, &REJILLA_VIVA_MS).expect("matriz de cascada");
    let curl = hodge_curl_share(&m).expect("tres nodos con flujo");
    assert!(
        curl < 0.05,
        "jerarquía perfecta L->S1->S2 con curl = {curl:.4}; z L->S1 = {:.2}, z S1->S2 = {:.2}, \
         z L->S2 = {:.2}, inversas S1->L = {:.2}, S2->S1 = {:.2}, S2->L = {:.2}",
        m[0][1],
        m[1][2],
        m[0][2],
        m[1][0],
        m[2][1],
        m[2][0]
    );
}
