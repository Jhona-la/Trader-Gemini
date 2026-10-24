//! C-F3.3 (Línea C): contratos de fuente sobre la ruta de la confianza.
//!
//! Las ramas viven dentro de `process_event` (privado) y no hay forma de
//! aislarlas sin un fixture que las dispare una a una. Estas pruebas leen el
//! código de PRODUCCIÓN de `src/lib.rs` (lo que va antes del primer
//! `#[cfg(test)]`, para que los módulos de test del propio archivo no las
//! satisfagan en falso) sin espacios, y comprueban una propiedad estructural:
//!
//! - D-752: toda rama que registra cierres en su `TasaAcierto` debe leer ese
//!   historial (`conviccion_de_rama`). La rama 12 (impulso de flujo) se
//!   registra y nunca se lee (F3.3-09).
//! - #624/H2 (F3.3-03): la posición vive a la τ a la que habló el voto.
//!   Cerrado por CL-85. CL-94 retira de aquí su contrato de fuente, que
//!   exigía el literal `tensor_cont.expected_lifetime_ms`: CL-85 lee la misma
//!   escala con `GodEngineCore::escala_que_hablo` y lo rompía por la forma.
//!   Lo sustituyen dos pruebas de propiedad en `cf3_una_sola_tau.rs`:
//!   `cf3_la_rama_11_vive_a_la_escala_que_hablo` (la orden de la rama 11
//!   viaja a τ_k) y `cl85_la_escala_que_hablo_es_la_del_orquestador` (τ_k es
//!   la `expected_lifetime_ms` del orquestador, ±1 ms).
//! - La rama 13 no debe contar dos veces la misma lectura del ML: su piso usa
//!   `normalized_directional_divergence(swing_nn_pred, …)` y después la puerta
//!   ML de `puertas_del_continuo` vuelve a multiplicar por (1 + 0,5·d) con el
//!   mismo ensamble (F3.3-07).
//! - F3.3-10: el núcleo recortaba `free_cap` en línea con una copia de
//!   `SpectralRegimeField::margin_multiplier`. Lo cerró F1-054 (AGY,
//!   508f1597) quitando el recorte: el margen admisible lo decide
//!   `PortfolioOrchestrator::available_margin_headroom` (risk-engine). T12
//!   queda como guarda de regresión.
//!
//! Son contratos de fuente: frágiles ante un refactor que cambie los nombres.
//! Si un arreglo legítimo los rompe por la forma y no por el fondo, se
//! sustituyen por una prueba de propiedad sobre la función extraída (lo dice
//! cada test). Convención: `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el
//! assert intacto mientras el hallazgo siga abierto.

/// Código de producción de lib.rs, sin espacios ni saltos de línea.
fn codigo_produccion() -> String {
    include_str!("../src/lib.rs")
        .split("#[cfg(test)]")
        .next()
        .expect("lib.rs vacío")
        .split_whitespace()
        .collect()
}

/// Trozo de `codigo` entre la primera aparición de `desde` y la siguiente de
/// `hasta` (incluida).
fn tramo<'a>(codigo: &'a str, desde: &str, hasta: &str) -> Option<&'a str> {
    let i = codigo.find(desde)?;
    let j = codigo[i..].find(hasta)?;
    Some(&codigo[i..i + j + hasta.len()])
}

/// T9 (F3.3-09). La rama 12 (FlowImpulse) etiqueta sus intenciones con
/// `volume_flow_rate = 12.0`, y el cierre registra el resultado en
/// `rama_registro[coin][12]`. Su confianza debe pasar por
/// `conviccion_de_rama`: o bien aparece `registro_ramas[12]`, o el tramo del
/// impulso (desde la llamada a `evaluate_flow_impulse` hasta
/// `fast_intent = impulso_intent;`) llama a `conviccion_de_rama`.
#[test]
fn cf3_rama_12_lee_su_historial() {
    let src = codigo_produccion();
    let bloque = tramo(
        &src,
        "evaluate_flow_impulse(",
        "fast_intent=impulso_intent;",
    )
    .expect("premisa: no encuentro el tramo de la rama 12; actualizar el test");
    assert!(
        bloque.contains("volume_flow_rate=12.0"),
        "premisa: el tramo encontrado no es el de la rama 12"
    );
    let lee = src.contains("registro_ramas[12]") || bloque.contains("conviccion_de_rama(");
    assert!(
        lee,
        "la rama 12 registra cierres pero su confianza no pasa por conviccion_de_rama"
    );
}

/// T11 (F3.3-07). Hoy el piso de la rama 13 llama dos veces (largo y corto)
/// a `normalized_directional_divergence(swing_nn_pred, ml_model_base)`, y la
/// puerta ML de `puertas_del_continuo` vuelve a aplicar el mismo ensamble.
/// Esperado: 0 usos en el piso.
///
/// Si el arreglo elige, en cambio, saltarse el impulso de la puerta para la
/// rama 13, esta prueba se sustituye por una de propiedad sobre la función
/// extraída.
#[test]
fn cf3_rama_13_cuenta_el_ml_una_sola_vez() {
    let src = codigo_produccion();
    let n = src
        .matches("normalized_directional_divergence(swing_nn_pred")
        .count();
    println!(
        "observado: {n} usos de normalized_directional_divergence(swing_nn_pred en producción"
    );
    assert_eq!(
        n, 0,
        "el piso de la rama 13 y la puerta ML leen el mismo ML: {n} usos en el piso"
    );
}

/// T12 (F3.3-10, cerrado por F1-054 de AGY, 508f1597). El núcleo recortaba
/// `free_cap` en línea con `(1 − 0,95·flux).clamp(0,05; 1)` bajo marea
/// adversa: una copia de `SpectralRegimeField::margin_multiplier` que su
/// test de espejo no ejecutaba. F1-054 quitó el recorte; el margen admisible
/// lo decide `PortfolioOrchestrator::available_margin_headroom` (risk-engine)
/// y el núcleo sólo descuenta el margen ya comprometido. Guarda de
/// regresión: el núcleo no vuelve a recortar `free_cap` por su cuenta. Entre
/// el cálculo de `free_cap` y su puerta (`margin_req <= free_cap`), `free_cap`
/// se asigna una sola vez y no aparece `margin_multiplier`: así fallan el
/// `*=` antiguo, un sombreado `let free_cap = free_cap * …`, la misma copia
/// con otra variable y un recorte en la propia puerta (que rompe la premisa).
/// Si un rediseño legítimo necesita aquí un recorte espectral, debe llamar a
/// la función con contrato de quantum-arena y este test se sustituye por una
/// prueba de propiedad sobre ella.
#[test]
fn cf3_colchon_sin_copia_en_linea() {
    let src = codigo_produccion();
    assert!(
        !src.contains("free_cap*="),
        "el núcleo vuelve a recortar free_cap en línea; el margen admisible lo decide PortfolioOrchestrator::available_margin_headroom"
    );
    assert!(
        !src.contains("0.95*directional_flux"),
        "reaparece en el núcleo la copia en línea de SpectralRegimeField::margin_multiplier"
    );
    assert!(
        src.contains("letfree_cap=(current_cap-total_used).max(0.0);"),
        "premisa: no encuentro el cálculo de free_cap del núcleo; actualizar el test"
    );
    let bloque = tramo(
        &src,
        "letfree_cap=(current_cap-total_used).max(0.0);",
        "margin_req<=free_cap{",
    );
    assert!(
        bloque.is_some(),
        "premisa: no encuentro el tramo de free_cap a su puerta (margin_req <= free_cap); actualizar el test"
    );
    let bloque = bloque.unwrap_or_default();
    assert_eq!(
        bloque.matches("free_cap=").count(),
        1,
        "el núcleo reasigna free_cap entre su cálculo y la puerta de margen"
    );
    assert!(
        !bloque.contains("margin_multiplier"),
        "el núcleo recorta free_cap con un multiplicador espectral antes de la puerta"
    );
}
