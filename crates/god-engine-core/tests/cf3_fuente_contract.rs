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
//! - #624/H2: la posición vive a la τ a la que habló el voto. El orquestador
//!   devuelve `TensorDecision::expected_lifetime_ms`; la rama 11 mide su
//!   convicción a τ_dom y se dimensiona a `fast_duration_ms` (F3.3-03).
//! - La rama 13 no debe contar dos veces la misma lectura del ML: su piso usa
//!   `normalized_directional_divergence(swing_nn_pred, …)` y después la puerta
//!   ML de `puertas_del_continuo` vuelve a multiplicar por (1 + 0,5·d) con el
//!   mismo ensamble (F3.3-07).
//! - El colchón de `free_cap` debe usar la función con contrato
//!   (`SpectralRegimeField::{long_,short_,}margin_multiplier`), no una copia
//!   en línea que su test de espejo no ejecuta (F3.3-10).
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

/// T10 (F3.3-03). La rama 11 toma la convicción del consenso tensorial a
/// τ_dom (`tensor_cont.net_confidence`) y la duración de la micro τ
/// (`fast_duration_ms`, recortada a [30 s; 30 min]). El contrato #624/H2
/// (orchestrator.rs, `expected_lifetime_ms`) dice que la posición vive a la
/// escala que habló. Se exige que el núcleo consuma
/// `tensor_cont.expected_lifetime_ms`.
///
/// Nota del escéptico de completitud: tras la puerta QO-586 la τ se reescribe
/// dos veces (hurst_duration_modulation y la media geométrica con la τ
/// resonante); arreglar sólo la rama 11 no consigue «una sola τ».
#[test]
#[ignore = "OPEN F3.3-03 (Línea C): la rama 11 mide su convicción a τ_dom y se dimensiona a fast_duration_ms; TensorDecision::expected_lifetime_ms no tiene lector en el núcleo"]
fn cf3_rama_11_vive_a_la_escala_que_hablo() {
    let src = codigo_produccion();
    // Diagnóstico: la expresión de duración del literal de la rama 11.
    let literal = tramo(
        &src,
        "confidence:conviccion_de_rama(&registro_ramas[11]",
        "volume_flow_rate:11.0",
    )
    .unwrap_or("(no encontrado)");
    let duracion = literal
        .split("expected_duration_ms:")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .unwrap_or("(no encontrada)");
    println!("observado: la rama 11 emite expected_duration_ms: {duracion}");
    assert!(
        src.contains("tensor_cont.expected_lifetime_ms"),
        "la rama 11 toma la convicción a τ_dom y la duración de la micro τ ({duracion})"
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

/// T12 (F3.3-10). El colchón de `free_cap` reescribe en línea
/// `(1 − 0,95·flux).clamp(0,05; 1)` con marea adversa, que es la fórmula de
/// `SpectralRegimeField::margin_multiplier` (quantum-arena), sin llamador
/// vivo. `cushion_symmetry_contract.rs` prueba una copia. Se exige que
/// lib.rs llame a la función con contrato.
#[test]
#[ignore = "OPEN F3.3-10 (Línea C): el colchón de free_cap duplica en línea SpectralRegimeField::margin_multiplier y su test de espejo prueba una copia"]
fn cf3_colchon_usa_la_funcion_con_contrato() {
    let src = codigo_produccion();
    let en_linea = src.contains("free_cap*=(1.0-0.95*directional_flux)");
    println!("observado: fórmula del colchón en línea = {en_linea}");
    assert!(
        src.contains("margin_multiplier("),
        "el colchón de free_cap duplica en línea SpectralRegimeField::margin_multiplier; su test de espejo prueba una copia"
    );
}
