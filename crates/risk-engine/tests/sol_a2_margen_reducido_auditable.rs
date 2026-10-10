//! SOL-A2 — el recorte de margen era SILENCIOSO.
//!
//! Hallazgo: en `evaluate_quantum_order`, cuando el margen ya dimensionado
//! excede el techo de seguridad, el código lo reducía (`final_margin =
//! safe_limit`) y, si hacía falta, SUBÍA el apalancamiento para seguir
//! alcanzando el nocional mínimo — sin contador, sin causa y sin rastro por
//! moneda. El consejo no podía responder a preguntas básicas: ¿cuántas
//! órdenes se redimensionan?, ¿en qué monedas?, ¿por qué factor?
//!
//! Este contrato NO cambia la matemática del recorte (eso exige medir el
//! impacto en OOS y decidir con el consejo). Exige lo mínimo defendible:
//! que el recorte sea AUDITABLE y POR MONEDA.

use std::sync::Arc;

/// El recorte de margen debe escribir telemetría por moneda: contador de
/// eventos y cociente de recorte (margen impuesto / margen pedido).
#[test]
fn el_recorte_de_margen_queda_registrado_por_moneda() {
    let src: String = include_str!("../src/lib.rs").split_whitespace().collect();

    // 1) El recorte debe incrementar un contador POR MONEDA (no un println,
    //    no un contador global sin símbolo: en multiactivo eso es opaco).
    assert!(
        src.contains("set_for_coin(coin_id,\"sol_a2_margen_reducido\""),
        "el recorte de margen debe contarse por moneda"
    );

    // 2) Debe conservar el factor de recorte: cuánto del tamaño pedido
    //    sobrevivió. Sin esto, "se recortó" no dice nada cuantitativo.
    assert!(
        src.contains("sol_a2_margen_reducido_cociente"),
        "el recorte debe publicar el cociente impuesto/pedido"
    );

    // 3) Y debe medirse ANTES de mutar el margen (un cociente calculado
    //    después de la asignación sería siempre 1.0 — métrica decorativa).
    let pos_cociente = src
        .find("let cociente_recorte=")
        .or_else(|| src.find("letcociente_recorte="))
        .expect("debe existir el cálculo del cociente");
    let pos_mutacion = src
        .find("final_margin=safe_limit;")
        .expect("debe existir la mutación del margen");
    assert!(
        pos_cociente < pos_mutacion,
        "el cociente debe calcularse ANTES de reducir el margen"
    );
}

/// La telemetría del recorte es por moneda: dos monedas distintas no deben
/// compartir contador (un pool global haría imposible atribuir el síntoma en
/// un universo multiactivo).
#[test]
fn la_telemetria_del_recorte_no_es_global() {
    let src: String = include_str!("../src/lib.rs").split_whitespace().collect();
    // Si se hubiera usado `set` global, no habría coin_id en la llamada.
    assert!(
        src.contains("get_for_coin_or(coin_id,\"sol_a2_margen_reducido\""),
        "el contador debe leerse por moneda"
    );
    assert!(
        !src.contains("registry.set(\"sol_a2_margen_reducido\""),
        "el contador NO debe ser un registro global"
    );
}

/// Recordatorio ejecutable del defecto de fondo (todavía abierto): el recorte
/// puede ir acompañado de un RESCATE de apalancamiento para alcanzar el
/// nocional mínimo. Eso no es gratis: más apalancamiento no mejora la
/// viabilidad (D-750) y sí aumenta el coste de fricción. Queda como deuda
/// medida por este mismo contrato.
#[test]
fn el_recorte_puede_rescatar_apalancamiento_y_eso_sigue_siendo_deuda() {
    let src: String = include_str!("../src/lib.rs").split_whitespace().collect();
    let recorta = src.contains("final_margin=safe_limit;");
    let rescata = src.contains("letneeded_leverage=");
    assert!(
        recorta && rescata,
        "se espera el recorte con el rescate de apalancamiento documentado"
    );
    // El invariante terminal posterior (rej por nocional mínimo) sigue
    // siendo la red de seguridad; el contrato lo deja constancia.
    assert!(
        src.contains("returnrej(6);") || src.contains("returnrej(REJ_MIN_NOTIONAL);"),
        "debe existir el rechazo terminal por nocional mínimo"
    );
}

/// Sanidad: el módulo de régimen de capital expone la función de viabilidad
/// que el recorte NO debe sustituir (subir apalancamiento no hace viable una
/// orden que no cabe en el tope de riesgo — D-750).
#[test]
fn la_viabilidad_no_se_resuelve_con_apalancamiento() {
    // 100 $ de mínimo y stop del 5% en cuenta de 13 $: 38% del capital.
    assert!(!risk_engine::capital_regime::orden_viable(100.0, 0.05, 13.0, 0.25));
    // El mismo símbolo con stop del 0,5% cabe.
    assert!(risk_engine::capital_regime::orden_viable(100.0, 0.005, 13.0, 0.25));
}

#[allow(dead_code)]
fn _arena(_: Arc<()>) {}
