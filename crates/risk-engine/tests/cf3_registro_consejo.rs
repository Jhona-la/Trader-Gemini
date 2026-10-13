//! C-F3 / F3.4-07 (Línea C, entradas del registro de vetos): los vetos de
//! asiento del consejo no tienen entrada en el registro.
//!
//! Reformulación (hallazgo disputado por los dos escépticos). El borrador
//! proponía `cf3_v_logic_007_pinea_el_consejo`: exigir que el test pineado
//! por V-LOGIC-007 («confianza del consejo») ejecute `deliberar`. Esa verdad
//! es errónea y se DESCARTA: la causa de V-LOGIC-007 es «fusión
//! espectral/council por debajo del umbral del gen» y sus datos «council_fused,
//! thresholds del genoma activo», es decir, REJ_CONFIDENCE del risk-engine, y
//! su test (`council_confidence_threshold_respects_graceful_cold_modulation`)
//! comprueba exactamente ese rechazo. El nombre es engañoso, pero no
//! certifica en falso a `deliberar` (risk-engine ni siquiera depende de
//! metacortex-engine). Forzarla empujaría a reescribir un test coherente.
//!
//! Lo que sí es un defecto, y lo que esta prueba exige (corrección de
//! completitud del hallazgo): los vetos de asiento de
//! `ConsejoDeliberacion::deliberar_traced` no tienen entrada en el registro.
//! Son seis, uno por asiento con `is_veto` (consejo_seniors.rs):
//! - EnteMercado: cascada de liquidaciones > cascade_severity_breaker
//!   (cortacircuitos irrevocable);
//! - Causal: do_calculus_risk (VPIN) > umbral causal sin ruptura alineada;
//! - Riesgo: drawdown > máximo del asiento (irrevocable);
//! - Ejecucion: deslizamiento efectivo > máximo;
//! - Teleonomia: utilidad esperada bajo el suelo con wr bajo;
//! - AuditorInterno: do_calculus_risk > 0,92.
//! La cabecera del registro lo declara como deuda («Las puertas internas del
//! consejo/ramas se añaden por ola»): esta prueba la deja fijada como OPEN y
//! se activa cuando cada asiento tenga su entrada.
//!
//! Criterio de cobertura: una entrada cubre el veto de un asiento si sus
//! campos de texto (id, nombre, causa, datos) contienen «consejo» y el nombre
//! de la variante de `SeniorRole` del asiento. risk-engine no depende de
//! metacortex-engine, así que los nombres se fijan aquí como texto.
//!
//! Convención: una prueba que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use risk_engine::veto_registry::{buscar, REGISTRO_VETOS};

/// Variantes de `SeniorRole` que pueden vetar en el consejo.
const ASIENTOS_CON_VETO: [&str; 6] = [
    "EnteMercado",
    "Causal",
    "Riesgo",
    "Ejecucion",
    "Teleonomia",
    "AuditorInterno",
];

/// Lectura corregida de V-LOGIC-007: su entrada describe REJ_CONFIDENCE del
/// risk-engine y su test pineado comprueba ese rechazo. Pasa hoy; fija que la
/// premisa del borrador (un certificado falso de `deliberar`) no se sostiene.
#[test]
fn cf3_v_logic_007_describe_rej_confidence_del_riesgo() {
    let e = buscar("V-LOGIC-007").expect("V-LOGIC-007 existe");
    let test = e.test.expect("V-LOGIC-007 pinea un test");
    let src = include_str!("veto_logic_contracts.rs");
    let ini = src
        .find(&format!("fn {test}("))
        .expect("el test pineado está en veto_logic_contracts.rs");
    let resto = &src[ini..];
    let fin = resto.find("#[test]").unwrap_or(resto.len());
    let cuerpo = &resto[..fin];
    println!(
        "observado: V-LOGIC-007 «{}», causa «{}»; el test pineado contiene REJ_CONFIDENCE: {}; contiene deliberar: {}",
        e.nombre,
        e.causa,
        cuerpo.contains("REJ_CONFIDENCE"),
        cuerpo.contains("deliberar")
    );
    assert!(
        e.causa.contains("umbral del gen"),
        "la causa de V-LOGIC-007 debe ser el umbral de confianza del gen: «{}»",
        e.causa
    );
    assert!(
        cuerpo.contains("REJ_CONFIDENCE"),
        "el test pineado por V-LOGIC-007 debe comprobar la razón REJ_CONFIDENCE"
    );
}

/// Esperado: cada uno de los seis asientos con veto tiene al menos una
/// entrada del registro que lo nombra junto a «consejo». Cerrado en Ola Ω93 (F3.4-07).
#[test]
fn cf3_registro_cubre_los_vetos_de_asiento_del_consejo() {
    let texto = |e: &risk_engine::veto_registry::EntradaVeto| {
        format!("{} {} {} {}", e.id, e.nombre, e.causa, e.datos)
    };
    let faltan: Vec<&str> = ASIENTOS_CON_VETO
        .iter()
        .copied()
        .filter(|asiento| {
            !REGISTRO_VETOS.iter().any(|e| {
                let t = texto(e);
                t.contains("consejo") && t.contains(asiento)
            })
        })
        .collect();
    println!(
        "observado: {} entradas en el registro; asientos sin entrada: {:?}",
        REGISTRO_VETOS.len(),
        faltan
    );
    assert!(
        faltan.is_empty(),
        "vetos de asiento del consejo sin entrada en el registro: {faltan:?}"
    );
}
