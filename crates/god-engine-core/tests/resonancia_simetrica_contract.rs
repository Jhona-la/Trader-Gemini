//! CL-31 — la rama 15 (resonancia en τ*) trata la persistencia sin lado.
//! Función pura; sin exchange ni estado.
use god_engine_core::confluencia_resonante;

/// H2-11 (G2-15, RONDA 3) — LOS CINCO CORTES DE LA RESONANCIA, PINNEADOS.
///
/// Eran literales dispersos calibrados a mano (0.38/0.22/0.52/0.12/±2e-4);
/// ahora son constantes públicas nombradas. Este contrato fija los VALORES
/// y las fronteras exactas (justo-adentro/justo-afuera de cada corte) para
/// que cualquier recalibración futura — propia o del genoma — sea un cambio
/// VISIBLE que exige oráculo, no un drift silencioso.
#[test]
fn h2_11_cortes_de_resonancia_pinneados() {
    use god_engine_core::{
        COHERENCIA_MINIMA, FUSED_UMBAL_MODERADO, FUSED_UMBAL_PLENO, HURST_CONTINUACION,
        MAREA_MACRO_TOLERANCIA,
    };
    // (1) Valores exactos.
    assert_eq!(FUSED_UMBAL_PLENO, 0.38);
    assert_eq!(FUSED_UMBAL_MODERADO, 0.22);
    assert_eq!(HURST_CONTINUACION, 0.52);
    assert_eq!(COHERENCIA_MINIMA, 0.12);
    assert_eq!(MAREA_MACRO_TOLERANCIA, 0.00020);
    // (2) Frontera del umbral pleno (browniano, sin continuación):
    // justo-afuera veta, justo-adentro pasa (con coherencia y marea sanas).
    let eps = 1e-9;
    assert_eq!(
        confluencia_resonante(FUSED_UMBAL_PLENO - eps, 0.50, 0.2, 0.2, 0.0),
        None
    );
    assert!(matches!(
        confluencia_resonante(FUSED_UMBAL_PLENO + eps, 0.50, 0.2, 0.2, 0.0),
        Some((true, _))
    ));
    // (3) Zona moderada: sin continuación veta; con continuación (h en la
    // frontera exacta — >= es inclusivo) pasa.
    assert_eq!(
        confluencia_resonante(FUSED_UMBAL_MODERADO + eps, 0.50, 0.2, 0.2, 0.0),
        None
    );
    assert!(matches!(
        confluencia_resonante(
            FUSED_UMBAL_MODERADO + eps,
            HURST_CONTINUACION,
            0.2,
            0.2,
            0.0
        ),
        Some((true, _))
    ));
    // (4) Coherencia mínima: justo-bajo veta.
    assert_eq!(
        confluencia_resonante(0.40, 0.50, COHERENCIA_MINIMA - eps, 0.2, 0.0),
        None
    );
    // (5) Marea adversa justo-más-allá-de-tolerancia veta.
    assert_eq!(
        confluencia_resonante(0.40, 0.50, 0.2, 0.2, -MAREA_MACRO_TOLERANCIA - eps),
        None
    );
}

#[test]
fn cl31_el_espejo_de_una_entrada_es_la_entrada_contraria() {
    for &fused in &[0.25, 0.30, 0.45] {
        for &h in &[0.40, 0.50, 0.55, 0.70] {
            for &marea in &[0.0, 0.0001] {
                let largo = confluencia_resonante(fused, h, 0.2, 0.2, marea);
                let corto = confluencia_resonante(-fused, h, 0.2, 0.2, -marea);
                assert_eq!(
                    largo.map(|(es_largo, conf)| (!es_largo, conf)),
                    corto,
                    "fused {fused}, h {h}, marea {marea}"
                );
            }
        }
    }
}

#[test]
fn cl31_en_la_zona_moderada_decide_la_continuacion() {
    // Browniano (h = 0,5): la zona moderada no basta a ningún lado.
    assert_eq!(confluencia_resonante(0.30, 0.50, 0.2, 0.2, 0.0), None);
    assert_eq!(confluencia_resonante(-0.30, 0.50, 0.2, 0.2, 0.0), None);
    // Continuación: basta a los dos lados, con el mismo bono.
    let largo = confluencia_resonante(0.30, 0.60, 0.2, 0.2, 0.0);
    let corto = confluencia_resonante(-0.30, 0.60, 0.2, 0.2, 0.0);
    assert!(matches!(largo, Some((true, _))), "{largo:?}");
    assert!(matches!(corto, Some((false, _))), "{corto:?}");
}
