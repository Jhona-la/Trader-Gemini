//! CL-31 — la rama 15 (resonancia en τ*) trata la persistencia sin lado.
//! Función pura; sin exchange ni estado.
use god_engine_core::confluencia_resonante;

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
