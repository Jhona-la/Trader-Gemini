//! CL-162: el daemon no explora el Kelly legado (genes 10 y 11).
//!
//! Desde F1-116 + F1-087 ninguna orden del examen walk-forward toma su
//! pérdida al stop del Kelly legado: LH-04 dimensiona con evidencia y el
//! resto va a la orden mínima. El pre-examen, en cambio, sí lo leía
//! (`kelly_at_tau(16 s)`), así que los candidatos se ordenaban por una
//! coordenada que en el examen real no fija ningún tamaño: a través del
//! margen legado sólo elige el apalancamiento de la orden mínima (D-130) y,
//! con él, la puerta REJ_FEE_IMPACT. La restauración de genes no
//! observables los devuelve al incumbente: todos los candidatos de la ronda
//! llegan al pre-examen con la misma Kelly.

use evolution_engine::online_daemon::restaurar_genes_no_observables_en_examen;
use quantum_arena::genome::SuperGenotype;

const WF_BAR_MS: f64 = 16_000.0;

#[test]
fn cl162_la_restauracion_devuelve_el_kelly_legado_del_incumbente() {
    let incumbent = SuperGenotype::new_baseline(0.0002, 0.0005);
    let mut candidate = incumbent.clone();
    candidate.scalp_kelly_fraction = 2.0;
    candidate.swing_kelly_fraction = 1.0;
    assert_ne!(
        candidate.kelly_at_tau(WF_BAR_MS).to_bits(),
        incumbent.kelly_at_tau(WF_BAR_MS).to_bits(),
        "premisa: la perturbación mueve la Kelly que lee el pre-examen"
    );

    restaurar_genes_no_observables_en_examen(&mut candidate, &incumbent);

    assert_eq!(candidate.scalp_kelly_fraction.to_bits(), incumbent.scalp_kelly_fraction.to_bits());
    assert_eq!(candidate.swing_kelly_fraction.to_bits(), incumbent.swing_kelly_fraction.to_bits());
    assert_eq!(
        candidate.kelly_at_tau(WF_BAR_MS).to_bits(),
        incumbent.kelly_at_tau(WF_BAR_MS).to_bits(),
        "el pre-examen recibe la Kelly del incumbente"
    );
}

#[test]
fn cl162_la_restauracion_no_toca_el_tope_vivo_del_gen_33() {
    let incumbent = SuperGenotype::new_baseline(0.0002, 0.0005);
    let mut candidate = incumbent.clone();
    candidate.kelly_clamp_max = 0.2;
    restaurar_genes_no_observables_en_examen(&mut candidate, &incumbent);
    assert_eq!(candidate.kelly_clamp_max.to_bits(), 0.2_f64.to_bits());
}
