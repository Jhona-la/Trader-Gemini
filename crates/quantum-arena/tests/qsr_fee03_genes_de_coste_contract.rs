//! QSR FEE-03 — los genes del MODELO DE COSTE no evolucionan.
//!
//! `latency_penalty_ms` (vec[68]) y `base_slippage_floor` (vec[69]) fijan lo
//! que se cobra cada candidato: la fricción de la puerta de riesgo
//! (`tp_sl::roundtrip_friction`), la física de llenado del simulador
//! (`reality_physics`) y la valoración a mercado del bosque sombra
//! (`random_forest::equity_mtm`). Cada candidato se aplica a su propia arena,
//! así que se juzga con SU modelo de coste: uno que baja su deslizamiento o su
//! latencia paga menos en el examen, admite más operaciones en la puerta y
//! mejora su aptitud sin predecir mejor. La selección premia entonces
//! cobrarse menos, y el genoma promovido lleva ese coste rebajado a la puerta
//! viva, que admite órdenes que no cubren el coste real.
//!
//! Lo correcto es que todos los candidatos de una ronda se midan con el mismo
//! coste (el del incumbente): estos dos genes se copian sin mutar, como los
//! congelados de LG-18/LG-21/F1-084/F1-085, y un genoma aleatorio nace con el
//! coste de la línea base, no con uno sorteado. El vector sigue en 144 genes y
//! los slots 68 y 69 no se mueven (compatibilidad con los genomas guardados).
//! Medir el deslizamiento y la latencia reales y publicarlos es otro átomo.

use quantum_arena::genome::SuperGenotype;

const SLOT_LATENCIA: usize = 68;
const SLOT_DESLIZAMIENTO: usize = 69;

fn genes_de_coste(g: &SuperGenotype) -> [u64; 2] {
    [
        g.latency_penalty_ms.to_bits(),
        g.base_slippage_floor.to_bits(),
    ]
}

/// Un incumbente con valores lejos de la línea base, para que un congelado
/// que se copiara de la base (y no del padre) también se detecte.
fn incumbente() -> SuperGenotype {
    let mut g = SuperGenotype::default();
    g.latency_penalty_ms = 61.25;
    g.base_slippage_floor = 0.000_337;
    g
}

#[test]
fn fee03_la_mutacion_copia_los_genes_de_coste_del_padre() {
    let padre = incumbente();
    let antes = genes_de_coste(&padre);
    for tasa in [0.05_f64, 0.15, 0.5, 1.0] {
        for semilla in 0..256u64 {
            let hijo = padre.mutate_cmaes_seeded(tasa, semilla);
            assert_eq!(
                genes_de_coste(&hijo),
                antes,
                "un gen de coste mutó (tasa {tasa}, semilla {semilla}): el candidato se juzgaría con su propio coste"
            );
        }
    }
}

#[test]
fn fee03_control_los_genes_que_deciden_si_mutan() {
    // Sin este control, un mutate que no mutara nada pasaría el test de arriba.
    // Se mira un gen anterior al slot 68 y otro posterior al 69: quitar dos
    // llamadas al generador desplaza los sorteos de los genes que vienen
    // después, y deben seguir mutando.
    let padre = incumbente();
    let (mut antes_muta, mut despues_muta) = (false, false);
    for semilla in 0..64u64 {
        let hijo = padre.mutate_cmaes_seeded(1.0, semilla);
        antes_muta |= hijo.global_leverage.to_bits() != padre.global_leverage.to_bits()
            || hijo.max_fee_pct.to_bits() != padre.max_fee_pct.to_bits();
        despues_muta |= hijo.margin_cushion_pct.to_bits() != padre.margin_cushion_pct.to_bits()
            || hijo.temporal_scale.to_bits() != padre.temporal_scale.to_bits();
    }
    assert!(
        antes_muta,
        "los genes vivos anteriores al slot 68 sí deben mutar"
    );
    assert!(
        despues_muta,
        "los genes vivos posteriores al slot 69 sí deben mutar"
    );
}

#[test]
fn fee03_dos_fundadores_se_miden_con_un_solo_coste() {
    // walkforward_evolver siembra con la semilla del almacén y con
    // `new_baseline`. Como la mutación copia el coste del padre, sin igualarlo
    // cada linaje conservaría el de su fundador.
    let semilla = incumbente();
    let base = SuperGenotype::new_baseline(0.0002, 0.0005);
    assert_ne!(
        genes_de_coste(&base),
        genes_de_coste(&semilla),
        "precondición: los fundadores tienen costes distintos"
    );
    let base_igualada = base.clone().con_coste_de(&semilla);
    assert_eq!(genes_de_coste(&base_igualada), genes_de_coste(&semilla));
    // El resto del genoma de la base no cambia.
    let (mut vb, mut vi) = (base.to_vector(), base_igualada.to_vector());
    vb[SLOT_LATENCIA] = 0.0;
    vb[SLOT_DESLIZAMIENTO] = 0.0;
    vi[SLOT_LATENCIA] = 0.0;
    vi[SLOT_DESLIZAMIENTO] = 0.0;
    assert_eq!(
        vb.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        vi.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "con_coste_de sólo toca los slots 68 y 69"
    );
    // Los descendientes de ambos linajes comparten el coste del examinador.
    for semilla_rng in 0..32u64 {
        let a = semilla.mutate_cmaes_seeded(0.20, semilla_rng);
        let b = base_igualada.mutate_cmaes_seeded(0.20, semilla_rng);
        assert_eq!(genes_de_coste(&a), genes_de_coste(&b));
    }
}

#[test]
fn fee03_un_genoma_aleatorio_nace_con_el_coste_de_la_linea_base() {
    let base = genes_de_coste(&SuperGenotype::default());
    for _ in 0..64 {
        let g = SuperGenotype::new_random();
        assert_eq!(
            genes_de_coste(&g),
            base,
            "new_random sorteó el modelo de coste"
        );
    }
}

#[test]
fn fee03_los_slots_68_y_69_conservan_su_sitio() {
    assert_eq!(SuperGenotype::DIMENSION, 144);
    let g = incumbente();
    let v = g.to_vector();
    assert_eq!(v[SLOT_LATENCIA].to_bits(), g.latency_penalty_ms.to_bits());
    assert_eq!(
        v[SLOT_DESLIZAMIENTO].to_bits(),
        g.base_slippage_floor.to_bits()
    );
    let r =
        SuperGenotype::try_from_vector(&v).expect("el vector de un genoma válido se reconstruye");
    assert_eq!(genes_de_coste(&r), genes_de_coste(&g));
}
