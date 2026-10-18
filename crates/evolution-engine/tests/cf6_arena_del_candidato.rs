//! CL-142c (C-F6): el examen del demonio y los universos sombra corren con el
//! genoma de su candidato.
//!
//! CL-142b cambió cómo nacen sus arenas (`from_genome_in_own_stack` en vez de
//! leer el genoma del disco). La prueba de paridad de `cf6_logs_acotados`
//! compara los dos constructores, pero no mira los sitios reales: si el
//! examen recibiera otro genoma (el de por defecto, el base en vez de la
//! mutación), seguiría en verde y la evolución puntuaría siempre lo mismo.

use evolution_engine::online_daemon::{genoma_del_examen, WfMarketEnv};
use evolution_engine::random_forest::ShadowForest;
use quantum_arena::genome::SuperGenotype;
use quantum_arena::GlobalArena;

/// El genoma tal como lo ve un arena construido desde él.
fn visto(g: &SuperGenotype) -> Vec<f64> {
    let arena = GlobalArena::from_genome_in_own_stack(13.0, g);
    g.apply_to_arena(&arena);
    SuperGenotype::current_from_arena(&arena).to_vector()
}

fn mercado() -> WfMarketEnv {
    WfMarketEnv { observed_half_spread_pct: 1e-4, tick_pct: 1e-5, max_leverage: 20.0 }
}

/// Esperado: el núcleo del examen ve exactamente el genoma del candidato, y
/// no el de por defecto.
#[test]
fn cl142c_el_examen_ve_el_genoma_del_candidato() {
    let candidato = SuperGenotype::new_baseline(0.0002, 0.0005).mutate_cmaes_seeded(0.15, 20261018);
    let examen = genoma_del_examen(&candidato, &mercado()).to_vector();
    assert_eq!(examen, visto(&candidato));
    assert_ne!(examen, visto(&SuperGenotype::default()), "la mutación debe notarse");
}

/// Esperado: cada universo sombra ve su propio genoma; el 0 es el base y los
/// demás son mutaciones distintas de él.
#[test]
fn cl142c_cada_sombra_ve_su_genoma() {
    let base = SuperGenotype::new_baseline(0.0002, 0.0005);
    let bosque = ShadowForest::new(13.0, base.clone(), 3);
    assert_eq!(bosque.engines.len(), 3);
    for (i, (motor, genoma)) in bosque.engines.iter().zip(&bosque.genomes).enumerate() {
        assert_eq!(SuperGenotype::current_from_arena(&motor.arena).to_vector(), visto(genoma), "universo {i}");
    }
    assert_eq!(visto(&bosque.genomes[0]), visto(&base));
    assert_ne!(visto(&bosque.genomes[1]), visto(&base), "las sombras mutan");
}
