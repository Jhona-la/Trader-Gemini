//! C-F3 / F3.6-07 (Línea C): la vida de la rama 12 no debe ser un gen.
//!
//! Teoría que se prueba (una sola τ): la τ con que se dimensiona y vive una
//! orden debe ser la escala del mercado que la generó, medida, no un
//! parámetro evolutivo. `FlowImpulseEngine::evaluate_flow_impulse` no lee
//! ninguna τ del mercado: su `expected_duration_ms` es
//! máx(gen `base_duration_ms`, 15 000). Aguas abajo el núcleo mezcla esa
//! duración en log al 50 % con la τ resonante, así que el gen fija la mitad
//! del logaritmo del horizonte.
//!
//! Solución conocida (propiedad): el mismo mercado (OBI = OFI = 0,8,
//! λ/μ̂ = 10, entropía 0) y la misma τ publicada para la moneda
//! (`consenso_espectral_tau` = 300 000 ms) deben dar la misma vida con el gen
//! en 10 000 y en 100 000 ms. Igualdad exacta (son enteros de ms).
//! El impulso es significativo: exceso = (10 − 1,6)/1,6 = 5,25,
//! z = 5,25/(1/1,6) = 8,4 > 1 y coherencia = √(0,8·tanh(5,25)) ≈ 0,894 > 0,1.
//!
//! Convención: un test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use quantum_arena::GlobalArena;
use signal_engine::flow_impulse::FlowImpulseEngine;
use std::sync::atomic::Ordering;

fn duracion_con_gen(gen_ms: f64) -> u64 {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let c = &arena.config;
    c.weight_obi.store(0.5, Ordering::Relaxed);
    c.weight_ofi.store(0.5, Ordering::Relaxed);
    c.tensor_poly_a.store(0.0, Ordering::Relaxed);
    c.tensor_poly_b.store(0.0, Ordering::Relaxed);
    c.turbo_z_score_stdev.store(1.0, Ordering::Relaxed);
    c.turbo_coherence_threshold.store(0.1, Ordering::Relaxed);
    c.base_duration_ms.store(gen_ms, Ordering::Relaxed);
    // Mismo mercado y misma τ medida publicada para la moneda.
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_tau", 300_000.0);
    FlowImpulseEngine::evaluate_flow_impulse(&arena, 0.8, 0.8, 10.0, 0.0, 100.0, 0.002, 0)
        .expect("impulso significativo: z = 8,4 > 1 y coherencia 0,89 > 0,1")
        .expected_duration_ms
}

#[test]
#[ignore = "OPEN F3.6-07 (Línea C, flow_impulse): expected_duration_ms = máx(gen base_duration_ms, 15 000), sin leer ninguna τ medida; el mismo mercado vive 15 000 o 100 000 ms según el gen"]
fn cf3_6_rama12_la_vida_no_es_un_gen() {
    let (corta, larga) = (duracion_con_gen(10_000.0), duracion_con_gen(100_000.0));
    println!("observado: gen 10 000 -> {corta} ms; gen 100 000 -> {larga} ms");
    assert_eq!(
        corta, larga,
        "mismo mercado, vida distinta según el gen base_duration_ms: {} frente a {} ms",
        corta, larga
    );
}
