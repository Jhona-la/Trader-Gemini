//! C-F3 / F3.1-04 (Línea C): el TTL del veredicto espectral no puede durar
//! más que la vida que el orquestador fija a la posición que ese veredicto
//! dirige.
//!
//! Teoría que se prueba (una sola τ): el orquestador hace vivir la posición a
//! min(τ_dom, 12 h) y nunca menos de 30 s (orchestrator.rs, rama
//! `espectral_activo` de `expected_lifetime_ms`). El veredicto que la dirige
//! caduca a los `ttl_consenso_ms(τ_dom)` (skill_motores.rs). Si el TTL supera
//! la vida, un dominante de una escala que nadie opera sigue dirigiendo el
//! tensor después de que su propia posición habría expirado. Con
//! τ_k = 4^k·10⁻⁶ ms, la primera escala que rompe la cota es k = 23
//! (τ ≈ 7,04·10⁷ ms ≈ 19,5 h > 12 h); con k = 31 el TTL es de ~146 años.
//!
//! Prueba de propiedad sobre un barrido determinista de las 32 escalas.
//! Tolerancia: 1 ms, el truncado de `as u64`.
//!
//! Convención: un test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto. Para
//! correrlo: `cargo test ... -- --include-ignored`.

use quantum_arena::temporal_spectrum::{
    SPECTRUM_SCALES_MS, TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS,
};
use signal_engine::skill_motores::ttl_consenso_ms;

/// Verdad: para toda escala, ttl ≤ max(min(τ, 12 h), 30 s) + 1 ms.
/// Hoy: ttl = max(τ, 30 s) sin techo; en k = 23 da 7,04·10⁷ ms frente a una
/// vida de 4,32·10⁷ ms.
#[test]
#[ignore = "OPEN F3.1-04 (Qoder, skill_motores): ttl_consenso_ms = max(τ_dom, 30 s) sin acotar por TAU_ANCHOR_SLOW_MS; desde k = 23 el dominante sobrevive a la vida min(τ, 12 h) que fija el orquestador"]
fn cf3_ttl_del_veredicto_no_supera_la_vida_que_fija_el_orquestador() {
    let mut primera_rota: Option<usize> = None;
    for (k, &tau) in SPECTRUM_SCALES_MS.iter().enumerate() {
        let ttl = ttl_consenso_ms(tau) as f64;
        let vida = tau.min(TAU_ANCHOR_SLOW_MS).max(TAU_ANCHOR_FAST_MS);
        if ttl > vida + 1.0 && primera_rota.is_none() {
            primera_rota = Some(k);
        }
        println!("observado: k = {k}: τ = {tau:.3e} ms, TTL {ttl:.3e} ms, vida {vida:.3e} ms");
    }
    for (k, &tau) in SPECTRUM_SCALES_MS.iter().enumerate() {
        let ttl = ttl_consenso_ms(tau) as f64;
        // orchestrator.rs: la posición vive min(τ, 12 h), y nunca menos de 30 s.
        let vida = tau.min(TAU_ANCHOR_SLOW_MS).max(TAU_ANCHOR_FAST_MS);
        assert!(
            ttl <= vida + 1.0,
            "escala {k}: τ = {tau:.3e} ms, TTL {ttl:.3e} ms > vida {vida:.3e} ms \
             (primera escala rota: {primera_rota:?})"
        );
    }
}
