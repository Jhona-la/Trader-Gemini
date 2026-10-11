//! C-F3 · F3.7-03 (Línea C): un spot congelado (no puesto a 0) debe
//! disparar el TTL anti-staleness de StatArb.
//!
//! Teoría que se prueba: el TTL de 30 s (R6-C4) existe para que la SDE OU de
//! la base futuro-spot no compare el futuro con un spot que ya no es el del
//! mercado. La edad que importa es la del ÚLTIMO DATO DE SPOT, no la del
//! último tick del futuro. En producción, cuando cae el stream spot, el
//! único escritor (`apply_spot_tick`, data-pipeline) descarta bid/ask ≤ 0,
//! así que la arena CONSERVA el último spot > 0: el spot no se pone a 0. El
//! núcleo llama a `update_with_clock` en cada evento con el reloj del
//! futuro, la edad medida es siempre 0 y el TTL no dispara nunca: la SDE
//! vota la dislocación del futuro contra un spot fijo. El test vigente
//! (statarb_live_physics_contract.rs) sólo pasa porque pone el spot a 0.
//!
//! Generador: el de `ola73_spot_stale_mas_alla_del_ttl_re_publica_cero` (12
//! ticks alternos de ±25 pb con spot en 50 000), pero el spot se deja
//! congelado, como ocurre en producción.
//!
//! Convención (fase C-F3): un test que hoy falla por la causa de un hallazgo
//! lleva `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};

/// Verdad: tras +31 s sin refrescar el spot (TTL 30 s), la z publicada es
/// la abstención exacta `Some(0.0)`. Precondición: la SDE maduró con spot
/// fresco y publicó una z finita no nula.
#[test]
#[ignore = "OPEN F3.7-03 (Línea C, núcleo; data-pipeline debe publicar la edad del spot): la arena no guarda la marca de tiempo del spot; el núcleo alimenta la SDE en cada evento con el reloj del futuro, así que la edad medida es 0 y un spot congelado sigue votando"]
fn cf3_spot_congelado_mas_alla_del_ttl_publica_abstencion() {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    arena.update_spot_data(0, 50_000.0, 50_000.2, 1.0, 1.0);
    let mut ts = 1_000u64;
    for i in 0..12 {
        let fut = 50_000.0 * if i % 2 == 0 { 1.0025 } else { 0.9975 };
        let _ = core.process_tick_dual(
            0,
            fut - 0.5,
            fut + 0.5,
            1.0,
            1.0,
            ts,
            &[0.0; 54],
            true,
            true,
        );
        ts += 1_000;
    }
    let z0 = arena
        .registry
        .get_scoped_strict("BTCUSDT", "statarb_ou_zscore")
        .expect("precondición: SDE madura");
    assert!(z0.is_finite() && z0 != 0.0, "precondición: z = {z0}");
    // El stream spot cae: apply_spot_tick rechaza valores ≤ 0, así que la
    // arena conserva el último spot. No se llama a update_spot_data.
    ts += 31_000;
    let _ = core.process_tick_dual(0, 50_100.0, 50_101.0, 1.0, 1.0, ts, &[0.0; 54], true, true);
    let z = arena
        .registry
        .get_scoped_strict("BTCUSDT", "statarb_ou_zscore");
    println!("observado: z madura = {z0}, z tras 31 s con el spot congelado = {z:?}");
    assert_eq!(
        z,
        Some(0.0),
        "spot sin refrescar durante más de 30 s: debe publicar abstención"
    );
}
