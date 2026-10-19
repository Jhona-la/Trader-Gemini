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
//! Activada por CL-79: la arena guarda la marca de llegada del spot
//! (`CoinArena::spot_ts_ms`, en el reloj de eventos del núcleo) y el núcleo
//! mide el TTL con la edad del spot.
//!
//! CL-97 (F3.7-02 v2): la SDE se sustituyó por una prueba de reversión sobre
//! pares síncronos, que con un spot congelado ya no forma pares. Las dos
//! pruebas de StatArb parten de la reversión demostrada (`comun_base_spot`).

mod comun_base_spot;

use comun_base_spot::*;
use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};

/// Verdad: con el spot sin refrescar más de 30 s (TTL), la z publicada es
/// la abstención exacta `Some(0.0)`, aunque la prueba siga demostrada (la
/// t½ publicada sigue siendo la cota ≤ 2τ*): lo que la silencia es el TTL.
/// El futuro sigue con un evento por segundo (el stream de futuros no cae)
/// durante 45 s. Precondición: la prueba demostró la reversión y publicó
/// z ≠ 0, y en algún evento con el spot caducado la prueba seguía
/// demostrada (τ* puede saltar a un nodo sin madurar).
///
/// CL-97 (F3.7-02 v2): antes la precondición era la SDE OU madura con 12
/// ticks contra un spot fijo; ahora un spot que no se refresca no forma
/// pares, y la precondición es la reversión demostrada sobre pares
/// síncronos (`nucleo_demostrado`).
#[test]
fn cf3_spot_congelado_mas_alla_del_ttl_publica_abstencion() {
    let _g = bloqueo();
    let (arena, mut core, mut m, mut ts) = nucleo_demostrado(6);
    let z0 = leer(&arena, "statarb_ou_zscore").expect("precondición: z publicada");
    assert!(z0.is_finite() && z0 != 0.0, "precondición: z = {z0}");
    // El stream spot cae: apply_spot_tick rechaza valores ≤ 0, así que la
    // arena conserva el último spot. No se llama a update_spot_data.
    let ultimo_spot = ts - 1_000;
    let (mut caducados, mut demostrados) = (0u32, 0u32);
    for _ in 0..45 {
        let (f, _) = m.paso();
        evento(&arena, &mut core, ts, f, None);
        if ts - ultimo_spot > 30_000 {
            caducados += 1;
            let z = leer(&arena, "statarb_ou_zscore");
            assert_eq!(z, Some(0.0), "spot sin refrescar {} s: debe publicar abstención", (ts - ultimo_spot) / 1_000);
            let (th, tau) = (leer(&arena, "statarb_half_life_ms"), leer(&arena, "dominant_tau_ms"));
            if matches!((th, tau), (Some(th), Some(tau)) if th <= 2.0 * tau) {
                demostrados += 1;
            }
        }
        ts += 1_000;
    }
    println!("observado: z demostrada = {z0}; con el spot caducado, {demostrados}/{caducados} eventos con la prueba aún demostrada, todos con z = 0");
    assert!(demostrados > 0, "precondición: la prueba sigue demostrada en algún evento con el spot caducado");
}

/// CL-79. Verdad: la frescura es por LLEGADA del spot, no por cambio de su
/// valor. Con la reversión demostrada y el spot reescrito cada segundo con
/// el mismo valor durante 60 s (el futuro quieto, como un mercado parado),
/// cada llegada forma un par y la z no se abstiene.
#[test]
fn cl79_spot_repetido_sigue_fresco() {
    let _g = bloqueo();
    let (arena, mut core, mut m, mut ts) = nucleo_demostrado(8);
    let (f, sp) = m.paso();
    for _ in 0..60 {
        evento(&arena, &mut core, ts, f, Some(sp));
        ts += 1_000;
    }
    let z = leer(&arena, "statarb_ou_zscore");
    let (th, tau) = (leer(&arena, "statarb_half_life_ms"), leer(&arena, "dominant_tau_ms"));
    println!("observado: z con el spot repetido 60 s = {z:?}, t½ = {th:?}, τ* = {tau:?}");
    assert!(
        matches!((th, tau), (Some(th), Some(tau)) if th <= 2.0 * tau),
        "precondición: la prueba sigue demostrada (t½ = {th:?}, τ* = {tau:?})"
    );
    assert!(matches!(z, Some(v) if v.is_finite() && v != 0.0), "z = {z:?}");
}

/// Núcleo con 12 ticks alternos de ±25 pb y el spot en 50 000 (fixture del
/// z de la base, CL-79/CL-92).
fn nucleo_maduro() -> (std::sync::Arc<GlobalArena>, GodEngineCore, u64) {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    arena.update_spot_data(0, 50_000.0, 50_000.2, 1.0, 1.0);
    let mut ts = 1_000u64;
    for i in 0..12 {
        let fut = 50_000.0 * if i % 2 == 0 { 1.0025 } else { 0.9975 };
        let _ = core.process_tick_dual(0, fut - 0.5, fut + 0.5, 1.0, 1.0, ts, &[0.0; 54], true, true);
        ts += 1_000;
    }
    (arena, core, ts)
}

/// CL-79. Verdad: con el spot congelado más allá del TTL tampoco hay base:
/// `vecm_zscore` se abstiene (0), aunque el spot siga > 0 en la arena.
///
/// CL-92 (spot_z-2, revisión de los PR #43–#55): el nodo τ_k de la base da z
/// cuando lleva τ_k de observaciones, y el z se interpola entre dos nodos
/// (τ_17 = 17,2 s, τ_18 = 68,7 s). Con los 12 s de `nucleo_maduro` ningún
/// nodo tenía historia: el z ya era 0 antes de congelar el spot, y la prueba
/// pasaba aunque el TTL no existiera. Ahora hay 90 s más de spot fresco y la
/// precondición exige un z no nulo antes de congelarlo.
#[test]
fn cl79_spot_congelado_silencia_la_base() {
    let _g = bloqueo();
    let (arena, mut core, mut ts) = nucleo_maduro();
    for i in 0..90 {
        arena.update_spot_data(0, 50_000.0, 50_000.2, 1.0, 1.0);
        let fut = 50_000.0 * if i % 2 == 0 { 1.0025 } else { 0.9975 };
        let _ = core.process_tick_dual(0, fut - 0.5, fut + 0.5, 1.0, 1.0, ts, &[0.0; 54], true, true);
        ts += 1_000;
    }
    let z0 = arena.registry.get_scoped_strict("BTCUSDT", "vecm_zscore");
    assert!(
        matches!(z0, Some(v) if v.is_finite() && v != 0.0),
        "precondición: con 102 s de spot fresco la base da z ≠ 0, y da {z0:?}"
    );
    let _ = core.process_tick_dual(0, 50_100.0, 50_101.0, 1.0, 1.0, ts + 31_000, &[0.0; 54], true, true);
    let z = arena.registry.get_scoped_strict("BTCUSDT", "vecm_zscore");
    println!("observado: vecm_zscore con el spot fresco = {z0:?}, tras 31 s congelado = {z:?}");
    assert_eq!(z, Some(0.0));
}
