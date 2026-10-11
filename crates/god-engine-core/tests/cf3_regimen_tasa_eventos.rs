//! C-F3 · F3.7-06 (Línea C): el régimen global no debe depender de la tasa
//! de eventos de libro.
//!
//! Teoría que se prueba: el régimen (símplex p_range, p_bull, p_crash,
//! p_chaos) describe el PROCESO de precio en tiempo físico. Un libro
//! duplicado a +500 ms con el mismo mid no añade información (retorno 0, la
//! misma trayectoria de precio en tiempo físico, las mismas velas de 1 min),
//! así que no puede cambiar el régimen. Hoy el símplex se recalcula en
//! eventos de libro del slot 0 con z_btc = (EMA20 − EMA200)/EMA200 entre
//! σ_tick·k(20, 200), con EMAs de 20/200 EVENTOS y una σ por EVENTO: con el
//! doble de eventos la deriva por evento se reduce a la mitad y la σ por
//! evento en 1/√2, así que la parte de z que viene de la deriva cae en 1/√2.
//!
//! Por qué la prueba mide p_bull relativo y promediado (corrección del
//! borrador, que pedía |Δp_bull| < 0,05 en el último evento): p_bull =
//! w_trend·σ((z − 1,96)/0,5), y w_trend sale del Hurst muestreado al cierre
//! de la vela de 1 min (D-615b, ya en tiempo físico e idéntico en las dos
//! variantes). Con un paseo sin memoria el Hurst queda neutro (0,5) y
//! w_trend ≈ 0,0193, así que |Δp_bull| nunca puede pasar de 0,0193: la
//! tolerancia absoluta de 0,05 no podía fallar. Además, con la deriva del
//! borrador (2e-5 por s, z ≈ 11) la sigmoide satura en las dos variantes.
//! Se mide la media de p_bull en las dos últimas horas (tras cada evento del
//! segundo original, no tras los duplicados) y se exige que la razón B/A
//! esté a menos del 10 % de 1. Con un régimen en tiempo físico las dos
//! variantes verían las mismas entradas en cada segundo y la razón sería 1;
//! el ruido es común a las dos (misma semilla), así que el 10 % es holgura,
//! no error de muestreo.
//!
//! Generador: LCG; ln p con deriva de 5e-6 por segundo y ruido uniforme de
//! ±5e-5 (σ_tick ≈ 2,9e-5); 4 h de libros a 1 s. Deriva elegida para que la
//! parte de z que viene de ella quede en la zona sensible de la sigmoide:
//! z ≈ deriva·90/(σ_tick·k) con el retardo 90 = (200 − 20)/2 de las EMAs y
//! k = ema_spread_sd(20, 200) ≈ 6,07 ⇒ z_A ≈ 2,6 y z_B ≈ 1,85. En la
//! variante B cada evento va seguido de un libro idéntico a +500 ms.
//!
//! Convención (fase C-F3): un test que hoy falla por la causa de un hallazgo
//! lleva `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::atomic::Ordering;

/// Media de p_bull leída tras cada evento original de las dos últimas horas,
/// y el símplex final (p_range, p_bull, p_crash, p_chaos).
fn correr(duplicar: bool) -> (f64, [f64; 4]) {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    let mut lcg = 12_345u64;
    let mut ln_p = 50_000f64.ln();
    let mut ts = 1_000u64;
    let libro = |core: &mut GodEngineCore, mid: f64, t: u64| {
        let _ = core.process_event(
            0,
            false,
            false,
            true,
            mid,
            0.0,
            mid - 0.5,
            mid + 0.5,
            1.0,
            1.0,
            0.0,
            0.0,
            t,
            false,
            &[0.0; 54],
            false,
        );
    };
    let n = 4 * 3_600u64;
    let (mut suma, mut cuenta) = (0.0f64, 0usize);
    for i in 0..n {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((lcg >> 11) as f64) / (1u64 << 53) as f64 - 0.5;
        ln_p += 5e-6 + 1e-4 * u;
        let mid = ln_p.exp();
        libro(&mut core, mid, ts);
        if i >= n / 2 {
            suma += arena.regime_p_bull.load(Ordering::Relaxed);
            cuenta += 1;
        }
        if duplicar {
            libro(&mut core, mid, ts + 500);
        }
        ts += 1_000;
    }
    (
        suma / cuenta as f64,
        [
            arena.regime_p_range.load(Ordering::Relaxed),
            arena.regime_p_bull.load(Ordering::Relaxed),
            arena.regime_p_crash.load(Ordering::Relaxed),
            arena.regime_p_chaos.load(Ordering::Relaxed),
        ],
    )
}

/// Verdad: |media p_bull(B)/media p_bull(A) − 1| ≤ 0,10. Duplicar un libro
/// sin información no cambia el proceso de precio.
#[test]
#[ignore = "OPEN F3.7-06 (Línea C): el símplex de régimen usa EMAs de 20/200 eventos y una σ por evento (reloj de eventos); la z de la tendencia, y con ella p_bull, cambian con la tasa de libros del slot 0"]
fn cf3_regimen_no_depende_de_la_tasa_de_eventos() {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let (media_a, fin_a) = correr(false);
    let (media_b, fin_b) = correr(true);
    println!(
        "observado: media p_bull A (1 libro/s) = {media_a:.6}, B (2 libros/s) = {media_b:.6}, razón B/A = {:.4}; símplex final [range, bull, crash, chaos] A = {fin_a:?}, B = {fin_b:?}",
        media_b / media_a
    );
    assert!(
        media_a.is_finite() && media_b.is_finite() && media_a > 0.0,
        "p_bull no publicado"
    );
    assert!(
        (media_b / media_a - 1.0).abs() <= 0.10,
        "p_bull depende de la tasa de eventos: media {media_a:.6} con 1 libro/s frente a {media_b:.6} con 2 (razón {:.4})",
        media_b / media_a
    );
}
