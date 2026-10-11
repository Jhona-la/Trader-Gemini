//! C-F3 · F3.6-03 (Línea C): la entropía del libro que lee el votante
//! Rényi-Tsallis debe aportar algo que OBI no tenga.
//!
//! Teoría que se prueba: OBI = (b − a)/(b + a) y la fracción del libro del
//! lado comprador p_bid = b/(b + a) = (1 + OBI)/2 son la MISMA variable. Una
//! entropía de Tsallis binaria q = 1,5 sobre {p_bid, 1 − p_bid}, normalizada
//! por su máximo S_max = (1 − 2·0,5^1,5)/0,5, es entonces una función
//! determinista h(OBI) del mismo evento: no añade información (ni historia,
//! ni profundidad por niveles, ni ventana). Si la entropía publicada
//! (`tsallis_q_entropy`) es h(OBI) en todos los estados, el votante es otra
//! copia de OBI en el consenso.
//!
//! Generador: 120 ticks de la moneda 0 por `process_tick_dual`, rotando seis
//! libros (pares con el MISMO OBI a distinta profundidad: (10, 2) y
//! (100, 20); (500, 100) y (1, 50) con otros OBI), mid 100 + sin(0,37·i) e
//! intervalos irregulares de 250 ms a 5,65 s.
//!
//! Convención (fase C-F3): un test que hoy falla por la causa de un hallazgo
//! lleva `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};

/// h(OBI): Tsallis binario q = 1,5 normalizado de {(1 + OBI)/2, (1 − OBI)/2},
/// con el mismo recorte de probabilidades [1e-4; 0,9999] que el escritor.
fn h(obi: f64) -> f64 {
    let p_bid = ((1.0 + obi) / 2.0).clamp(0.0001, 0.9999);
    let p_ask = ((1.0 - obi) / 2.0).clamp(0.0001, 0.9999);
    ((1.0 - (p_bid.powf(1.5) + p_ask.powf(1.5))) / 0.5 / 0.585_786_437_626_905).clamp(0.0, 1.0)
}

/// Verdad: debe existir algún estado con |T − h(OBI)| > 1e-6 (1e-6 está muy
/// por encima del redondeo de powf, ~1e-15, y muy por debajo de cualquier
/// diferencia informativa). Precondición: el escritor corrió (OBI y T
/// finitos en el registro de la moneda).
#[test]
#[ignore = "OPEN F3.6-03 (Línea C): p_bid = bid_qty/(bid_qty + ask_qty) = (1 + OBI)/2 en el mismo evento, así que la «entropía» publicada es h(OBI) sin ventana ni historia y el votante Rényi-Tsallis cuenta OBI otra vez"]
fn cf3_6_entropia_publicada_no_es_funcion_de_obi() {
    symbols::update_dynamic_universe(vec!["CF3ENTUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "CF3ENTUSDT",
    )]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    let omni = [0.0f64; 54];
    let libros = [
        (10.0, 2.0),
        (100.0, 20.0),
        (2.0, 10.0),
        (6.0, 6.0),
        (1.0, 50.0),
        (500.0, 100.0),
    ];
    let mut t: u64 = 1_700_000_000_000;
    let (mut escritos, mut max_dev) = (0usize, 0.0f64);
    for i in 0..120usize {
        let (bq, aq) = libros[i % libros.len()];
        let mid = 100.0 + (i as f64 * 0.37).sin();
        t += 250 + (i as u64 % 7) * 900;
        let _ = core.process_tick_dual(0, mid - 0.01, mid + 0.01, bq, aq, t, &omni, false, false);
        let obi = arena
            .registry
            .get_for_coin_or(0, "order_book_imbalance", f64::NAN);
        let ent = arena
            .registry
            .get_for_coin_or(0, "tsallis_q_entropy", f64::NAN);
        if obi.is_finite() && ent.is_finite() {
            escritos += 1;
            max_dev = max_dev.max((ent - h(obi)).abs());
        }
    }
    println!("observado: {escritos} estados escritos, máx |T − h(OBI)| = {max_dev:e}");
    assert!(escritos > 0, "arnés: el escritor de la entropía no corrió");
    assert!(
        max_dev > 1e-6,
        "la entropía publicada es una función exacta de OBI (desviación máxima {:e}): el votante no aporta otra fuente",
        max_dev
    );
}
