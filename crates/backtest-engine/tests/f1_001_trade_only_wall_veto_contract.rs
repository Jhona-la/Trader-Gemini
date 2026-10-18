//! CONTRATO FORMAL: F1-001 (Fable / Ola Ω151)
//!
//! Desacoplamiento de muros L2 en modo `trade_only`.
//!
//! En modo `trade_only: true` (utilizado por `evolution.rs` para IS y OOS sobre
//! datos aggTrades), las cantidades `bid_qty` y `ask_qty` provienen de la
//! codificación del agresor (ej. `binance_vision_sync.rs`). No son muros reales
//! del libro de órdenes L2.
//!
//! Este contrato certifica:
//! 1. En `trade_only: true`, `update_l2_depth` no contamina los muros; `l2_bid_wall`
//!    y `l2_ask_wall` se preservan en `0.0`.
//! 2. `muro_en_contra` permanece estrictamente inerte (`false`) para largos y cortos.
//! 3. En `trade_only: false` (modo bookTicker con profundidad real), `update_l2_depth`
//!    refleja fielmente las cantidades del libro.

use backtest_engine::booktick_replay::{
    run_booktick_replay_with_observer, ReplayConfig, ReplayTick,
};
use god_engine_core::calibration::muro_en_contra;
use quantum_arena::genome::SuperGenotype;
use std::sync::atomic::Ordering::Relaxed;

#[test]
fn contract_f1_001_trade_only_keeps_walls_neutral_and_veto_inert() {
    // 300 ticks con desequilibrio artificial extremo (1.0 vs 99.0)
    let ticks: Vec<ReplayTick> = (0..300u64)
        .map(|i| ReplayTick {
            ts_ms: 1_700_000_000_000 + i * 1_000,
            bid: 100.0,
            ask: 100.01,
            bid_qty: 1.0,
            ask_qty: 99.0,
        })
        .collect();

    // Caso 1: trade_only = true (aggTrades sin libro real)
    let mut cfg_trade = ReplayConfig::default();
    cfg_trade.trade_only = true;

    let mut muros_trade_only = Vec::new();
    let _ = run_booktick_replay_with_observer(
        &ticks,
        &SuperGenotype::default(),
        None,
        &cfg_trade,
        |i, core| {
            if i > 0 {
                let c = &core.arena.coins[0];
                muros_trade_only.push((c.l2_bid_wall.load(Relaxed), c.l2_ask_wall.load(Relaxed)));
            }
        },
    );

    assert!(!muros_trade_only.is_empty(), "El replay trade_only debe procesar eventos");
    for (bid_w, ask_w) in &muros_trade_only {
        assert_eq!(*bid_w, 0.0, "En trade_only l2_bid_wall debe ser 0.0");
        assert_eq!(*ask_w, 0.0, "En trade_only l2_ask_wall debe ser 0.0");
        assert!(
            !muro_en_contra(*bid_w, *ask_w, 2.0, true),
            "muro_en_contra largo no debe dispararse con muros neutros"
        );
        assert!(
            !muro_en_contra(*bid_w, *ask_w, 2.0, false),
            "muro_en_contra corto no debe dispararse con muros neutros"
        );
    }

    // Caso 2: trade_only = false (bookTicker con profundidad real)
    let mut cfg_book = ReplayConfig::default();
    cfg_book.trade_only = false;

    let mut muros_book = Vec::new();
    let _ = run_booktick_replay_with_observer(
        &ticks,
        &SuperGenotype::default(),
        None,
        &cfg_book,
        |i, core| {
            if i > 0 {
                let c = &core.arena.coins[0];
                muros_book.push((c.l2_bid_wall.load(Relaxed), c.l2_ask_wall.load(Relaxed)));
            }
        },
    );

    assert!(!muros_book.is_empty(), "El replay bookTicker debe procesar eventos");
    let (ultimo_bid, ultimo_ask) = muros_book.last().copied().unwrap();
    assert_eq!(ultimo_bid, 1.0, "En bookTicker l2_bid_wall debe reflejar el bid_qty real");
    assert_eq!(ultimo_ask, 99.0, "En bookTicker l2_ask_wall debe reflejar el ask_qty real");
}
