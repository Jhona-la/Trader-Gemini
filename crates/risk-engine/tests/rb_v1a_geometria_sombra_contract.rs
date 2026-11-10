//! CONTRATO RB-V1a (Codex RB, 2026-11-10): la sombra de una orden validada
//! REAL lleva su geometría y su fricción.
//!
//! QUÉ: el núcleo registra en el libro contrafactual cada intención que el
//! riesgo validó y que después vetan el consejo o el ML-GATE. La geometría
//! sale de `MedidaOrden::geometria_sombra`, que la deriva de la medida
//! pendiente sin consumirla.
//! POR QUÉ: si la sombra usara otra vara (otro TP, otro stop u otra fricción)
//! que la del riesgo, juzgaría una operación distinta de la vetada.
//! CÓMO: con la cuenta sana del contrato C07 (la misma de C-F5c) se valida
//! una orden real y se comprueba que (1) TP y SL coinciden con los precios de
//! la orden respecto a la entrada, (2) leer la medida no la consume y (3) la
//! fricción y el TP coinciden con lo que la cuenta mide al abrirla (`w`, `l`).

use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc};

/// Cuenta sana de C07/C-F5c: 13 USD, stop y TP explícitos a 99 y 102 desde 100.
fn cuenta(capital: f64) -> (Arc<GlobalArena>, SignalIntent) {
    quantum_arena::symbol_registry::update_registry(vec![
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "BTCUSDT".into(),
            step_size: 0.001,
            tick_size: 0.01,
            min_qty: 0.001,
            min_notional: 5.0,
            max_leverage: 20,
            maker_fee: 0.0002,
            taker_fee: 0.0004,
            is_shadow: false,
        },
    ]);
    let arena = GlobalArena::build_in_own_stack(capital);
    let coin = &arena.coins[0];
    coin.current_price.store(100.0, Relaxed);
    coin.current_atr.store(1.0, Relaxed);
    coin.hurst_exponent.store(0.5, Relaxed);
    coin.metrics.trade_count.store(100, Relaxed);
    coin.metrics.profit_factor.store(2.0, Relaxed);
    coin.metrics.win_rate.store(0.75, Relaxed);
    coin.metrics.kelly_fraction.store(0.15, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.2, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.min_confidence_btc.store(0.62, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        tp_price_target: 102.0,
        sl_price_target: 99.0,
        ..SignalIntent::default()
    };
    (arena, intent)
}

#[test]
fn rb_v1a_la_sombra_lleva_la_geometria_y_la_friccion_de_la_orden_validada() {
    let (arena, intent) = cuenta(13.0);
    let mut riesgo = RiskEngine::new(13.0);
    let orden = riesgo.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        orden.signal,
        SignalType::Long,
        "{}",
        risk_engine::reject_report()
    );

    let entrada = arena.coins[0].current_price.load(Relaxed);
    let Some(g) = riesgo.medida_validada(0).and_then(|m| m.geometria_sombra()) else {
        panic!("una orden validada deja una medida con geometría");
    };
    let tp = (orden.tp_target - entrada) / entrada;
    let sl = (entrada - orden.sl_target) / entrada;
    let cota = 8.0 * f64::EPSILON * (tp + sl + g.friccion_pct);
    println!(
        "observado: entrada {entrada}, TP {:.6} frente a {tp:.6}, SL {:.6} frente a {sl:.6}, fricción {:.6}",
        g.tp_pct, g.sl_pct, g.friccion_pct
    );
    assert!(
        (g.tp_pct - tp).abs() <= cota,
        "TP de la sombra = TP de la orden"
    );
    assert!(
        (g.sl_pct - sl).abs() <= cota,
        "SL de la sombra = SL de la orden"
    );
    assert!(
        g.friccion_pct > 0.0,
        "la orden paga comisiones: fricción positiva"
    );

    // Leer no consume: la apertura toma la misma medida y la registra.
    assert!(
        riesgo.medida_validada(1).is_none(),
        "la pendiente es de la moneda 0"
    );
    let Some(m) = riesgo.tomar_medida_validada(0) else {
        panic!("leer la medida en sombra no la consume");
    };
    assert_eq!(riesgo.medida_validada(0), None, "tomarla sí la consume");
    RiskEngine::registrar_medida(&arena, &m);
    let w = arena.ganancia_en_stops.load(Relaxed);
    let l = arena.perdida_en_stops.load(Relaxed);
    // La primera operación fija las medias (C-F5c usa la misma cota, 1e-12).
    let f_medida = (l - 1.0) * sl;
    assert!(
        (f_medida - g.friccion_pct).abs() < 1e-12,
        "fricción {f_medida} frente a {}",
        g.friccion_pct
    );
    assert!(
        (w * sl + f_medida - g.tp_pct).abs() < 1e-12,
        "w·SL + f es el TP de la sombra"
    );
}
