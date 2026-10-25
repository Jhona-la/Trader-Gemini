//! CL-173 (fila N12 del censo C-F7b): el tope de ruina F1-092 acota la
//! pérdida al stop CON la fricción de ida y vuelta.
//!
//! F1-092 recortaba el margen cuando margen·L·SL pasaba de tope·C, sin c
//! (`tp_sl::roundtrip_friction`): tras un rescate de nocional mínimo la orden
//! salía con una pérdida real al stop N·(SL + c) por encima del tope. Ahora
//! el tope se calcula con `kelly_envelope::nocional_por_riesgo(tope, C, SL, c)`,
//! la misma función con la que dimensiona LH-04 (CL-159).
//!
//! Fixture de lote grueso (paso 0,5 a precio 100: la orden mínima del lote son
//! 50 USD) con 140 USD, régimen estándar (sin el recorte micro de Sol-A2) y un
//! stop tan ancho que la liquidación deja L = 1. El stop se elige para que la
//! orden mínima quepa con c en el tope (CL-172 la deja pasar) pero el piso
//! ×1,0005 del rescate la empuje por encima.

use quantum_arena::GlobalArena;
use risk_engine::{capital_regime, kelly_envelope, ruin, tp_sl, RiskEngine};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

const PRECIO: f64 = 100.0;
const LOTE_GRUESO: usize = 0;
const BTC: usize = 1;

fn registro() {
    std::env::set_var("TG_GENOME_ENV", "backtest");
    static REGISTRY: std::sync::Once = std::sync::Once::new();
    REGISTRY.call_once(|| {
        quantum_arena::symbol_registry::update_registry(vec![
            quantum_arena::symbol_registry::SymbolSpec {
                symbol: "LOTEGRUESOUSDT".into(),
                step_size: 0.5,
                tick_size: 0.01,
                min_qty: 0.5,
                min_notional: 5.0,
                max_leverage: 20,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            },
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
    });
}

/// Arena con una moneda de métricas (n, win rate, PF, Kelly).
fn arena(capital: f64, coin: usize, n: usize, wr: f64, pf: f64) -> std::sync::Arc<GlobalArena> {
    registro();
    let arena = GlobalArena::build_in_own_stack(capital);
    arena.coins[coin].current_price.store(PRECIO, Relaxed);
    arena.coins[coin].current_atr.store(1.0, Relaxed);
    arena.coins[coin].hurst_exponent.store(0.5, Relaxed);
    arena.coins[coin].metrics.trade_count.store(n, Relaxed);
    arena.coins[coin].metrics.profit_factor.store(pf, Relaxed);
    arena.coins[coin].metrics.win_rate.store(wr, Relaxed);
    arena.coins[coin]
        .metrics
        .kelly_fraction
        .store(0.15, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.2, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena
}

/// c con las entradas del gate (taker, piso, ATR/precio, latencia 0).
fn friccion(arena: &GlobalArena, coin: usize) -> f64 {
    let taker = arena
        .config
        .fee_rates_for(coin)
        .map_or(f64::NAN, |(_, t)| t);
    let piso = arena.config.base_slippage_floor.load(Relaxed).max(0.00001);
    let atr_pct = arena.coins[coin].current_atr.load(Relaxed) / PRECIO;
    let c = tp_sl::roundtrip_friction(taker, piso, atr_pct, 0.0);
    assert!(c.is_finite() && c > 0.0, "premisa: c = {c}");
    c
}

fn intencion(sl_price_target: f64) -> SignalIntent {
    SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        expected_duration_ms: 60_000,
        tp_price_target: 300.0,
        sl_price_target,
        ..SignalIntent::default()
    }
}

#[test]
fn f1_092_acota_la_perdida_al_stop_con_la_friccion() {
    const C: f64 = 140.0;
    // (b) LH-04 rescatada al mínimo del lote y (c) la sonda.
    for (clase, n, wr) in [("LH-04 rescatada", 100_usize, 0.75), ("sonda", 0, 0.0)] {
        let arena = arena(C, LOTE_GRUESO, n, wr, 2.0);
        let c = friccion(&arena, LOTE_GRUESO);
        let tope = ruin::clamp_ruin(1.0, ruin::conservative_loss_q(wr, n as f64));
        assert_eq!(
            tope, 0.25,
            "premisa ({clase}): el tope es el axioma del 25 %"
        );
        let n_min = capital_regime::nocional_minimo_ejecutable(
            capital_regime::effective_min_notional(5.0) + 0.1,
            0.5,
            0.5,
            PRECIO,
        )
        .unwrap_or(f64::NAN);
        assert_eq!(n_min, 50.0, "premisa: la orden mínima del lote");
        // El stop deja la orden mínima en el tope con c (viable por CL-172)
        // y el piso ×1,0005 del rescate la pasa; sin c no la pasaba.
        let sl = tope * C / (n_min * 1.00025) - c;
        assert!(n_min * (sl + c) <= tope * C, "premisa: viable con c");
        assert!(
            n_min * 1.0005 * sl <= tope * C,
            "premisa: sin c el F1-092 de main no muerde"
        );
        assert!(
            n_min * 1.0005 * (sl + c) > tope * C,
            "premisa: con c el piso ×1,0005 pasa el tope"
        );
        let sl_target = PRECIO * (1.0 - sl);
        let out =
            RiskEngine::new(C).evaluate_quantum_order(LOTE_GRUESO, &intencion(sl_target), &arena);
        let sl_orden = (PRECIO - out.sl_target) / PRECIO;
        let nocional = out.volume_usd * out.leverage;
        println!(
            "{clase}: SL {sl_orden:.6}, c {c:.6}: {:?}, {} USD ×{} = {nocional} USD, pérdida al stop {} (tope {}); {}",
            out.signal,
            out.volume_usd,
            out.leverage,
            nocional * (sl_orden + c),
            tope * C,
            risk_engine::reject_report()
        );
        assert_eq!(
            out.signal,
            SignalType::Long,
            "premisa ({clase}): la orden sale"
        );
        assert!(
            nocional >= n_min * (1.0 - 1e-9),
            "({clase}) la orden sigue en el mínimo del lote: {nocional}"
        );
        assert!(
            nocional * (sl_orden + c) <= tope * C * (1.0 + 1e-9),
            "({clase}) la pérdida al stop con la fricción no pasa del tope: {} > {}",
            nocional * (sl_orden + c),
            tope * C
        );
    }
}

#[test]
fn f1_092_no_recorta_lh04_con_f_igual_al_tope() {
    // LH-04 sin rescate con f = κ = tope = 0,25: el tope con la fricción no
    // la recorta ni en un ulp, porque usa la misma función y los mismos
    // argumentos que LH-04. L ≥ 2 para que el cociente por L cuente.
    const C: f64 = 1_000.0;
    let arena = arena(C, BTC, 500, 0.95, 10.0);
    let c = friccion(&arena, BTC);
    let tope = ruin::clamp_ruin(1.0, ruin::conservative_loss_q(0.95, 500.0));
    assert_eq!(tope, 0.25, "premisa: el tope es el axioma del 25 %");
    let out = RiskEngine::new(C).evaluate_quantum_order(BTC, &intencion(65.0), &arena);
    println!(
        "{:?}, {} USD ×{}, SL {}; f_realizado {}; sol_a2 {}; {}",
        out.signal,
        out.volume_usd,
        out.leverage,
        out.sl_target,
        arena
            .registry
            .get_for_coin_or(BTC, "lh04_f_realizado", f64::NAN),
        arena
            .registry
            .get_for_coin_or(BTC, "sol_a2_margen_reducido", 0.0),
        risk_engine::reject_report()
    );
    assert_eq!(out.signal, SignalType::Long, "premisa: la orden sale");
    assert!(out.leverage >= 2.0, "premisa: L ≥ 2, {}", out.leverage);
    assert_eq!(
        arena
            .registry
            .get_for_coin_or(BTC, "sol_a2_margen_reducido", 0.0),
        0.0,
        "premisa: sin Sol-A2"
    );
    let sl = (PRECIO - out.sl_target) / PRECIO;
    let esperado =
        kelly_envelope::nocional_por_riesgo(0.25, C, sl, c).unwrap_or(f64::NAN) / out.leverage;
    assert_eq!(
        out.volume_usd.to_bits(),
        esperado.to_bits(),
        "{} vs {esperado}",
        out.volume_usd
    );
}
