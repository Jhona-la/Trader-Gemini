//! CONTRATO RB-V1b/RB-V1c (Codex RB, 2026-11-10): la geometría y la promesa que
//! el riesgo publica para la sombra son las de la intención evaluada, o ninguna.
//!
//! QUÉ: `RiskEngine::geometria_evaluada` lleva el TP, el SL, la fricción de ida
//! y vuelta y el τ con que `evaluate_single_intent` evaluó la intención. El
//! núcleo la usa para juzgar en sombra tanto los vetos posteriores al riesgo
//! (consejo, ML-GATE) como los rechazos del propio riesgo posteriores a la
//! geometría (confianza, EV, viabilidad, nocional, orquestador, fee…).
//! POR QUÉ: si la sombra usara otra vara que la del riesgo juzgaría una
//! operación distinta de la rechazada; y una geometría rancia de una
//! evaluación anterior atribuiría a un rechazo una operación que nunca se
//! evaluó.
//! CÓMO: con la cuenta sana del contrato C07 (la misma de C-F5c):
//! 1. orden validada: TP y SL coinciden con sus precios respecto a la entrada,
//!    τ con el de la orden, y la fricción con la que mide la cuenta al abrirla;
//! 2. rechazo POSTERIOR a la geometría (confianza): la geometría sigue
//!    publicada, con el TP y el SL de la intención;
//! 3. rechazo ANTERIOR (moneda sin spec, intención plana): `None`, también
//!    después de una evaluación que sí la había publicado.

use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Mutex};

/// Las pruebas cambian el registro global de símbolos.
static SERIE: Mutex<()> = Mutex::new(());

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
fn rb_v1b_la_orden_validada_publica_su_geometria_exacta() {
    let _serie = SERIE.lock().unwrap_or_else(|p| p.into_inner());
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
    let Some(g) = riesgo.geometria_evaluada(0) else {
        panic!("una orden validada publica su geometría");
    };
    let tp = (orden.tp_target - entrada) / entrada;
    let sl = (entrada - orden.sl_target) / entrada;
    let cota = 8.0 * f64::EPSILON * (tp + sl + g.friccion_pct);
    println!(
        "observado: TP {:.6} frente a {tp:.6}, SL {:.6} frente a {sl:.6}, fricción {:.6}, τ {} ms",
        g.tp_pct, g.sl_pct, g.friccion_pct, g.tau_ms
    );
    assert!(
        (g.tp_pct - tp).abs() <= cota,
        "TP de la sombra = TP de la orden"
    );
    assert!(
        (g.sl_pct - sl).abs() <= cota,
        "SL de la sombra = SL de la orden"
    );
    assert_eq!(
        g.tau_ms, orden.tau_ms,
        "τ de la sombra = τ con el que se dimensionó"
    );
    assert!(
        g.friccion_pct > 0.0,
        "la orden paga comisiones: fricción positiva"
    );

    // La fricción es la misma que la cuenta mide al abrir la orden (C-F5c):
    // con la primera operación, l = (SL + f)/SL y w = (TP − f)/SL exactos.
    // RB-V1c: la promesa de la puerta de EV es p·TP − (1−p)·SL con la p de la
    // intención (0,9) y la geometría de la orden: 0,9·0,02 − 0,1·0,01 = 0,017.
    let Some(promesa) = riesgo.promesa_evaluada(0) else {
        panic!("una orden validada publica lo que promete");
    };
    println!("observado: promesa bruta {promesa:.6} (esperada 0,017)");
    assert!(
        (promesa - (0.9 * tp - 0.1 * sl)).abs() <= cota,
        "promesa {promesa}"
    );
    assert_eq!(
        riesgo.promesa_evaluada(1),
        None,
        "la promesa es de la moneda 0"
    );

    let Some(m) = riesgo.tomar_medida_validada(0) else {
        panic!("la orden validada deja su medida pendiente");
    };
    RiskEngine::registrar_medida(&arena, &m);
    let w = arena.ganancia_en_stops.load(Relaxed);
    let l = arena.perdida_en_stops.load(Relaxed);
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

#[test]
fn rb_v1b_un_rechazo_posterior_a_la_geometria_la_conserva() {
    let _serie = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, intent) = cuenta(13.0);
    let mut riesgo = RiskEngine::new(13.0);
    let dudosa = SignalIntent {
        confidence: 0.30,
        ..intent
    };
    let orden = riesgo.evaluate_quantum_order(0, &dudosa, &arena);
    assert_eq!(orden.signal, SignalType::Flat);
    assert_eq!(
        risk_engine::ultimo_rechazo(),
        Some(risk_engine::REJ_CONFIDENCE),
        "{}",
        risk_engine::reject_report()
    );
    let Some(g) = riesgo.geometria_evaluada(0) else {
        panic!("la confianza se juzga después de la geometría");
    };
    assert_eq!(
        riesgo.promesa_evaluada(0),
        None,
        "RB-V1c: la confianza se juzga ANTES del EV: no hay promesa"
    );
    let cota = 8.0 * f64::EPSILON;
    assert!(
        (g.tp_pct - 0.02).abs() <= cota,
        "TP de la intención: 102 desde 100"
    );
    assert!(
        (g.sl_pct - 0.01).abs() <= cota,
        "SL de la intención: 99 desde 100"
    );
    assert!(g.friccion_pct > 0.0 && g.tau_ms > 0.0);
}

#[test]
fn rb_v1b_un_rechazo_anterior_a_la_geometria_no_publica_ninguna() {
    let _serie = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, intent) = cuenta(13.0);
    let mut riesgo = RiskEngine::new(13.0);
    let orden = riesgo.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        orden.signal,
        SignalType::Long,
        "{}",
        risk_engine::reject_report()
    );
    assert!(riesgo.geometria_evaluada(0).is_some());

    // La moneda 1 no tiene spec: el riesgo la rechaza antes de la geometría y
    // la de la evaluación anterior no puede atribuírsele.
    let orden = riesgo.evaluate_quantum_order(1, &intent, &arena);
    assert_eq!(orden.signal, SignalType::Flat);
    assert_eq!(
        riesgo.geometria_evaluada(1),
        None,
        "sin geometría propia, ninguna"
    );
    assert_eq!(
        riesgo.geometria_evaluada(0),
        None,
        "y la de la evaluación anterior se borró al empezar esta"
    );
    assert_eq!(riesgo.promesa_evaluada(0), None, "la promesa también");

    let plana = SignalIntent {
        signal: SignalType::Flat,
        ..intent
    };
    assert_eq!(
        riesgo.evaluate_quantum_order(0, &plana, &arena).signal,
        SignalType::Flat
    );
    assert_eq!(riesgo.geometria_evaluada(0), None);
}
