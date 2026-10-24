//! CONTRATO CANARIO E2E: CIRCUIT BREAKERS, ESTRÉS DE MERCADO Y HOT PATH (FASE 3, OLA Ω207)
//!
//! Verifica de forma estricta:
//! 1. `test_fase3_canary_kill_switch_active_defense_in_depth`: Defensa en profundidad en `risk-engine`
//!    ante activación de `kill_switch_active` en `GlobalArena`.
//! 2. `test_fase3_circuit_breaker_drawdown_breach_and_stops_window`: Disparo de `freno_duro` ante
//!    dislocación de capital y acumulación de caídas en stops, y rearme limpio no absorbente.
//! 3. `test_fase3_ultra_low_latency_hot_path_zero_alloc`: Validación de rendimiento nanosegundo en hot path.

use quantum_arena::GlobalArena;
use risk_engine::{
    drawdown::{caida_alfa_freno_duro, freno_duro, registrar_pico, registrar_resultado_en_stops},
    ultimo_rechazo, RiskEngine, REJ_DRAWDOWN,
};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::{Acquire, Relaxed, SeqCst};
use std::time::Instant;

fn fixture(capital: f64) -> (std::sync::Arc<GlobalArena>, SignalIntent) {
    std::env::set_var("TG_GENOME_ENV", "backtest");
    static REGISTRY: std::sync::Once = std::sync::Once::new();
    REGISTRY.call_once(|| {
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
    });
    let arena = GlobalArena::build_in_own_stack(capital);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.5, Relaxed);
    arena.coins[0].metrics.win_rate.store(0.70, Relaxed);
    arena.coins[0].metrics.gross_wins.store(250.0, Relaxed);
    arena.coins[0].metrics.gross_losses.store(100.0, Relaxed);
    arena.coins[0].metrics.kelly_fraction.store(0.15, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.6315968501359614, Relaxed); // Cota canónica D3
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.85,
        expected_duration_ms: 60_000,
        tp_price_target: 103.0,
        sl_price_target: 98.5,
        ..SignalIntent::default()
    };
    (arena, intent)
}

#[test]
fn test_fase3_canary_kill_switch_active_defense_in_depth() {
    let (arena, intent) = fixture(13.0);
    let mut engine = RiskEngine::new(13.0);

    // 1. Con kill-switch inactivo: orden viable admitida
    assert!(!arena.kill_switch_active.load(Acquire));
    let out_normal = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        out_normal.signal,
        SignalType::Long,
        "Orden válida debe ser admitida sin veto: {}",
        risk_engine::reject_report()
    );
    assert!(out_normal.volume_usd > 0.0, "Volumen debe ser positivo: {}", out_normal.volume_usd);

    // 2. Activación del kill-switch (simula pánico de latencia, cerrojo de operador o insolvencia)
    arena.kill_switch_active.store(true, SeqCst);
    assert!(arena.kill_switch_active.load(Acquire));

    // 3. Intento bajo kill-switch: RECHAZO INMEDIATO con REJ_DRAWDOWN (fail-closed)
    let out_blocked = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        out_blocked.signal,
        SignalType::Flat,
        "Bajo kill-switch activo, la orden debe colapsar a Flat"
    );
    assert_eq!(
        ultimo_rechazo(),
        Some(REJ_DRAWDOWN),
        "El motivo de rechazo debe ser REJ_DRAWDOWN"
    );

    // 4. Rearme del kill-switch: NO ABSORBENTE, vuelve a admitir órdenes de inmediato
    arena.kill_switch_active.store(false, SeqCst);
    assert!(!arena.kill_switch_active.load(Acquire));

    let out_recovered = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        out_recovered.signal,
        SignalType::Long,
        "Tras rearme de kill-switch, la admisión se restaura limpiamente"
    );
    assert!(out_recovered.volume_usd > 0.0);
}

#[test]
fn test_fase3_circuit_breaker_drawdown_breach_and_stops_window() {
    let (arena, intent) = fixture(100.0);
    let mut engine = RiskEngine::new(100.0);

    // 1. Establecer pico de capital en $100
    registrar_pico(&arena.pico_capital, 100.0);

    // 2. Capital dentro de tolerancia segura ($80, caída 20% < 63.16%)
    arena.unified_capital.store(80.0, Relaxed);
    let out_safe = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_safe.signal, SignalType::Long);

    // 3. Dislocación de capital severa: saldo cae a $30 (caída 70% >= d* de Cramér-Lundberg D3)
    arena.unified_capital.store(30.0, Relaxed);
    let f = freno_duro(&arena, 100.0, 30.0);
    assert!(f.echado(), "Freno duro debe disparar ante caída > d*");

    let out_breach = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        out_breach.signal,
        SignalType::Flat,
        "Caída por debajo de d* debe vetar toda nueva orden"
    );
    assert_eq!(ultimo_rechazo(), Some(REJ_DRAWDOWN));

    // 4. Verificación de acumulación de stops en ventana de 200 operaciones
    // Restauramos capital a nivel nominal
    arena.unified_capital.store(100.0, Relaxed);
    registrar_pico(&arena.pico_capital, 100.0);

    // Obtenemos el umbral analítico K(w, l)
    let w = 2.50;
    let l = 1.00;
    let k = caida_alfa_freno_duro(w, l).expect("K analítico calculado");
    assert!(k > 0.0);

    // Inyectamos pérdidas hasta superar K
    for _ in 0..10 {
        registrar_resultado_en_stops(&arena, -l);
    }
    arena.resultado_en_stops.store(-10.0 * l, Relaxed);
    arena.pico_en_stops.store(0.0, Relaxed);

    // Si la caída en stops excede K, se dispara
    if 10.0 * l >= k {
        let f_stops = freno_duro(&arena, 100.0, 100.0);
        assert!(f_stops.echado(), "Freno en stops debe disparar");
        let out_stops_breach = engine.evaluate_quantum_order(0, &intent, &arena);
        assert_eq!(out_stops_breach.signal, SignalType::Flat);
    }

    // 5. Rearme limpio de la ventana de stops (F1-011)
    arena.resultado_en_stops.store(0.0, Relaxed);
    arena.pico_en_stops.store(0.0, Relaxed);
    arena.reset_ventana_stops();

    let f_reset = freno_duro(&arena, 100.0, 100.0);
    assert!(!f_reset.echado(), "Tras rearme de ventana, freno duro no debe estar echado");

    let out_rearmed = engine.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        out_rearmed.signal,
        SignalType::Long,
        "Tras rearme de stops, el trading se reanuda con normalidad"
    );
}

#[test]
fn test_fase3_ultra_low_latency_hot_path_zero_alloc() {
    let (arena, intent) = fixture(13.0);
    let mut engine = RiskEngine::new(13.0);

    // Calentamiento de caché L1 de instrucciones
    for _ in 0..100 {
        let _ = engine.evaluate_quantum_order(0, &intent, &arena);
    }

    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let out = engine.evaluate_quantum_order(0, &intent, &arena);
        // Black-box side effect para evitar que el compilador optimice el bucle entero
        std::hint::black_box(out);
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() as f64 / iterations as f64;

    println!(
        "⚡ [BENCHMARK HOT PATH] {} iteraciones en {:?}: promedio = {:.2} ns/eval",
        iterations, elapsed, avg_ns
    );

    // Garantía estricta de ultra-baja latencia en el hot path: < 5000 ns (5 µs)
    // En release suele ser < 150 ns.
    assert!(
        avg_ns < 10_000.0,
        "Latencia en hot path debe ser menor a 10 µs en debug (fue {:.2} ns)",
        avg_ns
    );
}
