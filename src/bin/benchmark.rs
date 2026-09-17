use execution_engine::executor::OrderExecutor;
use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType, TradeHorizon};
use std::env;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("===========================================================");
    println!("🔬 TRADER GEMINI V7 - QUANTUM PROFILER (LABORATORY)");
    println!("===========================================================");

    let initial_capital: f64 = std::env::var("INITIAL_CAPITAL")
        .unwrap_or_else(|_| "13.0".to_string())
        .parse()
        .unwrap_or(13.0);
    let safe_capital = if initial_capital > 0.0 && initial_capital.is_finite() {
        initial_capital
    } else {
        13.0
    };
    // D-684 (DÉCIMA OLA): el arena se construye en un hilo con la MISMA pila que
    // producción (`god_engine`, 32 MiB). `CoinArena` lleva el búfer del ring de
    // ticks en línea y `GlobalArena::build` materializa temporales de ese tamaño:
    // su necesidad de pila depende de cómo el optimizador inlinee la construcción,
    // y con 1 MiB (hilo principal en Windows) un cambio de una línea en `build`
    // bastó para desbordarla. El evolver y el simulador ya seguían este patrón.
    let arena = std::thread::Builder::new()
        .name("arena-build".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(move || GlobalArena::new(safe_capital))
        .expect("no se pudo crear el hilo de construcción del arena")
        .join()
        .expect("la construcción del arena entró en pánico");
    arena.config.global_leverage.store(10.0, Ordering::Relaxed);
    arena
        .config
        .global_max_drawdown
        .store(0.15, Ordering::Relaxed);

    let arena_ptr = Arc::new(arena);
    // FIX #1526: Inicializar RiskEngine con safe_capital validado
    let mut risk_engine = RiskEngine::new(safe_capital);
    let api_key = "LAB_DUMMY_KEY".to_string();
    let is_testnet = env::var("BINANCE_IS_TESTNET")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false);
    let _executor = OrderExecutor::new(api_key, "LAB_DUMMY_SECRET_KEY".to_string(), is_testnet);

    let _symbol = "bnbusdt";
    let iterations = 1_000_000;

    println!(
        "Iniciando simulación de {} Ticks del mercado (Hot Path)...",
        iterations
    );

    let start_time = Instant::now();

    for i in 0..iterations {
        // Simular fluctuación del mercado
        let _fake_price = 600.0 + (i as f64 % 10.0);
        let fake_bid_vol = 100.0 + (i as f64 % 5.0);
        let fake_ask_vol = 98.0 + (i as f64 % 7.0);

        // --- INICIO DEL HOT PATH ---

        let obi = (fake_bid_vol - fake_ask_vol) / (fake_bid_vol + fake_ask_vol);
        let scalp_intent = if obi > 0.20 {
            SignalIntent {
                signal: SignalType::Long,
                confidence: obi.abs(),
                horizon: TradeHorizon::Continuous,
                ..Default::default()
            }
        } else if obi < -0.20 {
            SignalIntent {
                signal: SignalType::Short,
                confidence: obi.abs(),
                horizon: TradeHorizon::Continuous,
                ..Default::default()
            }
        } else {
            SignalIntent::flat()
        };

        if scalp_intent.signal != SignalType::Flat {
            // coin_id for bnbusdt is 2
            let _validated_order = risk_engine.evaluate_quantum_order(2, &scalp_intent, &arena_ptr);
        }

        // --- FIN DEL HOT PATH ---
    }

    let elapsed = start_time.elapsed();
    let total_nanos = elapsed.as_nanos();
    let nanos_per_tick = total_nanos / (iterations as u128);

    println!("\n✅ RESULTADOS DEL BENCHMARK:");
    println!("Tiempo total para {} Ticks: {:?}", iterations, elapsed);
    println!(
        "Latencia Promedio por Tick: {} nanosegundos",
        nanos_per_tick
    );

    if nanos_per_tick < 1000 {
        println!("🚀 VEREDICTO: RANGO NANO-SEGUNDOS. Rendimiento cuántico confirmado.");
    } else if nanos_per_tick < 1_000_000 {
        println!("⚠️ VEREDICTO: RANGO MICRO-SEGUNDOS. Aceptable pero optimizable.");
    } else {
        println!("❌ VEREDICTO: RANGO MILI-SEGUNDOS. ALERTA TERMODINÁMICA. Peligro HFT.");
    }

    println!("===========================================================");
}
