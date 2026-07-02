use execution_engine::executor::OrderExecutor;
use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{ScalpEngine, SwingEngine, SignalType};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("===========================================================");
    println!("🔬 TRADER GEMINI V5 - QUANTUM PROFILER (LABORATORY)");
    println!("===========================================================");

    let initial_capital = 13.0;
    let arena = GlobalArena::new(initial_capital);
    arena.config.global_leverage.store(10.0, Ordering::Relaxed);
    arena.config.global_max_drawdown.store(0.15, Ordering::Relaxed);

    let arena_ptr = Arc::new(arena);
    let mut scalp_engine = ScalpEngine::new();
    let _swing_engine = SwingEngine::default();
    let mut risk_engine = RiskEngine::new(initial_capital);
    let api_key = "LAB_DUMMY_KEY".to_string();
    let api_secret = "LAB_DUMMY_SECRET_KEY".to_string();
    let _executor = OrderExecutor::new(api_key, api_secret);

    let _symbol = "bnbusdt";
    let iterations = 1_000_000;
    
    println!("Iniciando simulación de {} Ticks del mercado (Hot Path)...", iterations);

    let start_time = Instant::now();

    for i in 0..iterations {
        // Simular fluctuación del mercado
        let _fake_price = 600.0 + (i as f64 % 10.0);
        let fake_bid_vol = 100.0 + (i as f64 % 5.0);
        let fake_ask_vol = 98.0 + (i as f64 % 7.0);
        
        // --- INICIO DEL HOT PATH ---
        
        let obi_threshold = arena_ptr.config.scalp_obi_threshold.load(Ordering::Relaxed);
        let scalp_intent = scalp_engine.evaluate_microstructure(fake_bid_vol, fake_ask_vol, obi_threshold);
        
        if scalp_intent.signal != SignalType::Flat {
            // coin_id for bnbusdt is 2
            let _validated_order = risk_engine.evaluate_order(2, scalp_intent, signal_engine::SignalIntent::flat(), &arena_ptr);
        }
        
        // --- FIN DEL HOT PATH ---
    }

    let elapsed = start_time.elapsed();
    let total_nanos = elapsed.as_nanos();
    let nanos_per_tick = total_nanos / (iterations as u128);

    println!("\n✅ RESULTADOS DEL BENCHMARK:");
    println!("Tiempo total para {} Ticks: {:?}", iterations, elapsed);
    println!("Latencia Promedio por Tick: {} nanosegundos", nanos_per_tick);
    
    if nanos_per_tick < 1000 {
        println!("🚀 VEREDICTO: RANGO NANO-SEGUNDOS. Rendimiento cuántico confirmado.");
    } else if nanos_per_tick < 1_000_000 {
        println!("⚠️ VEREDICTO: RANGO MICRO-SEGUNDOS. Aceptable pero optimizable.");
    } else {
        println!("❌ VEREDICTO: RANGO MILI-SEGUNDOS. ALERTA TERMODINÁMICA. Peligro HFT.");
    }
    
    println!("===========================================================");
}
