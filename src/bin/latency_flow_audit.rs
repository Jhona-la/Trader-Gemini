use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("🚀 GOD ENGINE - DEEP LATENCY FLOW AUDIT");
    println!("=======================================");
    println!("Initialize Isolated Execution Environment...");

    // Create Arena
    // D-684 (DÉCIMA OLA): el arena se construye en un hilo con la MISMA pila que
    // producción (`god_engine`, 32 MiB). `CoinArena` lleva el búfer del ring de
    // ticks en línea y `GlobalArena::build` materializa temporales de ese tamaño:
    // su necesidad de pila depende de cómo el optimizador inlinee la construcción,
    // y con 1 MiB (hilo principal en Windows) un cambio de una línea en `build`
    // bastó para desbordarla. El evolver y el simulador ya seguían este patrón.
    // El capital lo inyecta la API más tarde.
    let arena = std::thread::Builder::new()
        .name("arena-build".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(move || Arc::new(GlobalArena::new(0.0)))
        .expect("no se pudo crear el hilo de construcción del arena")
        .join()
        .expect("la construcción del arena entró en pánico");
    let mut engine = GodEngineCore::new(Arc::clone(&arena));

    let total_ticks = 1_000_000;
    println!(
        "Feeding {} Synthetic L1 Ticks (Bid/Ask Updates) into GodEngineCore...",
        total_ticks
    );

    let mut latencies_ns = Vec::with_capacity(total_ticks);
    let omni_features = [0.0; 54]; // Mock dummy features

    let mut bid = 60000.0;
    let mut ask = 60000.1;
    let mut bid_qty = 1.0;
    let mut ask_qty = 1.0;

    // Warmup round (JIT compilation and cache warmup)
    for i in 0..10_000 {
        engine.process_event(
            0,       // coin_id = 0 (BTC)
            false,   // is_trade
            false,   // is_kline_closed
            true,    // is_depth
            60000.0, // current_price
            0.0,     // _trade_qty
            bid,
            ask,
            bid_qty,
            ask_qty,
            0.1,      // depth_obi
            0.0,      // depth_micro_div
            i as u64, // event_time_ms
            false,    // latency_panic
            &omni_features,
            false, // is_buyer_maker
        );
        bid += 0.01;
        ask += 0.01;
    }

    // Real Benchmark Loop
    for i in 0..total_ticks {
        let start = Instant::now();

        let _result = engine.process_event(
            0,       // coin_id = 0 (BTC)
            false,   // is_trade
            false,   // is_kline_closed
            true,    // is_depth
            60000.0, // current_price
            0.0,     // _trade_qty
            bid,
            ask,
            bid_qty,
            ask_qty,
            0.1,                // depth_obi
            0.0,                // depth_micro_div
            (i as u64) + 10000, // event_time_ms
            false,              // latency_panic
            &omni_features,
            false, // is_buyer_maker
        );

        latencies_ns.push(start.elapsed().as_nanos());

        // Random walk for dynamic state
        bid += 0.01;
        ask += 0.01;
        if bid > 70000.0 {
            bid = 60000.0;
            ask = 60000.1;
        }
        bid_qty = if i % 2 == 0 { 2.0 } else { 1.5 };
        ask_qty = if i % 3 == 0 { 3.0 } else { 1.0 };
    }

    // Sort for percentiles
    latencies_ns.sort_unstable();

    let total_time_ns: u128 = latencies_ns.iter().sum();
    let avg_ns = total_time_ns / (total_ticks as u128);
    let p50 = latencies_ns[(total_ticks as f64 * 0.50) as usize];
    let p90 = latencies_ns[(total_ticks as f64 * 0.90) as usize];
    let p99 = latencies_ns[(total_ticks as f64 * 0.99) as usize];
    let p99_9 = latencies_ns[(total_ticks as f64 * 0.999) as usize];
    let max = latencies_ns[total_ticks - 1];
    let min = latencies_ns[0];

    println!("---------------------------------------");
    println!("📊 LATENCY RESULTS (Nanoseconds):");
    println!("Total Ticks Processed : {}", total_ticks);
    println!("Average Latency       : {} ns", avg_ns);
    println!("Minimum Latency       : {} ns", min);
    println!("Median (P50) Latency  : {} ns", p50);
    println!("90th Pctl Latency     : {} ns", p90);
    println!("99th Pctl Latency     : {} ns", p99);
    println!("99.9th Pctl Latency   : {} ns", p99_9);
    println!("Maximum Jitter        : {} ns", max);
    println!("---------------------------------------");

    if p99 < 10000 {
        println!("✅ AUDIT PASSED: System meets hyper-efficiency requirements (<10us P99).");
    } else {
        println!(
            "⚠️ AUDIT WARNING: Potential bottleneck detected at P99 ({} ns).",
            p99
        );
    }
}
