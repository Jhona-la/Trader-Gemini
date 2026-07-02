use std::sync::Arc;
use std::sync::atomic::Ordering;
use god_engine_core::{GodEngineCore, GlobalArena};

fn main() {
    let _ = trader_gemini_v5::telemetry_server::init_telemetry(8000); // init random stuff just in case
    
    // Load config
    let mut arena = GlobalArena::new();
    arena.config.scalp_kelly_fraction.store(0.2, Ordering::Relaxed);
    let arena_arc = Arc::new(arena);
    
    // Load NanoForest
    let path = "models/BTCUSDT_SCALP.json";
    if let Err(e) = god_engine_core::ml_inference::NanoForest::load_global("BTCUSDT_SCALP", path) {
        println!("Error loading NanoForest: {}", e);
        return;
    }
    
    let mut core = GodEngineCore::new(arena_arc.clone());
    
    // Feed it some fake prices to see if atr_pct and v_t evolve
    println!("Testing feature engine and NanoForest...");
    let mut price = 60000.0;
    
    for i in 0..100 {
        price += (i as f64 % 5.0) * 10.0 - 20.0; // random walk
        let bid = price - 0.1;
        let ask = price + 0.1;
        
        // Pass event
        core.process_event(0, bid, ask, 1.0, 1000 + i);
        
        let features = core.feature_engines[0].get_features();
        let ml_prob = core.scalp_forest.as_ref().map(|f| f.predict(&features) as f64).unwrap_or(0.5);
        let v_t = features[4];
        let atr_pct = features[5];
        
        if i % 10 == 0 {
            println!("Tick {}: Price={:.2} ml_prob={:.4} v_t={:.4} atr_pct={:.6}", i, price, ml_prob, v_t, atr_pct);
        }
    }
}
