//! Sonda B2.5: reproduce el camino EXACTO de inferencia en vivo para
//! BTCUSDT_MOTOR y diagnostica por qué el motor vivo ml=0.5000 constante.
use god_engine_core::stateful_engine::StatefulEngine;

#[tokio::main]
async fn main() {
    // 1. Cargar como el motor (orden real del filesystem)
    println!("── load_global json:");
    match god_engine_core::ml_inference::NanoForest::load_global("BTCUSDT_MOTOR", "models/BTCUSDT_MOTOR.json") {
        Ok(()) => println!("   ok"),
        Err(e) => println!("   ❌ {}", e),
    }
    let g = god_engine_core::ml_inference::NanoForest::get_global("BTCUSDT_MOTOR");
    println!("   get_global tras json: {}", g.is_some());

    // 2. Motor de features con ticks REALES del archivo de julio (mmap):
    // los sintéticos daban todo finito; si el vivo falla, la diferencia
    // está en los datos reales.
    let f = std::fs::File::open("data/BTCUSDT_ticks_REAL.bin").expect("abrir ticks");
    let mmap = unsafe { memmap2::MmapOptions::new().map(&f) }.expect("mmap");
    let header_off = if mmap.len() >= 8 && &mmap[..8] == b"TGMTICK1" { 8usize } else { 0 };
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct BinTick { ts: u64, bid: f64, ask: f64, bq: f64, aq: f64 }
    let n = (mmap.len() - header_off) / std::mem::size_of::<BinTick>();
    let raw = unsafe {
        std::slice::from_raw_parts(mmap.as_ptr().add(header_off) as *const BinTick, n)
    };
    let use_n = n.min(50_000);
    println!("── replay de {} ticks reales de julio", use_n);
    let mut engine = StatefulEngine::new();
    let mut preds: Vec<f32> = Vec::new();
    let mut none_count = 0usize;
    let mut bad_idx_hits = [0usize; 44];
    for i in 0..use_n {
        let t = &raw[i];
        if t.bid <= 0.0 || t.ask <= 0.0 || t.bid > t.ask { continue; }
        let mid = (t.bid + t.ask) / 2.0;
        let vol = t.bq + t.aq;
        engine.process_tick(mid, vol, t.ts);
        engine.update_trade_flow(vol, t.bq > t.aq);
        let _ = engine.update_ofi(t.bid, t.ask, t.bq, t.aq);
        if i % 137 == 0 && i > 200 {
            let swing = engine.get_universal_features();
            let spectral = engine.get_spectral_ml_features();
            let mut input = [0f32; 44];
            input[..34].copy_from_slice(&swing);
            input[34..].copy_from_slice(&spectral);
            if let Some(fr) = god_engine_core::ml_inference::NanoForest::get_global("BTCUSDT_MOTOR") {
                match fr.predict(&input) {
                    Some(p) => preds.push(p),
                    None => {
                        none_count += 1;
                        for (k, v) in input.iter().enumerate() {
                            if !v.is_finite() { bad_idx_hits[k] += 1; }
                        }
                    }
                }
            }
        }
    }
    if none_count > 0 {
        let bad: Vec<String> = bad_idx_hits.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(k, c)| format!("[{}]={}", k, c)).collect();
        println!("   predict=None ×{} — índices no finitos: {:?}", none_count, bad);
    }
    println!("── predicciones: {}", preds.len());
    if !preds.is_empty() {
        let mn = preds.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = preds.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("   min={:.4} max={:.4} últimas: {:?}", mn, mx, &preds[preds.len().saturating_sub(5)..]);
    }
}
