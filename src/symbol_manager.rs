use std::time::Duration;

use crate::config::TensorConfig;

pub async fn evolve_symbols_daemon() {
    println!("🌍 [SYMBOL MANAGER] Radar Cuántico Global activado (Modo Eficiente 1H)...");
    
    // Configure polling interval for 1 hour to save CPU/Network instead of opening a raw firehose WS stream
    let mut interval = tokio::time::interval(Duration::from_secs(3600));
    
    loop {
        interval.tick().await;
        
        println!("🔄 [SYMBOL MANAGER] Recalculando Top Dinámico usando métricas reales (REST API)...");
        let limit = quantum_arena::symbols::get_active_universe_size();
        let is_testnet = false;
        
        let specs_res = data_pipeline::dynamic_ranker::fetch_dynamic_universe(limit, is_testnet).await;
        
        if let Ok(specs) = specs_res {
            let top_symbols: Vec<String> = specs.into_iter().map(|s| s.symbol).collect();
            let config_bytes = tokio::fs::read("data/dynamic_config.bin").await.unwrap_or_default();
            let mut config: TensorConfig = match bincode::deserialize(&config_bytes) {
                Ok(cfg) => cfg,
                Err(_) => {
                    let config_str = tokio::fs::read_to_string("data/dynamic_config.json").await.unwrap_or_else(|_| "".to_string());
                    serde_json::from_str(&config_str).unwrap_or_else(|_| TensorConfig {
                        symbols: top_symbols.clone(),
                        is_testnet: false,
                    })
                }
            };

            let mut changed = false;
            if config.symbols.len() != top_symbols.len() { changed = true; }
            else {
                for (a, b) in config.symbols.iter().zip(top_symbols.iter()) {
                    if a != b { changed = true; break; }
                }
            }

            if changed {
                println!("🔄 [SYMBOL MANAGER] Cambio de Régimen! Nuevos símbolos detectados: {:?}", top_symbols);
                config.symbols = top_symbols.clone();
                if let Ok(encoded) = bincode::serialize(&config) {
                    let _ = tokio::fs::write("data/dynamic_config.bin", encoded).await;
                    if let Ok(json_str) = serde_json::to_string_pretty(&config) {
                        let _ = tokio::fs::write("data/dynamic_config.json", json_str).await;
                    }
                }
            }
        } else if let Err(e) = specs_res {
            println!("⚠️ [SYMBOL MANAGER] Fallo al recuperar universo dinámico: {}", e);
        }
    }
}
