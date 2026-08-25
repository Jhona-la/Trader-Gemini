use std::time::Duration;

use crate::config::TensorConfig;

pub async fn evolve_symbols_daemon() {
    println!("🌍 [SYMBOL MANAGER] Radar Cuántico Global activado (Modo Eficiente 1H)...");

    // Configure polling interval for 1 hour to save CPU/Network instead of opening a raw firehose WS stream
    let mut interval = tokio::time::interval(Duration::from_secs(3600));

    loop {
        interval.tick().await;

        println!(
            "🔄 [SYMBOL MANAGER] Recalculando Top Dinámico usando métricas reales (REST API)..."
        );
        let limit = quantum_arena::symbols::get_active_universe_size();
        // FIX #1421: Determinar entorno dinámicamente según variables de configuración
        let is_testnet = crate::env_manager::EnvManager::is_demo_env();

        let specs_res =
            data_pipeline::dynamic_ranker::fetch_dynamic_universe(limit, is_testnet).await;

        if let Ok(specs) = specs_res {
            // F4.6 — HISTÉRESIS ANTI-THRASH: un símbolo en el borde (posición
            // N/N+1) no debe entrar/salir cada hora. Regla: los incumbentes
            // que sigan en el top (limit+3) CONSERVAN su asiento; los nuevos
            // solo entran por asientos realmente liberados.
            let current: Vec<String> = quantum_arena::symbols::get_active_universe();
            let raw_top_symbols: Vec<String> = specs.iter().map(|s| s.symbol.clone()).collect();
            let top_symbols = merge_universe_with_hysteresis(&current, &raw_top_symbols, limit);

            let is_testnet_env = crate::env_manager::EnvManager::is_demo_env();
            let config_bytes = tokio::fs::read("data/dynamic_config.bin")
                .await
                .unwrap_or_default();
            let mut config: TensorConfig = match bincode::deserialize(&config_bytes) {
                Ok(cfg) => cfg,
                Err(_) => {
                    let config_str = tokio::fs::read_to_string("data/dynamic_config.json")
                        .await
                        .unwrap_or_else(|_| "".to_string());
                    // FIX #1460: Fallback coherente con el entorno de ejecución activo
                    serde_json::from_str(&config_str).unwrap_or_else(|_| TensorConfig {
                        symbols: top_symbols.clone(),
                        is_testnet: is_testnet_env,
                    })
                }
            };

            let mut changed = false;
            if config.symbols.len() != top_symbols.len() {
                changed = true;
            } else {
                for (a, b) in config.symbols.iter().zip(top_symbols.iter()) {
                    if a != b {
                        changed = true;
                        break;
                    }
                }
            }

            if changed {
                println!(
                    "🔄 [SYMBOL MANAGER] Cambio de Régimen! Nuevos símbolos detectados: {:?}",
                    top_symbols
                );
                config.symbols = top_symbols.clone();
                if let Ok(encoded) = bincode::serialize(&config) {
                    let _ = tokio::fs::write("data/dynamic_config.bin", encoded).await;
                    if let Ok(json_str) = serde_json::to_string_pretty(&config) {
                        let _ = tokio::fs::write("data/dynamic_config.json", json_str).await;
                    }
                }
                // F4.6 — ACTUALIZACIÓN EN VIVO: antes solo se reescribían los
                // archivos y el motor seguía con el universo VIEJO hasta
                // reiniciar (cambio fantasma). Ahora:
                // 1) el universo dinámico del arena se actualiza al instante;
                // 2) los SymbolSpec (tick/step/minNotional REALES de cada
                //    símbolo nuevo) entran al registro — sin esto, ejecutar un
                //    símbolo nuevo usaba defaults con riesgo de -4014/-4164.
                quantum_arena::symbols::update_dynamic_universe(top_symbols.clone());
                quantum_arena::symbol_registry::update_registry(specs);
                println!(
                    "✅ [SYMBOL MANAGER] Universo vivo + registro de specs actualizados ({} símbolos)",
                    top_symbols.len()
                );
            }
        } else if let Err(e) = specs_res {
            println!(
                "⚠️ [SYMBOL MANAGER] Fallo al recuperar universo dinámico: {}",
                e
            );
        }
    }
}

pub fn merge_universe_with_hysteresis(current: &[String], top_candidates: &[String], limit: usize) -> Vec<String> {
    let margin_set: std::collections::HashSet<&String> =
        top_candidates.iter().take(limit.saturating_add(3)).collect();

    let mut merged: Vec<String> = Vec::with_capacity(limit);
    for incumbent in current {
        if merged.len() >= limit {
            break;
        }
        if margin_set.contains(incumbent) {
            merged.push(incumbent.clone());
        }
    }
    for candidate in top_candidates {
        if merged.len() >= limit {
            break;
        }
        if !merged.contains(candidate) {
            merged.push(candidate.clone());
        }
    }
    if merged.len() == limit {
        merged
    } else {
        top_candidates.iter().take(limit).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_universe_with_hysteresis() {
        let current = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string(), "SOLUSDT".to_string()];
        let candidates = vec![
            "BTCUSDT".to_string(),
            "BNBUSDT".to_string(), // new entrant
            "ETHUSDT".to_string(),
            "SOLUSDT".to_string(), // incumbent still in top limit+3
            "XRPUSDT".to_string(),
        ];
        let limit = 3;
        let merged = merge_universe_with_hysteresis(&current, &candidates, limit);
        assert_eq!(merged.len(), 3);
        assert!(merged.contains(&"BTCUSDT".to_string()));
        assert!(merged.contains(&"ETHUSDT".to_string()));
        assert!(merged.contains(&"SOLUSDT".to_string()));
    }
}
