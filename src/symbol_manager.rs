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

        // B3.37 — POLO COMPLETO para anclar el universo al ROSTER: 3× limit
        // no alcanza — medido en testnet, los símbolos con modelo validado
        // (ADA 86, NEAR 111, ATOM 162, BNB 182 de 573 por score de momentum)
        // quedaban FUERA de un pool de 78 y la intersección roster∩pool era
        // VACÍA. Se pide TODO el ranking (una sola llamada más de exchangeInfo):
        // el merge asienta al roster donde esté, y el momentum llena el resto.
        let pool_limit = 1000;
        let specs_res =
            data_pipeline::dynamic_ranker::fetch_dynamic_universe(pool_limit, is_testnet).await;

        if let Ok(specs) = specs_res {
            // F4.6 — HISTÉRESIS ANTI-THRASH: un símbolo en el borde (posición
            // N/N+1) no debe entrar/salir cada hora. Regla: los incumbentes
            // que sigan en el top (limit+3) CONSERVAN su asiento; los nuevos
            // solo entran por asientos realmente liberados.
            let current: Vec<String> = quantum_arena::symbols::get_active_universe();
            let raw_top_symbols: Vec<String> = specs.iter().map(|s| s.symbol.clone()).collect();
            let roster = load_model_roster();
            let roster_in_pool: Vec<String> = raw_top_symbols
                .iter()
                .filter(|s| roster.contains(*s))
                .cloned()
                .collect();
            println!(
                "🧬 [SYMBOL MANAGER] B3.37 roster: {} modelos activos · pool {} · roster∩pool {} {:?}",
                roster.len(),
                raw_top_symbols.len(),
                roster_in_pool.len(),
                roster_in_pool
            );
            let top_symbols =
                merge_universe_with_hysteresis(&current, &raw_top_symbols, limit, &roster);

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

/// B3.37 — ROSTER de modelos validados: símbolos con `models/{SYM}_MOTOR.json`
/// activo (los `_SCALP_CANDIDATE.json` NO cuentan — aún no pasaron el gate
/// cross-month). Se lee del FILESYSTEM (no de GLOBAL_FORESTS) porque el
/// daemon rota el universo ANTES de que el motor caliente los modelos:
/// los archivos ya están, la memoria del proceso aún no.
fn load_model_roster() -> std::collections::HashSet<String> {
    let mut roster = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir("models") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with("_MOTOR.json") && !name.contains("_CANDIDATE") {
                let sym = name.trim_end_matches("_MOTOR.json").to_string();
                if !sym.is_empty() {
                    roster.insert(sym);
                }
            }
        }
    }
    roster
}

pub fn merge_universe_with_hysteresis(
    current: &[String],
    top_candidates: &[String],
    limit: usize,
    roster: &std::collections::HashSet<String>,
) -> Vec<String> {
    let margin_set: std::collections::HashSet<&String> = top_candidates
        .iter()
        .take(limit.saturating_add(3))
        .collect();

    let mut merged: Vec<String> = Vec::with_capacity(limit);

    // B3.37 — PRIORIDAD 1: asientos de ROSTER. Un símbolo con modelo
    // validado es el ÚNICO que puede pasar el gate B3.25 (has_roster_model):
    // si la rotación los expulsa, el sistema se queda estructuralmente sin
    // nada operable por bueno que sea su momentum. Orden: por score del
    // escáner (posición en top_candidates) — los que se mueven HOY primero.
    for candidate in top_candidates {
        if merged.len() >= limit {
            break;
        }
        if roster.contains(candidate) && !merged.contains(candidate) {
            merged.push(candidate.clone());
        }
    }

    // F4.6 — PRIORIDAD 2: incumbentes dentro del margen anti-thrash.
    for incumbent in current {
        if merged.len() >= limit {
            break;
        }
        if margin_set.contains(incumbent) && !merged.contains(incumbent) {
            merged.push(incumbent.clone());
        }
    }

    // PRIORIDAD 3: exploradores sin modelo (recolección de datos — el gate
    // B3.25 los mantiene sin permiso de entrada; alimentan al escáner y al
    // futuro entrenamiento).
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
        let mut fallback: Vec<String> = top_candidates.iter().take(limit).cloned().collect();
        for candidate in top_candidates.iter().skip(limit) {
            if fallback.len() >= limit {
                break;
            }
            if roster.contains(candidate) && !fallback.contains(candidate) {
                fallback.push(candidate.clone());
            }
        }
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roster_of(syms: &[&str]) -> std::collections::HashSet<String> {
        syms.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_merge_universe_with_hysteresis() {
        let current = vec![
            "BTCUSDT".to_string(),
            "ETHUSDT".to_string(),
            "SOLUSDT".to_string(),
        ];
        let candidates = vec![
            "BTCUSDT".to_string(),
            "BNBUSDT".to_string(), // new entrant
            "ETHUSDT".to_string(),
            "SOLUSDT".to_string(), // incumbent still in top limit+3
            "XRPUSDT".to_string(),
        ];
        let limit = 3;
        let merged = merge_universe_with_hysteresis(
            &current,
            &candidates,
            limit,
            &roster_of(&["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]),
        );
        assert_eq!(merged.len(), 3);
        assert!(merged.contains(&"BTCUSDT".to_string()));
        assert!(merged.contains(&"ETHUSDT".to_string()));
        assert!(merged.contains(&"SOLUSDT".to_string()));
    }

    /// B3.37 — regresión del caso medido en vivo (v41/v42): el escáner rota
    /// a movers SIN modelo y el roster queda fuera del top-N. El merge debe
    /// rescatar a los modelados aunque no sean top-movers: sin esto, la
    /// intersección universo∩roster queda VACÍA y el gate B3.25 veta toda
    /// entrada (cero trades eterno con modelos sanos).
    #[test]
    fn test_b3_37_roster_symbols_survive_rotation_to_unmodeled_movers() {
        let current: Vec<String> = vec![]; // arranque frío
        let candidates: Vec<String> = [
            "MUSDT", // top movers de testnet: NINGUNO con modelo
            "USELESSUSDT",
            "ACHUSDT",
            "VVVUSDT",
            "ZKUSDT",
            "NEARUSDT", // modelado, hoy callado (puesto 6)
            "BNBUSDT",  // modelado, puesto 7
            "XRPUSDT",  // modelado, puesto 8
        ]
        .iter()
        .map(|s| format!("{}USDT", s))
        .collect();
        let roster = roster_of(&["NEARUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]);
        let limit = 5;

        let merged = merge_universe_with_hysteresis(&current, &candidates, limit, &roster);
        assert_eq!(merged.len(), limit);
        // Los tres modelados del pool entran SÍ o SÍ…
        assert!(merged.contains(&"NEARUSDT".to_string()));
        assert!(merged.contains(&"BNBUSDT".to_string()));
        assert!(merged.contains(&"XRPUSDT".to_string()));
        // …y desplazan a los exploradores sin modelo del fondo del pool.
        assert!(!merged.contains(&"ZKUSDT".to_string()) || !merged.contains(&"VVVUSDT".to_string()));
        // Los primeros puestos del escáner (momentum alto) conservan asiento.
        assert!(merged.contains(&"MUSDT".to_string()));
    }
}
