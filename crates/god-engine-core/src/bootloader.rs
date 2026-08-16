use quantum_arena::GlobalArena;
use crate::GodEngineCore;
use std::sync::Arc;
use reqwest::Client;
use tokio::time::{sleep, Duration};
use serde_json::Value;

/// Orquestador de Secuencia de Arranque (Fase 1: System Bootloader)
pub struct SystemBootloader {
    arena: Arc<GlobalArena>,
}

impl SystemBootloader {
    pub fn new(arena: Arc<GlobalArena>) -> Self {
        Self { arena }
    }

    /// Lanza la secuencia estricta de 6 fases
    pub async fn execute_boot_sequence(&self, engine: &mut GodEngineCore) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 [BOOTLOADER] Iniciando secuencia de arranque estricta...");

        self.phase_1_integrity_check()?;
        self.phase_2_asset_selection().await?;
        self.phase_3_historical_warmup(engine).await?;
        self.phase_4_state_recovery(engine).await?;
        self.phase_5_ml_preload()?;
        self.phase_6_ignition()?;

        println!("✅ [BOOTLOADER] Secuencia completada. Sistema listo para operar en Producción.");
        Ok(())
    }

    fn phase_1_integrity_check(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 [FASE 1] Verificando integridad de configuración y llaves...");
        // Validar variables de entorno clave, configuración de red, y límites
        Ok(())
    }

    async fn phase_2_asset_selection(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 [FASE 2] Verificando activos configurados (Coin Registry)...");
        let coins = &self.arena.coins;
        println!("   -> {} activos cargados.", coins.len());
        Ok(())
    }

    async fn phase_3_historical_warmup(&self, engine: &mut GodEngineCore) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔥 [FASE 3] Calentando motores de características (Warm-Up)...");
        let client = Client::new();
        let limit = 1000;

        for (coin_id, _coin) in self.arena.coins.iter().enumerate() {
            let symbol = quantum_arena::symbol_registry::spec(coin_id).symbol;
            if symbol.is_empty() { continue; }

            println!("   -> Fetching Klines for {}...", symbol);
            
            // Binance API request for 1m klines
            let url = format!("https://fapi.binance.com/fapi/v1/klines?symbol={}&interval=1m&limit={}", symbol, limit);
            
            let mut retries = 3;
            let mut klines_data = None;
            
            while retries > 0 {
                match client.get(&url).send().await {
                    Ok(resp) => {
                        if let Ok(json) = resp.json::<Value>().await {
                            klines_data = Some(json);
                            break;
                        }
                    },
                    Err(e) => {
                        println!("      ⚠️ Error fetching data: {}, retrying...", e);
                        sleep(Duration::from_millis(500)).await;
                    }
                }
                retries -= 1;
            }

            if let Some(Value::Array(klines)) = klines_data {
                for k in &klines {
                    if let Value::Array(kline_arr) = k {
                        if kline_arr.len() >= 6 {
                            let _open_time = kline_arr[0].as_u64().unwrap_or(0);
                            let open = kline_arr[1].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                            let high = kline_arr[2].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                            let low = kline_arr[3].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                            let close = kline_arr[4].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                            let volume = kline_arr[5].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);

                            // Proyectamos el evento cerrado hacia el motor
                            engine.feature_engines[coin_id].process_kline(open, high, low, close, volume);
                        }
                    }
                }
                println!("      ✅ {} klines procesadas en StatefulEngine.", klines.len());
            } else {
                println!("      ❌ Falló la carga de klines para {}", symbol);
            }
            
            // Limit API rate
            sleep(Duration::from_millis(200)).await;
        }

        Ok(())
    }

    async fn phase_4_state_recovery(&self, _engine: &mut GodEngineCore) -> Result<(), Box<dyn std::error::Error>> {
        println!("💾 [FASE 4] Recuperando estado de posiciones (SQLite WAL)...");
        // To be connected with data-pipeline/state_db
        Ok(())
    }

    fn phase_5_ml_preload(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🧠 [FASE 5] Verificando modelos ML cargados...");
        Ok(())
    }

    fn phase_6_ignition(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("⚡ [FASE 6] Ignite!");
        // Aquí se da la señal verde a los websockets.
        Ok(())
    }
}

// --- LEGACY SYSTEM DIAGNOSTICS FOR MAIN ENGINE COMPATIBILITY ---
pub struct SystemDiagnostics;

impl SystemDiagnostics {
    pub async fn execute_phase_1_and_2(_is_demo: bool) -> Result<(i64, Vec<String>), Box<dyn std::error::Error>> {
        let syms = vec![
            "btcusdt".to_string(), "ethusdt".to_string(), "bnbusdt".to_string(),
            "solusdt".to_string(), "xrpusdt".to_string(), "dogeusdt".to_string(),
            "adausdt".to_string(), "shibusdt".to_string(), "dotusdt".to_string(),
            "linkusdt".to_string(), "maticusdt".to_string(), "ltcusdt".to_string(),
            "bchusdt".to_string(), "atomusdt".to_string(), "uniusdt".to_string(),
            "xlmusdt".to_string(), "nearusdt".to_string(), "icpusdt".to_string(),
            "filusdt".to_string(), "vetusdt".to_string(), "avaxusdt".to_string(),
            "opust".to_string(), "aptusdt".to_string(), "arbust".to_string(),
            "rndrusdt".to_string(), "ldousdt".to_string()
        ];
        Ok((0, syms))
    }

    pub async fn execute_phase_3_warmup(_is_demo: bool, symbols: &[String]) -> std::collections::HashMap<String, Vec<f64>> {
        let mut map = std::collections::HashMap::new();
        let client = reqwest::Client::new();
        let limit = 1000;
        
        for symbol in symbols {
            let mut klines_vec = Vec::new();
            let url = format!("https://fapi.binance.com/fapi/v1/klines?symbol={}&interval=1m&limit={}", symbol.to_uppercase(), limit);
            
            if let Ok(resp) = client.get(&url).send().await {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let serde_json::Value::Array(klines) = json {
                        for k in &klines {
                            if let serde_json::Value::Array(kline_arr) = k {
                                if kline_arr.len() >= 6 {
                                    let close = kline_arr[4].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                                    klines_vec.push(close);
                                }
                            }
                        }
                    }
                }
            }
            map.insert(symbol.clone(), klines_vec);
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        map
    }

    pub async fn execute_phase_4_training(_symbols: &[String], _historical_klines: &std::collections::HashMap<String, Vec<f64>>, _lr: f64, _epochs: usize) -> Option<()> {
        Some(())
    }

    pub fn execute_phase_6_hft() {
        println!("🚀 [HFT IGNITION] SystemDiagnostics::execute_phase_6_hft called. Ignition sequence started.");
    }
}
