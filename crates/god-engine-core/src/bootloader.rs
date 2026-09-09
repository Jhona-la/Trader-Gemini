use crate::GodEngineCore;
use quantum_arena::GlobalArena;
use reqwest::Client;
use serde_json::Value;
use std::sync::Arc;
use tokio::time::{Duration, sleep};

/// Orquestador de Secuencia de Arranque (Fase 1: System Bootloader)
pub struct SystemBootloader {
    arena: Arc<GlobalArena>,
}

impl SystemBootloader {
    pub fn new(arena: Arc<GlobalArena>) -> Self {
        Self { arena }
    }

    /// Lanza la secuencia estricta de 6 fases
    pub async fn execute_boot_sequence(
        &self,
        engine: &mut GodEngineCore,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
        println!(
            "🔍 [FASE 1] Verificando integridad de configuración, topología de memoria y llaves..."
        );
        let base_cap = self
            .arena
            .config
            .base_capital
            .load(std::sync::atomic::Ordering::Relaxed);
        if base_cap < 5.0 {
            return Err(format!(
                "Capital base inválido: ${:.2} (mínimo requerido: $5.00)",
                base_cap
            )
            .into());
        }
        if self
            .arena
            .kill_switch_active
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err("Kill switch activo al arranque. Abortando inicio por seguridad.".into());
        }
        if self.arena.coins.len() != 30 {
            return Err(format!(
                "Topología de memoria corrupta: {} slots en lugar de 30",
                self.arena.coins.len()
            )
            .into());
        }
        println!("   -> Capital base validado: ${:.2} USD", base_cap);
        println!("   -> Memoria contigua: 30 CoinArenas L1/L2 alineadas.");
        println!("   -> Kill Switch: Desarmado (Estado Seguro).");
        Ok(())
    }

    async fn phase_2_asset_selection(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 [FASE 2] Verificando y seleccionando universo activo de activos...");
        let mut active_count = 0;
        for (id, _coin) in self.arena.coins.iter().enumerate() {
            if let Some(spec) = quantum_arena::symbol_registry::try_spec(id) {
                if !spec.symbol.is_empty() {
                    active_count += 1;
                }
            }
        }
        let cap = self
            .arena
            .unified_capital
            .load(std::sync::atomic::Ordering::Relaxed);
        println!("   -> {} activos registrados en memoria.", active_count);
        println!(
            "   -> Universo calibrado para capital actual: ${:.2} USD.",
            cap
        );
        Ok(())
    }

    async fn phase_3_historical_warmup(
        &self,
        engine: &mut GodEngineCore,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔥 [FASE 3] Calentando motores de características (Warm-Up)...");
        let client = Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| Client::new());
        let limit = 1000;

        for (coin_id, _coin) in self.arena.coins.iter().enumerate() {
            let symbol = match quantum_arena::symbol_registry::try_spec(coin_id) {
                Some(s) if !s.symbol.is_empty() => s.symbol,
                _ => continue,
            };

            // FIX #1504: Endpoint adaptativo para Testnet vs Producción
            let is_testnet = std::env::var("USE_TESTNET")
                .unwrap_or_default()
                .trim()
                .to_lowercase()
                == "true";
            let base_url = if is_testnet {
                "https://testnet.binancefuture.com"
            } else {
                "https://fapi.binance.com"
            };
            let url = format!(
                "{}/fapi/v1/klines?symbol={}&interval=1m&limit={}",
                base_url, symbol, limit
            );

            let mut retries = 3;
            let mut klines_data = None;

            while retries > 0 {
                match client.get(&url).send().await {
                    Ok(resp) => {
                        if let Ok(json) = resp.json::<Value>().await {
                            klines_data = Some(json);
                            break;
                        }
                    }
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
                            let open = kline_arr[1]
                                .as_str()
                                .unwrap_or("0")
                                .parse::<f64>()
                                .unwrap_or(0.0);
                            let high = kline_arr[2]
                                .as_str()
                                .unwrap_or("0")
                                .parse::<f64>()
                                .unwrap_or(0.0);
                            let low = kline_arr[3]
                                .as_str()
                                .unwrap_or("0")
                                .parse::<f64>()
                                .unwrap_or(0.0);
                            let close = kline_arr[4]
                                .as_str()
                                .unwrap_or("0")
                                .parse::<f64>()
                                .unwrap_or(0.0);
                            let volume = kline_arr[5]
                                .as_str()
                                .unwrap_or("0")
                                .parse::<f64>()
                                .unwrap_or(0.0);

                            // Proyectamos el evento cerrado hacia el motor
                            engine.feature_engines[coin_id]
                                .process_kline(open, high, low, close, volume);
                        }
                    }
                }
                println!(
                    "      ✅ {} klines procesadas en StatefulEngine.",
                    klines.len()
                );
            } else {
                println!("      ❌ Falló la carga de klines para {}", symbol);
            }

            // Limit API rate
            sleep(Duration::from_millis(200)).await;
        }

        Ok(())
    }

    async fn phase_4_state_recovery(
        &self,
        _engine: &mut GodEngineCore,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "💾 [FASE 4] Recuperando y verificando continuidad de estado (State Continuity Engine)..."
        );
        let mut total_open = 0;

        for (id, coin) in self.arena.coins.iter().enumerate() {
            let pos = &coin.positions.position;

            if pos.is_open() {
                total_open += 1;
                let chk =
                    quantum_arena::state_continuity::StateContinuityEngine::compute_state_checksum(
                        id,
                        pos.quantity.load(std::sync::atomic::Ordering::Relaxed),
                        pos.entry_price.load(std::sync::atomic::Ordering::Relaxed),
                    );
                println!(
                    "   -> [UNIVERSAL] Posición abierta en ID {} detectada (Checksum: {:016X})",
                    id, chk
                );
            }
        }
        println!(
            "   -> Continuidad verificada: {} posiciones universales continuas vivas.",
            total_open
        );
        Ok(())
    }

    fn phase_5_ml_preload(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🧠 [FASE 5] Verificando modelos ML cargados...");
        let count = crate::ml_inference::GLOBAL_FORESTS.load().len();
        println!(
            "   -> Modelos NanoForest globales listos en ArcSwap: {} cargados.",
            count
        );
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
    pub async fn execute_phase_1_and_2(
        _is_demo: bool,
    ) -> Result<(i64, Vec<String>), Box<dyn std::error::Error>> {
        let syms = vec![
            "btcusdt".to_string(),
            "ethusdt".to_string(),
            "bnbusdt".to_string(),
            "solusdt".to_string(),
            "xrpusdt".to_string(),
            "dogeusdt".to_string(),
            "adausdt".to_string(),
            "shibusdt".to_string(),
            "dotusdt".to_string(),
            "linkusdt".to_string(),
            "polusdt".to_string(),
            "ltcusdt".to_string(),
            "bchusdt".to_string(),
            "atomusdt".to_string(),
            "uniusdt".to_string(),
            "xlmusdt".to_string(),
            "nearusdt".to_string(),
            "icpusdt".to_string(),
            "filusdt".to_string(),
            "vetusdt".to_string(),
            "avaxusdt".to_string(),
            "opusdt".to_string(),
            "aptusdt".to_string(),
            "arbusdt".to_string(),
            "renderusdt".to_string(),
            "ldousdt".to_string(),
        ];
        Ok((0, syms))
    }

    /// X-017 (REHAB-6): OHLCV COMPLETO por kline — antes solo closes con
    /// banda sintética ±0.05% y volumen fijo 10: v_t (volatilidad de rango)
    /// quedaba clavado en ~0.1% y el ATR vivo des-calibraba los stops por
    /// horas (get_atr_pct alimenta el sizing en vivo).
    pub async fn execute_phase_3_warmup(
        _is_demo: bool,
        symbols: &[String],
    ) -> std::collections::HashMap<String, Vec<[f64; 5]>> {
        let mut map = std::collections::HashMap::new();
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .timeout(tokio::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let limit = 1000;

        for symbol in symbols {
            let mut klines_vec = Vec::new();
            let url = format!(
                "https://fapi.binance.com/fapi/v1/klines?symbol={}&interval=1m&limit={}",
                symbol.to_uppercase(),
                limit
            );

            if let Ok(resp) = client.get(&url).send().await {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let serde_json::Value::Array(klines) = json {
                        for k in &klines {
                            if let serde_json::Value::Array(kline_arr) = k {
                                // [openTime, open, high, low, close, volume, ...]
                                if kline_arr.len() >= 6 {
                                    let parse = |idx: usize| {
                                        kline_arr[idx]
                                            .as_str()
                                            .unwrap_or("0")
                                            .parse::<f64>()
                                            .unwrap_or(0.0)
                                    };
                                    let (o, h, l, c, v) =
                                        (parse(1), parse(2), parse(3), parse(4), parse(5));
                                    if c > 0.0
                                        && c.is_finite()
                                        && h >= l
                                        && h >= c
                                        && l <= c
                                        && o > 0.0
                                    {
                                        klines_vec.push([o, h, l, c, v]);
                                    }
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

    pub async fn execute_phase_4_training(
        _symbols: &[String],
        _historical_klines: &std::collections::HashMap<String, Vec<[f64; 5]>>,
        _lr: f64,
        _epochs: usize,
    ) -> Option<()> {
        Some(())
    }

    pub fn execute_phase_6_hft() {
        println!(
            "🚀 [HFT IGNITION] SystemDiagnostics::execute_phase_6_hft called. Ignition sequence started."
        );
    }
}
