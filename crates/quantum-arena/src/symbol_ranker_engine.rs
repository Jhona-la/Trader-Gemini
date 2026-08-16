use reqwest::Client;
use serde_json::Value;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use super::symbols::{update_dynamic_universe};
use crate::symbol_registry::{SymbolSpec, update_registry};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct UniverseSymbol {
    pub symbol: String,
    pub volume: f64,
    pub trades: f64,
    pub volatility: f64,
    pub liquidity_density: f64,
    pub order_flow_imbalance: f64,
    pub tensor_score: f64,
}

/// 👁️ MOTOR DE RASTREO OMNISCIENTE (V10)
/// Rastrea continuamente TODO el universo de Binance Futures (~250+ pares)
/// Aplica tensores matemáticos a todo el mercado y selecciona el Top 10 hiper-líquido.
pub struct SymbolRankerEngine {
    client: Client,
    pub tracked_universe: Arc<tokio::sync::RwLock<HashMap<String, UniverseSymbol>>>,
    pub arena: Arc<crate::GlobalArena>,
}

impl SymbolRankerEngine {
    pub fn new(arena: Arc<crate::GlobalArena>) -> Self {
        Self {
            client: Client::new(),
            tracked_universe: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            arena,
        }
    }

    /// Tarea principal asíncrona (Daemon)
    pub async fn run_daemon(self: Arc<Self>) {
        loop {
            // Evaluamos la selección (V10: Adaptabilidad con ExchangeInfo)
            match self.fetch_and_rank_symbols().await {
                Ok((top_30, new_specs)) => {
                    telemetry_engine::telemetry!("🌐 [OMNISCIENT TRACKER] Nuevo Universo Cuántico Top 30 descubierto.");
                    update_registry(new_specs);
                    update_dynamic_universe(&top_30);
                }
                Err(e) => {
                    telemetry_engine::telemetry!("⚠️ [OMNISCIENT TRACKER] Fallo al evaluar el mercado global de Binance: {}", e);
                }
            }
            let interval = self.arena.config.symbol_ranker_interval_secs.load(std::sync::atomic::Ordering::Relaxed) as u64;
            sleep(Duration::from_secs(interval.max(10))).await;
        }
    }

    /// Consulta a Binance, filtra pares USDT y rankea TODO el mercado matemáticamente (Zero-Hardcode)
    async fn fetch_and_rank_symbols(&self) -> Result<(Vec<String>, Vec<SymbolSpec>), Box<dyn std::error::Error>> {
        // 1. Fetch Exchange Info para obtener la precisión real de Binance
        let is_testnet = std::env::var("USE_TESTNET").unwrap_or_default().trim().to_lowercase() == "true";
        let base_url = if is_testnet {
            "https://testnet.binancefuture.com"
        } else {
            "https://fapi.binance.com"
        };
        
        let ex_url = format!("{}/fapi/v1/exchangeInfo", base_url);
        let ex_info: Value = self.client.get(&ex_url).send().await?.json().await?;
        let symbols_info = ex_info["symbols"].as_array().ok_or("No symbols array")?;
        
        let url = format!("{}/fapi/v1/ticker/24hr", base_url);
        let res = self.client.get(url).send().await?.json::<Vec<Value>>().await?;
        
        let mut candidates = Vec::new();
        // Estructura temporal para asociar SymbolSpec a los candidatos
        let mut specs_map = HashMap::new();
        
        for item in res {
            if let Some(symbol) = item["symbol"].as_str() {
                if symbol.ends_with("USDT") && !symbol.contains("_") {
                    let volume = item["quoteVolume"].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                    let trades = item["count"].as_u64().unwrap_or(0) as f64;
                    let last = item["lastPrice"].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                    let price_change_pct = item["priceChangePercent"].as_str().unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                    
                    if let Some(info) = symbols_info.iter().find(|s| s["symbol"].as_str().unwrap_or("") == symbol) {
                        let status = info["status"].as_str().unwrap_or("");
                        if status != "TRADING" { continue; }
                        
                        let mut step_size = 0.0;
                        let mut min_qty = 0.0;
                        let mut tick_size = 0.0;
                        let mut min_notional = 0.0;
                        
                        if let Some(filters) = info["filters"].as_array() {
                            for f in filters {
                                if f["filterType"] == "LOT_SIZE" {
                                    step_size = f["stepSize"].as_str().unwrap_or("0.0").parse().unwrap_or(0.0);
                                    min_qty = f["minQty"].as_str().unwrap_or("0.0").parse().unwrap_or(0.0);
                                } else if f["filterType"] == "PRICE_FILTER" {
                                    tick_size = f["tickSize"].as_str().unwrap_or("0.0").parse().unwrap_or(0.0);
                                } else if f["filterType"] == "MIN_NOTIONAL" {
                                    min_notional = f["notional"].as_str().unwrap_or("0.0").parse().unwrap_or(0.0);
                                }
                            }
                        }
                        
                        // Filtrar con umbrales mínimos. La dinámica del volumen percentile viene después en el sort.
                        // Usamos un base de 1,000,000 para no iterar sobre monedas completamente muertas, 
                        // pero la selección final la hace el ranker.
                        // No hardcoded filters. We process all active TRADING pairs.
                        if last > 0.000001 && volume > 0.0 && trades > 0.0 {
                            let maker_fee = self.arena.config.live_maker_fee.load(std::sync::atomic::Ordering::Relaxed);
                            let taker_fee = self.arena.config.live_taker_fee.load(std::sync::atomic::Ordering::Relaxed);
                            let fee_roundtrip_pct = maker_fee + taker_fee;
                            
                            let fee_in_price = last * fee_roundtrip_pct;
                            let breakeven_ticks = fee_in_price / tick_size.max(0.000000001);
                            
                            let liquidity_density = volume / trades.max(1.0);
                            let trade_frequency = trades / 1440.0;
                            
                            let momentum_volatility_ratio = price_change_pct.abs() / (tick_size * 1000.0).max(0.1);
                            let tick_advantage = (20.0 / breakeven_ticks.max(1.0)).min(50.0);
                            
                            let liquidity_momentum = (liquidity_density.sqrt() * trade_frequency.sqrt()) / 100.0;
                            let score = liquidity_momentum * momentum_volatility_ratio * tick_advantage;
                            
                            // Dynamic leverage cap based on volatility instead of hardcoded symbol matches.
                            // The lower the volatility relative to step_size, the higher leverage allowed, capped at genome.
                            let base_max_lev = self.arena.config.leverage_cap.load(std::sync::atomic::Ordering::Relaxed);
                            // We adjust leverage down for highly volatile/unpredictable assets
                            let lev_penalty = (momentum_volatility_ratio / 10.0).clamp(1.0, 5.0);
                            let max_lev = (base_max_lev / lev_penalty) as u32;
                            
                            let spec = SymbolSpec {
                                symbol: symbol.to_string(),
                                step_size,
                                tick_size,
                                min_qty,
                                min_notional,
                                max_leverage: max_lev.max(10).min(125), // Safe bounds
                                maker_fee,
                                taker_fee,
                            };
                            
                            specs_map.insert(symbol.to_string(), spec);
                                      
                            candidates.push(UniverseSymbol {
                                symbol: symbol.to_string(),
                                volume,
                                trades,
                                volatility: momentum_volatility_ratio,
                                liquidity_density: volume / trades,
                                order_flow_imbalance: 0.0,
                                tensor_score: score,
                            });
                        }
                    }
                }
            }
        }
        
        {
            let mut map = self.tracked_universe.write().await;
            map.clear();
            for c in &candidates {
                map.insert(c.symbol.clone(), c.clone());
            }
        }
        
        // Ordenamiento tensorial para sacar el Top
        candidates.sort_by(|a, b| b.tensor_score.partial_cmp(&a.tensor_score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Inject BTC and ETH implicitly if they are missing
        let mut final_coins = Vec::new();
        if let Some(btc) = candidates.iter().find(|c| c.symbol == "BTCUSDT") {
            final_coins.push(btc.clone());
        }
        if let Some(eth) = candidates.iter().find(|c| c.symbol == "ETHUSDT") {
            if final_coins.iter().all(|c| c.symbol != "ETHUSDT") {
                final_coins.push(eth.clone());
            }
        }
        
        let max_universe_size = 30; // Limite lógico para no saturar memoria innecesariamente
        for coin in candidates {
            if final_coins.len() >= max_universe_size { break; }
            if final_coins.iter().all(|c| c.symbol != coin.symbol) {
                final_coins.push(coin);
            }
        }
        
        let mut top_30: Vec<String> = Vec::with_capacity(final_coins.len());
        let mut new_specs: Vec<SymbolSpec> = Vec::with_capacity(final_coins.len());
        
        for coin in final_coins {
            top_30.push(coin.symbol.clone());
            new_specs.push(specs_map[&coin.symbol].clone());
        }
        
        Ok((top_30, new_specs))
    }
}
