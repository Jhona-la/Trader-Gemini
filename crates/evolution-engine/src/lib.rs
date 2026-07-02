use quantum_arena::{GlobalArena, TickEvent};
use god_engine_core::GodEngineCore;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::time::sleep;
use rayon::prelude::*;
use rand::RngExt;

pub mod meta;
pub mod polars_evolver;

use meta::MetaEvolver;

#[derive(Clone, Debug)]
pub struct Genotype {
    pub global_leverage: f64,
    pub trend_threshold: f64,
    pub maker_spread_pct: f64,
    pub maker_obi_threshold: f64,
    pub scalp_tp: f64,
    pub scalp_sl: f64,
    pub swing_tp: f64,
    pub swing_sl: f64,
    pub scalp_z_target: f64,
    pub capital_split_scalp: f64,
}

impl Default for Genotype {
    fn default() -> Self {
        Self {
            global_leverage: 20.0,
            trend_threshold: 0.65,
            maker_spread_pct: 0.0003,
            maker_obi_threshold: 0.7,
            scalp_tp: 0.002,
            scalp_sl: 0.002,
            swing_tp: 0.005,
            swing_sl: 0.002,
            scalp_z_target: 2.0,
            capital_split_scalp: 0.5,
        }
    }
}

impl Genotype {
    pub fn mutate(&self, rate: f64) -> Self {
        let mut rng = rand::rng();
        let mut mutate_val = |base: f64, min_val: f64, max_val: f64| -> f64 {
            let change = base * rate * rng.random_range(-1.0..1.0);
            (base + change).clamp(min_val, max_val)
        };

        Self {
            global_leverage: mutate_val(self.global_leverage, 10.0, 50.0),
            trend_threshold: mutate_val(self.trend_threshold, 0.4, 0.8),
            maker_spread_pct: mutate_val(self.maker_spread_pct, 0.0001, 0.0020),
            maker_obi_threshold: mutate_val(self.maker_obi_threshold, 0.5, 0.95),
            scalp_tp: mutate_val(self.scalp_tp, 0.001, 0.010),
            scalp_sl: mutate_val(self.scalp_sl, 0.001, 0.010),
            swing_tp: mutate_val(self.swing_tp, 0.002, 0.020),
            swing_sl: mutate_val(self.swing_sl, 0.001, 0.010),
            scalp_z_target: mutate_val(self.scalp_z_target, 1.0, 4.0),
            capital_split_scalp: mutate_val(self.capital_split_scalp, 0.2, 0.8),
        }
    }
}

pub struct EvolutionEngine {
    arena: Arc<GlobalArena>,
}

impl EvolutionEngine {
    pub fn new(arena: Arc<GlobalArena>) -> Self {
        Self { arena }
    }

    pub async fn start_evolution_loop(&self) {
        println!("🧠 [TRUE EVOLUTION] Motor de Inteligencia Artificial Live Iniciado (CMA-ES).");
        
        let mut current_alpha = Genotype::default();
        let mut mutation_rate = 0.1; // Empieza con 10% de exploración
        let meta_evolver = MetaEvolver::new(self.arena.clone());

        loop {
            sleep(Duration::from_secs(15)).await;
            
            // FASE 9: Auto-Evolución y Detección de Degradación
            let mut total_wr = 0.0;
            let mut valid_coins = 0;
            for coin_id in 0..self.arena.coins.len() {
                let coin_wr = self.arena.coins[coin_id].scalp.win_rate.load(Ordering::Relaxed);
                if self.arena.coins[coin_id].current_price.load(Ordering::Relaxed) > 0.0 {
                    total_wr += coin_wr;
                    valid_coins += 1;
                }
            }
            if valid_coins > 0 {
                let avg_wr = total_wr / valid_coins as f64;
                if avg_wr < 0.45 {
                    println!("🚨 [DEGRADACIÓN DETECTADA] Win Rate Global {:.2}%. Disparando Auto-Reentrenamiento HFT...", avg_wr * 100.0);
                    match tokio::process::Command::new("cargo")
                        .args(["run", "--release", "--bin", "train_nano_forest"])
                        .spawn()
                    {
                        Ok(mut child) => {
                            tokio::spawn(async move {
                                let _ = child.wait().await;
                                println!("✅ [AUTO-REENTRENAMIENTO] NanoForest reentrenado. El hot-reload lo tomará en el próximo ciclo HFT.");
                                // Forzamos carga inmediata en cache global
                                let _ = god_engine_core::ml_inference::NanoForest::load_global("BTCUSDT_SCALP", "models/BTCUSDT_SCALP.json");
                            });
                        }
                        Err(e) => eprintln!("❌ [ERROR] Falló al lanzar reentrenamiento: {}", e),
                    }
                }
            }
            
            println!("🧠 [TRUE EVOLUTION] Extrayendo ventana de memoria a corto plazo (LockFreeRing)...");
            
            let max_capacity = self.arena.coins.len() * 32768;
            let mut all_ticks = Vec::with_capacity(max_capacity);
            
            for coin_id in 0..self.arena.coins.len() {
                let current_price = self.arena.coins[coin_id].current_price.load(Ordering::Relaxed);
                if current_price == 0.0 { continue; }
                
                let ticks = self.arena.coins[coin_id].tick_ring.snapshot_recent(32768);
                
                let mut timestamp = 10000u64;
                for ct in ticks {
                    if ct.bid_price > 0.0 {
                        all_ticks.push(TickEvent {
                            timestamp,
                            coin_id,
                            bid_price: ct.bid_price,
                            ask_price: ct.ask_price,
                            bid_qty: ct.bid_qty,
                            ask_qty: ct.ask_qty,
                        });
                        timestamp += 100;
                    }
                }
            }
            
            if all_ticks.is_empty() {
                continue;
            }
            
            all_ticks.sort_by_key(|t| t.timestamp);
            let ticks_len = all_ticks.len();
            println!("🧠 [TRUE EVOLUTION] Entrenando sobre {} ticks reales. Mutation Rate: {:.1}%", ticks_len, mutation_rate * 100.0);
            
            let pop_size = 40;
            let mut population: Vec<Genotype> = Vec::with_capacity(pop_size);
            
            // 1 clon exacto de Alpha actual
            population.push(current_alpha.clone());
            // 30 mutaciones alrededor del Alpha
            for _ in 1..30 {
                population.push(current_alpha.mutate(mutation_rate));
            }
            // 9 completamente aleatorias (Exploración pura)
            for _ in 30..pop_size {
                population.push(Genotype::default().mutate(0.5));
            }
            
            let initial_capital = self.arena.unified_capital.load(Ordering::Relaxed);
            
            let mut results: Vec<_> = population
                .par_iter()
                .map(|genome| {
                    let genome_clone = genome.clone();
                    let test_arena = Arc::new(GlobalArena::new(initial_capital));
                            
                            test_arena.config.global_leverage.store(genome_clone.global_leverage, Ordering::Relaxed);
                            test_arena.config.trend_threshold.store(genome_clone.trend_threshold, Ordering::Relaxed);
                            test_arena.config.maker_spread_pct.store(genome_clone.maker_spread_pct, Ordering::Relaxed);
                            test_arena.config.maker_obi_threshold.store(genome_clone.maker_obi_threshold, Ordering::Relaxed);
                            test_arena.config.scalp_tp_base.store(genome_clone.scalp_tp, Ordering::Relaxed);
                            test_arena.config.scalp_sl_base.store(genome_clone.scalp_sl, Ordering::Relaxed);
                            test_arena.config.swing_tp_base.store(genome_clone.swing_tp, Ordering::Relaxed);
                            test_arena.config.swing_sl_base.store(genome_clone.swing_sl, Ordering::Relaxed);
                            test_arena.config.global_max_drawdown.store(0.95, Ordering::Relaxed);
                            
                            let mut engine = GodEngineCore::new(test_arena.clone());
                            let mut total_trades = 0;
                            
                            let mut equity_curve = Vec::with_capacity(100);
                            let mut last_trades = 0;
                            
                            for tick in &all_ticks {
                                test_arena.update_market_data(tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty);
                                let (new_sc, new_sw, closed_sc, closed_sw) = engine.process_event(
                                    tick.coin_id,
                                    false, false, true,
                                    tick.bid_price, 0.0,
                                    tick.bid_price, tick.ask_price,
                                    tick.bid_qty, tick.ask_qty,
                                    0.5, 0.0,
                                    tick.timestamp,
                                    false, &[0.0; 54]);
                                if new_sc.is_some() || new_sw.is_some() || closed_sc.is_some() || closed_sw.is_some() {
                                    total_trades += 1;
                                }
                                
                                if total_trades > last_trades {
                                    equity_curve.push(test_arena.unified_capital.load(Ordering::Relaxed));
                                    last_trades = total_trades;
                                }
                            }
                            
                            let final_cap = test_arena.unified_capital.load(Ordering::Relaxed);
                            let pnl = final_cap - initial_capital;
                            
                            let sharpe = if equity_curve.len() > 2 && pnl > 0.0 {
                                let mut returns = Vec::with_capacity(equity_curve.len());
                                for i in 1..equity_curve.len() {
                                    returns.push((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]);
                                }
                                let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
                                let variance = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
                                let std_dev = variance.sqrt();
                                if std_dev > 0.0 {
                                    (mean_ret / std_dev) * (returns.len() as f64).sqrt()
                                } else {
                                    0.0
                                }
                            } else {
                                if pnl > 0.0 { 0.1 } else { -1.0 }
                            };
                            
                            (genome_clone, pnl, total_trades, sharpe)
                })
                .collect();
                
            // Ordenar por Sharpe Ratio en lugar de solo PnL absoluto
            results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
            let alpha = &results[0];
            
            if alpha.3 > 0.5 && alpha.1 > 0.0 {
                println!("🧬 [ALPHA HOT-SWAP] Nuevo Genoma Evolucionado! Sharpe: {:.2} | PnL: +${:.2} ({} trades)", alpha.3, alpha.1, alpha.2);
                println!("   => Leverage: {:.2}x | Scalp TP: {:.2}% | Spread: {:.4}%", alpha.0.global_leverage, alpha.0.scalp_tp * 100.0, alpha.0.maker_spread_pct * 100.0);
                
                // Actualizar Alpha y reducir mutación (explotación)
                current_alpha = alpha.0.clone();
                mutation_rate = (mutation_rate * 0.9).max(0.02);
                
                self.arena.config.global_leverage.store(alpha.0.global_leverage, Ordering::Relaxed);
                self.arena.config.trend_threshold.store(alpha.0.trend_threshold, Ordering::Relaxed);
                self.arena.config.maker_spread_pct.store(alpha.0.maker_spread_pct, Ordering::Relaxed);
                self.arena.config.scalp_tp_base.store(alpha.0.scalp_tp, Ordering::Relaxed);
                self.arena.config.scalp_sl_base.store(alpha.0.scalp_sl, Ordering::Relaxed);
            } else {
                println!("🛡️ [TRUE EVOLUTION] Decadencia del Sharpe ({:.2}). Aumentando entropía para explorar.", alpha.3);
                // Si el Sharpe decae, aumentamos la tasa de mutación (exploración)
                mutation_rate = (mutation_rate * 1.5).min(0.5);
            }
            
            // Reflexión Arquitectónica de Fase 9
            meta_evolver.audit_system_architecture(alpha.3);
        }
    }
}




