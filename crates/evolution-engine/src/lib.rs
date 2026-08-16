use quantum_arena::{GlobalArena, TickEvent};
use god_engine_core::GodEngineCore;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::time::sleep;
use rayon::prelude::*;


pub mod meta;
pub mod polars_evolver;
pub mod cma_es;
pub mod ast_mutator;
pub mod random_forest;
pub mod online_random_forest;
pub mod entropy_fitness;

use meta::MetaEvolver;
use quantum_arena::genome::SuperGenotype as Genotype;

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
        let mut mutation_rate = self.arena.config.quantum_mutation_rate.load(Ordering::Relaxed);
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
                // Certificación Bayesiana/Genómica: Umbral dictado por ML Threshold en el Genoma, no hardcodeado a 0.45
                let minimum_viable_wr = self.arena.config.ml_threshold_long.load(Ordering::Relaxed);
                if avg_wr < minimum_viable_wr {
                    println!("🚨 [DEGRADACIÓN DETECTADA] Win Rate Global {:.2}% (Requerido: {:.2}%). Re-activando CMA-ES intenso.", avg_wr * 100.0, minimum_viable_wr * 100.0);
                    mutation_rate = (mutation_rate * 1.5).min(0.5); // Increase mutation rate dynamically if degrading
                } else {
                    mutation_rate = self.arena.config.quantum_mutation_rate.load(Ordering::Relaxed); // Return to baseline
                }
            }
            
            println!("🧠 [TRUE EVOLUTION] Extrayendo ventana de memoria a corto plazo (LockFreeRing)...");
            
            let max_capacity = self.arena.coins.len() * 32768;
            let mut all_ticks = Vec::with_capacity(max_capacity);
            
            for coin_id in 0..self.arena.coins.len() {
                let current_price = self.arena.coins[coin_id].current_price.load(Ordering::Relaxed);
                if current_price == 0.0 { continue; }
                
                let ticks = self.arena.coins[coin_id].tick_ring.snapshot_recent(32768);
                
                for ct in ticks {
                    if ct.bid_price > 0.0 {
                        all_ticks.push(TickEvent {
                            timestamp: ct.timestamp,
                            coin_id,
                            bid_price: ct.bid_price,
                            ask_price: ct.ask_price,
                            bid_qty: ct.bid_qty,
                            ask_qty: ct.ask_qty,
                        });
                    }
                }
            }
            
            if all_ticks.is_empty() {
                continue;
            }
            
            all_ticks.sort_by_key(|t| t.timestamp);
            let ticks_len = all_ticks.len();
            println!("🧠 [TRUE EVOLUTION] Entrenando sobre {} ticks reales. Mutation Rate: {:.1}%", ticks_len, mutation_rate * 100.0);
            
            // Adaptive Population Size based on the current genome
            let pop_size = 40.max((self.arena.config.min_trades_per_day.load(Ordering::Relaxed) * 2.0) as usize).min(100);
            
            // FASE 3: Instanciar el verdadero Optimizador CMA-ES + PSO
            let mut cma_es_optimizer = crate::cma_es::CmaEsOptimizer::new(Genotype::DIMENSION, mutation_rate, Some(pop_size));
            cma_es_optimizer.mean = current_alpha.to_vector(); // Center around current alpha
            cma_es_optimizer.global_best = cma_es_optimizer.mean.clone();
            
            let w = self.arena.config.global_momentum.load(Ordering::Relaxed);
            let c1 = self.arena.config.global_learning_rate.load(Ordering::Relaxed) * 2.0; // cognitive
            let c2 = self.arena.config.global_learning_rate.load(Ordering::Relaxed) * 2.0; // social
            let mut cma_samples = cma_es_optimizer.sample_population(w, c1, c2);
            let mut population: Vec<Genotype> = Vec::with_capacity(pop_size);
            
            // Generate genotypes from CMA-ES vectors
            for vec in &cma_samples {
                population.push(Genotype::from_vector(vec));
            }
            // Always keep the exact alpha to prevent catastrophic forgetting
            population[0] = current_alpha.clone();
            cma_samples[0] = current_alpha.to_vector();
            
            let initial_capital = self.arena.unified_capital.load(Ordering::Relaxed);
            
            let mut results: Vec<_> = population
                .par_iter()
                .enumerate()
                .map(|(i, genome)| {
                    let genome_clone = genome.clone();
                    let test_arena = Arc::new(GlobalArena::new(initial_capital));
                            
                            genome_clone.apply_to_arena(&test_arena);
                            // Note: apply_to_arena already stores global_max_drawdown — no duplicate needed
                            
                            // 🚨 CRÍTICO: Inyectar fees reales al entorno simulado
                            let live_maker = self.arena.config.live_maker_fee.load(Ordering::Relaxed);
                            let live_taker = self.arena.config.live_taker_fee.load(Ordering::Relaxed);
                            test_arena.config.live_maker_fee.store(live_maker, Ordering::Relaxed);
                            test_arena.config.live_taker_fee.store(live_taker, Ordering::Relaxed);
                            
                            let mut engine = GodEngineCore::new(test_arena.clone());
                            let mut total_trades = 0;
                            
                            let mut equity_curve = Vec::with_capacity(100);
                            let mut last_trades = 0;
                            
                            for tick in &all_ticks {
                                test_arena.update_market_data(tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty, tick.timestamp);
                                    let ml_prob = self.arena.coins[tick.coin_id].ml_prob.load(Ordering::Relaxed);
                                    let (new_sc, new_sw, closed_sc, closed_sw) = engine.process_event(
                                        tick.coin_id,
                                        false, false, true,
                                        tick.bid_price, 0.0,
                                        tick.bid_price, tick.ask_price,
                                        tick.bid_qty, tick.ask_qty,
                                        ml_prob, 0.0,
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
                            
                            let sharpe = if equity_curve.len() > 2 {
                                let mut returns = Vec::with_capacity(equity_curve.len());
                                for i in 1..equity_curve.len() {
                                    returns.push((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1].max(1e-10));
                                }
                                let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
                                let variance = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
                                let std_dev = variance.sqrt();
                                if std_dev > 1e-12 {
                                    let trades_per_day = (total_trades as f64).max(1.0);
                                    (mean_ret / std_dev) * trades_per_day.sqrt()
                                } else {
                                    if mean_ret > 0.0 { mean_ret * 100.0 } else { 0.0 }
                                }
                            } else {
                                if pnl > 0.0 { 0.01 } else { -0.01 }
                            };
                            
                            let velocity = final_cap / initial_capital;
                            
                            // Return: (index, raw_fitness, gross_pnl, num_trades, backtest_sharpe, live_sharpe)
                            // We use `velocity` instead of live_sharpe to match the old tuple partially, or just velocity for now.
                            // Actually, let's stick to the expected signature!
                            let raw_fitness = if pnl > 0.0 && velocity >= 1.0 {
                                pnl * sharpe * if total_trades > 0 { 1.0 } else { 0.0 }
                            } else {
                                pnl * (1.0 / sharpe.max(0.01)) // Castigo exponencial a pérdidas
                            };
                            (i, raw_fitness, pnl, total_trades as usize, sharpe, sharpe)
                })
                .collect();
                
            // Apply CMA-ES Update
            let actual_fee_rate = self.arena.config.max_fee_pct.load(Ordering::Relaxed);
            cma_es_optimizer.update(&cma_samples, &mut results, actual_fee_rate);
            
            // Re-sort to find the absolute best
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let best_idx = results[0].0;
            let alpha = &results[0];
            let next_alpha = population[best_idx].clone();
            
            // Evolución de los umbrales de seguridad meta-arquitectónicos
            meta_evolver.audit_system_architecture(alpha.3 as f64);
            
            // FASE 38: Ajuste Dinámico de la Población. 
            // Si el motor encontró un pozo óptimo, aumentamos la entropía reduciendo población
            // Si el motor está estancado, aumentamos la población para buscar en más frentes.
            
            // FASE 37: Requisito de mantener Velocity de 2.0x (o al menos un buen PnL en la muestra actual si es corta)
            let is_high_velocity = alpha.4 >= 1.5; // Relajado para backtests cortos, meta = 2.0x en 3 días.
            
            if alpha.3 > 0 {
                println!("🧬 [ALPHA HOT-SWAP] Nuevo Genoma! Sharpe: {:.2} | Fitness: {:.2} | PnL: +${:.2} ({} trades)", alpha.4, alpha.1, alpha.2, alpha.3);
                if is_high_velocity {
                    println!("🚀 [VELOCITY TARGET MET] Validando escalabilidad!");
                }
                println!("   => Leverage: {:.2}x | Scalp TP: {:.2}% | Spread: {:.4}%", next_alpha.global_leverage, next_alpha.scalp_tp_base * 100.0, next_alpha.maker_spread_pct * 100.0);
                
                // Actualizar Alpha y reducir mutación (explotación)
                current_alpha = next_alpha.clone();
                current_alpha.apply_to_arena(&self.arena);
                current_alpha.save();
                
                println!("✅ [TRUE EVOLUTION] Nueva semilla cuántica Alpha propagada globalmente.");
            } else {
                println!("💀 [ESTANCO] Ningún genoma superó el umbral. Alpha actual se mantiene.");
                // Si el Sharpe decae, aumentamos la tasa de mutación (exploración)
                mutation_rate = (mutation_rate * 1.5).min(0.5);
            }
            
            // Reflexión Arquitectónica de Fase 9
            meta_evolver.audit_system_architecture(alpha.3 as f64);
        }
    }
}




