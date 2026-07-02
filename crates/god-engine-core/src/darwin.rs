use quantum_arena::GlobalArena;
use crate::GodEngineCore;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use rayon::prelude::*;

use rand::RngExt;
use quantum_arena::tick_source::TickEvent;

/// Axioma X: The Darwin Daemon
/// Continuous Online Evolution. Evaluates the recent market microstructure
/// and dynamically hot-swaps parameters without stopping the live engine.

#[derive(Debug, Clone)]
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
    pub min_confidence: f64,
    pub explosive_leverage_multiplier: f64,
}

impl Genotype {
    pub fn new_random() -> Self {
        Self {
            global_leverage: rand::rng().random_range(10.0..125.0),
            trend_threshold: rand::rng().random_range(0.3..0.8),
            maker_spread_pct: rand::rng().random_range(0.0001..0.0020),
            maker_obi_threshold: rand::rng().random_range(0.3..0.9),
            scalp_tp: rand::rng().random_range(0.001..0.010),
            scalp_sl: rand::rng().random_range(0.001..0.005),
            swing_tp: rand::rng().random_range(0.002..0.015),
            swing_sl: rand::rng().random_range(0.001..0.005),
            scalp_z_target: rand::rng().random_range(1.0..4.0),
            capital_split_scalp: rand::rng().random_range(0.1..1.0),
            min_confidence: rand::rng().random_range(0.5..0.95),
            explosive_leverage_multiplier: rand::rng().random_range(1.0..5.0),
        }
    }

    pub fn current_from_arena(arena: &GlobalArena) -> Self {
        Self {
            global_leverage: arena.config.global_leverage.load(Ordering::Relaxed),
            trend_threshold: arena.config.trend_threshold.load(Ordering::Relaxed),
            maker_spread_pct: arena.config.maker_spread_pct.load(Ordering::Relaxed),
            maker_obi_threshold: arena.config.maker_obi_threshold.load(Ordering::Relaxed),
            scalp_tp: arena.config.scalp_tp_base.load(Ordering::Relaxed),
            scalp_sl: arena.config.scalp_sl_base.load(Ordering::Relaxed),
            swing_tp: arena.config.swing_tp_base.load(Ordering::Relaxed),
            swing_sl: arena.config.swing_sl_base.load(Ordering::Relaxed),
            scalp_z_target: arena.config.scalp_obi_threshold.load(Ordering::Relaxed),
            capital_split_scalp: arena.config.capital_split_scalp.load(Ordering::Relaxed),
            min_confidence: arena.config.min_confidence_btc.load(Ordering::Relaxed),
            explosive_leverage_multiplier: arena.config.explosive_leverage_multiplier.load(Ordering::Relaxed),
        }
    }

    pub fn apply_to_arena(&self, arena: &GlobalArena) {
        arena.config.global_leverage.store(self.global_leverage, Ordering::Relaxed);
        // Kelly fractions are pure math now, removed from Darwin
        arena.config.trend_threshold.store(self.trend_threshold, Ordering::Relaxed);
        arena.config.maker_spread_pct.store(self.maker_spread_pct, Ordering::Relaxed);
        arena.config.maker_obi_threshold.store(self.maker_obi_threshold, Ordering::Relaxed);
        arena.config.scalp_tp_base.store(self.scalp_tp, Ordering::Relaxed);
        arena.config.scalp_sl_base.store(self.scalp_sl, Ordering::Relaxed);
        arena.config.swing_tp_base.store(self.swing_tp, Ordering::Relaxed);
        arena.config.swing_sl_base.store(self.swing_sl, Ordering::Relaxed);
        arena.config.scalp_obi_threshold.store(self.scalp_z_target, Ordering::Relaxed);
        arena.config.capital_split_scalp.store(self.capital_split_scalp, Ordering::Relaxed);
        arena.config.min_confidence_btc.store(self.min_confidence, Ordering::Relaxed);
        arena.config.explosive_leverage_multiplier.store(self.explosive_leverage_multiplier, Ordering::Relaxed);
    }
}

pub struct DarwinDaemon {
    pub live_arena: Arc<GlobalArena>,
}

impl DarwinDaemon {
    pub fn new(live_arena: Arc<GlobalArena>) -> Self {
        Self { live_arena }
    }

    /// Extacts the recent ticks from the live arena, sorts them, and runs a fast GA
    pub fn evolve_online(&self) {
        let mut master_stream = Vec::with_capacity(4 * 32768);
        
        // 1. Extract memory snapshot (lock-free: snapshot_recent never blocks the writer)
        for coin_id in 0..4 {
            let ticks = self.live_arena.coins[coin_id].tick_ring.snapshot_recent(32768);
            
            for tick in ticks {
                master_stream.push(TickEvent {
                    coin_id,
                    timestamp: 0,
                    bid_price: tick.bid_price,
                    ask_price: tick.ask_price,
                    bid_qty: tick.bid_qty,
                    ask_qty: tick.ask_qty,
                });
            }
        }
        
        if master_stream.is_empty() {
            return;
        }
        
        println!("[Darwin] Extracted {} recent ticks. Starting online evolution...", master_stream.len());

        let pop_size = 20; // Fast mini-evolution
        let generations = 5;
        let mutation_rate = 0.3;
        
        let mut population: Vec<Genotype> = (0..pop_size).map(|_| Genotype::new_random()).collect();
        // Ensure current active genotype is in the pool (Elitism baseline)
        let current_active = Genotype::current_from_arena(&self.live_arena);
        population[0] = current_active.clone();

        let initial_capital = 13.0; // Standardize for testing fitness
        let mut best_all_time = (population[0].clone(), 0.0_f64);

        for generation in 1..=generations {
            let mut results: Vec<_> = population
                .par_iter()
                .map(|genome| {
                    let arena = Arc::new(GlobalArena::new(initial_capital));
                    genome.apply_to_arena(&arena);
                    arena.config.global_max_drawdown.store(0.95, Ordering::Relaxed);
                    
                    let mut engine = GodEngineCore::new(arena.clone());
                    let mut max_drawdown = 0.0;
                    let mut peak_capital = initial_capital;

                    for tick in &master_stream {
                        arena.update_market_data(tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty);
                        let (_sc, _sw, c_sc, c_sw, _) = engine.process_tick(
                            tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty, tick.timestamp, &[0.0; 54]);
                        
                        if c_sc.is_some() || c_sw.is_some() {
                            let current_cap = arena.unified_capital.load(Ordering::Relaxed);
                            if current_cap > peak_capital { peak_capital = current_cap; }
                            let dd = (peak_capital - current_cap) / peak_capital;
                            if dd > max_drawdown { max_drawdown = dd; }
                        }
                    }

                    let final_cap = arena.unified_capital.load(Ordering::Relaxed);
                    let fitness = (final_cap - initial_capital) * (1.0 - max_drawdown);
                    (genome.clone(), final_cap, fitness)
                })
                .collect();
                
            results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            let best_gen = &results[0];
            
            if best_gen.2 > best_all_time.1 {
                best_all_time = (best_gen.0.clone(), best_gen.2);
            }
            
            if generation == generations { break; }
            
            let mut next_gen = Vec::with_capacity(pop_size);
            for i in 0..(pop_size / 4) { next_gen.push(results[i].0.clone()); } // Top 25% elites
            
            while next_gen.len() < pop_size {
                let p1 = &results[rand::rng().random_range(0..(pop_size/2))].0;
                let p2 = &results[rand::rng().random_range(0..(pop_size/2))].0;
                
                let mut child = Genotype {
                    global_leverage: if rand::rng().random_bool(0.5) { p1.global_leverage } else { p2.global_leverage },
                    trend_threshold: if rand::rng().random_bool(0.5) { p1.trend_threshold } else { p2.trend_threshold },
                    maker_spread_pct: if rand::rng().random_bool(0.5) { p1.maker_spread_pct } else { p2.maker_spread_pct },
                    maker_obi_threshold: if rand::rng().random_bool(0.5) { p1.maker_obi_threshold } else { p2.maker_obi_threshold },
                    scalp_tp: if rand::rng().random_bool(0.5) { p1.scalp_tp } else { p2.scalp_tp },
                    scalp_sl: if rand::rng().random_bool(0.5) { p1.scalp_sl } else { p2.scalp_sl },
                    swing_tp: if rand::rng().random_bool(0.5) { p1.swing_tp } else { p2.swing_tp },
                    swing_sl: if rand::rng().random_bool(0.5) { p1.swing_sl } else { p2.swing_sl },
                    scalp_z_target: if rand::rng().random_bool(0.5) { p1.scalp_z_target } else { p2.scalp_z_target },
                    capital_split_scalp: if rand::rng().random_bool(0.5) { p1.capital_split_scalp } else { p2.capital_split_scalp },
                    min_confidence: if rand::rng().random_bool(0.5) { p1.min_confidence } else { p2.min_confidence },
                    explosive_leverage_multiplier: if rand::rng().random_bool(0.5) { p1.explosive_leverage_multiplier } else { p2.explosive_leverage_multiplier },
                };
                
                if rand::rng().random_bool(mutation_rate) { child.global_leverage *= rand::rng().random_range(0.8..1.2); }
                if rand::rng().random_bool(mutation_rate) { child.trend_threshold *= rand::rng().random_range(0.9..1.1); }
                if rand::rng().random_bool(mutation_rate) { child.maker_spread_pct *= rand::rng().random_range(0.5..2.0); }
                if rand::rng().random_bool(mutation_rate) { child.maker_obi_threshold *= rand::rng().random_range(0.8..1.2); }
                if rand::rng().random_bool(mutation_rate) { child.scalp_tp *= rand::rng().random_range(0.7..1.5); }
                if rand::rng().random_bool(mutation_rate) { child.scalp_sl *= rand::rng().random_range(0.7..1.5); }
                if rand::rng().random_bool(mutation_rate) { child.swing_tp *= rand::rng().random_range(0.7..1.5); }
                if rand::rng().random_bool(mutation_rate) { child.swing_sl *= rand::rng().random_range(0.7..1.5); }
                if rand::rng().random_bool(mutation_rate) { child.scalp_z_target *= rand::rng().random_range(0.8..1.2); }
                if rand::rng().random_bool(mutation_rate) { child.capital_split_scalp *= rand::rng().random_range(0.8..1.2); }
                if rand::rng().random_bool(mutation_rate) { child.min_confidence *= rand::rng().random_range(0.9..1.1); }
                if rand::rng().random_bool(mutation_rate) { child.explosive_leverage_multiplier *= rand::rng().random_range(0.5..2.0); }
                
                child.global_leverage = child.global_leverage.clamp(1.0, 125.0);
                child.trend_threshold = child.trend_threshold.clamp(0.1, 0.9);
                child.maker_spread_pct = child.maker_spread_pct.clamp(0.0001, 0.05);
                child.maker_obi_threshold = child.maker_obi_threshold.clamp(0.1, 0.95);
                child.scalp_tp = child.scalp_tp.clamp(0.0005, 0.02);
                child.scalp_sl = child.scalp_sl.clamp(0.0005, 0.01);
                child.swing_tp = child.swing_tp.clamp(0.001, 0.03);
                child.swing_sl = child.swing_sl.clamp(0.0005, 0.01);
                child.scalp_z_target = child.scalp_z_target.clamp(0.5, 5.0);
                child.capital_split_scalp = child.capital_split_scalp.clamp(0.1, 1.0);
                child.min_confidence = child.min_confidence.clamp(0.5, 0.99);
                child.explosive_leverage_multiplier = child.explosive_leverage_multiplier.clamp(1.0, 10.0);
                
                next_gen.push(child);
            }
            population = next_gen;
        }

        let baseline_results = [current_active];
        let baseline_fitness = {
            let arena = Arc::new(GlobalArena::new(initial_capital));
            baseline_results[0].apply_to_arena(&arena);
            let mut engine = GodEngineCore::new(arena.clone());
            let mut max_drawdown = 0.0;
            let mut peak_capital = initial_capital;
            for tick in &master_stream {
                arena.update_market_data(tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty);
                let (_sc, _sw, c_sc, c_sw, _) = engine.process_tick(
                    tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty, tick.timestamp, &[0.0; 54]);
                if c_sc.is_some() || c_sw.is_some() {
                    let cap = arena.unified_capital.load(Ordering::Relaxed);
                    if cap > peak_capital { peak_capital = cap; }
                    let dd = (peak_capital - cap) / peak_capital;
                    if dd > max_drawdown { max_drawdown = dd; }
                }
            }
            let final_cap = arena.unified_capital.load(Ordering::Relaxed);
            (final_cap - initial_capital) * (1.0 - max_drawdown)
        };

        println!("[Darwin] Online Evolution Complete.");
        println!("         Current Active Fitness: {:.4}", baseline_fitness);
        println!("         Evolved Genome Fitness: {:.4}", best_all_time.1);

        // If the new genome is at least 5% better than the current one on recent data, Hot-Swap!
        if best_all_time.1 > baseline_fitness * 1.05 {
            println!("[Darwin] 🧬 HOT-SWAPPING ACTIVE GENOME! Market regime shift detected.");
            best_all_time.0.apply_to_arena(&self.live_arena);
        } else {
            println!("[Darwin] 🛡️ Current genome is still optimal for this regime.");
        }
    }
}


