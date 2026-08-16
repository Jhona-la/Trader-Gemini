use crate::core::state::GlobalArena;
use crate::core::god_engine_core::GodEngineCore;
use crate::core::config::Genome;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use rayon::prelude::*;
use rand::Rng;
use crate::core::tick_source::TickEvent;

/// Axioma X: The Darwin Daemon
/// Continuous Online Evolution. Evaluates the recent market microstructure
/// and dynamically hot-swaps parameters without stopping the live engine.
pub struct DarwinDaemon {
    pub live_arena: Arc<GlobalArena>,
    pub current_genome: std::sync::Mutex<Genome>,
}

impl DarwinDaemon {
    pub fn new(live_arena: Arc<GlobalArena>) -> Self {
        Self { 
            live_arena,
            current_genome: std::sync::Mutex::new(Genome::bootstrap_seed()),
        }
    }

    /// Mutates the genome slightly for the evolutionary algorithm
    fn mutate_genome(genome: &mut Genome, rate: f64) {
        let mut rng = rand::thread_rng();
        
        // Global
        if rng.gen_bool(rate) { genome.global.global_leverage *= rng.gen_range(0.8..1.2); }
        if rng.gen_bool(rate) { genome.global.trend_threshold *= rng.gen_range(0.9..1.1); }
        if rng.gen_bool(rate) { genome.global.range_threshold *= rng.gen_range(0.9..1.1); }
        if rng.gen_bool(rate) { genome.global.funding_rate_sensitivity *= rng.gen_range(0.8..1.2); }
        
        // Scalp
        if rng.gen_bool(rate) { genome.scalp.capital_split *= rng.gen_range(0.8..1.2); }
        if rng.gen_bool(rate) { genome.scalp.ml_threshold *= rng.gen_range(0.8..1.2); }
        if rng.gen_bool(rate) { genome.scalp.tp_atr_mult *= rng.gen_range(0.7..1.5); }
        if rng.gen_bool(rate) { genome.scalp.sl_atr_mult *= rng.gen_range(0.7..1.5); }
        if rng.gen_bool(rate) { genome.scalp.trailing_activation_atr *= rng.gen_range(0.8..1.2); }
        
        // Swing
        if rng.gen_bool(rate) { genome.swing.tp_atr_mult *= rng.gen_range(0.7..1.5); }
        if rng.gen_bool(rate) { genome.swing.sl_atr_mult *= rng.gen_range(0.7..1.5); }
        if rng.gen_bool(rate) { genome.swing.min_confidence *= rng.gen_range(0.9..1.1); }
        if rng.gen_bool(rate) { genome.swing.trailing_activation_atr *= rng.gen_range(0.8..1.2); }
        
        // Clamp basic limits
        genome.global.global_leverage = genome.global.global_leverage.clamp(1.0, 125.0);
        genome.scalp.capital_split = genome.scalp.capital_split.clamp(0.1, 1.0);
        genome.scalp.tp_atr_mult = genome.scalp.tp_atr_mult.clamp(0.0005, 0.02);
        genome.scalp.sl_atr_mult = genome.scalp.sl_atr_mult.clamp(0.0005, 0.01);
        genome.swing.tp_atr_mult = genome.swing.tp_atr_mult.clamp(0.001, 0.03);
        genome.swing.sl_atr_mult = genome.swing.sl_atr_mult.clamp(0.0005, 0.01);
        genome.swing.min_confidence = genome.swing.min_confidence.clamp(0.5, 0.99);
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
                    timestamp: tick.timestamp_ms,
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
        
        let current_active = {
            let guard = self.current_genome.lock().unwrap();
            guard.clone()
        };

        let mut population: Vec<Genome> = Vec::with_capacity(pop_size);
        population.push(current_active.clone());
        for _ in 1..pop_size {
            let mut p = current_active.clone();
            Self::mutate_genome(&mut p, mutation_rate);
            population.push(p);
        }

        let initial_capital = self.live_arena.unified_capital.load(Ordering::Relaxed);
        let mut best_all_time = (current_active.clone(), f64::MIN);
        
        // let omni_features = self.live_arena.get_omni_features(); // unused

        for generation in 1..=generations {
            let mut results: Vec<_> = population
                .par_iter()
                .map(|genome| {
                    let mut genome_cfg = genome.clone();
                    genome_cfg.global.base_capital = initial_capital;
                    let config = crate::core::config::QuantumConfig::new_from_genome(&genome_cfg);
                    let arena = Arc::new(GlobalArena::new(config));
                    arena.config.global_max_drawdown.store(0.95, Ordering::Relaxed);
                    
                    let mut engine = GodEngineCore::new(arena.clone());
                    let mut max_drawdown = 0.0;
                    let mut peak_capital = initial_capital;

                    for tick in &master_stream {
                        arena.update_market_data(tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty, tick.timestamp);
                        let (_sc, _sw, c_sc, c_sw, _, _, _) = engine.process_tick(
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
                let p1 = &results[rand::thread_rng().gen_range(0..(pop_size/2))].0;
                let p2 = &results[rand::thread_rng().gen_range(0..(pop_size/2))].0;
                
                // Crossover
                let mut child = if rand::thread_rng().gen_bool(0.5) { p1.clone() } else { p2.clone() };
                
                // Mutate
                Self::mutate_genome(&mut child, mutation_rate);
                
                next_gen.push(child);
            }
            population = next_gen;
        }

        let baseline_fitness = {
            let mut genome_cfg = current_active.clone();
            genome_cfg.global.base_capital = initial_capital;
            let config = crate::core::config::QuantumConfig::new_from_genome(&genome_cfg);
            let arena = Arc::new(GlobalArena::new(config));
            let mut engine = GodEngineCore::new(arena.clone());
            let mut max_drawdown = 0.0;
            let mut peak_capital = initial_capital;
            for tick in master_stream {
                arena.update_market_data(tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty, tick.timestamp);
                let (_sc, _sw, c_sc, c_sw, _, _, _) = engine.process_tick(
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
            let mut max_vol: f64 = 0.0;
            for coin in self.live_arena.coins.iter() {
                let v = coin.atr_pct.load(Ordering::Relaxed);
                if v > max_vol { max_vol = v; }
            }
            let margin = self.live_arena.used_margin.load(Ordering::Relaxed);
            let cap = self.live_arena.unified_capital.load(Ordering::Relaxed);

            match self.live_arena.registry.validate_hot_swap_risk(best_all_time.0.global.global_leverage, max_vol, margin, cap) {
                Ok(_) => {
                    println!("[Darwin] 🧬 HOT-SWAPPING ACTIVE GENOME! Market regime shift detected.");
                    
                    let old = &current_active;
                    let new = &best_all_time.0;
                    
                    // --- 🔬 DELTA TRACKER FORENSE ---
                    println!("================== 🧬 OMNISCIENT MUTATION DELTA TRACKER 🧬 ==================");
                    println!("  Global Leverage   : {:.2}x  ->  {:.2}x  ({:+.2}%)", old.global.global_leverage, new.global.global_leverage, ((new.global.global_leverage - old.global.global_leverage)/old.global.global_leverage) * 100.0);
                    println!("  Trend Threshold   : {:.2}  ->  {:.2}  ({:+.2}%)", old.global.trend_threshold, new.global.trend_threshold, ((new.global.trend_threshold - old.global.trend_threshold)/old.global.trend_threshold) * 100.0);
                    println!("  Capital Split     : {:.2}  ->  {:.2}", old.scalp.capital_split, new.scalp.capital_split);
                    println!("  Scalp TP / SL     : {:.4}/{:.4}  ->  {:.4}/{:.4}", old.scalp.tp_atr_mult, old.scalp.sl_atr_mult, new.scalp.tp_atr_mult, new.scalp.sl_atr_mult);
                    println!("  Swing TP / SL     : {:.4}/{:.4}  ->  {:.4}/{:.4}", old.swing.tp_atr_mult, old.swing.sl_atr_mult, new.swing.tp_atr_mult, new.swing.sl_atr_mult);
                    println!("  Scalp Trail Act.  : {:.2}  ->  {:.2}", old.scalp.trailing_activation_atr, new.scalp.trailing_activation_atr);
                    println!("=============================================================================");

                    self.live_arena.config.update_from_genome(&best_all_time.0);
                    let mut guard = self.current_genome.lock().unwrap();
                    *guard = best_all_time.0.clone();
                },
                Err(e) => {
                    println!("🛡️ [Darwin VETO] Mutación rechazada por OmniscientRegistry: {}", e);
                }
            }
        } else {
            println!("[Darwin] 🛡️ Current genome is still optimal for this regime.");
        }
    }
}
