use god_engine_core::GodEngineCore;
use quantum_arena::{GlobalArena, TickEvent};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::time::sleep;

pub mod anti_bias_governor;
pub mod ast_mutator;
pub mod cma_es;
pub mod crossover_cauchy;
pub mod entropy_fitness;
pub mod meta;
pub mod moe_neat_arena;
pub mod neat;
pub mod online_daemon;
pub mod online_random_forest;
pub mod polars_evolver;
pub mod random_forest;

pub use crossover_cauchy::EvolutionaryOperators;
pub use moe_neat_arena::{fast_non_dominated_sort, ParetoCandidate};
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
        let mut mutation_rate = self
            .arena
            .config
            .quantum_mutation_rate
            .load(Ordering::Relaxed);
        let meta_evolver = MetaEvolver::new(self.arena.clone());
        let quantum_evolver = metacortex_engine::QuantumEvolver::new();

        loop {
            // FIX BLOQUEO #1: Reducir sleep de 15s a 5s para micro-capital ($13)
            // La velocidad de iteración evolutiva es crítica con capital bajo.
            sleep(Duration::from_secs(5)).await;

            // FASE 9: Auto-Evolución y Detección de Degradación
            let mut total_wr = 0.0;
            let mut valid_coins = 0;
            for coin_id in 0..self.arena.coins.len() {
                let coin_wr = self.arena.coins[coin_id]
                    .metrics
                    .win_rate
                    .load(Ordering::Relaxed);
                if self.arena.coins[coin_id]
                    .current_price
                    .load(Ordering::Relaxed)
                    > 0.0
                {
                    total_wr += coin_wr;
                    valid_coins += 1;
                }
            }
            if valid_coins > 0 {
                let avg_wr = total_wr / valid_coins as f64;
                // Umbral de viabilidad de Win Rate estadístico (no confundir con umbral de confianza ML de 70%)
                let minimum_viable_wr = (self.arena.config.ml_threshold_long.load(Ordering::Relaxed) * 0.70).clamp(0.40, 0.60);
                if avg_wr < minimum_viable_wr {
                    println!(
                        "🚨 [DEGRADACIÓN DETECTADA] Win Rate Global {:.2}% (Requerido: {:.2}%). Re-activando CMA-ES intenso.",
                        avg_wr * 100.0,
                        minimum_viable_wr * 100.0
                    );
                    mutation_rate = (mutation_rate * 1.5).min(0.5); // Increase mutation rate dynamically if degrading
                } else {
                    mutation_rate = self
                        .arena
                        .config
                        .quantum_mutation_rate
                        .load(Ordering::Relaxed); // Return to baseline
                }
            }

            println!(
                "🧠 [TRUE EVOLUTION] Extrayendo ventana de memoria a corto plazo (LockFreeRing)..."
            );

            let max_capacity = self.arena.coins.len() * 4096;
            let mut all_ticks = Vec::with_capacity(max_capacity);

            for coin_id in 0..self.arena.coins.len() {
                let current_price = self.arena.coins[coin_id]
                    .current_price
                    .load(Ordering::Relaxed);
                if current_price == 0.0 {
                    continue;
                }

                let ticks = self.arena.coins[coin_id].tick_ring.snapshot_recent(4096);

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
            println!(
                "🧠 [TRUE EVOLUTION] Entrenando sobre {} ticks reales. Mutation Rate: {:.1}%",
                ticks_len,
                mutation_rate * 100.0
            );

            // Adaptive Population Size based on the current genome
            let pop_size = 40
                .max((self.arena.config.min_trades_per_day.load(Ordering::Relaxed) * 2.0) as usize)
                .min(100);

            // FASE 3: Instanciar el verdadero Optimizador CMA-ES + PSO
            let mut cma_es_optimizer = crate::cma_es::CmaEsOptimizer::new(
                Genotype::DIMENSION,
                mutation_rate,
                Some(pop_size),
            );
            cma_es_optimizer.mean = current_alpha.to_vector(); // Center around current alpha
            cma_es_optimizer.global_best = cma_es_optimizer.mean.clone();

            let w = self.arena.config.global_momentum.load(Ordering::Relaxed);
            let c1 = self
                .arena
                .config
                .global_learning_rate
                .load(Ordering::Relaxed)
                * 2.0; // cognitive
            let c2 = self
                .arena
                .config
                .global_learning_rate
                .load(Ordering::Relaxed)
                * 2.0; // social
            let cma_samples = cma_es_optimizer.sample_population(w, c1, c2);
            let mut population: Vec<Genotype> = Vec::with_capacity(pop_size);

            // Generate genotypes from CMA-ES vectors
            for vec in &cma_samples {
                population.push(Genotype::from_vector(vec));
            }
            // FIX BLOQUEO #1: El Alpha ya es el centroide del CMA-ES (mean).
            // NO sobrescribimos la población manualmente para no destruir la matriz de covarianza
            // y permitir una exploración y explotación matemáticamente puras.

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
                    test_arena
                        .config
                        .live_maker_fee
                        .store(live_maker, Ordering::Relaxed);
                    test_arena
                        .config
                        .live_taker_fee
                        .store(live_taker, Ordering::Relaxed);

                    let mut engine = GodEngineCore::new(test_arena.clone());
                    let mut total_trades = 0;

                    let mut equity_curve = Vec::with_capacity(100);
                    let mut last_trades = 0;

                    // FIX BLOQUEO #1: Walk-Forward Split 70/30 para evitar sobreajuste
                    let train_end = (all_ticks.len() * 7) / 10;
                    let train_ticks = &all_ticks[..train_end];
                    let oos_ticks = &all_ticks[train_end..];

                    // FASE TRAIN: Evaluar sobre 70% de los ticks
                    for tick in train_ticks {
                        test_arena.update_market_data(
                            tick.coin_id,
                            tick.bid_price,
                            tick.ask_price,
                            tick.bid_qty,
                            tick.ask_qty,
                            tick.timestamp,
                        );
                        let _ml_prob = self.arena.coins[tick.coin_id]
                            .ml_prob
                            .load(Ordering::Relaxed);

                        // FIX BLOQUEO #8: Purgar Falsa Omnisciencia (Data Leakage)
                        // Alinear omni_features con buffers incrementales causales locales. 
                        // Prohibido leer de GLOBAL_TELEONOMIA en simulaciones, contiene datos vivos.
                        let mut omni_live = [0.0f64; 54];
                        let live_vol = tick.bid_qty + tick.ask_qty;
                        let live_ofi = if live_vol > 0.0 { (tick.bid_qty - tick.ask_qty) / live_vol } else { 0.0 };
                        
                        // Inyectamos el flujo micro-estructural ultra rápido en las primeras dimensiones
                        omni_live[0] = tick.bid_price;
                        omni_live[10] = live_vol * live_ofi.abs();
                        omni_live[30] = tick.bid_qty - tick.ask_qty;
                        omni_live[39] = live_ofi;

                        let (new_order, closed_order) = engine.process_event(
                            tick.coin_id,
                            true,  // is_trade = true para evaluar scalp
                            false,
                            true,  // is_depth = true para actualizar features
                            tick.bid_price,
                            live_vol,
                            tick.bid_price,
                            tick.ask_price,
                            tick.bid_qty,
                            tick.ask_qty,
                            live_ofi,
                            0.0,
                            tick.timestamp,
                            false,
                            &omni_live,
                            false, // is_buyer_maker desconocido en replay
                        );
                        if new_order.is_some() || closed_order.is_some() {
                            total_trades += 1;
                        }

                        if total_trades > last_trades {
                            equity_curve.push(test_arena.unified_capital.load(Ordering::Relaxed));
                            last_trades = total_trades;
                        }
                    }

                    // FASE OOS: Validar sobre 30% restante (walk-forward)
                    let oos_capital_start = test_arena.unified_capital.load(Ordering::Relaxed);
                    for tick in oos_ticks {
                        test_arena.update_market_data(
                            tick.coin_id,
                            tick.bid_price,
                            tick.ask_price,
                            tick.bid_qty,
                            tick.ask_qty,
                            tick.timestamp,
                        );
                        let _ml_prob = self.arena.coins[tick.coin_id]
                            .ml_prob
                            .load(Ordering::Relaxed);
                        let live_vol = tick.bid_qty + tick.ask_qty;
                        let live_ofi = if live_vol > 0.0 { (tick.bid_qty - tick.ask_qty) / live_vol } else { 0.0 };
                        let mut omni_oos = [0.0f64; 54];
                        omni_oos[0] = tick.bid_price;
                        omni_oos[39] = live_ofi;
                        omni_oos[30] = tick.bid_qty - tick.ask_qty;

                        let (_, closed_order) = engine.process_event(
                            tick.coin_id,
                            true,
                            false,
                            true,
                            tick.bid_price,
                            live_vol,
                            tick.bid_price,
                            tick.ask_price,
                            tick.bid_qty,
                            tick.ask_qty,
                            live_ofi,
                            0.0,
                            tick.timestamp,
                            false,
                            &omni_oos,
                            false, // is_buyer_maker desconocido en replay
                        );
                        if closed_order.is_some() {
                            total_trades += 1;
                        }
                    }

                    let final_cap = test_arena.unified_capital.load(Ordering::Relaxed);
                    let pnl = final_cap - initial_capital;

                    let sharpe = if equity_curve.len() > 2 {
                        let mut returns = Vec::with_capacity(equity_curve.len());
                        for i in 1..equity_curve.len() {
                            returns.push(
                                (equity_curve[i] - equity_curve[i - 1])
                                    / equity_curve[i - 1].max(1e-10),
                            );
                        }
                        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
                        let variance = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
                            / returns.len() as f64;
                        let std_dev = variance.sqrt();
                        if std_dev > 1e-12 {
                            let trades_per_day = (total_trades as f64).max(1.0);
                            (mean_ret / std_dev) * trades_per_day.sqrt()
                        } else {
                            if mean_ret > 0.0 {
                                mean_ret * 100.0
                            } else {
                                0.0
                            }
                        }
                    } else {
                        if pnl > 0.0 { 0.01 } else { -0.01 }
                    };

                    let velocity = final_cap / initial_capital;

                    // FIX BLOQUEO #1: OOS Walk-Forward Penalty (Matemática continua sin asimetrías abruptas)
                    // Multiplicador simétrico de castigo.
                    let oos_capital_end = test_arena.unified_capital.load(Ordering::Relaxed);
                    let oos_pnl = oos_capital_end - oos_capital_start;
                    let oos_penalty = if oos_pnl < 0.0 { 1.5 } else { 1.0 }; // Penalización del 50% extra en pérdidas
                    
                    let raw_fitness = if pnl > 0.0 && velocity >= 1.0 {
                        (pnl * sharpe.max(0.01) * if total_trades > 0 { 1.0 } else { 0.0 }) / oos_penalty
                    } else {
                        // Castigo monótono, sin división que corrompa la convexidad
                        (pnl.min(0.0) * (1.0 + sharpe.abs()) * oos_penalty) - (if total_trades == 0 { 1.0 } else { 0.0 })
                    };

                    // Normalización logarítmica simétrica para estabilizar matriz de covarianza CMA-ES
                    let normalized_fitness = raw_fitness.signum() * (1.0 + raw_fitness.abs()).ln();

                    // D-136: Empaquetar velocity real en el índice 5 en lugar de duplicar sharpe
                    (i, normalized_fitness, pnl, total_trades as usize, sharpe, velocity)
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

            // D-140: Pasar el Sharpe real (alpha.4) a la auditoría meta-arquitectónica
            meta_evolver.audit_system_architecture(alpha.4);

            // FASE 38: Ajuste Dinámico de la Población.
            // Si el motor encontró un pozo óptimo, aumentamos la entropía reduciendo población
            // Si el motor está estancado, aumentamos la población para buscar en más frentes.

            // D-136: Evaluar velocidad de capital real desde alpha.5
            let is_high_velocity = alpha.5 >= 1.5; // Relajado para backtests cortos, meta = 2.0x en 3 días.

            // D-129: Exigir trades >= 1 Y PnL estrictamente positivo Y fitness positivo antes de hacer hot-swap
            if alpha.3 > 0 && alpha.2 > 0.0 && alpha.1 > 0.0 {
                println!(
                    "🧬 [ALPHA HOT-SWAP] Nuevo Genoma! Sharpe: {:.2} | Fitness: {:.2} | PnL: +${:.2} ({} trades)",
                    alpha.4, alpha.1, alpha.2, alpha.3
                );
                if is_high_velocity {
                    println!("🚀 [VELOCITY TARGET MET] Validando escalabilidad!");
                }
                println!(
                    "   => Leverage: {:.2}x | Scalp TP: {:.2}% | Spread: {:.4}%",
                    next_alpha.global_leverage,
                    next_alpha.scalp_tp_base * 100.0,
                    next_alpha.maker_spread_pct * 100.0
                );

                // Actualizar Alpha y reducir mutación (explotación)
                // N-02 — EMBUDO ÚNICO: el hot-swap pasa ANTES por
                // GenomeEnvelope::promote (gate de dimensionalidad/bounds/RR);
                // solo si el almacén lo sanciona se aplica al arena. El viejo
                // apply+save() directo era un BYPASS del gate que podía
                // instalar un genoma inválido en vivo y escribía la ruta
                // paralela genesis_genome.json sin auditoría ni rollback.
                match quantum_arena::genome_store::GenomeEnvelope::promote(
                    next_alpha.clone(),
                    "evolution_engine",
                    &format!(
                        "alpha hot-swap: sharpe {:.2}, fitness {:.2}, pnl +${:.2} ({} trades)",
                        alpha.4, alpha.1, alpha.2, alpha.3
                    ),
                ) {
                    Ok(env) => {
                        current_alpha = env.genome.clone();
                        env.genome.apply_to_arena(&self.arena);
                        println!(
                            "✅ [TRUE EVOLUTION] Generación {} sancionada y propagada.",
                            env.generation
                        );
                    }
                    Err(e) => println!(
                        "🚫 [N-02] Promoción rechazada por el gate — arena queda intacto: {}",
                        e
                    ),
                }

                println!("✅ [TRUE EVOLUTION] Nueva semilla cuántica Alpha propagada globalmente.");
            } else {
                println!("💀 [ESTANCO] Ningún genoma superó el umbral. Alpha actual se mantiene.");
                // Si el Sharpe decae, aumentamos la tasa de mutación (exploración)
                mutation_rate = (mutation_rate * 1.5).min(0.5);

                // FIX BLOQUEO #7: Colapso cuántico para salir del pozo de estancamiento local
                let latest_ts = all_ticks.last().map(|t| t.timestamp).unwrap_or(42);
                use metacortex_engine::consejo_seniors::TradingHorizon;
                // Add mode depending on the context, defaulting to Scalping if not specified.
                let mode = TradingHorizon::Scalping; // Or extract from context if available
                let q_state = quantum_evolver.anneal_and_collapse(latest_ts, 0.25, mode);
                println!(
                    "⚛️ [QUANTUM-EVOLVER] Recocido cuántico activado. Energy: {:.4} | Window: {} | Thresh: {:.2} | VolMult: {:.2}",
                    q_state.energy, q_state.window_size, q_state.threshold, q_state.volume_multiplier
                );
                current_alpha.dynamic_atr_min = (q_state.threshold * 0.0005).clamp(0.0001, 0.005);
                current_alpha.target_volatility = (q_state.volume_multiplier * 0.01).clamp(0.005, 0.08);
                current_alpha.funding_rate_sensitivity = q_state.funding_weight.clamp(0.0, 3.0);
                // N-02: la micro-mutación del recocido también pasa por el
                // embudo — nada toca el arena sin sanción del almacén.
                match quantum_arena::genome_store::GenomeEnvelope::promote(
                    current_alpha.clone(),
                    "quantum_evolver_collapse",
                    "recocido cuántico anti-estancamiento (3 genes)",
                ) {
                    Ok(env) => {
                        current_alpha = env.genome.clone();
                        env.genome.apply_to_arena(&self.arena);
                    }
                    Err(e) => println!(
                        "🚫 [N-02] Recocido rechazado por el gate — sin cambio: {}",
                        e
                    ),
                }
            }

            // Reflexión Arquitectónica de Fase 9
            meta_evolver.audit_system_architecture(alpha.3 as f64);
        }
    }
}
