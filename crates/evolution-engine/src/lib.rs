use god_engine_core::GodEngineCore;
use quantum_arena::{GlobalArena, TickEvent};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::time::sleep;

pub mod fitness;
pub mod selection_stats;
pub mod anti_bias_governor;
pub mod ast_mutator;
pub mod cma_es;
pub mod crossover_cauchy;
pub mod entropy_fitness;
pub mod meta;
pub mod moe_neat_arena;
pub mod neat;
pub mod online_daemon;
pub mod return_evidence;
pub mod online_random_forest;
pub mod polars_evolver;
pub mod random_forest;

pub use crossover_cauchy::EvolutionaryOperators;
use meta::MetaEvolver;
pub use moe_neat_arena::{ParetoCandidate, fast_non_dominated_sort};
use quantum_arena::genome::SuperGenotype as Genotype;

/// MOD3/5-011 (INFORME 14) — macro CMA-ES CONGELADA: fallbacks de los slots
/// macro del tensor 54D que la evaluación viva inyecta a `process_event`.
///
/// Antes eran literales 2024 duplicados inline en DOS bloques (train y OOS)
/// — el entrenamiento servía un mercado macro que ya no existe y ningún
/// cambio aquí era visible. Se parametrizan como ÚNICO punto de verdad,
/// alineados con los defaults de `OmniState::new()` (data-pipeline).
///
/// TODO: cablear a omni_state real (MOD3/5-011). Este loop solo recibe
/// `Arc<GlobalArena>`, cuyo registry NO expone índices macro (nadie los
/// escribe ahí) — hace falta pasar el `Arc<OmniState>` del host
/// (god_engine.rs) a `start_evolution_loop` y reemplazar estas consts por
/// `omni_state.get_features()` en vivo.
pub mod frozen_macro {
    pub const DXY: f64 = 104.2;
    pub const SP500: f64 = 5120.0;
    pub const NASDAQ: f64 = 18100.0;
    pub const VIX: f64 = 18.5;
    pub const US10Y: f64 = 4.25;
    pub const GOLD: f64 = 2320.0;
    pub const OIL_WTI: f64 = 81.0;
    pub const FED_RATE: f64 = 5.25;
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
        let mut mutation_rate = self
            .arena
            .config
            .quantum_mutation_rate
            .load(Ordering::Relaxed);
        let meta_evolver = MetaEvolver::new(self.arena.clone());
        let mut cma_es_optimizer: Option<crate::cma_es::CmaEsOptimizer> = None;

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
                let minimum_viable_wr =
                    (self.arena.config.ml_threshold_long.load(Ordering::Relaxed) * 0.70)
                        .clamp(0.40, 0.60);
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

            // FASE 3: Persistir el verdadero Optimizador CMA-ES + PSO sin destruir la matriz de covarianza (D-140)
            if cma_es_optimizer.is_none() {
                let mut opt = match crate::cma_es::CmaEsOptimizer::try_new(
                    Genotype::DIMENSION,
                    mutation_rate,
                    Some(pop_size),
                ) {
                    Ok(opt) => opt,
                    Err(error) => {
                        eprintln!("[CMA] Configuración rechazada: {error:?}");
                        continue;
                    }
                };
                // D-405: Centroide en hipercubo canónico unitario [0.0, 1.0]^DIMENSION
                opt.mean = current_alpha.to_normalized_vector();
                opt.global_best = opt.mean.clone();
                cma_es_optimizer = Some(opt);
            }
            let optimizer = cma_es_optimizer.as_mut().expect("initialized above");
            // FMT-013: persist CSA learning; only CHANGES in the supervisor
            // level rescale sigma. This is not a covariance/path restart.
            let step = match optimizer.apply_exploration_level(mutation_rate) {
                Ok(step) => step,
                Err(error) => {
                    eprintln!("[CMA] Ajuste de exploración rechazado: {error:?}");
                    continue;
                }
            };
            println!(
                "[CMA PASO] generación={} sigma={:.8e}->{:.8e} nivel={:.8e}->{:.8e} lambda={} solicitada={}",
                optimizer.generation, step.previous_sigma, step.sigma,
                step.previous_level, step.level, optimizer.lambda, pop_size
            );

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
            let cma_samples = optimizer.sample_population(w, c1, c2);
            let mut population: Vec<Genotype> = Vec::with_capacity(cma_samples.len());

            // Generate genotypes from CMA-ES vectors via proyección afín canónica (D-405)
            for vec in &cma_samples {
                population.push(Genotype::from_normalized_vector(vec));
            }
            // FIX BLOQUEO #1: El Alpha ya es el centroide del CMA-ES (mean).
            // NO sobrescribimos la población manualmente para no destruir la matriz de covarianza
            // y permitir una exploración y explotación matemáticamente puras.

            let initial_capital = self.arena.unified_capital.load(Ordering::Relaxed);

            let chunk_size = rayon::current_num_threads().max(1);
            let mut results: Vec<_> = Vec::with_capacity(population.len());

            for (chunk_idx, chunk) in population.chunks(chunk_size).enumerate() {
                let chunk_res: Vec<_> = chunk
                    .par_iter()
                    .enumerate()
                    .map(|(local_i, genome)| {
                        let i = chunk_idx * chunk_size + local_i;
                        let genome_clone = genome.clone();
                        let test_arena = GlobalArena::build_in_own_stack(initial_capital);

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
                        let mut last_train_minute = 0u64;
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

                            // D-514: Mapeo topológico multidimensional completo 1:1 para CMA-ES (Cero ceguera dimensional)
                            let mut omni_live = [0.0f64; 54];
                            let live_vol = tick.bid_qty + tick.ask_qty;
                            let live_ofi = if live_vol > 0.0 {
                                (tick.bid_qty - tick.ask_qty) / live_vol
                            } else {
                                0.0
                            };
                            let mid = (tick.bid_price + tick.ask_price) * 0.5;
                            let spread_bps = if mid > 0.0 {
                                (tick.ask_price - tick.bid_price) / mid * 10000.0
                            } else {
                                0.0
                            };

                            omni_live[0] = mid;
                            omni_live[1] = live_vol;
                            omni_live[2] = spread_bps;
                            omni_live[3] = (live_vol * mid) / 1000.0;
                            omni_live[4] = (live_ofi * 100.0).clamp(-100.0, 100.0);
                            omni_live[5] = 50.0;
                            omni_live[10] = live_vol * live_ofi.abs();
                            omni_live[11] = 0.0001;
                            omni_live[21] = frozen_macro::DXY;
                            omni_live[22] = frozen_macro::SP500;
                            omni_live[23] = frozen_macro::NASDAQ;
                            omni_live[24] = frozen_macro::VIX;
                            omni_live[25] = frozen_macro::US10Y;
                            omni_live[26] = frozen_macro::GOLD;
                            omni_live[27] = frozen_macro::OIL_WTI;
                            omni_live[29] = frozen_macro::FED_RATE;
                            omni_live[30] = tick.bid_qty - tick.ask_qty;
                            omni_live[31] = (tick.bid_qty - tick.ask_qty) * 1.2;
                            omni_live[39] = live_ofi;
                            omni_live[48] = live_ofi.clamp(-1.0, 1.0);
                            omni_live[49] = (live_vol / 100.0).tanh().clamp(-1.0, 1.0);

                            // D-142: Activar is_kline_closed en fronteras de 1 minuto para evaluar genes de swing
                            let cur_minute = tick.timestamp / 60_000;
                            let is_kline_closed =
                                if last_train_minute > 0 && cur_minute > last_train_minute {
                                    last_train_minute = cur_minute;
                                    true
                                } else {
                                    if last_train_minute == 0 {
                                        last_train_minute = cur_minute;
                                    }
                                    false
                                };

                            let (new_order, closed_order) = engine.process_event(
                                tick.coin_id,
                                true, // is_trade = true para evaluar scalp
                                is_kline_closed,
                                true, // is_depth = true para actualizar features
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
                                equity_curve
                                    .push(test_arena.unified_capital.load(Ordering::Relaxed));
                                last_trades = total_trades;
                            }
                        }

                        // FASE OOS: Validar sobre 30% restante (walk-forward)
                        let oos_capital_start = test_arena.unified_capital.load(Ordering::Relaxed);
                        let mut last_oos_minute = 0u64;
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
                            let live_ofi = if live_vol > 0.0 {
                                (tick.bid_qty - tick.ask_qty) / live_vol
                            } else {
                                0.0
                            };
                            // D-514: Mapeo topológico multidimensional completo 1:1 para OOS walk-forward
                            let mut omni_oos = [0.0f64; 54];
                            let mid = (tick.bid_price + tick.ask_price) * 0.5;
                            let spread_bps = if mid > 0.0 {
                                (tick.ask_price - tick.bid_price) / mid * 10000.0
                            } else {
                                0.0
                            };

                            omni_oos[0] = mid;
                            omni_oos[1] = live_vol;
                            omni_oos[2] = spread_bps;
                            omni_oos[3] = (live_vol * mid) / 1000.0;
                            omni_oos[4] = (live_ofi * 100.0).clamp(-100.0, 100.0);
                            omni_oos[5] = 50.0;
                            omni_oos[10] = live_vol * live_ofi.abs();
                            omni_oos[11] = 0.0001;
                            omni_oos[21] = frozen_macro::DXY;
                            omni_oos[22] = frozen_macro::SP500;
                            omni_oos[23] = frozen_macro::NASDAQ;
                            omni_oos[24] = frozen_macro::VIX;
                            omni_oos[25] = frozen_macro::US10Y;
                            omni_oos[26] = frozen_macro::GOLD;
                            omni_oos[27] = frozen_macro::OIL_WTI;
                            omni_oos[29] = frozen_macro::FED_RATE;
                            omni_oos[30] = tick.bid_qty - tick.ask_qty;
                            omni_oos[31] = (tick.bid_qty - tick.ask_qty) * 1.2;
                            omni_oos[39] = live_ofi;
                            omni_oos[48] = live_ofi.clamp(-1.0, 1.0);
                            omni_oos[49] = (live_vol / 100.0).tanh().clamp(-1.0, 1.0);

                            // D-142: Activar is_kline_closed en fronteras de 1 minuto OOS
                            let cur_minute = tick.timestamp / 60_000;
                            let is_kline_closed =
                                if last_oos_minute > 0 && cur_minute > last_oos_minute {
                                    last_oos_minute = cur_minute;
                                    true
                                } else {
                                    if last_oos_minute == 0 {
                                        last_oos_minute = cur_minute;
                                    }
                                    false
                                };

                            let (_, closed_order) = engine.process_event(
                                tick.coin_id,
                                true,
                                is_kline_closed,
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
                            let variance =
                                returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
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

                        // D-653/D-654 (DÉCIMA OLA) — APTITUD ÚNICA Y CORRECTA.
                        //
                        // Sustituye a `pnl · sharpe / oos_penalty`, que:
                        //   · crecía LINEALMENTE con el apalancamiento (sharpe
                        //     es invariante de escala, pnl no) y sin cota, de
                        //     modo que la evolución escogía invariablemente el
                        //     genoma más apalancado; y
                        //   · puntuaba la INACCIÓN (−1,0) muy por encima de la
                        //     pérdida moderada (−112,5), convirtiendo la
                        //     parálisis operativa en el óptimo local más
                        //     accesible del paisaje.
                        //
                        // La nueva es crecimiento logarítmico penalizado por
                        // ruina: invariante de escala, cóncava en la riqueza y
                        // coherente con el dimensionamiento de Kelly del motor.
                        let max_dd = {
                            let mut peak = initial_capital.max(1e-12);
                            let mut worst = 0.0f64;
                            for &e in equity_curve.iter() {
                                if e > peak {
                                    peak = e;
                                }
                                if peak > 0.0 {
                                    worst = worst.max((peak - e) / peak);
                                }
                            }
                            worst.clamp(0.0, 1.0)
                        };
                        // MOD3/5-012 (INFORME 14): 3 trades en una ventana de
                        // 4096 ticks (minutos de mercado) es el mínimo
                        // estadístico para una señal direccional. El gen
                        // `min_trades_per_day` del campeón (50.0) exigía aquí
                        // un ritmo IMPOSIBLE en la ventana corta ⇒ todo genoma
                        // era INVIABLE ⇒ estancamiento crónico → recocido
                        // cuántico sin evaluación real. El gen sigue gobernando
                        // el ritmo DIARIO del motor; en ESTE loop el mínimo se
                        // acota a 3 para que la aptitud mida y no filtre.
                        let min_trades_required = self
                            .arena
                            .config
                            .min_trades_per_day
                            .load(Ordering::Relaxed)
                            .clamp(1.0, 3.0) as u32;
                        let normalized_fitness =
                            crate::fitness::compute(&crate::fitness::FitnessInputs {
                                initial_capital,
                                final_capital: final_cap,
                                max_drawdown_pct: max_dd,
                                total_trades: total_trades as u32,
                                min_trades_required,
                                oos_start_capital: oos_capital_start,
                                oos_end_capital: test_arena
                                    .unified_capital
                                    .load(Ordering::Relaxed),
                            });

                        (
                            i,
                            normalized_fitness,
                            pnl,
                            total_trades as usize,
                            sharpe,
                            velocity,
                        )
                    })
                    .collect();
                results.extend(chunk_res);
            }

            // Apply CMA-ES Update (D-140: Preservar matriz de covarianza viva)
            let actual_fee_rate = self.arena.config.max_fee_pct.load(Ordering::Relaxed);
            if let Some(opt) = cma_es_optimizer.as_mut() {
                // The sixth result is capital growth (velocity), NOT live Sharpe.
                // No candidate-specific live reference was measured in this replay.
                let sigma_before = opt.sigma;
                let outcome = opt.update_backtest_only(&cma_samples, &mut results, actual_fee_rate);
                println!(
                    "[CMA EVALUACIÓN] generación={} resultado={outcome:?} sigma={sigma_before:.8e}->{:.8e}",
                    opt.generation, opt.sigma
                );
                if outcome != crate::cma_es::UpdateOutcome::Updated {
                    // A malformed/insufficient generation cannot promote a genome
                    // or trigger the unvalidated anti-stagnation fallback below.
                    continue;
                }
            }

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
                // U-6: motor continuo — un solo modo.
                let mode = TradingHorizon::Continuous;
                // QO-M2.1: quantum_evolver DELETED — valor neutro del genoma
                current_alpha.dynamic_atr_min = 0.0012;
                current_alpha.target_volatility =
                    (1.0f64 * 0.01).clamp(0.005, 0.08);
                current_alpha.funding_rate_sensitivity = 0.5f64.clamp(0.0, 3.0);
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
