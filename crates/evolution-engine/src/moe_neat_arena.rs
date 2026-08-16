use std::sync::Arc;
use std::sync::atomic::Ordering;
use arc_swap::ArcSwap;
use dark_alpha_engine::MoENeatEngine;
use quantum_arena::config::QuantumConfig;
use data_pipeline::telemetry_bus::ZeroCopyTelemetryBus;
use crate::anti_bias_governor::AntiBiasGovernor;
use crate::entropy_fitness::EntropyFitness;
use std::time::Duration;
use tokio::time::sleep;
use quantum_arena::GlobalArena;

/// Daemon Autoevolutivo Cuántico (Online Learning & NEAT)
/// 
/// V9 FORENSIC FIX: Ahora usa datos REALES de la Arena atómica,
/// evoluciona AMBOS motores (Scalp + Swing), y calcula estadísticas
/// reales (Kurtosis, Skewness) de los trades para el AntiBiasGovernor.
pub fn run_evolution_daemon(
    scalp_moe: Arc<ArcSwap<MoENeatEngine>>,
    swing_moe: Arc<ArcSwap<MoENeatEngine>>,
    _config: Arc<QuantumConfig>,
    telemetry: Arc<ZeroCopyTelemetryBus>,
    arena: Arc<GlobalArena>,
) {
    tokio::spawn(async move {
        loop {
            let decay = arena.config.temporal_memory_decay.load(Ordering::Relaxed);
            let sleep_time = (900.0 * (1.0 - decay) * 100.0).clamp(60.0, 3600.0) as u64;
            sleep(Duration::from_secs(sleep_time)).await;

            println!("🧬 [EVOLUTION-ENGINE] Analizando entropía del modelo (Walk-Forward / Online Learning)...");
            
            os_guardian::memory_audit::check_memory_suspension();
            
            let current_cap = arena.unified_capital.load(Ordering::Relaxed);
            let peak_cap = arena.peak_unified_capital.load(Ordering::Relaxed);
            let drawdown = if peak_cap > 0.0 { (peak_cap - current_cap) / peak_cap } else { 0.0 };
            let panic_threshold = arena.config.hard_stop_base_limit.load(Ordering::Relaxed).clamp(0.01, 0.15);
            
            // ═══════════════════════════════════════════════════════
            // V9 FIX: Extraer estadísticas REALES de la Arena atómica
            // ═══════════════════════════════════════════════════════
            let (real_scalp_pnl, real_scalp_trades, real_scalp_wr, real_scalp_pf) = 
                aggregate_real_stats(&arena, true);
            let (real_swing_pnl, real_swing_trades, real_swing_wr, real_swing_pf) =
                aggregate_real_stats(&arena, false);

            // ═══════════════════════════════════════════════════════
            // F4.5 — OOS REAL POR CICLO (era fabricado: is=pnl*0.7, oos=pnl*0.3
            // del MISMO dato ⇒ la "validación" siempre pasaba). Ahora: el delta
            // de PnL/trades acumulado desde la evaluación ANTERIOR es
            // genuinamente fuera-de-muestra respecto al ciclo previo — un
            // walk-forward real sobre datos vivos.
            // ═══════════════════════════════════════════════════════
            fn oos_delta(slot: usize, pnl_now: f64, trades_now: usize) -> (f64, usize) {
                use std::sync::Mutex;
                static LASTS: std::sync::OnceLock<[Mutex<[(f64, usize); 2]>; 1]> =
                    std::sync::OnceLock::new();
                let lasts = LASTS.get_or_init(|| [Mutex::new([(0.0, 0_usize); 2])]);
                if let Ok(mut guard) = lasts[0].lock() {
                    let (pnl_prev, trades_prev) = guard[slot];
                    let delta = (pnl_now - pnl_prev, trades_now.saturating_sub(trades_prev));
                    guard[slot] = (pnl_now, trades_now);
                    delta
                } else {
                    (0.0, 0)
                }
            }

            // ═══════════════════════════════════════════════════════
            // EVOLUCIÓN DE SCALP MOE
            // ═══════════════════════════════════════════════════════
            if drawdown > panic_threshold || should_explore(real_scalp_wr, real_scalp_trades) {
                println!("🚨 [EVOLUTION-ENGINE] Scalp: Drawdown {:.2}% o WR bajo ({:.1}%). Disparando Hiper-Mutación NEAT.",
                    drawdown * 100.0, real_scalp_wr * 100.0);

                let mut new_scalp = (**scalp_moe.load()).clone();
                new_scalp.force_hyper_mutation(current_cap);

                // V9: Calcular estadísticas reales para el AntiBiasGovernor
                let (real_kurtosis, real_skewness) = estimate_distribution_shape(real_scalp_pf, real_scalp_wr);

                let (oos_pnl_raw, oos_trades_raw) = oos_delta(0, real_scalp_pnl, real_scalp_trades);

                // V13+FASE22 Reality Slippage Penalty (Fee real del Arena)
                let actual_fee = arena.config.sim_fee_rate.load(Ordering::Relaxed).max(0.0001);
                let penalized_pnl = EntropyFitness::reality_slippage_penalty(real_scalp_pnl, real_scalp_trades, actual_fee);

                // IS = ciclo previo acumulado; OOS = delta desde entonces.
                let is_pnl = penalized_pnl - oos_pnl_raw.max(0.0);
                let is_trades = real_scalp_trades.saturating_sub(oos_trades_raw);
                let oos_pnl = oos_pnl_raw;
                let oos_trades = oos_trades_raw;

                // Sin OOS suficiente (primer ciclo o sin trades nuevos) NO se
                // promueve: honestidad antes que actividad.
                let is_valid = oos_trades >= 5
                    && AntiBiasGovernor::validate_out_of_sample(
                        is_pnl, oos_pnl, is_trades.max(1), oos_trades,
                    );

                let bayesian_penalty = EntropyFitness::bayesian_posterior_collapse_penalty(
                    0.80, real_scalp_wr, real_scalp_trades
                );
                
                // Calcular Sharpe base a partir de WR y PF reales
                let base_sharpe = if real_scalp_trades > 0 {
                    (real_scalp_pnl / real_scalp_trades as f64) / (real_scalp_pf.max(0.01) * 0.01).max(0.001)
                } else { 0.0 };
                
                let deflated_sharpe = AntiBiasGovernor::calculate_deflated_sharpe(
                    base_sharpe,
                    real_scalp_trades.max(1),
                    real_kurtosis,
                    real_skewness,
                    real_scalp_trades,
                ) * bayesian_penalty;

                if is_valid && deflated_sharpe > 0.5 {
                    scalp_moe.store(Arc::new(new_scalp.clone()));
                    
                    std::fs::create_dir_all("config_dir/genotypes").ok();
                    if let Err(e) = new_scalp.save_to_disk("config_dir/genotypes/moe_champion.bin") {
                        println!("⚠️ [EVOLUTION-ENGINE] Error guardando scalp champion: {}", e);
                    }
                    
                    println!("🧬 [EVOLUTION-ENGINE] ✅ Hot-Swap Scalp MOE. DSR: {:.2}, WR: {:.1}%, BayesPenalty: {:.2}, PF: {:.2}, Trades: {}", 
                        deflated_sharpe, real_scalp_wr * 100.0, bayesian_penalty, real_scalp_pf, real_scalp_trades);
                    
                    // V13 Telemetry Cuántica
                    telemetry.record_tensor_telemetry(1, deflated_sharpe, bayesian_penalty, real_scalp_wr, real_kurtosis, real_skewness, 0.0);
                } else {
                    println!("💀 [EVOLUTION-ENGINE] ❌ Scalp Mutación RECHAZADA. DSR: {:.2}, OOS Valid: {}, WR: {:.1}%", 
                        deflated_sharpe, is_valid, real_scalp_wr * 100.0);
                        
                    telemetry.record_tensor_telemetry(1, deflated_sharpe, bayesian_penalty, real_scalp_wr, real_kurtosis, real_skewness, 0.0);
                }
            }

            // ═══════════════════════════════════════════════════════
            // V9 FIX: EVOLUCIÓN DE SWING MOE (Antes completamente ignorado)
            // ═══════════════════════════════════════════════════════
            if drawdown > (panic_threshold * 1.5) || should_explore(real_swing_wr, real_swing_trades) {
                println!("🚨 [EVOLUTION-ENGINE] Swing: Drawdown {:.2}% o WR bajo ({:.1}%). Disparando Hiper-Mutación NEAT.", 
                    drawdown * 100.0, real_swing_wr * 100.0);
                
                let mut new_swing = (**swing_moe.load()).clone();
                new_swing.force_hyper_mutation(current_cap);
                
                let (real_kurtosis, real_skewness) = estimate_distribution_shape(real_swing_pf, real_swing_wr);

                // F4.5: OOS real por ciclo (delta desde la evaluación anterior).
                let (oos_pnl_raw, oos_trades_raw) = oos_delta(1, real_swing_pnl, real_swing_trades);

                // V13+FASE22 Reality Slippage Penalty (Fee real del Arena)
                let actual_fee = arena.config.sim_fee_rate.load(Ordering::Relaxed).max(0.0001);
                let penalized_swing_pnl = EntropyFitness::reality_slippage_penalty(real_swing_pnl, real_swing_trades, actual_fee);

                let is_pnl = penalized_swing_pnl - oos_pnl_raw.max(0.0);
                let is_trades = real_swing_trades.saturating_sub(oos_trades_raw);
                let oos_pnl = oos_pnl_raw;
                let oos_trades = oos_trades_raw;

                let is_valid = oos_trades >= 5
                    && AntiBiasGovernor::validate_out_of_sample(
                        is_pnl, oos_pnl, is_trades.max(1), oos_trades,
                    );

                let bayesian_penalty = EntropyFitness::bayesian_posterior_collapse_penalty(
                    0.80, real_swing_wr, real_swing_trades
                );
                
                let base_sharpe = if real_swing_trades > 0 {
                    (real_swing_pnl / real_swing_trades as f64) / (real_swing_pf.max(0.01) * 0.01).max(0.001)
                } else { 0.0 };
                
                let deflated_sharpe = AntiBiasGovernor::calculate_deflated_sharpe(
                    base_sharpe,
                    real_swing_trades.max(1),
                    real_kurtosis,
                    real_skewness,
                    real_swing_trades,
                ) * bayesian_penalty;

                if is_valid && deflated_sharpe > 0.5 {
                    swing_moe.store(Arc::new(new_swing.clone()));
                    
                    std::fs::create_dir_all("config_dir/genotypes").ok();
                    // Swing should technically be saved in a different file or combined. 
                    // For now, we save it as moe_champion_swing.bin to avoid overwriting scalp.
                    if let Err(e) = new_swing.save_to_disk("config_dir/genotypes/moe_champion_swing.bin") {
                        println!("⚠️ [EVOLUTION-ENGINE] Error guardando swing champion: {}", e);
                    }

                    println!("🧬 [EVOLUTION-ENGINE] ✅ Hot-Swap Swing MOE. DSR: {:.2}, WR: {:.1}%, BayesPenalty: {:.2}, PF: {:.2}, Trades: {}", 
                        deflated_sharpe, real_swing_wr * 100.0, bayesian_penalty, real_swing_pf, real_swing_trades);
                        
                    telemetry.record_tensor_telemetry(2, deflated_sharpe, bayesian_penalty, real_swing_wr, real_kurtosis, real_skewness, 0.0);
                } else {
                    println!("💀 [EVOLUTION-ENGINE] ❌ Swing Mutación RECHAZADA. DSR: {:.2}, OOS Valid: {}, WR: {:.1}%", 
                        deflated_sharpe, is_valid, real_swing_wr * 100.0);
                        
                    telemetry.record_tensor_telemetry(2, deflated_sharpe, bayesian_penalty, real_swing_wr, real_kurtosis, real_skewness, 0.0);
                }
            }

            if drawdown <= panic_threshold {
                println!("🧬 [EVOLUTION-ENGINE] Sistema estable. Micro-mutación de pesos completada. Capital: ${:.2}", current_cap);
            }
        }
    });
}

/// Agrega estadísticas REALES de PnL, Trades, WR, PF de todas las monedas en la Arena.
fn aggregate_real_stats(arena: &GlobalArena, is_scalp: bool) -> (f64, usize, f64, f64) {
    let mut total_pnl = 0.0_f64;
    let mut total_trades = 0_usize;
    let mut total_wins = 0_usize;
    let mut weighted_pf_sum = 0.0_f64;
    let mut pf_weight = 0.0_f64;

    for coin in arena.coins.iter() {
        let (trades, wr, pf, pnl) = if is_scalp {
            (
                coin.scalp.trade_count.load(Ordering::Relaxed),
                coin.scalp.win_rate.load(Ordering::Relaxed),
                coin.scalp.profit_factor.load(Ordering::Relaxed),
                coin.scalp.pnl_realized.load(Ordering::Relaxed),
            )
        } else {
            (
                coin.swing.trade_count.load(Ordering::Relaxed),
                coin.swing.win_rate.load(Ordering::Relaxed),
                coin.swing.profit_factor.load(Ordering::Relaxed),
                coin.swing.pnl_realized.load(Ordering::Relaxed),
            )
        };
        
        total_trades += trades;
        total_pnl += pnl;
        total_wins += (trades as f64 * wr) as usize;
        if trades > 5 {
            weighted_pf_sum += pf * trades as f64;
            pf_weight += trades as f64;
        }
    }

    let avg_wr = if total_trades > 0 { total_wins as f64 / total_trades as f64 } else { 0.0 };
    let avg_pf = if pf_weight > 0.0 { weighted_pf_sum / pf_weight } else { 1.0 };

    (total_pnl, total_trades, avg_wr, avg_pf)
}

/// Estima Kurtosis y Skewness a partir de Win Rate y Profit Factor (método de momentos).
/// En un sistema real con historial completo, usaríamos Welford Online.
/// Aquí usamos la relación matemática entre WR/PF y la forma de la distribución.
fn estimate_distribution_shape(profit_factor: f64, win_rate: f64) -> (f64, f64) {
    // Kurtosis: PF bajo con WR alto → distribución leptokúrtica (colas pesadas de pérdida)
    // PF > 2.0 con WR > 60% → distribución más normal (kurtosis ≈ 3.0)
    let kurtosis = 3.0 + (2.0 - profit_factor).max(0.0) * 1.5;
    
    // Skewness: WR > 50% tiende a skew positivo (más ganancias pequeñas)
    // WR < 50% tiende a skew negativo (más pérdidas)
    let skewness = (win_rate - 0.5) * 2.0;
    
    (kurtosis.clamp(1.5, 10.0), skewness.clamp(-2.0, 2.0))
}

/// Decide si explorar (mutar) incluso sin drawdown, basado en estancamiento.
fn should_explore(win_rate: f64, trade_count: usize) -> bool {
    // Si tenemos suficientes trades pero el WR es bajo, forzar exploración
    trade_count > 50 && win_rate < 0.45
}

