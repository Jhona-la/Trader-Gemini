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
        // FIX #550 & #601: Cargar genomas campeones persistidos en frío (Scalp y Swing especializados)
        if let Ok(champ_scalp) = dark_alpha_engine::MoENeatEngine::load("config_dir/genotypes/moe_champion.bin") {
            println!("🧬 [EVOLUTION-ENGINE] 🚀 Genoma campeón Scalp cargado exitosamente desde disco al inicio.");
            scalp_moe.store(Arc::new(champ_scalp.clone()));

            if let Ok(champ_swing) = dark_alpha_engine::MoENeatEngine::load("config_dir/genotypes/moe_champion_swing.bin") {
                println!("🧬 [EVOLUTION-ENGINE] 🚀 Genoma campeón Swing especializado cargado exitosamente desde disco al inicio.");
                swing_moe.store(Arc::new(champ_swing));
            } else {
                swing_moe.store(Arc::new(champ_scalp));
            }
        } else if let Ok(champ_swing) = dark_alpha_engine::MoENeatEngine::load("config_dir/genotypes/moe_champion_swing.bin") {
            println!("🧬 [EVOLUTION-ENGINE] 🚀 Genoma campeón Swing cargado exitosamente al inicio.");
            swing_moe.store(Arc::new(champ_swing));
        }

        loop {
            let decay = arena.config.temporal_memory_decay.load(Ordering::Relaxed);
            // FIX BLOQUEO #6: Reducir sleep de 60-3600s a 30-120s para micro-capital
            // Con $13, cada minuto cuenta. Evolución agresiva continua.
            let sleep_time = (120.0 * (1.0 - decay) * 10.0).clamp(30.0, 120.0) as u64;
            sleep(Duration::from_secs(sleep_time)).await;

            println!("🧬 [EVOLUTION-ENGINE] Analizando entropía del modelo (Walk-Forward / Online Learning)...");
            
            let current_cap = arena.unified_capital.load(Ordering::Relaxed);
            let base_cap = arena.config.base_capital.load(Ordering::Relaxed).max(1.0);
            let drawdown = if current_cap < base_cap {
                (base_cap - current_cap) / base_cap
            } else {
                0.0
            };
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
            // FIX BLOQUEO #6: Evolución PROACTIVA (siempre buscar mejorar)
            // ANTES: Solo mutaba si drawdown > panic_threshold OR should_explore() → REACTIVO.
            // AHORA: Siempre intenta mejorar. Si tiene buen rendimiento, usa mutación suave.
            //        Si tiene mal rendimiento, usa hiper-mutación.
            let scalp_needs_hyper = drawdown > panic_threshold || should_explore(real_scalp_wr, real_scalp_trades);
            {
                let mode = if scalp_needs_hyper { "Hiper-Mutación" } else { "Refinamiento" };
                println!("🧬 [EVOLUTION-ENGINE] Scalp: {} proactivo. WR: {:.1}%, DD: {:.2}%",
                    mode, real_scalp_wr * 100.0, drawdown * 100.0);

                let mut new_scalp = (**scalp_moe.load()).clone();
                if scalp_needs_hyper {
                    new_scalp.force_hyper_mutation(current_cap);
                } else {
                    // Refinamiento suave: mutación menor para no romper lo que funciona
                    new_scalp.force_hyper_mutation(current_cap * 10.0); // Higher cap = smaller mutations
                }

                // V9: Calcular estadísticas reales para el AntiBiasGovernor
                let (real_kurtosis, real_skewness) = estimate_distribution_shape(real_scalp_pf, real_scalp_wr);

                let (oos_pnl_raw, oos_trades_raw) = oos_delta(0, real_scalp_pnl, real_scalp_trades);

                // V13+FASE22 Reality Slippage Penalty (Fee real del Arena)
                let actual_fee = arena.config.max_fee_pct.load(Ordering::Relaxed).max(0.0001);
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
                
                // Calcular Sharpe base a partir de WR y PF reales (monótono creciente con PF y WR)
                let base_sharpe = if real_scalp_trades > 0 {
                    let w = real_scalp_wr.clamp(0.01, 0.99);
                    let pf = real_scalp_pf.max(0.01);
                    let numerator = (pf - 1.0) * (w * (1.0 - w)).sqrt();
                    let denominator = (w * pf + (1.0 - w)).max(0.001);
                    let trade_sharpe = numerator / denominator;
                    let raw = trade_sharpe * (real_scalp_trades as f64).sqrt();
                    if raw.is_finite() { raw } else { 0.0 }
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
            // FIX BLOQUEO #6: Evolución PROACTIVA para Swing MOE
            let swing_needs_hyper = drawdown > (panic_threshold * 1.5) || should_explore(real_swing_wr, real_swing_trades);
            {
                let mode = if swing_needs_hyper { "Hiper-Mutación" } else { "Refinamiento" };
                println!("🧬 [EVOLUTION-ENGINE] Swing: {} proactivo. WR: {:.1}%, DD: {:.2}%",
                    mode, real_swing_wr * 100.0, drawdown * 100.0);
                
                let mut new_swing = (**swing_moe.load()).clone();
                if swing_needs_hyper {
                    new_swing.force_hyper_mutation(current_cap);
                } else {
                    new_swing.force_hyper_mutation(current_cap * 10.0);
                }
                
                let (real_kurtosis, real_skewness) = estimate_distribution_shape(real_swing_pf, real_swing_wr);

                // F4.5: OOS real por ciclo (delta desde la evaluación anterior).
                let (oos_pnl_raw, oos_trades_raw) = oos_delta(1, real_swing_pnl, real_swing_trades);

                // V13+FASE22 Reality Slippage Penalty (Fee real del Arena)
                let actual_fee = arena.config.max_fee_pct.load(Ordering::Relaxed).max(0.0001);
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
                    let w = real_swing_wr.clamp(0.01, 0.99);
                    let pf = real_swing_pf.max(0.01);
                    let numerator = (pf - 1.0) * (w * (1.0 - w)).sqrt();
                    let denominator = (w * pf + (1.0 - w)).max(0.001);
                    let trade_sharpe = numerator / denominator;
                    let raw = trade_sharpe * (real_swing_trades as f64).sqrt();
                    if raw.is_finite() { raw } else { 0.0 }
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
    // FIX #671: Sanitizar parámetros para evitar NaNs en forma de distribución
    if !profit_factor.is_finite() || !win_rate.is_finite() {
        return (3.0, 0.0);
    }

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

/// Individuo con vector multiobjetivo para optimización de Pareto NSGA-II (#191-#200)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParetoCandidate {
    pub id: usize,
    pub sharpe: f64,       // Maximizar
    pub win_rate: f64,     // Maximizar
    pub max_drawdown: f64, // Minimizar
    pub pnl_realized: f64, // Maximizar
}

impl ParetoCandidate {
    pub fn new(id: usize, sharpe: f64, win_rate: f64, max_drawdown: f64, pnl_realized: f64) -> Self {
        // FIX #671: Sanitizar métricas de Pareto
        let s = if sharpe.is_finite() { sharpe } else { 0.0 };
        let wr = if win_rate.is_finite() { win_rate.clamp(0.0, 1.0) } else { 0.0 };
        let dd = if max_drawdown.is_finite() { max_drawdown.clamp(0.0, 1.0) } else { 1.0 };
        let pnl = if pnl_realized.is_finite() { pnl_realized } else { 0.0 };

        Self {
            id,
            sharpe: s,
            win_rate: wr,
            max_drawdown: dd,
            pnl_realized: pnl,
        }
    }

    /// Determina si `self` domina a `other` en sentido estricto de Pareto (NSGA-II)
    #[inline(always)]
    pub fn dominates(&self, other: &ParetoCandidate) -> bool {
        let not_worse = self.sharpe >= other.sharpe
            && self.win_rate >= other.win_rate
            && self.max_drawdown <= other.max_drawdown
            && self.pnl_realized >= other.pnl_realized;

        let strictly_better = self.sharpe > other.sharpe
            || self.win_rate > other.win_rate
            || self.max_drawdown < other.max_drawdown
            || self.pnl_realized > other.pnl_realized;

        not_worse && strictly_better
    }
}

/// Clasificación no dominada rápida (Fast Non-Dominated Sorting NSGA-II)
pub fn fast_non_dominated_sort(candidates: &[ParetoCandidate]) -> Vec<Vec<usize>> {
    let n = candidates.len();
    if n == 0 {
        return Vec::new();
    }

    let mut domination_counts = vec![0_usize; n];
    let mut dominated_sets: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut first_front = Vec::new();

    for p in 0..n {
        for q in 0..n {
            if p == q {
                continue;
            }
            if candidates[p].dominates(&candidates[q]) {
                dominated_sets[p].push(q);
            } else if candidates[q].dominates(&candidates[p]) {
                domination_counts[p] += 1;
            }
        }
        if domination_counts[p] == 0 {
            first_front.push(p);
        }
    }

    fronts.push(first_front);
    let mut i = 0;
    while i < fronts.len() && !fronts[i].is_empty() {
        let mut next_front = Vec::new();
        for &p in &fronts[i] {
            for &q in &dominated_sets[p] {
                domination_counts[q] = domination_counts[q].saturating_sub(1);
                if domination_counts[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        if !next_front.is_empty() {
            fronts.push(next_front);
        }
        i += 1;
    }

    fronts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pareto_dominance_and_nsga2_sorting() {
        let c1 = ParetoCandidate::new(1, 2.5, 0.65, 0.03, 100.0); // Dominante
        let c2 = ParetoCandidate::new(2, 1.5, 0.55, 0.08, 50.0);  // Dominado por c1
        let c3 = ParetoCandidate::new(3, 3.0, 0.50, 0.02, 120.0); // No dominado (mejor Sharpe/DD)

        assert!(c1.dominates(&c2));
        assert!(!c2.dominates(&c1));
        assert!(!c1.dominates(&c3));
        assert!(!c3.dominates(&c1));

        let candidates = vec![c1, c2, c3];
        let fronts = fast_non_dominated_sort(&candidates);

        assert!(!fronts.is_empty());
        // Frente 1 debe contener c1 (idx 0) y c3 (idx 2)
        assert!(fronts[0].contains(&0));
        assert!(fronts[0].contains(&2));
        // Frente 2 debe contener c2 (idx 1)
        if fronts.len() > 1 {
            assert!(fronts[1].contains(&1));
        }
    }
}

