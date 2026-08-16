use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use crate::simulation::backtest_engine::{UnifiedConfig, run_backtest_native};


static EVOLUTION_RUNNING: AtomicBool = AtomicBool::new(false);

/// Demonio Evolutivo en Hilo Background (OS-Priority = Idle)
pub fn start_polars_evolver_daemon(
    closes: Arc<Vec<f64>>,
    highs: Arc<Vec<f64>>,
    lows: Arc<Vec<f64>>,
    volumes: Arc<Vec<f64>>,
    base_config: UnifiedConfig,
) {
    if EVOLUTION_RUNNING.swap(true, Ordering::SeqCst) {
        return; // Ya está corriendo
    }
    
    // Configuramos este hilo con prioridad muy baja (Idle) usando OS-Guardian / Windows API
    #[cfg(windows)]
    unsafe {
        let thread = windows::Win32::System::Threading::GetCurrentThread();
        let _ = windows::Win32::System::Threading::SetThreadPriority(
            thread,
            windows::Win32::System::Threading::THREAD_PRIORITY_IDLE,
        );
    }
    
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
            
        rt.block_on(async {
            loop {
                // Dormir 1 hora entre evoluciones masivas
                sleep(Duration::from_secs(3600)).await;
                
                println!("[EVOLVER] 🧬 Iniciando mutación cuántica (Polars Vectorized)...");
                
                let mut best_sharpe = 0.0;
                let mut best_cfg = base_config.clone();
                
                // Simular variaciones de hiperparámetros masivos (Búsqueda de Rentabilidad Exponencial)
                for tp in [0.006, 0.01, 0.025] {
                    for sl in [0.003, 0.005, 0.015] {
                        for lev in [30.0, 50.0, 100.0] {
                            let mut test_cfg = base_config.clone();
                            test_cfg.tp_pct = tp;
                            test_cfg.sl_pct = sl;
                            test_cfg.scalp_leverage = lev;
                            test_cfg.dyn_atr_min = 0.0001; // FIjo por ahora para no explotar combinaciones
                            test_cfg.dyn_obi = 0.10;
                            test_cfg.dyn_ema = 0.00005;
                            test_cfg.dyn_ofi = 0.05; 
                            
                            let mut pnl = vec![0.0];
                            let mut stats = vec![0.0; 4];
                            let _final_cap = run_backtest_native(
                                &closes, &highs, &lows, &volumes, &test_cfg, &mut pnl, &mut stats, "SIM"
                            );
                            
                            // Native backtest populates stats exactly as follows:
                            // 0: net_win_rate, 1: trades, 2: final_cap, 3: max_dd, 4: sharpe
                            let trades = stats[1];
                            let final_capital = stats[2];
                            let max_dd = stats[3];
                            let wr = stats[0];
                            
                            let growth_pct = (final_capital - test_cfg.starting_capital) / test_cfg.starting_capital;
                            
                            // Penalización severa por Drawdown 
                            // Ojo: En este caso no tenemos acceso al crate EntropyFitness (parece ser código copiado).
                            // Implementaremos la misma función matemática aquí en línea:
                            let threshold = 0.05;
                            let dd_penalty = if max_dd <= threshold {
                                1.0
                            } else {
                                f64::exp(-(max_dd - threshold) * 20.0).clamp(0.01, 1.0)
                            };
                            
                            let sharpe = growth_pct * wr * (trades.sqrt() / 10.0) * dd_penalty; // Pseudo-Sharpe con premio por volumen y penalizado por DD
                            
                            if sharpe > best_sharpe && growth_pct > 0.0 {
                                best_sharpe = sharpe;
                                best_cfg = test_cfg.clone();
                            }
                        }
                    }
                }
                
                println!("[EVOLVER] 🏆 Nuevo genotipo élite encontrado (Pseudo-Sharpe: {:.2})", best_sharpe);
                
                let mut new_genome = crate::core::config::Genome::bootstrap_seed();
                new_genome.global.global_leverage = best_cfg.scalp_leverage;
                new_genome.scalp.tp_atr_mult = best_cfg.tp_pct;
                new_genome.scalp.sl_atr_mult = best_cfg.sl_pct;
                new_genome.swing.tp_atr_mult = best_cfg.tp_pct * 2.0;
                new_genome.swing.sl_atr_mult = best_cfg.sl_pct * 2.0;
                new_genome.swing.ml_threshold_long = best_cfg.ml_threshold_l;
                new_genome.swing.ml_threshold_short = best_cfg.ml_threshold_s;
                
                crate::core::storage::GenomeStorage::save_atomic(&new_genome, "config_dir/genotypes/candidate_genome.rkyv")
                    .unwrap_or_else(|e| eprintln!("❌ Polars Evolver failed to save candidate: {}", e));
                println!("[EVOLVER] 💾 candidate_genome.rkyv generado para Shadow Certification.");
            }
        });
    });
}
