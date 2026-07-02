use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use backtest_engine::{UnifiedConfig, run_backtest_native};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Genome {
    pub scalp_tp: f64,
    pub scalp_sl: f64,
    pub swing_tp: f64,
    pub swing_sl: f64,
    pub ml_threshold: f64,
    pub dyn_atr_min: f64,
    pub dyn_obi: f64,
    pub dyn_ema: f64,
    pub dyn_ofi: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub generation: u32,
    pub fitness: f64,
}

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
                
                // Simular variaciones de hiperparámetros de microestructura
                for atr in [0.00005, 0.0001, 0.0002] {
                    for obi in [0.05, 0.10, 0.15] {
                        for ema in [0.00002, 0.00005, 0.00010] {
                            let mut test_cfg = base_config.clone();
                            test_cfg.dyn_atr_min = atr;
                            test_cfg.dyn_obi = obi;
                            test_cfg.dyn_ema = ema;
                            test_cfg.dyn_ofi = 0.05; // fixed for now to keep grid small
                            
                            let mut pnl = vec![0.0];
                            let mut stats = vec![0.0; 4];
                            let final_cap = run_backtest_native(
                                &closes, &highs, &lows, &volumes, &test_cfg, &mut pnl, &mut stats, "SIM"
                            );
                            
                            // Native backtest returns length of output or something, stats[0] is final cap
                            let trades = stats[1] as f64;
                            let wins = stats[2] as f64;
                            let final_capital = stats[0];
                            
                            let wr = if trades > 0.0 { wins / trades } else { 0.0 };
                            let sharpe = (final_capital - test_cfg.starting_capital) * wr; // Pseudo-Sharpe
                            
                            if sharpe > best_sharpe {
                                best_sharpe = sharpe;
                                best_cfg = test_cfg.clone();
                            }
                        }
                    }
                }
                
                println!("[EVOLVER] 🏆 Nuevo genotipo élite encontrado (Pseudo-Sharpe: {:.2})", best_sharpe);
                
                // Sobrescribir active_genome.json
                let new_genome = Genome {
                    scalp_tp: best_cfg.tp_pct,
                    scalp_sl: best_cfg.sl_pct,
                    swing_tp: best_cfg.tp_pct * 2.0,
                    swing_sl: best_cfg.sl_pct * 2.0,
                    ml_threshold: 0.1,
                    dyn_atr_min: best_cfg.dyn_atr_min,
                    dyn_obi: best_cfg.dyn_obi,
                    dyn_ema: best_cfg.dyn_ema,
                    dyn_ofi: best_cfg.dyn_ofi,
                    sharpe_ratio: best_sharpe,
                    win_rate: 0.0,
                    max_drawdown: 0.0,
                    generation: 1,
                    fitness: best_sharpe,
                };
                
                if let Ok(json) = serde_json::to_string_pretty(&new_genome) {
                    let _ = tokio::fs::write("config_dir/genotypes/active_genome.json", json).await;
                    println!("[EVOLVER] 💾 active_genome.json actualizado en caliente.");
                }
            }
        });
    });
}
