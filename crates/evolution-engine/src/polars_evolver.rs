use backtest_engine::run_backtest_native;
use quantum_arena::genome::SuperGenotype as Genotype;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::time::sleep;

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
    base_config: Genotype,
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

                // Simular variaciones de hiperparámetros estocásticos (1000 iteraciones cuánticas)
                for _ in 0..1000 {
                    let mut test_cfg = Genotype::new_random();
                    // Merge some base properties or rely entirely on random

                    // F3.3: buffers según el CONTRATO del motor (antes: pnl de
                    // tamaño 1 y stats de 4 → OOB garantizado al primer trade).
                    let n = closes.len();
                    let mut pnl = vec![0.0; n];
                    let mut stats = vec![0.0; backtest_engine::STATS_LEN];
                    let _final_cap = run_backtest_native(
                        &closes, &highs, &lows, &volumes, &test_cfg, &mut pnl, &mut stats, "SIM",
                        100.0,
                    );

                    // Native backtest populates stats exactly as follows:
                    // 0: net_win_rate, 1: trades, 2: final_cap, 3: max_dd, 4: sharpe
                    let trades = stats[1];
                    let final_capital = stats[2];
                    let max_dd = stats[3];
                    let wr = stats[0];

                    // Penalización severa por Drawdown
                    let dd_penalty =
                        crate::entropy_fitness::EntropyFitness::drawdown_adversarial_penalty(
                            max_dd,
                            test_cfg.global_max_drawdown,
                        );

                    // Penalización por significancia estadística (Cero ghost code)
                    let min_trades_penalty = if trades < 15.0 {
                        (trades / 15.0).max(0.1)
                    } else {
                        1.0
                    };

                    // Pseudo-Sharpe Cuántico con Penalización
                    let sharpe = (final_capital - 100.0) * wr * dd_penalty * min_trades_penalty;

                    if sharpe > best_sharpe {
                        best_sharpe = sharpe;
                        best_cfg = test_cfg.clone();
                    }
                }

                if best_sharpe > 0.0 {
                    println!(
                        "[EVOLVER] 🏆 Nuevo genotipo élite encontrado (Pseudo-Sharpe: {:.2})",
                        best_sharpe
                    );

                    // F4.3 — FIX CORRUPCIÓN DE SCHEMA: este bloque escribía un
                    // struct REPORT legacy (14 campos) sobre active_genome.json,
                    // donde el loader espera el SuperGenotype completo (100+ genes)
                    // — la siguiente promo habría dejado el motor en baseline
                    // silencioso. Ahora: embudo único con linaje y espejo atómico.
                    match quantum_arena::genome_store::GenomeEnvelope::promote(
                        best_cfg.clone(),
                        "polars_evolver",
                        &format!("pseudo-sharpe {:.4}", best_sharpe),
                    ) {
                        Ok(env) => println!(
                            "[EVOLVER] 💾 Genoma generación {} promovido (padre {}).",
                            env.generation, env.parent_generation
                        ),
                        Err(e) => println!("[EVOLVER] ❌ promo falló: {}", e),
                    }
                } // Cierra if best_sharpe > 0.0
            } // Cierra loop
        }); // Cierra rt.block_on
    }); // Cierra thread::spawn
} // Cierra fn
