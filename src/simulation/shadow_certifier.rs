use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use crate::core::state::GlobalArena;
use std::path::Path;
use std::sync::atomic::Ordering;

/// Demonio que gestiona la Evaluación de Candidatos en el proceso Shadow
pub async fn shadow_tournament_daemon(arena_real: Arc<GlobalArena>, shadow_arenas: Vec<Arc<GlobalArena>>) {
    println!("[SHADOW TOURNAMENT] 🏆 Torneo de Sombras iniciado con {} participantes.", shadow_arenas.len());
    let candidate_path = "config_dir/genotypes/candidate_genome.rkyv";
    let _prod_path = "config_dir/genotypes/production.rkyv";
    let mut last_candidate_time = std::time::SystemTime::UNIX_EPOCH;
    
    // Almacena el último genoma cargado en cada Shadow Arena
    let mut shadow_genomes: Vec<Option<crate::core::config::Genome>> = vec![None; shadow_arenas.len()];

    loop {
        sleep(Duration::from_secs(60)).await;

        let mut real_pnl = 0.0;
        for i in 0..30 {
            real_pnl += arena_real.coins[i].scalp.pnl_realized.load(Ordering::Relaxed)
                      + arena_real.coins[i].swing.pnl_realized.load(Ordering::Relaxed);
        }

        let mut best_shadow_idx = 0;
        let mut best_shadow_pnl = -999999.0;
        let mut worst_shadow_idx = 0;
        let mut worst_shadow_pnl = 999999.0;

        for (idx, arena) in shadow_arenas.iter().enumerate() {
            let mut shadow_pnl = 0.0;
            for i in 0..30 {
                shadow_pnl += arena.coins[i].scalp.pnl_realized.load(Ordering::Relaxed)
                            + arena.coins[i].swing.pnl_realized.load(Ordering::Relaxed);
            }
            if shadow_pnl > best_shadow_pnl {
                best_shadow_pnl = shadow_pnl;
                best_shadow_idx = idx;
            }
            if shadow_pnl < worst_shadow_pnl {
                worst_shadow_pnl = shadow_pnl;
                worst_shadow_idx = idx;
            }
        }

        // 1. Promoción Cuántica del Campeón (Filtro de Realidad Bayesiana & Z-Test)
        let mut best_shadow_trades = 0;
        let mut best_shadow_wins = 0.0;
        let mut real_trades = 0;
        let mut real_wins = 0.0;
        
        for i in 0..30 {
            // Shadow Engine Metrics
            let tr = shadow_arenas[best_shadow_idx].coins[i].scalp.trade_count.load(Ordering::Relaxed)
                   + shadow_arenas[best_shadow_idx].coins[i].swing.trade_count.load(Ordering::Relaxed);
            let wr_sc = shadow_arenas[best_shadow_idx].coins[i].scalp.win_rate.load(Ordering::Relaxed);
            let wr_sw = shadow_arenas[best_shadow_idx].coins[i].swing.win_rate.load(Ordering::Relaxed);
            best_shadow_trades += tr;
            best_shadow_wins += (shadow_arenas[best_shadow_idx].coins[i].scalp.trade_count.load(Ordering::Relaxed) as f64 * wr_sc)
                              + (shadow_arenas[best_shadow_idx].coins[i].swing.trade_count.load(Ordering::Relaxed) as f64 * wr_sw);
            
            // Real Engine Metrics (for Null Hypothesis)
            let r_tr = arena_real.coins[i].scalp.trade_count.load(Ordering::Relaxed)
                     + arena_real.coins[i].swing.trade_count.load(Ordering::Relaxed);
            let r_wr_sc = arena_real.coins[i].scalp.win_rate.load(Ordering::Relaxed);
            let r_wr_sw = arena_real.coins[i].swing.win_rate.load(Ordering::Relaxed);
            real_trades += r_tr;
            real_wins += (arena_real.coins[i].scalp.trade_count.load(Ordering::Relaxed) as f64 * r_wr_sc)
                       + (arena_real.coins[i].swing.trade_count.load(Ordering::Relaxed) as f64 * r_wr_sw);
        }

        let min_trades_required = 30; // Muestra estadística mínima para Z-Test
        let mut p_value = 1.0;
        let mut z_score = 0.0;

        if best_shadow_trades >= min_trades_required {
            let p_hat = best_shadow_wins / (best_shadow_trades as f64);
            // FIX #1466: Clamping de p0 para evitar colapso de varianza binomial en 0 o 1
            let p0 = if real_trades >= 10 { real_wins / (real_trades as f64) } else { 0.55 };
            let p0_clamped = p0.clamp(0.01, 0.99);
            
            // Estadístico Z para una proporción (One-sample Z-test)
            let variance = (p0_clamped * (1.0 - p0_clamped)) / (best_shadow_trades as f64);
            if variance > 0.0 {
                let z = (p_hat - p0_clamped) / variance.sqrt();
                if z.is_finite() {
                    z_score = z;
                    // Aproximación de P-value para distribución normal estándar (One-sided)
                    p_value = (0.5 * (1.0 - libm::erf(z_score / std::f64::consts::SQRT_2))).clamp(0.0, 1.0);
                }
            }
        }

        // Calculamos la divergencia estadística entre Shadow y Real
        let reality_gap = (best_shadow_pnl - real_pnl).abs();
        
        let inverse_shannon_penalty = if reality_gap > 0.0 && reality_gap.is_finite() {
            let prob_divergence = (reality_gap / (real_pnl.abs() + 1.0)).clamp(0.0, 0.999);
            if prob_divergence > 0.0 {
                (-(prob_divergence * prob_divergence.ln()) * 100.0).clamp(0.0, 1000.0)
            } else {
                50.0 
            }
        } else {
            0.0
        };
        
        let deflated_shadow_pnl = best_shadow_pnl - inverse_shannon_penalty;

        // FASE 28: Parity Verification
        // Rigor Estadístico: P-value < 0.05 significa significancia estadística (95% confianza)
        if deflated_shadow_pnl > real_pnl * 1.02 && deflated_shadow_pnl > 0.0 && p_value < 0.05 {
            println!("[SHADOW TOURNAMENT] 👑 Shadow Engine {} superó al Real con Significancia Estadística! (P-value: {:.4}, Z: {:.2}, Shadow Neto: {:.2}, Real: {:.2})", 
                best_shadow_idx, p_value, z_score, deflated_shadow_pnl, real_pnl);
                
            if let Some(genome) = &shadow_genomes[best_shadow_idx] {
                arena_real.config.update_from_genome(genome);
                println!("[SHADOW TOURNAMENT] ⚡ Genoma promovido a PRODUCCIÓN!");
                
                for i in 0..30 {
                    shadow_arenas[best_shadow_idx].coins[i].scalp.pnl_realized.store(0.0, Ordering::Relaxed);
                    shadow_arenas[best_shadow_idx].coins[i].swing.pnl_realized.store(0.0, Ordering::Relaxed);
                    shadow_arenas[best_shadow_idx].coins[i].scalp.trade_count.store(0, Ordering::Relaxed);
                    shadow_arenas[best_shadow_idx].coins[i].swing.trade_count.store(0, Ordering::Relaxed);
                }
            }
        } else if best_shadow_pnl > real_pnl {
            println!("💀 [SHADOW TOURNAMENT] Falsa Alarma o Significancia Insuficiente. Shadow {} PnL ({:.2}), Penalty ({:.2}), P-value ({:.4}, req < 0.05).", 
                best_shadow_idx, best_shadow_pnl, inverse_shannon_penalty, p_value);
        }

        // 2. Reemplazo del Peor (Inyección de nuevos candidatos)
        if Path::new(candidate_path).exists() {
            if let Ok(m) = std::fs::metadata(candidate_path) {
                if let Ok(modified) = m.modified() {
                    if modified > last_candidate_time {
                        println!("[SHADOW TOURNAMENT] 🧪 Inyectando nuevo candidato al peor Shadow Engine (Idx: {})...", worst_shadow_idx);
                        if let Some(genome) = crate::core::storage::GenomeStorage::load_zero_copy(candidate_path) {
                            shadow_arenas[worst_shadow_idx].config.update_from_genome(&genome);
                            shadow_genomes[worst_shadow_idx] = Some(genome);
                            last_candidate_time = modified;
                            
                            // Reset metrics for this shadow
                            for i in 0..30 {
                                shadow_arenas[worst_shadow_idx].coins[i].scalp.pnl_realized.store(0.0, Ordering::Relaxed);
                                shadow_arenas[worst_shadow_idx].coins[i].swing.pnl_realized.store(0.0, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
        }
    }
}


/// Demonio que corre en el proceso Real (Producción). Observa production.rkyv y hace Hot-Reload
pub async fn hot_reload_daemon(arena_real: Arc<GlobalArena>) {
    println!("[HOT RELOAD] ⚡ Demonio de Hot-Reload iniciado en Producción.");
    let prod_path = "config_dir/genotypes/production.rkyv";
    let mut last_prod_time = std::time::SystemTime::UNIX_EPOCH;

    // Setup inicial de tiempo
    if Path::new(prod_path).exists() {
        if let Ok(m) = std::fs::metadata(prod_path) {
            if let Ok(modified) = m.modified() {
                last_prod_time = modified;
            }
        }
    }

    loop {
        sleep(Duration::from_secs(5)).await; // Poll frecuente pero ligero

        if Path::new(prod_path).exists() {
            if let Ok(m) = std::fs::metadata(prod_path) {
                if let Ok(modified) = m.modified() {
                    if modified > last_prod_time {
                        println!("[HOT RELOAD] 🚨 Nuevo Genoma Productivo Detectado! Iniciando inyección lock-free...");
                        if let Some(genome) = crate::core::storage::GenomeStorage::load_zero_copy(prod_path) {
                            // Hot-Swap Atómico a Producción en RAM (Picosegundos)
                            arena_real.config.update_from_genome(&genome);
                            last_prod_time = modified;
                            println!("[HOT RELOAD] ✅ Hot-Swap Atómico Completado con Cero Downtime.");
                        }
                    }
                }
            }
        }
    }
}

