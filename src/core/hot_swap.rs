use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use crate::core::state::GlobalArena;
use crate::core::config::Genome;

pub fn spawn_hot_swap_watcher(arena: Arc<GlobalArena>) {
    tokio::spawn(async move {
        let genome_path = "config_dir/genotypes/candidate_genome.rkyv";
        let mut last_modified = std::time::SystemTime::UNIX_EPOCH;

        loop {
            sleep(Duration::from_secs(5)).await;

            if let Ok(metadata) = std::fs::metadata(genome_path) {
                if let Ok(modified) = metadata.modified() {
                    if modified > last_modified {
                        last_modified = modified;

                        // Intentar cargar y aplicar el nuevo genoma (Hot-Swap)
                        // NOTA DE AUDITORÍA: Usamos fs::read en vez de mmap porque mmap2 en Windows 
                        // bloquea el archivo, impidiendo que el shadow_trader lo sobrescriba atómicamente.
                        // Al ser un archivo de ~100 bytes, fs::read toma nanosegundos desde el cache del SO.
                        match std::fs::read(genome_path) {
                            Ok(bytes) => {
                                // Deserializar con rkyv
                                match rkyv::from_bytes::<Genome>(&bytes) {
                                    Ok(genome) => {
                                        let mut max_vol: f64 = 0.0;
                                        for coin in arena.coins.iter() {
                                            let v = coin.atr_pct.load(std::sync::atomic::Ordering::Relaxed);
                                            if v > max_vol { max_vol = v; }
                                        }
                                        let margin = arena.used_margin.load(std::sync::atomic::Ordering::Relaxed);
                                        let cap = arena.unified_capital.load(std::sync::atomic::Ordering::Relaxed);

                                        match arena.registry.validate_hot_swap_risk(genome.global.global_leverage, max_vol, margin, cap) {
                                            Ok(_) => {
                                                println!("🧬 [HOT-SWAP] Nuevo genoma candidato detectado y validado. Aplicando mutación en caliente...");
                                                arena.config.update_from_genome(&genome);
                                                println!("✅ [HOT-SWAP] Mutación aplicada exitosamente sin detener el motor.");
                                            },
                                            Err(e) => {
                                                println!("🛡️ [HOT-SWAP VETO] Mutación rechazada por OmniscientRegistry: {}", e);
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        eprintln!("⚠️ [HOT-SWAP] Error al deserializar genoma: {:?}", e);
                                    }
                                }
                            },
                            Err(e) => {
                                eprintln!("⚠️ [HOT-SWAP] Error leyendo archivo de mutación: {:?}", e);
                            }
                        }
                    }
                }
            }
        }
    });
}
