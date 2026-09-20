use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

use quantum_engine::config::TensorConfig;

#[derive(Serialize, Deserialize, Debug)]
pub struct NanoForest {
    pub children_left: Vec<i32>,
    pub children_right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f32>,
    pub value: Vec<f32>,
}

fn main() {
    println!("🚀 [CONFIG COMPILER] Starting Zero-Copy Compilation...");

    // FIX #733: Manejo de errores defensivo sin unwrap en compilación de configuraciones
    // Compile DynamicConfig
    if let Ok(mut file) = File::open("data/dynamic_config.json") {
        let mut contents = String::new();
        if file.read_to_string(&mut contents).is_ok() {
            if let Ok(config) = serde_json::from_str::<TensorConfig>(&contents) {
                if let Ok(encoded) = bincode::serialize(&config) {
                    if let Ok(mut bin_file) = File::create("data/dynamic_config.bin") {
                        let _ = bin_file.write_all(&encoded);
                        println!(
                            "✅ dynamic_config.json -> dynamic_config.bin ({} bytes)",
                            encoded.len()
                        );
                    }
                }
            }
        }
    } else {
        println!("⚠️ dynamic_config.json not found.");
    }

    // Compile NanoForest
    if let Ok(mut file) = File::open("models/nano_forest.json") {
        let mut contents = String::new();
        if file.read_to_string(&mut contents).is_ok() {
            if let Ok(forest) = serde_json::from_str::<NanoForest>(&contents) {
                if let Ok(encoded) = bincode::serialize(&forest) {
                    if let Ok(mut bin_file) = File::create("models/nano_forest.bin") {
                        let _ = bin_file.write_all(&encoded);
                        println!(
                            "✅ nano_forest.json -> nano_forest.bin ({} bytes)",
                            encoded.len()
                        );
                    }
                }
            }
        }
    } else {
        println!("⚠️ nano_forest.json not found.");
    }

    // Sincronizar Genoma Campeón a todos los entornos (active.json, active_genome.json, backtest/active.json)
    println!("🧬 [CONFIG COMPILER] Synchronizing Champion Genome...");
    if let Ok(data) = std::fs::read_to_string("config_dir/genotypes/quantum_champion.json") {
        if let Ok(genome) = serde_json::from_str::<quantum_arena::genome::SuperGenotype>(&data) {
            let sanitized = quantum_arena::genome::SuperGenotype::from_vector(&genome.to_vector());
            // CERT-F-14: el "entorno default" hacía remove_var(TG_GENOME_ENV)
            // + promote — que desde D-651 escribe a config_dir/genomes/
            // __UNSET__/ y FALLA silenciosamente (o peor, sin atomic_write
            // crearían un silo basura). Sincronización EXPLÍCITA por entorno
            // de la lista cerrada: demo y backtest. prod NO se toca aquí —
            // la promoción a producción sigue siendo un paso humano
            // deliberado (promote_across_env).
            for target_env in ["demo", "backtest"] {
                std::env::set_var("TG_GENOME_ENV", target_env);
                match quantum_arena::genome_store::GenomeEnvelope::promote(
                    sanitized.clone(),
                    "quantum_champion_sync",
                    &format!(
                        "Sincronización de genoma calibrado sin ruido browniano (RR >= 2.25, SL 0.80%) → {}",
                        target_env
                    ),
                ) {
                    Ok(env) => println!(
                        "✅ [{}] GenomeEnvelope promovido a generación {}",
                        target_env, env.generation
                    ),
                    Err(e) => eprintln!("❌ [{}] Error promoviendo genoma: {}", target_env, e),
                }
            }
        } else {
            eprintln!("⚠️ quantum_champion.json no pudo deserializarse como SuperGenotype.");
        }
    } else {
        eprintln!("⚠️ config_dir/genotypes/quantum_champion.json no encontrado.");
    }

    println!("🏁 Compilation Finished.");
}
