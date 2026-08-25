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

    println!("🏁 Compilation Finished.");
}
