use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[derive(Serialize, Deserialize, Debug)]
pub struct DynamicConfig {
    pub sl_pct: f32,
    pub tp_pct: f32,
    pub ml_threshold_l: f32,
    pub ml_threshold_s: f32,
    pub tech_threshold_l: f32,
    pub tech_threshold_s: f32,
    #[serde(default)]
    pub scalp_leverage: f32,
    #[serde(default)]
    pub swing_leverage: f32,
}

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

    // Compile DynamicConfig
    if let Ok(mut file) = File::open("data/dynamic_config.json") {
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        if let Ok(config) = serde_json::from_str::<DynamicConfig>(&contents) {
            let encoded = bincode::serialize(&config).unwrap();
            let mut bin_file = File::create("data/dynamic_config.bin").unwrap();
            bin_file.write_all(&encoded).unwrap();
            println!("✅ dynamic_config.json -> dynamic_config.bin ({} bytes)", encoded.len());
        }
    } else {
        println!("⚠️ dynamic_config.json not found.");
    }

    // Compile NanoForest
    if let Ok(mut file) = File::open("models/nano_forest.json") {
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        if let Ok(forest) = serde_json::from_str::<NanoForest>(&contents) {
            let encoded = bincode::serialize(&forest).unwrap();
            let mut bin_file = File::create("models/nano_forest.bin").unwrap();
            bin_file.write_all(&encoded).unwrap();
            println!("✅ nano_forest.json -> nano_forest.bin ({} bytes)", encoded.len());
        }
    } else {
        println!("⚠️ nano_forest.json not found.");
    }
    
    println!("🏁 Compilation Finished.");
}
