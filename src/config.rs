use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DynamicConfig {
    #[serde(default = "default_symbols")]
    pub symbols: Vec<String>,
    pub sl_pct: f32,
    pub tp_pct: f32,
    pub ml_threshold_l: f32,
    pub ml_threshold_s: f32,
    pub tech_threshold_l: f32,
    pub tech_threshold_s: f32,
    #[serde(default = "default_scalp_lev")]
    pub scalp_leverage: f32,
    #[serde(default = "default_swing_lev")]
    pub swing_leverage: f32,
}

fn default_symbols() -> Vec<String> { vec!["btcusdt".to_string(), "ethusdt".to_string()] }
fn default_scalp_lev() -> f32 { 50.0 }
fn default_swing_lev() -> f32 { 15.0 }

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NanoForest {
    pub children_left: Vec<i32>,
    pub children_right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f32>,
    pub value: Vec<f32>,
}
