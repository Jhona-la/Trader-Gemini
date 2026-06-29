use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NanoForest {
    pub children_left: Vec<i32>,
    pub children_right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f32>,
    pub value: Vec<f32>,
}
