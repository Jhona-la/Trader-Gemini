use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorConfig {
    pub symbols: Vec<String>,
    pub is_testnet: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NanoForest {
    pub children_left: Vec<i32>,
    pub children_right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f32>,
    pub value: Vec<f32>,
}
