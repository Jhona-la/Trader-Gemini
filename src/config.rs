use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorConfig {
    pub symbols: Vec<String>,
    pub is_testnet: bool,
}
