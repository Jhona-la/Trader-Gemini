use rand::Rng;

pub struct ChaosMonkey {
    enabled: bool,
    disconnect_probability: f64, // Probability per tick
}

impl ChaosMonkey {
    pub fn new() -> Self {
        let enabled = std::env::var("BINANCE_CHAOS_MODE").unwrap_or_else(|_| "False".to_string()) == "True";
        Self {
            enabled,
            disconnect_probability: 0.0001, // 0.01% chance per tick to disconnect
        }
    }

    /// Determines if the stream should artificially drop the connection
    pub fn should_disconnect(&self) -> bool {
        if !self.enabled {
            return false;
        }
        
        let mut rng = rand::thread_rng();
        rng.gen_bool(self.disconnect_probability)
    }
}
