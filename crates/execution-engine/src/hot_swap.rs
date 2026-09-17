use std::sync::atomic::{AtomicBool, Ordering};

pub struct HotSwapController {
    pub is_production: AtomicBool,
    pub estasis_threshold: f64,
}

impl HotSwapController {
    pub fn new(threshold: f64) -> Self {
        Self {
            is_production: AtomicBool::new(false),
            estasis_threshold: threshold,
        }
    }

    /// Evaluates if the system has reached statistical profitability (Probability Stasis)
    /// to warrant a hot-swap from Demo to Production.
    pub fn evaluate_estasis(&self, current_win_rate: f64, expected_ev: f64) -> bool {
        if !self.is_production.load(Ordering::Relaxed) {
            if current_win_rate.is_finite() && expected_ev.is_finite() {
                if current_win_rate > 0.55 && expected_ev > self.estasis_threshold {
                    return true; // Threshold met, ready for swap
                }
            }
        }
        false
    }

    /// Executes the hot swap
    pub fn execute_swap(&self) {
        println!("🔥 [HOT-SWAP] Executing Zero-Downtime Transition to PRODUCTION MAINNET!");
        self.is_production.store(true, Ordering::Release);
        // Here we would signal the websocket client to reconnect using real API keys
        // without dropping the in-memory state of the AI models.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_swap_controller_evaluate_and_swap() {
        let controller = HotSwapController::new(0.02);
        assert!(!controller.is_production.load(Ordering::Relaxed));

        // WR <= 0.55 or EV <= threshold should not trigger
        assert!(!controller.evaluate_estasis(0.50, 0.05));
        assert!(!controller.evaluate_estasis(0.60, 0.01));

        // NaN inputs must return false
        assert!(!controller.evaluate_estasis(f64::NAN, 0.05));
        assert!(!controller.evaluate_estasis(0.60, f64::NAN));

        // Valid conditions trigger true
        assert!(controller.evaluate_estasis(0.65, 0.03));

        // Execute swap
        controller.execute_swap();
        assert!(controller.is_production.load(Ordering::Relaxed));

        // Once in production, evaluate_estasis returns false
        assert!(!controller.evaluate_estasis(0.65, 0.03));
    }
}
