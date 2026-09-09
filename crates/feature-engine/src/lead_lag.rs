use std::collections::VecDeque;
use std::f64;

/// ⚡ MOTOR DE LIDERAZGO MACRO MICROESTRUCTURAL LEAD-LAG (CROSS-ASSET ALPHA MATRIX)
/// Rastra la velocidad de propagación de ráfagas institucionales desde BTC/ETH hacia altcoins.
/// Permite entrar en altcoins en nanosegundos antes de que el libro reaccione al impulso macro.
#[derive(Debug, Clone)]
pub struct LeadLagAlphaEngine {
    pub btc_ofi_buffer: VecDeque<f64>,
    pub eth_ofi_buffer: VecDeque<f64>,
    pub max_window: usize,
    pub btc_ewma: f64,
    pub eth_ewma: f64,
}

impl LeadLagAlphaEngine {
    pub fn new(max_window: usize) -> Self {
        Self {
            btc_ofi_buffer: VecDeque::with_capacity(max_window),
            eth_ofi_buffer: VecDeque::with_capacity(max_window),
            max_window: max_window.max(20),
            btc_ewma: 0.0,
            eth_ewma: 0.0,
        }
    }

    #[inline(always)]
    pub fn update_leader(&mut self, is_btc: bool, ofi: f64) {
        if !ofi.is_finite() {
            return;
        }
        if is_btc {
            self.btc_ewma = if self.btc_ofi_buffer.is_empty() {
                ofi
            } else {
                self.btc_ewma * 0.70 + ofi * 0.30
            };
            if self.btc_ofi_buffer.len() >= self.max_window {
                self.btc_ofi_buffer.pop_front();
            }
            self.btc_ofi_buffer.push_back(ofi);
        } else {
            self.eth_ewma = if self.eth_ofi_buffer.is_empty() {
                ofi
            } else {
                self.eth_ewma * 0.70 + ofi * 0.30
            };
            if self.eth_ofi_buffer.len() >= self.max_window {
                self.eth_ofi_buffer.pop_front();
            }
            self.eth_ofi_buffer.push_back(ofi);
        }
    }

    /// Calcula la señal de propagación Cross-Asset para un altcoin en nanosegundos
    #[inline(always)]
    pub fn predict_altcoin_impulse(&self, alt_ofi: f64) -> (f64, f64) {
        if !alt_ofi.is_finite() {
            return (0.0, 0.0);
        }
        let btc_last = self.btc_ofi_buffer.back().cloned().unwrap_or(0.0);
        let eth_last = self.eth_ofi_buffer.back().cloned().unwrap_or(0.0);

        let btc_composite = btc_last * 0.60 + self.btc_ewma * 0.40;
        let eth_composite = eth_last * 0.60 + self.eth_ewma * 0.40;

        // Score de liderazgo ponderado (BTC 60%, ETH 40%)
        let leader_momentum = btc_composite * 0.60 + eth_composite * 0.40;

        // Si el líder tiene una ráfaga fuerte (|leader| > 0.50) pero el altcoin aún no ha reaccionado (|alt| < 0.25)
        let lead_lag_divergence = if leader_momentum.abs() > 0.50 && alt_ofi.abs() < 0.25 {
            leader_momentum - alt_ofi
        } else {
            0.0
        };

        (leader_momentum, lead_lag_divergence)
    }
}

impl Default for LeadLagAlphaEngine {
    fn default() -> Self {
        Self::new(50)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lead_lag_divergence_prediction() {
        let mut engine = LeadLagAlphaEngine::new(30);
        engine.update_leader(true, 0.80); // Fuerte impulso BTC
        engine.update_leader(false, 0.70); // Fuerte impulso ETH

        // Altcoin con OFI plano (0.05) -> debe detectar lead-lag divergence
        let (leader_mom, div) = engine.predict_altcoin_impulse(0.05);
        assert!(leader_mom > 0.70);
        assert!(div > 0.60);

        // Altcoin ya reaccionó (0.80) -> divergencia neutral
        let (_, div_reacted) = engine.predict_altcoin_impulse(0.80);
        assert_eq!(div_reacted, 0.0);
    }

    #[test]
    fn test_lead_lag_nan_and_empty_buffer_immunity() {
        let mut engine = LeadLagAlphaEngine::new(30);
        engine.update_leader(true, f64::NAN);
        engine.update_leader(false, f64::NAN);

        let (leader_mom, div) = engine.predict_altcoin_impulse(f64::NAN);
        assert_eq!(leader_mom, 0.0);
        assert_eq!(div, 0.0);
    }
}
