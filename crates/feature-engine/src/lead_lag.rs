use std::f64;
use std::collections::VecDeque;

/// ⚡ MOTOR DE LIDERAZGO MACRO MICROESTRUCTURAL LEAD-LAG (CROSS-ASSET ALPHA MATRIX)
/// Rastra la velocidad de propagación de ráfagas institucionales desde BTC/ETH hacia altcoins.
/// Permite entrar en altcoins en nanosegundos antes de que el libro reaccione al impulso macro.
#[derive(Debug, Clone)]
pub struct LeadLagAlphaEngine {
    pub btc_ofi_buffer: VecDeque<f64>,
    pub eth_ofi_buffer: VecDeque<f64>,
    pub max_window: usize,
}

impl LeadLagAlphaEngine {
    pub fn new(max_window: usize) -> Self {
        Self {
            btc_ofi_buffer: VecDeque::with_capacity(max_window),
            eth_ofi_buffer: VecDeque::with_capacity(max_window),
            max_window: max_window.max(20),
        }
    }

    #[inline(always)]
    pub fn update_leader(&mut self, is_btc: bool, ofi: f64) {
        let buffer = if is_btc { &mut self.btc_ofi_buffer } else { &mut self.eth_ofi_buffer };
        if buffer.len() >= self.max_window {
            buffer.pop_front();
        }
        buffer.push_back(ofi);
    }

    /// Calcula la señal de propagación Cross-Asset para un altcoin en nanosegundos
    #[inline(always)]
    pub fn predict_altcoin_impulse(&self, alt_ofi: f64) -> (f64, f64) {
        let btc_last = self.btc_ofi_buffer.back().cloned().unwrap_or(0.0);
        let eth_last = self.eth_ofi_buffer.back().cloned().unwrap_or(0.0);

        // Score de liderazgo ponderado (BTC 60%, ETH 40%)
        let leader_momentum = btc_last * 0.60 + eth_last * 0.40;

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
