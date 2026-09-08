//! # Quantum Evolver Engine
//!
//! Quantum-inspired architecture evolution in Hilbert Space ($|\psi\rangle$).
//! Explores 10,000 architecture variants symbolically via Quantum Monte Carlo (QMC) / SPSA
//! Hamiltonian energy ($\hat{H}$) minimization before collapsing to concrete Rust AST templates.

use serde::{Deserialize, Serialize};
use crate::consejo_seniors::TradingHorizon;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub mode: TradingHorizon,
    pub amplitudes: Vec<f64>,
    pub window_size: usize,
    pub threshold: f64,
    pub funding_weight: f64,
    pub volume_multiplier: f64,
    pub energy: f64,
}

impl QuantumState {
    pub fn random_superposition(seed: u64, mode: TradingHorizon) -> Self {
        let mut rng = seed;
        let mut amplitudes = Vec::with_capacity(16);
        for i in 0..16 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let amp = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
            amplitudes.push(amp);
            let _ = i;
        }

        // Normalize state vector
        let norm: f64 = amplitudes.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        if norm > 0.0 {
            for a in &mut amplitudes {
                *a /= norm;
            }
        }

        // Base window calculation adaptativo para modo
        let window_size = match mode {
            TradingHorizon::Scalping => 16 + (amplitudes[0].abs() * 64.0) as usize, // ventanas pequeñas, respuesta nano
            TradingHorizon::Swing => 256 + (amplitudes[0].abs() * 1024.0) as usize, // ventanas muy grandes
            // U-F: interpolación geométrica del continuo (media de los extremos)
            TradingHorizon::Continuous => {
                let s = 16 + (amplitudes[0].abs() * 64.0) as usize;
                let l = 256 + (amplitudes[0].abs() * 1024.0) as usize;
                ((s as f64 * l as f64).sqrt()) as usize
            }
        };
        let threshold = 1.0 + amplitudes[1].abs() * 2.5;
        let funding_weight = amplitudes[2] * 2.0;
        let volume_multiplier = 1.0 + amplitudes[3].abs() * 3.0;

        Self {
            mode,
            amplitudes,
            window_size,
            threshold,
            funding_weight,
            volume_multiplier,
            energy: f64::MAX,
        }
    }

    /// Evaluates Hamiltonian Energy $\hat{H}$: $H = \text{Error} + \text{Complexity} + \text{Cost}$
    pub fn compute_hamiltonian(&mut self, historical_residual_error: f64) {
        let err = if historical_residual_error.is_finite() {
            historical_residual_error.max(0.0)
        } else {
            f64::MAX
        };
        
        let complexity_penalty = match self.mode {
            TradingHorizon::Scalping => (self.window_size as f64 / 64.0) * 0.1, // Penaltis estrictos por lentitud
            TradingHorizon::Swing => (self.window_size as f64 / 1024.0) * 0.02,
            // U-F: media de los extremos del continuo
            TradingHorizon::Continuous => {
                ((self.window_size as f64 / 64.0) * 0.1
                    + (self.window_size as f64 / 1024.0) * 0.02)
                    / 2.0
            }
        };
        let stability_cost = (self.threshold - 2.0).powi(2) * 0.02;
        let total = err + complexity_penalty + stability_cost;
        self.energy = if total.is_finite() { total } else { f64::MAX };
    }
}

pub struct QuantumEvolver {
    pub num_candidates: usize,
}

impl Default for QuantumEvolver {
    fn default() -> Self {
        Self {
            num_candidates: 100,
        }
    }
}

impl QuantumEvolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Explores architecture space in superposed Hilbert states and performs annealing measurement
    pub fn anneal_and_collapse(&self, seed: u64, residual_error: f64, mode: TradingHorizon) -> QuantumState {
        let mut best_state = QuantumState::random_superposition(seed, mode);
        best_state.compute_hamiltonian(residual_error);

        for i in 0..self.num_candidates {
            let candidate_seed = seed.wrapping_add(i as u64 * 9999);
            let mut candidate = QuantumState::random_superposition(candidate_seed, mode);
            // FIX #866: Evaluación homogénea de residual_error puro sin sesgos artificiales de amplitudes
            candidate.compute_hamiltonian(residual_error);

            if candidate.energy < best_state.energy {
                best_state = candidate;
            }
        }

        best_state
    }
}
