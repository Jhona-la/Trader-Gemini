// =====================================================================
// crates/risk-engine/src/hayden_preskill_scrambling.rs
//
// FASE 108 (§108): AdS/CFT Holographic Complexity, Ryu-Takayanagi Area
// & Hayden-Preskill Quantum Scrambling.
//
// Mathematical Framework:
// - Hayden-Preskill Information Recovery Protocol (2007):
//   Decodability of quantum information from black hole Hawking radiation
//   past the Page time, with reconstruction fidelity F_HP = 1 - 2^{-c_bits}.
// - Scrambling Time & Maldacena-Shenker-Stanford (MSS) Chaos Bound (2016):
//   Quantum Lyapunov exponent bounded by \lambda_L <= 2\pi T_eff.
// - Holographic Ryu-Takayanagi Entanglement Entropy in AdS_3:
//   S_A = (c/3) * ln((L / \pi \epsilon) * sin(\pi |A| / L)).
// - Tripartite Mutual Information:
//   I_3(A:B:C) = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC.
//   Fast scramblers exhibit negative I_3 <= 0.
// - Vetos:
//   1. HAYDEN_PRESKILL_FAST_SCRAMBLING_VETO: information decodability collapse.
//   2. MSS_CHAOS_BOUND_VIOLATION_VETO: violation of the quantum chaos bound.
//
// Invariants:
// - 100% Rust nativo, hot-path zero allocation (O(1) stack).
// - Zero panic/unwrap/expect (rb_p1 strict compliance).
// =====================================================================

use core::f64::consts::PI;

pub const HP_HISTORY_CAPACITY: usize = 32;

/// Configuration for the Hayden-Preskill Quantum Scrambling Engine.
#[derive(Debug, Clone, Copy)]
pub struct HaydenPreskillConfig {
    pub central_charge: f64,
    pub uv_cutoff_epsilon: f64,
    pub min_recovery_fidelity: f64,
    pub max_mss_tolerance: f64,
    pub scale_min: f64,
    pub scale_max: f64,
}

impl Default for HaydenPreskillConfig {
    fn default() -> Self {
        Self {
            central_charge: 1.5,
            uv_cutoff_epsilon: 0.01,
            min_recovery_fidelity: 0.65,
            max_mss_tolerance: 0.25,
            scale_min: 0.10,
            scale_max: 1.25,
        }
    }
}

/// Evaluation report from the Hayden-Preskill Quantum Scrambling Engine.
#[derive(Debug, Clone, Copy)]
pub struct HaydenPreskillReport {
    pub ryu_takayanagi_entropy: f64,
    pub tripartite_mutual_information: f64, // I_3 \le 0 for scramblers
    pub quantum_lyapunov_exponent: f64,     // \lambda_L
    pub mss_chaos_bound: f64,               // 2\pi T_eff
    pub recovery_fidelity: f64,             // \mathcal{F}_HP \in [0.0, 1.0]
    pub hayden_preskill_veto: bool,
    pub mss_chaos_veto: bool,
    pub slot_scale: f64,
}

impl Default for HaydenPreskillReport {
    fn default() -> Self {
        Self {
            ryu_takayanagi_entropy: 0.5,
            tripartite_mutual_information: -0.2,
            quantum_lyapunov_exponent: 0.1,
            mss_chaos_bound: 1.0,
            recovery_fidelity: 1.0,
            hayden_preskill_veto: false,
            mss_chaos_veto: false,
            slot_scale: 1.0,
        }
    }
}

/// Engine evaluating holographic entanglement, scrambling, and chaos bounds.
#[derive(Debug, Clone)]
pub struct HaydenPreskillEngine {
    config: HaydenPreskillConfig,
    otoc_buffer: [f64; HP_HISTORY_CAPACITY],
    head: usize,
    count: usize,
}

impl HaydenPreskillEngine {
    pub fn new(config: HaydenPreskillConfig) -> Self {
        Self {
            config,
            otoc_buffer: [0.0; HP_HISTORY_CAPACITY],
            head: 0,
            count: 0,
        }
    }

    pub fn with_default() -> Self {
        Self::new(HaydenPreskillConfig::default())
    }

    /// Reset internal history buffers.
    pub fn reset(&mut self) {
        self.otoc_buffer = [0.0; HP_HISTORY_CAPACITY];
        self.head = 0;
        self.count = 0;
    }

    /// Evaluate scrambling, Ryu-Takayanagi entropy and MSS chaos bound.
    pub fn update(
        &mut self,
        drawdown_pct: f64,
        vol_norm: f64,
        price_ret: f64,
        spread_bps: f64,
    ) -> HaydenPreskillReport {
        let dd = if drawdown_pct.is_finite() && drawdown_pct >= 0.0 {
            drawdown_pct.clamp(0.0, 0.99)
        } else {
            0.0
        };
        let vol = if vol_norm.is_finite() && vol_norm > 0.0 {
            vol_norm.clamp(0.01, 10.0)
        } else {
            1.0
        };
        let ret = if price_ret.is_finite() {
            price_ret
        } else {
            0.0
        };
        let spr = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.1, 50.0)
        } else {
            1.0
        };

        // Out-of-time-ordered correlator (OTOC) proxy from squared commutator [ret, vol]
        let otoc_val = (ret.abs() * 50.0 + vol * 0.20 + (spr / 10.0)).clamp(0.001, 100.0);
        self.otoc_buffer[self.head] = otoc_val;
        self.head = (self.head + 1) % HP_HISTORY_CAPACITY;
        if self.count < HP_HISTORY_CAPACITY {
            self.count += 1;
        }

        // Effective Hawking temperature T_eff of capital horizon
        // Proportional to volatility and inverted by capital cushion (1 - dd)
        let t_eff = ((vol * 0.15 + (spr / 20.0)) / (1.0 - dd + 1e-4)).clamp(0.05, 5.0);
        let mss_bound = 2.0 * PI * t_eff;

        // Estimate quantum Lyapunov exponent \lambda_L from growth of OTOC
        let lambda_l = self.estimate_lyapunov_exponent();

        // Holographic Ryu-Takayanagi Entanglement Entropy S_A
        // Subsystem size |A| modulated by exposure/drawdown
        let l_total = 1.0f64;
        let subregion_a = (0.20 + dd * 0.60).clamp(0.05, 0.95);
        let argument = (PI * subregion_a / l_total).sin();
        let rt_entropy = if argument > 1e-6 {
            let log_term = ((l_total / (PI * self.config.uv_cutoff_epsilon)) * argument)
                .max(1.0)
                .ln();
            (self.config.central_charge / 3.0) * log_term
        } else {
            0.1
        };

        // Tripartite Mutual Information I_3(A:B:C)
        // For fast scramblers, I_3 is negative. Drawdown and turbulence deepen negative scrambling,
        // but runaway decoherence drives it to collapse.
        let i3 = -0.50 * rt_entropy * (1.0 + (vol * 0.30));

        // Hayden-Preskill information recovery fidelity
        // F_HP = 1 - 2^{-c_bits}, where c_bits is the effective radiation margin
        let radiation_margin = (1.0 - dd) * (mss_bound / (lambda_l + 1e-4));
        let recovery_fidelity = (1.0 - (-radiation_margin).exp()).clamp(0.0, 1.0);

        // Veto conditions:
        // 1. HAYDEN_PRESKILL_FAST_SCRAMBLING_VETO: recovery fidelity collapses
        let hayden_preskill_veto =
            recovery_fidelity < self.config.min_recovery_fidelity || dd > 0.85;

        // 2. MSS_CHAOS_BOUND_VIOLATION_VETO: Lyapunov exponent violates quantum chaos bound
        let mss_violation = lambda_l - mss_bound;
        let mss_chaos_veto = mss_violation > self.config.max_mss_tolerance;

        // Slot scale modulation
        let raw_scale = if hayden_preskill_veto || mss_chaos_veto {
            self.config.scale_min
        } else {
            let fidelity_boost = (recovery_fidelity - self.config.min_recovery_fidelity)
                / (1.0 - self.config.min_recovery_fidelity + 1e-6);
            let chaos_margin = (mss_bound - lambda_l).max(0.0) / (mss_bound + 1.0);
            1.0 + (fidelity_boost * 0.10) + (chaos_margin * 0.10)
        };

        let slot_scale = raw_scale.clamp(self.config.scale_min, self.config.scale_max);

        HaydenPreskillReport {
            ryu_takayanagi_entropy: rt_entropy,
            tripartite_mutual_information: i3,
            quantum_lyapunov_exponent: lambda_l,
            mss_chaos_bound: mss_bound,
            recovery_fidelity,
            hayden_preskill_veto,
            mss_chaos_veto,
            slot_scale,
        }
    }

    /// Estimate Lyapunov exponent from exponential growth rate of OTOC buffer
    fn estimate_lyapunov_exponent(&self) -> f64 {
        if self.count < 4 {
            return 0.1;
        }

        let n = self.count.min(HP_HISTORY_CAPACITY);
        let idx_now = (self.head + HP_HISTORY_CAPACITY - 1) % HP_HISTORY_CAPACITY;
        let idx_prev = (self.head + HP_HISTORY_CAPACITY - 1 - (n / 2)) % HP_HISTORY_CAPACITY;

        let val_now = self.otoc_buffer[idx_now];
        let val_prev = self.otoc_buffer[idx_prev];

        if val_prev > 1e-6 && val_now > val_prev {
            let delta_t = (n / 2) as f64;
            ((val_now / val_prev).ln() / delta_t).clamp(0.01, 15.0)
        } else {
            0.10
        }
    }
}
