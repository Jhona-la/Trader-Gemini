// =====================================================================
// crates/feature-engine/src/geometric_langlands_hecke.rs
//
// FASE 106 (§106): Geometric Langlands Program & Automorphic Hecke
// Eigensheaves over Microstructure.
//
// Mathematical Framework:
// - Geometric Langlands Correspondence (Beilinson-Drinfeld, Gaitsgory, Frenkel):
//   Equivalence between D-modules on Bun_G(X) and coherent sheaves on Loc_G(X).
// - Automorphic Hecke Algebra: Action of local Hecke operators T_p at prime
//   lags p \in {2, 3, 5, 7, 11, 13, 17, 19} over the order flow discrete spectrum.
// - Satake Isomorphism & Ramanujan-Petersson Conjecture:
//   Temperedness condition |a_p| <= 2 * p^{(k-1)/2}. For normalized k=1, |a_p| <= 2.
// - Automorphic L-function at central critical point s = 1/2:
//   L(1/2, \pi) = \prod_p (1 - a_p p^{-1/2} + p^{-1})^{-1}.
// - Veto: LANGLANDS_AUTOMORPHIC_ANOMALY_VETO when order flow violates the
//   Ramanujan temperedness bound, indicating predatory non-unitary manipulation.
//
// Invariants:
// - 100% Rust nativo, hot-path zero allocation (O(1) stack).
// - Zero panic/unwrap/expect (rb_p1 strict compliance).
// =====================================================================

pub const NUM_PRIME_SCALES: usize = 8;
pub const DEFAULT_PRIME_SCALES: [u32; NUM_PRIME_SCALES] = [2, 3, 5, 7, 11, 13, 17, 19];
pub const HE_BUFFER_CAPACITY: usize = 32;

/// Configuration for the Geometric Langlands Hecke Eigensheaf Engine.
#[derive(Debug, Clone, Copy)]
pub struct GeometricLanglandsHeckeConfig {
    pub prime_scales: [u32; NUM_PRIME_SCALES],
    pub ramanujan_bound: f64,
    pub max_anomaly_tolerance: f64,
    pub scale_min: f64,
    pub scale_max: f64,
}

impl Default for GeometricLanglandsHeckeConfig {
    fn default() -> Self {
        Self {
            prime_scales: DEFAULT_PRIME_SCALES,
            ramanujan_bound: 2.0,
            max_anomaly_tolerance: 0.65,
            scale_min: 0.10,
            scale_max: 1.25,
        }
    }
}

/// Evaluation report from the Geometric Langlands Hecke Eigensheaf Engine.
#[derive(Debug, Clone, Copy)]
pub struct GeometricLanglandsHeckeReport {
    pub hecke_eigenvalues: [f64; NUM_PRIME_SCALES],
    pub satake_parameters: [f64; NUM_PRIME_SCALES],
    pub automorphic_l_function_central: f64,
    pub ramanujan_violation_count: u32,
    pub max_hecke_deviation: f64,
    pub langlands_anomaly_veto: bool,
    pub slot_scale: f64,
}

impl Default for GeometricLanglandsHeckeReport {
    fn default() -> Self {
        Self {
            hecke_eigenvalues: [0.0; NUM_PRIME_SCALES],
            satake_parameters: [0.0; NUM_PRIME_SCALES],
            automorphic_l_function_central: 1.0,
            ramanujan_violation_count: 0,
            max_hecke_deviation: 0.0,
            langlands_anomaly_veto: false,
            slot_scale: 1.0,
        }
    }
}

/// Engine evaluating automorphic Hecke eigensheaves across prime microstructure lags.
#[derive(Debug, Clone)]
pub struct GeometricLanglandsHeckeEngine {
    config: GeometricLanglandsHeckeConfig,
    return_buffer: [f64; HE_BUFFER_CAPACITY],
    ofi_buffer: [f64; HE_BUFFER_CAPACITY],
    head: usize,
    count: usize,
}

impl GeometricLanglandsHeckeEngine {
    pub fn new(config: GeometricLanglandsHeckeConfig) -> Self {
        Self {
            config,
            return_buffer: [0.0; HE_BUFFER_CAPACITY],
            ofi_buffer: [0.0; HE_BUFFER_CAPACITY],
            head: 0,
            count: 0,
        }
    }

    pub fn with_default() -> Self {
        Self::new(GeometricLanglandsHeckeConfig::default())
    }

    /// Reset internal history buffers.
    pub fn reset(&mut self) {
        self.return_buffer = [0.0; HE_BUFFER_CAPACITY];
        self.ofi_buffer = [0.0; HE_BUFFER_CAPACITY];
        self.head = 0;
        self.count = 0;
    }

    /// Update engine state with latest market microstructure inputs.
    pub fn update(
        &mut self,
        price_ret: f64,
        ofi: f64,
        volume_norm: f64,
        spread_bps: f64,
    ) -> GeometricLanglandsHeckeReport {
        // Sanitize inputs
        let ret = if price_ret.is_finite() {
            price_ret
        } else {
            0.0
        };
        let ofi_clean = if ofi.is_finite() {
            ofi.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let vol = if volume_norm.is_finite() && volume_norm > 0.0 {
            volume_norm
        } else {
            1.0
        };
        let spr = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps
        } else {
            1.0
        };

        // Store raw inputs into circular buffers
        self.return_buffer[self.head] = ret;
        self.ofi_buffer[self.head] = ofi_clean;
        self.head = (self.head + 1) % HE_BUFFER_CAPACITY;
        if self.count < HE_BUFFER_CAPACITY {
            self.count += 1;
        }

        if self.count < 6 {
            return GeometricLanglandsHeckeReport::default();
        }

        // Predatory coupling multiplier from volume-to-spread concentration
        let vol_spr_ratio = (vol / (spr + 0.1)).clamp(0.1, 5.0);
        let coupling = if vol_spr_ratio > 1.5 {
            1.0 + (vol_spr_ratio - 1.5) * 0.75
        } else {
            1.0
        };

        let mut hecke_eigenvalues = [0.0; NUM_PRIME_SCALES];
        let mut satake_parameters = [0.0; NUM_PRIME_SCALES];
        let mut violations = 0u32;
        let mut max_dev = 0.0f64;
        let mut l_central = 1.0f64;

        // Compute local Hecke eigenvalues T_p across prime lags
        for (i, &p) in self.config.prime_scales.iter().enumerate() {
            let lag = p as usize;
            let (eigenvalue, satake) = self.compute_hecke_eigenvalue(lag, coupling);
            hecke_eigenvalues[i] = eigenvalue;
            satake_parameters[i] = satake;

            let dev = (eigenvalue.abs() - self.config.ramanujan_bound).max(0.0);
            if dev > max_dev {
                max_dev = dev;
            }

            if dev > self.config.max_anomaly_tolerance {
                violations += 1;
            }

            // Local Euler factor: (1 - a_p p^{-1/2} + p^{-1})^{-1}
            let p_f = p as f64;
            let p_half = p_f.sqrt();
            let denom = 1.0 - (eigenvalue / p_half) + (1.0 / p_f);
            if denom.abs() > 1e-6 {
                l_central *= 1.0 / denom;
            }
        }

        // Bound L-function to finite positive range
        let l_central_clean = if l_central.is_finite() && l_central > 0.0 {
            l_central.clamp(0.01, 100.0)
        } else {
            1.0
        };

        // Veto condition: violation of Ramanujan-Petersson temperedness
        let langlands_anomaly_veto = violations > 0 || max_dev > self.config.max_anomaly_tolerance;

        // Slot scale modulation
        let raw_scale = if langlands_anomaly_veto {
            self.config.scale_min
        } else {
            let stability_bonus =
                (1.0 - (max_dev / self.config.max_anomaly_tolerance)).clamp(0.0, 1.0);
            let l_mod = (l_central_clean.ln() * 0.05).clamp(-0.15, 0.15);
            1.0 + (stability_bonus * 0.10) + l_mod
        };

        let slot_scale = raw_scale.clamp(self.config.scale_min, self.config.scale_max);

        GeometricLanglandsHeckeReport {
            hecke_eigenvalues,
            satake_parameters,
            automorphic_l_function_central: l_central_clean,
            ramanujan_violation_count: violations,
            max_hecke_deviation: max_dev,
            langlands_anomaly_veto,
            slot_scale,
        }
    }

    /// Compute Hecke eigenvalue T_p for a given prime lag
    fn compute_hecke_eigenvalue(&self, lag: usize, coupling: f64) -> (f64, f64) {
        if self.count <= lag + 2 {
            return (0.0, 0.0);
        }

        let n = self.count.min(HE_BUFFER_CAPACITY);
        let mut sum_r = 0.0;
        let mut sum_o = 0.0;
        for i in 0..n {
            let idx = (self.head + HE_BUFFER_CAPACITY - 1 - i) % HE_BUFFER_CAPACITY;
            sum_r += self.return_buffer[idx];
            sum_o += self.ofi_buffer[idx];
        }
        let mean_r = sum_r / (n as f64);
        let mean_o = sum_o / (n as f64);

        let mut cov_ret = 0.0;
        let mut cov_ofi = 0.0;
        let mut var_ret = 0.0;
        let mut var_ofi = 0.0;
        let mut pairs = 0usize;

        for k in 0..(n - lag) {
            let idx_now = (self.head + HE_BUFFER_CAPACITY - 1 - k) % HE_BUFFER_CAPACITY;
            let idx_lag = (self.head + HE_BUFFER_CAPACITY - 1 - k - lag) % HE_BUFFER_CAPACITY;

            let r_now = self.return_buffer[idx_now] - mean_r;
            let r_lag = self.return_buffer[idx_lag] - mean_r;
            let o_now = self.ofi_buffer[idx_now] - mean_o;
            let o_lag = self.ofi_buffer[idx_lag] - mean_o;

            cov_ret += r_now * r_lag;
            cov_ofi += o_now * o_lag;
            var_ret += r_now * r_now;
            var_ofi += o_now * o_now;
            pairs += 1;
        }

        if pairs == 0 || (var_ret <= 1e-12 && var_ofi <= 1e-12) {
            return (0.0, 0.0);
        }

        let corr_ret = if var_ret > 1e-12 {
            (cov_ret / var_ret).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let corr_ofi = if var_ofi > 1e-12 {
            (cov_ofi / var_ofi).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Base Hecke eigenvalue: a_p = corr_ret + corr_ofi in [-2.0, 2.0]
        let base_eigenvalue = corr_ret + corr_ofi;
        let eigenvalue = base_eigenvalue * coupling;

        // Satake parameter \alpha_p: normalized to [-1.0, 1.0] under temperedness
        let satake = eigenvalue * 0.50;

        (eigenvalue, satake)
    }
}
