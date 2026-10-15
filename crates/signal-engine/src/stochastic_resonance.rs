use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// ⚡ OLA Ω129: Parámetros desacoplados y configurables para el motor de resonancia estocástica
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticResonanceParams {
    /// Ganancia de base cuando la señal está lejos de la resonancia (default: 1.0)
    pub base_gain: f64,
    /// Ganancia pico máxima en resonancia |s| = σ (default: 2.0)
    pub peak_gain: f64,
    /// Factor de nitidez/calidad Q del pico de resonancia (default: 1.0)
    pub q_sharpness: f64,
    /// Suelo mínimo de varianza de ruido para estabilidad numérica (default: 1e-6)
    pub noise_floor: f64,
}

impl Default for StochasticResonanceParams {
    fn default() -> Self {
        Self {
            base_gain: 1.0,
            peak_gain: 2.0,
            q_sharpness: 1.0,
            noise_floor: 1e-6,
        }
    }
}

impl StochasticResonanceParams {
    pub fn validate(&self) -> Result<(), String> {
        if !self.base_gain.is_finite() || !(0.1..=5.0).contains(&self.base_gain) {
            return Err(format!("Invalid base_gain: {}", self.base_gain));
        }
        if !self.peak_gain.is_finite()
            || !(1.0..=10.0).contains(&self.peak_gain)
            || self.peak_gain < self.base_gain
        {
            return Err(format!("Invalid peak_gain: {}", self.peak_gain));
        }
        if !self.q_sharpness.is_finite() || !(0.1..=10.0).contains(&self.q_sharpness) {
            return Err(format!("Invalid q_sharpness: {}", self.q_sharpness));
        }
        if !self.noise_floor.is_finite() || !(1e-12..=1e-2).contains(&self.noise_floor) {
            return Err(format!("Invalid noise_floor: {}", self.noise_floor));
        }
        Ok(())
    }
}

/// ⚡ ALGORITMO #98: FILTRO DE RESONANCIA ESTOCÁSTICA CUÁNTICA (STOCHASTIC RESONANCE ENGINE)
/// Utiliza el ruido de fondo de microestructura para amplificar señales sub-umbral débiles de Alpha,
/// mediante el fenómeno de resonancia estocástica no-lineal en pozos de potencial simétricos.
#[derive(Clone)]
#[repr(C, align(64))]
pub struct StochasticResonanceEngine {
    pub params: StochasticResonanceParams,
    registry: Option<Arc<OmniscientRegistry>>,
}

impl Default for StochasticResonanceEngine {
    fn default() -> Self {
        Self {
            params: StochasticResonanceParams::default(),
            registry: None,
        }
    }
}

impl std::fmt::Debug for StochasticResonanceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StochasticResonanceEngine")
            .field("params", &self.params)
            .finish()
    }
}

impl StochasticResonanceEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_params(params: StochasticResonanceParams) -> Self {
        Self {
            params,
            registry: None,
        }
    }

    pub fn with_params(mut self, params: StochasticResonanceParams) -> Self {
        if params.validate().is_ok() {
            self.params = params;
        }
        self
    }

    pub fn update_params(&mut self, params: StochasticResonanceParams) {
        if params.validate().is_ok() {
            self.params = params;
        }
    }

    pub fn params(&self) -> &StochasticResonanceParams {
        &self.params
    }

    /// Amplifica una señal combinándola con la intensidad de ruido σ con parámetros desacoplados
    #[inline(always)]
    pub fn amplify_signal_with_params(
        weak_signal: f64,
        noise_variance: f64,
        params: &StochasticResonanceParams,
    ) -> f64 {
        let safe_signal = if weak_signal.is_finite() {
            weak_signal
        } else {
            0.0
        };
        let safe_noise = if noise_variance.is_finite() && noise_variance > 0.0 {
            noise_variance.max(params.noise_floor)
        } else {
            params.noise_floor
        };
        // σ = raíz de la varianza del ruido.
        let sigma = safe_noise.sqrt();
        // U-invertida en escala logarítmica con factor Q: pico exacto en |s| = σ.
        let raw_ratio = (safe_signal.abs().max(1e-12) / sigma).ln() * params.q_sharpness;
        let ratio = if raw_ratio.is_finite() {
            raw_ratio.clamp(-50.0, 50.0)
        } else {
            0.0
        };
        let cosh_val = (ratio.exp() + (-ratio).exp()) * 0.5;
        let delta_gain = (params.peak_gain - params.base_gain).max(0.0);
        let resonance_gain = params.base_gain + delta_gain / cosh_val;
        let res = safe_signal * resonance_gain;
        if res.is_finite() {
            res
        } else {
            0.0
        }
    }

    /// Amplifica una señal combinándola con la intensidad de ruido σ en O(1)
    /// Simétrica para señales Long (> 0) y Short (< 0) mediante pozo de potencial bi-estable.
    /// #650 (Ola 50) — RESONANCIA CANÓNICA en U-invertida: la ganancia
    /// `1 + sech(ln(|s|/σ))` PICA en |s| = σ (el régimen de resonancia:
    /// señal al nivel del ruido, donde el ruido ayuda a cruzar el umbral),
    /// decae a ×1 cuando |s| ≫ σ (señal fuerte pasa SIN cambio) y ×1 cuando
    /// |s| ≪ σ (señal enterrada: no hay dirección que amplificar). C∞,
    /// acotada (ganancia ∈ [1, 2]), antisimétrica.
    #[inline(always)]
    pub fn amplify_signal_with_noise(weak_signal: f64, noise_variance: f64) -> f64 {
        Self::amplify_signal_with_params(
            weak_signal,
            noise_variance,
            &StochasticResonanceParams::default(),
        )
    }

    /// #614 (Ola 36) — VOTO ESPECTRAL de la resonancia estocástica: el
    /// pozo bi-estable amplifica la señal SUB-UMBRAL de cada escala —
    /// x(τ) débil relativo al ruido se POTENCIA (resonancia), x(τ) fuerte
    /// pasa sin cambio. La escala que opera en el régimen de resonancia
    /// (señal ≈ ruido) vota más fuerte que su amplitud cruda sugiere.
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
        varianza_ruido: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        crate::voto_espectral::VotoEspectral::desde_espectro(desplazamientos, |x| {
            Self::amplify_signal_with_noise(x, varianza_ruido)
        })
    }
}

impl QuantumStrategy for StochasticResonanceEngine {
    fn name(&self) -> &str {
        "StochasticResonanceEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() {
            None
        } else {
            Some(symbol)
        };
        let cid_opt = if symbol.is_empty() {
            None
        } else {
            Some(coin_id)
        };
        let registry = match self.registry.as_ref() {
            Some(r) => r,
            None => return 0.0,
        };
        let weak_signal = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "weak_alpha_signal",
                "StochasticResonanceEngine",
            )
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_imbalance",
                    "StochasticResonanceEngine",
                )
            })
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "alpha_signal",
                    "StochasticResonanceEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        let noise_variance = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "microstructure_noise_variance",
                "StochasticResonanceEngine",
            )
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "tick_variance",
                    "StochasticResonanceEngine",
                )
            })
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "atr_pct",
                    "StochasticResonanceEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0001);

        if !weak_signal.is_finite() || !noise_variance.is_finite() {
            return 0.0;
        }

        let peak_gain = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "stochastic_res_peak_gain",
                "StochasticResonanceEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(self.params.peak_gain);
        let q_sharpness = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "stochastic_res_q",
                "StochasticResonanceEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(self.params.q_sharpness);
        let mut active_params = self.params;
        if peak_gain.is_finite() && peak_gain >= active_params.base_gain {
            active_params.peak_gain = peak_gain;
        }
        if q_sharpness.is_finite() && q_sharpness > 0.05 {
            active_params.q_sharpness = q_sharpness;
        }

        Self::amplify_signal_with_params(weak_signal, noise_variance, &active_params).clamp(-1.0, 1.0)
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_resonance_amplification() {
        let weak = 0.1;
        let amp = StochasticResonanceEngine::amplify_signal_with_noise(weak, 0.05);
        assert!(amp > weak);
        assert!(!amp.is_nan() && !amp.is_infinite());
    }

    #[test]
    fn test_stochastic_resonance_nan_and_negative_symmetry() {
        let weak_short = -0.1;
        let amp_short = StochasticResonanceEngine::amplify_signal_with_noise(weak_short, 0.05);
        assert!(amp_short < weak_short);
        assert!(amp_short.is_finite());

        let nan_res = StochasticResonanceEngine::amplify_signal_with_noise(f64::NAN, 0.05);
        assert!(nan_res.is_finite() && nan_res == 0.0);
    }

    #[test]
    fn test_stochastic_resonance_engine_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("weak_alpha_signal", 0.2);
        registry.set("microstructure_noise_variance", 0.01);

        let mut engine = StochasticResonanceEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        assert!(
            eval > 0.2,
            "Resonancia estocástica debe amplificar la señal sub-umbral"
        );
        assert!(eval <= 1.0);
    }
}

#[cfg(test)]
mod qo_614_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;

    #[test]
    fn qo_614_resonancia_amplifica_lo_sub_umbral_por_escala() {
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = 0.01 * (k as f64 - 15.5); // señales débiles simétricas
        }
        let sigma2 = 0.05; // ruido DOMINA la señal (régimen de resonancia)
        let voto = StochasticResonanceEngine::voto_espectral(&x, sigma2);
        // Antisimetría: el pozo bi-estable es simétrico.
        for kk in 0..ESCALAS_VOTO {
            let idx = ESCALAS_VOTO - 1 - kk;
            if (x[kk] + x[idx]).abs() < 1e-12 && x[kk] != 0.0 {
                assert!(
                    (voto.en_escala(kk) + voto.en_escala(idx)).abs() < 1e-10,
                    "antisimetría en {}: {} vs {}",
                    kk,
                    voto.en_escala(kk),
                    voto.en_escala(idx)
                );
            }
        }
        // La amplificación es real: |voto| > |x| en el régimen sub-umbral.
        for kk in 0..ESCALAS_VOTO {
            if x[kk].abs() > 1e-6 {
                assert!(
                    voto.en_escala(kk).abs() > x[kk].abs(),
                    "la resonancia amplifica la señal débil en {}: {} vs {}",
                    kk,
                    voto.en_escala(kk).abs(),
                    x[kk].abs()
                );
            }
        }
        // #650 — RESONANCIA CANÓNICA: la amplificación es MÁXIMA en el
        // régimen de resonancia (|s|≈σ) y decae a ×1 cuando la señal
        // domina (pasa casi sin cambio). Nota: el piso de varianza 1e-6
        // (σ ≥ 1e-3) acota cuán "dominante" puede ser la señal; usamos la
        // escala de mayor |x| (k=31: 0.155, ratio 155σ) para verla pasar.
        let voto_snr_alto = StochasticResonanceEngine::voto_espectral(&x, 1e-6);
        let amplificado = voto.en_escala(31).abs() / x[31].abs();
        let dominante = voto_snr_alto.en_escala(31).abs() / x[31].abs();
        assert!(
            amplificado > dominante,
            "régimen de resonancia amplifica más que señal dominante: {amplificado} vs {dominante}"
        );
        // Señal dominante ⇒ ganancia ≈ 1 (pasa sin cambio, <1.5%).
        assert!(
            (dominante - 1.0).abs() < 0.015,
            "señal dominante pasa sin cambio: ganancia {dominante}"
        );
        // Ganancia pico EXACTO en |s| = σ: ×2.
        let sigma = 0.05f64.sqrt();
        let pico = StochasticResonanceEngine::amplify_signal_with_noise(sigma, 0.05);
        assert!((pico - 2.0 * sigma).abs() < 1e-9, "pico {}", pico);
        // Enterrada (|s|≪σ): pasa casi sin cambio.
        let enterrada = StochasticResonanceEngine::amplify_signal_with_noise(1e-9, 0.05);
        assert!(enterrada.abs() < 1e-8);
    }

}
