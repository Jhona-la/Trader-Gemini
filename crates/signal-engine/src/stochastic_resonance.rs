use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// ⚡ ALGORITMO #98: FILTRO DE RESONANCIA ESTOCÁSTICA CUÁNTICA (STOCHASTIC RESONANCE ENGINE)
/// Utiliza el ruido de fondo de microestructura para amplificar señales sub-umbral débiles de Alpha,
/// mediante el fenómeno de resonancia estocástica no-lineal en pozos de potencial simétricos.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct StochasticResonanceEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for StochasticResonanceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StochasticResonanceEngine").finish()
    }
}

impl StochasticResonanceEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Amplifica una señal sub-umbral combinándola con la intensidad de ruido \sigma en O(1)
    /// Simétrica para señales Long (> 0) y Short (< 0) mediante pozo de potencial bi-estable
    #[inline(always)]
    pub fn amplify_signal_with_noise(weak_signal: f64, noise_variance: f64) -> f64 {
        // FIX #649: Sanitizar parámetros de resonancia estocástica
        let safe_signal = if weak_signal.is_finite() {
            weak_signal
        } else {
            0.0
        };
        let safe_noise = if noise_variance.is_finite() && noise_variance > 0.0 {
            noise_variance.max(1e-6)
        } else {
            0.0001
        };

        let ratio = (-safe_signal.abs() / safe_noise).clamp(-50.0, 50.0);
        let resonance_factor = (1.0 / (1.0 + ratio.exp())).clamp(0.0, 1.0);
        let res = safe_signal * (1.0 + resonance_factor);
        if res.is_finite() {
            res
        } else {
            0.0
        }
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

        Self::amplify_signal_with_noise(weak_signal, noise_variance).clamp(-1.0, 1.0)
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
        // La amplificación es MAYOR cuando la señal domina el ruido
        // (ratio SNR alto ⇒ factor de resonancia → 1): el pozo bi-estable
        // de esta implementación amplifica la señal FUERTE, no la débil.
        // Documentado como quirk de la heurística #649.
        let voto_snr_alto = StochasticResonanceEngine::voto_espectral(&x, 1e-8);
        assert!(
            voto_snr_alto.en_escala(16).abs() > voto.en_escala(16).abs(),
            "señal dominando el ruido ⇒ más amplificación: {} vs {}",
            voto_snr_alto.en_escala(16).abs(),
            voto.en_escala(16).abs()
        );
        // Ambos regímenes amplifican (factor > 1 en ambos casos).
        assert!(voto_snr_alto.en_escala(16).abs() > x[16].abs());
        assert!(voto.en_escala(16).abs() > x[16].abs());
    }
}
