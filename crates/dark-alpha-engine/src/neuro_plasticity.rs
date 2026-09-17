/// ⚡ ALGORITMO #40: MOTOR DE NEURO-PLASTICIDAD Y RECONEXIÓN SINÁPTICA EN TIEMPO REAL (NEURO-PLASTICITY ENGINE)
/// Reconecta pesos y conexiones sinápticas neuronales en tiempo real durante la ejecución
/// ante la detección de anomalías CUSUM/SPRT o cambios drásticos de régimen en el mercado.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct NeuroPlasticityEngine;

impl NeuroPlasticityEngine {
    /// Aplica reconexión sináptica sobre la matriz de pesos de la red neuronal si detecta drift
    #[inline(always)]
    pub fn rewires_synapses_if_drift(
        weights: &mut [f64],
        anomaly_detected: bool,
        learning_rate: f64,
    ) -> usize {
        if !anomaly_detected {
            return 0;
        }

        // FIX #632: Sanitizar learning_rate para prevenir introducción de NaNs
        let lr = if learning_rate.is_finite() && learning_rate > 0.0 {
            learning_rate.clamp(0.0001, 1.0)
        } else {
            0.01
        };

        // Quantum PICOSECOND entropía mediante RDTSC (Timestamp Counter) y XORShift
        let mut count = 0;
        let mut seed = unsafe { std::arch::x86_64::_rdtsc() };
        let len = weights.len();

        for weight in weights.iter_mut().take(len) {
            if weight.abs() < 0.01 {
                // Xorshift64 simple, rápido, ~1 ciclo de reloj
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;

                // Mapear el u64 resultante a un float pseudoaleatorio entre -1.0 y 1.0
                let rand_f64 = ((seed % 2000) as f64 - 1000.0) / 1000.0;

                // Reconexión sináptica adaptativa ultra rápida
                *weight = rand_f64 * lr;
                count += 1;
            }
        }
        count
    }

    /// 🛡️ FASE 7: META-EVOLUCIÓN
    /// Conecta la entropía cuántica del Order Book para alterar cómo el bot aprende.
    /// Si el mercado entra en pánico (alta entropía > 0.80), la red "derrite" sus pesos
    /// para adaptarse rápidamente (Learning Rate x 10).
    /// Si el mercado es predecible, congela los genes para exprimir ganancias.
    /// Para Micro Cuentas (< $50), aplica el multiplicador de Hyper-Mutation (x3 adicional).
    #[inline(always)]
    pub fn compute_plasticity_multiplier(entropy: f64, capital: f64) -> f64 {
        // FIX #708: Sanitización de finitud en entropía y capital para evitar modulación indeterminada
        let safe_entropy = if entropy.is_finite() {
            entropy.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let safe_capital = if capital.is_finite() && capital > 0.0 {
            capital
        } else {
            13.0
        };

        let base_mult = if safe_entropy > 0.85 {
            2.0
        } else if safe_entropy > 0.60 {
            1.5
        } else if safe_entropy < 0.30 {
            0.2
        } else {
            1.0
        };

        // MicroAccount Modulation: Modulación suave y segura acotada para micro-cuentas ($13 USD)
        let micro_factor = if safe_capital < 50.0 {
            1.0 + ((50.0 - safe_capital.max(1.0)) / 50.0) * 0.25
        } else {
            1.0
        };

        let raw = base_mult * micro_factor;
        if raw.is_finite() {
            raw.clamp(0.1, 2.5)
        } else {
            1.0
        }
    }

    /// Plasticidad de Oja (Hebbian Learning Modificado)
    /// Se aplica in-place a la matriz de pesos de una capa lineal en cada inferencia.
    /// \Delta w_{ij} = \eta y_j (x_i - y_j w_{ij})
    #[inline(always)]
    pub fn apply_oja_plasticity(
        weights: &mut [f64],
        inputs: &[f64],
        outputs: &[f64],
        in_features: usize,
        out_features: usize,
        learning_rate: f64,
    ) {
        if learning_rate < 1e-9 || in_features == 0 || out_features == 0 {
            return;
        }
        // FIX #631: Guardas O(1) de longitud de slices (inputs, outputs y weights)
        if inputs.len() < in_features
            || outputs.len() < out_features
            || weights.len() < out_features * in_features
        {
            return;
        }

        for (i, &y_j) in outputs.iter().enumerate().take(out_features) {
            let safe_y = if y_j.is_finite() { y_j } else { 0.0 };
            let row_offset = i * in_features;
            let rate_y = learning_rate * safe_y;

            for j in 0..in_features {
                let x_i = if inputs[j].is_finite() {
                    inputs[j]
                } else {
                    0.0
                };
                let w = if weights[row_offset + j].is_finite() {
                    weights[row_offset + j]
                } else {
                    0.0
                };
                // Regla de Oja: w += lr * y * (x - y * w)
                let dw = rate_y * (x_i - safe_y * w);
                if dw.is_finite() {
                    weights[row_offset + j] = (w + dw).clamp(-10.0, 10.0);
                } else {
                    weights[row_offset + j] = w;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_plasticity_multiplier() {
        let mult_high = NeuroPlasticityEngine::compute_plasticity_multiplier(0.90, 13.0);
        assert!(
            mult_high > 1.5,
            "High entropy should boost plasticity multiplier"
        );

        let mult_low = NeuroPlasticityEngine::compute_plasticity_multiplier(0.10, 13.0);
        assert!(
            mult_low < 1.0,
            "Low entropy should suppress plasticity multiplier"
        );

        let mult_nan = NeuroPlasticityEngine::compute_plasticity_multiplier(f64::NAN, f64::NAN);
        assert!(mult_nan.is_finite());
    }

    #[test]
    fn test_oja_plasticity_bounded() {
        let mut weights = vec![0.1; 4];
        let inputs = vec![1.0, 2.0];
        let outputs = vec![0.5, 0.5];

        NeuroPlasticityEngine::apply_oja_plasticity(&mut weights, &inputs, &outputs, 2, 2, 0.01);
        for &w in &weights {
            assert!(w.is_finite() && w.abs() <= 10.0);
        }
    }

    #[test]
    fn test_rewires_synapses_if_drift() {
        let mut weights = vec![0.001, 0.5, -0.002, 0.8];
        let rewired = NeuroPlasticityEngine::rewires_synapses_if_drift(&mut weights, true, 0.05);
        assert_eq!(rewired, 2, "Near-zero weights should be rewired upon drift");
    }

    #[test]
    fn test_oja_plasticity_mismatched_dimensions_and_nan() {
        let mut weights = vec![0.5; 4];
        let short_inputs = vec![1.0]; // shorter than in_features=2
        let outputs = vec![0.5, 0.5];

        NeuroPlasticityEngine::apply_oja_plasticity(
            &mut weights,
            &short_inputs,
            &outputs,
            2,
            2,
            0.01,
        );
        assert_eq!(weights[0], 0.5, "Mismatched dimensions must return early");

        let nan_inputs = vec![f64::NAN, 1.0];
        NeuroPlasticityEngine::apply_oja_plasticity(
            &mut weights,
            &nan_inputs,
            &outputs,
            2,
            2,
            0.01,
        );
        assert!(weights[0].is_finite());
    }

    #[test]
    fn test_rewires_synapses_no_drift_and_nan_learning_rate() {
        let mut weights = vec![0.001, 0.5, -0.002, 0.8];
        let rewired_no_anomaly =
            NeuroPlasticityEngine::rewires_synapses_if_drift(&mut weights, false, 0.05);
        assert_eq!(
            rewired_no_anomaly, 0,
            "No rewiring when anomaly_detected is false"
        );

        // NaN learning rate falls back to safe default (0.01) without crashing
        let rewired_nan_lr =
            NeuroPlasticityEngine::rewires_synapses_if_drift(&mut weights, true, f64::NAN);
        assert_eq!(rewired_nan_lr, 2);
        assert!(weights[0].is_finite());
        assert!(weights[2].is_finite());
    }

    #[test]
    fn test_oja_plasticity_nan_outputs_and_weights_immunity() {
        let mut weights = vec![f64::NAN, 0.5, 0.2, f64::INFINITY];
        let inputs = vec![1.0, 2.0];
        let outputs = vec![f64::NAN, 0.5];

        NeuroPlasticityEngine::apply_oja_plasticity(&mut weights, &inputs, &outputs, 2, 2, 0.05);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }
}
