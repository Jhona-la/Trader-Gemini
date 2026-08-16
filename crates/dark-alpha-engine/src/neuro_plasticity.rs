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
                *weight = rand_f64 * learning_rate;
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
        let mut multiplier = if entropy > 0.85 {
            // Caos absoluto: Neuro-plasticidad extrema
            10.0
        } else if entropy > 0.60 {
            // Transición: Plasticidad elevada
            2.5
        } else if entropy < 0.30 {
            // Estructura extrema: Congelamiento epigenético (Histone Silencing)
            0.1
        } else {
            // Mercado normal
            1.0
        };

        // MicroAccountHyperMutation: Si la cuenta es micro (< $50), necesitamos reaccionar
        // 3 veces más rápido para evitar quiebra.
        if capital < 50.0 {
            multiplier *= 3.0;
        }

        multiplier
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
        if learning_rate < 1e-9 {
            return;
        }
        for (i, &y_j) in outputs.iter().enumerate().take(out_features) {
            let row_offset = i * in_features;
            let rate_y = learning_rate * y_j;

            for j in 0..in_features {
                let x_i = inputs[j];
                let w = weights[row_offset + j];
                // Regla de Oja: w += lr * y * (x - y * w)
                let dw = rate_y * (x_i - y_j * w);
                weights[row_offset + j] += dw;
            }
        }
    }
}
