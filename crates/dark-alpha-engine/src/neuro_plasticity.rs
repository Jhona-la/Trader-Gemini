use serde::{Deserialize, Serialize};

/// Parámetros adaptativos del motor de neuro-plasticidad.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NeuroPlasticityParams {
    pub min_multiplier: f64,
    pub max_multiplier: f64,
    pub center_entropy: f64,
    pub steepness: f64,
    pub micro_boost_cap: f64,
    pub micro_capital_ref: f64,
}

impl Default for NeuroPlasticityParams {
    fn default() -> Self {
        Self {
            min_multiplier: 0.20,
            max_multiplier: 2.00,
            center_entropy: 0.50,
            steepness: 6.00,
            micro_boost_cap: 0.25,
            micro_capital_ref: 50.0,
        }
    }
}

impl NeuroPlasticityParams {
    #[inline(always)]
    pub fn validate(&self) -> bool {
        self.min_multiplier.is_finite()
            && self.max_multiplier.is_finite()
            && self.min_multiplier > 0.0
            && self.max_multiplier >= self.min_multiplier
            && self.center_entropy.is_finite()
            && (0.01..=0.99).contains(&self.center_entropy)
            && self.steepness.is_finite()
            && self.steepness > 0.0
            && self.micro_boost_cap.is_finite()
            && self.micro_boost_cap >= 0.0
            && self.micro_capital_ref.is_finite()
            && self.micro_capital_ref > 0.0
    }
}

/// ⚡ ALGORITMO #40: MOTOR DE NEURO-PLASTICIDAD (NEURO-PLASTICITY ENGINE)
/// Modula la plasticidad de la red con la entropía del libro
/// (`compute_plasticity_multiplier_with_params`) y aplica la regla de Oja.
/// RET-35 (censo C-F7b, F55): la reconexión sináptica ante drift se retiró
/// sin llamador.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct NeuroPlasticityEngine;

impl NeuroPlasticityEngine {
    /// 🛡️ FASE 7: META-EVOLUCIÓN CONTINUA C^1
    /// Conecta la entropía cuántica del Order Book para alterar cómo el bot aprende de forma continua.
    /// Si el mercado entra en alta entropía (pánico / turbulencia), la red incrementa suavemente
    /// su plasticidad hasta max_multiplier para adaptarse velozmente.
    /// Si el mercado es altamente predecible (baja entropía), reduce suavemente la tasa de mutación
    /// hacia min_multiplier para explotar el régimen estacionario sin sobreajuste.
    /// Modulación suave para micro-cuentas usando ponderación monótona del
    /// capital REAL declarado; capital ausente (NaN/≤0) abstiene la modulación.
    #[inline(always)]
    pub fn compute_plasticity_multiplier_with_params(
        params: &NeuroPlasticityParams,
        entropy: f64,
        capital: f64,
    ) -> f64 {
        let safe_params = if params.validate() {
            *params
        } else {
            NeuroPlasticityParams::default()
        };

        // Sanitización de finitud en entropía y capital para evitar modulación indeterminada
        let safe_entropy = if entropy.is_finite() {
            entropy.clamp(0.0, 1.0)
        } else {
            safe_params.center_entropy
        };
        // CL-15 (GLM 144): capital inválido ⇒ ABSTENCIÓN de la modulación por
        // capital (micro_factor neutro 1.0). Inventar 13.0 asumiría una
        // micro-cuenta inexistente y aplicaría el micro_boost caprichosamente.
        let micro_factor = if capital.is_finite() && capital > 0.0 {
            let micro_ratio = ((safe_params.micro_capital_ref - capital)
                / safe_params.micro_capital_ref)
                .clamp(0.0, 1.0);
            1.0 + micro_ratio * safe_params.micro_boost_cap
        } else {
            1.0
        };

        // Rampa continua C^1 vía sigmoide logística suave (sin escalones discretos heurísticos)
        let delta = safe_entropy - safe_params.center_entropy;
        let t = 1.0 / (1.0 + (-safe_params.steepness * delta).exp());
        let base_mult = safe_params.min_multiplier
            + (safe_params.max_multiplier - safe_params.min_multiplier) * t;

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
        // RET-35: la envoltura sin parámetros se retiró sin llamador; se prueba
        // la función viva con los parámetros por defecto, que es lo que delegaba.
        let p = NeuroPlasticityParams::default();
        let mult_high =
            NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(&p, 0.90, 13.0);
        assert!(
            mult_high > 1.5,
            "High entropy should boost plasticity multiplier"
        );

        let mult_low =
            NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(&p, 0.10, 13.0);
        assert!(
            mult_low < 1.0,
            "Low entropy should suppress plasticity multiplier"
        );

        let mult_nan = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(
            &p,
            f64::NAN,
            f64::NAN,
        );
        assert!(mult_nan.is_finite());
    }

    /// CL-15 (GLM 144): capital inválido ⇒ ABSTENCIÓN de la modulación por
    /// capital (idéntico a un capital grande); NO se asume una micro-cuenta
    /// de 13 USD inexistente.
    #[test]
    fn cl15_capital_invalido_abstiene_la_modulacion_por_micro_cuenta() {
        let p = NeuroPlasticityParams::default();
        let m_nan = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(
            &p, 0.9, f64::NAN,
        );
        let m_neg = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(
            &p, 0.9, -5.0,
        );
        let m_grande = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(
            &p,
            0.9,
            p.micro_capital_ref * 10.0,
        );
        assert!(
            (m_nan - m_grande).abs() < 1e-12,
            "capital NaN ⇒ modulación OFF (neutro), no micro-cuenta"
        );
        assert!(
            (m_neg - m_grande).abs() < 1e-12,
            "capital negativo ⇒ modulación OFF (neutro)"
        );
        let m_micro = NeuroPlasticityEngine::compute_plasticity_multiplier_with_params(
            &p, 0.9, 1.0,
        );
        assert!(
            m_micro > m_grande,
            "el micro_boost sólo con capital REAL pequeño declarado"
        );
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
