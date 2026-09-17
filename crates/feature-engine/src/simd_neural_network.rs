/// 🛡️ ALGORITMO #154: MOTOR NEURONAL ULTRALIGERO (SIMD CPU MLP)
/// Deep Learning sin GPU, ejecutado en cachés L1/L2 mediante auto-vectorización del compilador (SIMD).
/// Diseñado para 34 entradas, 16 neuronas ocultas y 2 salidas.

#[derive(Clone, Debug)]
#[repr(C, align(64))]
pub struct SimdNeuralNet {
    // Pesos Capa 1: [Entradas x Ocultas] -> [34 x 16] = 544
    pub w1: [[f64; 16]; 34],
    // Sesgo Capa 1: 16
    pub b1: [f64; 16],

    // Pesos Capa 2: [Ocultas x Salidas] -> [16 x 2] = 32
    pub w2: [[f64; 2]; 16],
    // Sesgo Capa 2: 2
    pub b2: [f64; 2],
}

impl Default for SimdNeuralNet {
    fn default() -> Self {
        let mut w1 = [[0.0; 16]; 34];
        let mut w2 = [[0.0; 2]; 16];

        // Inicialización He/Xavier ortogonal determinista para romper la simetría neuronal (BUG-680)
        for (i, row) in w1.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                *w = ((i * 17 + j * 31 + 7) as f64).sin() * 0.2425;
            }
        }
        for (j, row) in w2.iter_mut().enumerate() {
            for (k, w) in row.iter_mut().enumerate() {
                *w = ((j * 13 + k * 29 + 11) as f64).sin() * 0.3333;
            }
        }

        Self {
            w1,
            b1: [0.0; 16],
            w2,
            b2: [0.0; 2],
        }
    }
}

impl SimdNeuralNet {
    /// Inferencia Forward Pass (O(1) memoria, microsegundos)
    /// Devuelve [Probabilidad Long, Probabilidad Short]
    #[inline(always)]
    pub fn infer(&self, inputs: &[f64; 34]) -> [f64; 2] {
        let mut clean_inputs = [0.0; 34];
        for (clean, &raw) in clean_inputs.iter_mut().zip(inputs.iter()) {
            *clean = if raw.is_finite() {
                raw.clamp(-100.0, 100.0)
            } else {
                0.0
            };
        }

        let mut hidden = [0.0; 16];

        // Multiplicación de Matrices Capa 1 (Vectorización SIMD 256-bit Pura vía Iteradores)
        for (&in_val, w1_row) in clean_inputs.iter().zip(self.w1.iter()) {
            for (h, &w) in hidden.iter_mut().zip(w1_row.iter()) {
                *h += in_val * w;
            }
        }

        // Aplicar Sesgo y Activación Leaky ReLU (Capa Oculta) para evitar Fugas de Gradiente (Dying ReLU)
        for (h, &b) in hidden.iter_mut().zip(self.b1.iter()) {
            let raw_h = *h + b;
            *h = if raw_h > 0.0 { raw_h } else { raw_h * 0.01 };
        }

        let mut outputs = [0.0; 2];

        // Multiplicación Capa 2
        for (&h_val, w2_row) in hidden.iter().zip(self.w2.iter()) {
            for (out, &w) in outputs.iter_mut().zip(w2_row.iter()) {
                *out += h_val * w;
            }
        }

        // Activación de Salida (Softmax Numéricamente Estable para evitar overflow a Inf/NaN)
        outputs[0] += self.b2[0];
        outputs[1] += self.b2[1];

        let max_val = outputs[0].max(outputs[1]);
        let exp_0 = (outputs[0] - max_val).exp();
        let exp_1 = (outputs[1] - max_val).exp();
        let sum_exp = exp_0 + exp_1;

        if sum_exp > 0.0 && sum_exp.is_finite() {
            [
                (exp_0 / sum_exp).clamp(0.0001, 0.9999),
                (exp_1 / sum_exp).clamp(0.0001, 0.9999),
            ]
        } else {
            [0.5, 0.5] // Neutral fallback
        }
    }

    /// Actualización online con clipping de gradientes y decaimiento L2 (previene explosión de pesos)
    #[inline(always)]
    pub fn train_step(&mut self, inputs: &[f64; 34], target_idx: usize, learning_rate: f64) {
        let mut clean_inputs = [0.0; 34];
        for (clean, &raw) in clean_inputs.iter_mut().zip(inputs.iter()) {
            *clean = if raw.is_finite() {
                raw.clamp(-100.0, 100.0)
            } else {
                0.0
            };
        }
        let lr = if learning_rate.is_finite() {
            learning_rate.clamp(1e-5, 0.1)
        } else {
            0.001
        };
        let probs = self.infer(&clean_inputs);

        // Error de salida con Cross-Entropy
        let target = if target_idx == 0 {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        };
        let mut d_out = [0.0; 2];
        for (d, (&p, &t)) in d_out.iter_mut().zip(probs.iter().zip(target.iter())) {
            *d = (p - t).clamp(-2.0, 2.0);
        }

        // Forward activations para backprop
        let mut hidden = [0.0; 16];
        for (&in_val, w1_row) in clean_inputs.iter().zip(self.w1.iter()) {
            for (h, &w) in hidden.iter_mut().zip(w1_row.iter()) {
                *h += in_val * w;
            }
        }

        // FIX #391: Aislando la derivada transitoria antes de mutar la capa profunda (Fuga de Gradientes)
        let mut d_hidden = [0.0; 16];

        // Fase 1: Computar el error que fluye hacia la capa oculta sin mutar pesos (Matemáticamente puro)
        for ((&h_val, &b1), (d_h, w2_row)) in hidden
            .iter()
            .zip(self.b1.iter())
            .zip(d_hidden.iter_mut().zip(self.w2.iter()))
        {
            let raw_h = h_val + b1;
            let relu_grad = if raw_h > 0.0 { 1.0 } else { 0.01 };

            // ⚡ MOTOR CUÁNTICO AVX2: Zipping estricto purga bounds-checking
            for (&w2_val, &d_out_val) in w2_row.iter().zip(d_out.iter()) {
                *d_h += d_out_val * w2_val * relu_grad;
            }
        }

        // Fase 2: Aplicar mutaciones L2 a w2 y b2 (Después de propagar gradientes)
        for ((&h_val, &b1), w2_row) in hidden.iter().zip(self.b1.iter()).zip(self.w2.iter_mut()) {
            let raw_h = h_val + b1;
            let h_act = if raw_h > 0.0 { raw_h } else { raw_h * 0.01 };
            for (w2_current, &d_out_val) in w2_row.iter_mut().zip(d_out.iter()) {
                let grad_w2 = (d_out_val * h_act).clamp(-5.0, 5.0);
                *w2_current = (*w2_current * 0.9999 - lr * grad_w2).clamp(-10.0, 10.0);
            }
        }

        // Actualizar sesgos capa 2
        for (b, &d) in self.b2.iter_mut().zip(d_out.iter()) {
            *b = (*b - lr * d).clamp(-5.0, 5.0);
        }

        // Actualizar capa 1
        for (&in_val, w1_row) in clean_inputs.iter().zip(self.w1.iter_mut()) {
            for (w1_val, &d_h) in w1_row.iter_mut().zip(d_hidden.iter()) {
                let grad_w1 = (d_h * in_val).clamp(-5.0, 5.0);
                *w1_val = (*w1_val * 0.9999 - lr * grad_w1).clamp(-10.0, 10.0);
            }
        }
        for (b, &d) in self.b1.iter_mut().zip(d_hidden.iter()) {
            *b = (*b - lr * d).clamp(-5.0, 5.0);
        }
    }

    /// Inferencia Cuantizada Int8 en Caché L1 (Punto #078)
    /// Transforma el cálculo matricial en operaciones enteras en registros SIMD.
    /// Devuelve [Probabilidad Long, Probabilidad Short] normalizadas en f64.
    #[inline(always)]
    pub fn infer_quantized_i8(&self, inputs: &[f64; 34]) -> [f64; 2] {
        let mut clean_inputs = [0i32; 34];
        for (clean, &raw) in clean_inputs.iter_mut().zip(inputs.iter()) {
            let val = if raw.is_finite() {
                raw.clamp(-10.0, 10.0)
            } else {
                0.0
            };
            *clean = (val * 12.7) as i32; // Escala fija Q7
        }

        let mut hidden = [0i32; 16];
        for (&in_val, w1_row) in clean_inputs.iter().zip(self.w1.iter()) {
            for (h, &w) in hidden.iter_mut().zip(w1_row.iter()) {
                let w_i8 = (w * 12.7).clamp(-127.0, 127.0) as i32;
                *h += in_val * w_i8;
            }
        }

        // Activación Leaky ReLU con sesgo cuantizado
        let mut hidden_act = [0f64; 16];
        for (h_act, (&h, &b)) in hidden_act.iter_mut().zip(hidden.iter().zip(self.b1.iter())) {
            let b_scaled = (b * 161.29) as i32;
            let raw_h = h + b_scaled;
            let val = if raw_h > 0 {
                raw_h as f64
            } else {
                raw_h as f64 * 0.01
            };
            *h_act = val / 161.29;
        }

        let mut outputs = [self.b2[0], self.b2[1]];
        for (&h_val, w2_row) in hidden_act.iter().zip(self.w2.iter()) {
            for (out, &w) in outputs.iter_mut().zip(w2_row.iter()) {
                *out += h_val * w;
            }
        }

        let max_val = outputs[0].max(outputs[1]);
        let exp_0 = (outputs[0] - max_val).exp();
        let exp_1 = (outputs[1] - max_val).exp();
        let sum_exp = exp_0 + exp_1;

        if sum_exp > 0.0 && !sum_exp.is_nan() {
            [
                (exp_0 / sum_exp).clamp(0.0001, 0.9999),
                (exp_1 / sum_exp).clamp(0.0001, 0.9999),
            ]
        } else {
            [0.5, 0.5]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_neural_net_numerical_stability() {
        let mut net = SimdNeuralNet::default();
        // Set very high bias to test softmax overflow immunity
        net.b2 = [1000.0, 1000.0];
        let inputs = [100.0; 34];
        let out = net.infer(&inputs);

        assert!(!out[0].is_nan(), "Output 0 must not be NaN");
        assert!(!out[1].is_nan(), "Output 1 must not be NaN");
        assert!(
            (out[0] + out[1] - 1.0).abs() < 1e-4,
            "Softmax sum must equal 1.0"
        );
    }

    #[test]
    fn test_simd_neural_net_train_step_bounded() {
        let mut net = SimdNeuralNet::default();
        let inputs = [0.5; 34];
        for _ in 0..100 {
            net.train_step(&inputs, 0, 0.01);
        }
        let out = net.infer(&inputs);
        assert!(
            out[0] > out[1],
            "Net should learn class 0 probability increase"
        );
        assert!(out[0].is_finite() && out[1].is_finite());
    }

    #[test]
    fn test_simd_neural_net_nan_input_immunity() {
        let net = SimdNeuralNet::default();
        let mut inputs = [0.5; 34];
        inputs[0] = f64::NAN;
        inputs[1] = f64::INFINITY;
        inputs[2] = f64::NEG_INFINITY;
        let out = net.infer(&inputs);
        assert!(out[0].is_finite());
        assert!(out[1].is_finite());
        assert!((out[0] + out[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_simd_neural_net_quantized_i8_inference() {
        let net = SimdNeuralNet::default();
        let inputs = [0.5; 34];
        let out = net.infer_quantized_i8(&inputs);
        assert!(out[0].is_finite() && out[1].is_finite());
        assert!((out[0] + out[1] - 1.0).abs() < 1e-4);
        assert!(out[0] > 0.0 && out[1] > 0.0);
    }

    #[test]
    fn test_simd_neural_net_train_step_nan_lr_immunity() {
        let mut net = SimdNeuralNet::default();
        let nan_inputs = [f64::NAN; 34];
        net.train_step(&nan_inputs, 0, f64::NAN);
        let out = net.infer(&nan_inputs);
        assert!(out[0].is_finite());
        assert!(out[1].is_finite());
    }
}
