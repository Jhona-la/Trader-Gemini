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
        for i in 0..34 {
            for j in 0..16 {
                w1[i][j] = ((i * 17 + j * 31 + 7) as f64).sin() * 0.2425;
            }
        }
        for j in 0..16 {
            for k in 0..2 {
                w2[j][k] = ((j * 13 + k * 29 + 11) as f64).sin() * 0.3333;
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
        for i in 0..34 {
            clean_inputs[i] = if inputs[i].is_finite() { inputs[i].clamp(-100.0, 100.0) } else { 0.0 };
        }

        let mut hidden = [0.0; 16];
        
        // Multiplicación de Matrices Capa 1 (El compilador usa AVX/SIMD automáticamente aquí)
        for i in 0..34 {
            let in_val = clean_inputs[i];
            for j in 0..16 {
                hidden[j] += in_val * self.w1[i][j];
            }
        }
        
        // Aplicar Sesgo y Activación ReLU (Capa Oculta)
        for j in 0..16 {
            hidden[j] = (hidden[j] + self.b1[j]).max(0.0);
        }

        let mut outputs = [0.0; 2];
        
        // Multiplicación Capa 2
        for j in 0..16 {
            let h_val = hidden[j];
            for k in 0..2 {
                outputs[k] += h_val * self.w2[j][k];
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
        for i in 0..34 {
            clean_inputs[i] = if inputs[i].is_finite() { inputs[i].clamp(-100.0, 100.0) } else { 0.0 };
        }
        let lr = if learning_rate.is_finite() { learning_rate.clamp(1e-5, 0.1) } else { 0.001 };
        let probs = self.infer(&clean_inputs);

        // Error de salida con Cross-Entropy
        let target = if target_idx == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
        let d_out = [
            (probs[0] - target[0]).clamp(-2.0, 2.0),
            (probs[1] - target[1]).clamp(-2.0, 2.0),
        ];

        // Forward activations para backprop
        let mut hidden = [0.0; 16];
        for i in 0..34 {
            let in_val = clean_inputs[i];
            for j in 0..16 {
                hidden[j] += in_val * self.w1[i][j];
            }
        }
        let mut d_hidden = [0.0; 16];
        for j in 0..16 {
            let h_val = (hidden[j] + self.b1[j]).max(0.0);
            let relu_grad = if h_val > 0.0 { 1.0 } else { 0.0 };

            for k in 0..2 {
                let w2_current = self.w2[j][k];
                // FIX #391: Propagar el error a la capa oculta usando el peso original antes de la mutación
                d_hidden[j] += d_out[k] * w2_current * relu_grad;

                let grad_w2 = (d_out[k] * h_val).clamp(-5.0, 5.0);
                self.w2[j][k] = (w2_current * 0.9999 - lr * grad_w2).clamp(-10.0, 10.0);
            }
        }

        // Actualizar sesgos capa 2
        self.b2[0] = (self.b2[0] - lr * d_out[0]).clamp(-5.0, 5.0);
        self.b2[1] = (self.b2[1] - lr * d_out[1]).clamp(-5.0, 5.0);

        // Actualizar capa 1
        for i in 0..34 {
            let in_val = clean_inputs[i];
            for j in 0..16 {
                let grad_w1 = (d_hidden[j] * in_val).clamp(-5.0, 5.0);
                self.w1[i][j] = (self.w1[i][j] * 0.9999 - lr * grad_w1).clamp(-10.0, 10.0);
            }
        }
        for j in 0..16 {
            self.b1[j] = (self.b1[j] - lr * d_hidden[j]).clamp(-5.0, 5.0);
        }
    }

    /// Inferencia Cuantizada Int8 en Caché L1 (Punto #078)
    /// Transforma el cálculo matricial en operaciones enteras en registros SIMD.
    /// Devuelve [Probabilidad Long, Probabilidad Short] normalizadas en f64.
    #[inline(always)]
    pub fn infer_quantized_i8(&self, inputs: &[f64; 34]) -> [f64; 2] {
        let mut clean_inputs = [0i32; 34];
        for i in 0..34 {
            let val = if inputs[i].is_finite() { inputs[i].clamp(-10.0, 10.0) } else { 0.0 };
            clean_inputs[i] = (val * 12.7) as i32; // Escala fija Q7
        }

        let mut hidden = [0i32; 16];
        for i in 0..34 {
            let in_val = clean_inputs[i];
            for j in 0..16 {
                let w_i8 = (self.w1[i][j] * 12.7).clamp(-127.0, 127.0) as i32;
                hidden[j] += in_val * w_i8;
            }
        }

        // Activación ReLU con sesgo cuantizado
        let mut hidden_act = [0f64; 16];
        for j in 0..16 {
            let b_scaled = (self.b1[j] * 161.29) as i32;
            let val = (hidden[j] + b_scaled).max(0);
            hidden_act[j] = val as f64 / 161.29;
        }

        let mut outputs = [self.b2[0], self.b2[1]];
        for j in 0..16 {
            let h_val = hidden_act[j];
            for k in 0..2 {
                outputs[k] += h_val * self.w2[j][k];
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
        assert!((out[0] + out[1] - 1.0).abs() < 1e-4, "Softmax sum must equal 1.0");
    }

    #[test]
    fn test_simd_neural_net_train_step_bounded() {
        let mut net = SimdNeuralNet::default();
        let inputs = [0.5; 34];
        for _ in 0..100 {
            net.train_step(&inputs, 0, 0.01);
        }
        let out = net.infer(&inputs);
        assert!(out[0] > out[1], "Net should learn class 0 probability increase");
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

