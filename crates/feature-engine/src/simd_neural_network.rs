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
        Self {
            w1: [[0.01; 16]; 34],
            b1: [0.0; 16],
            w2: [[0.01; 2]; 16],
            b2: [0.0; 2],
        }
    }
}

impl SimdNeuralNet {
    /// Inferencia Forward Pass (O(1) memoria, microsegundos)
    /// Devuelve [Probabilidad Long, Probabilidad Short]
    #[inline(always)]
    pub fn infer(&self, inputs: &[f64; 34]) -> [f64; 2] {
        let mut hidden = [0.0; 16];
        
        // Multiplicación de Matrices Capa 1 (El compilador usa AVX/SIMD automáticamente aquí)
        for i in 0..34 {
            let in_val = inputs[i];
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

        // Activación de Salida (Softmax manual en 2 variables)
        outputs[0] += self.b2[0];
        outputs[1] += self.b2[1];
        
        let exp_0 = outputs[0].exp();
        let exp_1 = outputs[1].exp();
        let sum_exp = exp_0 + exp_1;
        
        if sum_exp > 0.0 {
            [exp_0 / sum_exp, exp_1 / sum_exp]
        } else {
            [0.5, 0.5] // Neutral fallback
        }
    }
}
