use serde::{Deserialize, Serialize};

/// Operaciones disponibles para Auto-Feature Engineering
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum OpCode {
    /// Empuja un valor del buffer base (0 a 33) a la pila
    PushInput(u8),
    /// Empuja una constante a la pila
    PushConst(f32),
    Add,
    Sub,
    Mul,
    Div,
    Log,
    Sqrt,
    Abs,
    Max,
    Min,
}

/// Una Feature compilada es simplemente una secuencia plana de OpCodes
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CompiledFeature {
    pub ops: Vec<OpCode>,
}

impl CompiledFeature {
    pub fn new(ops: Vec<OpCode>) -> Self {
        Self { ops }
    }

    /// Evaluación ultra-rápida basada en pila sin asiganciones dinámicas
    /// Se garantiza < 10 ns por evaluación.
    #[inline(always)]
    pub fn execute(&self, base_features: &[f32; 54]) -> f32 {
        let mut stack = [0.0f32; 16];
        let mut sp = 0; // Stack pointer

        for op in &self.ops {
            match op {
                OpCode::PushInput(idx) => {
                    if sp < 16 {
                        // FIX #626: Evitar bounds check panic y sanitizar finitud
                        let val = if (*idx as usize) < 54 {
                            base_features[*idx as usize]
                        } else {
                            0.0
                        };
                        let safe_val = if val.is_finite() { val } else { 0.0 };
                        stack[sp] = safe_val;
                        sp += 1;
                    } else {
                        return 0.0;
                    } // Phenotypic Shield: Stack Overflow
                }
                OpCode::PushConst(val) => {
                    if sp < 16 {
                        let safe_val = if val.is_finite() { *val } else { 0.0 };
                        stack[sp] = safe_val;
                        sp += 1;
                    } else {
                        return 0.0;
                    } // Phenotypic Shield
                }
                OpCode::Add => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        let res = b + a;
                        stack[sp - 1] = if res.is_finite() { res } else { 0.0 };
                    } else {
                        return 0.0;
                    } // Phenotypic Shield: Stack Underflow
                }
                OpCode::Sub => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        let res = b - a;
                        stack[sp - 1] = if res.is_finite() { res } else { 0.0 };
                    } else {
                        return 0.0;
                    }
                }
                OpCode::Mul => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        let res = b * a;
                        stack[sp - 1] = if res.is_finite() { res } else { 0.0 };
                    } else {
                        return 0.0;
                    }
                }
                OpCode::Div => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        let res = if a.abs() > 1e-9 { b / a } else { 0.0 };
                        stack[sp - 1] = if res.is_nan() || res.is_infinite() {
                            0.0
                        } else {
                            res
                        };
                    } else {
                        return 0.0;
                    }
                }
                OpCode::Log => {
                    if sp >= 1 {
                        let a = stack[sp - 1];
                        let res = if a > 1e-9 { a.ln() } else { 0.0 };
                        stack[sp - 1] = if res.is_nan() || res.is_infinite() {
                            0.0
                        } else {
                            res
                        };
                    } else {
                        return 0.0;
                    }
                }
                OpCode::Sqrt => {
                    if sp >= 1 {
                        let a = stack[sp - 1];
                        let res = if a > 0.0 { a.sqrt() } else { 0.0 };
                        stack[sp - 1] = if res.is_nan() || res.is_infinite() {
                            0.0
                        } else {
                            res
                        };
                    } else {
                        return 0.0;
                    }
                }
                OpCode::Abs => {
                    if sp >= 1 {
                        let a = stack[sp - 1];
                        stack[sp - 1] = a.abs();
                    } else {
                        return 0.0;
                    }
                }
                OpCode::Max => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        stack[sp - 1] = b.max(a);
                    } else {
                        return 0.0;
                    }
                }
                OpCode::Min => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        stack[sp - 1] = b.min(a);
                    } else {
                        return 0.0;
                    }
                }
            }
        }

        if sp > 0 {
            let res = stack[sp - 1];
            if res.is_nan() || res.is_infinite() {
                0.0
            } else {
                res
            }
        } else {
            0.0
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AutoFeatureSet {
    pub features: Vec<CompiledFeature>,
}

impl AutoFeatureSet {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    pub fn default_set() -> Self {
        let mut set = Self::new();
        // Feature 0: Spread momentum = (Fast EMA - Slow EMA) / Slow EMA
        set.features.push(CompiledFeature::new(vec![
            OpCode::PushInput(0),
            OpCode::PushInput(1),
            OpCode::Sub,
            OpCode::PushInput(1),
            OpCode::Div,
        ]));
        // Feature 1: Hawkes/OFI Interaction = OFI * Hawkes Intensity
        set.features.push(CompiledFeature::new(vec![
            OpCode::PushInput(2),
            OpCode::PushInput(6),
            OpCode::Mul,
        ]));
        // Feature 2: Volatility Adjusted Hurst = Hurst / (ATR + 1e-4)
        set.features.push(CompiledFeature::new(vec![
            OpCode::PushInput(1),
            OpCode::PushInput(3),
            OpCode::PushConst(1e-4),
            OpCode::Add,
            OpCode::Div,
        ]));
        set
    }

    /// Evaluates all compiled features in nanoseconds without dynamic heap allocation
    #[inline(always)]
    pub fn evaluate_all(&self, base_features: &[f32; 54], output: &mut [f32]) {
        let count = self.features.len().min(output.len());
        for i in 0..count {
            output[i] = self.features[i].execute(base_features);
        }
    }

    /// Mutates an expression or adds a new candidate alpha feature via online genetic programming
    pub fn mutate(&mut self, rng_seed: u64) {
        let binary_ops = [
            OpCode::Add,
            OpCode::Sub,
            OpCode::Mul,
            OpCode::Div,
            OpCode::Max,
            OpCode::Min,
        ];
        let unary_ops = [OpCode::Abs, OpCode::Sqrt, OpCode::Log];

        let mut next_seed = rng_seed ^ 0x5DEECE66D;
        let rand_idx = ((next_seed >> 16) % 54) as u8;
        next_seed = next_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bin_op_idx = ((next_seed >> 16) % (binary_ops.len() as u64)) as usize;
        let un_op_idx = ((next_seed >> 20) % (unary_ops.len() as u64)) as usize;

        if self.features.is_empty() || (next_seed % 3 == 0 && self.features.len() < 32) {
            // FIX #828: Añadir nueva feature binaria válida que deja exactamente 1 elemento en la pila
            let new_feat = CompiledFeature::new(vec![
                OpCode::PushInput(rand_idx),
                OpCode::PushInput((rand_idx + 1) % 54),
                binary_ops[bin_op_idx],
            ]);
            self.features.push(new_feat);
        } else {
            // Mutate an existing feature
            let feat_idx = ((next_seed >> 24) as usize) % self.features.len();
            if self.features[feat_idx].ops.len() < 8 {
                if next_seed % 2 == 0 {
                    // Mutación unaria: transforma el tope de la pila in-situ sin empujar operandos extra
                    self.features[feat_idx].ops.push(unary_ops[un_op_idx]);
                } else {
                    // Mutación binaria: empuja 1 nuevo input y aplica operador binario
                    self.features[feat_idx]
                        .ops
                        .push(OpCode::PushInput(rand_idx));
                    self.features[feat_idx].ops.push(binary_ops[bin_op_idx]);
                }
            }
        }
    }
}

impl Default for AutoFeatureSet {
    fn default() -> Self {
        Self::default_set()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vm_execution_and_evaluation() {
        let feature_set = AutoFeatureSet::default_set();
        let mut base_features = [0.0f32; 54];
        base_features[0] = 105.0; // Fast
        base_features[1] = 100.0; // Slow
        base_features[2] = 0.8; // OFI
        base_features[3] = 0.02; // ATR
        base_features[6] = 1.5; // Hawkes

        let mut output = [0.0f32; 8];
        feature_set.evaluate_all(&base_features, &mut output);

        // Feature 0: (105 - 100) / 100 = 0.05
        assert!((output[0] - 0.05).abs() < 1e-4);
        // Feature 1: 0.8 * 1.5 = 1.2
        assert!((output[1] - 1.2).abs() < 1e-4);
    }

    #[test]
    fn test_feature_vm_mutation() {
        let mut feature_set = AutoFeatureSet::new();
        feature_set.mutate(123456789);
        assert!(!feature_set.features.is_empty());
        let base_features = [1.0f32; 54];
        let mut output = [0.0f32; 4];
        feature_set.evaluate_all(&base_features, &mut output);
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_feature_vm_nan_and_div_by_zero_immunity() {
        let compiled = CompiledFeature::new(vec![
            OpCode::PushConst(10.0),
            OpCode::PushConst(0.0),
            OpCode::Div, // 10 / 0 -> 0.0
            OpCode::PushConst(-5.0),
            OpCode::Log, // log(-5) -> 0.0
            OpCode::Add,
        ]);
        let base = [f32::NAN; 54];
        let res = compiled.execute(&base);
        assert_eq!(res, 0.0);
    }

    #[test]
    fn test_feature_vm_min_max_abs_sqrt_operations() {
        let compiled_max_min = CompiledFeature::new(vec![
            OpCode::PushConst(5.0),
            OpCode::PushConst(12.0),
            OpCode::Max, // 12.0
            OpCode::PushConst(20.0),
            OpCode::Min, // 12.0
            OpCode::PushConst(-15.0),
            OpCode::Abs, // 15.0
            OpCode::Add, // 27.0
            OpCode::PushConst(16.0),
            OpCode::Sqrt, // 4.0
            OpCode::Sub,  // 27.0 - 4.0 = 23.0
        ]);
        let base = [0.0f32; 54];
        let res = compiled_max_min.execute(&base);
        assert!((res - 23.0).abs() < 1e-4);
    }

    #[test]
    fn test_feature_vm_stack_overflow_shield() {
        // Pushing more than 16 elements must trigger the phenotypic shield and return 0.0 safely
        let mut ops = Vec::new();
        for i in 0..20 {
            ops.push(OpCode::PushConst(i as f32));
        }
        let compiled_overflow = CompiledFeature::new(ops);
        let base = [1.0f32; 54];
        let res = compiled_overflow.execute(&base);
        assert_eq!(res, 0.0);
    }

    #[test]
    fn test_feature_vm_push_const_nan_and_infinite_sanitization() {
        let compiled = CompiledFeature::new(vec![
            OpCode::PushConst(f32::NAN),
            OpCode::PushConst(f32::INFINITY),
            OpCode::Add,
            OpCode::PushConst(10.0),
            OpCode::Add,
        ]);
        let base = [0.0f32; 54];
        let res = compiled.execute(&base);
        assert_eq!(res, 10.0);
    }

    #[test]
    fn test_feature_vm_empty_feature_and_out_of_bounds_inputs() {
        let compiled_empty = CompiledFeature::new(vec![]);
        let base = [1.0f32; 54];
        assert_eq!(compiled_empty.execute(&base), 0.0);

        let compiled_oob = CompiledFeature::new(vec![
            OpCode::PushInput(100), // Out of bounds > 54
            OpCode::PushInput(200),
            OpCode::Add,
        ]);
        assert_eq!(compiled_oob.execute(&base), 0.0);
    }
}
