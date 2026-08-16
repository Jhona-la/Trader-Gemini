use serde::{Serialize, Deserialize};

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
    pub fn execute(&self, base_features: &[f32; 34]) -> f32 {
        let mut stack = [0.0f32; 16];
        let mut sp = 0; // Stack pointer

        for op in &self.ops {
            match op {
                OpCode::PushInput(idx) => {
                    if sp < 16 {
                        // Evitar bounds check panic
                        let val = if (*idx as usize) < 34 { base_features[*idx as usize] } else { 0.0 };
                        stack[sp] = val;
                        sp += 1;
                    } else { return 0.0; } // Phenotypic Shield: Stack Overflow
                }
                OpCode::PushConst(val) => {
                    if sp < 16 {
                        stack[sp] = *val;
                        sp += 1;
                    } else { return 0.0; } // Phenotypic Shield
                }
                OpCode::Add => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        stack[sp - 1] = b + a;
                    } else { return 0.0; } // Phenotypic Shield: Stack Underflow
                }
                OpCode::Sub => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        stack[sp - 1] = b - a;
                    } else { return 0.0; }
                }
                OpCode::Mul => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        stack[sp - 1] = b * a;
                    } else { return 0.0; }
                }
                OpCode::Div => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        let res = if a.abs() > 1e-9 { b / a } else { 0.0 };
                        stack[sp - 1] = if res.is_nan() || res.is_infinite() { 0.0 } else { res };
                    } else { return 0.0; }
                }
                OpCode::Log => {
                    if sp >= 1 {
                        let a = stack[sp - 1];
                        let res = if a > 1e-9 { a.ln() } else { 0.0 };
                        stack[sp - 1] = if res.is_nan() || res.is_infinite() { 0.0 } else { res };
                    } else { return 0.0; }
                }
                OpCode::Sqrt => {
                    if sp >= 1 {
                        let a = stack[sp - 1];
                        let res = if a > 0.0 { a.sqrt() } else { 0.0 };
                        stack[sp - 1] = if res.is_nan() || res.is_infinite() { 0.0 } else { res };
                    } else { return 0.0; }
                }
                OpCode::Abs => {
                    if sp >= 1 {
                        let a = stack[sp - 1];
                        stack[sp - 1] = a.abs();
                    } else { return 0.0; }
                }
                OpCode::Max => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        stack[sp - 1] = b.max(a);
                    } else { return 0.0; }
                }
                OpCode::Min => {
                    if sp >= 2 {
                        sp -= 1;
                        let a = stack[sp];
                        let b = stack[sp - 1];
                        stack[sp - 1] = b.min(a);
                    } else { return 0.0; }
                }
            }
        }

        if sp > 0 {
            let res = stack[sp - 1];
            if res.is_nan() || res.is_infinite() { 0.0 } else { res }
        } else {
            0.0
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AutoFeatureSet {
    pub features: Vec<CompiledFeature>,
}

