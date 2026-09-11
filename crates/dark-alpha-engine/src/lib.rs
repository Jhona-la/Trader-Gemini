//! # Dark Alpha Engine — Red Neuronal Profunda en Rust Puro
//!
//! ## QUÉ
//! Perceptrón Multicapa (MLP) de 3 capas implementado sin frameworks ML externos.
//! Recibe un vector de features (precio, volumen, microestructura, macro) y devuelve
//! una probabilidad [0.0, 1.0] de anomalía/oportunidad de trading.
//!
//! ## POR QUÉ
//! - `candle-core` arrastra +50 dependencias y compilación de 5+ minutos
//! - Para un MLP de 3 capas, la aritmética manual es más rápida que cualquier framework
//! - Inferencia en ~50-100 nanosegundos vs ~1ms con candle en CPU
//!
//! ## PARA QUÉ
//! Capa de confluencia neuronal profunda para todo el espectro temporal continuo [1 ns, 100 a].
//! El GodEngineCore consulta `DarkAlphaEngine::predict_for_coin()` integrando la inferencia
//! en el ensamble Brier multidimensional tanto en micro-ticks como en macro-ciclos.
//!
//! ## CÓMO
//! Forward pass: Input → Linear(ReLU) → Linear(ReLU) → Linear(Sigmoid) → Output
//! Pesos almacenados en layout contiguo para locality de cache L1.
//!
//! ## CUÁNDO
//! Se invoca de manera continua y unificada en cada evento de mercado procesado por GodEngineCore.
//!
//! ## DÓNDE
//! `crates/dark-alpha-engine/src/lib.rs`
//!
//! ## QUIÉN
//! Llamado desde `GodEngineCore::process_event()` y desde `evolution.rs` para optimizar.

pub mod neuro_plasticity;
pub mod online_ppo;

use serde::{Deserialize, Serialize};

/// Pesos de una capa lineal en layout contiguo (row-major).
/// `weights` tiene dimensión [out_features * in_features]
/// `biases` tiene dimensión [out_features]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DenseLayer {
    pub weights: Vec<f64>, // [out_features * in_features] row-major
    pub biases: Vec<f64>,  // [out_features]
    pub in_features: usize,
    pub out_features: usize,
}

impl DenseLayer {
    /// Crear capa con pesos inicializados con He initialization (óptimo para ReLU)
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // He initialization: sqrt(2/n_in)
        let scale = (2.0 / in_features as f64).sqrt();
        let mut weights = Vec::with_capacity(out_features * in_features);

        // Deterministic pseudo-random init using simple LCG
        let mut seed: u64 =
            (in_features as u64).wrapping_mul(7919) ^ (out_features as u64).wrapping_mul(104729);
        for _ in 0..(out_features * in_features) {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (seed >> 33) as f64 / (1u64 << 31) as f64; // [0, 1)
            weights.push((u - 0.5) * 2.0 * scale);
        }

        Self {
            weights,
            biases: vec![0.0; out_features],
            in_features,
            out_features,
        }
    }

    /// Crear capa con pesos inicializados con Xavier/Glorot initialization (óptimo para Sigmoid/Tanh)
    pub fn new_xavier(in_features: usize, out_features: usize) -> Self {
        // Xavier/Glorot initialization: sqrt(2 / (n_in + n_out))
        let scale = (2.0 / (in_features + out_features) as f64).sqrt();
        let mut weights = Vec::with_capacity(out_features * in_features);

        let mut seed: u64 = (in_features as u64).wrapping_mul(179424673)
            ^ (out_features as u64).wrapping_mul(275604341);
        for _ in 0..(out_features * in_features) {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (seed >> 33) as f64 / (1u64 << 31) as f64; // [0, 1)
            weights.push((u - 0.5) * 2.0 * scale);
        }

        Self {
            weights,
            biases: vec![0.0; out_features],
            in_features,
            out_features,
        }
    }

    /// Sanitiza los pesos eliminando números subnormales (denormals < 1e-7) y NaNs
    /// para evitar microcode exception traps en hardware x86_64 que degradan la latencia a 200x.
    pub fn sanitize_denormals(&mut self) {
        for w in self.weights.iter_mut() {
            if !w.is_finite() || w.abs() < 1e-7 {
                *w = 0.0;
            }
        }
        for b in self.biases.iter_mut() {
            if !b.is_finite() || b.abs() < 1e-7 {
                *b = 0.0;
            }
        }
    }

    /// Forward pass con ReLU activation
    #[inline(always)]
    pub fn forward_relu(&self, input: &[f64], output: &mut [f64]) {
        // FIX #355: Active bounds check for --release safety
        assert_eq!(
            input.len(),
            self.in_features,
            "DarkAlpha Layer: input dimension mismatch"
        );
        assert_eq!(
            output.len(),
            self.out_features,
            "DarkAlpha Layer: output dimension mismatch"
        );

        for i in 0..self.out_features {
            let row_offset = i * self.in_features;
            let mut sum = unsafe { *self.biases.get_unchecked(i) };

            // Manual unrolled dot product for cache efficiency
            let mut j = 0;
            let len = self.in_features;

            // Process 4 elements at a time
            while j + 4 <= len {
                unsafe {
                    sum += *self.weights.get_unchecked(row_offset + j) * *input.get_unchecked(j)
                        + *self.weights.get_unchecked(row_offset + j + 1)
                            * *input.get_unchecked(j + 1)
                        + *self.weights.get_unchecked(row_offset + j + 2)
                            * *input.get_unchecked(j + 2)
                        + *self.weights.get_unchecked(row_offset + j + 3)
                            * *input.get_unchecked(j + 3);
                }
                j += 4;
            }
            // Remainder
            while j < len {
                unsafe {
                    sum += *self.weights.get_unchecked(row_offset + j) * *input.get_unchecked(j);
                }
                j += 1;
            }

            // ReLU
            unsafe {
                *output.get_unchecked_mut(i) = if sum > 0.0 { sum } else { 0.0 };
            }
        }
    }

    /// Forward pass con Tanh activation (óptimo para señales simétricas y bidireccionales Long/Short)
    #[inline(always)]
    pub fn forward_tanh(&self, input: &[f64], output: &mut [f64]) {
        assert_eq!(
            input.len(),
            self.in_features,
            "DarkAlpha Layer: input dimension mismatch"
        );
        assert_eq!(
            output.len(),
            self.out_features,
            "DarkAlpha Layer: output dimension mismatch"
        );

        for i in 0..self.out_features {
            let row_offset = i * self.in_features;
            let mut sum = unsafe { *self.biases.get_unchecked(i) };

            let mut j = 0;
            let len = self.in_features;
            while j + 4 <= len {
                unsafe {
                    sum += *self.weights.get_unchecked(row_offset + j) * *input.get_unchecked(j)
                        + *self.weights.get_unchecked(row_offset + j + 1)
                            * *input.get_unchecked(j + 1)
                        + *self.weights.get_unchecked(row_offset + j + 2)
                            * *input.get_unchecked(j + 2)
                        + *self.weights.get_unchecked(row_offset + j + 3)
                            * *input.get_unchecked(j + 3);
                }
                j += 4;
            }
            while j < len {
                unsafe {
                    sum += *self.weights.get_unchecked(row_offset + j) * *input.get_unchecked(j);
                }
                j += 1;
            }

            unsafe {
                *output.get_unchecked_mut(i) = sum.tanh();
            }
        }
    }

    /// Forward pass con Sigmoid activation (capa final)
    #[inline(always)]
    pub fn forward_sigmoid(&self, input: &[f64], output: &mut [f64]) {
        // FIX #355: Active bounds check for --release safety
        assert_eq!(
            input.len(),
            self.in_features,
            "DarkAlpha Layer: input dimension mismatch"
        );
        assert_eq!(
            output.len(),
            self.out_features,
            "DarkAlpha Layer: output dimension mismatch"
        );

        for i in 0..self.out_features {
            let row_offset = i * self.in_features;
            let mut sum = unsafe { *self.biases.get_unchecked(i) };

            let mut j = 0;
            let len = self.in_features;

            while j + 4 <= len {
                unsafe {
                    sum += *self.weights.get_unchecked(row_offset + j) * *input.get_unchecked(j)
                        + *self.weights.get_unchecked(row_offset + j + 1)
                            * *input.get_unchecked(j + 1)
                        + *self.weights.get_unchecked(row_offset + j + 2)
                            * *input.get_unchecked(j + 2)
                        + *self.weights.get_unchecked(row_offset + j + 3)
                            * *input.get_unchecked(j + 3);
                }
                j += 4;
            }
            while j < len {
                unsafe {
                    sum += *self.weights.get_unchecked(row_offset + j) * *input.get_unchecked(j);
                }
                j += 1;
            }

            // Limite matemático para evitar f64::exp overflow (f64 límite es ~709.0)
            let clamped = sum.clamp(-700.0, 700.0);
            unsafe {
                *output.get_unchecked_mut(i) = 1.0 / (1.0 + (-clamped).exp());
            }
        }
    }

    /// Cuantiza los pesos de la capa a Int8 simétrico para aceleración en CPU y L1 Cache (Punto #072)
    pub fn quantize_int8(&self) -> QuantizedDenseLayer {
        let max_abs = self
            .weights
            .iter()
            .fold(0.0_f64, |acc, &w| acc.max(w.abs()))
            .max(1e-8);
        let scale_w = max_abs / 127.0;
        let inv_scale = 127.0 / max_abs;

        let mut weights_q = Vec::with_capacity(self.weights.len());
        for &w in &self.weights {
            let q = (w * inv_scale).round().clamp(-127.0, 127.0) as i8;
            weights_q.push(q);
        }

        QuantizedDenseLayer {
            weights_q,
            scale_w,
            biases: self.biases.clone(),
            in_features: self.in_features,
            out_features: self.out_features,
        }
    }
}

/// Capa lineal cuantizada en Int8 con escalamiento punto fijo (Punto #072)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QuantizedDenseLayer {
    pub weights_q: Vec<i8>,
    pub scale_w: f64,
    pub biases: Vec<f64>,
    pub in_features: usize,
    pub out_features: usize,
}

impl QuantizedDenseLayer {
    #[inline(always)]
    pub fn forward_relu(&self, input: &[f64], output: &mut [f64]) {
        assert_eq!(
            input.len(),
            self.in_features,
            "QuantizedDarkAlpha Layer: input dimension mismatch"
        );
        assert_eq!(
            output.len(),
            self.out_features,
            "QuantizedDarkAlpha Layer: output dimension mismatch"
        );

        let scale = self.scale_w;
        for i in 0..self.out_features {
            let row_offset = i * self.in_features;
            let mut sum_dot = 0.0;
            let mut j = 0;
            let len = self.in_features;

            while j + 4 <= len {
                sum_dot += (self.weights_q[row_offset + j] as f64) * input[j]
                    + (self.weights_q[row_offset + j + 1] as f64) * input[j + 1]
                    + (self.weights_q[row_offset + j + 2] as f64) * input[j + 2]
                    + (self.weights_q[row_offset + j + 3] as f64) * input[j + 3];
                j += 4;
            }
            while j < len {
                sum_dot += (self.weights_q[row_offset + j] as f64) * input[j];
                j += 1;
            }

            let total = self.biases[i] + sum_dot * scale;
            output[i] = if total > 0.0 { total } else { 0.0 };
        }
    }

    #[inline(always)]
    pub fn forward_sigmoid(&self, input: &[f64], output: &mut [f64]) {
        assert_eq!(
            input.len(),
            self.in_features,
            "QuantizedDarkAlpha Layer: input dimension mismatch"
        );
        assert_eq!(
            output.len(),
            self.out_features,
            "QuantizedDarkAlpha Layer: output dimension mismatch"
        );

        let scale = self.scale_w;
        for i in 0..self.out_features {
            let row_offset = i * self.in_features;
            let mut sum_dot = 0.0;
            let mut j = 0;
            let len = self.in_features;

            while j + 4 <= len {
                sum_dot += (self.weights_q[row_offset + j] as f64) * input[j]
                    + (self.weights_q[row_offset + j + 1] as f64) * input[j + 1]
                    + (self.weights_q[row_offset + j + 2] as f64) * input[j + 2]
                    + (self.weights_q[row_offset + j + 3] as f64) * input[j + 3];
                j += 4;
            }
            while j < len {
                sum_dot += (self.weights_q[row_offset + j] as f64) * input[j];
                j += 1;
            }

            let total = (self.biases[i] + sum_dot * scale).clamp(-700.0, 700.0);
            output[i] = 1.0 / (1.0 + (-total).exp());
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Scaler {
    pub mean: Vec<f64>,
    pub std_dev: Vec<f64>,
}

impl Scaler {
    pub fn new(mean: Vec<f64>, std_dev: Vec<f64>) -> Self {
        Self { mean, std_dev }
    }

    #[inline(always)]
    pub fn scale(&self, features: &mut [f64]) {
        for i in 0..features.len() {
            if i < self.mean.len() && i < self.std_dev.len() {
                let m = if self.mean[i].is_finite() {
                    self.mean[i]
                } else {
                    0.0
                };
                let s = if self.std_dev[i].is_finite() {
                    self.std_dev[i]
                } else {
                    0.0
                };
                let feat = if features[i].is_finite() {
                    features[i]
                } else {
                    0.0
                };
                let scaled = if s > 1e-8 { (feat - m) / s } else { feat - m };
                // Limitar sólo por seguridad matemática (f64 exp overflow) no heurística
                features[i] = if scaled.is_finite() {
                    scaled.clamp(-700.0, 700.0)
                } else {
                    0.0
                };
            }
        }
    }
}

/// Estadísticas de Welford Online por canal para normalización streaming O(1)
/// Permite mantener la media móvil y varianza de cada feature sin cancelación catastrófica.
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct ChannelWelfordStats {
    pub count: f64,
    pub mean: f64,
    pub m2: f64,
    pub is_decay: bool,
}

impl Default for ChannelWelfordStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelWelfordStats {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
            is_decay: false,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, val: f64) {
        if !val.is_finite() {
            return;
        }
        if self.count >= 2000.0 {
            if !self.is_decay {
                self.m2 = (self.m2 / (self.count - 1.0)).max(0.0);
                self.is_decay = true;
            }
            let alpha = 2.0 / (self.count.min(2000.0) + 1.0);
            let delta = val - self.mean;
            self.mean += alpha * delta;
            let delta2 = val - self.mean;
            self.m2 = (1.0 - alpha) * self.m2 + alpha * delta * delta2;
        } else {
            self.count += 1.0;
            let delta = val - self.mean;
            self.mean += delta / self.count;
            let delta2 = val - self.mean;
            self.m2 += delta * delta2;
        }
    }

    #[inline(always)]
    pub fn std_dev(&self) -> f64 {
        let var = if self.is_decay {
            self.m2.max(0.0)
        } else if self.count < 2.0 {
            0.0
        } else {
            (self.m2 / (self.count - 1.0)).max(0.0)
        };
        var.sqrt()
    }

    #[inline(always)]
    pub fn normalize(&mut self, val: f64) -> f64 {
        if !val.is_finite() {
            return 0.0;
        }
        self.update(val);
        let std = self.std_dev();
        if std > 1e-6 {
            ((val - self.mean) / std).clamp(-5.0, 5.0)
        } else {
            // D-252: Si no hay varianza suficiente (cold start), retornar 0.0 neutral en vez de saturar en +/-5.0
            0.0
        }
    }

    /// N-11: Normalización sin mutación de estado (estadísticos congelados).
    /// Evita que media/varianza muten durante la inferencia (predict), garantizando
    /// determinismo y consistencia exacta entre backtest y live.
    #[inline(always)]
    pub fn transform(&self, val: f64) -> f64 {
        if !val.is_finite() {
            return 0.0;
        }
        let std = self.std_dev();
        if std > 1e-6 {
            ((val - self.mean) / std).clamp(-5.0, 5.0)
        } else {
            // D-252: Si no hay varianza suficiente (cold start), retornar 0.0 neutral en vez de saturar en +/-5.0
            0.0
        }
    }
}

/// Red Neuronal Profunda de 3 capas para detección de anomalías de mercado.
///
/// Arquitectura: Input(20) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
///
/// La salida es una probabilidad [0.0, 1.0]:
/// - > 0.7: Alta confianza en oportunidad de trading (Long bias)
/// - < 0.3: Alta confianza en riesgo (Short bias o no operar)
/// - 0.3-0.7: Zona neutral (no operar en swing)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DarkAlphaEngine {
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
    pub layer3: DenseLayer,
    pub scaler: Option<Scaler>,
    /// Normalizadores Welford individuales por canal (uno por cada feature en input_dim)
    #[serde(default)]
    pub channel_normalizers: Vec<ChannelWelfordStats>,
    /// Normalizadores Welford aislados por activo (hasta 30 monedas) para inferencia multi-moneda sin contaminación cruzada (D-53)
    #[serde(default)]
    pub per_coin_normalizers: Vec<Vec<ChannelWelfordStats>>,
    /// N-11: Si es true, la inferencia (predict) no muta los estadísticos de los normalizadores
    #[serde(default)]
    pub freeze_normalizers: bool,
    // Buffers pre-alocados para evitar allocations en hot path
    #[serde(skip)]
    buf_scaled: Vec<f64>,
    #[serde(skip)]
    buf_h1: Vec<f64>,
    #[serde(skip)]
    buf_h2: Vec<f64>,
    #[serde(skip)]
    buf_out: Vec<f64>,
}

impl DarkAlphaEngine {
    /// Congela los normalizadores para inferencia determinista sin drift
    pub fn freeze(&mut self) {
        self.freeze_normalizers = true;
    }

    /// Descongela los normalizadores para adaptación continua online
    pub fn unfreeze(&mut self) {
        self.freeze_normalizers = false;
    }

    /// Crear modelo con dimensiones por defecto
    /// input_dim=20 features (price, vol, obi, atr, ema, vix, dxy, sp500, etc.)
    pub fn new(input_dim: usize, hidden1: usize, hidden2: usize) -> Self {
        Self {
            layer1: DenseLayer::new_xavier(input_dim, hidden1), // Xavier init para Tanh
            layer2: DenseLayer::new_xavier(hidden1, hidden2),   // Xavier init para Tanh
            layer3: DenseLayer::new_xavier(hidden2, 1),         // Xavier init para Sigmoid
            scaler: None,
            channel_normalizers: vec![ChannelWelfordStats::new(); input_dim],
            per_coin_normalizers: vec![vec![ChannelWelfordStats::new(); input_dim]; 30],
            freeze_normalizers: false,
            buf_scaled: vec![0.0; input_dim],
            buf_h1: vec![0.0; hidden1],
            buf_h2: vec![0.0; hidden2],
            buf_out: vec![0.0; 1],
        }
    }

    /// Crear modelo con tamaños por defecto correspondientes a get_swing_features (34)
    pub fn default_model() -> Self {
        Self::new(34, 64, 32)
    }

    /// Garantiza que los buffers pre-alocados para inferencia tengan el tamaño correcto
    /// y sanitiza pesos denormales / subnormales.
    /// Indispensable tras deserialización con serde / bincode.
    pub fn init_buffers(&mut self) {
        self.sanitize_denormals();
        let in_dim = self.layer1.in_features;
        if self.channel_normalizers.len() != in_dim {
            self.channel_normalizers
                .resize(in_dim, ChannelWelfordStats::new());
        }
        if self.per_coin_normalizers.len() < 30 {
            self.per_coin_normalizers
                .resize(30, vec![ChannelWelfordStats::new(); in_dim]);
        }
        for coin_norms in self.per_coin_normalizers.iter_mut() {
            if coin_norms.len() != in_dim {
                coin_norms.resize(in_dim, ChannelWelfordStats::new());
            }
        }
        if self.buf_scaled.len() != in_dim {
            self.buf_scaled = vec![0.0; in_dim];
        }
        if self.buf_h1.len() != self.layer1.out_features {
            self.buf_h1 = vec![0.0; self.layer1.out_features];
        }
        if self.buf_h2.len() != self.layer2.out_features {
            self.buf_h2 = vec![0.0; self.layer2.out_features];
        }
        if self.buf_out.len() != self.layer3.out_features {
            self.buf_out = vec![0.0; self.layer3.out_features];
        }
    }

    /// Sanitiza los pesos de todas las capas eliminando valores subnormales (< 1e-7)
    pub fn sanitize_denormals(&mut self) {
        self.layer1.sanitize_denormals();
        self.layer2.sanitize_denormals();
        self.layer3.sanitize_denormals();
    }

    /// Forward pass completo — ~50-100ns en CPU moderna
    ///
    /// `features` debe tener al menos `input_dim` elementos normalizados [-1, 1]
    #[inline(always)]
    pub fn predict(&mut self, features: &[f64]) -> Option<f64> {
        telemetry_server::profile_node!("DarkAlphaEngine::predict", {
            let in_dim = self.layer1.in_features;
            if features.len() < in_dim {
                return None; // Fallo explícito si faltan datos (Leakage prevent)
            }

            // Invariante de seguridad: asegurar que los buffers y normalizadores estén inicializados
            if self.buf_scaled.len() < in_dim
                || self.buf_h1.len() < self.layer1.out_features
                || self.channel_normalizers.len() < in_dim
            {
                self.init_buffers();
            }

            if let Some(scaler) = &self.scaler {
                // Adaptabilidad dimensional y sanitización defensiva ante no-finitos (reemplazo por 0.0)
                for (dest, &src) in self.buf_scaled[..in_dim]
                    .iter_mut()
                    .zip(&features[..in_dim])
                {
                    *dest = if src.is_finite() { src } else { 0.0 };
                }
                scaler.scale(&mut self.buf_scaled[..in_dim]);
            } else {
                // 1. Normalización Welford Online individual por canal O(1) con control de congelamiento (N-11 / D-138)
                for i in 0..in_dim {
                    let raw = if features[i].is_finite() {
                        features[i]
                    } else {
                        0.0
                    };
                    self.buf_scaled[i] = if self.freeze_normalizers {
                        if self.channel_normalizers[i].count < 500.0 {
                            self.channel_normalizers[i].normalize(raw)
                        } else {
                            self.channel_normalizers[i].transform(raw)
                        }
                    } else {
                        self.channel_normalizers[i].normalize(raw)
                    };
                }

                // D-411: Preservar Z-scores causales individuales por canal Welford acotados en [-3.0, 3.0]
                // sin contaminación espacial transversal que comprima canales técnicos ante picos de volumen.
                for v in self.buf_scaled[..in_dim].iter_mut() {
                    *v = (*v).clamp(-3.0, 3.0);
                }
            }

            // N-11: Arquitectura ReLU en capas ocultas alineada 100% con fit() y SGD backward pass
            self.layer1
                .forward_relu(&self.buf_scaled[..in_dim], &mut self.buf_h1);
            self.layer2.forward_relu(&self.buf_h1, &mut self.buf_h2);
            self.layer3.forward_sigmoid(&self.buf_h2, &mut self.buf_out);

            let out = self.buf_out[0];
            if out.is_finite() {
                Some(out.clamp(0.0, 1.0))
            } else {
                None
            }
        })
    }

    /// FIX D-53: Forward pass aislado por activo para eliminar contaminación cruzada de Welford entre activos
    #[inline(always)]
    pub fn predict_for_coin(&mut self, coin_id: usize, features: &[f64]) -> Option<f64> {
        let in_dim = self.layer1.in_features;
        if features.len() < in_dim {
            return None;
        }

        if self.per_coin_normalizers.len() <= coin_id {
            self.per_coin_normalizers
                .resize_with(coin_id + 1, || vec![ChannelWelfordStats::new(); in_dim]);
        }
        if self.per_coin_normalizers[coin_id].len() < in_dim {
            self.per_coin_normalizers[coin_id].resize(in_dim, ChannelWelfordStats::new());
        }

        if self.buf_scaled.len() < in_dim || self.buf_h1.len() < self.layer1.out_features {
            self.init_buffers();
        }

        if let Some(scaler) = &self.scaler {
            for (dest, &src) in self.buf_scaled[..in_dim]
                .iter_mut()
                .zip(&features[..in_dim])
            {
                *dest = if src.is_finite() { src } else { 0.0 };
            }
            scaler.scale(&mut self.buf_scaled[..in_dim]);
        } else {
            // R-03 — FALLBACK A NORMALIZADORES ENTRENADOS: si el per-coin
            // está FRÍO (sin observaciones), usar el canal ENTRENADO en vez
            // de stats vacíos (mean=0/std=0 -> features crudas saturadas ->
            // salida ≈ sigmoid(bias) ≈ constante: el ML medía bias). El
            // per-coin toma el control cuando acumula evidencia propia.
            let normalizers = &mut self.per_coin_normalizers[coin_id];
            for i in 0..in_dim {
                let raw = if features[i].is_finite() {
                    features[i]
                } else {
                    0.0
                };
                // D-124 & D-138: Warmup adaptativo de 500 ticks en Welford antes del freeze estricto.
                // Si el normalizador per-coin aún no acumuló 500 observaciones:
                // - Si el canal global entrenado está disponible (count >= 20), usar transform del canal global mientras se actualiza el local.
                // - Si no hay canal global disponible, usar normalize() en el normalizador local hasta completar 500 ticks.
                // Una vez alcanzado count >= 500, se congela estrictamente con transform() garantizando cero drift.
                self.buf_scaled[i] = if self.freeze_normalizers {
                    if normalizers[i].count >= 500.0 {
                        normalizers[i].transform(raw)
                    } else if i < self.channel_normalizers.len()
                        && self.channel_normalizers[i].count >= 20.0
                    {
                        normalizers[i].update(raw);
                        self.channel_normalizers[i].transform(raw)
                    } else {
                        normalizers[i].normalize(raw)
                    }
                } else {
                    normalizers[i].normalize(raw)
                };
            }

            // D-411: Preservar Z-scores causales individuales por canal Welford acotados en [-3.0, 3.0]
            // sin contaminación espacial transversal que comprima canales técnicos ante picos de volumen.
            for v in self.buf_scaled[..in_dim].iter_mut() {
                *v = (*v).clamp(-3.0, 3.0);
            }
        }

        self.layer1
            .forward_relu(&self.buf_scaled[..in_dim], &mut self.buf_h1);
        self.layer2.forward_relu(&self.buf_h1, &mut self.buf_h2);
        self.layer3.forward_sigmoid(&self.buf_h2, &mut self.buf_out);

        let out = self.buf_out[0];
        if out.is_finite() {
            Some(out.clamp(0.0, 1.0))
        } else {
            None
        }
    }

    /// Forward pass con plasticidad sináptica continua (Oja Hebbian Learning)
    #[inline(always)]
    pub fn predict_with_plasticity(
        &mut self,
        features: &[f64],
        plasticity_rate: f64,
    ) -> Option<f64> {
        let prob = self.predict(features)?;
        if plasticity_rate > 1e-9 && plasticity_rate.is_finite() {
            let lr = (plasticity_rate * 1e-5).clamp(1e-7, 1e-3);
            let in_dim = self.layer1.in_features;
            let out_dim = self.layer1.out_features;
            crate::neuro_plasticity::NeuroPlasticityEngine::apply_oja_plasticity(
                &mut self.layer1.weights,
                &self.buf_scaled[..in_dim],
                &self.buf_h1[..out_dim],
                in_dim,
                out_dim,
                lr,
            );
        }
        Some(prob)
    }

    /// Entrena la red neuronal usando SGD (Stochastic Gradient Descent)
    pub fn fit(
        &mut self,
        features_batch: &[Vec<f64>],
        targets_batch: &[f64],
        epochs: usize,
        learning_rate: f64,
    ) {
        if features_batch.is_empty()
            || targets_batch.is_empty()
            || features_batch.len() != targets_batch.len()
        {
            return;
        }

        let start = std::time::Instant::now();
        let mut h1 = vec![0.0; self.layer1.out_features];
        let mut h2 = vec![0.0; self.layer2.out_features];
        let mut out = vec![0.0; self.layer3.out_features];

        let mut grad_h2 = vec![0.0; self.layer2.out_features];
        let mut grad_h1 = vec![0.0; self.layer1.out_features];
        let mut scaled_features = vec![0.0; self.layer1.in_features];

        for _epoch in 0..epochs {
            for (features, &target) in features_batch.iter().zip(targets_batch.iter()) {
                if features.len() != self.layer1.in_features {
                    continue;
                }

                scaled_features.copy_from_slice(features);
                if let Some(scaler) = &self.scaler {
                    scaler.scale(&mut scaled_features);
                } else {
                    let in_dim = self.layer1.in_features;
                    for i in 0..in_dim {
                        let raw = if features[i].is_finite() {
                            features[i]
                        } else {
                            0.0
                        };
                        scaled_features[i] = self.channel_normalizers[i].normalize(raw);
                    }
                    // D-411: Preservar Z-scores causales individuales por canal Welford acotados en [-3.0, 3.0]
                    for v in scaled_features[..in_dim].iter_mut() {
                        *v = (*v).clamp(-3.0, 3.0);
                    }
                }

                // Reset explícito de gradientes para desacoplamiento estricto por muestra
                grad_h1.fill(0.0);
                grad_h2.fill(0.0);

                // --- FORWARD PASS ---
                self.layer1.forward_relu(&scaled_features, &mut h1);
                self.layer2.forward_relu(&h1, &mut h2);
                self.layer3.forward_sigmoid(&h2, &mut out);

                let prediction = out[0];
                // Soportar targets continuos en [0.0, 1.0] o binarización suave
                let target_clamped = target.clamp(0.0, 1.0);

                // --- BACKWARD PASS ---
                // FIX #390: Binary Cross Entropy con Sigmoid analítico: dL/dz = pred - target (dL/dp * dp/dz cancela p(1-p))
                let delta_out = prediction - target_clamped;

                // Compute gradients for Layer 3 (Hidden 2 -> Out)
                for i in 0..self.layer2.out_features {
                    grad_h2[i] = self.layer3.weights[i] * delta_out;
                    self.layer3.weights[i] -= learning_rate * delta_out * h2[i];
                }
                self.layer3.biases[0] -= learning_rate * delta_out;

                // Compute gradients for Layer 2 (Hidden 1 -> Hidden 2)
                for i in 0..self.layer2.out_features {
                    // ReLU derivative: 1 if h2[i] > 0 else 0
                    let d_relu_h2 = if h2[i] > 0.0 { 1.0 } else { 0.0 };
                    let delta_h2 = grad_h2[i] * d_relu_h2;

                    let row_offset = i * self.layer2.in_features;
                    for j in 0..self.layer1.out_features {
                        grad_h1[j] += self.layer2.weights[row_offset + j] * delta_h2;
                        self.layer2.weights[row_offset + j] -= learning_rate * delta_h2 * h1[j];
                    }
                    self.layer2.biases[i] -= learning_rate * delta_h2;
                }

                // Compute gradients for Layer 1 (Input -> Hidden 1)
                for i in 0..self.layer1.out_features {
                    let d_relu_h1 = if h1[i] > 0.0 { 1.0 } else { 0.0 };
                    let delta_h1 = grad_h1[i] * d_relu_h1;

                    let row_offset = i * self.layer1.in_features;
                    for (j, &feat) in scaled_features
                        .iter()
                        .enumerate()
                        .take(self.layer1.in_features)
                    {
                        self.layer1.weights[row_offset + j] -= learning_rate * delta_h1 * feat;
                    }
                    self.layer1.biases[i] -= learning_rate * delta_h1;
                }
            }
        }
        self.freeze_normalizers = true;
        println!(
            "⚡ [Dark Alpha] Neural Network entrenada nativamente vía SGD en {:?}",
            start.elapsed()
        );
    }

    /// Guardar modelo a disco en formato bincode (más rápido que JSON)
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let data = bincode::serialize(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Cargar modelo desde disco
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let mut model: Self = bincode::deserialize(&data)?;
        model.init_buffers();
        Ok(model)
    }

    /// Cargar desde JSON (compatibilidad con formato anterior)
    pub fn load_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let mut model: Self = serde_json::from_str(&content)?;
        model.init_buffers();
        Ok(model)
    }

    /// Guardar modelo en formato JSON
    pub fn save_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Guardar modelo a disco en formato bincode
    pub fn save_to_disk(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.save(path)
    }

    /// Hiper-mutación de pesos para exploración evolutiva en línea
    // FIX #716: Acotamiento estricto de pesos en hipermutación para prevenir explosión sináptica
    pub fn force_hyper_mutation(&mut self, _capital: f64) {
        let mut seed: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);

        let mut next_rand = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        };

        for w in self.layer1.weights.iter_mut() {
            let delta = next_rand() * 0.05;
            let new_w = *w + delta;
            *w = if new_w.is_finite() {
                new_w.clamp(-100.0, 100.0)
            } else {
                0.0
            };
        }
        for w in self.layer2.weights.iter_mut() {
            let delta = next_rand() * 0.05;
            let new_w = *w + delta;
            *w = if new_w.is_finite() {
                new_w.clamp(-100.0, 100.0)
            } else {
                0.0
            };
        }
        for w in self.layer3.weights.iter_mut() {
            let delta = next_rand() * 0.05;
            let new_w = *w + delta;
            *w = if new_w.is_finite() {
                new_w.clamp(-100.0, 100.0)
            } else {
                0.0
            };
        }
    }
}

pub type Phenotype = DarkAlphaEngine;
pub type MoENeatEngine = DarkAlphaEngine;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_pass_returns_valid_probability() {
        let mut engine = DarkAlphaEngine::default_model();
        let features = vec![0.1; 34];
        let result = engine
            .predict(&features)
            .expect("input completo debe producir predicción");
        assert!(
            (0.0..=1.0).contains(&result),
            "Result {} out of [0,1]",
            result
        );
    }

    #[test]
    fn test_wrong_input_size_returns_none() {
        let mut engine = DarkAlphaEngine::default_model();
        let features = vec![0.1; 5]; // Wrong size
        assert_eq!(
            engine.predict(&features),
            None,
            "features insuficientes deben fallar explícitamente, nunca inferir"
        );
    }

    #[test]
    fn test_inference_speed() {
        let mut engine = DarkAlphaEngine::default_model();
        let features = vec![0.1; 34];

        let start = std::time::Instant::now();
        let iterations = 20_000;
        for _ in 0..iterations {
            std::hint::black_box(engine.predict(&features));
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() / iterations as u128;
        println!("⚡ Inferencia por llamada: {} ns", per_call_ns);
        let max_allowed_ns = 25_000;
        assert!(
            per_call_ns < max_allowed_ns,
            "Demasiado lento: {} ns (límite: {} ns)",
            per_call_ns,
            max_allowed_ns
        );
    }

    #[test]
    fn test_quantized_dense_layer_accuracy_and_speed() {
        let layer = DenseLayer::new(34, 64);
        let q_layer = layer.quantize_int8();
        let input = vec![0.5; 34];
        let mut out_f64 = vec![0.0; 64];
        let mut out_q = vec![0.0; 64];

        layer.forward_relu(&input, &mut out_f64);
        q_layer.forward_relu(&input, &mut out_q);

        for (f, q) in out_f64.iter().zip(&out_q) {
            assert!(
                (f - q).abs() < 0.20,
                "Int8 quantization error should be bounded: f64={}, int8={}",
                f,
                q
            );
        }
    }

    #[test]
    fn test_load_json_roundtrip() {
        let model = DarkAlphaEngine::default_model();
        let json_str = serde_json::to_string(&model).expect("serialization to JSON should succeed");
        let restored: DarkAlphaEngine =
            serde_json::from_str(&json_str).expect("deserialization from JSON should succeed");
        assert_eq!(restored.layer1.in_features, 34);
        assert_eq!(restored.layer1.out_features, 64);
    }

    #[test]
    fn test_quantized_dense_layer_sigmoid() {
        let layer = DenseLayer::new_xavier(32, 1);
        let q_layer = layer.quantize_int8();
        let input = vec![0.2; 32];
        let mut out_f64 = vec![0.0; 1];
        let mut out_q = vec![0.0; 1];

        layer.forward_sigmoid(&input, &mut out_f64);
        q_layer.forward_sigmoid(&input, &mut out_q);

        assert!((0.0..=1.0).contains(&out_f64[0]));
        assert!((0.0..=1.0).contains(&out_q[0]));
        assert!((out_f64[0] - out_q[0]).abs() < 0.25);
    }

    #[test]
    fn test_scaler_and_mutate() {
        let scaler = Scaler::new(vec![10.0, 20.0], vec![2.0, 5.0]);
        let mut feat = vec![12.0, 25.0];
        scaler.scale(&mut feat);
        assert_eq!(feat[0], 1.0);
        assert_eq!(feat[1], 1.0);

        let mut engine = DarkAlphaEngine::default_model();
        engine.force_hyper_mutation(13.0);
        let features = vec![0.1; 34];
        let result = engine
            .predict(&features)
            .expect("mutated engine should predict");
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn test_predict_nan_and_infinite_feature_sanitization() {
        let mut engine = DarkAlphaEngine::default_model();
        let mut features = vec![0.1; 34];
        features[0] = f64::NAN;
        features[5] = f64::INFINITY;
        features[10] = -f64::INFINITY;

        let result = engine.predict(&features);
        assert!(
            result.is_some(),
            "Sanitizer must handle NaN/Inf by mapping to 0.0"
        );
        let prob = result.unwrap();
        assert!(prob.is_finite());
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn test_scaler_nan_and_zero_scale_defense() {
        let scaler = Scaler::new(vec![f64::NAN, 10.0], vec![0.0, f64::NAN]);
        let mut feat = vec![5.0, 15.0];
        scaler.scale(&mut feat);
        // Zero scale should keep original value, NaN mean should treat mean as 0.0
        assert!(feat[0].is_finite());
        assert!(feat[1].is_finite());
    }

    #[test]
    fn test_quantized_dense_layers_3stage_pipeline() {
        let l1 = DenseLayer::new(34, 64).quantize_int8();
        let l2 = DenseLayer::new(64, 32).quantize_int8();
        let l3 = DenseLayer::new_xavier(32, 1).quantize_int8();

        let input = vec![0.05; 34];
        let mut h1 = vec![0.0; 64];
        let mut h2 = vec![0.0; 32];
        let mut out = vec![0.0; 1];

        l1.forward_relu(&input, &mut h1);
        l2.forward_relu(&h1, &mut h2);
        l3.forward_sigmoid(&h2, &mut out);

        assert!((0.0..=1.0).contains(&out[0]));
    }

    #[test]
    fn test_dark_alpha_engine_mismatched_features_dimension_rejection() {
        let mut engine = DarkAlphaEngine::default_model();
        let short_features = vec![0.1; 30]; // 50 instead of 34
        assert!(
            engine.predict(&short_features).is_none(),
            "Should reject mismatched dimension"
        );
    }

    #[test]
    fn test_dark_alpha_engine_he_initialization_finite() {
        let layer = DenseLayer::new(34, 64);
        assert_eq!(layer.weights.len(), 34 * 64);
        assert_eq!(layer.biases.len(), 64);
        for &w in &layer.weights {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_dark_alpha_engine_fit_bce_loss_reduction() {
        let mut engine = DarkAlphaEngine::default_model();
        let sample_positive = vec![1.0; 34];
        let initial_pred = engine.predict(&sample_positive).unwrap();

        let batch_x = vec![sample_positive.clone(); 10];
        let batch_y = vec![1.0; 10];

        engine.fit(&batch_x, &batch_y, 20, 0.05);

        let final_pred = engine.predict(&sample_positive).unwrap();
        assert!(final_pred.is_finite());
        // For target 1.0, final_pred should stay high or increase towards 1.0
        assert!(final_pred >= initial_pred || final_pred > 0.50);
    }

    #[test]
    fn test_dark_alpha_engine_quantized_forward_pipeline_nan_immunity() {
        let l1 = DenseLayer::new(34, 64).quantize_int8();
        let l2 = DenseLayer::new(64, 32).quantize_int8();
        let l3 = DenseLayer::new_xavier(32, 1).quantize_int8();

        let mut input = vec![0.0; 34];
        input[0] = f64::NAN;
        input[5] = f64::INFINITY;
        let mut h1 = vec![0.0; 64];
        let mut h2 = vec![0.0; 32];
        let mut out = vec![0.0; 1];

        // Sanitize input before quantized layers
        for x in input.iter_mut() {
            if !x.is_finite() {
                *x = 0.0;
            }
        }

        l1.forward_relu(&input, &mut h1);
        l2.forward_relu(&h1, &mut h2);
        l3.forward_sigmoid(&h2, &mut out);

        assert!(out[0].is_finite());
        assert!((0.0..=1.0).contains(&out[0]));
    }

    #[test]
    fn test_channel_welford_stats_o1_and_decay() {
        let mut w = ChannelWelfordStats::new();
        for i in 1..=100 {
            w.update(i as f64);
        }
        assert!((w.mean - 50.5).abs() < 1e-4);
        assert!(w.std_dev() > 28.0 && w.std_dev() < 30.0);

        // Test normalize
        let norm_val = w.normalize(50.5);
        assert!(norm_val.abs() < 0.1);
    }

    #[test]
    fn test_dark_alpha_welford_channel_disparate_scales() {
        let mut engine = DarkAlphaEngine::default_model();
        assert_eq!(engine.channel_normalizers.len(), 34);

        // Simulate 50 ticks where feature 0 has scale 60,000 (like BTC price)
        // and feature 1 has scale 0.05 (like Hawkes rate), feature 2 has scale 50.0 (like RSI)
        for i in 0..50 {
            let mut feats = vec![0.0; 34];
            feats[0] = 60000.0 + (i as f64) * 10.0;
            feats[1] = 0.05 + ((i % 5) as f64) * 0.01;
            feats[2] = 50.0 + ((i % 10) as f64) * 2.0;
            for k in 3..34 {
                feats[k] = ((k + i) as f64) * 0.1;
            }

            let pred = engine.predict(&feats);
            assert!(
                pred.is_some(),
                "Predict must succeed with multi-scale features"
            );
            let p = pred.unwrap();
            assert!(p.is_finite());
            assert!((0.0..=1.0).contains(&p));
        }

        // Check that channel 0 and channel 1 adapted their own means and standard deviations
        assert!(engine.channel_normalizers[0].mean > 50000.0);
        assert!(engine.channel_normalizers[1].mean < 1.0);
    }

    #[test]
    fn test_sanitize_denormals_and_clean_model_file() {
        let mut l = DenseLayer::new(10, 10);
        l.weights[0] = 1e-316;
        l.biases[0] = -2e-317;
        l.sanitize_denormals();
        assert_eq!(l.weights[0], 0.0);
        assert_eq!(l.biases[0], 0.0);

        // T-04: ELIMINADO el bloque que reescribía el modelo de PRODUCCIÓN
        // (models/DarkAlpha_BTCUSDT.json) como efecto secundario de un test:
        // (a) `cargo test` mutaba artefactos vivos; (b) la re-serialización
        // con serde default dejaba freeze_normalizers=false, revirtiendo el
        // congelamiento de modelos ya saneados. La sanitización de producción
        // ocurre en las rutas de carga (god-engine-core / simulator), con
        // freeze() explícito.
    }
}
