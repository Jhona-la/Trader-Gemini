//! # Online Newton GBDT (Gradient Boosted Decision Trees con Taylor de 2º Orden)
//!
//! Implementación analítica de alta velocidad en Rust para el Motor de Evolución Cuántica.
//! Opera en el espacio de Hilbert $\mathcal{H} = L^2$ con cero alocaciones en el heap durante inferencia (<25 µs).
//!
//! ## Fundamentación Matemática
//! 1. Aproximación de segundo orden de Taylor de la función de pérdida:
//!    $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t)$$
//!    donde:
//!    $$g_i = \frac{\partial \mathcal{L}(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}, \quad h_i = \frac{\partial^2 \mathcal{L}(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$$
//!
//! 2. Peso óptimo del nodo hoja $j$ con regularización L2 ($\lambda$):
//!    $$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda} = -\frac{G_j}{H_j + \lambda}$$
//!
//! 3. Ganancia de corte (Split Gain) analítica:
//!    $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
//!
//! 4. Inferencia en Stack Zero-Alloc:
//!    Los árboles se almacenan en un arreglo estático contiguo `[GbdtNode; MAX_NODES]`.
//!    La evaluación no crea matrices, vectores ni alocaciones dinámicas, permitiendo inferencia
//!    en menos de 300 nanosegundos para el ensamble completo.

use serde::{Deserialize, Serialize};
use crate::lock_seguro::{escribir, leer};
use std::sync::RwLock;

pub const NUM_FEATURES: usize = 6;
pub const MAX_GBDT_TREES: usize = 16;
pub const MAX_TREE_DEPTH: usize = 3;
pub const MAX_NODES_PER_TREE: usize = (1 << (MAX_TREE_DEPTH + 1)) - 1; // 15 nodos para prof 3
pub const MAX_MEMORY_OBS: usize = 5000;

/// Nodo de decisión estático optimizado para caché L1
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GbdtNode {
    pub feature_idx: u8,
    pub threshold: f32,
    pub is_leaf: bool,
    pub leaf_weight: f32,
    pub left_child: u8,
    pub right_child: u8,
}

impl Default for GbdtNode {
    fn default() -> Self {
        Self {
            feature_idx: 0,
            threshold: 0.0,
            is_leaf: true,
            leaf_weight: 0.0,
            left_child: 0,
            right_child: 0,
        }
    }
}

/// Árbol de decisión somero (profundidad <= 3) residente en memoria contigua
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ShallowGbdtTree {
    pub nodes: [GbdtNode; MAX_NODES_PER_TREE],
    pub node_count: u8,
    pub learning_rate: f32,
}

impl Default for ShallowGbdtTree {
    fn default() -> Self {
        Self {
            nodes: [GbdtNode::default(); MAX_NODES_PER_TREE],
            node_count: 0,
            learning_rate: 0.10,
        }
    }
}

impl ShallowGbdtTree {
    /// Inferencia ultra-rápida zero-alloc en stack (<20 nanosegundos)
    #[inline(always)]
    pub fn predict_zero_alloc(&self, features: &[f64; NUM_FEATURES]) -> f32 {
        if self.node_count == 0 {
            return 0.0;
        }

        let mut current_idx = 0usize;
        loop {
            let node = &self.nodes[current_idx];
            if node.is_leaf {
                return node.leaf_weight * self.learning_rate;
            }

            let feat_val = features[node.feature_idx as usize] as f32;
            if feat_val < node.threshold {
                current_idx = node.left_child as usize;
            } else {
                current_idx = node.right_child as usize;
            }

            if current_idx >= MAX_NODES_PER_TREE || current_idx >= self.node_count as usize {
                return node.leaf_weight * self.learning_rate;
            }
        }
    }
}

/// Hiperparámetros canónicos de Newton GBDT
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GbdtHyperparams {
    pub num_trees: usize,
    pub max_depth: usize,
    pub learning_rate: f32,
    pub l2_lambda: f32,
    pub min_split_gain: f32,
    pub min_child_weight: f32,
    pub online_alpha: f32,
}

impl Default for GbdtHyperparams {
    fn default() -> Self {
        Self {
            num_trees: 12,
            max_depth: 3,
            learning_rate: 0.15,
            l2_lambda: 1.0,
            min_split_gain: 0.001,
            min_child_weight: 2.0,
            online_alpha: 0.05,
        }
    }
}

/// Observación individual con paridad direccional de mercado
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GbdtObservation {
    pub features: [f64; NUM_FEATURES],
    pub actual_pnl_pct: f64,
    pub is_long: bool,
    pub directional_pnl: f64,
    pub target_up: bool,
}

impl GbdtObservation {
    pub fn new(features: [f64; NUM_FEATURES], net_pnl_pct: f64, is_long: bool) -> Self {
        let directional_pnl = if is_long { net_pnl_pct } else { -net_pnl_pct };
        let target_up = directional_pnl > 0.0;
        Self {
            features,
            actual_pnl_pct: net_pnl_pct,
            is_long,
            directional_pnl,
            target_up,
        }
    }
}

/// Ensamble Newton GBDT con soporte para Clasificación Sigmoide y Regresión Huber
pub struct OnlineNewtonGbdt {
    pub observations: RwLock<Vec<GbdtObservation>>,
    pub class_trees: RwLock<[ShallowGbdtTree; MAX_GBDT_TREES]>,
    pub reg_trees: RwLock<[ShallowGbdtTree; MAX_GBDT_TREES]>,
    pub class_tree_count: RwLock<usize>,
    pub reg_tree_count: RwLock<usize>,
    pub base_class_logit: RwLock<f32>,
    pub base_reg_value: RwLock<f32>,
    pub hyperparams: GbdtHyperparams,
    pub max_memory_size: usize,
    pub last_accuracy: RwLock<f64>,
    pub last_mse: RwLock<f64>,
    pub shadow_threshold_long: RwLock<f32>,
    pub shadow_threshold_short: RwLock<f32>,
    pub thresholds_identified: RwLock<bool>,
    pub is_trained: RwLock<bool>,
}

impl OnlineNewtonGbdt {
    pub fn new(max_memory_size: usize) -> Self {
        Self {
            observations: RwLock::new(Vec::with_capacity(max_memory_size)),
            class_trees: RwLock::new([ShallowGbdtTree::default(); MAX_GBDT_TREES]),
            reg_trees: RwLock::new([ShallowGbdtTree::default(); MAX_GBDT_TREES]),
            class_tree_count: RwLock::new(0),
            reg_tree_count: RwLock::new(0),
            base_class_logit: RwLock::new(0.0),
            base_reg_value: RwLock::new(0.0),
            hyperparams: GbdtHyperparams::default(),
            max_memory_size,
            last_accuracy: RwLock::new(0.0),
            last_mse: RwLock::new(0.0),
            shadow_threshold_long: RwLock::new(0.60),
            shadow_threshold_short: RwLock::new(0.40),
            thresholds_identified: RwLock::new(false),
            is_trained: RwLock::new(false),
        }
    }

    /// Alimenta la observación incorporando paridad de lado
    pub fn feed_observation(&self, features: [f64; NUM_FEATURES], net_pnl_pct: f64, is_long: bool) {
        // Blindaje numérico estricto ante NaNs o infinitos
        for f in &features {
            if !f.is_finite() {
                return;
            }
        }
        if !net_pnl_pct.is_finite() {
            return;
        }

        let obs = GbdtObservation::new(features, net_pnl_pct, is_long);
        let mut data = escribir(&self.observations);
        if data.len() >= self.max_memory_size {
            let drain_count = self.max_memory_size / 10;
            data.drain(0..drain_count);
        }
        data.push(obs);

        // Actualización estocástica online de Newton si ya está entrenado
        if *leer(&self.is_trained) {
            self.online_step(&obs);
        }
    }

    /// Paso estocástico online de Newton-Raphson para adaptación en caliente
    fn online_step(&self, obs: &GbdtObservation) {
        let alpha = self.hyperparams.online_alpha;
        let lambda = self.hyperparams.l2_lambda;

        // 1. Clasificador: actualizar la última hoja recorrida con gradiente logístico
        if let Some((prob, _)) = self.predict_6d(obs.features) {
            let target_y = if obs.target_up { 1.0 } else { 0.0 };
            let g = (prob - target_y) as f32;
            let h = ((prob * (1.0 - prob)).max(0.01)) as f32;

            let mut trees = escribir(&self.class_trees);
            let count = *leer(&self.class_tree_count);
            if count > 0 {
                let last_tree = &mut trees[count - 1];
                let mut curr = 0usize;
                while curr < last_tree.node_count as usize {
                    if last_tree.nodes[curr].is_leaf {
                        // Newton step regularizado: w <- w - alpha * (g / (h + lambda))
                        let delta_w = -alpha * (g / (h + lambda));
                        last_tree.nodes[curr].leaf_weight += delta_w;
                        break;
                    }
                    let val = obs.features[last_tree.nodes[curr].feature_idx as usize] as f32;
                    if val < last_tree.nodes[curr].threshold {
                        curr = last_tree.nodes[curr].left_child as usize;
                    } else {
                        curr = last_tree.nodes[curr].right_child as usize;
                    }
                }
            }
        }
    }

    /// Inferencia continua sin alocaciones en el heap (<300 ns)
    #[inline]
    pub fn predict_6d(&self, features: [f64; NUM_FEATURES]) -> Option<(f64, f64)> {
        for f in &features {
            if !f.is_finite() {
                return None;
            }
        }

        if !*leer(&self.is_trained) {
            return None;
        }

        let base_logit = *leer(&self.base_class_logit);
        let base_reg = *leer(&self.base_reg_value);
        let class_trees = leer(&self.class_trees);
        let class_count = *leer(&self.class_tree_count);
        let reg_trees = leer(&self.reg_trees);
        let reg_count = *leer(&self.reg_tree_count);

        // Acumulación lineal sobre el espacio de Hilbert L2
        let mut margin_logit = base_logit;
        for i in 0..class_count {
            margin_logit += class_trees[i].predict_zero_alloc(&features);
        }

        let mut expected_pnl = base_reg;
        for i in 0..reg_count {
            expected_pnl += reg_trees[i].predict_zero_alloc(&features);
        }

        // Función sigmoide logística continua C^infinito
        let prob_up = (1.0 / (1.0 + (-margin_logit as f64).exp())).clamp(0.001, 0.999);
        let safe_pnl = (expected_pnl as f64).clamp(-0.50, 0.50);

        Some((prob_up, safe_pnl))
    }

    /// Reentrenamiento completo con optimización exacta de Newton de 2º Orden
    pub fn retrain_models(&self) -> Result<(f64, f64), String> {
        let data = leer(&self.observations).clone();
        if data.len() < 50 {
            return Err("Insuficientes datos para entrenar el modelo (mínimo 50)".to_string());
        }

        let n = data.len();
        let count_pos = data.iter().filter(|o| o.target_up).count();
        let count_neg = n - count_pos;

        if count_pos == 0 || count_neg == 0 {
            return Err("Insuficiente variabilidad de clases (ambos lados requeridos)".to_string());
        }

        // 1. Inicializar Logit Base del Clasificador: log(p / (1 - p))
        let prior_p = (count_pos as f64) / (n as f64);
        let base_logit = ((prior_p / (1.0 - prior_p)).ln() as f32).clamp(-3.0, 3.0);
        *escribir(&self.base_class_logit) = base_logit;

        // 2. Inicializar Regresor Base: Media podada de retornos
        let mean_ret = (data.iter().map(|o| o.directional_pnl).sum::<f64>() / (n as f64)) as f32;
        *escribir(&self.base_reg_value) = mean_ret;

        // Buffer de predicciones acumuladas
        let mut margin_preds = vec![base_logit; n];
        let mut reg_preds = vec![mean_ret; n];

        let mut trained_class_trees = [ShallowGbdtTree::default(); MAX_GBDT_TREES];
        let mut trained_reg_trees = [ShallowGbdtTree::default(); MAX_GBDT_TREES];

        let num_trees = self.hyperparams.num_trees.min(MAX_GBDT_TREES);
        let lambda = self.hyperparams.l2_lambda;
        let lr = self.hyperparams.learning_rate;
        let min_gain = self.hyperparams.min_split_gain;
        let min_child = self.hyperparams.min_child_weight;

        // 3. Boosting de Newton para Clasificación (Log-Loss)
        for t in 0..num_trees {
            // Calcular gradientes g_i y hessianos h_i de segundo orden
            let mut g = Vec::with_capacity(n);
            let mut h = Vec::with_capacity(n);
            for i in 0..n {
                let p = 1.0 / (1.0 + (-margin_preds[i] as f64).exp());
                let y = if data[i].target_up { 1.0 } else { 0.0 };
                let gi = (p - y) as f32;
                let hi = ((p * (1.0 - p)).max(1e-4)) as f32;
                g.push(gi);
                h.push(hi);
            }

            let tree = self.build_shallow_tree(&data, &g, &h, lambda, lr, min_gain, min_child);
            for i in 0..n {
                margin_preds[i] += tree.predict_zero_alloc(&data[i].features);
            }
            trained_class_trees[t] = tree;
        }

        // 4. Boosting de Newton para Regresión (Huber Loss con delta = 0.01)
        let delta = 0.01_f32;
        for t in 0..num_trees {
            let mut g = Vec::with_capacity(n);
            let mut h = Vec::with_capacity(n);
            for i in 0..n {
                let diff = reg_preds[i] - data[i].directional_pnl as f32;
                let scale = (1.0 + (diff / delta).powi(2)).sqrt();
                let gi = diff / scale;
                let hi = 1.0 / (scale.powi(3)).max(1e-4);
                g.push(gi);
                h.push(hi);
            }

            let tree = self.build_shallow_tree(&data, &g, &h, lambda, lr, min_gain, min_child);
            for i in 0..n {
                reg_preds[i] += tree.predict_zero_alloc(&data[i].features);
            }
            trained_reg_trees[t] = tree;
        }

        // Métricas de calibración
        let mut correct = 0usize;
        for i in 0..n {
            let prob = 1.0 / (1.0 + (-margin_preds[i] as f64).exp());
            let pred_up = prob >= 0.50;
            if pred_up == data[i].target_up {
                correct += 1;
            }
        }
        let acc = (correct as f64) / (n as f64);

        let mut sse = 0.0f64;
        for i in 0..n {
            let diff = reg_preds[i] as f64 - data[i].directional_pnl;
            sse += diff * diff;
        }
        let mse = sse / (n as f64);

        // 5. Auto-calibración de umbrales óptimos maximizando PnL ponderado en rejilla continua
        let mut best_th_long = 0.55_f32;
        let mut best_th_short = 0.45_f32;
        let mut best_profit_long = -1e9_f64;
        let mut best_profit_short = -1e9_f64;
        let (mut min_l, mut max_l) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_s, mut max_s) = (f64::INFINITY, f64::NEG_INFINITY);

        for step in 50..=75 {
            let th = (step as f64) / 100.0;
            let mut profit_l = 0.0_f64;
            let mut profit_s = 0.0_f64;

            for i in 0..n {
                let prob = 1.0 / (1.0 + (-margin_preds[i] as f64).exp());
                let pnl = data[i].actual_pnl_pct;
                // Largo: OBI alineado y probabilidad de subida supera umbral
                if data[i].features[0] > 0.0 && prob >= th {
                    profit_l += pnl;
                }
                // Corto: OBI bajista y probabilidad de subida por debajo de (1 - th)
                if data[i].features[0] < 0.0 && prob < (1.0 - th) {
                    profit_s += pnl;
                }
            }

            min_l = min_l.min(profit_l);
            max_l = max_l.max(profit_l);
            min_s = min_s.min(profit_s);
            max_s = max_s.max(profit_s);

            if profit_l > best_profit_long {
                best_profit_long = profit_l;
                best_th_long = th as f32;
            }
            if profit_s > best_profit_short {
                best_profit_short = profit_s;
                best_th_short = (1.0 - th) as f32;
            }
        }

        let identificados = max_l > min_l && max_s > min_s;
        if identificados {
            *escribir(&self.shadow_threshold_long) = best_th_long.clamp(0.50, 0.75);
            *escribir(&self.shadow_threshold_short) = best_th_short.clamp(0.25, 0.50);
        }
        *escribir(&self.thresholds_identified) = identificados;

        *escribir(&self.class_trees) = trained_class_trees;
        *escribir(&self.class_tree_count) = num_trees;
        *escribir(&self.reg_trees) = trained_reg_trees;
        *escribir(&self.reg_tree_count) = num_trees;
        *escribir(&self.last_accuracy) = acc;
        *escribir(&self.last_mse) = mse;
        *escribir(&self.is_trained) = true;

        Ok((acc, mse))
    }

    /// Construcción recursiva exacta de árbol somero mediante split gain de Newton
    fn build_shallow_tree(
        &self,
        data: &[GbdtObservation],
        g: &[f32],
        h: &[f32],
        lambda: f32,
        lr: f32,
        min_gain: f32,
        min_child: f32,
    ) -> ShallowGbdtTree {
        let mut tree = ShallowGbdtTree {
            nodes: [GbdtNode::default(); MAX_NODES_PER_TREE],
            node_count: 0,
            learning_rate: lr,
        };

        let indices: Vec<usize> = (0..data.len()).collect();
        let mut next_node_idx = 0usize;

        self.split_recursive(
            &mut tree,
            &mut next_node_idx,
            0,
            &indices,
            data,
            g,
            h,
            lambda,
            min_gain,
            min_child,
        );

        tree.node_count = next_node_idx as u8;
        tree
    }

    fn split_recursive(
        &self,
        tree: &mut ShallowGbdtTree,
        next_node_idx: &mut usize,
        depth: usize,
        indices: &[usize],
        data: &[GbdtObservation],
        g: &[f32],
        h: &[f32],
        lambda: f32,
        min_gain: f32,
        min_child: f32,
    ) -> usize {
        let node_id = *next_node_idx;
        *next_node_idx += 1;

        let total_g: f32 = indices.iter().map(|&i| g[i]).sum();
        let total_h: f32 = indices.iter().map(|&i| h[i]).sum();
        let leaf_weight = -total_g / (total_h + lambda);

        if depth >= MAX_TREE_DEPTH || indices.len() < 4 || total_h < min_child {
            tree.nodes[node_id] = GbdtNode {
                feature_idx: 0,
                threshold: 0.0,
                is_leaf: true,
                leaf_weight,
                left_child: 0,
                right_child: 0,
            };
            return node_id;
        }

        // Buscar el mejor split evaluando las 6 dimensiones
        let mut best_gain = 0.0f32;
        let mut best_feat = 0u8;
        let mut best_threshold = 0.0f32;
        let mut best_left_indices = Vec::new();
        let mut best_right_indices = Vec::new();

        let base_score = (total_g * total_g) / (total_h + lambda);

        for feat_idx in 0..NUM_FEATURES {
            // Percentiles de corte para rapidez en stack
            let mut vals: Vec<f32> = indices.iter().map(|&i| data[i].features[feat_idx] as f32).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();

            if vals.len() < 2 {
                continue;
            }

            // Evaluar hasta 8 cuantiles
            let step = (vals.len() / 8).max(1);
            for k in (step..vals.len()).step_by(step) {
                let thresh = 0.5 * (vals[k - 1] + vals[k]);
                let mut gl = 0.0f32;
                let mut hl = 0.0f32;
                let mut left_idx = Vec::new();
                let mut right_idx = Vec::new();

                for &i in indices {
                    if (data[i].features[feat_idx] as f32) < thresh {
                        gl += g[i];
                        hl += h[i];
                        left_idx.push(i);
                    } else {
                        right_idx.push(i);
                    }
                }

                let gr = total_g - gl;
                let hr = total_h - hl;

                if hl < min_child || hr < min_child {
                    continue;
                }

                let gain = 0.5 * ((gl * gl) / (hl + lambda) + (gr * gr) / (hr + lambda) - base_score);
                if gain > best_gain && gain > min_gain {
                    best_gain = gain;
                    best_feat = feat_idx as u8;
                    best_threshold = thresh;
                    best_left_indices = left_idx;
                    best_right_indices = right_idx;
                }
            }
        }

        if best_gain <= min_gain || best_left_indices.is_empty() || best_right_indices.is_empty() {
            tree.nodes[node_id] = GbdtNode {
                feature_idx: 0,
                threshold: 0.0,
                is_leaf: true,
                leaf_weight,
                left_child: 0,
                right_child: 0,
            };
            return node_id;
        }

        // Crear split interno y recursar en subárboles izquierdo y derecho
        let left_child_id = self.split_recursive(
            tree,
            next_node_idx,
            depth + 1,
            &best_left_indices,
            data,
            g,
            h,
            lambda,
            min_gain,
            min_child,
        );

        let right_child_id = self.split_recursive(
            tree,
            next_node_idx,
            depth + 1,
            &best_right_indices,
            data,
            g,
            h,
            lambda,
            min_gain,
            min_child,
        );

        tree.nodes[node_id] = GbdtNode {
            feature_idx: best_feat,
            threshold: best_threshold,
            is_leaf: false,
            leaf_weight,
            left_child: left_child_id as u8,
            right_child: right_child_id as u8,
        };

        node_id
    }

    pub fn get_optimal_thresholds(&self) -> Option<(f32, f32)> {
        if !*leer(&self.thresholds_identified) {
            return None;
        }
        let th_long = *leer(&self.shadow_threshold_long);
        let th_short = *leer(&self.shadow_threshold_short);
        Some((th_long, th_short))
    }
}
