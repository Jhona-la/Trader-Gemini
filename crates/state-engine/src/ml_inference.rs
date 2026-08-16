use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use arc_swap::ArcSwap;
use std::collections::HashMap;

lazy_static::lazy_static! {
    pub static ref GLOBAL_FORESTS: ArcSwap<HashMap<String, Arc<NanoForest>>> = ArcSwap::from_pointee(HashMap::new());
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NanoForestData {
    pub children_left: Vec<i32>,
    pub children_right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f32>,
    pub value: Vec<f32>,
    pub tree_offsets: Vec<i32>,
    pub init_score: f32,
    
    // Stats for online growth (Hoeffding / Mondrian approximation)
    pub leaf_samples: Option<Vec<u32>>,
    pub leaf_variance: Option<Vec<f32>>,
}

#[derive(Clone)]
pub struct NanoForest {
    data: NanoForestData,
}

impl NanoForest {
    pub fn from_data(data: NanoForestData) -> Self {
        Self { data }
    }

    pub fn load_model(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let bin_path = path.replace(".json", ".bin");
        let data: NanoForestData = if let Ok(bin_data) = std::fs::read(&bin_path) {
            bincode::deserialize(&bin_data)?
        } else {
            // Fallback to JSON and auto-compile to bin!
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let parsed: NanoForestData = serde_json::from_reader(reader)?;
            if let Ok(encoded) = bincode::serialize(&parsed) {
                let _ = std::fs::write(&bin_path, encoded);
            }
            parsed
        };
        Ok(NanoForest { data })
    }

    /// Loads the forest into the global static cache under a specific key
    pub fn load_global(key: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let forest = Self::load_model(path)?;
        let current_map = crate::ml_inference::GLOBAL_FORESTS.load();
        let mut new_map = (**current_map).clone();
        new_map.insert(key.to_string(), Arc::new(forest));
        crate::ml_inference::GLOBAL_FORESTS.store(Arc::new(new_map));
        Ok(())
    }

    /// Predicts using a specific global forest
    pub fn predict_global(key: &str, features: &[f64], clip_lower: f64, clip_upper: f64) -> f64 {
        let map = crate::ml_inference::GLOBAL_FORESTS.load();
        if let Some(forest) = map.get(key) {
            // Cognitive Integrity (Phase 9): Consumimos los límites de confianza desde el Genoma Evolutivo
            return forest.predict(features, clip_lower, clip_upper);
        }
        panic!("Axioma VII Violado: NanoForest '{}' no cargado.", key);
    }

    /// Fetches a clone of the global forest for hot-path use without RwLock
    pub fn get_global(key: &str) -> Option<Arc<Self>> {
        let map = crate::ml_inference::GLOBAL_FORESTS.load();
        if let Some(forest) = map.get(key) {
            return Some(Arc::clone(forest));
        }
        None
    }

    /// Evaluates a single tree. Returns the leaf value.
    #[inline(always)]
    fn evaluate_tree(&self, features: &[f64], tree_idx: usize) -> f64 {
        unsafe {
            let start_node = *self.data.tree_offsets.get_unchecked(tree_idx) as usize;
            let mut current_node = start_node;

            loop {
                let left_child = *self.data.children_left.get_unchecked(current_node);
                let right_child = *self.data.children_right.get_unchecked(current_node);

                if left_child == -1 && right_child == -1 {
                    // Leaf node
                    return *self.data.value.get_unchecked(current_node) as f64;
                }

                let feat_idx = *self.data.feature.get_unchecked(current_node) as usize;
                let threshold = *self.data.threshold.get_unchecked(current_node) as f64;

                if *features.get_unchecked(feat_idx) <= threshold {
                    current_node = left_child as usize;
                } else {
                    current_node = right_child as usize;
                }
            }
        }
    }

    /// Evaluates a tree and returns the index of the leaf node reached.
    #[inline(always)]
    fn get_leaf_node(&self, features: &[f64], tree_idx: usize) -> usize {
        unsafe {
            let start_node = *self.data.tree_offsets.get_unchecked(tree_idx) as usize;
            let mut current_node = start_node;

            loop {
                let left_child = *self.data.children_left.get_unchecked(current_node);
                let right_child = *self.data.children_right.get_unchecked(current_node);

                if left_child == -1 && right_child == -1 {
                    return current_node;
                }

                let feat_idx = *self.data.feature.get_unchecked(current_node) as usize;
                let threshold = *self.data.threshold.get_unchecked(current_node) as f64;

                if *features.get_unchecked(feat_idx) <= threshold {
                    current_node = left_child as usize;
                } else {
                    current_node = right_child as usize;
                }
            }
        }
    }

    /// Predicts the probability for the given features.
    pub fn predict(&self, features: &[f64], min_prob_clamp: f64, max_prob_clamp: f64) -> f64 {
        let n_trees = self.data.tree_offsets.len().saturating_sub(1);
        if n_trees == 0 {
            return 0.5;
        }

        let mut sum = 0.0;
        for i in 0..n_trees {
            sum += self.evaluate_tree(features, i);
        }

        let avg = sum / (n_trees as f64);
        // Trees predict an expected value in [-1.0, 1.0] (1.0 = Buy, -1.0 = Sell, 0.0 = Flat)
        // Average is in [-1.0, 1.0]. Convert linearly to probability [0.0, 1.0]
        let prob = (avg + 1.0) / 2.0;
        prob.clamp(min_prob_clamp, max_prob_clamp)
    }

    /// Online Learning: Actualiza los pesos de las hojas usando el resultado real (PnL / Label)
    /// y expande el árbol estocásticamente si una hoja acumula suficientes muestras.
    pub fn update_online(&mut self, features: &[f64], label: f64, lr: f64) {
        let n_trees = self.data.tree_offsets.len().saturating_sub(1);
        if n_trees == 0 { return; }

        // Inicializar arreglos de crecimiento si no existen
        if self.data.leaf_samples.is_none() {
            self.data.leaf_samples = Some(vec![0; self.data.value.len()]);
            self.data.leaf_variance = Some(vec![0.0; self.data.value.len()]);
        }

        let mut rng = rand::thread_rng();

        for i in 0..n_trees {
            let leaf_idx = self.get_leaf_node(features, i);
            
            // 1. SGD: Update leaf value
            let old_val = self.data.value[leaf_idx];
            // Gradient step
            let error = label - old_val as f64;
            self.data.value[leaf_idx] = (old_val as f64 + lr * error) as f32;
            
            // 2. Acumular estadísticas
            self.data.leaf_samples.as_mut().unwrap()[leaf_idx] += 1;
            self.data.leaf_variance.as_mut().unwrap()[leaf_idx] += (error * error) as f32; // aproximación rápida

            // 3. Crecimiento dinámico (Hoeffding Tree Splitting condition)
            // Si la hoja tiene más de 100 muestras y el error es alto, realizar un split aleatorio
            if self.data.leaf_samples.as_mut().unwrap()[leaf_idx] > 100 && self.data.leaf_variance.as_mut().unwrap()[leaf_idx] > 10.0 {
                // Seleccionamos un feature al azar para partir (Mondrian approach)
                use rand::Rng;
                let split_feature = rng.gen_range(0..features.len()) as i32;
                let split_threshold = features[split_feature as usize] as f32; // partir en el valor actual

                // Crear dos nuevos nodos hoja
                let new_left_idx = self.data.value.len() as i32;
                self.data.value.push(self.data.value[leaf_idx]); // Heredan el valor
                self.data.feature.push(-1);
                self.data.threshold.push(0.0);
                self.data.children_left.push(-1);
                self.data.children_right.push(-1);
                self.data.leaf_samples.as_mut().unwrap().push(0);
                self.data.leaf_variance.as_mut().unwrap().push(0.0);

                let new_right_idx = self.data.value.len() as i32;
                self.data.value.push(self.data.value[leaf_idx]);
                self.data.feature.push(-1);
                self.data.threshold.push(0.0);
                self.data.children_left.push(-1);
                self.data.children_right.push(-1);
                self.data.leaf_samples.as_mut().unwrap().push(0);
                self.data.leaf_variance.as_mut().unwrap().push(0.0);

                // Convertir la hoja actual en un nodo de decisión
                self.data.feature[leaf_idx] = split_feature;
                self.data.threshold[leaf_idx] = split_threshold;
                self.data.children_left[leaf_idx] = new_left_idx;
                self.data.children_right[leaf_idx] = new_right_idx;
                
                // Reset stats del nodo que ahora es de decisión
                self.data.leaf_samples.as_mut().unwrap()[leaf_idx] = 0;
                self.data.leaf_variance.as_mut().unwrap()[leaf_idx] = 0.0;
            }
        }
    }
}

