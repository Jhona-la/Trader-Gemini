use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

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
        
        // Verificar frescura: si el JSON es más nuevo que el BIN, el BIN es obsoleto
        let is_stale = match (std::fs::metadata(path), std::fs::metadata(&bin_path)) {
            (Ok(m_json), Ok(m_bin)) => {
                let t_json = m_json.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                let t_bin = m_bin.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                t_json > t_bin
            }
            _ => false,
        };

        let data: NanoForestData = if !is_stale && std::path::Path::new(&bin_path).exists() {
            match std::fs::read(&bin_path) {
                Ok(bin_data) => match bincode::deserialize(&bin_data) {
                    Ok(parsed) => parsed,
                    Err(_) => {
                        let file = File::open(path)?;
                        let reader = BufReader::new(file);
                        let parsed: NanoForestData = serde_json::from_reader(reader)?;
                        if let Ok(encoded) = bincode::serialize(&parsed) {
                            let _ = std::fs::write(&bin_path, encoded);
                        }
                        parsed
                    }
                },
                Err(_) => {
                    let file = File::open(path)?;
                    let reader = BufReader::new(file);
                    let parsed: NanoForestData = serde_json::from_reader(reader)?;
                    if let Ok(encoded) = bincode::serialize(&parsed) {
                        let _ = std::fs::write(&bin_path, encoded);
                    }
                    parsed
                }
            }
        } else {
            // Fallback to JSON and auto-compile fresh BIN!
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

    /// Predicts using a specific global forest.
    /// F5.4: firma Option ⇒ comportamiento Option. El panic anterior mataba el
    /// proceso (panic=abort) con posiciones abiertas si un modelo no estaba
    /// cargado. Sin modelo: None ⇒ el caller decide (neutral 0.5 o no-trade).
    pub fn predict_global(key: &str, features: &[f32]) -> Option<f32> {
        let map = crate::ml_inference::GLOBAL_FORESTS.load();
        map.get(key).and_then(|forest| forest.predict(features))
    }

    /// Fetches a clone of the global forest for hot-path use without RwLock
    pub fn get_global(key: &str) -> Option<Arc<Self>> {
        let map = crate::ml_inference::GLOBAL_FORESTS.load();
        if let Some(forest) = map.get(key) {
            return Some(Arc::clone(forest));
        }
        None
    }

    /// Evaluates a single tree. Returns the leaf value with bounds protection.
    #[inline(always)]
    fn evaluate_tree(&self, features: &[f32], tree_idx: usize) -> f32 {
        let start_node = self.data.tree_offsets[tree_idx] as usize;
        let mut current_node = start_node;

        loop {
            let left_child = self.data.children_left.get(current_node).copied().unwrap_or(-1);
            let right_child = self.data.children_right.get(current_node).copied().unwrap_or(-1);

            if left_child == -1 && right_child == -1 {
                // Leaf node
                return self.data.value.get(current_node).copied().unwrap_or(0.0);
            }

            let feat_idx = self.data.feature.get(current_node).copied().unwrap_or(-1);
            if feat_idx < 0 || (feat_idx as usize) >= features.len() {
                // Out of bounds feature protection: fallback to left leaf or 0.0
                return 0.0;
            }
            let threshold = self.data.threshold.get(current_node).copied().unwrap_or(0.0);

            if features[feat_idx as usize] <= threshold {
                if left_child < 0 { return 0.0; }
                current_node = left_child as usize;
            } else {
                if right_child < 0 { return 0.0; }
                current_node = right_child as usize;
            }
        }
    }

    /// Predicts the probability for the given features.
    pub fn predict(&self, features: &[f32]) -> Option<f32> {
        if self.data.tree_offsets.len() <= 1 || features.is_empty() {
            return None;
        }
        // FIX #688: Validar finitud de todos los features
        for &f in features {
            if !f.is_finite() {
                return None;
            }
        }

        let n_trees = self.data.tree_offsets.len() - 1;
        let mut sum = self.data.init_score;

        for i in 0..n_trees {
            sum += self.evaluate_tree(features, i);
        }

        // Apply sigmoid with clamping for numerical stability [-50, +50]
        let safe_sum = if sum.is_finite() { sum } else { 0.0 };
        let clamped_sum = (-safe_sum).clamp(-50.0, 50.0);
        let prob = 1.0 / (1.0 + clamped_sum.exp());
        Some(if prob.is_finite() { prob.clamp(0.0, 1.0) } else { 0.5 })
    }

    pub fn predict_raw(&self, features: &[f32]) -> (f32, f32) {
        let n_trees = self.data.tree_offsets.len().saturating_sub(1);
        let mut sum = self.data.init_score;
        for i in 0..n_trees {
            sum += self.evaluate_tree(features, i);
        }
        let safe_sum = if sum.is_finite() { sum } else { 0.0 };
        let clamped_sum = (-safe_sum).clamp(-50.0, 50.0);
        let prob = 1.0 / (1.0 + clamped_sum.exp());
        (sum, prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nano_forest_synthetic_prediction() {
        let data = NanoForestData {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![0, -1, -1],
            threshold: vec![0.5, 0.0, 0.0],
            value: vec![0.0, -0.5, 0.8],
            tree_offsets: vec![0, 3],
            init_score: 0.0,
        };
        let forest = NanoForest::from_data(data);

        // Feature 0 <= 0.5 -> leaf value -0.5 -> prob < 0.50
        let prob_low = forest.predict(&[0.2]).unwrap();
        assert!(prob_low < 0.50);

        // Feature 0 > 0.5 -> leaf value 0.8 -> prob > 0.50
        let prob_high = forest.predict(&[0.9]).unwrap();
        assert!(prob_high > 0.50);

        // Empty features -> None
        assert!(forest.predict(&[]).is_none());
    }

    #[test]
    fn test_nano_forest_nan_and_out_of_bounds_features() {
        let data = NanoForestData {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![5, -1, -1], // Index 5 out of bounds for 1-element input
            threshold: vec![0.5, 0.0, 0.0],
            value: vec![0.0, -0.5, 0.8],
            tree_offsets: vec![0, 3],
            init_score: 0.0,
        };
        let forest = NanoForest::from_data(data);

        // NaN feature -> None
        assert!(forest.predict(&[f32::NAN]).is_none());

        // Out of bounds feature index falls back safely
        let prob = forest.predict(&[0.2]);
        assert!(prob.is_some());
    }

    #[test]
    fn test_nano_forest_global_cache_and_clone() {
        let data = NanoForestData {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![0, -1, -1],
            threshold: vec![0.5, 0.0, 0.0],
            value: vec![0.0, -0.5, 0.8],
            tree_offsets: vec![0, 3],
            init_score: 0.0,
        };
        let forest = NanoForest::from_data(data);

        // Store directly in GLOBAL_FORESTS
        let current_map = GLOBAL_FORESTS.load();
        let mut new_map = (**current_map).clone();
        new_map.insert("TEST_MODEL".to_string(), Arc::new(forest));
        GLOBAL_FORESTS.store(Arc::new(new_map));

        let retrieved = NanoForest::get_global("TEST_MODEL");
        assert!(retrieved.is_some());

        let prob = NanoForest::predict_global("TEST_MODEL", &[0.8]);
        assert!(prob.is_some());
        assert!(prob.unwrap() > 0.50);

        let non_existent = NanoForest::predict_global("NON_EXISTENT", &[0.8]);
        assert!(non_existent.is_none());
    }
}

