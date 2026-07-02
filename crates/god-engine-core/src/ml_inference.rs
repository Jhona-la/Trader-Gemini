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
    pub fn predict_global(key: &str, features: &[f32]) -> f32 {
        let map = crate::ml_inference::GLOBAL_FORESTS.load();
        if let Some(forest) = map.get(key) {
            return forest.predict(features);
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
    fn evaluate_tree(&self, features: &[f32], tree_idx: usize) -> f32 {
        let start_node = self.data.tree_offsets[tree_idx] as usize;
        let mut current_node = start_node;

        loop {
            let left_child = self.data.children_left[current_node];
            let right_child = self.data.children_right[current_node];

            if left_child == -1 && right_child == -1 {
                // Leaf node
                return self.data.value[current_node];
            }

            let feat_idx = self.data.feature[current_node] as usize;
            let threshold = self.data.threshold[current_node];

            if features[feat_idx] <= threshold {
                current_node = left_child as usize;
            } else {
                current_node = right_child as usize;
            }
        }
    }

    /// Predicts the probability for the given features.
    pub fn predict(&self, features: &[f32]) -> f32 {
        let n_trees = self.data.tree_offsets.len() - 1;
        let mut sum = self.data.init_score;

        for i in 0..n_trees {
            sum += self.evaluate_tree(features, i);
        }

        // Apply sigmoid for GradientBoostingClassifier
        1.0 / (1.0 + (-sum).exp())
    }
}
