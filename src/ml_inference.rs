use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;

lazy_static::lazy_static! {
    pub static ref GLOBAL_FOREST: RwLock<Option<NanoForest>> = RwLock::new(None);
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NanoForestData {
    pub children_left: Vec<i32>,
    pub children_right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f32>,
    pub value: Vec<f32>,
    pub tree_offsets: Vec<i32>,
    pub init_score: f32,
}

pub struct NanoForest {
    data: NanoForestData,
}

impl NanoForest {
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

    /// Loads the forest into the global static cache
    pub fn load_global(path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let forest = Self::load_model(path)?;
        if let Ok(mut lock) = crate::ml_inference::GLOBAL_FOREST.write() {
            *lock = Some(forest);
        }
        Ok(())
    }

    /// Predicts using the global forest
    pub fn predict_global(features: &[f32]) -> f32 {
        if let Ok(lock) = crate::ml_inference::GLOBAL_FOREST.read() {
            if let Some(forest) = lock.as_ref() {
                return forest.predict(features);
            }
        }
        0.5 // Default probability if not loaded
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
