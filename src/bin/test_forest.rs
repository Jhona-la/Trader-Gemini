use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("models/BTCUSDT_SCALP.json")?;
    let reader = BufReader::new(file);
    let data: NanoForestData = serde_json::from_reader(reader)?;
    
    let n_trees = data.tree_offsets.len() - 1;
    for &val in &[0.0, 1.0, -1.0, 100.0, -100.0] {
        let features = [val; 12];
        let mut sum = data.init_score;
        for i in 0..n_trees {
            let start_node = data.tree_offsets[i] as usize;
            let mut current_node = start_node;

            loop {
                let left_child = data.children_left[current_node];
                let right_child = data.children_right[current_node];

                if left_child == -1 && right_child == -1 {
                    sum += data.value[current_node];
                    break;
                }

                let feat_idx = data.feature[current_node] as usize;
                let threshold = data.threshold[current_node];

                if features[feat_idx] <= threshold {
                    current_node = left_child as usize;
                } else {
                    current_node = right_child as usize;
                }
            }
        }
        let prob = 1.0 / (1.0 + (-sum).exp());
        println!("Val: {}, Sum: {}, Prob: {}", val, sum, prob);
    }
    Ok(())
}
