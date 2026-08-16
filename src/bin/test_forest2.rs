use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;

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
    println!("N Trees: {}", n_trees);

    let mut sum_leaf_values = 0.0;
    let mut num_leaves = 0;

    for i in 0..data.children_left.len() {
        if data.children_left[i] == -1 && data.children_right[i] == -1 {
            sum_leaf_values += data.value[i];
            num_leaves += 1;
        }
    }

    println!(
        "Num Leaves: {}, Avg Leaf Value: {}",
        num_leaves,
        sum_leaf_values / num_leaves as f32
    );
    Ok(())
}
