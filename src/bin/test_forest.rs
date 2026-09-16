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
    let file = std::fs::File::open("models/BTCUSDT_SCALP.json")?;
    let reader = std::io::BufReader::new(file);
    let json_data: god_engine_core::ml_inference::NanoForestData = serde_json::from_reader(reader)?;
    // B3.9 (auditoría): from_data ahora valida el contrato ML_VECTOR_DIM y
    // devuelve Result — un modelo incompatible se rechaza ruidoso aquí.
    let forest_from_json = god_engine_core::ml_inference::NanoForest::from_data(json_data)
        .expect("modelo fuera de contrato de dimensión");

    let forest_from_loader =
        god_engine_core::ml_inference::NanoForest::load_model("models/BTCUSDT_SCALP.json")?;

    let zero_34 = [0.0f32; 34];
    let (s_json, p_json) = forest_from_json.predict_raw(&zero_34);
    let (s_bin, p_bin) = forest_from_loader.predict_raw(&zero_34);

    println!("FROM JSON: sum = {:.4}, prob = {:.4}", s_json, p_json);
    println!("FROM BIN (loader): sum = {:.4}, prob = {:.4}", s_bin, p_bin);
    Ok(())
}
