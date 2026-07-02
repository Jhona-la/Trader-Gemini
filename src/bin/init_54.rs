fn main() {
    let model = dark_alpha_engine::DarkAlphaEngine::default_model();
    let file = std::fs::File::create("models/DarkAlpha_BTCUSDT.json").unwrap();
    serde_json::to_writer_pretty(file, &model).unwrap();
    println!("Model initialized with 54 features.");
}
