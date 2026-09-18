use god_engine_core::ml_inference::NanoForest;

fn main() {
    let forest = NanoForest::load_model("../../models/BTCUSDT_MOTOR.json").unwrap();
    println!("Loaded forest.");

    // Test with all zeros
    let features = vec![0.0; 54];
    let prob = forest.predict(&features).unwrap();
    println!("Prob (zeros): {:.4}", prob);

    // Test with all ones
    let features = vec![1.0; 54];
    let prob = forest.predict(&features).unwrap();
    println!("Prob (ones): {:.4}", prob);

    // Test with realistic-ish random numbers
    let mut features = vec![0.0; 54];
    for i in 0..54 {
        features[i] = (i as f32) / 54.0 - 0.5;
    }
    let prob = forest.predict(&features).unwrap();
    println!("Prob (range): {:.4}", prob);
}
