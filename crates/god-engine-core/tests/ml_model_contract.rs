use god_engine_core::ml_inference::{NanoForest, NanoForestData};

fn stump() -> NanoForestData {
    NanoForestData {
        children_left: vec![1, -1, -1],
        children_right: vec![2, -1, -1],
        feature: vec![0, -1, -1],
        threshold: vec![0.5, 0.0, 0.0],
        value: vec![0.0, -0.5, 0.8],
        tree_offsets: vec![0, 3],
        init_score: 0.0,
    }
}

#[test]
fn cycles_are_rejected_at_construction_without_running_them() {
    let mut m = stump();
    m.children_left[0] = 0;
    assert!(NanoForest::from_data(m).is_err());
    let mut m = stump();
    m.children_left[1] = 0;
    m.children_right[1] = 2;
    m.feature[1] = 0;
    assert!(NanoForest::from_data(m).is_err());
}
#[test]
fn unequal_parallel_arrays_are_rejected() {
    for k in 0..5 {
        let mut m = stump();
        match k {
            0 => {
                m.children_left.pop();
            }
            1 => {
                m.children_right.pop();
            }
            2 => {
                m.feature.pop();
            }
            3 => {
                m.threshold.pop();
            }
            _ => {
                m.value.pop();
            }
        }
        assert!(NanoForest::from_data(m).is_err(), "array {k}");
    }
}
#[test]
fn invalid_offsets_are_rejected() {
    for offsets in [
        vec![-1, 3],
        vec![1, 3],
        vec![0, 4],
        vec![0, 2],
        vec![0, 0, 3],
        vec![0, 3, 2],
    ] {
        let mut m = stump();
        m.tree_offsets = offsets;
        assert!(NanoForest::from_data(m).is_err());
    }
}
#[test]
fn invalid_children_and_split_features_are_rejected() {
    for child in [-2, -1, 3, i32::MAX] {
        let mut m = stump();
        m.children_left[0] = child;
        assert!(NanoForest::from_data(m).is_err());
    }
    let mut m = stump();
    m.feature[0] = -1;
    assert!(NanoForest::from_data(m).is_err());
}
#[test]
fn tree_edges_cannot_escape_their_offset_interval() {
    let mut m = stump();
    m.children_left.extend([1, -1, -1]);
    m.children_right.extend([2, -1, -1]);
    m.feature.extend([0, -1, -1]);
    m.threshold.extend([0.5, 0.0, 0.0]);
    m.value.extend([0.0, 2.0, 3.0]);
    m.tree_offsets.push(6);
    assert!(NanoForest::from_data(m).is_err());
}
#[test]
fn nonfinite_model_numbers_are_rejected() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut m = stump();
        m.init_score = invalid;
        assert!(NanoForest::from_data(m).is_err());
        let mut m = stump();
        m.value[1] = invalid;
        assert!(NanoForest::from_data(m).is_err());
        let mut m = stump();
        m.threshold[0] = invalid;
        assert!(NanoForest::from_data(m).is_err());
    }
}
#[test]
fn missing_required_features_are_not_neutral_predictions() {
    let mut m = stump();
    m.feature[0] = 5;
    let f = NanoForest::from_data(m).unwrap();
    assert!(f.predict(&[0.2]).is_none());
    assert!(f.predict(&[0.2; 6]).is_some());
}
#[test]
fn raw_invalid_inputs_do_not_produce_valid_evidence() {
    let f = NanoForest::from_data(stump()).unwrap();
    for input in [vec![], vec![f32::NAN], vec![f32::INFINITY]] {
        let (raw, probability) = f.predict_raw(&input);
        assert!(!raw.is_finite() && !probability.is_finite());
    }
}
#[test]
fn score_overflow_is_not_fabricated_probability_half() {
    let mut m = stump();
    m.init_score = f32::MAX;
    m.value[1] = f32::MAX;
    let f = NanoForest::from_data(m).unwrap();
    assert!(f.predict(&[0.1]).is_none());
    assert!(!f.predict_raw(&[0.1]).0.is_finite());
}
#[test]
fn valid_global_indices_and_constant_bias_keep_their_meaning() {
    let mut m = stump();
    m.children_left.extend([4, -1, -1]);
    m.children_right.extend([5, -1, -1]);
    m.feature.extend([0, -1, -1]);
    m.threshold.extend([0.5, 0.0, 0.0]);
    m.value.extend([0.0, 2.0, 3.0]);
    m.tree_offsets.push(6);
    assert_eq!(NanoForest::from_data(m).unwrap().predict_raw(&[0.1]).0, 1.5);
    let bias = NanoForestData {
        children_left: vec![],
        children_right: vec![],
        feature: vec![],
        threshold: vec![],
        value: vec![],
        tree_offsets: vec![0, 0],
        init_score: 3.0,
    };
    assert_eq!(
        NanoForest::from_data(bias).unwrap().predict_raw(&[0.0]).0,
        3.0
    );
}

#[test]
fn invalid_json_and_binary_are_rejected_before_cache_or_activation() {
    let base = std::env::temp_dir().join(format!(
        "tg_xx_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let json = base.with_extension("json");
    let bin = base.with_extension("bin");
    let mut invalid = stump();
    invalid.children_left[0] = 0;
    std::fs::write(&json, serde_json::to_vec(&invalid).unwrap()).unwrap();
    assert!(NanoForest::load_model(json.to_str().unwrap()).is_err());
    assert!(!bin.exists(), "rejected JSON must not create a cache");
    std::fs::write(&bin, bincode::serialize(&invalid).unwrap()).unwrap();
    assert!(NanoForest::load_model(bin.to_str().unwrap()).is_err());
    std::fs::remove_file(json).unwrap();
    std::fs::remove_file(bin).unwrap();
}
#[test]
fn a_deep_valid_tree_uses_iterative_validation_and_bounded_traversal() {
    let depth = 4096;
    let n = depth * 2 + 1;
    let mut m = NanoForestData {
        children_left: vec![-1; n],
        children_right: vec![-1; n],
        feature: vec![-1; n],
        threshold: vec![0.5; n],
        value: vec![1.0; n],
        tree_offsets: vec![0, n as i32],
        init_score: 0.0,
    };
    for i in 0..depth {
        m.children_left[i] = (i + 1) as i32;
        m.children_right[i] = (depth + 1 + i) as i32;
        m.feature[i] = 0;
    }
    let f = NanoForest::from_data(m).unwrap();
    assert_eq!(f.predict_raw_checked(&[0.0]).unwrap().0, 1.0);
}
#[test]
#[ignore = "read-only local inventory, not a portable CI fixture"]
fn inspect_saved_models_without_activation_or_cache_writes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut files: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    for path in files {
        let bytes = std::fs::read(&path).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        if json.get("tree_offsets").is_none() {
            println!(
                "{}: OTHER_MODEL_SCHEMA",
                path.file_name().unwrap().to_string_lossy()
            );
            continue;
        }
        let result = serde_json::from_slice::<NanoForestData>(&bytes)
            .map_err(|e| e.to_string())
            .and_then(NanoForest::from_data);
        println!(
            "{}: {}",
            path.file_name().unwrap().to_string_lossy(),
            match result {
                Ok(_) => "STRUCTURALLY_ACCEPTED".to_string(),
                Err(e) => e,
            }
        );
    }
}
