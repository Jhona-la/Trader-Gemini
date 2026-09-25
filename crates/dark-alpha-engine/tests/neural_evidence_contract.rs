use dark_alpha_engine::{ChannelWelfordStats, DarkAlphaEngine, DenseLayer, Scaler};

fn model() -> DarkAlphaEngine {
    let mut m = DarkAlphaEngine::new(1, 1, 1);
    for layer in [&mut m.layer1, &mut m.layer2, &mut m.layer3] {
        layer.weights[0] = 1.0;
        layer.biases[0] = 0.0;
    }
    m.scaler = Some(Scaler::new(vec![0.0], vec![1.0]));
    m
}

fn state(m: &DarkAlphaEngine) -> Vec<u8> {
    bincode::serialize(m).unwrap()
}

#[test]
fn buffer_initialization_preserves_small_normal_weights_and_prediction() {
    let mut m = model();
    m.layer1.weights[0] = 1e-8;
    m.layer2.weights[0] = 1e8;
    let expected = m.predict(&[1.0]).unwrap();
    let bytes = state(&m);
    m.init_buffers();
    assert_eq!(state(&m), bytes);
    assert_eq!(m.predict(&[1.0]), Some(expected));
}

#[test]
fn subnormal_cleanup_preserves_normal_numbers_and_does_not_hide_nan() {
    let mut layer = DenseLayer::new(3, 1);
    layer.weights = vec![1e-8, f64::from_bits(1), f64::NAN];
    layer.sanitize_denormals();
    assert_eq!(layer.weights[0], 1e-8);
    assert_eq!(layer.weights[1], 0.0);
    assert!(layer.weights[2].is_nan());
}

#[test]
fn numerical_model_corruption_is_rejected_at_validation() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut m = model();
        m.layer1.weights[0] = value;
        assert!(m.validate().is_err());
        assert!(m.predict(&[1.0]).is_none());
    }
}

#[test]
fn incomplete_or_invalid_scalers_are_not_mixed_with_raw_features() {
    for scaler in [
        Scaler::new(vec![], vec![]),
        Scaler::new(vec![f64::NAN], vec![1.0]),
        Scaler::new(vec![0.0], vec![-1.0]),
    ] {
        let mut m = model();
        m.scaler = Some(scaler);
        assert!(m.validate().is_err());
        assert!(m.predict_for_coin(0, &[1.0]).is_none());
    }
}

#[test]
fn binary_model_rejects_multiple_outputs() {
    let mut m = model();
    m.layer3 = DenseLayer::new(1, 2);
    assert!(m.validate().is_err());
    assert!(m.predict(&[1.0]).is_none());
}

#[test]
fn invalid_observation_is_absent_evidence_and_does_not_update_state() {
    let mut m = model();
    m.scaler = None;
    let before = state(&m);
    assert!(m.predict(&[f64::NAN]).is_none());
    assert!(m.predict_for_coin(0, &[f64::INFINITY]).is_none());
    assert_eq!(state(&m), before);
}

#[test]
fn finite_parameter_overflow_is_not_a_neutral_prediction() {
    let mut m = model();
    m.layer1.weights[0] = -f64::MAX;
    assert!(m.predict(&[2.0]).is_none());
}

fn frozen_model(count: f64) -> DarkAlphaEngine {
    let mut m = model();
    m.scaler = None;
    m.channel_normalizers[0] = ChannelWelfordStats {
        count,
        mean: 1.0,
        m2: 2.0 * (count - 1.0),
        is_decay: false,
    };
    m.freeze();
    m
}

#[test]
fn freeze_does_not_warm_up_or_mutate_global_or_local_statistics() {
    let mut m = frozen_model(2.0);
    let before = state(&m);
    assert!(m.predict(&[4.0]).is_some());
    assert!(m.predict_for_coin(0, &[4.0]).is_some());
    assert_eq!(state(&m), before);
}

#[test]
fn frozen_apis_use_the_same_available_global_coordinates() {
    let mut global = frozen_model(20.0);
    let mut per_coin = global.clone();
    assert_eq!(global.predict(&[4.0]), per_coin.predict_for_coin(0, &[4.0]));
}

#[test]
fn cold_frozen_model_abstains_without_learning_from_evaluation() {
    let mut m = model();
    m.scaler = None;
    m.freeze();
    let before = state(&m);
    assert!(m.predict(&[1.0]).is_none());
    assert!(m.predict_for_coin(0, &[1.0]).is_none());
    assert_eq!(state(&m), before);
}

#[test]
fn invalid_fit_batch_does_not_partially_change_parameters() {
    let mut m = model();
    let before = state(&m);
    m.fit(&[vec![1.0], vec![2.0]], &[1.0, f64::NAN], 1, 0.01);
    assert_eq!(state(&m), before);
}

#[test]
fn invalid_normalizer_state_is_rejected() {
    let mut m = frozen_model(20.0);
    m.channel_normalizers[0].m2 = -1.0;
    assert!(m.validate().is_err());
}

#[test]
fn json_and_binary_roundtrips_preserve_small_parameters_and_scores() {
    let mut original = model();
    original.layer1.weights[0] = 1e-8;
    original.layer2.weights[0] = 1e8;
    let expected = original.predict(&[1.0]).unwrap();
    let json: DarkAlphaEngine =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    let binary: DarkAlphaEngine = bincode::deserialize(&state(&original)).unwrap();
    for mut restored in [json, binary] {
        restored.validate().unwrap();
        restored.init_buffers();
        assert!((restored.predict(&[1.0]).unwrap() - expected).abs() < 1e-12);
        assert_eq!(restored.layer1.weights[0], 1e-8);
    }
}

#[test]
fn frozen_unknown_slot_uses_global_state_without_allocating_statistical_slots() {
    let mut m = frozen_model(20.0);
    let before = state(&m);
    assert!(m.predict_for_coin(usize::MAX, &[1.0]).is_some());
    assert!(state(&m) == before);
}

#[test]
fn frozen_local_fallback_never_updates_other_assets() {
    let mut m = frozen_model(2.0);
    m.per_coin_normalizers[3][0] = m.channel_normalizers[0];
    m.channel_normalizers[0] = ChannelWelfordStats::new();
    let before = state(&m);
    assert!(m.predict_for_coin(3, &[4.0]).is_some());
    assert!(m.predict_for_coin(4, &[4.0]).is_none());
    assert!(state(&m) == before);
}

#[test]
fn scaler_overflow_is_not_winsorized_to_zero() {
    let s = Scaler::new(vec![-f64::MAX], vec![1.0]);
    assert!(s.scale_checked(&mut [f64::MAX]).is_err());
    let mut m = model();
    m.scaler = Some(s);
    assert!(m.predict(&[f64::MAX]).is_none());
}

#[test]
fn finite_input_training_overflow_rolls_back_whole_fit() {
    let mut m = model();
    m.scaler = Some(Scaler::new(vec![-f64::MAX], vec![1.0]));
    let before = state(&m);
    m.fit(&[vec![0.0], vec![f64::MAX]], &[1.0, 0.0], 2, 0.1);
    assert!(state(&m) == before);
}

#[test]
fn adaptive_normalizer_preflights_all_channels_before_committing_moments() {
    let mut m = DarkAlphaEngine::new(2, 2, 1);
    m.predict(&[1.0, f64::MAX]).unwrap();
    let before = state(&m);
    assert!(m.predict(&[2.0, -f64::MAX]).is_none());
    assert!(state(&m) == before);
}
