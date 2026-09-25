use risk_engine::epigenetic_capital_alloc::AllocationError;
use risk_engine::epigenetic_capital_alloc::EpigeneticCapitalAllocEngine as Alloc;

#[test]
fn checked_api_exposes_invalidity_and_legacy_preserves_cardinality() {
    assert_eq!(
        Alloc::try_allocate_epigenetic_universe_margin(&[1.0, 2.0], &[1.0], 10.0),
        Err(AllocationError::LengthMismatch)
    );
    assert_eq!(
        Alloc::allocate_epigenetic_universe_margin(&[1.0, 2.0], &[1.0], 10.0),
        vec![0.0; 2]
    );
    assert_eq!(
        Alloc::try_allocate_epigenetic_universe_margin(&[1.0, f64::NAN], &[1.0; 2], 10.0),
        Err(AllocationError::NonFiniteScore { index: 1 })
    );
    assert_eq!(
        Alloc::try_allocate_epigenetic_universe_margin(&[1.0], &[f64::INFINITY], 10.0),
        Err(AllocationError::NonFiniteMethylation { index: 0 })
    );
    assert_eq!(
        Alloc::try_allocate_epigenetic_universe_margin(&[1.0], &[1.0], 0.0),
        Err(AllocationError::InvalidCapital)
    );
    assert_eq!(
        Alloc::try_allocate_epigenetic_universe_margin(&[], &[], 10.0),
        Ok(vec![])
    );
}

#[test]
fn fixed_and_universe_apis_agree_and_smallest_positive_score_survives() {
    let scores = [f64::from_bits(1); 10];
    for capital in [13.0, 30.0, 100.0, 300.0] {
        let fixed =
            Alloc::try_allocate_epigenetic_top10_margin_with_capital(&scores, &[0.5; 10], capital)
                .unwrap();
        let dynamic =
            Alloc::try_allocate_epigenetic_universe_margin(&scores, &[0.5; 10], capital).unwrap();
        assert_eq!(fixed.as_slice(), dynamic.as_slice());
        assert!((fixed.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn open_debt_top_k_does_not_guarantee_minimum_notional() {
    let weights = Alloc::allocate_epigenetic_universe_margin(&[1.0, 1e-8], &[1.0; 2], 13.0);
    assert!(weights[1] > 0.0 && weights[1] * 13.0 < 5.0);
}

#[test]
fn absent_positive_evidence_keeps_capital_unallocated() {
    for scores in [[0.0; 10], [-1.0; 10], [f64::NAN; 10]] {
        assert_eq!(
            Alloc::allocate_epigenetic_top10_margin_with_capital(&scores, &[1.0; 10], 1000.0),
            [0.0; 10]
        );
    }
}

#[test]
fn invalid_capital_does_not_become_a_thirteen_dollar_account() {
    for capital in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            Alloc::allocate_epigenetic_top10_margin_with_capital(&[1.0; 10], &[1.0; 10], capital),
            [0.0; 10]
        );
    }
}

#[test]
fn zero_score_assets_do_not_receive_fabricated_floor_weights() {
    let weights = Alloc::allocate_epigenetic_universe_margin(&[1.0, 0.0, -1.0], &[1.0; 3], 1000.0);
    assert_eq!(weights, vec![1.0, 0.0, 0.0]);
}

#[test]
fn normalization_survives_large_finite_scores() {
    let weights =
        Alloc::allocate_epigenetic_universe_margin(&[f64::MAX, f64::MAX], &[1.0; 2], 1000.0);
    assert_eq!(weights, vec![0.5, 0.5]);
}

#[test]
fn positive_score_rescaling_preserves_allocations() {
    let base =
        Alloc::allocate_epigenetic_universe_margin(&[1.0, 0.5, 0.25], &[0.2, 0.5, 0.8], 1000.0);
    let small = Alloc::allocate_epigenetic_universe_margin(
        &[1e-100, 0.5e-100, 0.25e-100],
        &[0.2, 0.5, 0.8],
        1000.0,
    );
    for (a, b) in base.iter().zip(small.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}
