//! Pure numerical/control tests. No live daemon, exchange, ledger, or genome IO.
use evolution_engine::return_evidence::{
    EvidenceError, EvidenceEwma, EwmaError, MeanStatistic, latch_degradation, summarize_returns,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn positive_rescaling_preserves_statistic_across_500_orders() {
    let base = [
        -0.02, 0.03, -0.01, 0.04, 0.01, -0.01, 0.05, 0.02, 0.06, 0.03,
    ];
    let reference = summarize_returns(&base).unwrap().studentized().unwrap();
    for scale in [1e-250, 1e-100, 1e-20, 1.0, 1e20, 1e100, 1e250] {
        let scaled: Vec<_> = base.iter().map(|x| x * scale).collect();
        let evidence = summarize_returns(&scaled).unwrap();
        assert!((evidence.studentized().unwrap() - reference).abs() < 1e-12);
    }
}

#[test]
fn sign_change_reverses_t_without_changing_dispersion() {
    let data = [-0.01, 0.03, 0.02, 0.04];
    let a = summarize_returns(&data).unwrap();
    let b = summarize_returns(&data.map(|x| -x)).unwrap();
    assert_eq!(a.sample_std, b.sample_std);
    assert_eq!(a.mean, -b.mean);
    assert_eq!(a.studentized(), b.studentized().map(|t| -t));
}

#[test]
fn zero_mean_is_a_valid_zero_statistic_not_missing_data() {
    let evidence = summarize_returns(&[-1.0, 1.0, -1.0, 1.0]).unwrap();
    assert_eq!(evidence.statistic, MeanStatistic::Studentized(0.0));
    assert!(!evidence.is_constant_loss());
}

#[test]
fn all_constant_signs_have_undefined_student_statistic() {
    for value in [-1.0, -0.0, 0.0, 1.0, f64::MAX] {
        let evidence = summarize_returns(&[value; 25]).unwrap();
        assert_eq!(evidence.n, 25);
        assert_eq!(evidence.mean, value);
        assert_eq!(evidence.sample_std, 0.0);
        assert_eq!(evidence.studentized(), None);
        assert_eq!(evidence.is_constant_loss(), value < 0.0);
    }
}

#[test]
fn few_observations_do_not_invent_an_estimate() {
    for data in [vec![], vec![1.0]] {
        assert_eq!(
            summarize_returns(&data),
            Err(EvidenceError::InsufficientObservations)
        );
    }
}

#[test]
fn two_observations_have_well_defined_descriptive_t() {
    let evidence = summarize_returns(&[1.0, 3.0]).unwrap();
    assert!((evidence.sample_std - 2.0_f64.sqrt()).abs() < 1e-15);
    assert!((evidence.studentized().unwrap() - 2.0).abs() < 1e-15);
    // Mathematical existence is not permission for a promotion or a p-value.
}

#[test]
fn nonfinite_values_reject_the_whole_batch_at_their_index() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            summarize_returns(&[0.1, bad, 0.2]),
            Err(EvidenceError::NonFiniteObservation { index: 1 })
        );
    }
}

#[test]
fn negative_tail_remains_in_evidence() {
    let mut data = vec![0.01; 30];
    data.push(-0.5);
    let evidence = summarize_returns(&data).unwrap();
    assert_eq!(evidence.n, 31);
    assert!(evidence.mean < 0.0);
    assert!(evidence.studentized().unwrap() < 0.0);
}

#[test]
fn tiny_nonconstant_variance_is_not_replaced_by_zero() {
    let evidence = summarize_returns(&[1.0, 1.0 + f64::EPSILON, 1.0, 1.0 + f64::EPSILON]).unwrap();
    assert!(evidence.sample_std > 0.0);
    assert!(evidence.studentized().unwrap().is_finite());
}

#[test]
fn unrepresentable_dispersion_is_an_error_not_a_neutral_score() {
    assert_eq!(
        summarize_returns(&[-f64::MAX, f64::MAX]),
        Err(EvidenceError::NumericalRange)
    );
}

#[test]
fn ewma_initialization_does_not_use_zero_as_a_sentinel() {
    let mut ewma = EvidenceEwma::default();
    assert_eq!(ewma.value(), None);
    assert_eq!(ewma.observe(1, 0.0, 0.1), Ok(true));
    assert_eq!(ewma.observe(2, 2.0, 0.1), Ok(true));
    assert_eq!(ewma.value(), Some(0.2));
}

#[test]
fn unchanged_or_older_observations_cannot_accumulate_evidence_by_polling() {
    let mut ewma = EvidenceEwma::default();
    ewma.observe(1, -0.4, 0.1).unwrap();
    ewma.observe(2, -2.0, 0.1).unwrap();
    let once = ewma.clone();
    for _ in 0..1000 {
        assert_eq!(ewma.observe(2, -2.0, 0.1), Ok(false));
        assert_eq!(ewma.observe(1, -100.0, 0.1), Ok(false));
    }
    assert_eq!(ewma, once);
    assert!((ewma.value().unwrap() + 0.56).abs() < 1e-15);
}

#[test]
fn invalid_ewma_inputs_do_not_consume_revision_or_mutate_state() {
    let mut ewma = EvidenceEwma::default();
    ewma.observe(1, 1.0, 0.1).unwrap();
    let before = ewma.clone();
    for weight in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert_eq!(ewma.observe(2, 2.0, weight), Err(EwmaError::InvalidWeight));
        assert_eq!(ewma, before);
    }
    for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(ewma.observe(2, score, 0.1), Err(EwmaError::InvalidScore));
        assert_eq!(ewma, before);
    }
    assert_eq!(ewma.observe(2, 2.0, 0.1), Ok(true));
}

#[test]
fn weight_one_tracks_only_new_observations() {
    let mut ewma = EvidenceEwma::default();
    ewma.observe(20, 1.0, 1.0).unwrap();
    ewma.observe(21, -1.0, 1.0).unwrap();
    assert_eq!(ewma.value(), Some(-1.0));
}

#[test]
fn safety_latch_truth_table_never_clears_an_existing_stop() {
    for initial in [false, true] {
        for degraded in [false, true] {
            let stop = AtomicBool::new(initial);
            assert_eq!(latch_degradation(&stop, degraded), !initial && degraded);
            assert_eq!(stop.load(Ordering::Acquire), initial || degraded);
        }
    }
}

#[test]
fn healthy_observations_never_rearm_a_latency_or_drawdown_stop() {
    let stop = AtomicBool::new(true);
    let mut ewma = EvidenceEwma::default();
    for revision in 1..=100 {
        ewma.observe(revision, 10.0, 0.1).unwrap();
        assert!(!latch_degradation(&stop, ewma.value().unwrap() < -1.5));
        assert!(stop.load(Ordering::Acquire));
    }
}

#[test]
fn constant_loss_can_latch_safety_without_fabricating_a_t_value() {
    let evidence = summarize_returns(&[-0.01; 25]).unwrap();
    assert_eq!(evidence.studentized(), None);
    let stop = AtomicBool::new(false);
    assert!(latch_degradation(&stop, evidence.is_constant_loss()));
    assert!(!latch_degradation(&stop, evidence.is_constant_loss()));
    assert!(stop.load(Ordering::Acquire));
}

#[test]
fn simultaneous_daemon_requests_have_one_latch_transition_and_no_release() {
    use std::sync::{Arc, Barrier};
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(16));
    let workers: Vec<_> = (0..16)
        .map(|i| {
            let stop = stop.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                latch_degradation(&stop, i % 2 == 0)
            })
        })
        .collect();
    let activations: usize = workers
        .into_iter()
        .map(|worker| usize::from(worker.join().unwrap()))
        .sum();
    assert_eq!(activations, 1);
    assert!(stop.load(Ordering::Acquire));
}
