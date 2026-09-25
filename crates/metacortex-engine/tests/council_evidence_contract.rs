//! XXXI: council admission and feedback contracts, synthetic memory only.
use metacortex_engine::consejo_seniors::*;

fn snapshot() -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        horizon: TradingHorizon::Continuous,
        book_imbalance: 0.85,
        hurst_exponent: 0.72,
        ml_prob: 0.70,
        fused_score: 0.50,
        persistence: 0.60,
        atr_pct: 0.001,
        loss_streak: 0,
        intended_direction: 1.0,
        do_calculus_risk: 0.05,
        causal_veto_threshold: 0.75,
        current_drawdown_pct: 0.05,
        estimated_slippage_bps: 1.0,
        dominant_tau_ms: 30_000.0,
        whale_burst_z: 0.0,
        liquidation_severity: 0.0,
        open_interest_norm: 0.0,
        spoof_score: 0.0,
        crowd_ls_ratio: 1.0,
        crowd_taker_ratio: 1.0,
        ml_model_base: 0.5,
    }
}

#[test]
fn formerly_unchecked_fields_reject_nonfinite_evidence() {
    let edits: [fn(&mut MarketSnapshotPayload, f64); 9] = [
        |p, x| p.causal_veto_threshold = x,
        |p, x| p.dominant_tau_ms = x,
        |p, x| p.whale_burst_z = x,
        |p, x| p.liquidation_severity = x,
        |p, x| p.open_interest_norm = x,
        |p, x| p.spoof_score = x,
        |p, x| p.crowd_ls_ratio = x,
        |p, x| p.crowd_taker_ratio = x,
        |p, x| p.ml_model_base = x,
    ];
    for (i, edit) in edits.iter().enumerate() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut p = snapshot();
            edit(&mut p, value);
            assert!(p.validate().is_err(), "field {i} accepts {value}");
            assert!(!ConsejoDeliberacion::new().deliberar(&p, 0.5).approved);
        }
    }
}

#[test]
fn invalid_finite_domains_are_not_permission() {
    let edits: [fn(&mut MarketSnapshotPayload); 12] = [
        |p| p.book_imbalance = 1.1,
        |p| p.fused_score = -1.1,
        |p| p.do_calculus_risk = -0.1,
        |p| p.current_drawdown_pct = -0.1,
        |p| p.estimated_slippage_bps = -1.0,
        |p| p.dominant_tau_ms = 0.0,
        |p| p.causal_veto_threshold = 1.1,
        |p| p.liquidation_severity = 1.1,
        |p| p.open_interest_norm = -0.1,
        |p| p.spoof_score = 1.1,
        |p| p.crowd_ls_ratio = -1.0,
        |p| p.ml_model_base = 1.1,
    ];
    for (i, edit) in edits.iter().enumerate() {
        let mut p = snapshot();
        edit(&mut p);
        assert!(p.validate().is_err(), "invalid domain {i}");
    }
}

#[test]
fn malformed_policy_parameters_are_rejected_explicitly() {
    let edits: [fn(&mut CouncilParams); 6] = [
        |p| p.approval_threshold = -1.0,
        |p| p.supermajority_override = f64::NAN,
        |p| p.override_signal_floor = -1.0,
        |p| p.override_penalty = 2.0,
        |p| p.shrinkage_k = -8.0,
        |p| p.cascade_severity_breaker = f64::NAN,
    ];
    for (i, edit) in edits.iter().enumerate() {
        let mut council = ConsejoDeliberacion::new();
        edit(&mut council.params);
        let result = council.deliberar(&snapshot(), 0.7);
        assert!(!result.approved, "invalid policy {i} accepted");
        assert!(
            result
                .dissenting_log
                .iter()
                .any(|o| o.justification.contains("integrity")),
            "invalid policy {i} was not diagnosed"
        );
    }
}

#[test]
fn invalid_external_multipliers_cannot_silently_fall_back() {
    for value in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        let mut weights = [1.0; 11];
        weights[0] = value;
        let result =
            ConsejoDeliberacion::new().deliberar_with_weights(&snapshot(), 0.7, Some(&weights));
        assert!(!result.approved, "invalid multiplier {value}");
    }
}

#[test]
fn aligned_breakout_exception_requires_actual_direction_agreement() {
    for direction in [-1.0, 1.0] {
        let mut p = snapshot();
        p.intended_direction = direction;
        p.book_imbalance = -0.5 * direction;
        p.do_calculus_risk = 0.8;
        assert!(SeniorCausal.evaluate(&p, 0.5).is_veto);
        p.book_imbalance = 0.5 * direction;
        assert!(!SeniorCausal.evaluate(&p, 0.5).is_veto);
        p.intended_direction = 0.0;
        assert!(SeniorCausal.evaluate(&p, 0.5).is_veto);
    }
}

#[test]
fn configured_cascade_breaker_is_effective_below_legacy_literal() {
    let mut council = ConsejoDeliberacion::new();
    council.params.cascade_severity_breaker = 0.5;
    let mut p = snapshot();
    p.liquidation_severity = 0.6;
    let result = council.deliberar(&p, 0.7);
    assert!(!result.approved);
    assert_eq!(result.vetoed_by, Some(SeniorRole::EnteMercado));
}

#[test]
fn teleonomy_has_no_directional_ml_signal_at_the_model_base() {
    let mut p = snapshot();
    p.ml_prob = 0.3;
    p.ml_model_base = 0.3;
    p.fused_score = 0.0;
    assert_eq!(SeniorTeleonomia.evaluate(&p, 0.7).signal_direction, 0.0);
}

#[test]
fn teleonomy_is_invariant_to_a_common_probability_base_translation() {
    let mut p = snapshot();
    p.ml_prob = 0.45;
    p.ml_model_base = 0.3;
    let before = SeniorTeleonomia.evaluate(&p, 0.7).signal_direction;
    p.ml_prob += 0.2;
    p.ml_model_base += 0.2;
    let after = SeniorTeleonomia.evaluate(&p, 0.7).signal_direction;
    assert!((before - after).abs() < 1e-12);
}

#[test]
fn abstentions_do_not_become_successes_or_trials() {
    let mut tracker = SeniorPerformanceTracker::new(10);
    tracker.record_outcome(&[0.0; 11], 0.01);
    assert_eq!(tracker.total_counts, [0; 11]);
    assert_eq!(tracker.correct_counts, [0; 11]);
}

#[test]
fn masked_modulators_remain_untrained_after_winning_outcomes() {
    let council = ConsejoDeliberacion::new();
    for _ in 0..10 {
        council.record_outcome(&[1.0; 11], 0.01);
    }
    let tracker = council.tracker.read().unwrap();
    for role in [
        SeniorRole::Volatilidad,
        SeniorRole::Riesgo,
        SeniorRole::EnteMercado,
    ] {
        assert_eq!(tracker.total_counts[role as usize], 0);
        assert_eq!(tracker.compute_weights()[role as usize], 1.0);
    }
}

#[test]
fn malformed_signal_vector_does_not_pollute_tracker_state() {
    let mut tracker = SeniorPerformanceTracker::new(10);
    let mut signals = [1.0; 11];
    signals[6] = f64::NAN;
    tracker.record_outcome(&signals, 0.01);
    assert_eq!(tracker.total_outcomes(), 0);
    assert_eq!(tracker.total_counts, [0; 11]);
}

#[test]
fn rolling_window_counts_only_nonzero_votes_when_evicted() {
    let mut tracker = SeniorPerformanceTracker::new(10);
    let mut signals = [0.0; 11];
    signals[0] = 1.0;
    tracker.record_outcome(&signals, 0.01);
    for _ in 0..10 {
        tracker.record_outcome(&[0.0; 11], 0.01);
    }
    assert_eq!(tracker.total_counts, [0; 11]);
    assert_eq!(tracker.correct_counts, [0; 11]);
}

#[test]
fn valid_default_policy_preserves_both_directions() {
    let council = ConsejoDeliberacion::new();
    let p = snapshot();
    assert!(council.deliberar(&p, 0.7).approved);
    let q = MarketSnapshotPayload {
        book_imbalance: -0.85,
        ml_prob: 0.3,
        fused_score: -0.5,
        intended_direction: -1.0,
        ..p
    };
    let result = council.deliberar(&q, 0.7);
    assert!(result.approved);
    assert!(result.final_signal < 0.0);
}

#[test]
fn a_legitimate_market_veto_is_not_an_integrity_error() {
    let mut p = snapshot();
    p.liquidation_severity = 0.99;
    assert!(p.validate().is_ok());
    let result = ConsejoDeliberacion::new().deliberar(&p, 0.7);
    assert!(!result.approved);
    assert_eq!(result.vetoed_by, Some(SeniorRole::EnteMercado));
}

#[test]
fn net_trade_agreement_is_symmetric_for_long_and_short_wins_and_losses() {
    for is_long in [true, false] {
        for pnl in [-0.01, 0.01] {
            let council = ConsejoDeliberacion::new();
            let side = if is_long { 1.0 } else { -1.0 };
            let mut signals = [0.0; 11];
            signals[0] = side;
            signals[6] = -side;
            council.record_trade_outcome(&signals, pnl, is_long);
            let tracker = council.tracker.read().unwrap();
            assert_eq!(tracker.correct_counts[0], usize::from(pnl > 0.0));
            assert_eq!(tracker.correct_counts[6], usize::from(pnl < 0.0));
            assert_eq!(tracker.total_counts[0], 1);
            assert_eq!(tracker.total_counts[6], 1);
            assert_eq!(tracker.total_counts[2], 0);
        }
    }
}

#[test]
fn malformed_win_rate_is_an_integrity_rejection() {
    for wr in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let result = ConsejoDeliberacion::new().deliberar(&snapshot(), wr);
        assert!(!result.approved);
        assert!(result.dissenting_log[0].justification.contains("integrity"));
    }
}

#[test]
fn positive_extreme_scales_remain_valid_inputs_not_invalid_data() {
    for tau in [1e-6, 1.0, 30_000.0, 43_200_000.0, 3.15576e12] {
        let mut p = snapshot();
        p.dominant_tau_ms = tau;
        assert!(p.validate().is_ok());
    }
}

#[test]
fn cascade_parameter_boundary_matches_a_strict_greater_than_contract() {
    let mut council = ConsejoDeliberacion::new();
    council.params.cascade_severity_breaker = 0.5;
    let mut p = snapshot();
    p.liquidation_severity = 0.5;
    assert!(council.deliberar(&p, 0.7).approved);
    p.liquidation_severity = 0.500001;
    assert_eq!(
        council.deliberar(&p, 0.7).vetoed_by,
        Some(SeniorRole::EnteMercado)
    );
}

#[test]
fn zero_prior_concentration_is_invalid_without_an_explicit_no_prior_mode() {
    let mut council = ConsejoDeliberacion::new();
    council.params.shrinkage_k = 0.0;
    assert!(!council.deliberar(&snapshot(), 0.7).approved);
}

#[test]
fn open_one_raw_directional_source_can_still_look_like_full_consensus() {
    let p = MarketSnapshotPayload {
        ml_prob: 0.5,
        ml_model_base: 0.5,
        fused_score: 0.0,
        persistence: 0.0,
        hurst_exponent: 0.5,
        ..snapshot()
    };
    let result = ConsejoDeliberacion::new().deliberar(&p, 0.7);
    assert!(
        result.approved,
        "OPEN: one OBI source plus its derived meta vote suffices"
    );
    assert_eq!(result.total_consensus_pct, 1.0);
}

#[test]
fn open_extreme_horizons_collapse_to_the_same_legacy_endpoints() {
    let mut p = snapshot();
    p.dominant_tau_ms = 1e-6;
    assert_eq!(p.spectral_s(), 0.0);
    p.dominant_tau_ms = 30_000.0;
    assert_eq!(p.spectral_s(), 0.0);
    p.dominant_tau_ms = 43_200_000.0;
    assert_eq!(p.spectral_s(), 1.0);
    p.dominant_tau_ms = 3.15576e12;
    assert_eq!(p.spectral_s(), 1.0);
}

struct WinRateProbe;
impl SeniorAgent for WinRateProbe {
    fn role(&self) -> SeniorRole {
        SeniorRole::Microestructura
    }
    fn evaluate(&self, _: &MarketSnapshotPayload, wr: f64) -> SeniorOpinion {
        SeniorOpinion {
            role: self.role(),
            signal_direction: wr,
            confidence: 1.0,
            weight: 1.0,
            is_veto: false,
            justification: "synthetic WR probe".into(),
        }
    }
}

#[test]
fn open_aggregate_trial_count_changes_shrinkage_without_asset_identity() {
    let mut council = ConsejoDeliberacion::new();
    council.agents = vec![Box::new(WinRateProbe)];
    let p = snapshot();
    assert_eq!(council.deliberar(&p, 0.1).final_signal, 0.5);
    for _ in 0..10 {
        council.record_outcome(&[0.0; 11], 0.01);
    }
    let result = council.deliberar(&p, 0.1);
    assert!((result.final_signal - 5.0 / 18.0).abs() < 1e-12);
    // XXXII repairs the extraction path, but the global/per-asset mismatch
    // above remains OPEN. Keep this diagnostic until that population is fixed.
    assert!((council.extract_senior_signals(&p, 0.1)[0] - result.final_signal).abs() < 1e-12);
}

// XXXII: roles are identities, not positions in the mutable agents vector.
#[test]
fn reordering_agents_cannot_reassign_external_weights() {
    let a = ConsejoDeliberacion::new();
    let mut b = ConsejoDeliberacion::new();
    b.agents.reverse();
    let weights = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2];
    let x = a.deliberar_with_weights(&snapshot(), 0.7, Some(&weights));
    let y = b.deliberar_with_weights(&snapshot(), 0.7, Some(&weights));
    assert_eq!(x.approved, y.approved);
    assert!((x.final_signal - y.final_signal).abs() < 1e-12);
    assert!((x.total_consensus_pct - y.total_consensus_pct).abs() < 1e-12);
}

#[test]
fn signal_extraction_uses_roles_not_vector_order() {
    let a = ConsejoDeliberacion::new();
    let mut b = ConsejoDeliberacion::new();
    b.agents.reverse();
    assert_eq!(a.extract_senior_signals(&snapshot(), 0.7), b.extract_senior_signals(&snapshot(), 0.7));
}

#[test]
fn duplicate_roles_are_an_integrity_failure_not_extra_votes() {
    let mut council = ConsejoDeliberacion::new();
    council.agents.push(Box::new(WinRateProbe));
    let result = council.deliberar(&snapshot(), 0.7);
    assert!(!result.approved);
    assert!(result.dissenting_log[0].justification.contains("integrity"));
}

struct MalformedOpinion;
impl SeniorAgent for MalformedOpinion {
    fn role(&self) -> SeniorRole { SeniorRole::Microestructura }
    fn evaluate(&self, _: &MarketSnapshotPayload, _: f64) -> SeniorOpinion {
        SeniorOpinion { role: self.role(), signal_direction: 100.0, confidence: 1.0,
            weight: 1.0, is_veto: false, justification: "invalid agent output".into() }
    }
}

#[test]
fn an_invalid_agent_output_cannot_be_clamped_into_approval() {
    let mut council = ConsejoDeliberacion::new();
    council.agents = vec![Box::new(MalformedOpinion)];
    assert!(!council.deliberar(&snapshot(), 0.7).approved);
}

struct CountingAgent(std::sync::Arc<std::sync::atomic::AtomicUsize>);
impl SeniorAgent for CountingAgent {
    fn role(&self) -> SeniorRole { SeniorRole::Microestructura }
    fn evaluate(&self, _: &MarketSnapshotPayload, wr: f64) -> SeniorOpinion {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        SeniorOpinion { role: self.role(), signal_direction: wr, confidence: 1.0,
            weight: 1.0, is_veto: false, justification: "one-pass probe".into() }
    }
}

#[test]
fn traced_decision_evaluates_each_agent_once_with_the_effective_wr() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut council = ConsejoDeliberacion::new();
    council.agents = vec![Box::new(CountingAgent(calls.clone()))];
    let trace = council.deliberar_traced(&snapshot(), 0.1, None);
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(trace.effective_win_rate, Some(0.5));
    assert_eq!(trace.aggregate_outcome_count, 0);
    assert_eq!(trace.signals[0], trace.consensus.final_signal);
    assert_eq!(trace.signals[0], trace.opinions[0].signal_direction);
    assert_eq!(trace.opinions.len(), 1);
}

#[test]
fn invalid_snapshot_is_rejected_before_agent_callbacks() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut council = ConsejoDeliberacion::new();
    council.agents = vec![Box::new(CountingAgent(calls.clone()))];
    let mut p = snapshot();
    p.ml_prob = f64::NAN;
    let trace = council.deliberar_traced(&p, 0.7, None);
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(!trace.consensus.approved);
    assert_eq!(trace.signals, [0.0; 11]);
    assert!(trace.opinions.is_empty());
    assert_eq!(trace.effective_win_rate, None);
}

#[test]
fn trace_preserves_all_opinions_even_when_consensus_is_vetoed() {
    let mut p = snapshot();
    p.liquidation_severity = 0.99;
    let trace = ConsejoDeliberacion::new().deliberar_traced(&p, 0.7, None);
    assert!(!trace.consensus.approved);
    assert_eq!(trace.opinions.len(), 11);
    for op in &trace.opinions { assert_eq!(trace.signals[op.role as usize], op.signal_direction); }
    assert!(trace.opinions.iter().any(|o| o.role == SeniorRole::EnteMercado && o.is_veto));
}

#[test]
fn poisoned_tracker_is_not_silently_interpreted_as_cold_start() {
    let council = ConsejoDeliberacion::new();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = council.tracker.write().unwrap();
        panic!("synthetic tracker poison");
    }));
    let trace = council.deliberar_traced(&snapshot(), 0.7, None);
    assert!(!trace.consensus.approved);
    assert!(trace.consensus.dissenting_log[0].justification.contains("poisoned"));
}

struct OpinionProbe { declared: SeniorRole, opinion: SeniorOpinion }
impl SeniorAgent for OpinionProbe {
    fn role(&self) -> SeniorRole { self.declared }
    fn evaluate(&self, _: &MarketSnapshotPayload, _: f64) -> SeniorOpinion { self.opinion.clone() }
}

#[test]
fn every_opinion_numeric_contract_and_role_are_validated() {
    let good = SeniorOpinion { role: SeniorRole::Microestructura, signal_direction: 1.0,
        confidence: 1.0, weight: 1.0, is_veto: false, justification: "probe".into() };
    let edits: [fn(&mut SeniorOpinion); 9] = [
        |o| o.role = SeniorRole::Ml, |o| o.signal_direction = f64::NAN,
        |o| o.signal_direction = -1.1, |o| o.confidence = f64::INFINITY,
        |o| o.confidence = -0.1, |o| o.confidence = 1.1,
        |o| o.weight = f64::NAN, |o| o.weight = f64::INFINITY, |o| o.weight = -1.0,
    ];
    for edit in edits {
        let mut opinion = good.clone(); edit(&mut opinion);
        let mut council = ConsejoDeliberacion::new();
        council.agents = vec![Box::new(OpinionProbe { declared: good.role, opinion })];
        let trace = council.deliberar_traced(&snapshot(), 0.7, None);
        assert!(!trace.consensus.approved);
        assert_eq!(trace.signals, [0.0; 11]);
        assert!(trace.consensus.dissenting_log[0].justification.contains("integrity"));
    }
}

#[test]
fn finite_weights_that_overflow_when_scaled_do_not_approve() {
    let mut council = ConsejoDeliberacion::new();
    council.agents = vec![Box::new(OpinionProbe { declared: SeniorRole::Ml,
        opinion: SeniorOpinion { role: SeniorRole::Ml, signal_direction: 1.0,
            confidence: 1.0, weight: f64::MAX, is_veto: false, justification: "overflow".into() } })];
    let trace = council.deliberar_traced(&snapshot(), 0.7, Some(&[5.0; 11]));
    assert!(!trace.consensus.approved);
    assert!(trace.consensus.dissenting_log[0].justification.contains("overflow"));
}

#[test]
fn dynamic_weights_follow_roles_when_the_topology_is_reordered() {
    let a = ConsejoDeliberacion::new();
    let mut b = ConsejoDeliberacion::new(); b.agents.reverse();
    let mut signals = [0.0; 11]; signals[0] = 1.0; signals[6] = -1.0;
    for _ in 0..10 { a.record_outcome(&signals, 0.1); b.record_outcome(&signals, 0.1); }
    let x = a.deliberar_traced(&snapshot(), 0.7, None);
    let y = b.deliberar_traced(&snapshot(), 0.7, None);
    assert_eq!(x.signals, y.signals);
    assert!((x.consensus.final_signal - y.consensus.final_signal).abs() < 1e-12);
    for op in x.opinions {
        let other = y.opinions.iter().find(|o| o.role == op.role).unwrap();
        assert_eq!(op.weight, other.weight);
    }
}

#[test]
fn a_zero_weight_is_valid_but_cannot_create_directional_capacity() {
    let mut council = ConsejoDeliberacion::new();
    council.agents = vec![Box::new(OpinionProbe { declared: SeniorRole::Ml,
        opinion: SeniorOpinion { role: SeniorRole::Ml, signal_direction: 1.0,
            confidence: 1.0, weight: 0.0, is_veto: false, justification: "disabled vote".into() } })];
    let trace = council.deliberar_traced(&snapshot(), 0.7, None);
    assert!(!trace.consensus.approved);
    assert_eq!(trace.effective_win_rate, Some(0.5));
    assert_eq!(trace.opinions.len(), 1);
    assert_eq!(trace.consensus.vetoed_by, None);
    assert_eq!(trace.signals[SeniorRole::Ml as usize], 0.0);
}
