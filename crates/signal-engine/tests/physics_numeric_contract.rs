use signal_engine::{
    coaxial_breakout::CoaxialBreakoutEngine, supersonic_shockwave::SupersonicShockwaveEngine,
};

#[test]
fn no_variation_is_not_evidence_of_maximal_relative_squeeze() {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    for atr in [[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0], [0.0005, -1.0, 0.05]] {
        assert!(
            CoaxialBreakoutEngine::evaluate_coaxial_breakout(
                &arena, atr[0], atr[1], atr[2], 100.0, true
            )
            .is_none(),
            "ATR={atr:?}"
        );
    }
}

#[test]
fn bounded_jump_remains_finite_for_large_finite_mach() {
    for mach in [1e155, 1e200, f64::MAX] {
        let value = SupersonicShockwaveEngine::compute_shockwave_jump(mach);
        assert!(value.is_finite(), "mach={mach}");
        assert!((value - 1.0_f64.tanh()).abs() < 1e-14);
    }
}

#[test]
fn jump_matches_original_expression_where_squaring_is_safe() {
    for mach in [1.0 + f64::EPSILON, 1.01, 2.0, 10.0, 1e100] {
        let reference = ((mach * mach - 1.0) / (mach * mach + 1.0)).tanh();
        let actual = SupersonicShockwaveEngine::compute_shockwave_jump(mach);
        assert!((actual - reference).abs() < 1e-14);
    }
    assert_eq!(SupersonicShockwaveEngine::compute_shockwave_jump(1.0), 0.0);
}

#[test]
fn zero_short_scale_variation_with_positive_denominators_is_valid() {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    assert!(CoaxialBreakoutEngine::evaluate_coaxial_breakout(
        &arena, 0.0, 0.005, 0.05, 100.0, true
    )
    .is_some());
}
