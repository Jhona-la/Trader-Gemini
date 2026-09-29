//! Conformal wiring/domain regression tests. Synthetic local events; no exchange.
use god_engine_core::{GodEngineCore, conformal::ConformalCalibrator};
use quantum_arena::{GlobalArena, symbol_registry, symbols};
use std::sync::{Arc, Mutex, atomic::Ordering::Relaxed};

static CORE_ENV: Mutex<()> = Mutex::new(());

fn fixture() -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec!["CFMAUSDT".into(), "CFMBUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("CFMAUSDT"),
        symbol_registry::get_official_binance_spec("CFMBUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    (arena, core)
}

fn publish(core: &mut GodEngineCore, coin_id: usize) {
    publish_with_entries(core, coin_id, true);
}

fn publish_with_entries(core: &mut GodEngineCore, coin_id: usize, allow_entries: bool) {
    let (entry, close, _) = core.process_tick_dual(
        coin_id,
        100.0,
        100.01,
        5.0,
        5.0,
        10_000,
        &[0.0; 54],
        true,
        allow_entries,
    );
    assert!(
        entry.is_none() && close.is_none(),
        "isolated analytical event only"
    );
}

#[test]
fn genome_target_reaches_the_local_calibrator_not_just_global() {
    let _guard = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = fixture();
    arena.config.conformal_alpha.store(0.03, Relaxed);
    publish(&mut core, 0);
    assert_eq!(core.conformal_by_coin[0].effective_alpha(), 0.03);
    assert_eq!(
        core.conformal_by_coin[1].effective_alpha(),
        0.10,
        "other asset untouched"
    );
    arena.config.conformal_alpha.store(0.18, Relaxed);
    publish(&mut core, 1);
    assert_eq!(core.conformal_by_coin[1].effective_alpha(), 0.18);
    assert_eq!(core.conformal_by_coin[0].effective_alpha(), 0.03);
}

#[test]
fn effective_alpha_telemetry_describes_the_instance_that_decides() {
    let _guard = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = fixture();
    for _ in 0..50 {
        core.conformal_by_coin[0].update(0.9, true);
    }
    let local = core.conformal_by_coin[0].effective_alpha();
    assert_ne!(
        local,
        core.conformal.effective_alpha(),
        "distinct witnesses"
    );
    publish(&mut core, 0);
    assert_eq!(
        arena
            .registry
            .get_for_coin_or(0, "conformal_alpha_eff", -1.0),
        local
    );
    assert_eq!(
        arena
            .registry
            .get_scoped_value_or("CFMAUSDT", "conformal_alpha_eff", -1.0),
        local
    );
    let p = arena.coins[0].ml_prob.load(Relaxed);
    assert_eq!(
        arena
            .registry
            .get_for_coin_or(0, "conformal_accept_long", -1.0),
        f64::from(core.conformal_by_coin[0].accepts(p))
    );
    assert_eq!(
        arena.registry.get_for_coin_or(0, "conformal_p_value", -1.0),
        core.conformal_by_coin[0].p_value(p)
    );
}

#[test]
fn evolved_target_changes_next_local_adaptation_without_resetting_state() {
    let _guard = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = fixture();
    for _ in 0..50 {
        core.conformal_by_coin[0].update(0.9, true);
    }
    let previous = core.conformal_by_coin[0].effective_alpha();
    let count = core.conformal_by_coin[0].observations();
    arena.config.conformal_alpha.store(0.20, Relaxed);
    publish(&mut core, 0);
    assert_eq!(core.conformal_by_coin[0].effective_alpha(), previous);
    assert_eq!(core.conformal_by_coin[0].observations(), count);
    core.conformal_by_coin[0].update(0.9, true);
    assert!(
        (core.conformal_by_coin[0].effective_alpha() - (previous + 0.20 / 200.0)).abs() < 1e-12,
        "the local update must consume the new gene target"
    );
}

#[test]
fn explicit_legacy_global_fallback_remains_consistent() {
    let _guard = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = fixture();
    core.conformal_by_coin.clear();
    arena.config.conformal_alpha.store(0.07, Relaxed);
    publish(&mut core, 0);
    assert_eq!(core.conformal.effective_alpha(), 0.07);
    assert_eq!(
        arena
            .registry
            .get_for_coin_or(0, "conformal_alpha_eff", -1.0),
        0.07
    );
}

#[test]
fn invalid_predictions_do_not_enter_calibration_or_adaptation() {
    let mut c = ConformalCalibrator::new();
    for _ in 0..40 {
        c.update(0.9, true);
    }
    let before = (
        c.observations(),
        c.effective_alpha(),
        c.adaptive_error_rate(),
        c.p_value(0.6),
    );
    for p in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        c.update(p, false);
    }
    assert_eq!(
        (
            c.observations(),
            c.effective_alpha(),
            c.adaptive_error_rate(),
            c.p_value(0.6)
        ),
        before
    );
}

#[test]
fn warmup_is_not_permission_to_accept_an_invalid_probability() {
    let mut c = ConformalCalibrator::new();
    for p in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        assert!(
            !c.accepts(p),
            "invalid input {p} must not be called admissible"
        );
    }
    assert!(c.accepts(0.5), "valid warmup policy is preserved");
    for _ in 0..40 {
        c.update(0.9, true);
    }
    assert!(
        !c.accepts(1.1),
        "invalid warm probability is not clipped into acceptance"
    );
}

#[test]
fn endpoints_remain_valid_scores_and_empirical_rank_uses_inclusive_ties() {
    let mut c = ConformalCalibrator::new();
    for _ in 0..30 {
        c.update(1.0, true);
    }
    assert_eq!(c.observations(), 30);
    assert_eq!(c.p_value_for(1.0, true), 1.0);
    assert_eq!(c.p_value_for(1.0, false), 1.0 / 31.0);
    c.update(0.0, false);
    assert_eq!(c.observations(), 31);
    assert_eq!(c.p_value_for(0.0, false), 1.0);
}

#[test]
fn entry_interlock_preserves_conformal_analytics_and_still_blocks_entries() {
    let _guard = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = fixture();
    arena.config.conformal_alpha.store(0.07, Relaxed);
    publish_with_entries(&mut core, 0, false);
    assert_eq!(core.conformal_by_coin[0].effective_alpha(), 0.07);
    assert_eq!(
        arena
            .registry
            .get_for_coin_or(0, "conformal_alpha_eff", -1.0),
        0.07
    );
}

#[test]
fn invalid_prediction_rank_is_an_explicit_rejection_sentinel() {
    let mut c = ConformalCalibrator::new();
    for warm in [false, true] {
        if warm {
            for _ in 0..40 {
                c.update(0.9, true);
            }
        }
        for p in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
            for y in [false, true] {
                assert_eq!(c.p_value_for(p, y), 0.0, "invalid is not an empirical rank");
            }
        }
    }
}

#[test]
fn clipped_adaptation_does_not_inherit_the_unprojected_aci_tracking_bound() {
    // Diagnostic counterexample, not a guarantee of selected-trade risk.
    let mut c = ConformalCalibrator::new();
    for _ in 0..5_000 {
        c.update(0.5, true);
    }
    assert_eq!(c.effective_alpha(), 0.5);
    assert_eq!(c.adaptive_error_rate(), 0.0);
    assert_eq!(c.p_value_for(0.5, true), 1.0);
    assert_eq!(c.p_value_for(0.5, false), 1.0);
    assert!(!c.accepts(0.5), "the set always contains both labels");
    let unprojected_bound = (0.9 + 1.0 / 200.0) / ((1.0 / 200.0) * 4_970.0);
    assert!((c.adaptive_error_rate() - 0.1).abs() > unprojected_bound);
}

#[test]
fn finite_rank_resolution_can_block_all_singletons_until_new_feedback() {
    // Diagnostic of the existing policy, not a license to remove risk gates.
    let mut c = ConformalCalibrator::new();
    c.set_target_alpha(0.01);
    for _ in 0..200 {
        c.update(0.9, true);
    }
    for p in [0.2, 0.3, 0.4] {
        c.update(p, false);
    }
    let alpha = c.effective_alpha();
    let min_rank = 1.0 / (c.observations() as f64 + 1.0);
    assert!(
        alpha < min_rank,
        "alpha={alpha}, rank resolution={min_rank}"
    );
    for i in 0..=1_000 {
        let p = i as f64 / 1_000.0;
        assert!(c.p_value_for(p, false) >= min_rank);
        assert!(!c.accepts(p));
    }
    assert_eq!(
        c.effective_alpha(),
        alpha,
        "queries do not restore feedback"
    );
}
