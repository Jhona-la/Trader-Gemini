//! Synthetic LOCAL close-path tests: no exchange, model publication or live bus.
use god_engine_core::{GodEngineCore, outcome_context::OutcomeContext};
use quantum_arena::{GlobalArena, position::PositionHorizon, symbol_registry};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, atomic::Ordering},
};

static ENVIRONMENT: Mutex<()> = Mutex::new(());

struct FixtureDirectory {
    root: PathBuf,
    previous: PathBuf,
}
impl FixtureDirectory {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("tg-xxix-close-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(root.join("data")).unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&root).unwrap();
        Self { root, previous }
    }
}
impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.previous).unwrap();
        // Only the uniquely created, canonicalized test directory may be removed.
        let exact = self.root.canonicalize().unwrap();
        assert!(exact.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        assert!(
            exact
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("tg-xxix-close-")
        );
        std::fs::remove_dir_all(exact).unwrap();
    }
}

fn core(context: OutcomeContext, confirmed: bool) -> (Arc<GlobalArena>, GodEngineCore) {
    quantum_arena::symbols::update_dynamic_universe(vec!["XXIXUSDT".into(),"XXIYUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("XXIXUSDT"),
        symbol_registry::get_official_binance_spec("XXIYUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new_with_outcome_context(arena.clone(), context);
    core.swing_nn = None;
    core.scalp_forest = None;
    let pos = &arena.coins[0].positions.position;
    assert!(pos.open_with_horizon(
        true,
        100.0,
        1.0,
        10.0,
        1000,
        101.0,
        99.0,
        PositionHorizon::Continuous
    ));
    pos.exchange_confirmed.store(confirmed, Ordering::Relaxed);
    arena.used_margin.store(10.0, Ordering::Relaxed);
    (arena, core)
}

fn close(core: &mut GodEngineCore, bid: f64) -> f64 {
    close_at(core, 0, bid)
}

fn close_at(core: &mut GodEngineCore, coin_id: usize, bid: f64) -> f64 {
    let (entry, proposal, _) = core.process_tick_dual(
        coin_id,
        bid,
        bid + 0.01,
        5.0,
        5.0,
        2000,
        &[0.0; 54],
        true,
        false,
    );
    assert!(entry.is_none(), "entry veto remains in force");
    let (_, pnl, qty) = proposal.expect("defensive close still reaches the caller");
    assert_eq!(qty, 1.0);
    pnl
}

#[test]
fn unconfirmed_exchange_proposal_does_not_train_or_change_capital() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut core) = core(OutcomeContext::ExchangeLocalEstimate, false);
    let capital = arena.unified_capital.load(Ordering::Relaxed);
    let wr = arena.coins[0].metrics.win_rate.load(Ordering::Relaxed);
    close(&mut core, 101.1);
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), capital);
    assert_eq!(
        arena.coins[0].metrics.trade_count.load(Ordering::Relaxed),
        0
    );
    assert_eq!(arena.coins[0].metrics.win_rate.load(Ordering::Relaxed), wr);
    assert_eq!(core.ppo_engine.reward_ema.load(Ordering::Relaxed), 0.0);
    assert_eq!(core.diag_close_total, 0);
    assert_eq!(core.diag_unverified_close_total, 1);
    assert!(!arena.coins[0].positions.position.is_open());
}

#[test]
fn simulation_learns_locally_without_shared_dataset_or_trauma_files() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let dir = FixtureDirectory::new();
    let (arena, mut core) = core(OutcomeContext::IsolatedSimulation, false);
    *arena.coins[0]
        .positions
        .position
        .nn_entry_tensor
        .lock()
        .unwrap() = vec![0.25; 54];
    let pnl = close(&mut core, 97.0);
    assert!(pnl < 0.0);
    assert_eq!(core.diag_close_total, 1);
    assert_eq!(
        arena.coins[0].metrics.trade_count.load(Ordering::Relaxed),
        1
    );
    assert!(arena.unified_capital.load(Ordering::Relaxed) < 100.0);
    assert!(
        !dir.root
            .join("data/dark_alpha_dataset_XXIXUSDT.csv")
            .exists()
    );
    assert!(!dir.root.join("memoria").exists());
}

#[test]
fn unconfirmed_loss_cannot_write_training_data_or_trauma() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let dir = FixtureDirectory::new();
    let (arena, mut engine) = core(OutcomeContext::ExchangeLocalEstimate, false);
    *arena.coins[0]
        .positions
        .position
        .nn_entry_tensor
        .lock()
        .unwrap() = vec![0.25; 54];
    assert!(close(&mut engine, 97.0) < 0.0);
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 100.0);
    assert_eq!(engine.diag_unverified_close_total, 1);
    assert!(
        !dir.root
            .join("data/dark_alpha_dataset_XXIXUSDT.csv")
            .exists()
    );
    assert!(!dir.root.join("memoria").exists());
}

#[test]
fn missing_evidence_for_one_asset_does_not_veto_another_assets_local_estimate() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut engine) = core(OutcomeContext::ExchangeLocalEstimate, false);
    let other = &arena.coins[1].positions.position;
    assert!(other.open_with_horizon(
        true,
        100.0,
        1.0,
        10.0,
        1000,
        101.0,
        99.0,
        PositionHorizon::Continuous
    ));
    other.exchange_confirmed.store(true, Ordering::Relaxed);
    arena.used_margin.store(20.0, Ordering::Relaxed);
    close(&mut engine, 101.1);
    assert!(other.is_open(), "the close gate must be asset-local");
    let other_pnl = close_at(&mut engine, 1, 101.1);
    assert_eq!(
        arena.coins[0].metrics.trade_count.load(Ordering::Relaxed),
        0
    );
    assert_eq!(
        arena.coins[1].metrics.trade_count.load(Ordering::Relaxed),
        1
    );
    assert!((arena.unified_capital.load(Ordering::Relaxed) - 100.0 - other_pnl).abs() < 1e-12);
    assert_eq!(engine.diag_unverified_close_total, 1);
    assert_eq!(engine.diag_close_total, 1);
}

#[test]
fn one_close_updates_the_policy_reward_once() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (_, mut core) = core(OutcomeContext::IsolatedSimulation, false);
    let pnl = close(&mut core, 101.1);
    let reward = pnl / 100.0;
    let expected_once = 0.05 * reward;
    let actual = core.ppo_engine.reward_ema.load(Ordering::Relaxed);
    assert!(
        (actual - expected_once).abs() < 1e-14,
        "actual={actual} expected_once={expected_once}"
    );
}

#[test]
fn provenance_matrix_is_not_a_market_regime() {
    for (context, confirmed, local, shared) in [
        (OutcomeContext::IsolatedSimulation, false, true, false),
        (OutcomeContext::IsolatedSimulation, true, true, false),
        (OutcomeContext::ExchangeLocalEstimate, false, false, false),
        (OutcomeContext::ExchangeLocalEstimate, true, true, true),
    ] {
        assert_eq!(context.permits_local_learning(confirmed), local);
        assert_eq!(context.permits_shared_estimate_outputs(confirmed), shared);
    }
}

#[test]
fn default_constructor_is_isolated_even_if_entry_flag_is_true() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let dir = FixtureDirectory::new();
    let (arena, _) = core(OutcomeContext::IsolatedSimulation, true);
    let mut engine = GodEngineCore::new(arena.clone());
    engine.swing_nn = None;
    engine.scalp_forest = None;
    *arena.coins[0]
        .positions
        .position
        .nn_entry_tensor
        .lock()
        .unwrap() = vec![0.25; 54];
    assert!(
        !dir.root.join("memoria").exists(),
        "constructor has no trauma-directory effect"
    );
    close(&mut engine, 97.0);
    assert_eq!(engine.diag_close_total, 1);
    assert!(
        !dir.root
            .join("data/dark_alpha_dataset_XXIXUSDT.csv")
            .exists()
    );
    assert!(!dir.root.join("memoria").exists());
}

#[test]
fn confirmed_exchange_entry_preserves_the_legacy_local_estimate_path() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut engine) = core(OutcomeContext::ExchangeLocalEstimate, true);
    let pnl = close(&mut engine, 101.1);
    assert!((arena.unified_capital.load(Ordering::Relaxed) - 100.0 - pnl).abs() < 1e-12);
    assert_eq!(engine.diag_close_total, 1);
    assert_eq!(engine.diag_unverified_close_total, 0);
    assert!(
        arena.coins[0]
            .positions
            .position
            .last_close_confirmed
            .load(Ordering::Relaxed)
    );
    // This is intentionally NOT proof of an exit fill: the API still returns an estimate.
}

#[test]
fn close_kelly_retains_the_trade_horizon_instead_of_using_cleared_slot() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut engine) = core(OutcomeContext::IsolatedSimulation, false);
    let tau = 1_000_000_000_u64;
    arena.coins[0]
        .positions
        .position
        .entry_tau_ms
        .store(tau, Ordering::Relaxed);
    // A continuous curve with 0.2 at 30s and 0.6 at the actual horizon.
    let b = 3.0_f64.ln() / ((tau as f64).ln() - 30_000.0_f64.ln());
    arena
        .config
        .kelly_curve_a
        .store(0.2_f64.ln() - b * 30_000.0_f64.ln(), Ordering::Relaxed);
    arena.config.kelly_curve_b.store(b, Ordering::Relaxed);
    arena
        .config
        .kelly_survival_cap_ratio
        .store(2.0, Ordering::Relaxed);
    arena.config.kelly_clamp_min.store(0.0, Ordering::Relaxed);
    arena.config.kelly_clamp_max.store(1.0, Ordering::Relaxed);
    let metrics = &arena.coins[0].metrics;
    metrics.trade_count.store(100, Ordering::Relaxed);
    metrics.win_rate.store(0.5, Ordering::Relaxed);
    metrics.gross_wins.store(1.3, Ordering::Relaxed);
    metrics.gross_losses.store(1.0, Ordering::Relaxed);
    close(&mut engine, 101.1);
    let spec = &engine.temporal_spectrum[0];
    let spectral_conf =
        (0.5 + spec.persistence_at(spec.dominant_tau_ms).clamp(-1.0, 1.0) * 0.5).clamp(0.0, 1.0);
    let expected_at = |horizon| {
        risk_engine::kelly::calculate_kelly_fraction(
            metrics.win_rate.load(Ordering::Relaxed),
            metrics.profit_factor.load(Ordering::Relaxed),
            arena.unified_capital.load(Ordering::Relaxed),
            arena.config.base_capital.load(Ordering::Relaxed),
            arena
                .config
                .kelly_survival_cap_ratio
                .load(Ordering::Relaxed),
            arena.config.kelly_expansion_mult.load(Ordering::Relaxed),
            0.0,
            1.0,
            arena.config.kelly_at_tau(horizon),
            spectral_conf,
        )
    };
    let expected = expected_at(tau as f64);
    let wrong_fallback = expected_at(30_000.0);
    assert!(
        (expected - wrong_fallback).abs() > 1e-4,
        "fixture must detect the lost horizon"
    );
    assert_eq!(
        arena.coins[0]
            .positions
            .position
            .entry_tau_ms
            .load(Ordering::Relaxed),
        0
    );
    let actual = metrics.kelly_fraction.load(Ordering::Relaxed);
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}, cleared-slot fallback={wrong_fallback}"
    );
}

/// OPEN FMT-232: passing means the defensive-policy gap is still reproduced.
#[test]
fn open_kill_switch_blocks_even_a_local_stop_close_proposal() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut engine) = core(OutcomeContext::IsolatedSimulation, false);
    arena.kill_switch_active.store(true, Ordering::Relaxed);
    let (entry, proposal, _) =
        engine.process_tick_dual(0, 97.0, 97.01, 5.0, 5.0, 2000, &[0.0; 54], true, false);
    assert!(entry.is_none());
    assert!(
        proposal.is_none(),
        "known limitation: kill switch suppresses local defense too"
    );
    assert!(arena.coins[0].positions.position.is_open());
    // Only this fixture's latch is changed; no operational flag is touched.
    arena.kill_switch_active.store(false, Ordering::Relaxed);
    assert!(close(&mut engine, 97.0) < 0.0);
    assert!(!arena.coins[0].positions.position.is_open());
}

#[test]
fn drift_entry_veto_preserves_local_defensive_close_and_blocks_new_entry() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut engine) = core(OutcomeContext::IsolatedSimulation, false);
    engine.set_drift_entry_veto(true);
    let (entry, close, maker) = engine.process_tick_dual(
        0, 97.0, 97.01, 5.0, 5.0, 2000, &[0.0; 54], true, true,
    );
    assert!(entry.is_none());
    assert!(maker.is_none());
    assert!(close.is_some());
    assert!(!arena.coins[0].positions.position.is_open());
    assert!(engine.drift_entry_veto());
    assert!(!arena.kill_switch_active.load(Ordering::Relaxed));
}

#[test]
fn releasing_drift_cannot_clear_an_independent_global_risk_latch() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut engine) = core(OutcomeContext::IsolatedSimulation, false);
    engine.set_drift_entry_veto(true);
    arena.kill_switch_active.store(true, Ordering::SeqCst);
    engine.set_drift_entry_veto(false);
    assert!(arena.kill_switch_active.load(Ordering::SeqCst));
    let (entry, _, maker) = engine.process_tick_dual(
        0, 97.0, 97.01, 5.0, 5.0, 2000, &[0.0; 54], true, true,
    );
    assert!(entry.is_none());
    assert!(maker.is_none());
}

/// XXXVII: the unscoped legacy adapter has no authority over manually opened slots.
#[test]
fn unowned_symbol_rollback_preserves_all_slots_and_margin() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, engine) = core(OutcomeContext::IsolatedSimulation, false);
    let other = &arena.coins[0].positions.slots()[0];
    assert!(other.open_with_horizon(true, 100.0, 2.0, 20.0, 900,
        101.0, 99.0, PositionHorizon::Continuous));
    other.exchange_confirmed.store(true, Ordering::Relaxed);
    arena.used_margin.store(30.0, Ordering::Relaxed);
    engine.rollback_position(0);
    assert!(other.is_open());
    assert!(arena.coins[0].positions.position.is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 30.0);
}

#[test]
fn winning_short_rewards_the_councils_short_prediction_not_its_opposite() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut core) = core(OutcomeContext::IsolatedSimulation, false);
    // Modify only this fixture's already-open position; the live account is untouched.
    let pos = &arena.coins[0].positions.position;
    pos.is_long.store(false, Ordering::Relaxed);
    pos.tp_price.store(99.0, Ordering::Relaxed);
    pos.sl_price.store(101.0, Ordering::Relaxed);
    // positions.position is slot 2 in positions.slots(), not slot 0.
    core.last_senior_signals[0][2][0] = -1.0;
    core.last_senior_signals[0][2][6] = 1.0;
    core.council_entry_evidence[0][2] = Some(god_engine_core::CouncilEntryEvidence {
        symbol: "XXIXUSDT".into(), generation: pos.generation.load(Ordering::Acquire),
        is_long: false, signals: core.last_senior_signals[0][2],
    });
    assert!(close(&mut core, 98.8) > 0.0);
    let tracker = core.consejo_deliberacion.tracker.read().unwrap();
    assert_eq!(tracker.total_counts[0], 1);
    assert_eq!(
        tracker.correct_counts[0], 1,
        "profitable short must reward bearish agreement"
    );
    assert_eq!(
        tracker.correct_counts[6], 0,
        "opposite bullish vote was not the winning action"
    );
}

#[test]
fn a_position_without_a_bound_council_decision_does_not_train_the_council() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (_, mut engine) = core(OutcomeContext::IsolatedSimulation, false);
    // Legacy cached votes are not evidence that this position was admitted.
    engine.last_senior_signals[0][2][0] = 1.0;
    assert!(close(&mut engine, 101.1) > 0.0);
    assert_eq!(engine.consejo_deliberacion.tracker.read().unwrap().total_outcomes(), 0);
    assert_eq!(engine.diag_unattributed_council_closes, 1);
}

fn bind_council_votes(engine: &mut GodEngineCore, generation: u64, is_long: bool, symbol: &str) {
    let mut signals = [0.0; 11]; signals[0] = if is_long { 1.0 } else { -1.0 };
    engine.council_entry_evidence[0][2] = Some(god_engine_core::CouncilEntryEvidence {
        symbol: symbol.into(), generation, is_long, signals,
    });
}

#[test]
fn bound_votes_are_consumed_once_and_do_not_survive_slot_reuse() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut engine) = core(OutcomeContext::IsolatedSimulation, false);
    let pos = &arena.coins[0].positions.position;
    bind_council_votes(&mut engine, pos.generation.load(Ordering::Acquire), true, "XXIXUSDT");
    assert!(close(&mut engine, 101.1) > 0.0);
    assert_eq!(engine.consejo_deliberacion.tracker.read().unwrap().total_outcomes(), 1);
    assert!(engine.council_entry_evidence[0][2].is_none());
    assert!(pos.open_with_horizon(true, 100.0, 1.0, 10.0, 1000, 101.0, 99.0, PositionHorizon::Continuous));
    assert!(close(&mut engine, 101.1) > 0.0);
    assert_eq!(engine.consejo_deliberacion.tracker.read().unwrap().total_outcomes(), 1);
    assert_eq!(engine.diag_unattributed_council_closes, 1);
}

#[test]
fn wrong_generation_side_or_symbol_cannot_receive_credit() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    for (generation_offset, side, symbol) in [(1, true, "XXIXUSDT"), (0, false, "XXIXUSDT"), (0, true, "XXIYUSDT")] {
        let (arena, mut engine) = core(OutcomeContext::IsolatedSimulation, false);
        let generation = arena.coins[0].positions.position.generation.load(Ordering::Acquire);
        bind_council_votes(&mut engine, generation + generation_offset, side, symbol);
        assert!(close(&mut engine, 101.1) > 0.0);
        assert_eq!(engine.consejo_deliberacion.tracker.read().unwrap().total_outcomes(), 0);
        assert!(engine.council_entry_evidence[0][2].is_none());
        assert_eq!(engine.diag_unattributed_council_closes, 1);
    }
}

#[test]
fn unconfirmed_exchange_close_consumes_binding_without_training() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut engine) = core(OutcomeContext::ExchangeLocalEstimate, false);
    bind_council_votes(&mut engine, arena.coins[0].positions.position.generation.load(Ordering::Acquire), true, "XXIXUSDT");
    assert!(close(&mut engine, 101.1) > 0.0);
    assert_eq!(engine.consejo_deliberacion.tracker.read().unwrap().total_outcomes(), 0);
    assert!(engine.council_entry_evidence[0][2].is_none());
    assert_eq!(engine.diag_unverified_close_total, 1);
}

#[test]
fn legacy_signal_mirror_cannot_overwrite_the_frozen_binding() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut engine) = core(OutcomeContext::IsolatedSimulation, false);
    bind_council_votes(&mut engine, arena.coins[0].positions.position.generation.load(Ordering::Acquire), true, "XXIXUSDT");
    engine.last_senior_signals[0][2] = [-1.0; 11];
    assert!(close(&mut engine, 101.1) > 0.0);
    assert_eq!(engine.consejo_deliberacion.tracker.read().unwrap().correct_counts[0], 1);
}

#[test]
fn core_does_not_consume_unscoped_process_global_liquidation_evidence() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (_, mut engine) = core(OutcomeContext::IsolatedSimulation, false);
    god_engine_core::liquidation_feed::take_pending();
    god_engine_core::liquidation_feed::bump(0.95);
    engine.process_tick_dual(0, 100.0, 100.01, 5.0, 5.0, 2000, &[0.0; 54], false, false);
    assert_eq!(engine.feature_engines[0].dark_alpha.current_severity, 0.0);
    assert_eq!(god_engine_core::liquidation_feed::take_pending(), 0.95);
}

fn liquidation(symbol: &str, time: u64) -> god_engine_core::liquidation_feed::LiquidationObservation {
    god_engine_core::liquidation_feed::LiquidationObservation {
        symbol: symbol.into(), event_time_ms: time, trade_time_ms: time-1,
        is_buy: false, reported_filled_notional: 1e6,
    }
}

#[test]
fn liquidation_scope_is_per_symbol_and_per_core_instance() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (_, mut a) = core(OutcomeContext::IsolatedSimulation, false);
    let (_, b) = core(OutcomeContext::IsolatedSimulation, false);
    a.observe_liquidation(liquidation("XXIXUSDT",1000)).unwrap();
    assert_eq!(a.liquidation_severity_at(0,1000),Ok(Some(1.0)));
    assert_eq!(a.liquidation_severity_at(1,1000),Ok(None));
    assert_eq!(b.liquidation_severity_at(0,1000),Ok(None));
    a.process_tick_dual(1,100.0,100.01,5.0,5.0,2000,&[0.0;54],false,false);
    assert_eq!(a.feature_engines[1].dark_alpha.current_severity,0.0);
    assert!(a.observe_liquidation(liquidation("UNKNOWNUSDT",1000)).is_err());
    assert_eq!(a.liquidation_diagnostics.unknown_symbol,1);
}

#[test]
fn repeated_ticks_do_not_consume_or_readd_snapshot_and_depth_uses_same_view() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (_, mut c) = core(OutcomeContext::IsolatedSimulation, false);
    c.observe_liquidation(liquidation("XXIXUSDT",1000)).unwrap();
    let expected=c.liquidation_severity_at(0,2000).unwrap().unwrap();
    for _ in 0..2 {
        c.process_tick_dual(0,100.0,100.01,5.0,5.0,2000,&[0.0;54],false,false);
        assert_eq!(c.feature_engines[0].dark_alpha.current_severity,expected);
        assert_eq!(c.liquidation_severity_at(0,2000),Ok(Some(expected)));
    }
    c.process_event(0,false,false,true,100.005,0.0,100.0,100.01,5.0,5.0,0.0,0.0,2000,true,&[0.0;54],false);
    assert_eq!(c.feature_engines[0].dark_alpha.current_severity,expected);
    assert_eq!(c.liquidation_severity_at(0,11000),Ok(Some(0.5)));
}

#[test]
fn future_liquidation_asof_rejects_new_entry_evidence_but_does_not_block_defensive_close() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (_, mut c) = core(OutcomeContext::IsolatedSimulation, false);
    c.observe_liquidation(liquidation("XXIXUSDT",3000)).unwrap();
    assert!(c.liquidation_severity_at(0,2000).is_err());
    assert!(close(&mut c,97.0) < 0.0);
    assert_eq!(c.feature_engines[0].dark_alpha.current_severity,0.0);
    assert_eq!(c.liquidation_diagnostics.invalid_as_of,1);
}

#[test]
fn active_universe_slot_reassignment_cannot_relabel_liquidation_evidence() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (_, mut c) = core(OutcomeContext::IsolatedSimulation, false);
    c.observe_liquidation(liquidation("XXIXUSDT",1000)).unwrap();
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("REPLACEMENTUSDT")]);
    quantum_arena::symbols::update_dynamic_universe(vec!["REPLACEMENTUSDT".into()]);
    assert_eq!(c.liquidation_severity_at(0,1000),Ok(None));
    assert!(c.observe_liquidation(liquidation("XXIXUSDT",2000)).is_err());
    c.observe_liquidation(liquidation("REPLACEMENTUSDT",2000)).unwrap();
    assert_eq!(c.liquidation_severity_at(0,2000),Ok(Some(1.0)));
}

#[test]
fn case_insensitive_registry_lookup_keeps_canonical_symbol_binding() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (_, mut c) = core(OutcomeContext::IsolatedSimulation, false);
    c.observe_liquidation(liquidation("xxixusdt",1000)).unwrap();
    assert_eq!(c.liquidation_severity_at(0,1000),Ok(Some(1.0)));
}

#[test]
fn observation_diagnostics_distinguish_duplicate_older_conflict_and_invalid() {
    use god_engine_core::liquidation_feed::ObservationUpdate as Update;
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (_, mut c) = core(OutcomeContext::IsolatedSimulation, false);
    assert_eq!(c.observe_liquidation(liquidation("XXIXUSDT",1000)),Ok(Update::Accepted));
    assert_eq!(c.observe_liquidation(liquidation("XXIXUSDT",1000)),Ok(Update::Duplicate));
    assert_eq!(c.observe_liquidation(liquidation("XXIXUSDT",999)),Ok(Update::Older));
    let mut conflict=liquidation("XXIXUSDT",1000); conflict.is_buy=true;
    assert_eq!(c.observe_liquidation(conflict),Ok(Update::ConflictingTimestamp));
    let mut invalid=liquidation("XXIXUSDT",2000); invalid.reported_filled_notional=f64::NAN;
    assert!(c.observe_liquidation(invalid).is_err());
    let d=&c.liquidation_diagnostics;
    assert_eq!((d.accepted,d.duplicates,d.older,d.conflicting,d.invalid),(1,1,1,1,1));
}
