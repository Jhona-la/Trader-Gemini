//! Local synthetic outcome contracts: no exchange or model promotion.
use god_engine_core::{
    GodEngineCore,
    ensemble::{ModelEnsemble, ModelId, SKILL_SPAN_BARS, SkillTracker},
};
use quantum_arena::{GlobalArena, position::PositionHorizon, symbol_registry, symbols};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, atomic::Ordering::Relaxed},
};

static ENV: Mutex<()> = Mutex::new(());
struct Sandbox {
    path: PathBuf,
    previous: PathBuf,
}
impl Sandbox {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "tg-outcome-contract-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&path).unwrap();
        Self { path, previous }
    }
}
impl Drop for Sandbox {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.previous).unwrap();
        let exact = self.path.canonicalize().unwrap();
        assert!(exact.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        assert!(
            exact
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("tg-outcome-contract-")
        );
        std::fs::remove_dir_all(exact).unwrap();
    }
}
fn fixture() -> (Arc<GlobalArena>, GodEngineCore) {
    fixture_with_context(god_engine_core::outcome_context::OutcomeContext::IsolatedSimulation)
}
fn fixture_with_context(
    context: god_engine_core::outcome_context::OutcomeContext,
) -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec!["OCAUSDT".into(), "OCBUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("OCAUSDT"),
        symbol_registry::get_official_binance_spec("OCBUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new_with_outcome_context(arena.clone(), context);
    core.swing_nn = None;
    core.scalp_forest = None;
    open_slot(&arena, 0, 2);
    (arena, core)
}
fn open_slot(arena: &GlobalArena, coin: usize, slot: usize) {
    assert!(
        arena.coins[coin]
            .positions
            .get_slot(slot)
            .open_with_horizon(
                true,
                100.0,
                1.0,
                10.0,
                1_000,
                101.0,
                99.0,
                PositionHorizon::Continuous
            )
    );
    arena.used_margin.fetch_add(10.0, Relaxed);
}
fn close(core: &mut GodEngineCore, coin: usize) {
    let (entry, exit, _) = core.process_tick_dual(
        coin, 101.1, 101.11, 5.0, 5.0, 2_000, &[0.0; 54], true, false,
    );
    assert!(entry.is_none());
    assert!(
        exit.is_some(),
        "a defensive local close must still be returned"
    );
}
fn opinions(e: &mut ModelEnsemble) {
    e.submit(ModelId::MotorForest, 0.9);
    e.submit(ModelId::DarkAlphaNN, 0.1);
}
#[test]
fn unbound_position_does_not_grade_recent_model_predictions() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _dir = Sandbox::new();
    let (_, mut core) = fixture();
    opinions(&mut core.ensembles[0]);
    let before = core.ensembles[0].log_briers();
    close(&mut core, 0);
    assert_eq!(
        core.ensembles[0].log_briers(),
        before,
        "recent votes are not entry evidence"
    );
}
#[test]
fn unbound_position_cannot_train_from_a_legacy_branch_mirror() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _dir = Sandbox::new();
    let (_, mut core) = fixture();
    core.rama_abierta[0] = Some(7);
    close(&mut core, 0);
    assert_eq!(core.rama_registro[0][7].n, 0);
}
#[test]
fn unbound_position_cannot_train_from_a_legacy_forest_mirror() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _dir = Sandbox::new();
    let (_, mut core) = fixture();
    core.bosque_voto_abierto[0] = Some(true);
    close(&mut core, 0);
    assert_eq!(core.bosque_registro[0].n, 0);
}
#[test]
fn invalid_predictions_are_absent_not_clamped_or_imputed() {
    for p in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        let mut e = ModelEnsemble::new();
        e.submit(ModelId::MotorForest, 0.8);
        e.submit(ModelId::MotorForest, p);
        assert_eq!(
            e.combined(),
            None,
            "invalid {p} cannot masquerade as a current vote"
        );
    }
}
#[test]
fn invalid_bar_labels_do_not_mutate_weights_or_consume_predictions() {
    for y in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 2.0, 0.5] {
        let mut e = ModelEnsemble::new();
        opinions(&mut e);
        let before = (e.log_briers(), e.combined());
        e.update_with_outcome(y);
        assert_eq!((e.log_briers(), e.combined()), before, "invalid label {y}");
    }
}
#[test]
fn invalid_trade_returns_do_not_mutate_weights() {
    for r in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut e = ModelEnsemble::new();
        opinions(&mut e);
        let before = e.log_briers();
        e.update_with_trade_outcome(true, r);
        assert_eq!(e.log_briers(), before, "invalid return {r}");
    }
}
#[test]
fn skill_tracker_rejects_impossible_probabilities_and_nonbinary_labels() {
    for (p, y) in [(-0.1, 1.0), (1.1, 0.0), (0.9, -1.0), (0.9, 2.0), (0.9, 0.5)] {
        let mut s = SkillTracker::default();
        for i in 0..(SKILL_SPAN_BARS as usize * 3) {
            s.record(
                if i % 2 == 0 { 0.8 } else { 0.2 },
                if i % 2 == 0 { 1.0 } else { 0.0 },
            );
        }
        let z = s.z();
        assert!(z.is_some());
        s.record(p, y);
        assert_eq!(s.z(), z);
    }
}
#[test]
fn endpoint_predictions_remain_valid_under_existing_numerical_regularization() {
    let mut e = ModelEnsemble::new();
    e.submit(ModelId::MotorForest, 0.0);
    assert_eq!(e.combined(), Some(0.001));
    e.submit(ModelId::MotorForest, 1.0);
    assert_eq!(e.combined(), Some(0.999));
}
#[test]
fn valid_bar_updates_still_reward_the_better_prediction() {
    let mut e = ModelEnsemble::new();
    opinions(&mut e);
    e.update_with_outcome(1.0);
    assert!(e.weights()[0] > e.weights()[1]);
    assert_eq!(e.combined(), None);
}

#[test]
fn a_sole_participating_model_is_not_erased_by_an_absent_models_weight() {
    let mut e = ModelEnsemble::new();
    e.submit(ModelId::MotorForest, 0.0);
    for _ in 0..2_000 {
        e.update_with_trade_outcome(true, 0.01);
    }
    assert_eq!(
        e.combined(),
        Some(0.001),
        "normalize over participating models"
    );
}

fn bind(
    core: &mut GodEngineCore,
    arena: &GlobalArena,
    coin: usize,
    slot: usize,
    branch: usize,
    up: bool,
) {
    let position = arena.coins[coin].positions.get_slot(slot);
    core.entry_learning_evidence[coin][slot] = Some(god_engine_core::EntryLearningEvidence {
        symbol: if coin == 0 { "OCAUSDT" } else { "OCBUSDT" }.into(),
        generation: position.generation.load(Relaxed),
        is_long: true,
        decision_time_ms: position.entry_time_ms.load(Relaxed),
        branch: Some(branch),
        forest_vote: Some(up),
        model_predictions: if up {
            [Some(0.9), Some(0.1)]
        } else {
            [Some(0.1), Some(0.9)]
        },
    });
}

#[test]
fn bound_close_grades_opening_not_recent_or_legacy_votes_and_only_once() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _dir = Sandbox::new();
    let (arena, mut core) = fixture();
    bind(&mut core, &arena, 0, 2, 7, true);
    core.rama_abierta[0] = Some(8);
    core.bosque_voto_abierto[0] = Some(false);
    core.ensembles[0].submit(ModelId::MotorForest, 0.1);
    core.ensembles[0].submit(ModelId::DarkAlphaNN, 0.9);
    close(&mut core, 0);
    assert_eq!(core.rama_registro[0][7].n, 1);
    assert_eq!(core.rama_registro[0][8].n, 0);
    assert_eq!(
        (core.bosque_registro[0].n, core.bosque_registro[0].aciertos),
        (1, 1)
    );
    assert!(core.ensembles[0].weights()[0] > core.ensembles[0].weights()[1]);
    assert!(core.entry_learning_evidence[0][2].is_none());
    let weights = core.ensembles[0].log_briers();
    open_slot(&arena, 0, 2);
    close(&mut core, 0);
    assert_eq!(core.ensembles[0].log_briers(), weights);
    assert_eq!(core.rama_registro[0][7].n, 1);
    assert_eq!(core.bosque_registro[0].n, 1);
}

#[test]
fn independent_slots_and_assets_keep_their_own_pending_evidence() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _dir = Sandbox::new();
    let (arena, mut core) = fixture();
    open_slot(&arena, 0, 0);
    open_slot(&arena, 1, 2);
    bind(&mut core, &arena, 0, 2, 7, true);
    bind(&mut core, &arena, 0, 0, 8, false);
    bind(&mut core, &arena, 1, 2, 9, true);
    close(&mut core, 0); // slot zero closes first, although its evidence was bound later
    assert!(core.entry_learning_evidence[0][0].is_none());
    assert!(core.entry_learning_evidence[0][2].is_some());
    assert_eq!(core.rama_registro[0][8].n, 1);
    assert_eq!(core.rama_registro[0][7].n, 0);
    assert_eq!(
        (core.bosque_registro[0].n, core.bosque_registro[0].aciertos),
        (1, 0)
    );
    assert_eq!(core.ensembles[1].log_briers(), [0.0, 0.0]);
    close(&mut core, 0);
    assert_eq!(core.rama_registro[0][7].n, 1);
    assert_eq!(
        (core.bosque_registro[0].n, core.bosque_registro[0].aciertos),
        (2, 1)
    );
    assert!(core.entry_learning_evidence[1][2].is_some());
    close(&mut core, 1);
    assert_eq!(core.rama_registro[1][9].n, 1);
    assert_eq!(core.rama_registro[0][9].n, 0);
}

#[test]
fn identity_mismatch_is_consumed_without_training() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _dir = Sandbox::new();
    for mismatch in 0..4 {
        let (arena, mut core) = fixture();
        bind(&mut core, &arena, 0, 2, 7, true);
        let e = core.entry_learning_evidence[0][2].as_mut().unwrap();
        match mismatch {
            0 => e.symbol = "OCBUSDT".into(),
            1 => e.generation += 1,
            2 => e.is_long = false,
            _ => e.decision_time_ms += 1,
        }
        close(&mut core, 0);
        assert!(core.entry_learning_evidence[0][2].is_none());
        assert_eq!(core.ensembles[0].log_briers(), [0.0, 0.0]);
        assert_eq!(core.rama_registro[0][7].n, 0);
        assert_eq!(core.bosque_registro[0].n, 0);
    }
}

#[test]
fn event_before_entry_cannot_train_even_with_matching_identity() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _dir = Sandbox::new();
    let (arena, mut core) = fixture();
    arena.coins[0]
        .positions
        .get_slot(2)
        .entry_time_ms
        .store(3_000, Relaxed);
    bind(&mut core, &arena, 0, 2, 7, true);
    close(&mut core, 0);
    assert!(core.entry_learning_evidence[0][2].is_none());
    assert_eq!(core.ensembles[0].log_briers(), [0.0, 0.0]);
    assert_eq!(core.rama_registro[0][7].n, 0);
}

#[test]
fn unconfirmed_exchange_close_consumes_evidence_without_learning() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _dir = Sandbox::new();
    let (arena, mut core) = fixture_with_context(
        god_engine_core::outcome_context::OutcomeContext::ExchangeLocalEstimate,
    );
    bind(&mut core, &arena, 0, 2, 7, true);
    close(&mut core, 0);
    assert!(core.entry_learning_evidence[0][2].is_none());
    assert_eq!(core.ensembles[0].log_briers(), [0.0, 0.0]);
    assert_eq!(core.rama_registro[0][7].n, 0);
    assert_eq!(core.bosque_registro[0].n, 0);
    assert_eq!(core.diag_unverified_close_total, 1);
}

#[test]
fn snapshot_api_grades_frozen_predictions_and_does_not_consume_current_event() {
    let mut e = ModelEnsemble::new();
    opinions(&mut e);
    let frozen = e.prediction_snapshot();
    e.update_with_outcome(1.0); // resetting the current bar does not erase frozen evidence
    assert_eq!(e.prediction_snapshot(), [None, None]);
    e.submit(ModelId::MotorForest, 0.1);
    e.submit(ModelId::DarkAlphaNN, 0.9);
    let current = e.prediction_snapshot();
    let before = e.log_briers();
    e.update_with_trade_snapshot(frozen, true, 0.01);
    let after = e.log_briers();
    assert!((after[0] - before[0] + 0.75 * 0.01).abs() < 1e-12);
    assert!((after[1] - before[1] + 0.75 * 0.81).abs() < 1e-12);
    assert_eq!(e.prediction_snapshot(), current);
}

#[test]
fn invalid_snapshot_component_cannot_poison_other_models() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        let mut e = ModelEnsemble::new();
        e.update_with_trade_snapshot([Some(invalid), Some(0.9)], true, 0.01);
        assert_eq!(e.log_briers()[0], 0.0);
        assert!(e.log_briers()[1].is_finite());
        assert!(e.log_briers()[1] < 0.0);
    }
}

#[test]
fn later_invalid_vote_does_not_erase_first_valid_bar_evidence() {
    let mut e = ModelEnsemble::new();
    e.submit(ModelId::MotorForest, 0.9);
    e.submit(ModelId::MotorForest, f64::NAN);
    assert_eq!(e.prediction_snapshot(), [None, None]);
    e.update_with_outcome(1.0);
    assert!((e.log_briers()[0] + 0.05 * 0.01).abs() < 1e-12);
}
