//! LG5 integration: real core state roundtrips, symbol remapping and the local
//! close eligibility gate. Fixtures are synthetic and isolated from live files.
use god_engine_core::calibration::WINDOW;
use god_engine_core::estado_aprendido::{EstadoAprendido, validar};
use god_engine_core::net_payoff::{EntryPayoffEvidence, source_for};
use god_engine_core::outcome_context::OutcomeContext;
use god_engine_core::{EntryLearningEvidence, GodEngineCore};
use quantum_arena::{GlobalArena, position::PositionHorizon, symbol_registry, symbols};
use std::{
    path::PathBuf,
    sync::{Mutex, atomic::Ordering::Relaxed},
};
use strategy_core::{NetPayoffSource, SignalType};

static ENVIRONMENT: Mutex<()> = Mutex::new(());

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
        let path =
            std::env::temp_dir().join(format!("tg-lg5-payoff-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        std::fs::create_dir(path.join("data")).unwrap();
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
                .starts_with("tg-lg5-payoff-")
        );
        std::fs::remove_dir_all(exact).unwrap();
    }
}

fn core(order: [&str; 2], context: OutcomeContext) -> GodEngineCore {
    symbols::update_dynamic_universe(order.iter().map(|s| (*s).to_owned()).collect());
    symbol_registry::update_registry(
        order
            .iter()
            .map(|s| symbol_registry::SymbolSpec {
                symbol: (*s).into(),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.001,
                min_notional: 5.0,
                max_leverage: 20,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            })
            .collect(),
    );
    let mut core =
        GodEngineCore::new_with_outcome_context(GlobalArena::build_in_own_stack(1000.0), context);
    core.swing_nn = None;
    core.scalp_forest = None;
    core
}

fn entry(i: u64, source: NetPayoffSource) -> EntryPayoffEvidence {
    EntryPayoffEvidence {
        slot: 0,
        generation: i + 1,
        is_long: true,
        entry_time_ms: 100 + 10 * i,
        score: 0.6,
        tau_ms: 60_000,
        entry_price: 100.0,
        initial_tp_price: 101.0,
        initial_sl_price: 99.0,
        source,
    }
}

#[test]
fn real_state_roundtrip_restores_by_symbol_after_coin_indices_are_reordered() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _sandbox = Sandbox::new();
    let mut old = core(["LG5AUSDT", "LG5BUSDT"], OutcomeContext::IsolatedSimulation);
    let a = entry(0, NetPayoffSource::LocalSimulation);
    let b = entry(0, NetPayoffSource::ExchangeLocalEstimate);
    assert!(old.net_payoff_by_coin[0].observe(a, 1.0, 100.0, 1.0, 105, 110));
    assert!(old.net_payoff_by_coin[1].observe(b, -2.0, 100.0, 1.0, 105, 110));
    let bytes = serde_json::to_vec(&old.estado_aprendido()).unwrap();
    let decoded: EstadoAprendido = serde_json::from_slice(&bytes).unwrap();
    validar(&decoded).unwrap();
    let mut restored = core(["LG5BUSDT", "LG5AUSDT"], OutcomeContext::IsolatedSimulation);
    assert_eq!(restored.restaurar_estado_aprendido(&decoded), 2);
    let a = restored.net_payoff_by_coin[1]
        .estimate(1, SignalType::Long, NetPayoffSource::LocalSimulation, 110)
        .unwrap();
    let b = restored.net_payoff_by_coin[0]
        .estimate(
            0,
            SignalType::Long,
            NetPayoffSource::ExchangeLocalEstimate,
            110,
        )
        .unwrap();
    assert_eq!((a.coin_id, a.mean_net_return_pct), (1, 0.01));
    assert_eq!((b.coin_id, b.mean_net_return_pct), (0, -0.02));
    assert!(
        restored.net_payoff_by_coin[0]
            .estimate(0, SignalType::Long, NetPayoffSource::LocalSimulation, 110)
            .is_none()
    );
    assert!(
        restored.net_payoff_by_coin[1]
            .estimate(
                1,
                SignalType::Long,
                NetPayoffSource::ExchangeLocalEstimate,
                110
            )
            .is_none()
    );
}

#[test]
fn legacy_platt_and_profit_metrics_do_not_fabricate_payoff_magnitudes() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _sandbox = Sandbox::new();
    let mut old = core(["LG5AUSDT", "LG5BUSDT"], OutcomeContext::IsolatedSimulation);
    old.alimentar_calibradores(0, 0.8, true, 1000);
    old.arena.coins[0].metrics.gross_wins.store(123.0, Relaxed);
    old.arena.coins[0].metrics.profit_factor.store(5.0, Relaxed);
    let mut value = serde_json::to_value(old.estado_aprendido()).unwrap();
    for coin in value["monedas"].as_array_mut().unwrap() {
        coin.as_object_mut().unwrap().remove("net_payoff");
    }
    let legacy: EstadoAprendido = serde_json::from_value(value).unwrap();
    validar(&legacy).unwrap();
    let mut restored = core(["LG5AUSDT", "LG5BUSDT"], OutcomeContext::IsolatedSimulation);
    assert_eq!(restored.restaurar_estado_aprendido(&legacy), 2);
    assert_eq!(restored.calibrator_by_coin[0].observations(), 1);
    for mode in [
        NetPayoffSource::LocalSimulation,
        NetPayoffSource::ExchangeLocalEstimate,
    ] {
        assert!(
            restored.net_payoff_by_coin[0]
                .estimate(0, SignalType::Long, mode, 2000)
                .is_none()
        );
    }
}

#[test]
fn restore_preserves_evicted_clock_and_slot_deduplication_watermarks() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _sandbox = Sandbox::new();
    let mut old = core(["LG5AUSDT", "LG5BUSDT"], OutcomeContext::IsolatedSimulation);
    let first = entry(0, NetPayoffSource::LocalSimulation);
    assert!(old.net_payoff_by_coin[0].observe(first, 1.0, 100.0, 1.0, 105, 1_000_000));
    for i in 1..=WINDOW as u64 {
        let e = entry(i, NetPayoffSource::LocalSimulation);
        assert!(old.net_payoff_by_coin[0].observe(
            e,
            1.0,
            100.0,
            1.0,
            e.entry_time_ms + 1,
            e.entry_time_ms + 1
        ));
    }
    let state = old.estado_aprendido();
    assert!(
        state.monedas[0]
            .net_payoff
            .rows
            .iter()
            .all(|r| r.observed_at_ms < 1_000_000)
    );
    assert_eq!(state.monedas[0].net_payoff.clock_ms, Some(1_000_000));
    let decoded: EstadoAprendido =
        serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
    let mut restored = core(["LG5AUSDT", "LG5BUSDT"], OutcomeContext::IsolatedSimulation);
    assert_eq!(restored.restaurar_estado_aprendido(&decoded), 2);
    assert_eq!(
        restored.net_payoff_by_coin[0].snapshot(),
        old.net_payoff_by_coin[0].snapshot()
    );
    assert!(
        restored.net_payoff_by_coin[0]
            .estimate(
                0,
                SignalType::Long,
                NetPayoffSource::LocalSimulation,
                999_999
            )
            .is_none()
    );
    assert!(!restored.net_payoff_by_coin[0].observe(first, 50.0, 100.0, 1.0, 106, 2_000_000));
    assert_eq!(
        restored.net_payoff_by_coin[0].snapshot(),
        old.net_payoff_by_coin[0].snapshot()
    );
}

#[test]
fn malformed_additive_snapshot_rejects_before_mutating_any_core_state() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _sandbox = Sandbox::new();
    let mut target = core(["LG5AUSDT", "LG5BUSDT"], OutcomeContext::IsolatedSimulation);
    let e = entry(0, NetPayoffSource::LocalSimulation);
    assert!(target.net_payoff_by_coin[0].observe(e, 1.0, 100.0, 1.0, 105, 105));
    let before = target.estado_aprendido();
    let mut corrupt = before.clone();
    corrupt.monedas[0].metricas.trade_count = 999;
    corrupt.monedas[1].net_payoff.watermarks.clear();
    assert!(validar(&corrupt).unwrap_err().contains("net_payoff"));
    assert_eq!(target.restaurar_estado_aprendido(&corrupt), 0);
    assert_eq!(target.estado_aprendido(), before);
}

#[test]
fn actual_close_consumes_ledger_and_only_eligible_matched_local_outcomes_train() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _sandbox = Sandbox::new();
    for (context, confirmed, matched, expected_n) in [
        (OutcomeContext::IsolatedSimulation, false, true, 1),
        (OutcomeContext::ExchangeLocalEstimate, false, true, 0),
        (OutcomeContext::ExchangeLocalEstimate, true, true, 1),
        (OutcomeContext::IsolatedSimulation, false, false, 0),
    ] {
        let mut core = core(["LG5AUSDT", "LG5BUSDT"], context);
        let arena = core.arena.clone();
        let pos = arena.coins[0].positions.get_slot(0);
        assert!(pos.open_with_tau_and_fee(
            true,
            100.0,
            1.0,
            10.0,
            1000,
            101.0,
            99.0,
            PositionHorizon::Continuous,
            0.5,
            0.6,
            0.2,
            60_000
        ));
        pos.exchange_confirmed.store(confirmed, Relaxed);
        arena.used_margin.store(10.0, Relaxed);
        let generation = pos.generation.load(Relaxed);
        core.entry_learning_evidence[0][0] = Some(EntryLearningEvidence {
            symbol: "LG5AUSDT".into(),
            generation,
            is_long: true,
            decision_time_ms: 1000,
            branch: None,
            forest_vote: None,
            model_predictions: [None; 2],
        });
        core.entry_payoff_evidence[0][0] = Some(EntryPayoffEvidence {
            slot: 0,
            generation: if matched {
                generation
            } else {
                generation.wrapping_add(1)
            },
            is_long: true,
            entry_time_ms: 1000,
            score: 0.6,
            tau_ms: 60_000,
            entry_price: 100.0,
            initial_tp_price: 101.0,
            initial_sl_price: 99.0,
            source: source_for(context),
        });
        let (new_order, close, _) =
            core.process_tick_dual(0, 101.1, 101.11, 5.0, 5.0, 2000, &[0.0; 54], true, false);
        assert!(new_order.is_none());
        let (_, net_pnl, qty) = close.expect("synthetic defensive close");
        assert!(
            core.entry_payoff_evidence[0][0].is_none(),
            "ledger consumed even if ineligible"
        );
        let estimate =
            core.net_payoff_by_coin[0].estimate(0, SignalType::Long, source_for(context), 2000);
        assert_eq!(estimate.map_or(0, |e| e.sample_count), expected_n);
        if let Some(e) = estimate {
            assert!((e.mean_net_return_pct - net_pnl / (100.0 * qty)).abs() < 1e-14);
            assert_eq!(e.source, source_for(context));
        }
    }
}
