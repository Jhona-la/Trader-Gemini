//! MG02/MG05: real estimators, real registry and the existing risk consumer.
use god_engine_core::evidence_publication::{publicar_coherencia, publicar_lundberg};
use omniscient_registry::OmniscientRegistry;
use quantum_arena::{
    GlobalArena, espectral_multiactivo::EspectralMultiactivo, state::CompactTick,
    temporal_spectrum::SPECTRUM_SCALES_MS,
};
use risk_engine::{RiskEngine, cramer_lundberg::EstimadorSiniestros};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{Arc, Once, atomic::Ordering::Relaxed};

const A: usize = 19;
const B: usize = 20;

fn est_maduro() -> EstimadorSiniestros {
    let mut est = EstimadorSiniestros::new();
    for i in 0..256 {
        est.observar(if i % 2 == 0 { 0.015 } else { -0.01 });
    }
    assert!(est.lundberg().unwrap() > 0.0);
    est
}

// Reader-only fixture: p=9/16 on +/-0.2 has R=ln(9/7)/0.2.
// Keep est_maduro unchanged for the independent publication contracts.
fn est_restrictivo_para_lector() -> EstimadorSiniestros {
    let mut est = EstimadorSiniestros::new();
    for i in 0..256 {
        est.observar(if i % 16 < 9 { 0.2 } else { -0.2 });
    }
    let r = est.lundberg().expect("mature positive-drift fixture");
    let analytic_r = (9.0_f64 / 7.0).ln() / 0.2;
    assert!((r - analytic_r).abs() < 1e-9, "R={r}, analytic={analytic_r}");
    assert!(r < 0.75 * 20.0_f64.ln());
    est
}

fn reader_group_cap(arena: &GlobalArena) -> f64 {
    let coin = &arena.coins[0];
    let q = risk_engine::ruin::conservative_loss_q(
        coin.metrics.win_rate.load(Relaxed),
        coin.metrics.trade_count.load(Relaxed) as f64,
    );
    let tope = risk_engine::ruin::clamp_ruin(1.0, q);
    assert!(tope.is_finite() && tope > 0.0);
    tope
}

fn seed_candidate_risk(arena: &GlobalArena) -> f64 {
    let tope = reader_group_cap(arena);
    arena.riesgo_por_operacion.store(0.75 * tope, Relaxed);
    tope
}

fn assert_lundberg_reader_preconditions(arena: &GlobalArena, r: f64, mature_ic: bool) {
    use risk_engine::correlation_guard::{
        calcular_riesgo_grupo, dependency_exposure, veto_por_riesgo_cramer_lundberg,
    };
    let tope = reader_group_cap(arena);
    let dependence = dependency_exposure(
        arena, 0, true, arena.config.global_correlation_threshold.load(Relaxed),
    ).expect("valid reader group");
    let mut riesgos = dependence.same_bet_riesgos_hibridos(tope / 8.0);
    riesgos.push(arena.riesgo_por_operacion.load(Relaxed));
    // Fixture specs have Lmax=1. A mature IC=1 tightens the actual peer group.
    let rho = if mature_ic && dependence.same_bet_positions > 0 {
        Some(1.0)
    } else {
        dependence.same_bet_rho_efectivo()
    };
    let group = calcular_riesgo_grupo(&riesgos, rho);
    let reference_cap = tope * (r / 20.0_f64.ln()).min(1.0);
    assert!(reference_cap > 0.0 && group > reference_cap && group < tope,
        "fixture must separate CL from the base cap: CL={reference_cap}, group={group}, base={tope}");
    if mature_ic {
        assert_eq!(dependence.same_bet_positions, 1);
        assert!((group - 0.875 * tope).abs() < 1e-12);
    }
    assert!(!veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope, None, 0.05));
    assert!(veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope, Some(r), 0.05));
}

fn multiactivo_maduro() -> EspectralMultiactivo {
    let mut ma = EspectralMultiactivo::new(3);
    let tau = SPECTRUM_SCALES_MS[A];
    // Main's Ville family gate needs more evidence than the old 40-pair
    // fixture. Preserve that gate: cold at 40, supported after 120 pairs.
    for i in 0..120u64 {
        let ts = ((i + 1) as f64 * tau) as u64;
        let r = if i % 2 == 0 { 0.02 } else { -0.02 };
        ma.observar_maduracion(0, tau, A, ts, r);
        ma.observar_maduracion(1, tau, A, ts, r);
        if i == 39 {
            assert_eq!(ma.coherencia_media_con_todas(0, A), None);
        }
    }
    assert_eq!(ma.coherencia_media_con_todas(0, A), Some(1.0));
    assert_eq!(ma.coherencia_media_con_todas(0, B), None);
    ma
}

#[test]
fn mg02_some_none_clears_r_and_margin_on_first_loss_of_evidence_and_recovers() {
    let registry = OmniscientRegistry::new();
    let mut est = est_maduro();
    let r = est.lundberg().unwrap();
    publicar_lundberg(&registry, 0, Some(r));
    assert_eq!(registry.get_for_coin_or(0, "lundberg_r_nocional", -1.0), r);
    assert_eq!(
        registry.get_for_coin_or(0, "lundberg_margen_5pct", -1.0),
        20.0_f64.ln() / r
    );
    for _ in 0..256 {
        est.observar(-0.01);
        let current = est.lundberg();
        publicar_lundberg(&registry, 0, current);
        assert_eq!(
            registry.get_for_coin_or(0, "lundberg_r_nocional", -1.0),
            current.unwrap_or(0.0)
        );
        assert_eq!(
            registry.get_for_coin_or(0, "lundberg_margen_5pct", -1.0),
            current.map(|r| 20.0_f64.ln() / r).unwrap_or(0.0)
        );
    }
    assert_eq!(est.lundberg(), None);
    for i in 0..256 {
        est.observar(if i % 2 == 0 { 0.015 } else { -0.01 });
        publicar_lundberg(&registry, 0, est.lundberg());
    }
    assert_eq!(registry.get_for_coin_or(0, "lundberg_r_nocional", 0.0), r);
}

#[test]
fn mg02_nonfinite_some_is_absence_and_finite_some_recovers() {
    let registry = OmniscientRegistry::new();
    let r = est_maduro().lundberg().unwrap();
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        publicar_lundberg(&registry, 0, Some(r));
        publicar_lundberg(&registry, 0, Some(invalid));
        assert_eq!(
            registry.get_for_coin_or(0, "lundberg_r_nocional", -9.0),
            0.0,
            "R={invalid}"
        );
        assert_eq!(
            registry.get_for_coin_or(0, "lundberg_margen_5pct", -9.0),
            0.0
        );
        publicar_lundberg(&registry, 0, Some(r));
        assert_eq!(registry.get_for_coin_or(0, "lundberg_r_nocional", -9.0), r);
        assert_eq!(
            registry.get_for_coin_or(0, "lundberg_margen_5pct", -9.0),
            20.0_f64.ln() / r
        );
    }
}

#[test]
fn mg05_a_cold_b_return_a_does_not_reuse_a_at_b() {
    let registry = OmniscientRegistry::new();
    let ma = multiactivo_maduro();
    publicar_coherencia(&registry, &ma, 0, SPECTRUM_SCALES_MS[A]);
    assert_eq!(registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0), 1.0);
    publicar_coherencia(&registry, &ma, 0, SPECTRUM_SCALES_MS[B]);
    assert_eq!(registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0), -1.0);
    publicar_coherencia(&registry, &ma, 0, SPECTRUM_SCALES_MS[A]);
    assert_eq!(registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0), 1.0);
}

#[test]
fn mg05_invalid_tau_invalidates_previous_ic_and_valid_tau_recovers() {
    let registry = OmniscientRegistry::new();
    let ma = multiactivo_maduro();
    for tau in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        publicar_coherencia(&registry, &ma, 0, SPECTRUM_SCALES_MS[A]);
        publicar_coherencia(&registry, &ma, 0, tau);
        assert_eq!(
            registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0),
            -1.0,
            "tau={tau}"
        );
    }
    publicar_coherencia(&registry, &ma, 0, SPECTRUM_SCALES_MS[A]);
    assert_eq!(registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0), 1.0);
}

#[test]
fn mg05_trade_event_without_depth_expires_previous_ic_in_real_core() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = god_engine_core::GodEngineCore::new(Arc::clone(&arena));
    arena.registry.set_for_coin(0, "qo_613_rho_tau", 0.9);
    // Cold spectrum, no mature pairs. Disable entries; no orders or live host.
    core.process_event(
        0, true, false, false, 100.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1_000, true, &[0.0; 54],
        false,
    );
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0),
        -1.0
    );
}

#[test]
fn mg05_direct_dual_tick_expires_previous_ic_in_real_core() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = god_engine_core::GodEngineCore::new(Arc::clone(&arena));
    arena.registry.set_for_coin(0, "qo_613_rho_tau", 0.9);
    // Backtest callers bypass process_event. No entries or live host.
    core.process_tick_dual(0, 100.0, 100.0, 1.0, 1.0, 1_000, &[0.0; 54], false, false);
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0),
        -1.0
    );
}

#[test]
fn mg05_event_expires_ic_before_kill_switch_early_return() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = god_engine_core::GodEngineCore::new(Arc::clone(&arena));
    arena.registry.set_for_coin(0, "qo_613_rho_tau", 0.9);
    arena.kill_switch_active.store(true, Relaxed);
    core.process_event(
        0, true, false, false, 100.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1_000, true, &[0.0; 54],
        false,
    );
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0),
        -1.0
    );
}

#[test]
fn expired_coin_masks_global_without_expiring_other_coins_or_symbol_scope() {
    let registry = OmniscientRegistry::new();
    let ma = multiactivo_maduro();
    registry.set("lundberg_r_nocional", 70.0);
    registry.set("lundberg_margen_5pct", 0.04);
    registry.set("qo_613_rho_tau", 0.9);
    registry.set_scoped("BTCUSDT", "qo_613_rho_tau", 0.7);
    publicar_lundberg(&registry, 1, Some(30.0));
    publicar_coherencia(&registry, &ma, 1, SPECTRUM_SCALES_MS[A]);
    publicar_lundberg(&registry, 0, None);
    publicar_coherencia(&registry, &ma, 0, SPECTRUM_SCALES_MS[B]);
    assert_eq!(
        registry.get_for_coin_or(0, "lundberg_r_nocional", -9.0),
        0.0
    );
    assert_eq!(
        registry.get_for_coin_or(0, "lundberg_margen_5pct", -9.0),
        0.0
    );
    assert_eq!(registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0), -1.0);
    assert_eq!(
        registry.get_for_coin_or(1, "lundberg_r_nocional", -9.0),
        30.0
    );
    assert_eq!(registry.get_for_coin_or(1, "qo_613_rho_tau", -9.0), 1.0);
    assert_eq!(
        registry.get_for_coin_or(2, "lundberg_r_nocional", -9.0),
        70.0
    );
    assert_eq!(registry.get_for_coin_or(2, "qo_613_rho_tau", -9.0), 0.9);
    assert_eq!(
        registry.get_scoped_value_or("BTCUSDT", "qo_613_rho_tau", -9.0),
        0.7
    );
    registry.set("lundberg_r_nocional", 80.0);
    registry.set("qo_613_rho_tau", 0.95);
    assert_eq!(
        registry.get_for_coin_or(0, "lundberg_r_nocional", -9.0),
        0.0
    );
    assert_eq!(registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0), -1.0);
    assert_eq!(
        registry.get_for_coin_or(2, "lundberg_r_nocional", -9.0),
        80.0
    );
}

fn risk_fixture() -> (Arc<GlobalArena>, SignalIntent) {
    static SPECS: Once = Once::new();
    SPECS.call_once(|| {
        quantum_arena::symbol_registry::update_registry(
            ["BTCUSDT", "ETHUSDT"]
                .iter()
                .map(|symbol| quantum_arena::symbol_registry::SymbolSpec {
                    symbol: (*symbol).into(),
                    step_size: 0.001,
                    tick_size: 0.01,
                    min_qty: 0.001,
                    min_notional: 5.0,
                    max_leverage: 1,
                    maker_fee: 0.0002,
                    taker_fee: 0.0004,
                    is_shadow: false,
                })
                .collect(),
        );
    });
    let arena = GlobalArena::build_in_own_stack(100.0);
    let coin = &arena.coins[0];
    coin.current_price.store(100.0, Relaxed);
    coin.current_atr.store(1.0, Relaxed);
    coin.hurst_exponent.store(0.5, Relaxed);
    coin.metrics.trade_count.store(100, Relaxed);
    coin.metrics.profit_factor.store(2.0, Relaxed);
    coin.metrics.win_rate.store(0.75, Relaxed);
    coin.metrics.kelly_fraction.store(0.15, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.2, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena
        .config
        .global_correlation_threshold
        .store(0.5, Relaxed);
    seed_candidate_risk(&arena);
    (
        arena,
        SignalIntent {
            signal: SignalType::Long,
            confidence: 0.9,
            expected_duration_ms: 60_000,
            ..SignalIntent::default()
        },
    )
}

#[test]
fn mg02_real_risk_reader_stops_using_expired_r_even_with_positive_global() {
    let (arena, intent) = risk_fixture();
    let mut risk = RiskEngine::new(100.0);
    let mut est = est_restrictivo_para_lector();
    let r = est.lundberg().unwrap();
    assert_lundberg_reader_preconditions(&arena, r, false);
    assert_eq!(risk.evaluate_quantum_order(0, &intent, &arena).signal, SignalType::Long);
    assert_eq!(arena.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 0.0);
    seed_candidate_risk(&arena);
    arena.registry.set("lundberg_r_nocional", r);
    publicar_lundberg(&arena.registry, 0, Some(r));
    assert_eq!(risk.evaluate_quantum_order(0, &intent, &arena).signal, SignalType::Flat);
    assert_eq!(arena.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 1.0);
    for _ in 0..256 {
        est.observar(-0.01);
    }
    assert_eq!(est.lundberg(), None);
    publicar_lundberg(&arena.registry, 0, est.lundberg());
    assert_eq!(arena.registry.get_for_coin_or(0, "lundberg_r_nocional", -9.0), 0.0);
    assert_eq!(arena.registry.get_value_fast("lundberg_r_nocional"), Some(r));
    seed_candidate_risk(&arena);
    assert_eq!(risk.evaluate_quantum_order(0, &intent, &arena).signal, SignalType::Long);
    assert_eq!(arena.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 1.0);
}

#[test]
fn expiring_one_arena_does_not_expire_another_arena_with_the_same_coin_id() {
    let (first, intent) = risk_fixture();
    let (second, _) = risk_fixture();
    let ma = multiactivo_maduro();
    let r = est_restrictivo_para_lector().lundberg().unwrap();
    for arena in [&first, &second] {
        let mut baseline = RiskEngine::new(100.0);
        assert_lundberg_reader_preconditions(arena, r, false);
        assert_eq!(baseline.evaluate_quantum_order(0, &intent, arena).signal, SignalType::Long);
        assert_eq!(arena.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 0.0);
        seed_candidate_risk(arena);
        publicar_lundberg(&arena.registry, 0, Some(r));
        publicar_coherencia(&arena.registry, &ma, 0, SPECTRUM_SCALES_MS[A]);
        assert_eq!(baseline.evaluate_quantum_order(0, &intent, arena).signal, SignalType::Flat);
        assert_eq!(arena.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 1.0);
    }
    publicar_lundberg(&first.registry, 0, None);
    publicar_coherencia(&first.registry, &ma, 0, SPECTRUM_SCALES_MS[B]);
    assert_eq!(
        first
            .registry
            .get_for_coin_or(0, "lundberg_r_nocional", -9.0),
        0.0
    );
    assert_eq!(
        first.registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0),
        -1.0
    );
    assert_eq!(
        second
            .registry
            .get_for_coin_or(0, "lundberg_r_nocional", -9.0),
        r
    );
    assert_eq!(
        second
            .registry
            .get_for_coin_or(0, "lundberg_margen_5pct", -9.0),
        20.0_f64.ln() / r
    );
    assert_eq!(
        second.registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0),
        1.0
    );
    let mut risk = RiskEngine::new(100.0);
    assert_eq!(
        risk.evaluate_quantum_order(0, &intent, &first).signal,
        SignalType::Long
    );
    assert_eq!(
        risk.evaluate_quantum_order(0, &intent, &second).signal,
        SignalType::Flat
    );
    assert_eq!(first.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 1.0);
    assert_eq!(second.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 2.0);
}

fn open_correlated_peer(arena: &GlobalArena) {
    open_correlated_peer_with_quantity(arena, 8.0);
}

fn open_correlated_peer_with_quantity(arena: &GlobalArena, quantity: f64) {
    for i in 0..200u64 {
        let t = i as f64 * 0.7;
        for coin in 0..2 {
            let change = if coin == 0 {
                t.sin()
            } else {
                0.6 * t.sin() + 0.8 * t.cos()
            };
            let price = 100.0 * (0.01 * change).exp();
            arena.coins[coin].tick_ring.push(CompactTick {
                timestamp: 1000 + i * 10,
                bid_price: price,
                ask_price: price,
                bid_qty: 1.0,
                ask_qty: 1.0,
            });
        }
    }
    arena.coins[1].positions.slots()[0].open(true, 100.0, quantity, 0.1, 1000, 102.0, 99.0);
    let d = risk_engine::correlation_guard::dependency_exposure(arena, 0, true, 0.5).unwrap();
    assert_eq!(d.same_bet_positions, 1);
    let rho = d.same_bet_rho_efectivo().unwrap();
    assert!(rho > 0.5 && rho < 0.8, "rho={rho}");
}

#[test]
fn mg05_real_risk_reader_stops_tightening_on_cold_b_and_resumes_on_a() {
    let (arena, intent) = risk_fixture();
    open_correlated_peer(&arena);
    let ma = multiactivo_maduro();
    let mut risk = RiskEngine::new(100.0);
    arena.registry.set("qo_613_rho_tau", 0.95);
    publicar_coherencia(&arena.registry, &ma, 0, SPECTRUM_SCALES_MS[A]);
    risk.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_aprietes", 0.0),
        1.0
    );
    publicar_coherencia(&arena.registry, &ma, 0, SPECTRUM_SCALES_MS[B]);
    risk.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_aprietes", 0.0),
        1.0
    );
    publicar_coherencia(&arena.registry, &ma, 0, SPECTRUM_SCALES_MS[A]);
    risk.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_aprietes", 0.0),
        2.0
    );
}

#[test]
fn legacy_finite_scoped_values_and_global_fallback_still_reach_real_risk_reader() {
    let (arena, intent) = risk_fixture();
    open_correlated_peer(&arena);
    let mut risk = RiskEngine::new(100.0);
    arena.registry.set("qo_613_rho_tau", 0.9);
    risk.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_aprietes", 0.0),
        1.0
    );
    arena.registry.set_for_coin(0, "qo_613_rho_tau", 0.7);
    risk.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_aprietes", 0.0),
        2.0
    );
    arena.registry.set_for_coin(0, "qo_613_rho_tau", -0.9);
    risk.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_aprietes", 0.0),
        2.0
    );
    let (other, intent) = risk_fixture();
    assert_eq!(risk.evaluate_quantum_order(0, &intent, &other).signal, SignalType::Long);
    seed_candidate_risk(&other);
    assert_lundberg_reader_preconditions(&other, 1.0, false);
    other.registry.set("lundberg_r_nocional", 1.0);
    assert_eq!(risk.evaluate_quantum_order(0, &intent, &other).signal, SignalType::Flat);
    assert_eq!(other.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 1.0);
    // A permissive scoped value must win over the restrictive global fallback.
    other.registry.set_for_coin(0, "lundberg_r_nocional", 8.0);
    assert_eq!(risk.evaluate_quantum_order(0, &intent, &other).signal, SignalType::Long);
    assert_eq!(other.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 1.0);
    seed_candidate_risk(&other);
    assert_lundberg_reader_preconditions(&other, 1.5, false);
    other.registry.set_for_coin(0, "lundberg_r_nocional", 1.5);
    assert_eq!(risk.evaluate_quantum_order(0, &intent, &other).signal, SignalType::Flat);
    assert_eq!(other.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 2.0);
    publicar_lundberg(&other.registry, 0, None);
    assert_eq!(other.registry.get_for_coin_or(0, "lundberg_r_nocional", -9.0), 0.0);
    assert_eq!(other.registry.get_value_fast("lundberg_r_nocional"), Some(1.0));
    seed_candidate_risk(&other);
    assert_eq!(risk.evaluate_quantum_order(0, &intent, &other).signal, SignalType::Long);
    assert_eq!(other.registry.get_for_coin_or(0, "qo_602_veto_lundberg", 0.0), 2.0);
}

#[test]
fn concurrent_readers_see_expiry_after_publication_without_cross_coin_reactivation() {
    let (arena, intent) = risk_fixture();
    let tope = seed_candidate_risk(&arena);
    // At IC=1, peer risk 0.125T plus candidate 0.75T stays below the base cap.
    let peer_quantity = 0.125 * tope * arena.unified_capital.load(Relaxed);
    open_correlated_peer_with_quantity(&arena, peer_quantity);
    let ma = multiactivo_maduro();
    let r = est_restrictivo_para_lector().lundberg().unwrap();
    assert_lundberg_reader_preconditions(&arena, r, true);
    let mut risk = RiskEngine::new(100.0);
    publicar_lundberg(&arena.registry, 0, Some(r));
    publicar_coherencia(&arena.registry, &ma, 0, SPECTRUM_SCALES_MS[A]);
    let publication_done = std::sync::atomic::AtomicBool::new(false);
    let reader_ready = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            reader_ready.wait();
            loop {
                // Overlap with the same-coin publisher. Each numeric key is
                // complete, but mixed generations across keys are allowed.
                let root = arena
                    .registry
                    .get_for_coin_or(0, "lundberg_r_nocional", -9.0);
                let margin = arena
                    .registry
                    .get_for_coin_or(0, "lundberg_margen_5pct", -9.0);
                let ic = arena.registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0);
                assert!(root == r || root == 0.0);
                assert!(margin == 20.0_f64.ln() / r || margin == 0.0);
                assert!(ic == 1.0 || ic == -1.0);
                if publication_done.load(Relaxed) {
                    break;
                }
                std::thread::yield_now();
            }
        });
        scope.spawn(|| {
            for _ in 0..256 {
                publicar_lundberg(&arena.registry, 1, Some(30.0));
                publicar_coherencia(&arena.registry, &ma, 1, SPECTRUM_SCALES_MS[A]);
                arena.registry.set("lundberg_r_nocional", 1.0);
                arena.registry.set("qo_613_rho_tau", 0.95);
                publicar_lundberg(&arena.registry, 1, None);
                publicar_coherencia(&arena.registry, &ma, 1, SPECTRUM_SCALES_MS[B]);
            }
        });
        let (published, read_phase) = std::sync::mpsc::channel();
        let (ack, await_read) = std::sync::mpsc::channel();
        let publisher_arena = Arc::clone(&arena);
        let publisher_ma = ma.clone();
        let publication_done = &publication_done;
        let reader_ready = &reader_ready;
        scope.spawn(move || {
            reader_ready.wait();
            for active in [true, false].repeat(16) {
                publicar_lundberg(&publisher_arena.registry, 0, active.then_some(r));
                publicar_coherencia(
                    &publisher_arena.registry,
                    &publisher_ma,
                    0,
                    SPECTRUM_SCALES_MS[if active { A } else { B }],
                );
                // Acquire/release handoff marks completed publication, not a
                // transaction spanning unrelated arena or order state.
                if published.send(active).is_err() || await_read.recv().is_err() {
                    break;
                }
            }
            publication_done.store(true, Relaxed);
        });
        let mut active_phases = 0.0;
        while let Ok(active) = read_phase.recv() {
            assert_eq!(
                arena
                    .registry
                    .get_for_coin_or(0, "lundberg_r_nocional", -9.0),
                if active { r } else { 0.0 }
            );
            assert_eq!(
                arena
                    .registry
                    .get_for_coin_or(0, "lundberg_margen_5pct", -9.0),
                if active { 20.0_f64.ln() / r } else { 0.0 }
            );
            assert_eq!(
                arena.registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0),
                if active { 1.0 } else { -1.0 }
            );
            // Keep the candidate risk fixed: admitted cold phases can update
            // its EWMA, which is independent of evidence publication.
            seed_candidate_risk(&arena);
            assert_lundberg_reader_preconditions(&arena, r, true);
            assert_eq!(
                risk.evaluate_quantum_order(0, &intent, &arena).signal,
                if active { SignalType::Flat } else { SignalType::Long }
            );
            if active {
                active_phases += 1.0;
            }
            assert_eq!(
                arena.registry.get_for_coin_or(0, "qo_613_aprietes", 0.0),
                active_phases
            );
            assert_eq!(
                arena
                    .registry
                    .get_for_coin_or(0, "qo_602_veto_lundberg", 0.0),
                active_phases
            );
            ack.send(()).unwrap();
        }
        assert_eq!(active_phases, 16.0);
    });
}
