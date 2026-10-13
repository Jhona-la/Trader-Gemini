//! R4: real block producer, pair estimator, publication and group-risk reader.
//! Synthetic prices and in-memory state; no exchange or engine process.
use god_engine_core::evidence_publication::publicar_coherencia;
use quantum_arena::{
    espectral_multiactivo::EspectralMultiactivo,
    state::CompactTick,
    temporal_spectrum::{TemporalSpectrum, SPECTRUM_SCALES_MS},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc};

const K: usize = 19;
const RIESGO_FIJO: f64 = 0.13;

fn risk_fixture() -> (Arc<GlobalArena>, SignalIntent) {
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
    let arena = GlobalArena::build_in_own_stack(100.0);
    let coin = &arena.coins[0];
    coin.current_price.store(100.0, Relaxed);
    coin.current_atr.store(1.0, Relaxed);
    coin.hurst_exponent.store(0.5, Relaxed);
    coin.metrics.trade_count.store(100, Relaxed);
    coin.metrics.profit_factor.store(2.0, Relaxed);
    coin.metrics.gross_wins.store(200.0, Relaxed);
    coin.metrics.gross_losses.store(100.0, Relaxed);
    assert!(risk_engine::evidence::profit_factor_lcb(200.0, 100.0, 100.0) > 1.0);
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
    arena.riesgo_por_operacion.store(RIESGO_FIJO, Relaxed);
    for i in 0..200u64 {
        let t = i as f64 * 0.7;
        for id in 0..2 {
            let change = if id == 0 {
                t.sin()
            } else {
                0.6 * t.sin() + 0.8 * t.cos()
            };
            let price = 100.0 * (0.01 * change).exp();
            arena.coins[id].tick_ring.push(CompactTick {
                timestamp: 1000 + i * 10,
                bid_price: price,
                ask_price: price,
                bid_qty: 1.0,
                ask_qty: 1.0,
            });
        }
    }
    // open takes quantity before margin: 0.5 * (100 - 74) / 100 = 0.13.
    // Notional and collateral are both 50 at the fixture's exchange leverage 1.
    arena.coins[1].positions.slots()[0].open(true, 100.0, 0.5, 50.0, 1000, 102.0, 74.0);
    let dep = risk_engine::correlation_guard::dependency_exposure(&arena, 0, true, 0.5).unwrap();
    assert_eq!(dep.open_positions, 1);
    assert_eq!(dep.same_bet_positions, 1);
    assert_eq!(dep.unknown_positions, 0);
    let rho = dep.same_bet_rho_efectivo().unwrap();
    assert!(rho > 0.5 && rho < 0.8, "nontrivial scalar exposure: {rho}");
    let q = risk_engine::ruin::conservative_loss_q(0.75, 100.0);
    let cap = risk_engine::ruin::clamp_ruin(1.0, q);
    assert_eq!(cap, 0.25);
    let mut riesgos = dep.same_bet_riesgos_hibridos(cap / 8.0);
    assert_eq!(riesgos, vec![RIESGO_FIJO]);
    riesgos.push(RIESGO_FIJO);
    assert!(risk_engine::correlation_guard::calcular_riesgo_grupo(&riesgos, Some(rho)) < cap);
    assert!(risk_engine::correlation_guard::calcular_riesgo_grupo(&riesgos, Some(0.999)) > cap);
    (
        arena,
        SignalIntent {
            signal: SignalType::Long,
            confidence: 0.9,
            expected_duration_ms: SPECTRUM_SCALES_MS[K].ceil() as u64,
            ..SignalIntent::default()
        },
    )
}

#[test]
fn flat_producer_blocks_cannot_publish_a_fictitious_risk_tightening() {
    let tau = SPECTRUM_SCALES_MS[K];
    let step = tau.ceil() as u64;
    let mut producers = [TemporalSpectrum::new(), TemporalSpectrum::new()];
    let mut ma = EspectralMultiactivo::new(2);
    for producer in &mut producers {
        producer.update(100.0, 1000);
    }
    for i in 1..=113u64 {
        let ts = 1000 + i * step;
        for (id, producer) in producers.iter_mut().enumerate() {
            producer.update(101.0, ts);
            let (close, r) = producer.ultimo_bloque_maduro(K).expect("real block closed");
            assert_eq!(close, ts);
            if i == 1 {
                assert!(r > 0.0);
            } else {
                assert_eq!(r, 0.0);
            }
            ma.observar_maduracion(id, tau, K, close, r);
        }
    }
    assert_eq!(ma.muestras_par(0, 1, K), Some(113));
    assert!(ma.coherencia_par(0, 1, K).unwrap() > 0.999);
    let (arena, intent) = risk_fixture();
    let mut risk = RiskEngine::new(100.0);
    arena.registry.set("qo_613_rho_tau", 0.95);
    publicar_coherencia(&arena.registry, &ma, 0, tau);
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0),
        -1.0,
        "one informative pair plus flat blocks is absence of veto evidence"
    );
    arena.riesgo_por_operacion.store(RIESGO_FIJO, Relaxed);
    let cold = risk.evaluate_quantum_order(0, &intent, &arena);
    let cold_reason = risk_engine::ultimo_rechazo();
    assert_eq!(
        cold.signal,
        SignalType::Long,
        "cold rejection: {cold_reason:?}"
    );
    assert_eq!(cold_reason, None);
    assert_eq!(cold.leverage, 1.0);
    assert!(cold.volume_usd > 0.0);
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_aprietes", 0.0),
        0.0
    );
    // Positive control: the same producer/publisher/consumer must still react
    // to new informative paired blocks. No threshold or family is relaxed.
    for i in 114..=233u64 {
        let ts = 1000 + i * step;
        let price = 101.0 * (0.001 * (i - 113) as f64).exp();
        for (id, producer) in producers.iter_mut().enumerate() {
            producer.update(price, ts);
            let (close, r) = producer.ultimo_bloque_maduro(K).unwrap();
            assert_eq!(close, ts);
            assert!(r > 0.0);
            ma.observar_maduracion(id, tau, K, close, r);
        }
    }
    assert_eq!(ma.muestras_par(0, 1, K), Some(233));
    assert!(ma.coherencia_par_veto(0, 1, K).unwrap() > 0.999);
    publicar_coherencia(&arena.registry, &ma, 0, tau);
    assert!(arena.registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0) > 0.999);
    // Admitting cold updates the risk EWMA: restore the same candidate budget.
    arena.riesgo_por_operacion.store(RIESGO_FIJO, Relaxed);
    let warm = risk.evaluate_quantum_order(0, &intent, &arena);
    let warm_reason = risk_engine::ultimo_rechazo();
    assert_eq!(warm.signal, SignalType::Flat);
    assert_eq!(warm_reason, Some(risk_engine::REJ_CORRELATION));
    assert_eq!(
        arena.registry.get_for_coin_or(0, "qo_613_aprietes", 0.0),
        1.0
    );
}

#[test]
fn replay_cannot_publish_evidence_before_113_unique_informative_pairs() {
    let tau = SPECTRUM_SCALES_MS[K];
    let step = tau.ceil() as u64;
    let mut ma = EspectralMultiactivo::new(2);
    let feed = |ma: &mut EspectralMultiactivo, range: std::ops::Range<u64>| {
        for i in range {
            let ts = (i + 1) * step;
            for id in 0..2 {
                ma.observar_maduracion(id, tau, K, ts, 0.02);
            }
        }
    };
    feed(&mut ma, 0..40);
    feed(&mut ma, 0..40);
    assert_eq!(
        ma.muestras_par(0, 1, K),
        Some(40),
        "historic replay must not add samples"
    );
    feed(&mut ma, 40..112);
    assert_eq!(ma.muestras_par(0, 1, K), Some(112));
    let registry = omniscient_registry::OmniscientRegistry::new();
    publicar_coherencia(&registry, &ma, 0, tau);
    assert_eq!(registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0), -1.0);
    feed(&mut ma, 112..113);
    assert_eq!(ma.muestras_par(0, 1, K), Some(113));
    publicar_coherencia(&registry, &ma, 0, tau);
    assert!(registry.get_for_coin_or(0, "qo_613_rho_tau", -9.0) > 0.999);
}
