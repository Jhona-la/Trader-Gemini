//! Contrato Formal de Certificación F1-092 (Ola Ω195)
//!
//! Veredicto: Derivación rigurosa de apalancamiento y cota estricta del Axioma de Ruina:
//! 1. Fail-closed estricto ante capital no finito o no positivo (retorna 1.0, erradicando fallback a 13.0).
//! 2. Exponente de Hurst acotado al dominio fractal físico [0.30, 0.75] con bonus neutral 1.0 en H=0.50.
//! 3. Desacoplamiento de `veto_threshold_btc` erradicando la heurística espuria sobre Kelly.
//! 4. Cota estricta del Axioma de Ruina sobre el riesgo realizado al stop (volume_usd * leverage * stop / capital <= tope_ruina).

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::{leverage_matrix::QuantumLeverageMatrix, RiskEngine};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

fn fixture(capital: f64) -> (Arc<GlobalArena>, SignalIntent) {
    static REGISTRY: Once = Once::new();
    REGISTRY.call_once(|| {
        update_registry(vec![SymbolSpec {
            symbol: "BTCUSDT".into(),
            step_size: 0.001,
            tick_size: 0.01,
            min_qty: 0.001,
            min_notional: 5.0,
            max_leverage: 20,
            maker_fee: 0.0002,
            taker_fee: 0.0004,
            is_shadow: false,
        }])
    });
    let a = GlobalArena::build_in_own_stack(capital);
    a.coins[0].current_price.store(100.0, Relaxed);
    a.coins[0].current_atr.store(1.0, Relaxed);
    a.coins[0].hurst_exponent.store(0.5, Relaxed);
    a.config.global_leverage.store(20.0, Relaxed);
    a.config.global_max_drawdown.store(0.2, Relaxed);
    a.config.kelly_clamp_min.store(0.01, Relaxed);
    a.config.kelly_clamp_max.store(0.25, Relaxed);
    a.config.latency_penalty_ms.store(0.0, Relaxed);
    a.config.live_taker_fee.store(0.0004, Relaxed);
    a.config.base_slippage_floor.store(0.00001, Relaxed);
    (
        a,
        SignalIntent {
            signal: SignalType::Long,
            confidence: 0.9,
            win_probability: 0.9,
            expected_duration_ms: 60_000,
            ..Default::default()
        },
    )
}

#[test]
fn test_f1_092_capital_nonfinite_fails_closed_to_one() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let signal = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.85,
        win_probability: 0.85,
        ..Default::default()
    };

    // Casos no finitos o no positivos deben retornar 1.0 (fail-closed seguro)
    for bad_cap in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0, -13.0] {
        let lev_curr = QuantumLeverageMatrix::calculate_dynamic_leverage(
            &signal, 0.0, bad_cap, 13.0, 0.001, 1.0, 0.5, 2.0, 0.60, 20.0, 5.0, 0.01, &arena,
        );
        assert_eq!(
            lev_curr, 1.0,
            "Capital actual no finito/no positivo ({bad_cap}) debe fallar cerrado a 1.0"
        );

        let lev_base = QuantumLeverageMatrix::calculate_dynamic_leverage(
            &signal, 0.0, 13.0, bad_cap, 0.001, 1.0, 0.5, 2.0, 0.60, 20.0, 5.0, 0.01, &arena,
        );
        assert_eq!(
            lev_base, 1.0,
            "Capital base no finito/no positivo ({bad_cap}) debe fallar cerrado a 1.0"
        );
    }
}

#[test]
fn test_f1_092_hurst_bonus_bounded_in_fractal_domain() {
    let arena = GlobalArena::build_in_own_stack(10_000.0);
    let signal = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.85,
        win_probability: 0.85,
        ..Default::default()
    };

    let lev = |h: f64| {
        QuantumLeverageMatrix::calculate_dynamic_leverage(
            &signal, 0.0, 10_000.0, 10_000.0, 0.001, 1.0, h, 2.0, 0.60, 50.0, 5.0, 0.01, &arena,
        )
    };

    let lev_neutral = lev(0.50);
    let lev_persistent = lev(0.75);
    let lev_antipersistent = lev(0.30);

    assert!(
        lev_persistent > lev_neutral,
        "Hurst persistente (0.75) debe dar mayor apalancamiento que neutral (0.50)"
    );
    assert!(
        lev_antipersistent < lev_neutral,
        "Hurst antipersistente (0.30) debe dar menor apalancamiento que neutral (0.50)"
    );

    // Valores extremos fuera de [0.30, 0.75] deben ser clampeados
    let lev_extreme_high = lev(1.50);
    let lev_extreme_low = lev(-0.50);
    assert!(
        (lev_extreme_high - lev_persistent).abs() < 1e-9,
        "Hurst > 0.75 debe clampearse a 0.75"
    );
    assert!(
        (lev_extreme_low - lev_antipersistent).abs() < 1e-9,
        "Hurst < 0.30 debe clampearse a 0.30"
    );
}

#[test]
fn test_f1_092_veto_threshold_btc_decoupled_from_kelly_multiplier() {
    let arena = GlobalArena::build_in_own_stack(10_000.0);
    let signal = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.85,
        win_probability: 0.85,
        ..Default::default()
    };

    arena.config.veto_threshold_btc.store(0.10, Relaxed);
    let lev1 = QuantumLeverageMatrix::calculate_dynamic_leverage(
        &signal, 0.0, 10_000.0, 10_000.0, 0.001, 1.0, 0.5, 2.0, 0.60, 50.0, 5.0, 0.01, &arena,
    );

    arena.config.veto_threshold_btc.store(0.99, Relaxed);
    let lev2 = QuantumLeverageMatrix::calculate_dynamic_leverage(
        &signal, 0.0, 10_000.0, 10_000.0, 0.001, 1.0, 0.5, 2.0, 0.60, 50.0, 5.0, 0.01, &arena,
    );

    assert_eq!(
        lev1, lev2,
        "El gen veto_threshold_btc NO debe alterar el apalancamiento Kelly"
    );
}

#[test]
fn test_f1_092_ruin_axiom_realized_risk_ceiling() {
    let (arena, intent) = fixture(13.0);
    let mut engine = RiskEngine::new(13.0);

    // Barremos varios stops y apalancamientos
    for stop_target in [0.01, 0.02, 0.04, 0.08, 0.12] {
        let mut custom_intent = intent.clone();
        custom_intent.sl_price_target = 100.0 * (1.0 - stop_target);
        custom_intent.tp_price_target = 100.0 * (1.0 + stop_target * 2.5);

        let out = engine.evaluate_quantum_order(0, &custom_intent, &arena);
        if out.signal == SignalType::Long {
            let entry = arena.coins[0].current_price.load(Relaxed);
            let sl_dist = (entry - out.sl_target).abs() / entry;
            let realized_risk_usd = out.volume_usd * out.leverage * sl_dist;
            let realized_risk_fraction = realized_risk_usd / 13.0;

            assert!(
                realized_risk_fraction <= 0.25 + 1e-6,
                "Riesgo realizado ({:.4} = {:.2}%) excede el axioma de ruina (25%) para stop {:.2}%",
                realized_risk_fraction,
                realized_risk_fraction * 100.0,
                stop_target * 100.0
            );
        }
    }
}
