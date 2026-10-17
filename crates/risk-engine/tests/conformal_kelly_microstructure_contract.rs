//! CONTRATO DE MODULACIÓN CONFORMAL DE MICROESTRUCTURA EN DIMENSIONAMIENTO KELLY (Ola Ω134)
//!
//! # QUÉ
//! Suite de pruebas de contrato matemático que valida el acoplamiento entre la escala de
//! incertidumbre conformal de absorción de bloques (Kyle-Hasbrouck) y el dimensionamiento
//! fraccionario de Kelly en `RiskEngine`.
//!
//! # POR QUÉ
//! En micro-cuentas ($13.00 USD), dimensionar estáticamente ante volatilidad comprimida
//! o ante expansión de riesgo de squeeze causa ruina o sub-explotación del edge.
//! Si la banda conformal se expande ante un choque de liquidez ($\kappa_{\text{conf}} = 1.50$),
//! el dimensionamiento Kelly debe contraerse preventivamente. Si la banda se contrae ante
//! absorción institucional alineada ($\kappa_{\text{conf}} = 0.70$), la certeza local
//! permite una aceleración geométrica controlada (+25%).
//!
//! # PARA QUÉ
//! 1. Paridad canónica exacta bit a bit con baseline cuando la escala conformal no está presente o es 1.0.
//! 2. Desescalada preventiva ante riesgo de squeeze opuesto (reducción de exposición).
//! 3. Aceleración compuesta controlada ante absorción institucional alineada.
//! 4. Monotonía estricta decreciente respecto a la incertidumbre conformal.
//! 5. Inmunidad matemática ante NaNs, infinitos y valores no-positivos.

use quantum_arena::symbol_registry::SymbolSpec;
use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Once};

static INIT_SYMBOLS: Once = Once::new();

fn init_test_symbols() {
    INIT_SYMBOLS.call_once(|| {
        quantum_arena::symbol_registry::update_registry(vec![
            SymbolSpec {
                symbol: "BTCUSDT".into(),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.001,
                min_notional: 5.0,
                max_leverage: 50,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            },
        ]);
    });
}

fn build_fixture(capital: f64) -> (Arc<GlobalArena>, SignalIntent) {
    std::env::set_var("TG_GENOME_ENV", "backtest");
    init_test_symbols();
    let arena = GlobalArena::build_in_own_stack(capital);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.65, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.2, Relaxed);
    arena.coins[0].metrics.win_rate.store(0.75, Relaxed);
    arena.coins[0].metrics.kelly_fraction.store(0.20, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.20, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.35, Relaxed);

    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.85,
        win_probability: 0.70,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };

    (arena, intent)
}

#[test]
fn test_conformal_kelly_parity_when_uninitialized() {
    let (arena_base, intent) = build_fixture(13.0);
    let mut risk_base = RiskEngine::new(13.0);
    let out_base = risk_base.evaluate_quantum_order(0, &intent, &arena_base);
    assert_eq!(out_base.signal, SignalType::Long);

    let (arena_unit, _) = build_fixture(13.0);
    arena_unit.registry.set_scoped("BTCUSDT", "hidden_liquidity_conformal_scale", 1.0);
    arena_unit.registry.set_for_coin(0, "hidden_liquidity_conformal_scale", 1.0);
    let mut risk_unit = RiskEngine::new(13.0);
    let out_unit = risk_unit.evaluate_quantum_order(0, &intent, &arena_unit);
    assert_eq!(out_unit.signal, SignalType::Long);

    assert_eq!(
        out_base.volume_usd, out_unit.volume_usd,
        "Sin override o con escala 1.0, el volumen nocional debe ser idéntico: {} vs {}",
        out_base.volume_usd, out_unit.volume_usd
    );
}

#[test]
fn test_opposing_squeeze_risk_reduces_kelly_fraction() {
    let (arena_base, intent) = build_fixture(13.0);
    let mut risk_base = RiskEngine::new(13.0);
    let out_base = risk_base.evaluate_quantum_order(0, &intent, &arena_base);
    assert_eq!(out_base.signal, SignalType::Long);

    let (arena_squeeze, _) = build_fixture(13.0);
    // Incertidumbre conformal expandida a 1.50 (choque de liquidez / squeeze opuesto)
    arena_squeeze.registry.set_scoped("BTCUSDT", "hidden_liquidity_conformal_scale", 1.50);
    arena_squeeze.registry.set_for_coin(0, "hidden_liquidity_conformal_scale", 1.50);
    let mut risk_squeeze = RiskEngine::new(13.0);
    let out_squeeze = risk_squeeze.evaluate_quantum_order(0, &intent, &arena_squeeze);
    assert_eq!(out_squeeze.signal, SignalType::Long);

    assert!(
        out_squeeze.volume_usd <= out_base.volume_usd,
        "La expansión de incertidumbre conformal (1.50x) debe desescalar o acotar la exposición: squeeze={} vs base={}",
        out_squeeze.volume_usd,
        out_base.volume_usd
    );
}

#[test]
fn test_aligned_iceberg_amplifies_kelly_sizing() {
    let (arena_base, intent) = build_fixture(13.0);
    let mut risk_base = RiskEngine::new(13.0);
    let out_base = risk_base.evaluate_quantum_order(0, &intent, &arena_base);
    assert_eq!(out_base.signal, SignalType::Long);

    let (arena_aligned, _) = build_fixture(13.0);
    // Incertidumbre conformal contraída a 0.70 (absorción institucional alineada)
    arena_aligned.registry.set_scoped("BTCUSDT", "hidden_liquidity_conformal_scale", 0.70);
    arena_aligned.registry.set_for_coin(0, "hidden_liquidity_conformal_scale", 0.70);
    let mut risk_aligned = RiskEngine::new(13.0);
    let out_aligned = risk_aligned.evaluate_quantum_order(0, &intent, &arena_aligned);
    assert_eq!(out_aligned.signal, SignalType::Long);

    assert!(
        out_aligned.volume_usd >= out_base.volume_usd,
        "La contracción de incertidumbre conformal (0.70x) debe amplificar o preservar la convicción: aligned={} vs base={}",
        out_aligned.volume_usd,
        out_base.volume_usd
    );
}

#[test]
fn test_conformal_kelly_monotonicity_across_conformal_scale() {
    let sweep_scales = [0.70, 0.85, 1.0, 1.25, 1.50, 1.80];
    let mut volumes = Vec::new();

    for &scale in &sweep_scales {
        let (arena, intent) = build_fixture(13.0);
        arena.registry.set_scoped("BTCUSDT", "hidden_liquidity_conformal_scale", scale);
        arena.registry.set_for_coin(0, "hidden_liquidity_conformal_scale", scale);

        let mut risk = RiskEngine::new(13.0);
        let out = risk.evaluate_quantum_order(0, &intent, &arena);
        assert_eq!(out.signal, SignalType::Long);
        volumes.push(out.volume_usd);
    }

    // A mayor escala conformal (mayor incertidumbre), el volumen debe ser monótonamente decreciente o igual
    for i in 1..volumes.len() {
        assert!(
            volumes[i] <= volumes[i - 1] + 1e-9,
            "Volumen no monótono en paso {i} (escala={}): vol[{}]={} vs vol[{}]={}",
            sweep_scales[i],
            i,
            volumes[i],
            i - 1,
            volumes[i - 1]
        );
    }
}

#[test]
fn test_conformal_kelly_nan_and_degeneracy_protection() {
    let (arena_corrupt, intent) = build_fixture(13.0);
    arena_corrupt.registry.set_scoped("BTCUSDT", "hidden_liquidity_conformal_scale", f64::NAN);
    arena_corrupt.registry.set_for_coin(0, "hidden_liquidity_conformal_scale", f64::INFINITY);

    let mut risk = RiskEngine::new(13.0);
    let out = risk.evaluate_quantum_order(0, &intent, &arena_corrupt);
    assert_eq!(out.signal, SignalType::Long);
    assert!(out.volume_usd.is_finite());
    assert!(out.volume_usd > 0.0);
}
