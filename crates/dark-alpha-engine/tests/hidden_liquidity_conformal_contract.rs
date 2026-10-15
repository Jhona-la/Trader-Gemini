//! Contract: Hidden Liquidity, Block Absorption and Conformal Risk Gating.
//!
//! Verifies:
//! 1. Parameter boundary invariants and fallback sanitization.
//! 2. Kyle-Hasbrouck absorption ratio and directional iceberg pressure.
//! 3. Conformal gate verdicts (veto on opposing institutional walls, deescalation, alignment bonus).
//! 4. NaN / Infinity / zero ATR robustness without panicking.

use dark_alpha_engine::hidden_liquidity::{
    BlockAbsorptionState, ConformalGateVerdict, HiddenLiquidityEngine, HiddenLiquidityParams,
};

#[test]
fn test_hidden_liquidity_params_validation_and_defaults() {
    let mut p = HiddenLiquidityParams::default();
    assert!(p.validate().is_ok());

    // kyle_lambda_scale must be positive finite
    p.kyle_lambda_scale = 0.0;
    assert!(p.validate().is_err());
    p.kyle_lambda_scale = f64::NAN;
    assert!(p.validate().is_err());
    p.kyle_lambda_scale = 1.0;

    // absorption_ratio_threshold must be > 0
    p.absorption_ratio_threshold = -0.1;
    assert!(p.validate().is_err());
    p.absorption_ratio_threshold = 1.25;

    // conformal_shrink_factor must be in (0, 1]
    p.conformal_shrink_factor = 0.0;
    assert!(p.validate().is_err());
    p.conformal_shrink_factor = 1.1;
    assert!(p.validate().is_err());
    p.conformal_shrink_factor = 0.70;

    // conformal_expansion_on_opposing must be >= 1.0
    p.conformal_expansion_on_opposing = 0.90;
    assert!(p.validate().is_err());
    p.conformal_expansion_on_opposing = 1.50;
    assert!(p.validate().is_ok());

    // Invalid params fallback to default in constructor
    let mut bad_params = HiddenLiquidityParams::default();
    bad_params.conformal_shrink_factor = -1.0;
    let engine = HiddenLiquidityEngine::new(bad_params);
    assert_eq!(engine.params(), &HiddenLiquidityParams::default());
}

#[test]
fn test_block_absorption_iceberg_detection_and_directional_pressure() {
    let engine = HiddenLiquidityEngine::default();
    let safe_atr = 100.0;

    // Case 1: Massive trade volume (500) absorbed in ask with minimal price displacement (0.01)
    // Visible ask depth is 100. Trade volume / visible depth = 5.0 >> 1.25 threshold.
    let state_ask = engine.compute_absorption(500.0, 100.0, 100.0, 0.01, safe_atr);
    assert!(state_ask.is_iceberg_active);
    assert!(state_ask.ask_absorption_ratio > 1.25);
    assert!(state_ask.net_iceberg_pressure > 0.0, "net pressure should be positive (ask absorption)");
    assert_eq!(state_ask.conformal_band_scale, 0.70);

    // Case 2: Massive trade volume absorbed in bid with minimal downward displacement (-0.01)
    let state_bid = engine.compute_absorption(500.0, 100.0, 100.0, -0.01, safe_atr);
    assert!(state_bid.is_iceberg_active);
    assert!(state_bid.bid_absorption_ratio > 1.25);
    assert!(state_bid.net_iceberg_pressure < 0.0, "net pressure should be negative (bid absorption)");
    assert_eq!(state_bid.conformal_band_scale, 0.70);

    // Case 3: Large price displacement (50.0 >> atr) dampens absorption ratio (no iceberg)
    let state_moving = engine.compute_absorption(500.0, 100.0, 100.0, 500.0, safe_atr);
    assert!(!state_moving.is_iceberg_active);
    assert_eq!(state_moving.conformal_band_scale, 1.0);
}

#[test]
fn test_conformal_risk_gate_verdicts() {
    let engine = HiddenLiquidityEngine::default();

    // Active heavy bid absorption (institutional distribution, net_pressure = -0.80)
    let state_distribution = BlockAbsorptionState {
        ask_absorption_ratio: 0.20,
        bid_absorption_ratio: 3.50,
        net_iceberg_pressure: -0.80,
        is_iceberg_active: true,
        conformal_band_scale: 0.70,
    };

    // Attempting Long against distribution -> VETO
    let verdict_long = engine.evaluate_conformal_gate(1.0, &state_distribution);
    assert!(matches!(
        verdict_long,
        ConformalGateVerdict::OpposingAbsorptionVeto { .. }
    ));

    // Attempting Short aligned with distribution -> ADMITTED with bonus
    let verdict_short = engine.evaluate_conformal_gate(-1.0, &state_distribution);
    match verdict_short {
        ConformalGateVerdict::Admitted { confidence_multiplier, band_scale } => {
            assert!(confidence_multiplier > 1.0, "aligned intent gets confidence bonus");
            assert_eq!(band_scale, 0.70);
        }
        _ => panic!("aligned trade should be admitted"),
    }

    // Moderate opposing absorption -> ProbeOnlyDeescalation
    let state_moderate = BlockAbsorptionState {
        ask_absorption_ratio: 0.50,
        bid_absorption_ratio: 1.80,
        net_iceberg_pressure: -0.45,
        is_iceberg_active: true,
        conformal_band_scale: 0.70,
    };
    let verdict_mod = engine.evaluate_conformal_gate(1.0, &state_moderate);
    assert!(matches!(
        verdict_mod,
        ConformalGateVerdict::ProbeOnlyDeescalation { max_lot_scale: 0.25, band_scale: 1.50 }
    ));
}

#[test]
fn test_nan_and_cold_start_immunity() {
    let engine = HiddenLiquidityEngine::default();

    // Cold start with zero trade volume
    let state_cold = engine.compute_absorption(0.0, 100.0, 100.0, 0.0, 1.0);
    assert!(!state_cold.is_iceberg_active);
    assert_eq!(state_cold.net_iceberg_pressure, 0.0);

    // NaN / Infinity inputs
    let state_nan = engine.compute_absorption(f64::NAN, 100.0, 100.0, 0.0, 1.0);
    assert!(!state_nan.is_iceberg_active);

    let state_inf_atr = engine.compute_absorption(100.0, 100.0, 100.0, 1.0, f64::INFINITY);
    assert!(!state_inf_atr.is_iceberg_active);

    // Gate evaluation with NaN intent
    let verdict_nan = engine.evaluate_conformal_gate(f64::NAN, &state_cold);
    assert!(matches!(
        verdict_nan,
        ConformalGateVerdict::Admitted { confidence_multiplier, band_scale } if confidence_multiplier == 1.0 && band_scale == 1.0
    ));
}
