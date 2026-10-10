//! CONTRATO FORMAL DE RIGOR DIMENSIONAL E HIDRODINÁMICA DE NAVIER-STOKES L2/L3 (OLA Ω65)
//!
//! Verifica analítica, física y numéricamente los axiomas del Número de Reynolds continuo,
//! la viscosidad cinemática con adelgazamiento por corte (shear-thinning), la invariancia
//! de escala de precios, y la disipación de Kolmogorov en microestructura de alta frecuencia.

use feature_engine::navier_stokes::{HydrodynamicRegime, NavierStokesReynoldsEngine};

#[test]
fn test_navier_stokes_reynolds_pure_dimensionless_invariance() {
    // Teorema Físico Fundamental: El número de Reynolds es ADIMENSIONAL.
    // Dos activos con magnitudes de precio radicalmente distintas (ej. BTC a $60,000 vs SOL a $150 vs DOGE a $0.15),
    // que experimentan la misma dinámica relativa (mismo retorno relativo en el mismo dt, mismo spread en %,
    // y misma relación de agresión taker/maker), DEBEN producir exactamente el mismo número de Reynolds.

    let mut engine_btc = NavierStokesReynoldsEngine::new();
    let mut engine_sol = NavierStokesReynoldsEngine::new();
    let mut engine_doge = NavierStokesReynoldsEngine::new();

    let p_btc = 60_000.0;
    let p_sol = 150.0;
    let p_doge = 0.15;

    let spread_pct = 0.0005; // 5 bps de spread relativo
    let atr_pct = 0.0020;    // 20 bps de ATR
    let rel_jump = 0.0010;   // 10 bps de salto en 50 ms

    // Warmup inicial idéntico
    let t0 = 1_000_000u64;
    engine_btc.update(p_btc, p_btc * (1.0 + spread_pct), 5.0, 5.0, 0.0, atr_pct, t0);
    engine_sol.update(p_sol, p_sol * (1.0 + spread_pct), 5.0, 5.0, 0.0, atr_pct, t0);
    engine_doge.update(p_doge, p_doge * (1.0 + spread_pct), 5.0, 5.0, 0.0, atr_pct, t0);

    // Evento dinámico con la misma perturbación relativa en dt = 50 ms
    let t1 = t0 + 50;
    let p_btc_1 = p_btc * (1.0 + rel_jump);
    let p_sol_1 = p_sol * (1.0 + rel_jump);
    let p_doge_1 = p_doge * (1.0 + rel_jump);

    // Mismo ratio de agresión: 2x profundidad pasiva en USD
    let trade_usd = 100_000.0;
    let btc_trade_qty = trade_usd / p_btc_1;
    let sol_trade_qty = trade_usd / p_sol_1;
    let doge_trade_qty = trade_usd / p_doge_1;

    let btc_depth_qty = 50_000.0 / p_btc_1;
    let sol_depth_qty = 50_000.0 / p_sol_1;
    let doge_depth_qty = 50_000.0 / p_doge_1;

    let re_btc = engine_btc.update(
        p_btc_1,
        p_btc_1 * (1.0 + spread_pct),
        btc_depth_qty,
        btc_depth_qty,
        btc_trade_qty,
        atr_pct,
        t1,
    );

    let re_sol = engine_sol.update(
        p_sol_1,
        p_sol_1 * (1.0 + spread_pct),
        sol_depth_qty,
        sol_depth_qty,
        sol_trade_qty,
        atr_pct,
        t1,
    );

    let re_doge = engine_doge.update(
        p_doge_1,
        p_doge_1 * (1.0 + spread_pct),
        doge_depth_qty,
        doge_depth_qty,
        doge_trade_qty,
        atr_pct,
        t1,
    );

    assert!(re_btc.is_finite() && re_sol.is_finite() && re_doge.is_finite());
    assert!(
        (re_btc - re_sol).abs() < 1e-6,
        "Invarianza dimensional violada entre BTC ({}) y SOL ({})",
        re_btc,
        re_sol
    );
    assert!(
        (re_btc - re_doge).abs() < 1e-6,
        "Invarianza dimensional violada entre BTC ({}) y DOGE ({})",
        re_btc,
        re_doge
    );
}

#[test]
fn test_navier_stokes_reynolds_shear_thinning_viscosity() {
    // Efecto Reológico de Adelgazamiento por Corte (Shear-Thinning):
    // Cuando el flujo agresor taker domina la profundidad del libro, la resistencia
    // disipativa pasiva colapsa, reduciendo la viscosidad cinemática nu y disparando Re.

    let mut engine_calm = NavierStokesReynoldsEngine::new();
    let mut engine_shear = NavierStokesReynoldsEngine::new();

    let p = 100.0;
    let t0 = 1_000_000u64;
    engine_calm.update(p, p + 0.10, 10.0, 10.0, 0.0, 0.0010, t0);
    engine_shear.update(p, p + 0.10, 10.0, 10.0, 0.0, 0.0010, t0);

    let t1 = t0 + 100;
    let p_next = p + 0.20; // Movimiento idéntico de precio

    // Calm: sin flujo agresor
    let re_calm = engine_calm.update(p_next, p_next + 0.10, 10.0, 10.0, 0.0, 0.0010, t1);
    // Shear: flujo agresor masivo (trade_qty = 500)
    let re_shear = engine_shear.update(p_next, p_next + 0.10, 10.0, 10.0, 500.0, 0.0010, t1);

    assert!(
        re_shear > re_calm,
        "El flujo agresor debe aumentar estrictamente el número de Reynolds: calm={}, shear={}",
        re_calm,
        re_shear
    );
    assert!(
        engine_shear.kinematic_viscosity < engine_calm.kinematic_viscosity,
        "El flujo agresor debe reducir la viscosidad cinemática por corte: visc_calm={}, visc_shear={}",
        engine_calm.kinematic_viscosity,
        engine_shear.kinematic_viscosity
    );
}

#[test]
fn test_navier_stokes_reynolds_laminar_share_monotonic_c_infinity() {
    // Contrato de Suavidad y Monotonía C^∞:
    // laminar_share = 1 / (1 + (Re / Re_crit)^2)
    // - Para Re = 0: laminar_share == 1.0
    // - Para Re = Re_crit (1.0): laminar_share == 0.50
    // - Para Re -> inf: laminar_share -> 0.0
    // - Estrictamente monótona decreciente.

    let mut prev_share = 1.05;
    for step in 0..100 {
        let re = step as f64 * 0.10;
        let share = NavierStokesReynoldsEngine::compute_laminar_share(re);

        assert!(
            share >= 0.0 && share <= 1.0,
            "laminar_share fuera de cota [0, 1]: {}",
            share
        );
        assert!(
            share <= prev_share,
            "Violación de monotonicidad decreciente en Re={}: prev={}, current={}",
            re,
            prev_share,
            share
        );
        prev_share = share;
    }

    assert_eq!(NavierStokesReynoldsEngine::compute_laminar_share(0.0), 1.0);
    assert_eq!(NavierStokesReynoldsEngine::compute_laminar_share(1.0), 0.50);
    assert!(NavierStokesReynoldsEngine::compute_laminar_share(5.0) < 0.04);
}

#[test]
fn test_navier_stokes_reynolds_kolmogorov_dissipation_positive_and_finite() {
    let mut engine = NavierStokesReynoldsEngine::new();
    let p = 50_000.0;

    engine.update(p, p + 1.0, 10.0, 10.0, 0.0, 0.0010, 1_000_000);
    let _ = engine.update(p + 50.0, p + 51.0, 10.0, 10.0, 50.0, 0.0010, 1_000_100);

    let eps = engine.energy_dissipation_rate;
    assert!(eps >= 0.0, "La disipación de Kolmogorov debe ser no-negativa: {}", eps);
    assert!(eps.is_finite(), "La disipación debe ser finita");
}

#[test]
fn test_navier_stokes_reynolds_fail_closed_nan_inf_immunity() {
    let mut engine = NavierStokesReynoldsEngine::new();

    // Entradas patológicas
    let re1 = engine.update(f64::NAN, 100.0, 1.0, 1.0, 0.0, 0.001, 1000);
    assert_eq!(re1, 0.0);

    let re2 = engine.update(100.0, f64::INFINITY, 1.0, 1.0, 0.0, 0.001, 1050);
    assert_eq!(re2, 0.0);

    // Spread negativo o invertido (ask <= bid)
    let re3 = engine.update(100.0, 99.0, 1.0, 1.0, 0.0, 0.001, 1100);
    assert_eq!(re3, 0.0);

    // Precios negativos
    let re4 = engine.update(-10.0, -9.0, 1.0, 1.0, 0.0, 0.001, 1150);
    assert_eq!(re4, 0.0);

    assert!(engine.reynolds_number.is_finite());
    assert!(engine.laminar_share.is_finite());
    assert!(engine.energy_dissipation_rate.is_finite());
    assert_eq!(engine.regime(), HydrodynamicRegime::Laminar);
}

#[test]
fn test_navier_stokes_reynolds_relaxation_time_adaptability() {
    // Escalamiento lineal con el tiempo de relajación característico τ_relax:
    // Re = (|u| * τ_relax * agresión) / L
    // Si τ_relax se duplica, Re se duplica exactamente.

    let engine_base = NavierStokesReynoldsEngine::new().with_relaxation_time_s(0.100);
    let engine_double = NavierStokesReynoldsEngine::new().with_relaxation_time_s(0.200);

    let mut e1 = engine_base;
    let mut e2 = engine_double;

    let p = 1000.0;
    e1.update(p, p + 1.0, 5.0, 5.0, 0.0, 0.001, 1_000_000);
    e2.update(p, p + 1.0, 5.0, 5.0, 0.0, 0.001, 1_000_000);

    let re1 = e1.update(p + 2.0, p + 3.0, 5.0, 5.0, 0.0, 0.001, 1_000_050);
    let re2 = e2.update(p + 2.0, p + 3.0, 5.0, 5.0, 0.0, 0.001, 1_000_050);

    assert!(
        (re2 - 2.0 * re1).abs() < 1e-9,
        "Duplicar tau_relax debe duplicar Re exactamente: re1={}, re2={}",
        re1,
        re2
    );
}
