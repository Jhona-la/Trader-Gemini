//! Contract test for Ola Ω171: F1-062 (Drift Accounting & Absorbing State Prevention)
//! and F1-063 (Warm Multi-Scale K-line/ATR Estimator Preservation on WS Reconnect).

use audit_engine::drift_auditor::{DriftAuditError, DriftAuditor, DriftRecovery, TradeResult};
use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::num::NonZeroU32;

/// F1-063: Verifica que `reset_engines` conserva el calentamiento multi-escala
/// de K-lines (v_t, EMAs de kline, sigma60) tras una reconexión de WS,
/// mientras limpia adecuadamente los buffers micro-estructurales del feed.
#[test]
fn f1_063_reconnect_preserves_warm_kline_and_atr_calibration() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut core = GodEngineCore::new(arena);

    // Inyectar 60 K-lines para calentar el motor de la moneda 0 (BTC)
    let mut base_price = 60000.0;
    for _i in 0..60 {
        let open = base_price;
        let high = base_price + 50.0;
        let low = base_price - 40.0;
        let close = base_price + 10.0;
        let vol = 15.0;
        core.feature_engines[0].process_kline(open, high, low, close, vol);
        base_price = close;
    }

    let warm_vt = core.feature_engines[0].v_t;
    let warm_vt_rasgo_ml = core.feature_engines[0].vt_rasgo_ml;
    let warm_sigma = core.feature_engines[0].sigma60.sigma_60s();

    assert!(warm_vt > 0.0, "Precondición: v_t debe estar calibrado y ser > 0");
    assert!(warm_vt_rasgo_ml > 0.0, "Precondición: vt_rasgo_ml debe ser > 0");
    assert!(warm_sigma.is_some_and(|s| s > 0.0), "Precondición: sigma60 debe ser > 0");

    // Simular evento de micro-estructura
    core.feature_engines[0].dir_velocity = 12.5;
    core.feature_engines[0].a_t = 3.2;

    // Simular centinela de reconexión del WS (ejecuta reset_engines)
    core.reset_engines();

    // Verificación: los estimadores de volatilidad y macro-tendencia se PRESERVAN
    assert_eq!(
        core.feature_engines[0].v_t, warm_vt,
        "F1-063: v_t debe preservarse idéntico tras reconexión"
    );
    assert_eq!(
        core.feature_engines[0].vt_rasgo_ml, warm_vt_rasgo_ml,
        "F1-063: vt_rasgo_ml debe preservarse idéntico tras reconexión"
    );
    assert_eq!(
        core.feature_engines[0].sigma60.sigma_60s(), warm_sigma,
        "F1-063: sigma60 debe preservarse idéntico tras reconexión"
    );

    // Verificación: buffers micro-estructurales del feed se limpiaron
    assert_eq!(core.feature_engines[0].dir_velocity, 0.0);
    assert_eq!(core.feature_engines[0].a_t, 0.0);

    // Verificación: un reset frío total sí resetea todo a 0
    core.reset_engines_cold();
    assert_eq!(core.feature_engines[0].v_t, 0.0, "reset_engines_cold debe reiniciar v_t a 0");
    assert_eq!(core.feature_engines[0].vt_rasgo_ml, 0.0, "reset_engines_cold debe reiniciar vt_rasgo_ml a 0");
    assert_eq!(core.feature_engines[0].kline_ema_fast, 0.0, "reset_engines_cold debe reiniciar EMAs a 0");
}

/// F1-062: Verifica la paridad estricta entre `centinela_contable` en O(1)
/// y la sombra sintética previa `audit_execution_checked` con `real * 0.95`.
#[test]
fn f1_062_centinela_contable_exact_parity_with_shadow_synthetic() {
    let auditor_a = DriftAuditor::new(0.05);
    let auditor_b = DriftAuditor::new(0.05);

    let test_returns = [
        0.005,
        -0.003,
        0.045,
        -0.049,
        0.50,
        -0.90,
        1.00,
        1.25, // corrupto (> 100% no apalancado)
        -1.50, // corrupto
        f64::NAN,
        f64::INFINITY,
    ];

    for &ret in &test_returns {
        let real_tr = TradeResult {
            symbol_id: 0,
            is_long: true,
            entry_price: 100.0,
            exit_price: 100.0 * (1.0 + ret),
            pnl_pct: ret,
            timestamp_ms: 1000,
        };
        let shadow_tr = TradeResult {
            symbol_id: 0,
            is_long: true,
            entry_price: 100.0,
            exit_price: 100.0 * (1.0 + ret * 0.95),
            pnl_pct: ret * 0.95,
            timestamp_ms: 1000,
        };

        let legacy_res = auditor_a.audit_execution_checked(&real_tr, &shadow_tr);
        let centinela_res = auditor_b.centinela_contable(ret);

        match (legacy_res, centinela_res) {
            (Ok(d1), Ok(d2)) => {
                assert!((d1 - d2).abs() < 1e-12, "Paridad de drift para ret = {ret}: {d1} vs {d2}");
            }
            (Err(DriftAuditError::NonfinitePnl), Err(DriftAuditError::NonfinitePnl)) => {}
            (Err(DriftAuditError::Exceeded { drift: d1, limit: l1 }), Err(DriftAuditError::Exceeded { drift: d2, limit: l2 })) => {
                assert!((d1 - d2).abs() < 1e-12, "Paridad de exceeded drift para ret = {ret}");
                assert_eq!(l1, l2);
            }
            (a, b) => {
                panic!("Discrepancia para ret = {ret}: legacy={a:?}, centinela={b:?}");
            }
        }
    }
}

/// F1-062: Verifica que `DriftRecovery` no queda atrapado en un estado absorbente
/// permanente cuando la cartera tiene 0 posiciones abiertas.
#[test]
fn f1_062_drift_recovery_absorbing_state_prevention() {
    let mut recovery = DriftRecovery::new(NonZeroU32::new(10).unwrap());

    // 1. Incidente contable bloquea el veto
    recovery.observe_at(&Err(DriftAuditError::NonfinitePnl), 10_000);
    assert!(recovery.is_blocked());
    assert_eq!(recovery.blocked_since_ms(), 10_000);

    // 2. Si todavía hay posiciones abiertas (open_positions = 2), la inactividad no desbloquea
    assert!(!recovery.try_idle_recovery(80_000, 60_000, 2));
    assert!(recovery.is_blocked());

    // 3. Si no hay posiciones abiertas pero el tiempo transcurrido es menor al cooldown (30s < 60s)
    assert!(!recovery.try_idle_recovery(40_000, 60_000, 0));
    assert!(recovery.is_blocked());

    // 4. Con 0 posiciones abiertas y timeout de seguridad de 60s cumplido (10s + 60s = 70s)
    assert!(recovery.try_idle_recovery(70_000, 60_000, 0));
    assert!(!recovery.is_blocked(), "El veto de deriva debe liberarse limpiamente tras 60s de inactividad segura");
    assert_eq!(recovery.blocked_since_ms(), 0);
}
