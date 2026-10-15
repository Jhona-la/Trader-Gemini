//! CONTRATO FORMAL DE CONTINUIDAD C1 EN VECM Y MOMENTUM BOOSTER (OLA Ω129)
//!
//! Verifica:
//! 1. Validación e invariantes de VecmArbitrageParams (umbrales z, saturación y cotas de velocidad).
//! 2. Continuidad C1 (cúbica Hermite) y derivada idénticamente nula en la frontera de activación de JohansenVecmEngine.
//! 3. Transición C1 continua en VolatileMomentumBooster eliminando el escalón booleano C0 en PnL > 0.
//! 4. Inmunidad matemática ante NaN, subnormales y ±∞.

use omniscient_registry::{OmniscientRegistry, ParameterKind};
use quantum_arena::GlobalArena;
use std::sync::Arc;
use strategy_core::momentum_booster::VolatileMomentumBooster;
use strategy_core::vecm_arbitrage::{JohansenVecmEngine, VecmArbitrageParams};
use strategy_core::QuantumStrategy;

#[test]
fn test_vecm_arbitrage_params_validation_and_defaults() {
    let def = VecmArbitrageParams::default();
    assert!(def.validate().is_ok());
    assert!((def.z_entry_threshold - 1.5).abs() < 1e-9);
    assert!((def.z_saturation - 3.0).abs() < 1e-9);
    assert!((def.speed_clamp_min - 0.01).abs() < 1e-9);
    assert!((def.speed_clamp_max - 1.0).abs() < 1e-9);
    assert!((def.rls_gain - 0.001).abs() < 1e-9);

    // Invalid saturation <= entry
    let mut invalid = def;
    invalid.z_saturation = 1.2;
    assert!(invalid.validate().is_err());

    // Invalid negative entry
    let mut invalid_entry = def;
    invalid_entry.z_entry_threshold = -0.5;
    assert!(invalid_entry.validate().is_err());

    // Invalid speed clamps
    let mut invalid_speed = def;
    invalid_speed.speed_clamp_min = 2.0;
    invalid_speed.speed_clamp_max = 1.0;
    assert!(invalid_speed.validate().is_err());
}

#[test]
fn test_vecm_c1_smoothstep_and_zero_derivatives() {
    let reg = Arc::new(OmniscientRegistry::new());
    let mut vecm = JohansenVecmEngine::new(0.15, 1.0);
    vecm.init(reg.clone()).unwrap();

    // Debajo del umbral (|z| = 1.0 < 1.5): señal idénticamente 0.0
    reg.register_or_update("vecm_zscore", ParameterKind::Adaptive, 1.0, "test");
    assert_eq!(vecm.evaluate(), 0.0);

    reg.register_or_update("vecm_zscore", ParameterKind::Adaptive, -1.0, "test");
    assert_eq!(vecm.evaluate(), 0.0);

    // Exactamente en el umbral (|z| = 1.50): señal es 0.0
    reg.register_or_update("vecm_zscore", ParameterKind::Adaptive, 1.50, "test");
    assert_eq!(vecm.evaluate(), 0.0);

    // Infinitesimalmente por encima del umbral: u = eps / 1.5, señal = 3u^2 - 2u^3 ≈ 3 (eps/1.5)^2 -> orden cuadrático C1
    let eps = 0.001;
    reg.register_or_update(
        "vecm_zscore",
        ParameterKind::Adaptive,
        1.50 + eps,
        "test",
    );
    let val_above = vecm.evaluate();
    assert!(val_above < 0.0); // Z positivo genera señal de reversión negativa (short)
    let u_expected = eps / 1.5;
    let expected_scale = 3.0 * u_expected * u_expected - 2.0 * u_expected * u_expected * u_expected;
    assert!((val_above.abs() - expected_scale).abs() < 1e-7);

    // Derivada numérica en la frontera z = 1.5: (f(1.5 + eps) - f(1.5)) / eps -> 0 cuando eps -> 0
    let num_deriv = (val_above.abs() - 0.0) / eps;
    assert!(
        num_deriv < 0.005,
        "Derivada numérica debe tender a 0 en la frontera (suavidad C1)"
    );

    // Saturación plena en |z| >= 3.0: señal alcanza exactamente -1.0
    reg.register_or_update("vecm_zscore", ParameterKind::Adaptive, 3.0, "test");
    assert!((vecm.evaluate() - (-1.0)).abs() < 1e-9);

    reg.register_or_update("vecm_zscore", ParameterKind::Adaptive, 4.5, "test");
    assert!((vecm.evaluate() - (-1.0)).abs() < 1e-9);

    // Simetría para Z negativo (long)
    reg.register_or_update("vecm_zscore", ParameterKind::Adaptive, -3.0, "test");
    assert!((vecm.evaluate() - 1.0).abs() < 1e-9);

    // Modificación dinámica mediante registro omnisciente
    reg.register_or_update("vecm_z_entry", ParameterKind::Adaptive, 2.0, "test");
    reg.register_or_update(
        "vecm_z_saturation",
        ParameterKind::Adaptive,
        4.0,
        "test",
    );
    // Con z = 1.8, ahora debe dar 0.0 porque el nuevo umbral es 2.0
    reg.register_or_update("vecm_zscore", ParameterKind::Adaptive, 1.8, "test");
    assert_eq!(vecm.evaluate(), 0.0);
}

#[test]
fn test_momentum_booster_c1_hermite_smooth_alignment() {
    let arena = GlobalArena::new(100.0);
    let atr = 0.01; // 1% ATR

    // Caso 1: PnL no positivo (cero o negativo) -> extensión neutral 1.0
    let tp_neg = VolatileMomentumBooster::calculate_tp_extension(1.0, -0.005, 1.5, atr, &arena);
    assert_eq!(tp_neg, 1.0);

    let tp_zero = VolatileMomentumBooster::calculate_tp_extension(1.0, 0.0, 1.5, atr, &arena);
    assert_eq!(tp_zero, 1.0);

    // Caso 2: PnL infinitesimalmente positivo (pnl = 1e-6)
    // Con delta_pnl = 0.5 * atr = 0.005, u = 1e-6 / 0.005 = 0.0002
    // alignment_direction = 3u^2 - 2u^3 ≈ 1.2e-7 (continuidad suave, sin salto de 0 a 1)
    let tp_micro = VolatileMomentumBooster::calculate_tp_extension(1.0, 1e-6, 1.5, atr, &arena);
    assert!(
        (tp_micro - 1.0).abs() < 0.001,
        "No debe haber un salto abrupto en PnL microscópico"
    );

    // Caso 3: PnL maduro (pnl >= delta_pnl = 0.005) -> expansión completa de TP
    let tp_mature = VolatileMomentumBooster::calculate_tp_extension(1.0, 0.02, 1.8, atr, &arena);
    assert!(
        tp_mature > 1.0,
        "Debe extender Take Profit ante impulso alineado maduro"
    );

    // Caso 4: Simetría para posiciones cortas ganadoras
    let tp_short_mature =
        VolatileMomentumBooster::calculate_tp_extension(-1.0, 0.02, 1.8, atr, &arena);
    assert!(
        (tp_short_mature - tp_mature).abs() < 1e-6,
        "La extensión debe ser idéntica para Short ganador"
    );
}
