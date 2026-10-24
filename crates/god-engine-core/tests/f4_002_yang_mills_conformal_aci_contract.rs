//! ⚡ CONTRATO FORMAL DE CAUSALIDAD GAUGE YANG-MILLS & CONFORMAL PREDICTION ACI (OLA Ω220 - ÁTOMO 4.2)
//!
//! Marco Teórico & Verificación Formal:
//! 1. Aceleración Adaptativa de tasa $\gamma_{\text{ACI}}(\mathcal{S}_{\text{YM}}) = \gamma_0 (1 + \tanh(\mathcal{S}_{\text{YM}}))$.
//! 2. Ensanchamiento de cota $\alpha_{\text{gauge}} = \alpha_{\text{eff}} (1 + 0.5 \min(\bar{H}, 1.0))$ ante pérdida de intercambiabilidad local.
//! 3. Veto direccional por colisión contra corriente restauradora $\mathcal{J}_i$ ($J_{\text{align}} < -0.70$).
//! 4. Inmunidad a valores no finitos (NaN, Inf) y trinquete estricto de pánicos.
//! 5. Publicación viva y lectura no nula en OmniscientRegistry tras tick dual en GodEngineCore.

use god_engine_core::conformal::ConformalCalibrator;
use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::atomic::Ordering::Relaxed;

#[test]
fn test_c1_yang_mills_action_accelerates_conformal_aci_adaptation() {
    let mut c_flat = ConformalCalibrator::new();
    let mut c_dislocated = ConformalCalibrator::new();
    c_flat.set_target_alpha(0.10);
    c_dislocated.set_target_alpha(0.10);

    // Warmup de 40 observaciones idénticas
    for _ in 0..40 {
        c_flat.update(0.85, true);
        c_dislocated.update(0.85, true);
    }
    assert!(c_flat.is_warm());
    assert!(c_dislocated.is_warm());
    assert!((c_flat.effective_alpha() - c_dislocated.effective_alpha()).abs() < 1e-12);

    // En calma gauge (S_YM = 0.0), gamma = gamma_0 = 1/200 = 0.005
    c_flat.update_with_gauge(0.85, false, 0.0);
    let gamma_flat = c_flat.last_gamma();
    assert!((gamma_flat - 0.005).abs() < 1e-9);

    // Con dislocación gauge activa (S_YM = 3.0), gamma = gamma_0 * (1 + tanh(3.0)) ~ 0.005 * 1.995 ~ 0.00997
    c_dislocated.update_with_gauge(0.85, false, 3.0);
    let gamma_dislocated = c_dislocated.last_gamma();
    assert!(
        gamma_dislocated > gamma_flat * 1.90,
        "gamma con S_YM=3.0 debe ser casi el doble de gamma_0: {gamma_dislocated}"
    );

    // Ante el fallo (err=1.0), el calibrador con dislocación reduce alpha_eff sustancialmente más rápido
    assert!(
        c_dislocated.effective_alpha() < c_flat.effective_alpha(),
        "Dislocación gauge debe acelerar la contracción de alpha_eff ante errores: dislocado={:.6} vs plano={:.6}",
        c_dislocated.effective_alpha(),
        c_flat.effective_alpha()
    );
}

#[test]
fn test_c2_holonomy_density_widens_conformal_significance_band() {
    let mut c = ConformalCalibrator::new();
    c.set_target_alpha(0.10);

    for _ in 0..50 {
        c.update(0.85, true);
    }
    assert!(c.is_warm());

    // Curvatura plana (H = 0.0): alpha_gauge coincide con alpha_eff
    let (_p_flat, _acc_flat, a_flat, _) = c.eval_prediction_gauge(0.85, 0.0, 0.0, true);
    assert!((a_flat - c.effective_alpha()).abs() < 1e-9);

    // Holonomía moderada (H = 0.50): alpha_gauge = alpha_eff * (1 + 0.25)
    let (_p_mid, _acc_mid, a_mid, _) = c.eval_prediction_gauge(0.85, 0.50, 0.0, true);
    assert!((a_mid - c.effective_alpha() * 1.25).abs() < 1e-9);

    // Holonomía extrema (H >= 1.0): alpha_gauge saturado a alpha_eff * 1.50
    let (_p_max, _acc_max, a_max, _) = c.eval_prediction_gauge(0.85, 1.80, 0.0, true);
    assert!((a_max - c.effective_alpha() * 1.50).abs() < 1e-9);
    assert!(a_max > a_mid && a_mid > a_flat);
}

#[test]
fn test_c3_conformal_gate_veto_symmetry_on_opposing_gauge_current() {
    let mut c = ConformalCalibrator::new();
    c.set_target_alpha(0.10);

    for _ in 0..50 {
        c.update(0.90, true);
    }
    assert!(c.is_warm());

    // 1. Long con corriente alineada (+0.50) -> admitido
    let (_p, acc_long_ok, _, j_long_ok) = c.eval_prediction_gauge(0.90, 0.0, 0.50, true);
    assert!(acc_long_ok);
    assert!((j_long_ok - 0.50).abs() < 1e-12);

    // 2. Long con corriente fuertemente opuesta (-0.71) -> VETO CONFORMAL ESTRICTO
    let (_p, acc_long_vetoed, _, j_long_vetoed) = c.eval_prediction_gauge(0.90, 0.0, -0.71, true);
    assert!(
        !acc_long_vetoed,
        "Long en contra de corriente gauge < -0.70 debe ser vetado"
    );
    assert!((j_long_vetoed - (-0.71)).abs() < 1e-12);

    // 3. Short con corriente alineada (-0.50 -> j_align = +0.50) -> admitido
    let (_p, acc_short_ok, _, j_short_ok) = c.eval_prediction_gauge(0.90, 0.0, -0.50, false);
    assert!(acc_short_ok);
    assert!((j_short_ok - 0.50).abs() < 1e-12);

    // 4. Short con corriente fuertemente opuesta (+0.75 -> j_align = -0.75) -> VETO CONFORMAL ESTRICTO
    let (_p, acc_short_vetoed, _, j_short_vetoed) = c.eval_prediction_gauge(0.90, 0.0, 0.75, false);
    assert!(
        !acc_short_vetoed,
        "Short en contra de corriente gauge > +0.70 debe ser vetado"
    );
    assert!((j_short_vetoed - (-0.75)).abs() < 1e-12);
}

#[test]
fn test_c4_gauge_conformal_nan_inf_immunity() {
    let mut c = ConformalCalibrator::new();

    // Ingesta de no-conformidad con valores degenerados
    c.update_with_gauge(f64::NAN, true, 1.0);
    c.update_with_gauge(0.8, false, f64::NAN);
    c.update_with_gauge(0.8, false, f64::INFINITY);
    assert_eq!(
        c.observations(),
        2,
        "NaN en p_hat debe ser rechazado, NaN/Inf en S_YM debe ser clamped a 0.0"
    );

    // Consulta con parámetros no finitos
    let (p1, acc1, a1, j1) = c.eval_prediction_gauge(f64::NAN, 0.5, 0.5, true);
    assert_eq!(p1, 0.0);
    assert!(!acc1);
    assert!(a1.is_finite());
    assert_eq!(j1, 0.0);

    let (p2, _acc2, a2, j2) = c.eval_prediction_gauge(0.8, f64::NAN, f64::NAN, true);
    assert!(p2.is_finite());
    assert!(a2.is_finite());
    assert_eq!(j2, 0.0);

    let (p3, _acc3, a3, j3) = c.eval_prediction_gauge(0.8, f64::INFINITY, -f64::INFINITY, true);
    assert!(p3.is_finite());
    assert!(a3.is_finite());
    assert_eq!(j3, -1.0, "-Inf en corriente debe clampear limpiamente a -1.0");
}

#[test]
fn test_c5_god_engine_registry_conformal_gauge_observables() {
    const SIM: &str = "BTCUSDT";
    symbols::update_dynamic_universe(vec![SIM.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIM)]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    arena.config.conformal_alpha.store(0.08, Relaxed);

    // 40 observaciones para calentar el calibrador
    for i in 0..40 {
        core.conformal_by_coin[0].update((i as f64 + 0.5) / 41.0, i % 2 == 0);
    }

    // Ejecutar tick dual
    let _ = core.process_tick_dual(0, 50_000.0, 50_001.0, 5.0, 5.0, 10_000, &[0.0; 54], true, false);

    let reg = &arena.registry;

    // Verificar que las nuevas claves están publicadas en el registro y son finitas
    let alpha_gauge = reg.get_for_coin_strict_or(0, "conformal_alpha_gauge", f64::NAN);
    let gamma_aci = reg.get_for_coin_strict_or(0, "conformal_gamma_aci", f64::NAN);
    let j_long = reg.get_for_coin_strict_or(0, "conformal_ym_alignment_long", f64::NAN);
    let j_short = reg.get_for_coin_strict_or(0, "conformal_ym_alignment_short", f64::NAN);
    let veto_long = reg.get_for_coin_strict_or(0, "conformal_gauge_veto_long", f64::NAN);
    let veto_short = reg.get_for_coin_strict_or(0, "conformal_gauge_veto_short", f64::NAN);

    assert!(
        alpha_gauge.is_finite() && alpha_gauge > 0.0 && alpha_gauge <= 0.50,
        "conformal_alpha_gauge válido: {alpha_gauge}"
    );
    assert!(
        gamma_aci.is_finite() && gamma_aci > 0.0 && gamma_aci <= 0.10,
        "conformal_gamma_aci válido: {gamma_aci}"
    );
    assert!(
        j_long.is_finite() && (-1.0..=1.0).contains(&j_long),
        "conformal_ym_alignment_long válido: {j_long}"
    );
    assert!(
        j_short.is_finite() && (-1.0..=1.0).contains(&j_short),
        "conformal_ym_alignment_short válido: {j_short}"
    );
    assert!(
        veto_long == 0.0 || veto_long == 1.0,
        "conformal_gauge_veto_long binario: {veto_long}"
    );
    assert!(
        veto_short == 0.0 || veto_short == 1.0,
        "conformal_gauge_veto_short binario: {veto_short}"
    );
    assert!(
        (j_long + j_short).abs() < 1e-12,
        "Simetría gauge antisimétrica: J_long + J_short == 0"
    );
}
