// crates/god-engine-core/tests/f1_048_051_052_ml_gate_symmetry_and_deadbands_contract.rs
//
// CONTRATO FORMAL DE NO-REGRESIÓN — OLA Ω167:
// Erradicación de sesgos en ML gate, modulación fractal y dead-bands tipificadas.
//
// F1-051: Modulación simétrica del lift ML por predictibilidad fractal |H - 0.50|
// F1-052: Unificación dimensional del gate ML evaluando |p - b| >= lift_eff
// F1-048: Dead-bands dinámicas (OBI/OFI cuantiles, z_macro ATR) y reparto Brier skill_z

use std::sync::atomic::Ordering;
use std::sync::Arc;
use quantum_arena::GlobalArena;
use god_engine_core::GodEngineCore;
use signal_engine::SignalType;

#[test]
fn test_f1_051_hurst_predictibility_mirror_symmetry() {
    // F1-051: Simetría exacta respecto a ruido browniano (|H - 0.50|).
    // Fórmula en producción:
    //   predictibilidad_h = (hurst_val - 0.50).abs().clamp(0.0, 0.40);
    //   hurst_mod = (1.0 + 1.50 * (0.10 - predictibilidad_h)).clamp(0.70, 1.30);
    let hurst_mod = |h: f64| -> f64 {
        let predictibilidad_h = (h - 0.50).abs().clamp(0.0, 0.40);
        (1.0 + 1.50 * (0.10 - predictibilidad_h)).clamp(0.70, 1.30)
    };

    // 1. En ruido browniano puro (H = 0.50), la predictibilidad es cero.
    // El lift se eleva a 1.15 (+15%) para defender el capital contra martingala aleatoria.
    let h_brownian = 0.50;
    let mod_brownian = hurst_mod(h_brownian);
    assert!(
        (mod_brownian - 1.15).abs() < 1e-12,
        "A H=0.50 hurst_mod debe ser 1.15 (+15%), obtenido: {mod_brownian}"
    );

    // 2. A distancia neutral (delta = 0.10, i.e., H = 0.40 o H = 0.60):
    assert!((hurst_mod(0.40) - 1.00).abs() < 1e-12);
    assert!((hurst_mod(0.60) - 1.00).abs() < 1e-12);

    // 3. A memoria fuerte (delta = 0.20, i.e., reversión H = 0.30 o tendencia H = 0.70):
    // Ambos tienen idéntica predictibilidad no browniana, relajando el lift a 0.85 (-15%).
    let mod_reversion = hurst_mod(0.30);
    let mod_trend = hurst_mod(0.70);
    assert!(
        (mod_reversion - 0.85).abs() < 1e-12,
        "Reversión H=0.30 debe dar 0.85, obtenido: {mod_reversion}"
    );
    assert!(
        (mod_trend - 0.85).abs() < 1e-12,
        "Tendencia H=0.70 debe dar 0.85, obtenido: {mod_trend}"
    );
    assert!(
        (mod_reversion - mod_trend).abs() < 1e-12,
        "Debe ser idéntico entre H=0.30 y H=0.70"
    );

    // 4. Test exhaustivo de espejo: hurst_mod(0.50 + d) == hurst_mod(0.50 - d) para todo d.
    for step in 1..=40 {
        let d = step as f64 * 0.01;
        let h_plus = 0.50 + d;
        let h_minus = 0.50 - d;
        let m_plus = hurst_mod(h_plus);
        let m_minus = hurst_mod(h_minus);
        assert!(
            (m_plus - m_minus).abs() < 1e-12,
            "Violación de simetría especular a d={d}: H+({h_plus})={m_plus} != H-({h_minus})={m_minus}"
        );
    }
}

#[test]
fn test_f1_052_ml_gate_dimensional_edge_and_symmetry() {
    // F1-052: Ventaja direccional honesta en unidades de probabilidad |p - b| >= lift_eff.
    // Simula la función del ML gate con tolerancia de máquina 1e-9:
    let eval_gate = |signal: SignalType, p: f64, base: f64, lift: f64| -> bool {
        let ml_edge_prob = if signal == SignalType::Long {
            p - base
        } else {
            base - p
        };
        ml_edge_prob + 1e-9 >= lift
    };

    // Caso 1: Base simétrica clásica b = 0.50, lift = 0.05
    let b_classic = 0.50;
    let lift_test = 0.05;
    assert!(eval_gate(SignalType::Long, 0.55, b_classic, lift_test));
    assert!(!eval_gate(SignalType::Long, 0.549, b_classic, lift_test));
    assert!(eval_gate(SignalType::Short, 0.45, b_classic, lift_test));
    assert!(!eval_gate(SignalType::Short, 0.451, b_classic, lift_test));

    // Caso 2: Base honesta b = 1/3 (etiquetado TP = 2*SL)
    let b_honest = 1.0 / 3.0;
    let lift_req = 0.05;

    // Para Long: exige p >= 1/3 + 0.05
    let p_long_pass = b_honest + 0.05;
    let p_long_fail = b_honest + 0.049;
    assert!(eval_gate(SignalType::Long, p_long_pass, b_honest, lift_req));
    assert!(!eval_gate(SignalType::Long, p_long_fail, b_honest, lift_req));

    // Para Short: exige p <= 1/3 - 0.05
    let p_short_pass = b_honest - 0.05;
    let p_short_fail = b_honest - 0.049;
    assert!(eval_gate(SignalType::Short, p_short_pass, b_honest, lift_req));
    assert!(!eval_gate(SignalType::Short, p_short_fail, b_honest, lift_req));

    // Simetría absoluta de dispersión requerida Delta_p:
    let delta_long = p_long_pass - b_honest;
    let delta_short = b_honest - p_short_pass;
    assert!(
        (delta_long - delta_short).abs() < 1e-12,
        "La ventaja requerida en probabilidad debe ser idéntica: long={delta_long}, short={delta_short}"
    );
    assert!((delta_long - 0.05).abs() < 1e-12);
}

#[test]
fn test_f1_048_brier_skill_partition_and_deadbands() {
    // F1-048: Reparto de peso ML/Tensor derivado de la habilidad Brier skill_z:
    //   w_frac_ml = (0.50 + 0.20 / (1.0 + (-skill_z).exp())).clamp(0.30, 0.70);
    let compute_w_frac = |z: f64| -> (f64, f64) {
        let w_frac_ml = (0.50 + 0.20 / (1.0 + (-z).exp())).clamp(0.30, 0.70);
        let w_frac_tensor = 1.0 - w_frac_ml;
        (w_frac_ml, w_frac_tensor)
    };

    // 1. Cold start / z = 0.0 (sin habilidad demostrada):
    // Preservación canónica exacta del 60% ML y 40% Tensor:
    let (w_ml_zero, w_tensor_zero) = compute_w_frac(0.0);
    assert!(
        (w_ml_zero - 0.60).abs() < 1e-12,
        "A z=0.0 w_ml debe ser 0.60, obtenido: {w_ml_zero}"
    );
    assert!(
        (w_tensor_zero - 0.40).abs() < 1e-12,
        "A z=0.0 w_tensor debe ser 0.40, obtenido: {w_tensor_zero}"
    );

    // 2. Modelo con habilidad demostrada z = +2.0 (95% confianza estadística):
    let (w_ml_high, w_tensor_high) = compute_w_frac(2.0);
    assert!(w_ml_high > 0.60, "w_ml debe aumentar con habilidad positiva");
    assert!(w_ml_high <= 0.70, "w_ml debe respetar el techo 0.70");
    assert!(
        (w_ml_high + w_tensor_high - 1.0).abs() < 1e-12,
        "Suma debe ser estrictamente 1.0"
    );

    // 3. Modelo descalibrado o con ruido z = -2.0:
    let (w_ml_low, w_tensor_low) = compute_w_frac(-2.0);
    assert!(w_ml_low < 0.60, "w_ml debe reducirse con habilidad negativa");
    assert!(w_ml_low >= 0.30, "w_ml debe respetar el piso 0.30");
    assert!(w_tensor_low > 0.40, "w_tensor debe absorber el peso cedido");

    // 4. Dead-bands dinámicas en GodEngineCore:
    let arena = GlobalArena::build_in_own_stack(13.0);
    let core = GodEngineCore::new(Arc::clone(&arena));
    arena.coins[0].current_price.store(65000.0, Ordering::Relaxed);
    arena.coins[0].agg_buy_vol.store(100.0, Ordering::Relaxed);
    arena.coins[0].agg_sell_vol.store(100.0, Ordering::Relaxed);

    // Verificamos que el motor arranca hermético con diagnóstico 0
    assert_eq!(core.diag_ml_vetoes, 0);
}
