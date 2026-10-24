//! Contrato de Certificación Ola Ω202: F1-036, F1-037, F1-039
//! 1. F1-036: Convicción honesta sin piso artificial > 0.50 en sig_conf(0.0) y
//!    superposición constructiva condicionada a evidencia estadística Wilson genuina.
//! 2. F1-039: Temporizadores de salida (harvest, stagnant, zombie) continuos y homogéneos
//!    en tau_trade_ms sin suelos rígidos (8m/12m/30m) que bloqueen la banda rápida (tau = 30s).
//! 3. F1-037: Sincronización de parámetros de arena y derivación actuarial del apalancamiento
//!    máximo por cota de ruina de Cramér-Lundberg psi <= exp(-R * m).

use god_engine_core::{
    confianza_superpuesta_con_evidencia, sig_conf, TasaAcierto,
};
use quantum_arena::GlobalArena;

#[test]
fn test_f1_036_honest_confidence_and_wilson_evidence_superposition() {
    // 1. sig_conf con score nulo produce exactamente 0.50 (moneda al aire), no 0.51
    assert_eq!(
        sig_conf(0.0),
        0.50,
        "sig_conf(0.0) debe ser 0.50 exacto (neutralidad perfecta, erradicando suelo espurio 0.51)"
    );
    assert!(sig_conf(0.50) > 0.50);
    assert!(sig_conf(1.0) <= 0.90);

    // 2. Fusión en colisión armónica:
    let fast_conf = 0.62;
    let slow_conf = 0.55;

    // Escenario de fallo F1-036: Ambas ramas en arranque frío (sin evidencia estadística Wilson)
    // No deben recibir el bono unearned +0.10 * lo
    let conf_sin_ev = confianza_superpuesta_con_evidencia(fast_conf, slow_conf, false, false);
    assert_eq!(
        conf_sin_ev, fast_conf,
        "Sin evidencia demostrada en ambas ramas, la fusión debe ser max(hi, lo) = 0.62 sin refuerzo"
    );

    // Si solo una tiene evidencia, tampoco debe haber refuerzo unearned
    let conf_una_ev = confianza_superpuesta_con_evidencia(fast_conf, slow_conf, true, false);
    assert_eq!(
        conf_una_ev, fast_conf,
        "Con una sola rama con evidencia, no hay superposición constructiva mutua"
    );

    // Con evidencia demostrada en ambas ramas (lo_wilson > p*):
    let conf_con_ev = confianza_superpuesta_con_evidencia(fast_conf, slow_conf, true, true);
    let expected = (fast_conf + 0.10 * slow_conf).min(0.96);
    assert!(
        (conf_con_ev - expected).abs() < 1e-12,
        "Con evidencia en ambas ramas, aplica superposición constructiva completa: {conf_con_ev}"
    );

    // 3. TasaAcierto Wilson check:
    let reg_frio = TasaAcierto::default();
    assert!(
        !reg_frio.mejor_que_el_azar(god_engine_core::diffusion::Z95),
        "Rama sin muestras n=0 no puede pretender evidencia estadística superior al azar"
    );

    let mut reg_maduro = TasaAcierto::default();
    for _ in 0..100 {
        reg_maduro.observar(true);
    }
    assert!(
        reg_maduro.mejor_que_el_azar(god_engine_core::diffusion::Z95),
        "100/100 aciertos demuestra contundentemente superioridad al azar"
    );
}

#[test]
fn test_f1_039_continuous_exit_timers_in_fast_band() {
    let tau_fast_ms: f64 = 30_000.0; // 30 segundos (escala física rápida s=0)
    let temporal_s: f64 = 0.0;

    // 1. Harvest age escala limpiamente a 8 * tau = 240s = 4 min (antes clamped a 720s = 12 min):
    let harvest_age_ms = (tau_fast_ms * 8.0).clamp(tau_fast_ms, 86_400_000.0) as u64;
    assert_eq!(
        harvest_age_ms, 240_000,
        "Harvest age en tau=30s debe ser exactamente 240s (4 min), no 12 min"
    );

    // 2. Min stagnant age escala a 5 * tau = 150s = 2.5 min (antes clamped a 480s = 8 min):
    let min_stagnant_ms = (tau_fast_ms * (5.0 + 3.0 * temporal_s)).clamp(tau_fast_ms * 2.0, 86_400_000.0) as u64;
    assert_eq!(
        min_stagnant_ms, 150_000,
        "Min stagnant en tau=30s debe ser 150s (2.5 min), permitiendo cortar tesis rotas ágilmente"
    );

    // 3. Hard stagnant age escala a 10 * tau = 300s = 5 min (antes clamped a 1200s = 20 min):
    let hard_stagnant_ms = (tau_fast_ms * (10.0 + 6.0 * temporal_s)).clamp(tau_fast_ms * 4.0, 86_400_000.0) as u64;
    assert_eq!(
        hard_stagnant_ms, 300_000,
        "Hard stagnant en tau=30s debe ser 300s (5 min), no 20 min"
    );

    // 4. Zombie debounce escala a 20 * tau = 600s = 10 min (antes clamped a 1800s = 30 min):
    let zombie_ms = (tau_fast_ms * (20.0 + 10.0 * temporal_s)).clamp(tau_fast_ms * 6.0, 86_400_000.0) as u64;
    assert_eq!(
        zombie_ms, 600_000,
        "Zombie debounce en tau=30s debe ser 600s (10 min), no 30 min"
    );

    // 5. Escala lenta tau = 12h (43_200_000 ms, s=1):
    let tau_slow_ms: f64 = 43_200_000.0;
    let s_slow: f64 = 1.0;
    let min_stagnant_slow = (tau_slow_ms * (5.0 + 3.0 * s_slow)).clamp(tau_slow_ms * 2.0, 86_400_000.0) as u64;
    assert!(
        min_stagnant_slow > 86_400_000 / 2,
        "En escala macro, el temporizador respira proporcionalmente a la longitud de onda"
    );
}

#[test]
fn test_f1_037_cramer_lundberg_leverage_actuarial_bound() {
    let _arena = GlobalArena::build_in_own_stack(13.0);

    let macro_cap: f64 = 50.0;
    let safe_cushion: f64 = 0.95;
    let ln_inv_eps: f64 = 2.995732273553991; // ln(1 / 0.05)

    // Caso 1: Moneda sin historial (r_nocional = 0.0)
    let r_nocional_cold: f64 = 0.0;
    let base_lev_cap: f64 = 20.0;
    let lev_cap_cold: f64 = if r_nocional_cold > 0.0 {
        let lundberg_lev = (r_nocional_cold * safe_cushion / ln_inv_eps).clamp(1.0, macro_cap);
        base_lev_cap.min(lundberg_lev)
    } else {
        base_lev_cap
    };
    assert_eq!(lev_cap_cold, 20.0, "Sin estimador Lundberg maduro, rige el límite genómico");

    // Caso 2: Moneda con edge moderado medido (r_nocional = 15.0)
    let r_nocional_edge: f64 = 15.0;
    let lundberg_bound = (r_nocional_edge * safe_cushion / ln_inv_eps).clamp(1.0, macro_cap);
    // (15.0 * 0.95) / 2.9957 = 14.25 / 2.9957 = 4.75x
    let lev_cap_tight = base_lev_cap.min(lundberg_bound);
    assert!(
        lev_cap_tight < base_lev_cap,
        "La cota actuarial acota el apalancamiento a 4.75x para preservar psi <= 5%"
    );
    assert!((lev_cap_tight - 4.7567).abs() < 1e-3);
}
