//! CL-51: el pico del veto de drawdown es de la cuenta, no del feed.

use risk_engine::RiskEngine;

#[test]
fn cl51_el_pico_solo_se_rebasa_si_cambia_la_cuenta() {
    let mut r = RiskEngine::new(13.0);
    r.seguir_base_de_cuenta(13.0);
    r.peak_capital = 20.0;
    // Misma cuenta: el máximo alcanzado se conserva.
    r.seguir_base_de_cuenta(13.0);
    assert_eq!(r.peak_capital, 20.0);
    // Bases no válidas no tocan nada.
    r.seguir_base_de_cuenta(f64::NAN);
    r.seguir_base_de_cuenta(0.0);
    assert_eq!(r.peak_capital, 20.0);
    // Cuenta nueva (transición demo→mainnet): el pico parte de su base.
    r.seguir_base_de_cuenta(50.0);
    assert_eq!(r.peak_capital, 50.0);
}

/// La primera base sólo se anota: el pico de construcción se respeta.
#[test]
fn cl51_la_primera_base_no_mueve_el_pico() {
    let mut r = RiskEngine::new(100.0);
    r.seguir_base_de_cuenta(13.0);
    assert_eq!(r.peak_capital, 100.0);
}
