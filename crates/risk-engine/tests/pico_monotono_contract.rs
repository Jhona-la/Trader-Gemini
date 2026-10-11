//! CL-51b — el pico del veto de drawdown es monótono dentro del proceso.
//!
//! La transición del calentamiento vuelve a publicar `base_capital` con el
//! balance de la MISMA cuenta (el entorno no cambia en caliente). Re-basar
//! el pico cada vez que esa base cambia (CL-51, primera versión) olvidaba
//! una caída real por un funding o una comisión de la espera.

use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

#[test]
fn cl51b_republicar_la_base_no_rebasa_el_pico() {
    let a = GlobalArena::build_in_own_stack(13.0);
    a.unified_capital.store(15.0, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        ..Default::default()
    };
    let mut r = RiskEngine::new(13.0);
    r.peak_capital = 20.0;
    let _ = r.evaluate_quantum_order(0, &intent, &a);
    // La transición publica el balance de la misma cuenta tras un funding.
    a.config.base_capital.store(12.98, Relaxed);
    let _ = r.evaluate_quantum_order(0, &intent, &a);
    assert_eq!(r.peak_capital, 20.0, "el máximo alcanzado no se olvida");
}
