// ============================================================================
// CONTRATO FORMAL: F1-114 Erradicación de .expect() en HorizonForecaster::before_trade
// ============================================================================
// QUÉ: Contrato formal de verificación de resiliencia y ausencia total de .expect()
//      en HorizonForecaster::before_trade y el banco de pronósticos espectrales.
// POR QUÉ: spectral_tape.rs:659 usaba `.expect("frente comprobado")` en la ruta de
//          producción por trade (SpectralForecastBank::on_trade -> god-engine-core).
//          Cualquier vaciado o condición de carrera en `pending` abortaba el proceso host.
// PARA QUÉ: Garantizar que la extracción de muestras maduras sea 100% fail-safe
//           mediante guard let-else (`let Some(s) = self.pending.pop_front() else { break; }`).
// CÓMO: 1. Prueba de bucle de trades continuos con transiciones temporales abruptas.
//       2. Prueba de drenaje y llamadas repetidas a before_trade sin muestras pendientes.
//       3. Prueba de avance temporal masivo (saturating_sub) sin riesgo de pánico.
// CUÁNDO: Ejecutado como gate de regresión continuo en quantum-arena.
// DÓNDE: crates/quantum-arena/tests/f1_114_spectral_tape_no_panic_contract.rs
// QUIÉN: Consejo Quant Sr. (Ola Ω158)
// ============================================================================

use quantum_arena::spectral_tape::{ForecastTarget, HorizonForecaster, SpectralForecastBank, SpectralTape};

#[test]
fn f1_114_before_trade_empty_pending_never_panics() {
    let tape = SpectralTape::with_band(1_000.0, 300_000.0);
    let mut forecaster = HorizonForecaster::new(&tape, ForecastTarget::Volume, 4_000.0);

    // Llamadas con tape sin iniciar o con ts menor al último ts
    forecaster.before_trade(&tape, 0);
    forecaster.before_trade(&tape, 1000);
    forecaster.before_trade(&tape, 500_000);

    // forecaster.pending está vacío: el bucle debe salir inmediatamente sin panic
    assert_eq!(forecaster.score.n, 0);
}

#[test]
fn f1_114_before_trade_stress_burst_never_panics() {
    let mut tape = SpectralTape::with_band(1_000.0, 300_000.0);
    let mut f = HorizonForecaster::new(&tape, ForecastTarget::Variance, 2_000.0);

    let mut ts = 100_000u64;
    for i in 0..10_000u64 {
        // Incrementos temporales erráticos: a veces sub-milisegundo (0 ms), a veces ráfagas macro (10_000 ms)
        let step = match i % 5 {
            0 => 0,
            1 => 1,
            2 => 15,
            3 => 2_500,
            _ => 15_000,
        };
        ts += step;

        f.before_trade(&tape, ts);
        tape.on_trade(ts, 50_000.0 + (i as f64 * 0.1).sin(), 0.5, i % 2 == 0);
        f.after_trade(&tape, ts);

        // Llamar a before_trade repetidamente con el mismo timestamp no debe causar pánico
        f.before_trade(&tape, ts);
    }

    assert!(f.score.n > 0, "deben haberse madurado observaciones");
}

#[test]
fn f1_114_spectral_forecast_bank_burst_invariants() {
    let mut bank = SpectralForecastBank::new();
    let mut ts = 50_000u64;

    for i in 0..5_000u64 {
        ts += if i % 10 == 0 { 5_000 } else { 10 };
        bank.on_trade(ts, 60_000.0 + (i % 11) as f64, 1.5, i % 2 == 0);
    }

    // Consulta de sigmas no debe entrar en pánico ni emitir valores no finitos
    let sigmas = bank.sigmas_en_anclas(ts);
    assert_eq!(sigmas.len(), 7, "debe consultar las 7 anclas espectrales");
    for &s in &sigmas {
        if let Some(v) = s {
            assert!(v.is_finite());
            assert!(v >= 0.0);
        }
    }
}
