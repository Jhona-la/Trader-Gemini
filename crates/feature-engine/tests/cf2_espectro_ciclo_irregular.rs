//! C-F2.3b (Línea C): el FFT de 64 eventos de `spectral.rs` frente a un ciclo
//! físico conocido.
//!
//! Teoría que se prueba: un seno de periodo físico 18 s observado durante
//! 72 s completa 4 periodos en la ventana, así que el bin dominante debe ser
//! el 4 (4 ciclos por ventana). Con paso uniforme (control) el tiempo de
//! eventos coincide con el físico y sale el 4. Con la misma ventana física y
//! eventos irregulares (ráfagas de 16 eventos a 10 ms alternas con tramos de
//! 16 eventos a 2,24 s) el motor, que trabaja en tiempo de EVENTOS, ve una
//! fase que avanza 2,24/18 ≈ 0,124 ciclos por evento sólo en los tramos
//! dispersos; los dos tramos están separados 32 índices y en fase, así que
//! los bins pares se suman, los impares se cancelan y se predice el bin 8.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use feature_engine::spectral::SpectralCycleEngine;

const PERIODO_S: f64 = 18.0;

/// 64 instantes en 4 bloques de 16: bloques pares con `dt_denso`, impares
/// con `dt_disperso`.
fn tiempos(dt_denso: f64, dt_disperso: f64) -> Vec<f64> {
    let mut t = Vec::with_capacity(64);
    let mut ahora = 0.0;
    for bloque in 0..4 {
        let dt = if bloque % 2 == 0 { dt_denso } else { dt_disperso };
        for _ in 0..16 {
            t.push(ahora);
            ahora += dt;
        }
    }
    t
}

fn bin_dominante(t: &[f64]) -> usize {
    let mut e = SpectralCycleEngine::new();
    for &ti in t {
        e.push((std::f64::consts::TAU * ti / PERIODO_S).sin());
    }
    e.analyze_spectrum().0
}

/// Verdad: bin 4 exacto. Con paso de 1,125 s la señal es sin(2πi/16), que
/// completa 4 ciclos en 64 muestras; el taper izquierdo de 16 muestras no
/// desplaza un pico centrado en un bin.
#[test]
fn cf2_ciclo_muestreo_uniforme_bin_4() {
    let t = tiempos(1.125, 1.125);
    assert!((t[63] + 1.125 - 72.0).abs() < 1e-9);
    assert_eq!(bin_dominante(&t), 4);
}

/// Verdad: bin 4 exacto, el mismo ciclo físico de 18 s en la misma ventana
/// de 72 s. Es determinista: no hay tolerancia.
#[test]
#[ignore = "OPEN F2.3b-04 (Línea C): el FFT trabaja en tiempo de eventos y el bin del mismo ciclo físico depende de la densidad de eventos (sale el 8, no el 4)"]
fn cf2_ciclo_muestreo_irregular_bin_4() {
    // Ráfagas de 16 eventos a 10 ms (cerca de sin = 0, en t = 0 y t = 36 s)
    // y tramos de 16 eventos a 2,24 s.
    let t = tiempos(0.01, 2.24);
    assert!((t[63] + 2.24 - 72.0).abs() < 1e-9);
    let bin = bin_dominante(&t);
    assert_eq!(
        bin, 4,
        "bin {bin}: el bin depende de la densidad de eventos, no del periodo físico"
    );
}
