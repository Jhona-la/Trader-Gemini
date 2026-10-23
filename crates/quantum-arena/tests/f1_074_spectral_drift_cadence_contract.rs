//! Contrato de Certificación F1-074 (Ola Ω178): Calibración física de `dominant_drift`
//! en unidades de la malla espectral (base 4, ln 4 = salto de nodo) y normalización
//! por cadencia macro de referencia (`CADENCIA_REGIMEN_MS` = 60s).
//!
//! Erradica la extrapolación espuria "por hora" (3.6e6) que amplificaba el ruido x60..x720,
//! saturaba `accel_fast` a 0 ó 1 en micro-oscilaciones de tau* y hacía oscilar el ~33%
//! del margen libre (`free_cap`) tick a tick.

use quantum_arena::spectral_regime::{SpectralRegimeField, SpectralRegimeParams};
use quantum_arena::temporal_spectrum::TemporalSpectrum;

const LN_4: f64 = 1.386_294_361_119_890_6; // ln(4) = 2 * ln(2)

fn create_warm_spectrum() -> TemporalSpectrum {
    let mut spec = TemporalSpectrum::new();
    for t in 0..120u64 {
        let p = 100.0 + 0.3 * ((t % 7) as f64) + 0.1 * ((t % 13) as f64);
        spec.update(p, 1_000 + t * 1_000);
    }
    spec
}

/// Contrato 1: Fluctuaciones realistas de tau* (Δln = 0.05 ≈ 5%) en ventanas intermedias
/// (10s..60s) producen derivas proporcionales |drift| ≪ 1.0 sin saturar `accel_fast`.
#[test]
fn f1_074_contrato_1_deriva_gradual_en_ruido_realista() {
    let spec = create_warm_spectrum();
    let cur_ln = spec.spectral_field(true).resonant_tau_ms.max(1e-6).ln();

    // Evaluar a diferentes elapsed_ms (10s, 20s, 30s, 60s) con Δln = -0.05 (salto moderado a rápido)
    for &elapsed_s in &[10.0, 20.0, 30.0, 60.0] {
        let elapsed_ms = elapsed_s * 1_000.0;
        let prev_ln = cur_ln + 0.05; // cur - prev = -0.05
        let campo = SpectralRegimeField::from_spectrum(&spec, Some(prev_ln), elapsed_ms);

        // A 30s: drift = (-0.05 / ln 4) * (60 / 30) ≈ -0.0721
        // A 10s: drift = (-0.05 / ln 4) * (60 / 10) ≈ -0.216
        // En ningún caso debe saturar a -1.0
        assert!(
            campo.dominant_drift.abs() < 0.35,
            "A {}s, Δln=-0.05 no debe saturar: drift = {}",
            elapsed_s,
            campo.dominant_drift
        );
        assert!(
            campo.dominant_drift < 0.0,
            "Deriva debe ser negativa ante migración hacia escalas rápidas"
        );

        // El componente accel_fast = (-drift).clamp(0, 1) debe ser pequeño:
        let accel_fast = (-campo.dominant_drift).clamp(0.0, 1.0);
        assert!(
            accel_fast < 0.35,
            "accel_fast no debe explotar por ruido moderado: {}",
            accel_fast
        );

        // F1-074: La contribución de accel_fast a crash_flux (w_accel_fast * accel_fast)
        // antes saltaba a 0.35 (35%) en cualquier oscilación mínima de tau*.
        // Ahora, con la calibración física, para Δln = -0.05 (salto del 5%):
        // a 30s: accel_fast ≈ 0.072 y su contribución es ≈ 0.025 (≪ 0.05).
        // a 10s: accel_fast ≈ 0.216 y su contribución es ≈ 0.075 (≪ 0.35).
        let accel_crash_contrib = 0.35 * accel_fast;
        if elapsed_s >= 30.0 {
            assert!(
                accel_crash_contrib < 0.05,
                "A {}s, contribución de accel_fast debe ser < 0.05: fue {}",
                elapsed_s,
                accel_crash_contrib
            );
        }
        assert!(
            accel_crash_contrib < 0.10,
            "A {}s, la contribución de accel_fast a crash_flux debe ser < 0.10 (fue {}), erradicando el salto del 35%",
            elapsed_s,
            accel_crash_contrib
        );
    }
}

/// Contrato 2: Una migración estructural de exactamente 1 nodo de la malla espectral
/// (Δln = ln 4) en la cadencia de referencia (60s) produce exactamente |drift| = 1.0.
#[test]
fn f1_074_contrato_2_salto_unitario_de_nodo_espectral() {
    let spec = create_warm_spectrum();
    let cur_ln = spec.spectral_field(true).resonant_tau_ms.max(1e-6).ln();

    // 1 nodo hacia rápido en 60s: drift = -1.0
    let campo_rapido = SpectralRegimeField::from_spectrum(&spec, Some(cur_ln + LN_4), 60_000.0);
    assert!(
        (campo_rapido.dominant_drift - (-1.0)).abs() < 1e-4,
        "1 nodo hacia rápido en 60s debe dar exactamente drift = -1.0: {}",
        campo_rapido.dominant_drift
    );

    // 1 nodo hacia lento en 60s: drift = +1.0
    let campo_lento = SpectralRegimeField::from_spectrum(&spec, Some(cur_ln - LN_4), 60_000.0);
    assert!(
        (campo_lento.dominant_drift - 1.0).abs() < 1e-4,
        "1 nodo hacia lento en 60s debe dar exactamente drift = +1.0: {}",
        campo_lento.dominant_drift
    );

    // 2 nodos hacia rápido en 60s (Δln = 2 * ln 4): se acota limpiamente a -1.0 sin desbordamiento
    let campo_extremo = SpectralRegimeField::from_spectrum(&spec, Some(cur_ln + 2.0 * LN_4), 60_000.0);
    assert!(
        (campo_extremo.dominant_drift - (-1.0)).abs() < 1e-4,
        "Migración extrema debe quedar acotada a -1.0: {}",
        campo_extremo.dominant_drift
    );
}

/// Contrato 3: Regularización continua C^1 en micro-intervalos (Hermite cúbico).
/// A elapsed_ms < tau_min_drift_res_ms (5s), la ponderación w_t amortigua suavemente
/// el micro-jitter sin saltos discretos ni división por cero.
#[test]
fn f1_074_contrato_3_regularizacion_c1_en_origen() {
    let spec = create_warm_spectrum();
    let cur_ln = spec.spectral_field(true).resonant_tau_ms.max(1e-6).ln();

    let mut prev_drift_abs = 0.0;
    // Evaluamos tiempos crecientes desde 1 ms hasta 5000 ms con salto constante Δln = 0.1
    for &elapsed_ms in &[1.0, 50.0, 200.0, 1_000.0, 2_500.0, 5_000.0] {
        let campo = SpectralRegimeField::from_spectrum(&spec, Some(cur_ln + 0.1), elapsed_ms);
        let drift_abs = campo.dominant_drift.abs();

        // A 1 ms debe ser prácticamente cero (< 1e-6)
        if elapsed_ms <= 1.0 {
            assert!(
                drift_abs < 1e-5,
                "A 1ms de ancla, drift debe amortiguarse a cero: {}",
                drift_abs
            );
        }

        // Crecimiento monótono suave de la confianza con el tiempo transcurrido
        assert!(
            drift_abs >= prev_drift_abs - 1e-6,
            "La regularización Hermite debe ser suave y monótona con elapsed_ms"
        );
        prev_drift_abs = drift_abs;
    }
}

/// Contrato 4: En una serie con jitter estocástico de tau*, `dominant_drift` no satura
/// y la oscilación tick a tick de `crash_flux` permanece estrictamente acotada,
/// erradicando la oscilación del 35% de margen libre por micro-ruido.
#[test]
fn f1_074_contrato_4_estabilidad_crash_flux_monte_carlo() {
    let spec = create_warm_spectrum();
    let base_ln = spec.spectral_field(true).resonant_tau_ms.max(1e-6).ln();

    let mut saturadas = 0usize;
    let total_evals = 500usize;
    let mut max_flux_jump = 0.0f64;
    let mut prev_flux = 0.0f64;

    // Simular 500 ticks con fluctuaciones estocásticas de tau* (±2% en ln tau)
    for i in 0..total_evals {
        // Generador pseudo-aleatorio determinista
        let u1 = ((i as f64 * 137.0 + 17.0) % 100.0) / 100.0 - 0.5; // [-0.5, 0.5]
        let jitter = u1 * 0.04; // ±2% de variación en ln tau
        let prev_ln = base_ln + jitter;

        // Evaluación a 30s de ancla (donde antes per_hour amplificaba x120 y saturaba al 100%)
        let elapsed_ms = 30_000.0;

        let campo = SpectralRegimeField::from_spectrum(&spec, Some(prev_ln), elapsed_ms);

        if campo.dominant_drift.abs() >= 0.99 {
            saturadas += 1;
        }
        if i > 0 {
            let jump = (campo.crash_flux - prev_flux).abs();
            if jump > max_flux_jump {
                max_flux_jump = jump;
            }
        }
        prev_flux = campo.crash_flux;
    }

    assert_eq!(
        saturadas, 0,
        "En presencia de jitter del 2%, no debe haber derivas saturadas al 99%: se encontraron {}",
        saturadas
    );

    // F1-074: Con la extrapolación espuria por hora, max_flux_jump era 0.35 (el 35% de margen libre
    // oscilaba como moneda al aire tick a tick). Con la calibración física en unidades de malla,
    // max_flux_jump es < 0.05.
    assert!(
        max_flux_jump < 0.05,
        "El salto tick a tick de crash_flux por jitter de tau* debe ser < 0.05 (fue {}), erradicando la oscilación del 35%",
        max_flux_jump
    );
}

/// Contrato 5: Blindaje estricto de números finitos IEEE-754 y desacoplamiento de parámetros.
#[test]
fn f1_074_contrato_5_inmunidad_ieee754_y_parametros_desacoplados() {
    let spec = create_warm_spectrum();

    let custom_params = SpectralRegimeParams {
        w_accel_fast: 0.40,
        w_adverse_tide: 0.25,
        w_coherence_collapse: 0.20,
        w_micro_anti: 0.15,
        tau_min_drift_res_ms: 10_000.0,
        cadence_regime_ms: 30_000.0, // cadencia acelerada a 30s
        margin_contraction_scale: 0.90,
        min_margin_floor: 0.10,
        w_quantum_coupling: 1.0,
    };

    // NaN y valores degenerados nunca deben generar pánico ni NaN
    let campo_nan = SpectralRegimeField::from_spectrum_with_params(
        &spec,
        Some(f64::NAN),
        f64::NAN,
        &custom_params,
    );
    assert_eq!(campo_nan.dominant_drift, 0.0);
    assert!(campo_nan.crash_flux.is_finite());

    let campo_inf = SpectralRegimeField::from_spectrum_with_params(
        &spec,
        Some(f64::INFINITY),
        f64::INFINITY,
        &custom_params,
    );
    assert_eq!(campo_inf.dominant_drift, 0.0);
    assert!(campo_inf.crash_flux.is_finite());

    let campo_neg = SpectralRegimeField::from_spectrum_with_params(
        &spec,
        Some(10.0),
        -500.0,
        &custom_params,
    );
    assert_eq!(campo_neg.dominant_drift, 0.0);

    // Con cadencia desacoplada a 30s, un salto de 1 nodo en 30s produce |drift| = 1.0
    let cur_ln = spec.spectral_field(true).resonant_tau_ms.max(1e-6).ln();
    let campo_custom = SpectralRegimeField::from_spectrum_with_params(
        &spec,
        Some(cur_ln + LN_4),
        30_000.0,
        &custom_params,
    );
    assert!(
        (campo_custom.dominant_drift - (-1.0)).abs() < 1e-4,
        "Con cadencia de 30s, salto de 1 nodo en 30s da drift = -1.0: {}",
        campo_custom.dominant_drift
    );
}
