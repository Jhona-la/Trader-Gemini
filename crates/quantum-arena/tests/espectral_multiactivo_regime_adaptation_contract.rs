//! Contrato formal de verificación: Adaptabilidad Dinámica y Continuidad C^1 en
//! Espectral Multiactivo y Campo de Régimen Espectral (Ola Ω127).
//!
//! Disciplina formal del Universo Multivariante Continuo Temporal Espectral:
//! 1. Desacoplamiento de constantes modulares duras hacia `EspectralMultiactivoParams`
//!    y `SpectralRegimeParams`.
//! 2. Mutabilidad y reconfiguración lock-free/adaptativa en runtime.
//! 3. Erradicación de discontinuidades de acantilado mediante factor de credibilidad
//!    continuo $C^1$ Hermite cúbico (smoothstep).
//! 4. Conservación estricta de invariantes de no-asfixia financiera y preservación de PnL.

use quantum_arena::espectral_multiactivo::{
    credibilidad_continua, EspectralMultiactivo, EspectralMultiactivoParams,
};
use quantum_arena::spectral_regime::{SpectralRegimeField, SpectralRegimeParams};
use quantum_arena::temporal_spectrum::{TemporalSpectrum, SPECTRUM_SCALES_MS};

#[test]
fn contrato_espectral_multiactivo_adaptacion_dinamica() {
    let initial_params = EspectralMultiactivoParams {
        recencia_x: 0.6,
        olvido: 0.05,
        muestras_maduras: 15,
        familia_veto_override: Some(50),
        lambda_adaptativa: false,
    };
    let mut ma = EspectralMultiactivo::with_params(4, initial_params);
    assert_eq!(ma.params().muestras_maduras, 15);
    assert_eq!(ma.params().recencia_x, 0.6);

    let tau = SPECTRUM_SCALES_MS[18]; // ~1.15 min
    for i in 0..14u64 {
        let ts = (i + 1) * 1000;
        let r = if i % 2 == 0 { 0.015 } else { -0.015 };
        ma.observar_maduracion(0, tau, 18, ts, r);
        ma.observar_maduracion(1, tau, 18, ts, r);
    }

    // A 14 muestras, con muestras_maduras = 15, el gate clásico retorna None
    assert_eq!(
        ma.coherencia_par(0, 1, 18),
        None,
        "con n=14 < 15 el gate estándar no debe abrir"
    );

    // Muestra 15: alcanza la madurez parametrizada
    let ts_15 = 15 * 1000;
    let r_15 = 0.015;
    ma.observar_maduracion(0, tau, 18, ts_15, r_15);
    ma.observar_maduracion(1, tau, 18, ts_15, r_15);

    let ic = ma.coherencia_par(0, 1, 18);
    assert!(
        ic.is_some(),
        "con n=15 >= 15 el par debe madurar y reportar IC"
    );
    assert!(
        ic.unwrap() > 0.90,
        "retornos idénticos deben reflejar correlación > 0.90: {}",
        ic.unwrap()
    );

    // Reconfiguración en caliente: aumentar la exigencia a 25 muestras
    let mut new_params = initial_params;
    new_params.muestras_maduras = 25;
    ma.update_params(new_params);
    assert_eq!(ma.params().muestras_maduras, 25);

    // Con el nuevo umbral de 25, 15 muestras ya no bastan
    assert_eq!(
        ma.coherencia_par(0, 1, 18),
        None,
        "tras reconfiguración adaptativa a 25 muestras, n=15 debe suspenderse honestamente"
    );

    // Alimentar hasta 25
    for i in 15..25u64 {
        let ts = (i + 1) * 1000;
        let r = if i % 2 == 0 { 0.015 } else { -0.015 };
        ma.observar_maduracion(0, tau, 18, ts, r);
        ma.observar_maduracion(1, tau, 18, ts, r);
    }
    assert!(
        ma.coherencia_par(0, 1, 18).is_some(),
        "al alcanzar n=25 el par vuelve a reportar IC maduro"
    );
}

#[test]
fn contrato_credibilidad_continua_hermite_c1_smoothstep() {
    let maduras = 40u64;

    // Condiciones de contorno
    assert_eq!(credibilidad_continua(0, maduras), 0.0, "en n=0 peso nulo");
    assert_eq!(
        credibilidad_continua(maduras, maduras),
        1.0,
        "en n=maduras peso unitario"
    );
    assert_eq!(
        credibilidad_continua(maduras + 20, maduras),
        1.0,
        "en n > maduras saturación suave a 1.0"
    );

    // Punto de inflexión en n = maduras / 2
    let mid = credibilidad_continua(20, maduras);
    assert!(
        (mid - 0.5).abs() < 1e-12,
        "en u=0.5, Hermite 3u^2 - 2u^3 = 0.5 exacto: {}",
        mid
    );

    // Monotonía estricta en el intervalo [0, maduras]
    let mut prev = -1.0f64;
    for n in 0..=maduras {
        let val = credibilidad_continua(n, maduras);
        assert!(val >= prev, "credibilidad debe ser monótona no decreciente");
        assert!(
            (0.0..=1.0).contains(&val),
            "credibilidad debe residir en [0, 1]"
        );
        prev = val;
    }

    // Suavidad C^1 en los extremos (derivada tiende a cero)
    let eps = credibilidad_continua(1, maduras);
    assert!(
        eps < 0.01,
        "derivada cero en origen asegura despegue cuadrático: {}",
        eps
    );
    let top_delta = 1.0 - credibilidad_continua(maduras - 1, maduras);
    assert!(
        top_delta < 0.01,
        "derivada cero en saturación asegura aterrizaje suave: {}",
        top_delta
    );
}

#[test]
fn contrato_coherencia_ponderada_par_preserva_continuo() {
    let mut ma = EspectralMultiactivo::new(4);
    let tau = SPECTRUM_SCALES_MS[19];

    // Con n=10 muestras (< 30)
    for i in 0..10u64 {
        let ts = (i + 1) * 1000;
        let r = 0.02;
        ma.observar_maduracion(0, tau, 19, ts, r);
        ma.observar_maduracion(1, tau, 19, ts, r);
    }

    // Gate discreto clásico: cerrado
    assert_eq!(ma.coherencia_par(0, 1, 19), None);

    // Operador continuo espectral: disponible con credibilidad parcial
    let ponderada = ma.coherencia_ponderada_par(0, 1, 19);
    assert!(ponderada.is_some(), "coherencia ponderada debe estar disponible");
    let (ic, cred) = ponderada.unwrap();
    assert!(ic > 0.95, "IC intrínseco capturado correctamente: {}", ic);
    assert!(
        cred > 0.0 && cred < 1.0,
        "credibilidad parcial no nula ni saturada: {}",
        cred
    );

    // A medida que n crece hacia 30, la credibilidad aumenta continuamente
    for i in 10..20u64 {
        let ts = (i + 1) * 1000;
        let r = 0.02;
        ma.observar_maduracion(0, tau, 19, ts, r);
        ma.observar_maduracion(1, tau, 19, ts, r);
    }
    let (_, cred_20) = ma.coherencia_ponderada_par(0, 1, 19).unwrap();
    assert!(
        cred_20 > cred,
        "credibilidad debe aumentar con mayor evidencia: {} > {}",
        cred_20,
        cred
    );
}

#[test]
fn contrato_spectral_regime_params_normalizacion_convexa_y_margen() {
    let mut spec = TemporalSpectrum::new();
    for t in 0..50u64 {
        spec.update(100.0 + (t as f64) * 0.1, 1000 + t * 1000);
    }

    let custom_params = SpectralRegimeParams {
        w_accel_fast: 0.50,
        w_adverse_tide: 0.30,
        w_coherence_collapse: 0.10,
        w_micro_anti: 0.10,
        tau_min_drift_res_ms: 3_000.0,
        cadence_regime_ms: 60_000.0,
        margin_contraction_scale: 0.80,
        min_margin_floor: 0.12,
        w_quantum_coupling: 1.0,
    };

    let field = SpectralRegimeField::from_spectrum_with_params(&spec, None, 10_000.0, &custom_params);
    assert!(field.crash_flux >= 0.0 && field.crash_flux <= 1.0);

    // Simular un campo adverso extremo
    let mut field_adverso = field;
    field_adverso.carrier_tide = -0.90;
    field_adverso.crash_flux = 1.0;

    let margin_mult = field_adverso.long_margin_multiplier_with_params(&custom_params);
    // Con contraction_scale = 0.80 y cf = 1.0: 1.0 - 0.80*1.0 = 0.20, clamped a min_margin_floor (0.12)
    assert!(
        (margin_mult - 0.20).abs() < 1e-6,
        "margen debe contraer según parámetros personalizados: {}",
        margin_mult
    );

    // Si la contracción llevara por debajo del piso
    let mut extreme_params = custom_params;
    extreme_params.margin_contraction_scale = 0.99;
    extreme_params.min_margin_floor = 0.15;
    let margin_extreme = field_adverso.long_margin_multiplier_with_params(&extreme_params);
    assert!(
        (margin_extreme - 0.15).abs() < 1e-6,
        "margen debe respetar el piso configurado para evitar asfixia: {}",
        margin_extreme
    );
}
