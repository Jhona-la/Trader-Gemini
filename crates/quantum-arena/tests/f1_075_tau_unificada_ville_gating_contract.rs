//! Contrato Formal de Verificación F1-075:
//! Gating de Ville para la Selección de τ Operativa Unificada y Estabilidad del Centroide de Hilbert.
//!
//! # QUÉ:
//! Erradicación de la oscilación discontinua entre el nodo discreto dominante (`dominant_tau_ms`)
//! y el centroide continuo de Hilbert (`continuous_resonant_tau_ms`) en `tau_operativa_unificada`.
//!
//! # POR QUÉ:
//! Anteriormente, `tau_operativa_unificada` aceptaba `dominant_tau_ms` con únicamente `habilidad_en(...) > 0`
//! (un IC de muestra ruidoso con $N \ge 30$, sin significancia estadística anytime-valid).
//! Cuando ninguna escala era significativa bajo Ville ($E_t < M/\alpha = 100$), `dominant_tau_ms` provenía
//! del argmax de energía o del peldaño LH-06(A). En una sesión fresca o en regímenes de bajo drift,
//! una oscilación de ruido en el signo del IC muestral (e.g. $+0.02 \to -0.01$) provocaba que el horizonte
//! temporal operativo $\tau_{\text{op}}$ saltara bruscamente $\times 8$ (e.g. de 68.7 s al centroide de 540 s),
//! desestabilizando el dimensionamiento de posición, el escalado de stop-loss / take-profit y el cálculo de Hurst.
//!
//! # PARA QUÉ:
//! Garantizar que el horizonte operativo del sistema pertenezca estrictamente al universo temporal continuo
//! de Hilbert $L^2(\mathbb{R})$, adoptando un nodo discreto únicamente cuando exista ventaja estadística demostrada
//! y validada fuera de muestra bajo la desigualdad de Ville para la familia de escalas operativas ($M=5$).
//!
//! # CÓMO:
//! 1. Exposición de `habilidad_significativa` en `ScaleState` y `habilidad_significativa_en` en `TemporalSpectrum`,
//!    evaluando `skill_e.significativo_familia(ESCALAS_OPERATIVAS_BANDA)` antes de admitir el IC.
//! 2. `tau_operativa_unificada` condiciona el uso de `dominant_tau_ms` a `habilidad_significativa_en(dominant_tau_ms) > 0`.
//! 3. Si no existe significancia estadística según Ville, el sistema opera con el centroide resonante continuo.
//!
//! # CUÁNDO:
//! Evaluado en cada tick de mercado en `process_event` y durante la apertura y ajuste dinámico de órdenes en el Core.
//!
//! # DÓNDE:
//! `crates/quantum-arena/src/temporal_spectrum.rs`.
//!
//! # QUIÉN:
//! Consejo de 10 Roles Senior / Quant Sr. — Ola Ω179 (Cierre F1-075).

use quantum_arena::temporal_spectrum::{
    TemporalSpectrum, SPECTRUM_SCALES_MS, TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS,
};

/// Contrato 1: Una escala coronada por energía con IC muestral positivo inmaduro/no-significativo
/// NO abandona el centroide continuo de Hilbert.
#[test]
fn f1_075_contrato_1_espectro_con_dominante_por_energia_e_ic_inmaduro_devuelve_centroide() {
    let mut spec = TemporalSpectrum::new();
    spec.set_test_time_window(0, 86_400_000, 10_000);
    spec.scales[18].signal = 0.9;
    spec.scales[18].persistence = 0.6;
    spec.scales[18].ewma_dev_vol = 0.01;

    // Escala 18 tiene N=35 muestras y un IC positivo (+0.18), pero NO es significativa bajo Ville (E_t = 1.0 < 100)
    spec.set_habilidad_para_test(18, 35, 0.18, false);

    spec.scales[19].signal = 0.3;
    spec.scales[19].persistence = 0.3;
    spec.scales[19].ewma_dev_vol = 0.01;

    spec.refresh_fusion();

    // 1. Telemetría reporta el IC crudo de la escala dominante
    assert_eq!(
        spec.habilidad_en(spec.dominant_tau_ms),
        Some(0.18),
        "La telemetría debe conservar el IC muestral crudo"
    );

    // 2. La habilidad significativa es None al no cruzar el test de Ville
    assert_eq!(
        spec.habilidad_significativa_en(spec.dominant_tau_ms),
        None,
        "Sin significancia Ville de familia, habilidad_significativa_en debe ser None"
    );

    // 3. tau_operativa_unificada debe devolver exactamente el centroide continuo
    let unificada = spec.tau_operativa_unificada();
    let centroide = spec.continuous_resonant_tau_ms();
    assert!(
        (unificada - centroide).abs() < 1e-9,
        "La tau unificada debe coincidir exactamente con el centroide continuo ({centroide}) vs ({unificada})"
    );

    // 4. Si el IC muestral oscila por ruido a negativo (-0.18), la tau unificada NO DEBE SALTAR
    spec.set_habilidad_para_test(18, 35, -0.18, false);
    spec.refresh_fusion();

    let unificada_neg = spec.tau_operativa_unificada();
    assert!(
        (unificada_neg - centroide).abs() < 1e-9,
        "Fluctuaciones de ruido en el signo del IC no significativo no deben alterar la tau operativa unificada"
    );
}

/// Contrato 2: Una escala con ventaja estadística validada por Ville fuera de muestra ($E_t \ge 100$)
/// adopta el nodo discreto dominante.
#[test]
fn f1_075_contrato_2_escala_con_habilidad_ville_significativa_adopta_nodo_discreto() {
    let mut spec = TemporalSpectrum::new();
    spec.set_test_time_window(0, 86_400_000, 10_000);
    spec.scales[18].signal = 0.4;
    spec.scales[18].persistence = 0.4;
    spec.scales[18].ewma_dev_vol = 0.01;

    // Escala 19 tiene N=75 muestras y un IC positivo (+0.35) con significancia Ville demostrada
    spec.scales[19].signal = 0.5;
    spec.scales[19].persistence = 0.5;
    spec.scales[19].ewma_dev_vol = 0.01;
    spec.set_habilidad_para_test(19, 75, 0.35, true);

    spec.refresh_fusion();

    assert_eq!(spec.dominant_tau_ms, SPECTRUM_SCALES_MS[19]);
    assert_eq!(
        spec.habilidad_significativa_en(spec.dominant_tau_ms),
        Some(0.35),
        "Escala con e-proceso validado debe reportar habilidad significativa"
    );

    let unificada = spec.tau_operativa_unificada();
    let esperado = SPECTRUM_SCALES_MS[19].clamp(TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS);
    assert!(
        (unificada - esperado).abs() < 1e-9,
        "Con habilidad demostrada por Ville, la tau unificada debe adoptar la escala dominante"
    );
}

/// Contrato 3: Una escala con e-proceso elevado pero con IC negativo $\le 0$ no es tomada como
/// horizonte directivo positivo; se recurre al centroide continuo.
#[test]
fn f1_075_contrato_3_escala_con_habilidad_ville_pero_ic_negativo_devuelve_centroide() {
    let mut spec = TemporalSpectrum::new();
    spec.set_test_time_window(0, 86_400_000, 10_000);
    spec.scales[18].signal = 0.4;
    spec.scales[18].persistence = 0.4;
    spec.scales[18].ewma_dev_vol = 0.01;

    // Escala 19 tiene capital Ville pero IC negativo (-0.35)
    spec.scales[19].signal = 0.5;
    spec.scales[19].persistence = 0.5;
    spec.scales[19].ewma_dev_vol = 0.01;
    spec.set_habilidad_para_test(19, 75, -0.35, true);

    spec.refresh_fusion();

    let unificada = spec.tau_operativa_unificada();
    let centroide = spec.continuous_resonant_tau_ms();
    assert!(
        (unificada - centroide).abs() < 1e-9,
        "Un IC negativo o nulo no debe gobernar la tau operativa unificada, debe volver al centroide"
    );
}

/// Contrato 4: En una serie de 100 ticks con fluctuaciones estocásticas de IC alrededor de cero,
/// la tau operativa unificada evoluciona de manera continua sin discontinuidades ni saltos $\ge 15\%$.
#[test]
fn f1_075_contrato_4_estabilidad_continua_sin_saltos_discretos_en_ruido() {
    let mut spec = TemporalSpectrum::new();
    spec.set_test_time_window(0, 86_400_000, 10_000);
    spec.scales[18].signal = 0.8;
    spec.scales[18].persistence = 0.6;
    spec.scales[18].ewma_dev_vol = 0.01;

    spec.scales[19].signal = 0.5;
    spec.scales[19].persistence = 0.4;
    spec.scales[19].ewma_dev_vol = 0.01;

    spec.refresh_fusion();

    let mut prev_tau = spec.tau_operativa_unificada();
    let mut rng_state = 0x1234_5678_u64;

    for _ in 0..100 {
        // Generador LCG determinista para ruido de IC entre -0.15 y +0.15 sin significancia
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (rng_state >> 32) as f64 / 4294967296.0;
        let noise_ic = -0.15 + 0.30 * u;

        spec.set_habilidad_para_test(18, 35, noise_ic, false);
        spec.refresh_fusion();

        let cur_tau = spec.tau_operativa_unificada();
        let salto_relativo = (cur_tau - prev_tau).abs() / prev_tau;

        assert!(
            salto_relativo < 0.10,
            "El horizonte operativo saltó de manera discontinua ({salto_relativo:.4}) debido a ruido no significativo"
        );
        prev_tau = cur_tau;
    }
}

/// Contrato 5: Inmunidad IEEE-754 y preservación estricta de cotas en todo el espacio operativo.
#[test]
fn f1_075_contrato_5_inmunidad_ieee754_y_cotas_operativas() {
    let spec = TemporalSpectrum::new();

    // Valores no finitos en habilidad_significativa_en
    assert_eq!(spec.habilidad_significativa_en(f64::NAN), None);
    assert_eq!(spec.habilidad_significativa_en(f64::INFINITY), None);
    assert_eq!(spec.habilidad_significativa_en(-1000.0), None);
    assert_eq!(spec.habilidad_significativa_en(0.0), None);

    // Estado en frío / no inicializado
    let tau_frio = spec.tau_operativa_unificada();
    assert!(tau_frio.is_finite(), "tau unificada debe ser finita en frío");
    assert!(
        tau_frio >= TAU_ANCHOR_FAST_MS && tau_frio <= TAU_ANCHOR_SLOW_MS,
        "tau unificada en frío ({tau_frio}) debe respetar las cotas [{TAU_ANCHOR_FAST_MS}, {TAU_ANCHOR_SLOW_MS}]"
    );
}
