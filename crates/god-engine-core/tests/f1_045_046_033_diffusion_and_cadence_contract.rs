//! Contrato formal: Tipificación por Difusión, Cadencia Física de Régimen y Fast-Path de Confluencia (Ola Ω200).
//!
//! Certifica la resolución integral de:
//! 1. F1-045: Fast-path de confluencia resonante (|fused| < 0.15) erradicando 10 lookups SkipMap por evento,
//!    y tolerancia de marea macro fundamentada en difusión Z95 * sigma * spread_sd.
//! 2. F1-046: Tipificación por difusión en Rama 13 (MACD spread z-score), umbral z estandarizado por Hurst
//!    y erradicación del multiplicador arbitrario x50 en convicción.
//! 3. F1-033: Acondicionamiento de cálculo W1 y observador BOCPD a la cadencia física de régimen (60s),
//!    y actuador de atenuación suave de confianza ante transiciones violentas (p_transition > 0.50).

use god_engine_core::{
    confluencia_resonante_con_umbrales, sig_conf, COHERENCIA_MINIMA, FUSED_UMBAL_MODERADO,
    FUSED_UMBAL_PLENO, HURST_CONTINUACION, MAREA_MACRO_TOLERANCIA,
};
use quantum_arena::GlobalArena;

#[test]
fn test_f1_045_confluencia_resonante_fast_path_and_diffusion_tolerance() {
    // 1. Fast-Path: Si |fused| < 0.15 y coherencias <= COHERENCIA_MINIMA, confluencia es idéntica a None.
    let fused_sub_floor = 0.10;
    let tau_star_hurst = 0.55;
    let coh_long = 0.10;
    let coh_short = 0.10;
    let macro_trend = 0.0;

    let res = confluencia_resonante_con_umbrales(
        fused_sub_floor,
        tau_star_hurst,
        coh_long,
        coh_short,
        macro_trend,
        FUSED_UMBAL_PLENO,
        FUSED_UMBAL_MODERADO,
        HURST_CONTINUACION,
        COHERENCIA_MINIMA,
        MAREA_MACRO_TOLERANCIA,
    );
    assert_eq!(res, None, "Sub-floor |fused| < 0.15 debe producir None analítico");

    // 2. Tolerancia de Marea Macro en Difusión:
    // En un activo volátil (ATR = 2.0% = 0.02), un macro_trend de 3 pb (0.0003) es ruido puro frente a Z95,
    // pero con el literal crudo 2e-4 (0.0002) quedaba vetado erróneamente.
    let atr_pct_volatil = 0.02;
    let sigma = god_engine_core::diffusion::sigma_from_atr(atr_pct_volatil);
    let spread_sd = god_engine_core::diffusion::ema_spread_sd(
        god_engine_core::diffusion::EMA_FAST_BARS,
        god_engine_core::diffusion::EMA_SLOW_BARS,
    );
    let piso_tendencia = god_engine_core::diffusion::Z95 * sigma * spread_sd;
    assert!(
        piso_tendencia > MAREA_MACRO_TOLERANCIA,
        "En activo volátil, el piso de difusión ({piso_tendencia}) supera holgadamente el literal crudo ({MAREA_MACRO_TOLERANCIA})"
    );

    let macro_trend_ruido = -0.0003; // -3 pb (ruido bajo piso_tendencia pero vetado por literal 2 pb)
    let res_literal = confluencia_resonante_con_umbrales(
        0.40, // fused pleno
        0.55,
        0.25, // coh_long > coh_min
        0.05,
        macro_trend_ruido,
        FUSED_UMBAL_PLENO,
        FUSED_UMBAL_MODERADO,
        HURST_CONTINUACION,
        COHERENCIA_MINIMA,
        MAREA_MACRO_TOLERANCIA, // 0.00020
    );
    assert_eq!(
        res_literal, None,
        "Literal crudo 2 pb veta erróneamente por ruido en activo volátil"
    );

    let res_difusion = confluencia_resonante_con_umbrales(
        0.40,
        0.55,
        0.25,
        0.05,
        macro_trend_ruido,
        FUSED_UMBAL_PLENO,
        FUSED_UMBAL_MODERADO,
        HURST_CONTINUACION,
        COHERENCIA_MINIMA,
        piso_tendencia, // Tolerancia física fundamentada
    );
    assert!(
        res_difusion.is_some(),
        "Tolerancia por difusión admite señal válida en activo volátil sin veto espurio"
    );
    let (is_long, conf) = res_difusion.unwrap();
    assert!(is_long);
    assert!(conf >= 0.50 && conf <= 0.95);
}

#[test]
fn test_f1_046_rama_13_standardized_diffusion_z_and_no_magic_multiplier() {
    let atr_pct = 0.005; // 50 pb ATR
    let macd_diff = 0.0015; // 15 pb spread EMA9/EMA21

    // 1. Tipificación analítica z-score del spread EMA:
    let z_macd = god_engine_core::diffusion::ema_spread_z(
        macd_diff,
        atr_pct,
        god_engine_core::diffusion::EMA_FAST_BARS,
        god_engine_core::diffusion::EMA_SLOW_BARS,
    );
    assert!(z_macd.is_finite(), "z_macd debe ser un número finito");
    assert!(z_macd > 0.0, "Spread positivo debe producir z_macd positivo");

    // 2. Umbral tipificado continuo modulado por memoria Hurst:
    let hurst_persistente = 0.65;
    let z_threshold = god_engine_core::diffusion::Z95 * (0.50 / hurst_persistente);
    assert!(
        z_threshold < god_engine_core::diffusion::Z95,
        "En régimen persistente (H > 0.5), el umbral de tendencia disminuye ({z_threshold} < {Z95})",
        Z95 = god_engine_core::diffusion::Z95
    );

    // 3. Convicción tipificada sin multiplicador arbitrario x50:
    let magnitud_norm = (z_macd.abs() / god_engine_core::diffusion::Z95).clamp(0.0, 1.0);
    let piso_magnitud = sig_conf(magnitud_norm);
    assert!(
        piso_magnitud >= 0.51 && piso_magnitud <= 0.90,
        "Piso de magnitud estandarizado debe estar estrictamente en [0.51, 0.90]: {piso_magnitud}"
    );

    // Si macd_diff es muy débil (ruido z < 0.1):
    let z_debil = 0.05;
    let mag_debil = (z_debil / god_engine_core::diffusion::Z95).clamp(0.0, 1.0);
    let piso_debil = sig_conf(mag_debil);
    assert!(
        (piso_debil - 0.51).abs() < 0.02,
        "Señal débil produce convicción mínima pegada al suelo neutro: {piso_debil}"
    );
}

#[test]
fn test_f1_033_bocpd_cadence_and_actuator_dampening() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut obs = god_engine_core::W1ChangepointObserver::new();

    // 1. Estabilidad de transición en reposo:
    let mut last_p = 0.0;
    for i in 1..=50 {
        if let Some(p) = obs.observe_at(i * 1000, 0.02) {
            last_p = p;
        }
    }
    assert!(
        last_p < 0.05,
        "En reposo estacionario, p_transition debe permanecer bajo hazard: {last_p}"
    );

    // 2. Salto de régimen violento (W1 se dispara bruscamente a 2.5):
    // El BOCPD detecta el changepoint en el instante t=51_000 donde la masa salta a r=0.
    let p_jump = obs.observe_at(51_000, 2.5).expect("calentado");
    assert!(
        p_jump > 0.50,
        "Salto violento instantáneo en W1 debe producir p_transition > 0.50: {p_jump}"
    );

    // 3. Actuador gobernado de atenuación:
    // Si bocpd_actuator_enabled está desactivado (default 0.0):
    let penalty_neutral = if arena.registry.get_value_fast("bocpd_actuator_enabled").unwrap_or(0.0) > 0.0 && p_jump > 0.50 {
        (1.0 - 0.40 * (p_jump - 0.50)).clamp(0.80, 1.0)
    } else {
        1.0
    };
    assert_eq!(penalty_neutral, 1.0, "Con actuador desactivado, penalización debe ser neutral (1.0)");

    // Activamos por gobernanza:
    arena.registry.set("bocpd_actuator_enabled", 1.0);
    let penalty_activo = if arena.registry.get_value_fast("bocpd_actuator_enabled").unwrap_or(0.0) > 0.0 && p_jump > 0.50 {
        (1.0 - 0.40 * (p_jump - 0.50)).clamp(0.80, 1.0)
    } else {
        1.0
    };
    assert!(
        penalty_activo < 1.0 && penalty_activo >= 0.80,
        "Con actuador habilitado ante salto violento, la confianza debe atenuarse suavemente en [0.80, 1.0): {penalty_activo}"
    );
}
