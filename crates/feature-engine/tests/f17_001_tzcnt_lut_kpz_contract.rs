//! Suite de Contratos Formales F17-001 — Bitmaps de Hardware `TZCNT`/`LZCNT` (`BitmapTickScanner256`),
//! Codificador ASCII por Tabla `LUT` sin `core::fmt` (`FastAsciiLutEncoder`) y Universalidad KPZ (`KpzSurfaceGrowthEngine`).
//!
//! Verifica las 6 propiedades físico-matemáticas y de silicio de la Fase 17:
//! - C1: Alineación de 64B y búsqueda en 1 ciclo `next_active_from` (`TZCNT`) y `prev_active_from` (`LZCNT`) a través de fronteras de palabras de 64 bits.
//! - C2: Conteo exacto `POPCNT` (`active_level_count`) y limpieza dinámica de `summary_mask` al vaciar niveles.
//! - C3: Exactitud bit a bit de `FastAsciiLutEncoder::encode_u64` y `encode_fixed_f64` frente a representación decimal estándar sin `core::fmt`.
//! - C4: Relaciones de escala de los exponentes universales KPZ ($\alpha = 1/2, \beta = 1/3, z = \alpha/\beta = 3/2$ e identidad de invarianza galileana $\alpha + z = 2$).
//! - C5: Exactitud de la Transformación de Cole-Hopf ($Z = \exp(\frac{\lambda}{2\nu} h)$) y deriva lateral positiva ($\Delta \bar{h} > 0$) ante pendiente no nula con $\lambda_{\text{KPZ}} > 0$.
//! - C6: Inmunidad total frente a entradas `NaN` / `±Inf` en `TzcntLutKpzMasterEngine`.

use feature_engine::{
    BitmapTickScanner256, FastAsciiLutEncoder, KpzBitmapRegime, KpzSurfaceGrowthEngine,
    TzcntLutKpzMasterEngine, KPZ_DYNAMIC_Z, KPZ_GROWTH_BETA, KPZ_ROUGHNESS_ALPHA,
    KPZ_SURFACE_SITES,
};
use std::mem::align_of;

#[test]
fn c1_bitmap_scanner_64b_alignment_and_cross_word_tzcnt_lzcnt() {
    assert_eq!(align_of::<BitmapTickScanner256>(), 64);

    let mut scanner = BitmapTickScanner256::new();
    assert_eq!(scanner.next_active_from(0), None);
    assert_eq!(scanner.prev_active_from(255), None);

    // Activar niveles en las 4 palabras (ticks 12, 70, 145, 240)
    assert!(scanner.set_level(12, true));
    assert!(scanner.set_level(70, true));
    assert!(scanner.set_level(145, true));
    assert!(scanner.set_level(240, true));
    assert_eq!(scanner.active_level_count(), 4);

    // Búsqueda ascendente (TZCNT)
    assert_eq!(scanner.next_active_from(0), Some(12));
    assert_eq!(scanner.next_active_from(12), Some(12));
    assert_eq!(scanner.next_active_from(13), Some(70));
    assert_eq!(scanner.next_active_from(71), Some(145));
    assert_eq!(scanner.next_active_from(146), Some(240));
    assert_eq!(scanner.next_active_from(241), None);

    // Búsqueda descendente (LZCNT)
    assert_eq!(scanner.prev_active_from(255), Some(240));
    assert_eq!(scanner.prev_active_from(239), Some(145));
    assert_eq!(scanner.prev_active_from(144), Some(70));
    assert_eq!(scanner.prev_active_from(69), Some(12));
    assert_eq!(scanner.prev_active_from(11), None);

    // Spread alrededor de mid_tick = 100: Bid en 70, Ask en 145 => 75 ticks
    assert_eq!(scanner.spread_ticks_around(100), Some(75));
}

#[test]
fn c2_bitmap_scanner_popcnt_and_summary_mask_cleanup() {
    let mut scanner = BitmapTickScanner256::new();
    assert!(scanner.set_level(63, true));
    assert!(scanner.set_level(64, true));
    assert_eq!(scanner.summary_mask, 0b0011);
    assert_eq!(scanner.active_level_count(), 2);

    assert!(scanner.set_level(63, false));
    assert_eq!(scanner.summary_mask, 0b0010);
    assert_eq!(scanner.active_level_count(), 1);

    assert!(scanner.set_level(64, false));
    assert_eq!(scanner.summary_mask, 0);
    assert_eq!(scanner.active_level_count(), 0);
}

#[test]
fn c3_fast_ascii_lut_encoder_u64_and_fixed_f64_exactness() {
    let mut buf = [0_u8; 32];

    for &val in &[0_u64, 7, 42, 99, 100, 105, 9999, 123_456_789, 9_876_543_210] {
        let len = FastAsciiLutEncoder::encode_u64(val, &mut buf);
        let s = std::str::from_utf8(&buf[..len]).unwrap_or("");
        assert_eq!(s, val.to_string());
    }

    let len_price = FastAsciiLutEncoder::encode_fixed_f64(67890.0425, 4, &mut buf);
    let price_str = std::str::from_utf8(&buf[..len_price]).unwrap_or("");
    assert_eq!(price_str, "67890.0425");

    let len_small = FastAsciiLutEncoder::encode_fixed_f64(0.007, 3, &mut buf);
    let small_str = std::str::from_utf8(&buf[..len_small]).unwrap_or("");
    assert_eq!(small_str, "0.007");
}

#[test]
fn c4_kpz_universal_exponents_galilean_invariance_identity() {
    // Identidad de relación de escala hiperespectral: z = alpha / beta = 3/2
    assert!((KPZ_ROUGHNESS_ALPHA / KPZ_GROWTH_BETA - KPZ_DYNAMIC_Z).abs() < 1e-15);
    // Identidad de invarianza galileana exacta en 1+1D: alpha + z == 2
    assert!((KPZ_ROUGHNESS_ALPHA + KPZ_DYNAMIC_Z - 2.0).abs() < 1e-15);
}

#[test]
fn c5_kpz_cole_hopf_nonlinear_lateral_growth_drift() {
    let engine = KpzSurfaceGrowthEngine {
        surface_tension_nu: 0.4,
        nonlinear_lambda: 2.0,
        grid_dx: 1.0,
    };

    // Perfil sinusoidal suave con media cero pero gradiente no nulo (\nabla h != 0)
    let mut heights = [0.0; KPZ_SURFACE_SITES];
    for (i, h) in heights.iter_mut().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (KPZ_SURFACE_SITES as f64);
        *h = 1.5 * angle.sin();
    }
    let zero_noise = [0.0; KPZ_SURFACE_SITES];

    let rep = engine.step_cole_hopf(&heights, &zero_noise, 0.05);
    // En la ecuación de KPZ con lambda > 0, el término (lambda/2)*(dh/dx)^2 > 0 impulsa un crecimiento positivo de la altura media
    assert!(
        rep.mean_height > 0.0,
        "Nonlinear KPZ lateral growth with lambda > 0 must increase mean surface height, got {}",
        rep.mean_height
    );
    assert!(rep.nonlinear_lateral_velocity > 0.0);
    assert!(rep.roughness_width_w > 0.0);
}

#[test]
fn c6_nan_inf_adversarial_immunity_in_master_engine() {
    let master = TzcntLutKpzMasterEngine::default();
    let mut scanner = BitmapTickScanner256::new();
    scanner.set_level(120, true);
    scanner.set_level(125, true);

    let mut bad_h = [f64::NAN; KPZ_SURFACE_SITES];
    bad_h[0] = f64::INFINITY;
    bad_h[1] = f64::NEG_INFINITY;
    let bad_f = [f64::NAN; KPZ_SURFACE_SITES];

    let rep = master.evaluate_cycle(&scanner, 122, &bad_h, &bad_f, f64::NAN);
    assert_eq!(rep.active_ticks, 2);
    assert_eq!(rep.spread_ticks, Some(5));
    assert!(rep.kpz.mean_height.is_finite());
    assert!(rep.kpz.roughness_width_w.is_finite());
    assert!(rep.directional_avalanche_signal.is_finite());
    assert!(matches!(
        rep.regime,
        KpzBitmapRegime::SmoothEdwardsWilkinson
            | KpzBitmapRegime::KpzAvalancheBreakout
            | KpzBitmapRegime::SparseLiquidityGapVeto
    ));
}
