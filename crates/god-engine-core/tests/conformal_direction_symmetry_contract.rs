//! K-26 / K-27 — CONTRATO DE INTEGRACIÓN: SIMETRÍA DIRECCIONAL CONFORMAL Y MICRO-TENDENCIA.
//!
//! Verifica matemáticamente y a nivel de motor:
//! 1. `p_win_directional` erradica el sesgo de base honesta: en neutralidad (p == base),
//!    ambos lados tienen la misma probabilidad proyectada p_win == base.
//! 2. Con base = 0.50 coincide idénticamente con el legado (1.0 - p).
//! 3. En el tick del motor con `ml_model_base = 0.25`, cuando el mercado es neutral,
//!    los registros conformal_p_value y conformal_p_value_short evalúan la misma probabilidad
//!    y no favorecen artificialmente a los cortos con 0.75.
//! 4. K-27: Simetría de microtendencia: micro_trend <= 0.0 en cortos y >= 0.0 en largos.

use god_engine_core::{
    GodEngineCore,
    calibration::p_win_directional,
    ml_inference::{NanoForest, NanoForestData},
};
use quantum_arena::{GlobalArena, symbol_registry, symbols};
use std::sync::{Arc, Mutex, atomic::Ordering::Relaxed};

static CORE_ENV: Mutex<()> = Mutex::new(());

fn bosque_constante(b: f64) -> NanoForest {
    NanoForest::from_data(NanoForestData {
        children_left: vec![],
        children_right: vec![],
        feature: vec![],
        threshold: vec![],
        value: vec![],
        tree_offsets: vec![0, 0],
        init_score: (b / (1.0 - b)).ln() as f32,
    })
    .expect("bosque de sólo sesgo")
}

fn fixture() -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec!["BTCSYM".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("BTCUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    (arena, core)
}

#[test]
fn k26_p_win_directional_mathematical_invariants() {
    // 1. Long es invariante de base y coincide con p
    for &p in &[0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0] {
        for &base in &[0.10, 0.25, 0.33, 0.50, 0.75] {
            assert_eq!(p_win_directional(true, p, base), p);
        }
    }

    // 2. Con base = 0.50, Short coincide exactamente con 1.0 - p
    for &p in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let legacy = 1.0 - p;
        let pw = p_win_directional(false, p, 0.50);
        assert!((pw - legacy).abs() < 1e-12, "p={p}: legacy={legacy}, pw={pw}");
    }

    // 3. Estado neutral: si p == base, ambos lados reciben base
    for &base in &[0.15, 0.20, 0.25, 0.30, 0.50, 0.70] {
        assert_eq!(p_win_directional(true, base, base), base);
        assert_eq!(p_win_directional(false, base, base), base);
    }

    // 4. Extremos del espectro de señal
    for &base in &[0.20, 0.25, 0.30, 0.50] {
        // Máximo impulso bajista (p = 0.0) -> Short tiene certeza 1.0, Long 0.0
        assert_eq!(p_win_directional(false, 0.0, base), 1.0);
        assert_eq!(p_win_directional(true, 0.0, base), 0.0);

        // Máximo impulso alcista (p = 1.0) -> Short tiene 0.0, Long certeza 1.0
        assert_eq!(p_win_directional(false, 1.0, base), 0.0);
        assert_eq!(p_win_directional(true, 1.0, base), 1.0);
    }

    // 5. REV-13: Robustez y neutralidad estricta ante NaN / Infinitos
    for &base in &[0.10, 0.20, 0.25, 0.33, 0.50, 0.70] {
        assert_eq!(p_win_directional(true, f64::NAN, base), base);
        assert_eq!(p_win_directional(false, f64::NAN, base), base);
        assert_eq!(p_win_directional(true, f64::INFINITY, base), base);
        assert_eq!(p_win_directional(false, f64::INFINITY, base), base);
    }
}

#[test]
fn k26_engine_publishes_symmetric_conformal_at_honest_neutrality() {
    let _guard = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    const BASE: f64 = 0.25;
    NanoForest::store_global("BTCSYM_MOTOR", bosque_constante(BASE));

    let (arena, mut core) = fixture();

    // Establecemos alpha objetivo
    arena.config.conformal_alpha.store(0.05, Relaxed);

    // Calentamos el calibrador con 40 cierres en rejilla (mínimo 30) para evaluar calibrado fuera de warmup
    for i in 0..40 {
        core.conformal_by_coin[0].update((i as f64 + 0.5) / 41.0, i % 2 == 0);
    }

    let (_, _, _) = core.process_tick_dual(
        0,
        100.0,
        100.01,
        5.0,
        5.0,
        10_000,
        &[0.0; 54],
        true,
        false, // Solo analítico
    );

    // Verificamos que el núcleo evaluó con la base honesta declarada del bosque
    let base = arena.registry.get_for_coin_or(0, "ml_model_base", -1.0);
    assert!(
        (base - BASE).abs() < 1e-6,
        "ml_model_base debe ser la base honesta declarada: {base}"
    );

    let ml_prob = arena.registry.get_for_coin_or(0, "ml_prob", -1.0);
    assert!(
        (ml_prob - BASE).abs() < 1e-6,
        "ml_prob debe coincidir con la base en ausencia de sesgo spot: {ml_prob}"
    );

    // Verificamos que ambos p-valores conformal existen, discriminan (p < 1.0) y son simétricos
    let p_long = arena.registry.get_for_coin_or(0, "conformal_p_value", -1.0);
    let p_short = arena.registry.get_for_coin_or(0, "conformal_p_value_short", -1.0);
    assert!(p_long > 0.0 && p_long < 1.0, "conformal_p_value fuera de rango: {p_long}");
    assert!(p_short > 0.0 && p_short < 1.0, "conformal_p_value_short fuera de rango: {p_short}");
    assert_eq!(
        p_long, p_short,
        "K-26: en neutralidad simétrica (p == base), p_long y p_short deben ser idénticos"
    );

    // Verificamos que la decisión de aceptación conformal coincide en ambos lados
    let acc_long = arena.registry.get_for_coin_or(0, "conformal_accept_long", -1.0);
    let acc_short = arena.registry.get_for_coin_or(0, "conformal_accept_short", -1.0);
    assert_eq!(
        acc_long, acc_short,
        "K-26: en neutralidad simétrica ambos lados comparten la aceptación"
    );
}

#[test]
fn k27_microtrend_reflection_symmetry_in_source() {
    let src = include_str!("../src/lib.rs");

    // Verificar que rama corta 1 exige micro_trend <= 0.0 (espejo exacto de larga >= 0.0)
    assert!(
        src.contains("micro_trend <= 0.0"),
        "K-27: la rama corta debe exigir micro_trend <= 0.0 estrictamente"
    );
    assert!(
        !src.contains("micro_trend <= 0.00003"),
        "K-27: no debe existir la relajación asimétrica de 3 bps (0.00003)"
    );
}
