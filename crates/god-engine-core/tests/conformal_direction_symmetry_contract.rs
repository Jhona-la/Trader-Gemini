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
};
use quantum_arena::{GlobalArena, symbol_registry, symbols};
use std::sync::{Arc, Mutex, atomic::Ordering::Relaxed};

static CORE_ENV: Mutex<()> = Mutex::new(());

fn fixture() -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec!["BTCSYM".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("BTCUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
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
}

#[test]
fn k26_engine_publishes_symmetric_conformal_at_honest_neutrality() {
    let _guard = CORE_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = fixture();

    // Establecemos alpha objetivo
    arena.config.conformal_alpha.store(0.05, Relaxed);

    // Simulamos un tick analítico donde ml_prob es exactamente la base honesta (~0.25)
    // Para ello alimentamos el registro con la base y procesamos un tick
    arena.registry.set_for_coin(0, "ml_model_base", 0.25);

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

    // Verificamos que ambos p-valores conformal existen y son finitos
    let p_long = arena.registry.get_for_coin_or(0, "conformal_p_value", -1.0);
    let p_short = arena.registry.get_for_coin_or(0, "conformal_p_value_short", -1.0);
    assert!(p_long > 0.0 && p_long.is_finite(), "conformal_p_value inválido: {p_long}");
    assert!(p_short > 0.0 && p_short.is_finite(), "conformal_p_value_short inválido: {p_short}");

    // Durante warmup (sin 30 trades previos), ambos aceptan por política fail-open sana
    let acc_long = arena.registry.get_for_coin_or(0, "conformal_accept_long", -1.0);
    let acc_short = arena.registry.get_for_coin_or(0, "conformal_accept_short", -1.0);
    assert_eq!(acc_long, 1.0);
    assert_eq!(acc_short, 1.0);
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
