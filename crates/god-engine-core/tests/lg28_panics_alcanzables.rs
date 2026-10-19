//! LG-28 (Línea G; auditoría de panics del camino vivo, ola 9): con
//! `panic = "abort"` en release, un panic alcanzable mata el proceso con
//! posiciones abiertas.
//!
//! GEC-LOW-01. El régimen global de BTC (`process_event`, moneda 0) hacía
//! `hurst_trend_threshold.clamp(0.50, 0.90)` y luego
//! `hurst_scalp_threshold.clamp(0.30, h_trend − 0.02)`. Un gen de tendencia
//! NaN (escrito por la evolución o por el registro) llegaba como NaN al
//! segundo `clamp` como cota superior, y `f64::clamp` exige `min <= max`:
//! panic en el primer tick de libro de BTC. Un umbral de caos NaN no
//! abortaba, pero anulaba en silencio el peso del caos.
//!
//! RED: con un gen de tendencia NaN el contrato entraba en pánico; con un
//! umbral de caos NaN el régimen salía distinto del de los genes base.
//! GREEN: NaN se lee como el valor del genoma base (φ − 1 y 0,7·(φ − 1)), y
//! el régimen coincide bit a bit con el de una corrida con esos genes.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering::Relaxed;

/// Corre 600 ticks de libro de BTC con los genes de Hurst dados y devuelve el
/// símplex del régimen.
fn regimen(tendencia: f64, caos: f64) -> [u64; 4] {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.config.hurst_trend_threshold.store(tendencia, Relaxed);
    arena.config.hurst_scalp_threshold.store(caos, Relaxed);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    let mut ts = 1_000u64;
    let mut lcg = 7_919u64;
    let mut precio = 50_000.0_f64;
    for _ in 0..600u64 {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((lcg >> 11) as f64) / (1u64 << 53) as f64 - 0.5;
        precio *= 1.0 + 4e-4 * u;
        ts += 1_000;
        let _ = core.process_event(
            0, false, false, true, precio, 0.0, precio - 0.5, precio + 0.5, 1.0, 1.0, 0.0, 0.0, ts,
            false, &[0.0; 54], false,
        );
    }
    [
        arena.regime_p_range.load(Relaxed).to_bits(),
        arena.regime_p_bull.load(Relaxed).to_bits(),
        arena.regime_p_crash.load(Relaxed).to_bits(),
        arena.regime_p_chaos.load(Relaxed).to_bits(),
    ]
}

#[test]
fn lg28_genes_de_hurst_nan_se_leen_como_el_genoma_base() {
    let phi_menos_uno = (5.0_f64.sqrt() - 1.0) / 2.0;
    let base = regimen(phi_menos_uno, phi_menos_uno * 0.7);
    assert!(base.iter().all(|b| f64::from_bits(*b).is_finite()));
    for (caso, tendencia, caos) in [
        ("tendencia NaN", f64::NAN, phi_menos_uno * 0.7),
        ("caos NaN", phi_menos_uno, f64::NAN),
        ("ambos NaN", f64::NAN, f64::NAN),
    ] {
        assert_eq!(regimen(tendencia, caos), base, "{caso}: mismo régimen que con los genes base");
    }
}

/// ±∞ ya saturaba en el `clamp` sin abortar; LG-28 no lo cambia.
#[test]
fn lg28_genes_de_hurst_infinitos_siguen_saturando() {
    assert_eq!(regimen(f64::INFINITY, f64::NEG_INFINITY), regimen(0.90, 0.30));
}
