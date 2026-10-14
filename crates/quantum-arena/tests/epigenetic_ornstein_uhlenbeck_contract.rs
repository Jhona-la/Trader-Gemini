//! Contrato formal: Proceso Estocástico Continuo Ornstein-Uhlenbeck para Memoria Epigenética (Ola Ω109).
//!
//! F3.4-07: Verifica que el sesgo epigenético (`epigenetic_bias`) experimente relajación
//! homeostática temporal continua hacia el equilibrio genómico $\mu = 1.0$ gobernada por un
//! proceso Ornstein-Uhlenbeck, erradicando la petrificación no-ergódica de rachas pasadas y
//! preservando la paridad canónica instantánea ante operaciones consecutivas de alta frecuencia.

use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering::Relaxed;

#[test]
fn test_epigenetic_bias_ou_zero_elapsed_parity() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];

    // 1. En arranque frío sin cierres previos (last_close = 0):
    coin.epigenetic_bias.store(1.50, Relaxed);
    coin.last_close_ts.store(0, Relaxed);
    assert_eq!(coin.get_active_epigenetic_bias(1_000_000), 1.50);

    // 2. Con evento concurrente o inmediatamente consecutivo (now_ms == last_close):
    coin.last_close_ts.store(5_000_000, Relaxed);
    assert_eq!(coin.get_active_epigenetic_bias(5_000_000), 1.50);

    // 3. Con now_ms < last_close (evento causal anterior):
    assert_eq!(coin.get_active_epigenetic_bias(4_999_999), 1.50);
}

#[test]
fn test_epigenetic_bias_ou_temporal_mean_reversion() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];

    let t0 = 10_000_000;
    coin.last_close_ts.store(t0, Relaxed);
    coin.epigenetic_bias.store(1.80, Relaxed);

    // Constante de tiempo tau = 1,800,000 ms (30 minutos)
    let _tau_ms = 1_800_000.0;

    // A: Tras 30 minutos (1 constante de tiempo): bias = 1.0 + 0.8 * e^(-1)
    let t1 = t0 + 1_800_000;
    let expected_t1 = 1.0 + 0.80 * (-1.0_f64).exp();
    let actual_t1 = coin.get_active_epigenetic_bias(t1);
    assert!(
        (actual_t1 - expected_t1).abs() < 1e-12,
        "t1: esperado {expected_t1}, obtenido {actual_t1}"
    );
    assert!(actual_t1 < 1.80 && actual_t1 > 1.0);

    // B: Tras 150 minutos (5 constantes de tiempo): bias = 1.0 + 0.8 * e^(-5) ~= 1.00539
    let t5 = t0 + 5 * 1_800_000;
    let expected_t5 = 1.0 + 0.80 * (-5.0_f64).exp();
    let actual_t5 = coin.get_active_epigenetic_bias(t5);
    assert!(
        (actual_t5 - expected_t5).abs() < 1e-12,
        "t5: esperado {expected_t5}, obtenido {actual_t5}"
    );
    assert!((actual_t5 - 1.0) < 0.01, "Tras 5 tau debe haber retornado casi al equilibrio");

    // C: Tras 24 horas (48 constantes de tiempo): bias converge exactamente a 1.0
    let t_inf = t0 + 24 * 3_600_000;
    let actual_inf = coin.get_active_epigenetic_bias(t_inf);
    assert!((actual_inf - 1.0).abs() < 1e-6);
}

#[test]
fn test_epigenetic_bias_ou_loss_penalty_recovery() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];

    let t0 = 20_000_000;
    coin.last_close_ts.store(t0, Relaxed);
    // Activo penalizado severamente tras pérdidas consecutivas a 0.50
    coin.epigenetic_bias.store(0.50, Relaxed);

    // Tras 30 minutos: 1.0 + (0.50 - 1.0) * e^(-1) = 1.0 - 0.5 * 0.367879 = 0.81606
    let t1 = t0 + 1_800_000;
    let expected = 1.0 - 0.50 * (-1.0_f64).exp();
    let actual = coin.get_active_epigenetic_bias(t1);
    assert!(
        (actual - expected).abs() < 1e-12,
        "Recuperación tras pérdida: esperado {expected}, obtenido {actual}"
    );
    assert!(actual > 0.50, "La penalización debe disiparse continuamente con el tiempo");
}

#[test]
fn test_epigenetic_feedback_advances_bias_before_update() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];

    // Inicializamos con sesgo inflado 1.80 a t=1,000
    coin.epigenetic_bias.store(1.80, Relaxed);
    coin.epigenetic_threshold_modifier.store(1.0, Relaxed);
    coin.last_close_ts.store(1_000, Relaxed);

    // Ocurre un nuevo cierre 10 constantes de tiempo después (t = 18_001_000)
    let close_ts = 1_000 + 10 * 1_800_000;
    let pnl_pct = 0.002; // Trade ganador
    coin.apply_spectral_epigenetic_feedback_with_time(pnl_pct, 10_000, 30_000.0, close_ts);

    // El sesgo previo 1.80 ya decayó a 1.0 + 0.8 * e^(-10) ~= 1.000036
    // El pnl_shift es 0.002 * (4.0 / 0.1) = 0.08
    // Por tanto, el nuevo sesgo almacenado NO debe ser 1.80 + 0.08 = 1.88,
    // sino aproximadamente 1.000036 + 0.08 ~= 1.080036!
    let new_stored = coin.epigenetic_bias.load(Relaxed);
    assert!(
        new_stored < 1.15,
        "El nuevo sesgo debe calcularse sobre el estado relajado ({new_stored} < 1.15)"
    );
    assert!(
        new_stored > 1.05,
        "El nuevo sesgo debe incorporar el beneficio del trade reciente ({new_stored} > 1.05)"
    );
}
