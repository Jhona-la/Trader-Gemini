//! Contrato formal F1-002, F1-003 y F1-004 (FASE 2 CULMINACIÓN, Ola Ω206):
//! - F1-002: Puente Browniano discreto y agresor Lee-Ready fluctuante para micro-ticks en examen walk-forward.
//! - F1-003: Omni 54D causal con instantánea macro congelada sin síntesis espuria y poda de escrituras huérfanas.
//! - F1-004: Shadow Forest libre de fuga temporal mediante snapshot de rasgos 6D en la apertura e ingesta única.

use evolution_engine::online_daemon::pseudo_gaussian_noise;
use quantum_arena::state::ClosedTradeRecord;
use quantum_arena::GlobalArena;

#[test]
fn test_f1_002_pseudo_gaussian_noise_properties() {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let n = 1000;
    let mut previous = 0.0;
    let mut signs_flip = 0;

    for i in 0..n {
        let z = pseudo_gaussian_noise(i as u64, (i % 8) as u64, 0);
        assert!(z.is_finite(), "El ruido browniano debe ser finito");
        assert!(z >= -3.5 && z <= 3.5, "El ruido debe estar acotado: {z}");
        sum += z;
        sum_sq += z * z;

        if i > 0 && ((z > 0.0 && previous < 0.0) || (z < 0.0 && previous > 0.0)) {
            signs_flip += 1;
        }
        previous = z;

        // Bit-exact determinismo y reproducibilidad
        let z_repeat = pseudo_gaussian_noise(i as u64, (i % 8) as u64, 0);
        assert_eq!(z, z_repeat, "El generador debe ser 100% determinista y reproducible");
    }

    let mean = sum / n as f64;
    let variance = (sum_sq / n as f64) - (mean * mean);
    assert!(mean.abs() < 0.20, "Media del ruido debe aproximarse a 0: {mean}");
    assert!(variance > 0.5 && variance < 1.5, "Varianza debe aproximarse a 1: {variance}");
    assert!(signs_flip > 100, "El signo debe fluctuar honestamente: {signs_flip} cambios");
}

#[test]
fn test_f1_004_position_and_coin_entry_shadow_features_lifecycle() {
    let arena = GlobalArena::new(13.0);
    let coin = &arena.coins[0];
    let pos = &coin.positions.slot_0;

    // Verificar estado inicial vacío
    assert!(pos.get_entry_shadow_features().is_none());
    assert!(coin.take_last_closed_trade().is_none());

    // Asignar rasgos 6D fotografiados en la APERTURA [obi, accel, spread_bps, atr_pct, hurst, velocity]
    let entry_features = [0.25, -0.15, 3.2, 0.0035, 0.62, 0.85];
    pos.set_entry_shadow_features(entry_features);

    // Lectura no destructiva pre-cierre
    let snapshot = pos.get_entry_shadow_features();
    assert_eq!(snapshot, Some(entry_features));

    // Consumir rasgos al cierre de la posición
    let taken = pos.take_entry_shadow_features();
    assert_eq!(taken, Some(entry_features));
    assert!(pos.get_entry_shadow_features().is_none(), "Debe quedar vacío tras take");

    // Registrar trade cerrado en CoinArena
    let record = ClosedTradeRecord {
        is_long: true,
        net_pnl_pct: 0.025,
        entry_shadow_features: entry_features,
        close_ts_ms: 1_700_000_000,
        generation: 42,
    };
    coin.record_closed_trade(record);

    // Ingesta única por online_daemon: primer consumo debe devolver el registro exacto
    let consumed = coin.take_last_closed_trade();
    assert_eq!(consumed, Some(record));

    // Segundo consumo debe devolver None (erradicando la doble contabilización)
    let second_attempt = coin.take_last_closed_trade();
    assert!(second_attempt.is_none(), "No debe existir doble ingesta del mismo trade cerrado");
}
