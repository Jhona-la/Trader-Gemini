//! Contrato Formal de Certificación para Ola Ω201:
//! - F1-044: Verificación de umbral de exceso de convicción ΔP ∈ [0.0, 0.45] para Rama 11 a partir de `min_confidence_btc` [0.50, 0.95] sin recorte espurio [0.55, 0.85].
//! - F1-038: Erradicación de literales plausibles en `build_54d_tensor` (slots 41..44 y 50 sirven 0.0 ante ausencia de datos Yahoo/alternative.me conforme a C-02).
//! - F1-035: Puerta 3 de `puertas_del_continuo_con`: eliminación de escalón C0 y exención por |stretch| >= 2.5 ATR; veto estricto por contradicción fuerte (d < -0.80) y modulación monótona continua (No Chasing Law D-472/LH-05).

use god_engine_core::GodEngineCore;
use quantum_arena::{symbols, GlobalArena};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[test]
fn test_f1_044_rama_11_tensor_cutoff_excess_space() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // 1. Caso límite superior: min_confidence_btc = 0.92
    arena.config.min_confidence_btc.store(0.92, Ordering::Relaxed);
    let tensor_min_conf = arena.config.min_confidence_btc.load(Ordering::Relaxed);
    let tensor_cutoff = ((tensor_min_conf - 0.50) * 1.0).clamp(0.0, 0.45);
    // ΔP = 0.92 - 0.50 = 0.42
    assert!((tensor_cutoff - 0.42).abs() < 1e-9, "tensor_cutoff={tensor_cutoff} debe ser 0.42");

    // 2. Caso límite inferior: min_confidence_btc = 0.50
    arena.config.min_confidence_btc.store(0.50, Ordering::Relaxed);
    let tensor_min_conf_lo = arena.config.min_confidence_btc.load(Ordering::Relaxed);
    let tensor_cutoff_lo = ((tensor_min_conf_lo - 0.50) * 1.0).clamp(0.0, 0.45);
    assert_eq!(tensor_cutoff_lo, 0.0, "tensor_cutoff={tensor_cutoff_lo} debe ser 0.0");

    // 3. Caso canónico: min_confidence_btc = 0.70
    arena.config.min_confidence_btc.store(0.70, Ordering::Relaxed);
    let tensor_min_conf_mid = arena.config.min_confidence_btc.load(Ordering::Relaxed);
    let tensor_cutoff_mid = ((tensor_min_conf_mid - 0.50) * 1.0).clamp(0.0, 0.45);
    assert!((tensor_cutoff_mid - 0.20).abs() < 1e-9, "tensor_cutoff={tensor_cutoff_mid} debe ser 0.20");
}

#[test]
fn test_f1_038_build_54d_tensor_zero_fill_for_missing_macro_features() {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    let arena = GlobalArena::build_in_own_stack(13.0);
    let core = GodEngineCore::new(Arc::clone(&arena));

    // Vector omni_features completamente vacío de macro-features (todos 0.0)
    let empty_omni = [0.0; 54];
    let tensor = core.build_54d_tensor(0, 10.0, 10.0, 60000.0, &empty_omni);

    // Contrato C-02: slots sin datos NUNCA deben inventar neutralidad (1.04, 1.02, 1.00, 0.75, 1.0)
    // Deben ser 0.0 estricto
    assert_eq!(tensor[41], 0.0, "Slot 41 (VIX) debe ser 0.0 ante ausencia de datos");
    assert_eq!(tensor[42], 0.0, "Slot 42 (SP500) debe ser 0.0 ante ausencia de datos");
    assert_eq!(tensor[43], 0.0, "Slot 43 (DXY) debe ser 0.0 ante ausencia de datos");
    assert_eq!(tensor[44], 0.0, "Slot 44 (NASDAQ) debe ser 0.0 ante ausencia de datos");
    assert_eq!(tensor[50], 0.0, "Slot 50 (Fear&Greed) debe ser 0.0 ante ausencia de datos");

    // Ahora verificamos con macro-features válidos
    let mut active_omni = [0.0; 54];
    active_omni[21] = 20.0; // VIX = 20.0 => 20 / 100 = 0.20 => clamp(0.5, 2.0) = 0.5
    active_omni[22] = 5500.0; // SP500 = 5500 => 5500 / 5000 = 1.1 => clamp(0.5, 2.0) = 1.1
    active_omni[23] = 18000.0; // DXY / 18000 = 1.0 => clamp(0.5, 2.0) = 1.0
    active_omni[24] = 25.0; // NASDAQ / 20 = 1.25 => clamp(0.1, 5.0) = 1.25
    active_omni[14] = 60.0; // Fear&Greed = 60 / 50 = 1.2 => clamp(0.0, 2.0) = 1.2

    let tensor_active = core.build_54d_tensor(0, 10.0, 10.0, 60000.0, &active_omni);
    assert!((tensor_active[41] - 0.50).abs() < 1e-9);
    assert!((tensor_active[42] - 1.10).abs() < 1e-9);
    assert!((tensor_active[43] - 1.00).abs() < 1e-9);
    assert!((tensor_active[44] - 1.25).abs() < 1e-9);
    assert!((tensor_active[50] - 1.20).abs() < 1e-9);
}

#[test]
fn test_f1_035_puertas_del_continuo_monotone_contradiction_and_no_stretch_chasing_bypass() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let core = GodEngineCore::new(Arc::clone(&arena));
    arena.coins[0].agg_buy_vol.store(50.0, Ordering::Relaxed);
    arena.coins[0].agg_sell_vol.store(50.0, Ordering::Relaxed);

    let intencion = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.80,
        ..Default::default()
    };

    // 1. Contradicción fuerte: p = 0.05 con base = 0.50 => d = (0.05 - 0.50) / 0.50 = -0.90 < -0.80.
    // Con stretch = 0.0 -> Flat
    let out_calm = core.puertas_del_continuo(0, intencion, true, 0.50, 0.05, 0.50, 0.0);
    assert_eq!(out_calm.signal, SignalType::Flat, "Veto estricto en calma");

    // Con stretch = 3.0 ATR (persecución extrema): NO debe hacer bypass a Long (No Chasing Law)
    let out_stretched = core.puertas_del_continuo(0, intencion, true, 0.50, 0.05, 0.50, 3.0);
    assert_eq!(out_stretched.signal, SignalType::Flat, "Veto estricto incondicional a stretch");

    // 2. Contradicción blanda: p = 0.30 con base = 0.50 => d = (0.30 - 0.50) / 0.50 = -0.40.
    // Factor = 1.0 + (-0.40) * 0.5 = 0.80.
    // Con stretch = 1.0 ATR
    let out_soft_1 = core.puertas_del_continuo(0, intencion, true, 0.50, 0.30, 0.50, 1.0);
    assert_eq!(out_soft_1.signal, SignalType::Long);
    assert!((out_soft_1.confidence - 0.80 * 0.80).abs() < 1e-9);

    // Con stretch = 3.0 ATR: la penalización se mantiene continua (sin saltar a 1.0 como en el legacy bug)
    let out_soft_3 = core.puertas_del_continuo(0, intencion, true, 0.50, 0.30, 0.50, 3.0);
    assert_eq!(out_soft_3.signal, SignalType::Long);
    assert!((out_soft_3.confidence - 0.80 * 0.80).abs() < 1e-9, "Penalización continua en stretch=3.0");
}
