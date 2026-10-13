//! LH-12: guardar y restaurar no rejuvenece evidencia aunque el reloj máximo
//! haya salido de la ventana de cierres por recepción tardía.
use god_engine_core::calibration::{SEMIVIDA_EVIDENCIA_MS, WINDOW};
use god_engine_core::estado_aprendido::{validar, EstadoAprendido};
use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, GlobalArena};

fn core() -> GodEngineCore {
    symbol_registry::update_registry(vec![symbol_registry::SymbolSpec {
        symbol: "BTCUSDT".into(),
        step_size: 0.001,
        tick_size: 0.01,
        min_qty: 0.001,
        min_notional: 5.0,
        max_leverage: 20,
        maker_fee: 0.0002,
        taker_fee: 0.0004,
        is_shadow: false,
    }]);
    GodEngineCore::new(GlobalArena::build_in_own_stack(1000.0))
}

#[test]
fn lh12_restore_keeps_clock_after_its_observation_is_evicted() {
    let mut old = core();
    let clock = 10 * SEMIVIDA_EVIDENCIA_MS as u64;
    old.alimentar_calibradores(0, 0.8, false, clock);
    for _ in 0..WINDOW {
        old.alimentar_calibradores(0, 0.8, false, 0);
    }
    let state = old.estado_aprendido();
    assert!(state.monedas[0].calibrador.iter().all(|o| o.2 == 0));
    assert_eq!(state.monedas[0].calibrador_reloj_ms, Some(clock));
    assert_eq!(state.calibrador_global_reloj_ms, Some(clock));
    let saved = serde_json::to_vec(&state).unwrap();
    let decoded: EstadoAprendido = serde_json::from_slice(&saved).unwrap();
    validar(&decoded).unwrap();
    let mut restored = core();
    assert_eq!(restored.restaurar_estado_aprendido(&decoded), 1);
    for (before, after) in [
        (&old.calibrator_by_coin[0], &restored.calibrator_by_coin[0]),
        (&old.confidence_calibrator, &restored.confidence_calibrator),
    ] {
        assert_eq!(after.evidence_clock_ms(), Some(clock));
        for t in [clock, clock + SEMIVIDA_EVIDENCIA_MS as u64] {
            assert!((before.calibrate_at(0.8, t) - after.calibrate_at(0.8, t)).abs() < 1e-8);
        }
    }
}

#[test]
fn lh12_legacy_snapshot_without_clock_uses_retained_dates() {
    let mut old = core();
    let h = SEMIVIDA_EVIDENCIA_MS as u64;
    for i in 0..20 {
        old.alimentar_calibradores(0, 0.8, i < 10, i * h);
    }
    let mut json = serde_json::to_value(old.estado_aprendido()).unwrap();
    json.as_object_mut()
        .unwrap()
        .remove("calibrador_global_reloj_ms");
    for coin in json["monedas"].as_array_mut().unwrap() {
        coin.as_object_mut().unwrap().remove("calibrador_reloj_ms");
    }
    let legacy: EstadoAprendido = serde_json::from_value(json).unwrap();
    validar(&legacy).unwrap();
    let mut restored = core();
    restored.restaurar_estado_aprendido(&legacy);
    assert_eq!(
        restored.confidence_calibrator.evidence_clock_ms(),
        Some(19 * h)
    );
    assert!(
        (old.confidence_calibrator.calibrate(0.8) - restored.confidence_calibrator.calibrate(0.8))
            .abs()
            < 1e-8
    );
}

#[test]
fn lh12_snapshot_rejects_clock_before_retained_observation() {
    let mut old = core();
    old.alimentar_calibradores(0, 0.8, false, 100);
    let mut state = old.estado_aprendido();
    state.monedas[0].calibrador_reloj_ms = Some(99);
    assert!(validar(&state).unwrap_err().contains("calibrador_reloj_ms"));
    state.monedas[0].calibrador_reloj_ms = Some(100);
    state.calibrador_global_reloj_ms = Some(99);
    assert!(validar(&state)
        .unwrap_err()
        .contains("calibrador_global_reloj_ms"));
}
