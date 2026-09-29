//! CL-16 — la adopción de posiciones remotas al arrancar usa una ranura
//! libre por pierna y sólo reserva margen si la apertura se aceptó.

use execution_engine::reconciliation::adoptar_en_ranura_libre;
use quantum_arena::position::PositionManager;
use std::sync::atomic::Ordering;

#[test]
fn cl16_las_dos_piernas_hedge_viven_en_ranuras_distintas() {
    let pm = PositionManager::default();
    let largo = adoptar_en_ranura_libre(&pm, true, 100.0, 0.5, 10.0, 1_000);
    let corto = adoptar_en_ranura_libre(&pm, false, 101.0, 0.3, 6.0, 1_000);
    let (Some(a), Some(b)) = (largo, corto) else {
        panic!("las dos piernas deben adoptarse: {largo:?} {corto:?}");
    };
    assert_ne!(a, b, "la segunda pierna no puede pisar a la primera");
    assert_eq!(pm.open_positions_count(), 2);
    assert!(pm.get_slot(a).is_long.load(Ordering::Relaxed));
    assert!(!pm.get_slot(b).is_long.load(Ordering::Relaxed));
    // Todo el margen que el llamador reserva queda en alguna ranura.
    assert!((pm.total_margin_used() - 16.0).abs() < 1e-12);
}

#[test]
fn cl16_la_primera_adopcion_sigue_en_la_ranura_de_siempre() {
    let pm = PositionManager::default();
    assert_eq!(adoptar_en_ranura_libre(&pm, true, 100.0, 0.5, 10.0, 1_000), Some(2));
    assert!(pm.position.is_open());
}

#[test]
fn cl16_sin_ranura_o_entrada_invalida_no_hay_adopcion() {
    let pm = PositionManager::default();
    assert_eq!(adoptar_en_ranura_libre(&pm, true, 0.0, 0.5, 10.0, 1_000), None);
    assert_eq!(adoptar_en_ranura_libre(&pm, true, 100.0, 0.0, 10.0, 1_000), None);
    assert_eq!(pm.open_positions_count(), 0);
    for _ in 0..3 {
        assert!(adoptar_en_ranura_libre(&pm, true, 100.0, 0.1, 2.0, 1_000).is_some());
    }
    assert_eq!(adoptar_en_ranura_libre(&pm, false, 100.0, 0.1, 2.0, 1_000), None);
    assert!((pm.total_margin_used() - 6.0).abs() < 1e-12);
}

/// El host reserva margen SÓLO tras una adopción aceptada y ya no escribe
/// la adopción en la ranura fija `positions.position`.
#[test]
fn cl16_el_host_adopta_por_ranura_y_reserva_solo_si_abre() {
    let host: String = include_str!("../../../src/bin/god_engine.rs")
        .split_whitespace()
        .collect();
    let fase5 = host
        .split("[FASE5]ADOPCIÓNDEESTADO")
        .nth(1)
        .expect("bloque FASE 5");
    let fase5 = &fase5[..fase5.find("letarena_telemetry=").expect("fin de FASE 5")];
    assert!(fase5.contains("adoptar_en_ranura_libre("));
    assert!(!fase5.contains("positions.position.open_with_horizon("));
    assert!(!fase5.contains("positions.position.exchange_confirmed"));
    assert!(!fase5.contains("positions.position.entry_tau_ms"));
    let reserva = fase5
        .find(".fetch_add(calculated_margin,")
        .expect("reserva de margen");
    let condicion = fase5
        .find("ifletSome(slot)=adopted_slot{")
        .expect("reserva condicionada a la adopción");
    assert!(condicion < reserva, "el margen se reserva dentro de la adopción aceptada");
}
