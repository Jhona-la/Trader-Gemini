//! CONTRATO FORMAL DE RESOLUCIÓN ESPECTRAL Y ADMISIÓN CONTINUA DE SLOTS (Ω32)
//!
//! # Objetivo del Contrato
//!
//! 1. Verificar que la admisión por defecto mantiene `DEFAULT_RESONANT_DELTA_LN = 0.80`
//!    rechazando colisiones destructivas en la misma escala y dirección.
//! 2. Verificar que con resolución unificada `UNIFIED_RESONANT_DELTA_LN = 0.60` (D-431),
//!    escalas en el intervalo [0.60, 0.80) —como tau = 30s y tau = 60s (|Δlnτ| ≈ 0.693)—
//!    son admitidas armónicamente en ranuras separadas.
//! 3. Comprobar la coexistencia ortogonal multiescala (ej. 15s rápido y 1h lento) en la misma dirección.
//! 4. Verificar que direcciones opuestas (Long vs Short) no colisionan por escala temporal.
//! 5. Validar que entradas no finitas o anómalas fallan de forma segura (fail-closed).

use quantum_arena::position::{PositionHorizon, PositionManager};

fn open_slot(manager: &PositionManager, slot: usize, is_long: bool, tau_ms: u64) {
    assert!(manager.get_slot(slot).open_with_tau_and_fee(
        is_long,
        100.0,
        1.0,
        10.0,
        1_000,
        105.0,
        95.0,
        PositionHorizon::Continuous,
        0.80,
        0.85,
        0.05,
        tau_ms,
    ));
}

#[test]
fn spectral_slot_resolucion_default_080_rechaza_colision_cercana() {
    let mgr = PositionManager::default();
    // Abrir ranura 0 con tau = 30,000 ms Long
    open_slot(&mgr, 0, true, 30_000);

    // Candidato con escala muy cercana: tau = 40,000 ms (|Δlnτ| = ln(4/3) ≈ 0.2877 < 0.80)
    let slot = mgr.find_resonant_slot(40_000.0, true);
    assert_eq!(slot, None, "Debe rechazar por colisión de banda bajo 0.80");
    assert_eq!(
        mgr.razon_sin_slot(40_000.0, true),
        PositionManager::RAZON_COLISION_BANDA
    );
}

#[test]
fn spectral_slot_resolucion_unificada_060_admite_el_hueco_intermedio() {
    let mgr = PositionManager::default();
    open_slot(&mgr, 0, true, 30_000);

    // Candidato con tau = 60,000 ms: |Δlnτ| = ln(2.0) ≈ 0.6931
    // Bajo el umbral 0.80 es rechazado:
    assert_eq!(mgr.find_resonant_slot(60_000.0, true), None);

    // Bajo el umbral unificado 0.60 (D-431), 0.6931 >= 0.60: ¡ES ADMITIDO EN RANURA 1!
    let slot_unificado = mgr.find_resonant_slot_with_threshold(
        60_000.0,
        true,
        PositionManager::UNIFIED_RESONANT_DELTA_LN,
    );
    assert_eq!(
        slot_unificado,
        Some(1),
        "El hueco [0.60, 0.80) debe ser admitido armónicamente bajo resolución unificada"
    );
}

#[test]
fn spectral_slot_coexistencia_ortogonal_multiescala() {
    let mgr = PositionManager::default();
    // 1. Abrir onda rápida: tau = 15s (15,000 ms) en Long
    let s0 = mgr.find_resonant_slot(15_000.0, true);
    assert_eq!(s0, Some(0));
    open_slot(&mgr, 0, true, 15_000);

    // 2. Onda lenta/macro: tau = 1 hora (3,600,000 ms) en Long
    // |ln(3,600,000) - ln(15,000)| = ln(240) ≈ 5.48 >> 0.80
    let s1 = mgr.find_resonant_slot(3_600_000.0, true);
    assert_eq!(s1, Some(1), "Escalas ortogonales deben coexistir en la misma dirección");
    open_slot(&mgr, 1, true, 3_600_000);

    // 3. Dirección opuesta Short en cualquier escala: no colisiona con Longs abiertos
    let s2 = mgr.find_resonant_slot(15_000.0, false);
    assert_eq!(s2, Some(2), "Short no colisiona con Longs en la misma escala");
}

#[test]
fn spectral_slot_entradas_no_finitas_fallan_seguro() {
    let mgr = PositionManager::default();
    open_slot(&mgr, 0, true, 30_000);

    for nan_val in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -50.0, 0.0, 5.0] {
        // En la misma dirección debe usar el fallback seguro (30,000 ms) y colisionar con la abierta
        assert_eq!(mgr.find_resonant_slot(nan_val, true), None);
        // En dirección opuesta Short, entra al primer slot libre (1)
        assert_eq!(mgr.find_resonant_slot(nan_val, false), Some(1));
    }
}
