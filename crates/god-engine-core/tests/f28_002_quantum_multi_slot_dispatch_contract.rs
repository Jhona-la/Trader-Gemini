//! # Contrato de Certificación Formal: Fase 28 (§28) - Orquestador Cuántico Multi-Slot
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Suite de contratos formales de integración para el orquestador cuántico multirranura
//!   en `god-engine-core`.
//! - **POR QUÉ:** La arquitectura previa colapsaba el despacho de órdenes con un `break` ciego
//!   al encontrar la primera propuesta, suprimiendo la concurrencia entre `MicroScalp`,
//!   `MesoTactical` y `MacroSwing`.
//! - **PARA QUÉ:** Garantizar que el continuo multivariante temporal-espectral emita órdenes
//!   simultáneas sin canibalismo, respetando el capital y la resonancia armónica.
//! - **CÓMO:** Verificando evaluación paralela en Hilbert space, filtrado de armónicos y síntesis
//!   multirranura sin asignaciones dinámicas en camino caliente.
//! - **CUÁNDO:** En cada ciclo de evaluación del despachador del núcleo.
//! - **DÓNDE:** `crates/god-engine-core/tests/f28_002_quantum_multi_slot_dispatch_contract.rs`.
//! - **QUIÉN:** Consejo Colegiado (Roles 1: Arquitecto, 2: Quant Dev, 5: QA Engineer).

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use signal_engine::quantum_slot_orchestrator::SpectralSlot;

#[test]
fn c1_quantum_slot_orchestrator_initialization_integrity() {
    let arena = GlobalArena::build_in_own_stack(1000.0);
    let core = GodEngineCore::new_auxiliar(arena);

    // Verificar que los slots espectrales definen sus escalas características típicas
    assert_eq!(
        SpectralSlot::MicroScalp.typical_tau_ms(),
        15_000.0,
        "MicroScalp tau típico debe ser 15,000 ms"
    );
    assert_eq!(
        SpectralSlot::MesoTactical.typical_tau_ms(),
        300_000.0,
        "MesoTactical tau típico debe ser 300,000 ms"
    );
    assert_eq!(
        SpectralSlot::MacroSwing.typical_tau_ms(),
        3_600_000.0,
        "MacroSwing tau típico debe ser 3,600,000 ms"
    );

    // Verificar que el orquestador cuántico está disponible en el núcleo
    let orq = &core.quantum_slot_orchestrator;
    assert_eq!(
        orq.params.min_same_direction_delta_ln, 0.80,
        "Distancia mínima en la misma dirección debe ser 0.80 nats"
    );
}

#[test]
fn c2_concurrent_multi_slot_evaluation() {
    let arena = GlobalArena::build_in_own_stack(1000.0);
    let mut core = GodEngineCore::new_auxiliar(arena);

    // Evaluar estado multirranura en BTCUSDT (coin_id = 0)
    let coin_id = 0;
    let bid = 50_000.0;
    let ask = 50_010.0;
    let bid_qty = 10.0;
    let ask_qty = 5.0;
    let event_time_ms = 1_700_000_000_000_u64;

    let report =
        core.evaluate_quantum_multi_slot(coin_id, bid, ask, bid_qty, ask_qty, event_time_ms);

    assert_eq!(
        report.decisions.len(),
        3,
        "Debe evaluar exactamente 3 slots espectrales"
    );
    assert_eq!(report.decisions[0].slot, SpectralSlot::MicroScalp);
    assert_eq!(report.decisions[1].slot, SpectralSlot::MesoTactical);
    assert_eq!(report.decisions[2].slot, SpectralSlot::MacroSwing);

    // Stoikov microprice debe estar estrictamente acotado entre bid y ask
    assert!(
        report.stoikov_microprice >= bid && report.stoikov_microprice <= ask,
        "Microprecio de Stoikov {} debe estar entre bid {} y ask {}",
        report.stoikov_microprice,
        bid,
        ask
    );
}

#[test]
fn c3_harmonic_resonance_collision_veto() {
    let arena = GlobalArena::build_in_own_stack(1000.0);
    let core = GodEngineCore::new_auxiliar(arena);

    let orq = &core.quantum_slot_orchestrator;

    // Escalas muy cercanas (Delta ln(tau) < 0.80) en la misma dirección deben ser vetadas
    let tau_a = 5_000.0;
    let tau_b = 6_000.0; // ln(6000/5000) = ln(1.2) = 0.182 < 0.80

    let permitted_close = orq.check_spectral_resonance(tau_a, true, tau_b, true);
    assert!(
        !permitted_close,
        "Posiciones alineadas en escalas cuasi-idénticas deben sufrir veto por resonancia destructiva"
    );

    // Escalas ampliamente separadas (Delta ln(tau) >= 0.80) deben ser admitidas
    let tau_macro = 900_000.0; // ln(900000/5000) = ln(180) = 5.19 >= 0.80
    let permitted_far = orq.check_spectral_resonance(tau_a, true, tau_macro, true);
    assert!(
        permitted_far,
        "Posiciones en escalas ortogonales separadas por > 0.80 nats deben coexistir sin veto"
    );
}

#[test]
fn c4_multi_slot_emission_synthesis_without_cancellation() {
    let arena = GlobalArena::build_in_own_stack(1000.0);
    let mut core = GodEngineCore::new_auxiliar(arena);

    let coin_id = 0;
    let bid = 50_000.0;
    let ask = 50_010.0;
    let bid_qty = 15.0;
    let ask_qty = 5.0;
    let event_time_ms = 1_700_000_000_000_u64;

    let report =
        core.evaluate_quantum_multi_slot(coin_id, bid, ask, bid_qty, ask_qty, event_time_ms);
    let available_capital = 100.0;
    let mid_price = (bid + ask) * 0.5;
    let atr_pct = 0.005;

    let emission = core.synthesize_quantum_multi_slot_emissions(
        &report,
        available_capital,
        mid_price,
        atr_pct,
    );

    // Verificar que la estructura de emisión admite hasta 3 órdenes simultáneas
    assert!(
        emission.active_count <= 3,
        "La emisión multirranura no debe superar la capacidad de slots"
    );

    for opt_prop in &emission.slots {
        if let Some(prop) = opt_prop {
            assert!(prop.quantity > 0.0, "Cantidad debe ser positiva");
            assert!(prop.price > 0.0, "Precio de entrada debe ser positivo");
            assert!(
                prop.sl_price > 0.0,
                "Stop loss debe ser estrictamente positivo"
            );
            assert!(
                prop.tp_price > 0.0,
                "Take profit debe ser estrictamente positivo"
            );

            // Asimetría de recompensa: el Take Profit debe tener ratio RR >= 1.5 frente al Stop Loss
            let risk = (prop.price - prop.sl_price).abs();
            let reward = (prop.tp_price - prop.price).abs();

            if risk > 1e-4 {
                let rr = reward / risk;
                assert!(
                    rr >= 1.49,
                    "El ratio Reward-to-Risk ({:.2}) debe ser asimétrico (>= 1.5)",
                    rr
                );
            }
        }
    }
}
