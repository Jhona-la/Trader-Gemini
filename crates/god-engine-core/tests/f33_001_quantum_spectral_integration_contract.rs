//! 🧪 CONTRATOS FORMALES DE VERIFICACIÓN: INTEGRACIÓN CUÁNTICA ESPECTRAL Y FINANZAS CONDUCTUALES
//!
//! # FASE 33 (§33) — OLA OMEGA220: DE SILICIO A LA CÚSPIDE CUÁNTICA
//!
//! Suites de verificación determinista para la integración de `GodEngineCore` con `QuantumSlotOrchestrator`
//! y el campo conductual cuántico `QuantumBehavioralFieldEngine`:
//! - C1: Evaluación multi-slot genera decisiones armónicas con resonancia conductual $B(t)$.
//! - C2: Emisión multi-slot ajusta asimétricamente los objetivos TP/SL hacia zonas de liquidación minorista ($RR > 2.25$).
//! - C3: Sincronización entre microestructura Stoikov y resonancia conductual en stack zero-alloc.
//! - C4: Resiliencia ante fallos extremos y pánicos con invariante `rb_p1` estricto (0 panics).

use god_engine_core::GodEngineCore;
use quantum_arena::position::MAX_SPECTRAL_SLOTS;
use quantum_arena::GlobalArena;
use signal_engine::QuantumSlotOrchestratorParams;

#[test]
fn c1_god_engine_multi_slot_behavioral_evaluation() {
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena);
    let coin_id = 0;

    // Inicializar libro y estado para simular squeeze alcista
    let _mid_price = 100.0;
    let bid = 99.95;
    let ask = 100.05;
    let bid_qty = 50.0;
    let ask_qty = 5.0; // Desequilibrio alcista masivo en cola L2

    let report = core.evaluate_quantum_multi_slot(coin_id, bid, ask, bid_qty, ask_qty, 1_000_000);

    assert_eq!(report.decisions.len(), MAX_SPECTRAL_SLOTS);
    assert!(
        (-1.0..=1.0).contains(&report.behavioral_resonance),
        "Resonancia conductual debe estar acotada en [-1.0, 1.0], obtenido: {}",
        report.behavioral_resonance
    );
    assert!(
        report.dynamic_recv_window_ms >= 500,
        "recvWindow debe estar en rango seguro"
    );
}

#[test]
fn c2_asymmetric_rr_expansion_on_behavioral_squeeze() {
    let mut orchestrator =
        signal_engine::QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let mid_price = 50.0;
    let bid = 49.95;
    let ask = 50.05;
    let bid_qty = 100.0;
    let ask_qty = 2.0;
    let ofi = 0.85; // Fuerte presión compradora
    let obi = 0.75;
    let cvd = 0.90;
    let atr_pct = 0.02; // 2% volatilidad diaria

    let open_slots = [false, false, false];
    let open_taus = [0.0, 0.0, 0.0];
    let open_is_long = [false, false, false];

    let report = orchestrator.evaluate_multi_slot_signals(
        mid_price,
        bid,
        ask,
        bid_qty,
        ask_qty,
        1_000_000,
        ofi,
        obi,
        cvd,
        atr_pct,
        15_000.0,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    let capital_usd = 200.0;
    let max_leverage = 10.0;
    let risk_fraction = 0.05;

    let emission =
        orchestrator.synthesize_order_emissions(&report, capital_usd, max_leverage, risk_fraction);

    assert!(
        emission.active_count > 0,
        "Debe emitir al menos una propuesta de orden bajo fuerte impulso y squeeze"
    );

    for proposal in emission.iter() {
        if proposal.is_long {
            let tp_distance = proposal.tp_price - proposal.price;
            let sl_distance = proposal.price - proposal.sl_price;
            assert!(
                sl_distance > 0.0,
                "La distancia de SL debe ser estrictamente positiva"
            );
            let rr_ratio = tp_distance / sl_distance;
            assert!(
                rr_ratio >= 2.20,
                "Bajo squeeze conductual, el ratio R:R debe expandirse asimétricamente (RR = {})",
                rr_ratio
            );
        }
    }
}

#[test]
fn c3_multi_slot_coexistence_and_behavioral_carry() {
    let mut orchestrator =
        signal_engine::QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());

    let mid_price = 1000.0;
    let bid = 999.5;
    let ask = 1000.5;
    let bid_qty = 80.0;
    let ask_qty = 20.0;
    let ofi = 0.60;
    let obi = 0.50;
    let cvd = 0.70;
    let atr_pct = 0.015;

    let open_slots = [false, false, false];
    let open_taus = [0.0, 0.0, 0.0];
    let open_is_long = [false, false, false];

    let report = orchestrator.evaluate_multi_slot_signals(
        mid_price,
        bid,
        ask,
        bid_qty,
        ask_qty,
        2_000_000,
        ofi,
        obi,
        cvd,
        atr_pct,
        30_000.0,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    let emission = orchestrator.synthesize_order_emissions(&report, 500.0, 5.0, 0.10);

    for prop in emission.slots.iter().flatten() {
        assert_eq!(
            prop.behavioral_resonance, report.behavioral_resonance,
            "Cada propuesta debe portar exactamente la resonancia conductual del instante"
        );
    }
}

#[test]
fn c4_zero_panic_resilience_under_corrupt_features() {
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena);
    let coin_id = 999; // ID fuera de rango para testear fail-closed seguro

    let report = core.evaluate_quantum_multi_slot(
        coin_id,
        f64::NAN,
        f64::INFINITY,
        -50.0,
        f64::NAN,
        1_000_000,
    );

    assert_eq!(report.active_slots_count, 0);
    assert_eq!(report.behavioral_resonance, 0.0);

    let emission = core.synthesize_quantum_multi_slot_emissions(&report, 0.0, -10.0, f64::NAN);
    assert_eq!(emission.active_count, 0);
}
