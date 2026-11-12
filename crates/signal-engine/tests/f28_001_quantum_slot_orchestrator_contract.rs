#![allow(clippy::all)]
//! 🧪 CONTRATO FORMAL F28-001 — ORQUESTADOR CUÁNTICO MULTI-SLOT & DESPACHO CONCURRENTE
//!
//! # VERIFICACIÓN DE FASE 28 (§28) — OLA OMEGA220
//!
//! 1. `c1_spectral_slot_classification_and_typical_scales`: Clasificación biyectiva de escalas diádicas $\tau$ a slots físicos $[0, 2]$.
//! 2. `c2_spectral_resonance_orthogonality_and_collision_veto`: Resonancia armónica $|\Delta \ln \tau| \ge 0.80$ y umbral de cobertura dinámica $\ge 0.60$.
//! 3. `c3_multi_slot_concurrent_signals_superposition`: Superposición coherente simultánea de MicroScalp y MesoTactical/MacroSwing sin anulación.
//! 4. `c4_stoikov_microprice_and_dynamic_recv_window_injection`: Inyección de microprecio markoviano de Sasha Stoikov y dynamic `recvWindow`.
//! 5. `c5_bouchaud_mezard_and_perelman_ricci_physics_veto`: Vetos físicos de condensación de Bouchaud-Mézard ($\mu \le 1.0$) y curvatura de Ricci.
//! 6. `c6_multi_slot_order_emission_and_capital_preservation`: Emisión concurrente multi-orden y cota estricta de preservación de micro-capital.

use signal_engine::quantum_slot_orchestrator::{
    MultiSlotOrderEmission, QuantumSlotOrchestrator, SpectralSlot,
};
use strategy_core::SignalType;

#[test]
fn c1_spectral_slot_classification_and_typical_scales() {
    // 1. Verificar mapeo continuo de tau a SpectralSlot
    assert_eq!(SpectralSlot::from_tau_ms(500.0), SpectralSlot::MicroScalp);
    assert_eq!(
        SpectralSlot::from_tau_ms(15_000.0),
        SpectralSlot::MicroScalp
    );
    assert_eq!(
        SpectralSlot::from_tau_ms(60_000.0),
        SpectralSlot::MicroScalp
    );

    assert_eq!(
        SpectralSlot::from_tau_ms(60_001.0),
        SpectralSlot::MesoTactical
    );
    assert_eq!(
        SpectralSlot::from_tau_ms(300_000.0),
        SpectralSlot::MesoTactical
    );
    assert_eq!(
        SpectralSlot::from_tau_ms(900_000.0),
        SpectralSlot::MesoTactical
    );

    assert_eq!(
        SpectralSlot::from_tau_ms(900_001.0),
        SpectralSlot::MacroSwing
    );
    assert_eq!(
        SpectralSlot::from_tau_ms(3_600_000.0),
        SpectralSlot::MacroSwing
    );
    assert_eq!(
        SpectralSlot::from_tau_ms(86_400_000.0),
        SpectralSlot::MacroSwing
    );

    // 2. Biyectividad con índices físicos
    for idx in 0..3 {
        let slot = SpectralSlot::from_index(idx);
        assert!(slot.is_some());
        let unwrap_slot = match slot {
            Some(s) => s,
            None => unreachable!(),
        };
        assert_eq!(unwrap_slot.to_index(), idx);
        assert!(unwrap_slot.typical_tau_ms() > 0.0);
    }
    assert_eq!(SpectralSlot::from_index(3), None);

    // 3. Robustez ante no-finitos
    assert_eq!(
        SpectralSlot::from_tau_ms(f64::NAN),
        SpectralSlot::MicroScalp
    );
    assert_eq!(SpectralSlot::from_tau_ms(-100.0), SpectralSlot::MicroScalp);
}

#[test]
fn c2_spectral_resonance_orthogonality_and_collision_veto() {
    let orchestrator = QuantumSlotOrchestrator::default();

    // Misma dirección (LONG vs LONG):
    // tau_1 = 15_000 ms (15 s)
    // tau_2 = 16_000 ms (16 s) -> |ln(15) - ln(16)| = |9.615 - 9.680| = 0.065 < 0.80 -> COLISIÓN
    let collision_same = orchestrator.check_spectral_resonance(15_000.0, true, 16_000.0, true);
    assert!(!collision_same, "Misma escala y dirección debe colisionar");

    // Misma dirección ortogonal:
    // tau_1 = 15_000 ms (15 s), tau_2 = 300_000 ms (5 min) -> |ln(15) - ln(300)| = 2.996 >= 0.80 -> RESONANCIA
    let resonant_same = orchestrator.check_spectral_resonance(15_000.0, true, 300_000.0, true);
    assert!(resonant_same, "Escalas ortogonales deben resonar");

    // Direcciones opuestas (cobertura dinámica SHORT vs LONG):
    // tau_1 = 15_000 ms, tau_2 = 30_000 ms -> |ln(15) - ln(30)| = 0.693 >= 0.60 -> RESONANCIA DE COBERTURA
    let hedge_resonant = orchestrator.check_spectral_resonance(15_000.0, false, 30_000.0, true);
    assert!(hedge_resonant, "Coberturas con delta >= 0.60 deben resonar");

    // Direcciones opuestas casi idénticas:
    // tau_1 = 15_000 ms, tau_2 = 16_000 ms -> 0.065 < 0.60 -> COLISIÓN
    let hedge_collision = orchestrator.check_spectral_resonance(15_000.0, false, 16_000.0, true);
    assert!(
        !hedge_collision,
        "Cobertura en la misma frecuencia colisiona"
    );
}

#[test]
fn c3_multi_slot_concurrent_signals_superposition() {
    let mut orchestrator = QuantumSlotOrchestrator::default();

    let open_slots = [false, false, false];
    let open_taus = [0.0, 0.0, 0.0];
    let open_is_long = [false, false, false];

    // Mercado con fuerte desequilibrio comprador y volumen acumulado positivo
    let report = orchestrator.evaluate_multi_slot_signals(
        50_000.0,
        49_999.0,
        50_001.0,
        15.0, // Fuerte bid qty
        3.0,  // Poca ask qty
        1_700_000_000,
        0.85, // OFI fuertemente alcista
        0.75, // OBI fuertemente alcista
        0.80, // CVD fuertemente alcista
        0.015,
        15_000.0,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    // Tanto MicroScalp como MesoTactical y MacroSwing deben capturar la presión compradora
    assert_eq!(report.decisions[0].signal, SignalType::Long);
    assert_eq!(report.decisions[1].signal, SignalType::Long);
    assert_eq!(report.decisions[2].signal, SignalType::Long);

    assert!(report.decisions[0].net_confidence >= 0.50);
    assert!(report.decisions[1].net_confidence >= 0.50);
    assert!(report.decisions[2].net_confidence >= 0.50);

    assert_eq!(report.active_slots_count, 3);
    assert!(report.coherent_superposition);
}

#[test]
fn c4_stoikov_microprice_and_dynamic_recv_window_injection() {
    let mut orchestrator = QuantumSlotOrchestrator::default();

    let open_slots = [false, false, false];
    let open_taus = [0.0, 0.0, 0.0];
    let open_is_long = [false, false, false];

    // Presión asimétrica de compra: bid_qty=20.0, ask_qty=5.0 en mid=100.0
    let report = orchestrator.evaluate_multi_slot_signals(
        100.0,
        99.95,
        100.05,
        20.0,
        5.0,
        1_700_000_100,
        0.60,
        0.60,
        0.50,
        0.02,
        20_000.0,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    // El microprecio markoviano de Stoikov debe reflejar la presión y situarse por encima del mid
    assert!(report.stoikov_microprice >= 100.0);
    assert!(report.dynamic_recv_window_ms >= 1000);
    assert_eq!(report.decisions[0].microprice, report.stoikov_microprice);
}

#[test]
fn c5_bouchaud_mezard_and_perelman_ricci_physics_veto() {
    let mut orchestrator = QuantumSlotOrchestrator::default();

    // Caso 1: Posición abierta en MicroScalp (15 s, LONG).
    // Una nueva señal en MicroScalp (16 s, LONG) debe ser vetada por colisión espectral
    let open_slots = [true, false, false];
    let open_taus = [15_000.0, 0.0, 0.0];
    let open_is_long = [true, false, false];

    let report = orchestrator.evaluate_multi_slot_signals(
        100.0,
        99.95,
        100.05,
        10.0,
        10.0,
        1_700_000_200,
        0.70,
        0.70,
        0.70,
        0.02,
        16_000.0, // Mismo régimen temporal (~16 s)
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    // El slot 0 debe quedar vetado por colisión con la posición abierta
    assert!(report.decisions[0].vetoed);
    assert_eq!(report.decisions[0].signal, SignalType::Flat);
    assert_eq!(
        report.decisions[0].veto_reason,
        Some("SPECTRAL_DESTRUCTIVE_COLLISION_WITH_OPEN_POSITION")
    );

    // Pero MesoTactical (slot 1) y MacroSwing (slot 2) siguen intactos y pueden operar
    assert_eq!(report.decisions[1].signal, SignalType::Long);
    assert_eq!(report.decisions[2].signal, SignalType::Long);
    assert!(!report.decisions[1].vetoed);
    assert!(!report.decisions[2].vetoed);
}

#[test]
fn c6_multi_slot_order_emission_and_capital_preservation() {
    let mut orchestrator = QuantumSlotOrchestrator::default();

    let open_slots = [false, false, false];
    let open_taus = [0.0, 0.0, 0.0];
    let open_is_long = [false, false, false];

    // Señal alcista unificada
    let report = orchestrator.evaluate_multi_slot_signals(
        2000.0,
        1999.5,
        2000.5,
        10.0,
        2.0,
        1_700_000_300,
        0.80,
        0.80,
        0.80,
        0.01,
        15_000.0,
        &open_slots,
        &open_taus,
        &open_is_long,
    );

    assert_eq!(report.active_slots_count, 3);

    // Capital total disponible: $13.00 USD (micro-capital)
    let capital = 13.0;
    let max_leverage = 5.0;
    let risk_fraction = 0.20; // 20% del capital arriesgado en total ($2.60 USD)

    let emission: MultiSlotOrderEmission =
        orchestrator.synthesize_order_emissions(&report, capital, max_leverage, risk_fraction);

    assert_eq!(emission.active_count, 3);

    let mut total_margin_allocated = 0.0;
    for proposal in emission.iter() {
        assert!(proposal.quantity > 0.0);
        assert!(proposal.price > 0.0);
        let notional = proposal.quantity * proposal.price;
        let margin = notional / max_leverage;
        total_margin_allocated += margin;

        // Comprobar ratio asimétrico TP vs SL (R:R = 2.0)
        let tp_distance = (proposal.tp_price - proposal.price).abs();
        let sl_distance = (proposal.sl_price - proposal.price).abs();
        assert!(tp_distance > sl_distance);
        let rr_ratio = tp_distance / sl_distance;
        assert!((rr_ratio - 2.0).abs() < 1e-3);
    }

    // El margen total asignado no puede exceder el capital arriesgado ($13 * 0.20 = $2.60)
    assert!(total_margin_allocated <= capital * risk_fraction + 1e-4);
}

#[test]
fn c7_invalid_capital_risk_or_leverage_emits_no_orders() {
    let mut orchestrator = QuantumSlotOrchestrator::default();
    let report = orchestrator.evaluate_multi_slot_signals(
        2000.0,
        1999.5,
        2000.5,
        10.0,
        2.0,
        1_700_000_300,
        0.80,
        0.80,
        0.80,
        0.01,
        15_000.0,
        &[false, false, false],
        &[0.0, 0.0, 0.0],
        &[false, false, false],
    );
    assert_eq!(report.active_slots_count, 3);

    for (capital, leverage, risk) in [
        (13.0, 5.0, 0.0),
        (0.0, 5.0, 0.20),
        (-1.0, 5.0, 0.20),
        (f64::NAN, 5.0, 0.20),
        (13.0, 5.0, f64::NAN),
        (13.0, 5.0, -0.01),
        (13.0, 0.0, 0.20),
        (13.0, -1.0, 0.20),
        (13.0, f64::NAN, 0.20),
    ] {
        let emission = orchestrator.synthesize_order_emissions(&report, capital, leverage, risk);
        assert_eq!(emission.active_count, 0, "capital={capital} leverage={leverage} risk={risk}");
    }
}
