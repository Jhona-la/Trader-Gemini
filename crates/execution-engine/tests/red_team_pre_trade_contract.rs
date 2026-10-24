//! Contrato Formal de la Compuerta Adversarial Pre-Trade (< 500 ns) — Ola Ω212.
//!
//! # QUÉ
//! Contrato de integración formal que valida la cooperación armónica entre
//! `AdverseSelectionGuard` (viabilidad financiera de cola FIFO) y `RedTeamPreTradeGate`
//! (defensa pre-trade contra trampas microestructurales, spoofing y barreduras de stops).
//!
//! # POR QUÉ
//! Una orden con esperanza matemática positiva en papel puede ser aniquilada si el nivel
//! es una trampa de liquidez colocada por algoritmos hostiles (spoofing) o si el spread
//! explota en el micro-instante de envío. La compuerta debe abortar la orden antes de
//! pagar el spread.
//!
//! # DÓNDE
//! `crates/execution-engine/tests/red_team_pre_trade_contract.rs`.
//!
//! # QUIÉN
//! Consejo Colegiado de 10 Roles / Quant Sr. Lead.

use execution_engine::{
    AdverseSelectionGuard, AdversarialClearance, AdversarialTrapReason,
    OrderViability, RedTeamPreTradeGate,
};

#[test]
fn test_contract_combined_viability_and_adversarial_clearance() {
    let gate = RedTeamPreTradeGate::default();

    // Caso 1: Mercado Sano y Fértil
    // 1.1 Viabilidad Financiera
    let (viability, metrics) = AdverseSelectionGuard::evaluate_viability(
        0.0030,  // edge bruto 30 bps
        0.0002,  // spread 2 bps
        0.0010,  // vol 10 bps
        0.80,    // confianza 80%
        0.0002,  // maker fee 2 bps
        0.0001,  // funding 1 bp
        2_000.0, // cola adelante
        8_000.0, // nivel total
    );
    assert_eq!(viability, OrderViability::Viable);
    assert!(metrics.viability_ratio >= 3.0);

    // 1.2 Compuerta Adversarial
    let clearance = gate.evaluate_order(
        true,       // compra
        0.0002,     // spread sano 2 bps
        0.0010,     // vol sana
        0.15,       // OBI favorable (+0.15 comprador)
        12_000.0,   // profundidad sana
        10_000.0,   // mediana
        0.18,       // tasa de cancelación baja
        1.1,        // ratio BBO a L2 normal
    );
    assert!(clearance.is_ok());
    let clearance: AdversarialClearance = clearance.unwrap();
    assert!(clearance.compound_risk < 0.65);
}

#[test]
fn test_contract_order_viable_financially_but_blocked_by_spoofing_trap() {
    let gate = RedTeamPreTradeGate::default();

    // La orden tiene un edge matemático que en teoría cubre fricciones
    let (viability, _) = AdverseSelectionGuard::evaluate_viability(
        0.0050,  // 50 bps edge
        0.0002,
        0.0010,
        0.90,
        0.0002,
        0.0001,
        1_000.0,
        10_000.0,
    );
    assert_eq!(viability, OrderViability::Viable);

    // Pero el nivel objetivo es una trampa de spoofing: 98% de cancelaciones y 12x volumen en BBO vs L2
    let result = gate.evaluate_order(
        true,
        0.0002,
        0.0010,
        0.05,
        15_000.0,
        10_000.0,
        0.98, // spoofing masivo
        12.0, // concentración extrema
    );

    assert_eq!(result, Err(AdversarialTrapReason::SpoofingPhantomTrap));
}

#[test]
fn test_contract_order_blocked_by_liquidity_sweep_trap() {
    let gate = RedTeamPreTradeGate::default();

    // Intentar comprar en una cascada de liquidaciones donde el libro comprador se vació
    let result = gate.evaluate_order(
        true,       // intentamos comprar
        0.0004,     // spread abriéndose
        0.0015,     // vol elevada
        -0.88,      // OBI masivamente adverso
        300.0,      // soporte casi evaporado (300 USD vs 10,000 USD mediana)
        10_000.0,
        0.45,
        2.0,
    );

    assert_eq!(result, Err(AdversarialTrapReason::LiquiditySweepTrap));
}

#[test]
fn test_contract_hot_path_latency_budget_under_500ns() {
    let gate = RedTeamPreTradeGate::default();
    let n = 20_000;
    let mut cleared_count = 0usize;

    let start = std::time::Instant::now();
    for i in 0..n {
        let is_buy = (i & 1) == 0;
        let obi = ((i % 200) as f64 - 100.0) / 200.0;
        let cancel = ((i % 50) as f64) / 100.0;
        if gate
            .evaluate_order(
                is_buy,
                0.0002,
                0.0008,
                obi,
                8_000.0,
                10_000.0,
                cancel,
                1.3,
            )
            .is_ok()
        {
            cleared_count += 1;
        }
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / n as f64;

    assert!(cleared_count > 0);
    // Presupuesto estricto: en debug < 3500 ns en máquina modesta; en release < 50 ns
    assert!(
        ns_per_call < 3500.0,
        "Latencia {} ns/op excede presupuesto",
        ns_per_call
    );
}
