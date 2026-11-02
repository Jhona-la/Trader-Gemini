//! Suite de Contratos Formales F16-001 — Libro Plano Structure-of-Arrays (`SoaOrderBookL2`),
//! Libro Sombra de Margen en Tránsito (`InFlightShadowLedger`) y Caos Cuántico OTOC / Cota MSS.
//!
//! Verifica las 6 propiedades físico-matemáticas y de silicio de la Fase 16:
//! - C1: Alineación de 64B e indexación directa $\mathcal{O}(1)$ bijeccional `price_to_slot` $\leftrightarrow$ `slot_to_price` en `SoaOrderBookL2`.
//! - C2: Actualización BBO en $\mathcal{O}(1)$, cálculo exacto de `micro_price` y recuperación automática del mejor nivel tras cancelación.
//! - C3: Deducción instantánea de margen y comisión en `InFlightShadowLedger`, rechazo de sobregiro y rechazo de `client_order_id` duplicado.
//! - C4: Liberación exacta por ACK (`release_by_order_id`) y recolección determinista de reservas expiradas por TTL (`reclaim_expired`).
//! - C5: Cota Universal de Caos de Maldacena-Shenker-Stanford ($\lambda_Q \le 2\pi T_{\text{eff}}$), crecimiento de $C(t)$ y tiempo de Ehrenfest $t_*$.
//! - C6: Inmunidad total frente a entradas `NaN` / `±Inf` en `SoaShadowOtocMasterEngine`.

use feature_engine::{
    InFlightShadowLedger, OtocShadowRegime, QuantumOtocEngine, SoaOrderBookL2,
    SoaShadowOtocMasterEngine,
};
use std::mem::align_of;

#[test]
fn c1_soa_orderbook_64b_alignment_and_o1_slot_bijection() {
    assert_eq!(align_of::<SoaOrderBookL2<128>>(), 64);

    let book: SoaOrderBookL2<128> = SoaOrderBookL2::new(50_000.0, 0.5);
    let mid_slot = book.price_to_slot(50_000.0);
    assert_eq!(mid_slot, Some(64));
    assert!((book.slot_to_price(64) - 50_000.0).abs() < 1e-12);

    let p_up = 50_010.5;
    let slot_up = book.price_to_slot(p_up);
    assert_eq!(slot_up, Some(64 + 21));
    assert!((book.slot_to_price(85) - p_up).abs() < 1e-12);

    // Precios fuera de la ventana contigua o inválidos devuelven None sin pánico
    assert_eq!(book.price_to_slot(99_999.0), None);
    assert_eq!(book.price_to_slot(-10.0), None);
    assert_eq!(book.price_to_slot(f64::NAN), None);
}

#[test]
fn c2_soa_orderbook_bbo_microprice_and_level_cancellation() {
    let mut book: SoaOrderBookL2<64> = SoaOrderBookL2::new(100.0, 0.1);
    assert!(book.update_bid(99.8, 10.0, 2));
    assert!(book.update_bid(99.9, 30.0, 5));
    assert!(book.update_ask(100.1, 10.0, 3));
    assert!(book.update_ask(100.2, 25.0, 4));

    // Micro-price en BBO (Bid 99.9 x 30, Ask 100.1 x 10):
    // P_micro = (30 * 100.1 + 10 * 99.9) / 40 = (3003 + 999) / 40 = 100.05
    let mp = book.micro_price();
    assert!((mp - 100.05).abs() < 1e-12, "Expected 100.05, got {mp}");

    // Cancelar el mejor Bid (99.9 -> qty 0.0) debe retroceder automáticamente a 99.8
    assert!(book.update_bid(99.9, 0.0, 0));
    let mp_after = book.micro_price();
    // Nuevo BBO: Bid 99.8 x 10, Ask 100.1 x 10 => punto medio 99.95
    assert!((mp_after - 99.95).abs() < 1e-12);
}

#[test]
fn c3_inflight_shadow_ledger_prevents_margin_overspend_and_duplicates() {
    let mut ledger: InFlightShadowLedger<8> = InFlightShadowLedger::new(5_000_000_000);
    let wallet_free = 100.0;

    // Orden 1: reserva 40.0 margen + 0.5 fee = 40.5 USD
    assert!(ledger.try_reserve_inflight(1001, 1, 40.0, 0.5, wallet_free, 1_000));
    assert!((ledger.effective_free_margin(wallet_free) - 59.5).abs() < 1e-12);

    // Intento con ID duplicado 1001 debe rechazarse
    assert!(!ledger.try_reserve_inflight(1001, 1, 10.0, 0.1, wallet_free, 1_100));

    // Orden 2: reserva 50.0 margen + 0.5 fee = 50.5 USD (quedan 9.0 USD libres)
    assert!(ledger.try_reserve_inflight(1002, 2, 50.0, 0.5, wallet_free, 1_200));
    assert!((ledger.effective_free_margin(wallet_free) - 9.0).abs() < 1e-12);

    // Orden 3 en ráfaga concurrente pide 15.0 USD (> 9.0 libres): debe ser bloqueada fail-closed
    assert!(!ledger.try_reserve_inflight(1003, 3, 15.0, 0.2, wallet_free, 1_300));
    assert_eq!(ledger.active_count, 2);
}

#[test]
fn c4_inflight_shadow_ledger_release_and_ttl_expiration() {
    let mut ledger: InFlightShadowLedger<8> = InFlightShadowLedger::new(1_000_000); // 1ms TTL
    let wallet_free = 50.0;

    assert!(ledger.try_reserve_inflight(2001, 1, 20.0, 0.2, wallet_free, 100));
    assert!(ledger.try_reserve_inflight(2002, 2, 15.0, 0.1, wallet_free, 500));
    assert_eq!(ledger.active_count, 2);

    // Liberar explícitamente la orden 2001 por llegada de ACK
    assert!(ledger.release_by_order_id(2001));
    assert_eq!(ledger.active_count, 1);
    assert!((ledger.total_blocked_usd() - 15.1).abs() < 1e-12);

    // En t = 1_600_000 ns, la orden 2002 superó su TTL y debe ser recolectada automáticamente
    let expired = ledger.reclaim_expired(1_600_000);
    assert_eq!(expired, 1);
    assert_eq!(ledger.active_count, 0);
    assert_eq!(ledger.total_blocked_usd(), 0.0);
    assert!((ledger.effective_free_margin(wallet_free) - 50.0).abs() < 1e-12);
}

#[test]
fn c5_quantum_otoc_maldacena_shenker_stanford_chaos_bound() {
    let engine = QuantumOtocEngine::default();
    let temp = 0.50;
    let expected_mss = 2.0 * std::f64::consts::PI * temp;

    // Acoplamiento débil => régimen integrable
    let weak = engine.evaluate_otoc(&[0.05, 0.04, 0.03, 0.02], temp, 0.5);
    assert!((weak.mss_chaos_bound - expected_mss).abs() < 1e-14);
    assert!(weak.quantum_lyapunov_lambda <= weak.mss_chaos_bound);
    assert!(weak.mss_saturation_ratio < 0.20);

    // Acoplamiento extremo => lambda_Q satura asintóticamente la cota de Maldacena-Shenker-Stanford sin violarla jamás
    let strong = engine.evaluate_otoc(&[40.0, -35.0, 45.0, -30.0], temp, 1.0);
    assert!(
        strong.quantum_lyapunov_lambda <= strong.mss_chaos_bound + 1e-12,
        "MSS universal chaos bound violated: {} > {}",
        strong.quantum_lyapunov_lambda,
        strong.mss_chaos_bound
    );
    assert!(strong.mss_saturation_ratio > 0.99);
    assert!(strong.otoc_correlator > weak.otoc_correlator);
    assert!(strong.ehrenfest_scrambling_time < weak.ehrenfest_scrambling_time);
}

#[test]
fn c6_nan_inf_adversarial_immunity_in_master_engine() {
    let master = SoaShadowOtocMasterEngine::default();
    let book: SoaOrderBookL2<32> = SoaOrderBookL2::new(f64::NAN, f64::NEG_INFINITY);
    let ledger: InFlightShadowLedger<4> = InFlightShadowLedger::default();
    let bad_couplings = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0];

    let rep = master.evaluate_cycle(
        &book,
        &ledger,
        f64::NAN,
        &bad_couplings,
        f64::NAN,
        f64::INFINITY,
    );

    assert!(rep.micro_price.is_finite());
    assert!(rep.depth_imbalance.is_finite());
    assert!(rep.effective_free_margin_usd.is_finite());
    assert!(rep.otoc.otoc_correlator.is_finite());
    assert!(rep.otoc.quantum_lyapunov_lambda <= rep.otoc.mss_chaos_bound + 1e-12);
    assert!(matches!(
        rep.regime,
        OtocShadowRegime::IntegrableLaminarFlow
            | OtocShadowRegime::InFlightMarginCongestion
            | OtocShadowRegime::FastScramblingChaosVeto
    ));
}
