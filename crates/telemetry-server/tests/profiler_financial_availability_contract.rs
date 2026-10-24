//! F1-099: un PnL ilegible no puede presentarse como 0.0 en el informe.
//!
//! Contrato sobre las funciones REALES de publicación financiera
//! (`Profiler::update_financials`, `update_financials_atomic`) y sobre la regla
//! de disponibilidad que usa el auditor: `is_finite()` ⇒ valor real;
//! no finito ⇒ `UNAVAILABLE`, nunca un cero fabricado.
//!
//! No hay sockets, órdenes, cuentas, modelos ni acceso a la API de Binance:
//! son valores deterministas. Un solo test posee los globales del proceso para
//! que los escenarios no se roben entre sí bajo el runner paralelo por defecto.

use std::sync::atomic::Ordering;
use telemetry_server::profiler::{
    Profiler, ROI_NET, TOTAL_FEES, TOTAL_GROSS_PNL, TOTAL_NET_PNL, WIN_RATE,
    update_financials_atomic,
};

#[test]
fn financial_availability_end_to_end() {
    unreadable_stays_unreadable();
    partial_corruption_does_not_fabricate_fees();
    genuine_zero_is_reported_as_zero();
    capital_without_value_makes_roi_unavailable();
    fresh_snapshot_claims_nothing();
    finite_values_are_reported_exactly();
}

fn unreadable_stays_unreadable() {
    let mut p = Profiler::new();
    p.update_financials(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    assert!(
        p.total_gross_pnl.is_nan(),
        "gross ilegible no puede ser 0.0"
    );
    assert!(p.net_pnl.is_nan(), "net ilegible no puede ser 0.0");
    assert!(
        p.total_fees.is_nan(),
        "fees no pueden derivarse de un neto inexistente"
    );
    assert!(p.win_rate.is_nan());
    assert!(p.roi_net.is_nan(), "ROI sin capital válido no es 0.0%");
}

fn partial_corruption_does_not_fabricate_fees() {
    // El caso vivo peligroso: bruto legible, neto no. El código anterior ponía
    // net = 0.0 y publicaba fees == bruto, es decir INVENTABA una comisión
    // igual a todo el bruto mientras afirmaba «Net PnL: 0.0000».
    let mut p = Profiler::new();
    p.update_financials(20.0, f64::NAN, 0.8, 13.0);
    assert_eq!(p.total_gross_pnl, 20.0, "el bruto legible se conserva");
    assert!(p.net_pnl.is_nan(), "el neto ilegible sigue ilegible");
    assert!(
        p.total_fees.is_nan(),
        "fees desconocidas, no inventadas como el bruto completo"
    );
    assert!(p.roi_net.is_nan(), "el ROI depende del neto");
}

fn genuine_zero_is_reported_as_zero() {
    // Un cero real (sin trades, o book exactamente en break-even) debe
    // reportarse como cero. «No disponible» es NaN, nunca «todo es cero».
    let mut p = Profiler::new();
    p.update_financials(0.0, 0.0, 0.0, 13.0);
    assert_eq!(p.total_gross_pnl, 0.0);
    assert_eq!(p.net_pnl, 0.0);
    assert_eq!(p.total_fees, 0.0);
    assert_eq!(p.win_rate, 0.0);
    assert!(p.roi_net.is_finite());
    assert_eq!(p.roi_net, 0.0);
}

fn capital_without_value_makes_roi_unavailable() {
    let mut p = Profiler::new();
    p.update_financials(5.0, 5.0, 1.0, 0.0);
    assert_eq!(p.total_gross_pnl, 5.0);
    assert_eq!(p.net_pnl, 5.0);
    assert_eq!(p.total_fees, 0.0);
    assert!(
        p.roi_net.is_nan(),
        "sin capital positivo el ROI es desconocido; 0.0% afirmaría que no se ganó nada"
    );
}

fn fresh_snapshot_claims_nothing() {
    // Antes de cualquier publicación financiera, el snapshot no puede afirmar
    // ceros. Default debe coincidir con new(): «sin datos», no «cero».
    for p in [Profiler::new(), Profiler::default()] {
        assert!(p.total_gross_pnl.is_nan());
        assert!(p.net_pnl.is_nan());
        assert!(p.total_fees.is_nan());
        assert!(p.win_rate.is_nan());
        assert!(p.roi_net.is_nan());
    }
}

fn finite_values_are_reported_exactly() {
    // Los atómicos guardan los bits crudos: lo que llegó se recupera tal cual
    // para forense, y el informe decide disponibilidad por is_finite().
    update_financials_atomic(-7.25, -6.0, 0.4, 50.0);
    assert_eq!(
        f64::from_bits(TOTAL_GROSS_PNL.load(Ordering::Relaxed)),
        -7.25
    );
    assert_eq!(f64::from_bits(TOTAL_NET_PNL.load(Ordering::Relaxed)), -6.0);
    assert_eq!(f64::from_bits(TOTAL_FEES.load(Ordering::Relaxed)), -1.25);
    assert_eq!(f64::from_bits(WIN_RATE.load(Ordering::Relaxed)), 0.4);
    assert!((f64::from_bits(ROI_NET.load(Ordering::Relaxed)) - (-12.0)).abs() < 1e-9);

    // Un ROI finito extremo se reporta tal como se midió: el clamp arbitrario
    // (-1000.0, 100000.0) reescribía una medición real.
    update_financials_atomic(2_000_000.0, 2_000_000.0, 1.0, 1.0);
    assert_eq!(f64::from_bits(TOTAL_FEES.load(Ordering::Relaxed)), 0.0);
    assert_eq!(
        f64::from_bits(ROI_NET.load(Ordering::Relaxed)),
        200_000_000.0
    );
}
