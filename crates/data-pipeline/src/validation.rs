//! VALIDACIÓN DE FRONTERA (F2.1) — la aduana de datos del sistema.
//!
//! QUÉ: todo dato que cruza una frontera (WS→motor, REST→features, disco→backtest)
//!      pasa por aquí ANTES de tocar estado, features o entrenamiento.
//! POR QUÉ: directriz — "la información puede llegar falsa o no llegar, y
//!      podríamos estar entrenando con valores sin sentido". Un NaN en una
//!      feature contamina el tensor; un book cruzado dispara señales falsas.
//! INVARIANTES por evento:
//!      1. Precios finitos, > 0.
//!      2. bid <= ask (libro cruzado = dato corrupto o replay desordenado).
//!      3. Spread relativo <= 50% (política heredada de admisión, NO prueba
//!         de corrupción ni umbral calibrado para todo activo).
//!      4. Cantidades finitas, >= 0.
//!      5. event_time > 0 (0 = campo sin parsear).
//! Contadores atómicos por razón — telemetría F6 leerá "cuánta basura
//!      rechazamos por tipo" sin tocar el hot-path.

use crate::parser::BookTickerEvent;
use std::sync::atomic::{AtomicU64, Ordering};

/// Política heredada, no ley de mercado. XXXIII conserva el umbral y corrige
/// solo la estabilidad numérica; calibración por activo/liquidez sigue pendiente.
const MAX_RELATIVE_SPREAD: f64 = 0.50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    PriceNotFinite,
    PriceNotPositive,
    CrossedBook,
    AbsurdSpread,
    QtyNotFinite,
    BadEventTime,
}

impl RejectReason {
    pub fn label(self) -> &'static str {
        match self {
            RejectReason::PriceNotFinite => "price_not_finite",
            RejectReason::PriceNotPositive => "price_not_positive",
            RejectReason::CrossedBook => "crossed_book",
            RejectReason::AbsurdSpread => "absurd_spread",
            RejectReason::QtyNotFinite => "qty_not_finite",
            RejectReason::BadEventTime => "bad_event_time",
        }
    }
}

macro_rules! reject_counter {
    ($name:ident) => {
        static $name: AtomicU64 = AtomicU64::new(0);
    };
}

reject_counter!(REJ_PRICE_FINITE);
reject_counter!(REJ_PRICE_POSITIVE);
reject_counter!(REJ_CROSSED);
reject_counter!(REJ_SPREAD);
reject_counter!(REJ_QTY);
reject_counter!(REJ_TIME);

pub fn count_reject(reason: RejectReason) {
    match reason {
        RejectReason::PriceNotFinite => REJ_PRICE_FINITE.fetch_add(1, Ordering::Relaxed),
        RejectReason::PriceNotPositive => REJ_PRICE_POSITIVE.fetch_add(1, Ordering::Relaxed),
        RejectReason::CrossedBook => REJ_CROSSED.fetch_add(1, Ordering::Relaxed),
        RejectReason::AbsurdSpread => REJ_SPREAD.fetch_add(1, Ordering::Relaxed),
        RejectReason::QtyNotFinite => REJ_QTY.fetch_add(1, Ordering::Relaxed),
        RejectReason::BadEventTime => REJ_TIME.fetch_add(1, Ordering::Relaxed),
    };
}

/// Snapshot de rechazos para telemetría (F6).
pub fn reject_snapshot() -> [(RejectReason, u64); 6] {
    [
        (
            RejectReason::PriceNotFinite,
            REJ_PRICE_FINITE.load(Ordering::Relaxed),
        ),
        (
            RejectReason::PriceNotPositive,
            REJ_PRICE_POSITIVE.load(Ordering::Relaxed),
        ),
        (
            RejectReason::CrossedBook,
            REJ_CROSSED.load(Ordering::Relaxed),
        ),
        (
            RejectReason::AbsurdSpread,
            REJ_SPREAD.load(Ordering::Relaxed),
        ),
        (RejectReason::QtyNotFinite, REJ_QTY.load(Ordering::Relaxed)),
        (RejectReason::BadEventTime, REJ_TIME.load(Ordering::Relaxed)),
    ]
}

/// Valida un BookTicker. Ok(()) = dato apto para el motor.
#[inline(always)]
pub fn validate_book_ticker(e: &BookTickerEvent) -> Result<(), RejectReason> {
    // 1) Finitud y positividad de precios
    if !e.bid_price.is_finite() || !e.ask_price.is_finite() {
        return Err(RejectReason::PriceNotFinite);
    }
    if e.bid_price <= 0.0 || e.ask_price <= 0.0 {
        return Err(RejectReason::PriceNotPositive);
    }
    // 2) Libro cruzado
    if e.bid_price > e.ask_price {
        return Err(RejectReason::CrossedBook);
    }
    // 3) Spread absurdo
    // Positive ordered prices imply r in [0,1]. Algebraically equivalent to
    // (ask-bid)/mid, without overflowing the sum of two finite prices.
    let ratio = e.bid_price / e.ask_price;
    let relative_spread = 2.0 * (1.0 - ratio) / (1.0 + ratio);
    if relative_spread > MAX_RELATIVE_SPREAD {
        return Err(RejectReason::AbsurdSpread);
    }
    // 4) Cantidades
    if !e.bid_qty.is_finite() || !e.ask_qty.is_finite() || e.bid_qty < 0.0 || e.ask_qty < 0.0 {
        return Err(RejectReason::QtyNotFinite);
    }
    // 5) Timestamp
    if e.event_time == 0 {
        return Err(RejectReason::BadEventTime);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(bid: f64, ask: f64, bq: f64, aq: f64) -> BookTickerEvent {
        BookTickerEvent {
            coin_id: 0,
            event_time: 1_700_000_000_000,
            bid_price: bid,
            bid_qty: bq,
            ask_price: ask,
            ask_qty: aq,
        }
    }

    #[test]
    fn accepts_sane_book() {
        assert!(validate_book_ticker(&ev(99.9, 100.1, 5.0, 3.0)).is_ok());
    }

    #[test]
    fn rejects_nan_and_zero_prices() {
        assert_eq!(
            validate_book_ticker(&ev(f64::NAN, 100.0, 1.0, 1.0)),
            Err(RejectReason::PriceNotFinite)
        );
        assert_eq!(
            validate_book_ticker(&ev(0.0, 100.0, 1.0, 1.0)),
            Err(RejectReason::PriceNotPositive)
        );
    }

    #[test]
    fn rejects_crossed_book_and_absurd_spread() {
        assert_eq!(
            validate_book_ticker(&ev(101.0, 100.0, 1.0, 1.0)),
            Err(RejectReason::CrossedBook)
        );
        // spread ~67%: bid 50 / ask 100 (mid 75, spread 50/75)
        assert_eq!(
            validate_book_ticker(&ev(50.0, 100.0, 1.0, 1.0)),
            Err(RejectReason::AbsurdSpread)
        );
    }

    #[test]
    fn rejects_bad_qty_and_time() {
        assert_eq!(
            validate_book_ticker(&ev(99.0, 101.0, f64::NAN, 1.0)),
            Err(RejectReason::QtyNotFinite)
        );
        let mut e = ev(99.0, 101.0, 1.0, 1.0);
        e.event_time = 0;
        assert_eq!(validate_book_ticker(&e), Err(RejectReason::BadEventTime));
    }

    #[test]
    fn test_reject_snapshot_and_counter_increment() {
        count_reject(RejectReason::PriceNotFinite);
        count_reject(RejectReason::CrossedBook);

        let snap = reject_snapshot();
        let finite_count = snap
            .iter()
            .find(|(r, _)| *r == RejectReason::PriceNotFinite)
            .unwrap()
            .1;
        let crossed_count = snap
            .iter()
            .find(|(r, _)| *r == RejectReason::CrossedBook)
            .unwrap()
            .1;

        assert!(finite_count >= 1);
        assert!(crossed_count >= 1);
    }
}
