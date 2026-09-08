use crate::atomic_float::AtomicF64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionHorizon {
    Continuous,
    Scalping,
    Swing,
}
/// Lock-free Position tracking for the Hot Path
#[repr(C, align(64))]
pub struct Position {
    pub is_open: AtomicBool,
    pub is_long: AtomicBool,
    pub horizon: std::sync::atomic::AtomicU8, // 0 = Continuous, 1 = Scalping, 2 = Swing
    pub entry_price: AtomicF64,
    pub quantity: AtomicF64,
    pub margin_used: AtomicF64,
    pub entry_time_ms: AtomicU64,
    pub trailing_phase: std::sync::atomic::AtomicU8,
    pub mfe_atr: AtomicF64,
    pub max_pnl_pct: AtomicF64,
    pub trail_stop: AtomicF64,
    pub tp_price: AtomicF64,
    pub sl_price: AtomicF64,
    pub ml_prediction: AtomicF64,
    pub confidence: AtomicF64,
    pub entry_fee: AtomicF64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            is_open: AtomicBool::new(false),
            is_long: AtomicBool::new(true),
            horizon: std::sync::atomic::AtomicU8::new(0),
            entry_price: AtomicF64::new(0.0),
            quantity: AtomicF64::new(0.0),
            margin_used: AtomicF64::new(0.0),
            entry_time_ms: AtomicU64::new(0),
            trailing_phase: std::sync::atomic::AtomicU8::new(0),
            mfe_atr: AtomicF64::new(0.0),
            max_pnl_pct: AtomicF64::new(0.0),
            trail_stop: AtomicF64::new(0.0),
            tp_price: AtomicF64::new(0.0),
            sl_price: AtomicF64::new(0.0),
            ml_prediction: AtomicF64::new(0.0),
            confidence: AtomicF64::new(0.0),
            entry_fee: AtomicF64::new(0.0),
        }
    }
}

impl Position {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        &self,
        is_long: bool,
        price: f64,
        qty: f64,
        margin: f64,
        current_time_ms: u64,
        tp: f64,
        sl: f64,
    ) {
        self.open_with_horizon(is_long, price, qty, margin, current_time_ms, tp, sl, PositionHorizon::Continuous);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_horizon(
        &self,
        is_long: bool,
        price: f64,
        qty: f64,
        margin: f64,
        current_time_ms: u64,
        tp: f64,
        sl: f64,
        horizon: PositionHorizon,
    ) {
        self.open_with_full_meta(is_long, price, qty, margin, current_time_ms, tp, sl, horizon, 0.0, 0.0);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_full_meta(
        &self,
        is_long: bool,
        price: f64,
        qty: f64,
        margin: f64,
        current_time_ms: u64,
        tp: f64,
        sl: f64,
        horizon: PositionHorizon,
        ml_pred: f64,
        conf: f64,
    ) {
        self.open_with_fee(is_long, price, qty, margin, current_time_ms, tp, sl, horizon, ml_pred, conf, 0.0);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_fee(
        &self,
        is_long: bool,
        price: f64,
        qty: f64,
        margin: f64,
        current_time_ms: u64,
        tp: f64,
        sl: f64,
        horizon: PositionHorizon,
        ml_pred: f64,
        conf: f64,
        entry_fee: f64,
    ) {
        let safe_price = if price.is_finite() && price > 0.0 { price } else { 1.0 };
        let safe_qty = if qty.is_finite() && qty > 0.0 { qty } else { 0.0 };
        let safe_margin = if margin.is_finite() && margin >= 0.0 { margin } else { 0.0 };
        let safe_tp = if tp.is_finite() && tp >= 0.0 { tp } else { 0.0 };
        let safe_sl = if sl.is_finite() && sl >= 0.0 { sl } else { 0.0 };
        let safe_ml = if ml_pred.is_finite() { ml_pred } else { 0.5 };
        let safe_conf = if conf.is_finite() { conf.clamp(0.0, 1.0) } else { 0.5 };
        let safe_fee = if entry_fee.is_finite() && entry_fee >= 0.0 { entry_fee } else { 0.0 };

        self.is_long.store(is_long, Ordering::Relaxed);
        // U-1 — encoding fiel del continuo: Continuous ocupa SU PROPIO slot
        // (2). Antes colisionaba con Swing (=1): una posición continua era
        // indistinguible de un swing al leer (la auditoría T-08/K-17).
        let h_val = match horizon {
            PositionHorizon::Scalping => 0,
            PositionHorizon::Swing => 1,
            PositionHorizon::Continuous => 2,
        };
        self.horizon.store(h_val, Ordering::Relaxed);
        self.entry_price.store(safe_price, Ordering::Relaxed);
        self.quantity.store(safe_qty, Ordering::Relaxed);
        self.margin_used.store(safe_margin, Ordering::Relaxed);
        self.entry_time_ms.store(current_time_ms, Ordering::Relaxed);
        self.trailing_phase.store(0, Ordering::Relaxed);
        self.mfe_atr.store(0.0, Ordering::Relaxed);
        self.max_pnl_pct.store(0.0, Ordering::Relaxed);
        self.trail_stop.store(0.0, Ordering::Relaxed);
        self.tp_price.store(safe_tp, Ordering::Relaxed);
        self.sl_price.store(safe_sl, Ordering::Relaxed);
        self.ml_prediction.store(safe_ml, Ordering::Relaxed);
        self.confidence.store(safe_conf, Ordering::Relaxed);
        self.entry_fee.store(safe_fee, Ordering::Relaxed);
        self.is_open.store(true, Ordering::Release);
    }

    #[inline(always)]
    pub fn horizon(&self) -> PositionHorizon {
        match self.horizon.load(Ordering::Acquire) {
            1 => PositionHorizon::Swing,
            2 => PositionHorizon::Continuous,
            _ => PositionHorizon::Scalping,
        }
    }

    pub fn close(&self) -> (bool, f64, f64, f64) {
        let (is_long, price, qty, margin, _fee) = self.close_with_fee();
        (is_long, price, qty, margin)
    }

    /// Closes position and returns (is_long, price, qty, margin, entry_fee)
    /// FIX #902: is_open set to false FIRST with Release to prevent torn reads.
    /// A concurrent reader checking is_open(Acquire) will see either:
    ///   (a) is_open=true with all valid fields (pre-close snapshot), or
    ///   (b) is_open=false (post-close, fields may be zeroed — reader skips).
    pub fn close_with_fee(&self) -> (bool, f64, f64, f64, f64) {
        // FIX #902 & #1201: Compare-and-swap atómico para garantizar que solo un hilo cierra la posición.
        // Si is_open ya era false, evita sobreescribir con ceros una nueva posición que se esté abriendo concurrentemente.
        if self.is_open.compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return (false, 0.0, 0.0, 0.0, 0.0);
        }

        // Step 2: Read all field values BEFORE clearing (order matters for correctness)
        let is_long = self.is_long.load(Ordering::Relaxed);
        let price = self.entry_price.load(Ordering::Relaxed);
        let qty = self.quantity.load(Ordering::Relaxed);
        let margin = self.margin_used.load(Ordering::Relaxed);
        let fee = self.entry_fee.load(Ordering::Relaxed);

        // Step 3: Clear all fields (Relaxed is fine — is_open=false already published via CAS)
        self.entry_price.store(0.0, Ordering::Relaxed);
        self.quantity.store(0.0, Ordering::Relaxed);
        self.margin_used.store(0.0, Ordering::Relaxed);
        self.entry_fee.store(0.0, Ordering::Relaxed);
        self.entry_time_ms.store(0, Ordering::Relaxed);
        self.trailing_phase.store(0, Ordering::Relaxed);
        self.mfe_atr.store(0.0, Ordering::Relaxed);
        self.max_pnl_pct.store(0.0, Ordering::Relaxed);
        self.trail_stop.store(0.0, Ordering::Relaxed);
        self.tp_price.store(0.0, Ordering::Relaxed);
        self.sl_price.store(0.0, Ordering::Relaxed);
        self.ml_prediction.store(0.0, Ordering::Relaxed);
        self.confidence.store(0.0, Ordering::Relaxed);

        (is_long, price, qty, margin, fee)
    }

    #[inline(always)]
    pub fn is_open(&self) -> bool {
        self.is_open.load(Ordering::Acquire)
    }
}

#[repr(C, align(64))]
#[derive(Default)]
pub struct PositionManager {
    pub scalp: Position,
    pub swing: Position,
    pub position: Position,
}

impl PositionManager {
    #[inline(always)]
    pub fn is_any_open(&self) -> bool {
        self.scalp.is_open() || self.swing.is_open() || self.position.is_open()
    }

    #[inline(always)]
    pub fn is_scalp_open(&self) -> bool {
        self.scalp.is_open()
    }

    #[inline(always)]
    pub fn is_swing_open(&self) -> bool {
        self.swing.is_open()
    }

    #[inline(always)]
    pub fn get_position(&self, horizon: PositionHorizon) -> &Position {
        match horizon {
            PositionHorizon::Scalping => &self.scalp,
            PositionHorizon::Swing => &self.swing,
            PositionHorizon::Continuous => &self.position,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_continuous_open_and_close() {
        let mgr = PositionManager::default();

        // Continuous Long
        mgr.position.open_with_horizon(
            true,
            60000.0,
            0.1,
            600.0,
            1000,
            60500.0,
            59500.0,
            PositionHorizon::Continuous,
        );

        assert!(mgr.position.is_open());
        assert!(mgr.is_any_open());
        assert_eq!(mgr.position.horizon(), PositionHorizon::Continuous);

        let (is_long, price, qty, _) = mgr.position.close();
        assert!(is_long);
        assert_eq!(price, 60000.0);
        assert_eq!(qty, 0.1);
        assert!(!mgr.position.is_open());
        assert!(!mgr.is_any_open());
    }

    #[test]
    fn test_position_atomic_close_idempotency() {
        let pos = Position::default();
        pos.open_with_fee(
            true, 50000.0, 1.0, 5000.0, 1000, 51000.0, 49000.0,
            PositionHorizon::Continuous, 0.8, 0.9, 2.5,
        );

        let (is_long, p, q, m, f) = pos.close_with_fee();
        assert!(is_long);
        assert_eq!(p, 50000.0);
        assert_eq!(qty_or_eq(q, 1.0), true);
        assert_eq!(m, 5000.0);
        assert_eq!(f, 2.5);

        // Segundo cierre debe retornar zeros (idempotente)
        let (is_long2, p2, _, _, _) = pos.close_with_fee();
        assert!(!is_long2);
        assert_eq!(p2, 0.0);
    }

    #[test]
    fn test_position_dual_scalp_swing_independence() {
        let mgr = PositionManager::default();

        // 1. Open Swing Long
        mgr.swing.open_with_horizon(
            true, 90000.0, 0.1, 900.0, 1000, 91500.0, 89300.0, PositionHorizon::Swing,
        );
        assert!(mgr.is_swing_open());
        assert!(!mgr.is_scalp_open());
        assert!(mgr.is_any_open());

        // 2. Open Scalp Short simultaneously without interfering
        mgr.scalp.open_with_horizon(
            false, 90200.0, 0.05, 450.0, 1050, 89900.0, 90350.0, PositionHorizon::Scalping,
        );
        assert!(mgr.is_swing_open());
        assert!(mgr.is_scalp_open());
        assert!(mgr.is_any_open());

        // 3. Scalp exits on TP
        let (is_long_sc, p_sc, q_sc, _) = mgr.scalp.close();
        assert!(!is_long_sc);
        assert_eq!(p_sc, 90200.0);
        assert_eq!(q_sc, 0.05);
        assert!(!mgr.is_scalp_open());
        // Swing remains OPEN!
        assert!(mgr.is_swing_open());

        // 4. Swing exits
        let (is_long_sw, p_sw, q_sw, _) = mgr.swing.close();
        assert!(is_long_sw);
        assert_eq!(p_sw, 90000.0);
        assert_eq!(q_sw, 0.1);
        assert!(!mgr.is_swing_open());
        assert!(!mgr.is_any_open());
    }

    fn qty_or_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }
}
