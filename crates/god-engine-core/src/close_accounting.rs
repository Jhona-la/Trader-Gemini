//! Accounting of a local close proposal; this is not exchange settlement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalCloseAccounting {
    pub event_time_ms: u64,
    pub quantity: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub gross_pnl: f64,
    pub entry_fee: f64,
    pub exit_fee: f64,
}

impl LocalCloseAccounting {
    pub fn total_fees(self) -> f64 {
        self.entry_fee + self.exit_fee
    }
}
