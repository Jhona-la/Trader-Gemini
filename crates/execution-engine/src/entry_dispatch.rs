//! Entry dispatch shared by the host and offline transport-contract tests.
//! It preserves named units and requires leverage confirmation before submission.
//! This is NOT a complete monetary-risk or exchange-filter projection.
use crate::{
    binance_api,
    executor::{ExecutionProvider, OrderExecutor},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntryRoute {
    Market,
    Maker { price: f64 },
    Iceberg { price: f64, visible_quantity: f64 },
}

#[derive(Debug, Clone, Copy)]
pub struct EntryRequest<'a> {
    pub symbol: &'a str,
    pub is_long: bool,
    /// Base-asset units, not quote currency or margin.
    pub quantity: f64,
    pub step_size: f64,
    pub tick_size: f64,
    pub leverage: u32,
    pub client_order_id: &'a str,
    pub route: EntryRoute,
}

impl EntryRequest<'_> {
    fn validate(&self, native_iceberg: bool) -> Result<(), String> {
        if self.symbol.is_empty()
            || !self.quantity.is_finite()
            || self.quantity <= 0.0
            || !self.step_size.is_finite()
            || self.step_size <= 0.0
            || !(1.0 / self.step_size).is_finite()
            || !(self.quantity * (1.0 / self.step_size)).is_finite()
            || !(1..=binance_api::MAX_INITIAL_LEVERAGE).contains(&self.leverage)
        {
            return Err("ENTRY_INVALID_DOMAIN: symbol, quantity, step or leverage".into());
        }
        let id = self.client_order_id;
        if id.is_empty()
            || id.len() > 36
            || !id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b':' | b'/' | b'_' | b'-'))
        {
            return Err("ENTRY_INVALID_ID: expected 1..36 permitted ASCII characters".into());
        }
        let price = match self.route {
            EntryRoute::Market => return Ok(()),
            EntryRoute::Maker { price } => price,
            EntryRoute::Iceberg {
                price,
                visible_quantity,
            } => {
                if !native_iceberg {
                    return Err(
                        "UNSUPPORTED_NATIVE_ICEBERG: selected adapter has no documented capability"
                            .into(),
                    );
                }
                if !visible_quantity.is_finite()
                    || visible_quantity <= 0.0
                    || visible_quantity > self.quantity
                {
                    return Err("ENTRY_INVALID_VISIBLE_QUANTITY".into());
                }
                price
            }
        };
        if !price.is_finite()
            || price <= 0.0
            || !self.tick_size.is_finite()
            || self.tick_size <= 0.0
        {
            return Err("ENTRY_INVALID_LIMIT_PRICE_OR_TICK".into());
        }
        Ok(())
    }
}

/// A narrow, statically dispatched seam; mocks need not implement protective exits.
#[allow(async_fn_in_trait)]
pub trait EntryTransport {
    fn supports_native_iceberg(&self) -> bool;
    async fn configure_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String>;
    async fn submit_entry(&self, request: &EntryRequest<'_>) -> Result<(), String>;
}

/// The caller owns reconciliation/rollback. Any error here prevents later stages;
/// configuration uncertainty is NOT treated as proof that account leverage is unchanged.
pub async fn dispatch_entry<T: EntryTransport>(
    transport: &T,
    request: &EntryRequest<'_>,
) -> Result<(), String> {
    request.validate(transport.supports_native_iceberg())?;
    transport
        .configure_leverage(request.symbol, request.leverage)
        .await
        .map_err(|e| format!("ENTRY_LEVERAGE_UNCONFIRMED: {e}"))?;
    transport.submit_entry(request).await
}

/// One identity per new intention. Callers must retain it for any retry/reconcile.
pub fn new_entry_client_id(is_long: bool) -> String {
    format!(
        "c{}_{}",
        if is_long { "L" } else { "S" },
        uuid::Uuid::now_v7().simple()
    )
}

impl EntryTransport for OrderExecutor {
    fn supports_native_iceberg(&self) -> bool {
        binance_api::SUPPORTS_NATIVE_ICEBERG
    }

    async fn configure_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String> {
        self.set_leverage(symbol, leverage).await
    }

    async fn submit_entry(&self, request: &EntryRequest<'_>) -> Result<(), String> {
        match request.route {
            EntryRoute::Market => {
                self.execute_raw_qty_with_client_id(
                    request.symbol,
                    request.is_long,
                    request.quantity,
                    request.step_size,
                    request.client_order_id,
                )
                .await
            }
            EntryRoute::Maker { price } => {
                self.execute_maker_chase(
                    request.symbol,
                    request.is_long,
                    request.quantity,
                    price,
                    request.step_size,
                    request.tick_size,
                    request.client_order_id,
                )
                .await
            }
            EntryRoute::Iceberg {
                price,
                visible_quantity,
            } => {
                self.execute_iceberg_limit(
                    request.symbol,
                    request.is_long,
                    request.quantity,
                    visible_quantity,
                    price,
                    request.step_size,
                    request.tick_size,
                    request.client_order_id,
                )
                .await
            }
        }
    }
}
