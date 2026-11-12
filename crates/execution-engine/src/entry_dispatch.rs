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
    Ioc { price: f64 },
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
            EntryRoute::Maker { price } | EntryRoute::Ioc { price } => price,
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

/// LG-10 (Línea G) — UNA RUTA QUE EL ADAPTADOR NO SOPORTA NO SE ELIGE.
///
/// El host enrutaba a `EntryRoute::Iceberg` toda entrada con nocional mayor
/// que el gen `iceberg_volume_threshold`, y `validate` la rechaza siempre que
/// el adaptador no declare iceberg nativo (Binance: `SUPPORTS_NATIVE_ICEBERG =
/// false`). Era un techo oculto al crecimiento: con el genoma de prod, ninguna
/// entrada por encima de 10 000 USD de nocional; con genomas aleatorios
/// (banda 100–2 000 USD), mucho antes. Ahora la ruta Iceberg sólo se elige con
/// capacidad nativa; sin ella la orden sigue por la ruta ordinaria.
pub fn admite_iceberg(soporta_iceberg_nativo: bool, nocional: f64, umbral: f64) -> bool {
    soporta_iceberg_nativo && nocional.is_finite() && umbral.is_finite() && nocional > umbral
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
            EntryRoute::Ioc { price } => {
                self.execute_ioc_order(
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

/// Resolve the original order without submitting a replacement. Active or
/// inconclusive evidence remains pending; only a terminal order carries a fill.
pub async fn resolve_terminal_entry(
    executor: &OrderExecutor,
    symbol: &str,
    client_order_id: &str,
    dispatch_error: &str,
) -> crate::ioc_evidence::DestinoReserva {
    let observation = executor.resolve_via_rest(symbol, client_order_id).await;
    let order = executor.registry().get(client_order_id);
    crate::ioc_evidence::destino_tras_consulta(dispatch_error, observation, order.as_ref())
}

pub const ENTRY_RECOVERY_DEADLINE: std::time::Duration = std::time::Duration::from_secs(8);

pub async fn resolve_terminal_entry_until(
    executor: &OrderExecutor,
    symbol: &str,
    client_order_id: &str,
    dispatch_error: &str,
    deadline: std::time::Duration,
) -> crate::ioc_evidence::DestinoReserva {
    let recovery = async {
        loop {
            let destination =
                resolve_terminal_entry(executor, symbol, client_order_id, dispatch_error).await;
            if !matches!(destination, crate::ioc_evidence::DestinoReserva::Conservar)
                || dispatch_error.starts_with("MAKER_CHASE_UNVERIFIED")
            {
                return destination;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
        }
    };
    match tokio::time::timeout_at(tokio::time::Instant::now() + deadline, recovery).await {
        Ok(destination) => destination,
        Err(_) => crate::ioc_evidence::DestinoReserva::CancelarPorDeadline,
    }
}

#[cfg(test)]
mod f28_terminal_recovery_tests {
    use super::*;
    use crate::ioc_evidence::DestinoReserva;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn active_partial_stays_pending_and_expired_partial_confirms_without_resubmission() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback listener");
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let server = tokio::spawn(async move {
            for status in ["PARTIALLY_FILLED", "EXPIRED"] {
                let (mut socket, _) = listener.accept().await.expect("accept query");
                let mut buffer = [0u8; 4096];
                let read = socket.read(&mut buffer).await.expect("read query");
                let request = String::from_utf8_lossy(&buffer[..read]);
                recorded
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(request.lines().next().unwrap_or("").to_string());
                let body = format!(
                    "{{\"symbol\":\"BTCUSDT\",\"clientOrderId\":\"f28_rest_partial\",\"orderId\":28028,\"side\":\"BUY\",\"status\":\"{status}\",\"origQty\":\"1.0\",\"executedQty\":\"0.4\",\"avgPrice\":\"100.0\",\"price\":\"100.0\"}}"
                );
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("write query");
                socket.shutdown().await.expect("close query");
            }
        });
        let executor = OrderExecutor::new(String::new(), "offline-contract".into(), false)
            .con_base_de_prueba(&base);
        executor.registry().register_intent(
            "f28_rest_partial",
            "BTCUSDT",
            "BUY",
            "LONG",
            "LIMIT",
            1.0,
            1_000,
        );

        let terminal = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            resolve_terminal_entry_until(
                &executor,
                "BTCUSDT",
                "f28_rest_partial",
                "AMBIGUOUS: contract",
                std::time::Duration::from_secs(5),
            ),
        )
        .await
        .expect("terminal recovery must not stall");
        server.await.unwrap();
        assert_eq!(terminal, DestinoReserva::Confirmar { ejecutada: 0.4 });
        let requests = requests.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(requests.len(), 2);
        for request in requests.iter() {
            assert!(request.starts_with("GET /fapi/v1/order?"), "{request}");
            assert!(
                request.contains("origClientOrderId=f28_rest_partial"),
                "{request}"
            );
        }
        assert_eq!(
            executor
                .registry()
                .get("f28_rest_partial")
                .map(|order| order.executed_qty),
            Some(0.4)
        );
    }
}

#[cfg(test)]
mod f28_deadline_recovery_tests {
    use super::*;
    use crate::ioc_evidence::DestinoReserva;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn persistent_partial_hits_deadline_without_confirming_or_resubmitting() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("loopback listener");
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let server = tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.expect("accept query");
                let mut buffer = [0u8; 4096];
                let read = socket.read(&mut buffer).await.expect("read query");
                let request = String::from_utf8_lossy(&buffer[..read]);
                recorded
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(request.lines().next().unwrap_or("").to_string());
                let body = "{\"symbol\":\"BTCUSDT\",\"clientOrderId\":\"f28_rest_deadline\",\"orderId\":28029,\"side\":\"BUY\",\"status\":\"PARTIALLY_FILLED\",\"origQty\":\"1.0\",\"executedQty\":\"0.4\",\"avgPrice\":\"100.0\",\"price\":\"100.0\"}";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("write query");
                socket.shutdown().await.expect("close query");
            }
        });
        let executor = OrderExecutor::new(String::new(), "offline-contract".into(), false)
            .con_base_de_prueba(&base);
        executor.registry().register_intent(
            "f28_rest_deadline",
            "BTCUSDT",
            "BUY",
            "LONG",
            "LIMIT",
            1.0,
            1_000,
        );

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            resolve_terminal_entry_until(
                &executor,
                "BTCUSDT",
                "f28_rest_deadline",
                "AMBIGUOUS: contract",
                std::time::Duration::from_secs(1),
            ),
        )
        .await
        .expect("the production deadline must fire");
        server.abort();
        assert_eq!(outcome, DestinoReserva::CancelarPorDeadline);
        let requests = requests.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!requests.is_empty());
        for request in requests.iter() {
            assert!(request.starts_with("GET /fapi/v1/order?"), "{request}");
            assert!(
                request.contains("origClientOrderId=f28_rest_deadline"),
                "{request}"
            );
        }
        let order = executor.registry().get("f28_rest_deadline").unwrap();
        assert_eq!(
            order.status,
            crate::order_registry::OrderStatus::PartiallyFilled
        );
        assert_eq!(order.executed_qty, 0.4);
    }
}
