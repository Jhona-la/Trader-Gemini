pub mod binance_api;
pub mod client;
pub mod executor;
pub mod hot_swap;
pub mod order_types;
pub mod router;
pub mod simulator;

pub use order_types::{Fill, OrderAck};

#[derive(Debug, Clone)]
pub struct ExecutionPayload {
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub order_type: String,
    pub time_in_force: String,
    pub position_side: String,
    /// F1.4: query EXACTA que se firmó. El envío SIEMPRE es base_url + signed_query
    /// + "&signature=" + signature. Invariante: se envía lo que se firma — elimina
    /// la clase completa de bugs "firmar una cosa, enviar otra".
    pub signed_query: String,
    /// F1.2: identificador idempotente de la orden (UUIDv7) para deduplicación
    /// y consulta post-timeout.
    pub client_order_id: String,
    pub signature: String,
    pub timestamp: u64,
}
