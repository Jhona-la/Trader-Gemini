pub mod binance_api;
pub mod client;
pub mod dynamic_symbols;
pub mod executor;
pub mod hot_swap;
pub mod order_registry;
pub mod order_types;
pub mod quantum_multiplexer;
pub mod quantum_socket;
pub mod reconciliation;
pub mod router;
pub mod shadow;
pub mod simulator;
pub mod user_data_stream;

pub use dynamic_symbols::DynamicSymbolSelector;
pub use hot_swap::HotSwapController;
pub use order_registry::{OrderRegistry, OrderStatus, TrackedOrder};
pub use order_types::{Fill, OrderAck};
pub use quantum_multiplexer::QuantumMultiplexer;
pub use quantum_socket::QuantumSocketPool;
pub use reconciliation::{reconcile, PositionRiskEntry, ReconciliationReport};
pub use router::QuantumOrderRouter;
pub use shadow::ShadowExecutor;
pub use user_data_stream::{AccountSink, RemotePosition, UserDataStreamer};

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
