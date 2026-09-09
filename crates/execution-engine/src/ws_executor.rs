//! Ejecución de órdenes por WebSocket API — STUB EXPLÍCITO (R1.5).
//!
//! ESTADO HONESTO: la ruta WS de ejecución NO está implementada. `is_connected()`
//! devuelve siempre `false`, por lo que `OrderExecutor` cae de forma inmediata y
//! determinista a la ruta REST (QuantumSocketPool HFT), que es la única ruta
//! sancionada para tocar dinero real.
//!
//! Historial: el archivo original se perdió sin commit; una reconstrucción y una
//! fachada posterior coexistieron sin firma HMAC real — cualquier activación
//! habría producido rechazos `-1022 Signature` en cada orden. Este stub elimina
//! esa trampa de activación: para implementar la vía WS se requiere (en este
//! orden) firma HMAC-SHA256 con params en orden canónico alfabético, canal
//! ACOTADO (no unbounded), sesión con heartbeat/reconnect, y ACK/resolución de
//! ambigüedad por clientOrderId ANTES de que `send_order_payload` devuelva Ok.
//!
//! R1.5: el secreto de firma NUNCA viaja en el mensaje — la firma se materializa
//! en el writer de la sesión (cuando exista), nunca en structs encolables.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Mensaje de orden para la futura vía WS — SIN credenciales: la firma
/// HMAC se calcula en el punto de envío, no se propaga por canales.
#[derive(Debug, Clone)]
pub struct WsOrderMessage {
    pub symbol: String,
    pub side: String,
    pub position_side: Option<String>,
    pub order_type: String,
    pub quantity: f64,
    pub price: Option<f64>,
    pub time_in_force: Option<String>,
    pub reduce_only: bool,
    pub client_order_id: String,
    pub timestamp: u64,
}

pub struct WsExecutor {
    connected: Arc<AtomicBool>,
}

impl WsExecutor {
    pub fn new(_api_key: String, _api_secret: String, _is_testnet: bool) -> Self {
        // Credenciales deliberadamente NO almacenadas: un stub no necesita
        // material secreto en memoria.
        Self {
            connected: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Siempre `false` mientras la vía WS no esté implementada: garantiza
    /// el fallback REST determinista de todos los call-sites del executor.
    #[inline(always)]
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    /// Contrato preservado para `OrderExecutor`. Con el stub, todo envío
    /// devuelve Err y el caller cae a REST — nunca éxito ficticio.
    #[allow(clippy::too_many_arguments)]
    pub fn send_order_payload(
        &self,
        _api_key: &str,
        _api_secret: &str,
        _symbol: &str,
        _side: &str,
        _position_side: Option<&str>,
        _order_type: &str,
        _quantity: f64,
        _price: Option<f64>,
        _time_in_force: Option<&str>,
        _reduce_only: bool,
        _client_order_id: &str,
        _timestamp: u64,
    ) -> Result<(), String> {
        if !self.is_connected() {
            return Err("WS_STUB: vía WS no implementada — usando fallback REST".to_string());
        }
        Err("WS_STUB: conectado pero sin transporte implementado".to_string())
    }
}
