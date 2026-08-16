use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::extract::State;
use serde::Serialize;
use tokio::sync::broadcast;
use std::sync::Arc;
use crate::lockfree_bus::LockFreeBus;
use crate::AppState;

#[derive(Copy, Clone, Default, Serialize)]
#[repr(C)]
pub struct StreamEvent {
    pub timestamp: u64,
    pub coin_id: usize,
    pub ml_prob: f32,
    pub z_score: f32,
    pub cvd: f32,
    pub ofi: f32,
    pub pnl_realized_scalp: f32,
    pub pnl_unrealized_scalp: f32,
}

// Zero-latency lock-free bus for passing events from GodEngine to the Telemetry background thread
lazy_static::lazy_static! {
    pub static ref TICK_STREAM_BUS: LockFreeBus<StreamEvent> = LockFreeBus::new();
}

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let tx = state.stream_tx.clone();
    ws.on_upgrade(move |socket| handle_socket(socket, tx))
}

async fn handle_socket(mut socket: WebSocket, tx: broadcast::Sender<String>) {
    let mut rx = tx.subscribe();
    
    // Send a welcome message
    let _ = socket.send(Message::Text(r#"{"status": "connected", "message": "Trader Gemini Realtime Feed"}"#.to_string())).await;
    
    while let Ok(msg) = rx.recv().await {
        if socket.send(Message::Text(msg)).await.is_err() {
            // Client disconnected
            break;
        }
    }
}
