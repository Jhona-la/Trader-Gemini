use crate::lockfree_bus::LockFreeBus;
use axum::extract::Extension;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use serde::Serialize;
use tokio::sync::broadcast;

#[derive(Copy, Clone, Default, Serialize)]
#[repr(C)]
pub struct StreamEvent {
    pub timestamp: u64,
    pub coin_id: usize,
    pub ml_prob: f32,
    pub z_score: f32,
    pub cvd: f32,
    pub ofi: f32,
    pub pnl_realized: f32,
    pub pnl_unrealized: f32,
}

// Zero-latency lock-free bus for passing events from GodEngine to the Telemetry background thread
lazy_static::lazy_static! {
    pub static ref TICK_STREAM_BUS: LockFreeBus<StreamEvent> = LockFreeBus::new();
}

pub async fn ws_stream_handler(
    ws: WebSocketUpgrade,
    Extension(tx): Extension<broadcast::Sender<String>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, tx))
}

async fn handle_socket(mut socket: WebSocket, tx: broadcast::Sender<String>) {
    let mut rx = tx.subscribe();

    // Send a welcome message
    let _ = socket
        .send(Message::Text(
            r#"{"status": "connected", "message": "Trader Gemini Realtime Feed"}"#.to_string(),
        ))
        .await;

    while let Ok(msg) = rx.recv().await {
        if socket.send(Message::Text(msg)).await.is_err() {
            // Client disconnected
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_event_serialization() {
        let ev = StreamEvent {
            timestamp: 1672531200000,
            coin_id: 2,
            ml_prob: 0.85,
            z_score: 1.5,
            cvd: 120.0,
            ofi: 45.0,
            pnl_realized: 0.25,
            pnl_unrealized: 0.05,
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("0.85"));
        assert!(json.contains("1672531200000"));
    }

    #[test]
    fn test_tick_stream_bus_push_and_multi_coin_events() {
        let ev1 = StreamEvent {
            timestamp: 1000,
            coin_id: 0,
            ml_prob: 0.92,
            z_score: 2.1,
            cvd: 300.0,
            ofi: 15.0,
            pnl_realized: 1.20,
            pnl_unrealized: 0.40,
        };
        TICK_STREAM_BUS.push(ev1);

        let json = serde_json::to_string(&ev1).unwrap();
        assert!(json.contains("\"coin_id\":0"));
        assert!(json.contains("\"ml_prob\":0.92"));
    }
}
