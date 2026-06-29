use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::fs;
use tokio::sync::broadcast;
use serde::Serialize;

#[derive(Clone, Serialize, Debug)]
pub enum TelemetryEvent {
    LatencyUpdate(u64),      // Nanoseconds
    LogUpdate(String, String), // (type, message) e.g., ("info", "Connected...")
    CapitalUpdate(f64),      // Current capital
    TensorUpdate([f32; 10]), // 10D State Vector
    OmniUpdate {
        latency_ms: u64,
        latency_panic: bool,
        dark_alpha: f64,
        scalp_pnl: f64,
        swing_pnl: f64,
    },
}

pub async fn start_server(tx: broadcast::Sender<TelemetryEvent>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let port = 8080;
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("🌐 [DASHBOARD] Embedded Rust Dashboard running on http://localhost:{}", port);

    loop {
        let (mut socket, _) = listener.accept().await?;
        let tx = tx.clone();
        
        tokio::spawn(async move {
            let mut buf = [0; 1024];
            if let Ok(n) = socket.read(&mut buf).await {
                if n == 0 { return; }
                let request = String::from_utf8_lossy(&buf[..n]);
                
                if request.starts_with("GET /api/stats") {
                    let config_str = fs::read_to_string("data/dynamic_config.json").unwrap_or_else(|_| "{}".to_string());
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                        config_str
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                } else if request.starts_with("GET /events") {
                    let response = "HTTP/1.1 200 OK\r\n\
                                    Content-Type: text/event-stream\r\n\
                                    Cache-Control: no-cache\r\n\
                                    Connection: keep-alive\r\n\
                                    Access-Control-Allow-Origin: *\r\n\r\n";
                    if socket.write_all(response.as_bytes()).await.is_err() {
                        return;
                    }
                    
                    let mut rx = tx.subscribe();
                    loop {
                        if let Ok(event) = rx.recv().await {
                            if let Ok(json) = serde_json::to_string(&event) {
                                let sse_msg = format!("data: {}\n\n", json);
                                if socket.write_all(sse_msg.as_bytes()).await.is_err() {
                                    break; // Client disconnected
                                }
                            }
                        }
                    }
                } else if request.starts_with("GET /style.css") {
                    let css = fs::read_to_string("static/style.css").unwrap_or_else(|_| "".to_string());
                    let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/css\r\n\r\n{}", css);
                    let _ = socket.write_all(response.as_bytes()).await;
                } else if request.starts_with("GET /app.js") {
                    let js = fs::read_to_string("static/app.js").unwrap_or_else(|_| "".to_string());
                    let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/javascript\r\n\r\n{}", js);
                    let _ = socket.write_all(response.as_bytes()).await;
                } else {
                    let html = fs::read_to_string("static/index.html").unwrap_or_else(|_| "<h1>Error: static/index.html not found</h1>".to_string());
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n{}",
                        html
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                }
            }
        });
    }
}
