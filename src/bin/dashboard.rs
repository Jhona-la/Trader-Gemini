use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = 8080;
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("🌐 [DASHBOARD] Rust Native Dashboard running on http://localhost:{}", port);

    loop {
        let (mut socket, _) = listener.accept().await?;
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
