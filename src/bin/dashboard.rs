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
                    let html = r#"<!DOCTYPE html>
<html>
<head>
    <title>Trader Gemini Dashboard</title>
    <style>
        body { background: #0a0a0a; color: #00ffcc; font-family: 'Courier New', monospace; padding: 20px; }
        .card { background: #111; padding: 20px; border: 1px solid #333; margin-bottom: 20px; border-radius: 8px;}
        h1 { text-shadow: 0 0 10px #00ffcc; }
        .val { color: #fff; font-weight: bold; }
    </style>
    <script>
        async function fetchStats() {
            try {
                let res = await fetch('/api/stats');
                let data = await res.json();
                document.getElementById('config').innerText = JSON.stringify(data, null, 2);
            } catch (e) {
                console.error(e);
            }
        }
        setInterval(fetchStats, 2000);
        window.onload = fetchStats;
    </script>
</head>
<body>
    <h1>🚀 God Engine - Quantum Dashboard</h1>
    <div class="card">
        <h2>Live Dynamic Config</h2>
        <pre id="config" class="val">Loading...</pre>
    </div>
</body>
</html>"#;
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
