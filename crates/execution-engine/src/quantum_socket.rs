use reqwest::{Client, ClientBuilder, header};
use std::time::Duration;

/// Hilo conductor de latencia ultra-baja para inyección directa de órdenes
/// V3 INSTITUCIONAL: Usamos Reqwest (Hyper) inyectado con esteroides HFT.
/// Elimina los riesgos de TCP crudo (chunked encoding, keep-alive drop, GZIP drift).
#[derive(Clone)]
pub struct QuantumSocketPool {
    host: String,
    _api_key: String,
    client: Client,
}

impl QuantumSocketPool {
    pub fn new(host: &str, _port: u16, api_key: String) -> Self {
        
        let mut headers = header::HeaderMap::new();
        headers.insert(
            "X-MBX-APIKEY",
            header::HeaderValue::from_str(&api_key).unwrap_or(header::HeaderValue::from_static(""))
        );
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/x-www-form-urlencoded")
        );
        headers.insert(
            header::CONNECTION,
            header::HeaderValue::from_static("keep-alive")
        );

        let client = ClientBuilder::new()
            .default_headers(headers)
            .tcp_nodelay(true) // FUNDAMENTAL: Desactiva el algoritmo de Nagle para enviar órdenes sin esperar a llenar buffers
            .tcp_keepalive(Duration::from_secs(15)) // FUNDAMENTAL: Latidos invisibles cada 15s para engañar a los routers y evitar drops fantasma
            .pool_idle_timeout(None) // EVITA QUE REQWEST CIERRE LA CONEXIÓN (MANTENER LA TUBERÍA SIEMPRE CALIENTE PARA EVITAR TLS HANDSHAKE)
            .pool_max_idle_per_host(50) // Mantenemos 50 tuberías ultra calientes a Binance listas para disparar
            .timeout(Duration::from_secs(3)) // Reducido a 3s para fallar ultra rápido y hacer retry inmediato
            .hickory_dns(true) // FUNDAMENTAL: Bypass the OS blocking DNS resolver for zero-latency lookups
            .use_rustls_tls() // TLS optimizado y seguro (rustls instead of native for hickory-dns compatibility)
            .build()
            .expect("Fallo al construir Cliente HFT Institucional");

        Self {
            host: host.to_string(),
            _api_key: api_key,
            client,
        }
    }

    /// Pre-calienta el pool lanzando pings a Binance para levantar los sockets TCP/TLS y dejarlos en idle.
    pub async fn warmup_pool(&self, count: usize) {
        let url = format!("https://{}/fapi/v1/ping", self.host);
        let mut handles = Vec::with_capacity(count);
        for _ in 0..count {
            let client = self.client.clone();
            let url_clone = url.clone();
            handles.push(tokio::spawn(async move {
                let _ = client.get(&url_clone).send().await;
            }));
        }
        for h in handles {
            let _ = h.await;
        }
    }

    /// Dispara el payload usando hiper-conexiones recicladas
    pub async fn fire_raw_payload(&self, method: &str, path_and_query: &str) -> Result<String, String> {
        let url = format!("https://{}{}", self.host, path_and_query);
        
        let req = match method {
            "GET" => self.client.get(&url),
            "POST" => self.client.post(&url),
            "DELETE" => self.client.delete(&url),
            "PUT" => self.client.put(&url),
            _ => return Err("Método HTTP no soportado".to_string()),
        };

        // El cliente de Reqwest ya maneja retries internos de bajo nivel si la conexión dropea en el handshake,
        // pero podemos implementar un soft-retry (1 intento) por si Binance nos manda un 502 repentino o se cae
        // la red del OS antes de salir.
        
        let mut retry_count = 0;
        
        loop {
            // Clonamos el request builder (es ultra ligero)
            let req_clone = req.try_clone().ok_or("No se puede clonar request HTTP")?;
            
            match req_clone.send().await {
                Ok(response) => {
                    // Hyper se encarga de parsear el Chunked Encoding, Content-Length y descomprimir automáticamente si es necesario.
                    match response.text().await {
                        Ok(text) => return Ok(text),
                        Err(e) => {
                            if retry_count == 0 {
                                retry_count += 1;
                                continue;
                            }
                            return Err(format!("Error leyendo payload: {}", e));
                        }
                    }
                },
                Err(e) => {
                    // Si el error es de conexión (ej. broken pipe, dns, timeout) intentamos 1 vez más
                    if e.is_connect() || e.is_timeout() {
                        if retry_count == 0 {
                            retry_count += 1;
                            continue;
                        }
                    }
                    return Err(format!("Error de Red al disparar orden: {}", e));
                }
            }
        }
    }
}
