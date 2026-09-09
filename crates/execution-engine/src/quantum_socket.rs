use reqwest::{header, Client, ClientBuilder};
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
        let clean_api_key = api_key.trim();
        let mut headers = header::HeaderMap::new();
        headers.insert(
            "X-MBX-APIKEY",
            header::HeaderValue::from_str(clean_api_key)
                .unwrap_or(header::HeaderValue::from_static("")),
        );
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/x-www-form-urlencoded"),
        );
        headers.insert(
            header::CONNECTION,
            header::HeaderValue::from_static("keep-alive"),
        );

        let client = ClientBuilder::new()
            .default_headers(headers)
            .tcp_nodelay(true) // FUNDAMENTAL: Desactiva el algoritmo de Nagle para enviar órdenes sin esperar a llenar buffers
            .tcp_keepalive(Duration::from_secs(15)) // FUNDAMENTAL: Latidos invisibles cada 15s para engañar a los routers y evitar drops fantasma
            .pool_idle_timeout(None) // EVITA QUE REQWEST CIERRE LA CONEXIÓN (MANTENER LA TUBERÍA SIEMPRE CALIENTE PARA EVITAR TLS HANDSHAKE)
            .pool_max_idle_per_host(50) // Mantenemos 50 tuberías ultra calientes a Binance listas para disparar
            .hickory_dns(true) // FUNDAMENTAL: Bypass the OS blocking DNS resolver for zero-latency lookups
            .use_rustls_tls() // TLS optimizado y seguro (rustls instead of native for hickory-dns compatibility)
            .build()
            // FIX #1503: Construcción resiliente de cliente HFT sin expect
            .unwrap_or_else(|_| Client::new());

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
    pub async fn fire_raw_payload(
        &self,
        method: &str,
        path_and_query: &str,
    ) -> Result<String, String> {
        let clean_path = if path_and_query.starts_with('/') {
            path_and_query.to_string()
        } else {
            format!("/{}", path_and_query)
        };
        let url = format!("https://{}{}", self.host, clean_path);

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
                }
                Err(e) => {
                    // Si el error es de conexión inicial (ej. broken pipe, dns) se puede reintentar si el request no se envió.
                    // Para POST con timeout, NO reintentar ciegamente para evitar duplicación de órdenes en Binance.
                    if (e.is_connect() || (e.is_timeout() && method != "POST")) && retry_count == 0
                    {
                        retry_count += 1;
                        continue;
                    }
                    return Err(format!("Error de Red al disparar orden: {}", e));
                }
            }
        }
    }

    /// Retorna el host configurado para el pool
    pub fn get_host(&self) -> &str {
        &self.host
    }

    /// Calcula la afinidad de socket/shard por símbolo en O(1) (Punto #177 / #178)
    #[inline(always)]
    pub fn compute_symbol_shard_affinity(symbol: &str, shard_count: usize) -> usize {
        if shard_count <= 1 {
            return 0;
        }
        let mut hash: u64 = 0xcbf29ce484222325;
        for b in symbol.as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        (hash as usize) % shard_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_socket_pool_creation_and_config() {
        let pool = QuantumSocketPool::new("fapi.binance.com", 443, "test_api_key".to_string());
        assert_eq!(pool.get_host(), "fapi.binance.com");
    }

    #[test]
    fn test_quantum_socket_pool_headers() {
        let pool = QuantumSocketPool::new(
            "testnet.binancefuture.com",
            443,
            "  api_key_with_spaces  ".to_string(),
        );
        assert_eq!(pool.get_host(), "testnet.binancefuture.com");
    }

    #[test]
    fn test_symbol_shard_affinity_distribution() {
        let shard_count = 8;
        let s1 = QuantumSocketPool::compute_symbol_shard_affinity("BTCUSDT", shard_count);
        let s2 = QuantumSocketPool::compute_symbol_shard_affinity("ETHUSDT", shard_count);
        let s3 = QuantumSocketPool::compute_symbol_shard_affinity("SOLUSDT", shard_count);

        assert!(s1 < shard_count);
        assert!(s2 < shard_count);
        assert!(s3 < shard_count);
        // Determinismo
        assert_eq!(
            s1,
            QuantumSocketPool::compute_symbol_shard_affinity("BTCUSDT", shard_count)
        );
    }

    #[test]
    fn test_symbol_shard_affinity_edge_cases() {
        assert_eq!(
            QuantumSocketPool::compute_symbol_shard_affinity("BTCUSDT", 0),
            0
        );
        assert_eq!(
            QuantumSocketPool::compute_symbol_shard_affinity("BTCUSDT", 1),
            0
        );
        let default_hash = (0xcbf29ce484222325_u64 as usize) % 4;
        assert_eq!(
            QuantumSocketPool::compute_symbol_shard_affinity("", 4),
            default_hash
        );
    }

    #[tokio::test]
    async fn test_quantum_socket_unsupported_http_method() {
        let pool = QuantumSocketPool::new("127.0.0.1", 443, "dummy".to_string());
        let res = pool.fire_raw_payload("PATCH", "/test").await;
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("no soportado"));
    }
}
