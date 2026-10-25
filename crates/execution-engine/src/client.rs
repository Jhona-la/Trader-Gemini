use arc_swap::ArcSwap;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Client, ClientBuilder};
use std::time::Duration;

/// Buffer stack para cero-alocación en el hot-path.
/// F1.2: 1024 bytes (un clientOrderId UUIDv7 + firma excedían el margen del
/// original de 512) y detección de overflow: jamás pánico ni truncamiento
/// silencioso — el caller consulta `is_overflow()` antes de firmar/enviar.
pub struct ZeroAllocBuffer {
    bytes: [u8; 1024],
    len: usize,
    overflow: bool,
}

impl ZeroAllocBuffer {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            bytes: [0; 1024],
            len: 0,
            overflow: false,
        }
    }

    #[inline(always)]
    pub fn push_str(&mut self, s: &str) {
        let b = s.as_bytes();
        if self.len + b.len() > self.bytes.len() {
            self.overflow = true;
            return;
        }
        self.bytes[self.len..self.len + b.len()].copy_from_slice(b);
        self.len += b.len();
    }

    #[inline(always)]
    pub fn is_overflow(&self) -> bool {
        self.overflow
    }

    #[inline(always)]
    pub fn push_f64(&mut self, v: f64) {
        let mut b = ryu::Buffer::new();
        self.push_str(b.format(v));
    }

    /// Formatea un valor float usando punto fijo en stack según el número de decimales (o None si no representable).
    /// Evita residuos float espurios que provocan el error Binance -1111 (precision overflow) sin forzar a entero pasos no representables.
    #[inline(always)]
    pub fn push_fixed_f64(&mut self, v: f64, decimals: Option<u8>) {
        if let Some(dec) = decimals {
            let s = crate::fixed_filter::format_decimal(v, dec);
            if !s.is_empty() {
                self.push_str(s.as_str());
                return;
            }
        }
        self.push_f64(v);
    }

    #[inline(always)]
    pub fn push_u64(&mut self, v: u64) {
        let mut b = itoa::Buffer::new();
        self.push_str(b.format(v));
    }

    #[inline(always)]
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).unwrap_or("")
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
        self.overflow = false;
    }
}

pub const BINANCE_BASE_URL: &str = "https://fapi.binance.com";
pub const BINANCE_TESTNET_URL: &str = "https://testnet.binancefuture.com";

#[derive(Debug, Default, Clone, Copy)]
pub struct BinanceRateLimits {
    pub weight_1m: Option<usize>,
    pub orders_10s: Option<usize>,
    pub orders_1m: Option<usize>,
}

#[inline(always)]
fn extract_limits(headers: &HeaderMap) -> BinanceRateLimits {
    let mut limits = BinanceRateLimits::default();
    if let Some(val) = headers.get("X-MBX-USED-WEIGHT-1M") {
        if let Ok(s) = val.to_str() {
            limits.weight_1m = s.parse().ok();
        }
    }
    if let Some(val) = headers.get("X-MBX-ORDER-COUNT-10S") {
        if let Ok(s) = val.to_str() {
            limits.orders_10s = s.parse().ok();
        }
    }
    if let Some(val) = headers.get("X-MBX-ORDER-COUNT-1M") {
        if let Ok(s) = val.to_str() {
            limits.orders_1m = s.parse().ok();
        }
    }
    limits
}

/// F1.8: Retry-After del 429 (segundos). Default 60 si Binance no lo envía:
/// conservador y documentado — cubre la ventana típica de reinicio de peso.
#[inline(always)]
fn extract_retry_after(headers: &HeaderMap) -> u64 {
    headers
        .get("Retry-After")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(60)
}

/// F1-005 (Ola Ω150): Clasificador canónico de control de cuota HTTP (429 / 418)
/// compartido por todas las rutas del cliente: los transportes de las peticiones
/// firmadas (`execute_order_payload`, `execute_order_payload_body`,
/// `cancel_order_payload`, `get_payload_with_timeout`, `post_payload`), cuyo
/// error registra el ejecutor, y las tres rutas sin ejecutor (hora del NTP, alta
/// y renovación del listenKey), que lo registran ellas (CL-189).
#[inline(always)]
pub fn classify_rate_limit_or_ban(status_code: u16, headers: &HeaderMap) -> Option<String> {
    if status_code == 429 {
        let retry_after = extract_retry_after(headers);
        Some(format!("HTTP_429_RATE_LIMITED retry_after={}", retry_after))
    } else if status_code == 418 {
        let retry_after = headers
            .get("Retry-After")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(300);
        Some(format!("HTTP_418_IP_BANNED retry_after={}", retry_after))
    } else {
        None
    }
}

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// N20 (CL-188): el freno 429/418 del exchange de un ejecutor, compartido por
/// todos los clones de su cliente (el user-data stream y el NTP los toman con
/// `client().clone()`): el límite es de la IP, no de quien envía. Antes vivía
/// en `OrderExecutor` y los clones no lo veían. Cada ejecutor crea el suyo
/// (`new` y `new_with_shared`), como antes.
pub(crate) struct FrenoExchange {
    /// F1.8: no enviar hasta este timestamp (cooldown post-429).
    pub(crate) cooldown_until_ms: AtomicU64,
    /// F1.8: 429s consecutivos; al tercero, RATE_BRAKE (M4-C02).
    pub(crate) consecutive_429: AtomicUsize,
    /// FIX M4-C02 (quinta ola, CRITICAL): freno TEMPORAL por rate-limit
    /// (3×429 escalado o HTTP 418). A diferencia del `kill_switch` del
    /// ejecutor —permanente y reservado para incidentes reales
    /// (immune/X-009/insolvencia)— este freno AUTO-EXPIRA en este timestamp,
    /// porque un 429/418 es transitorio (Binance indica retry_after). Antes:
    /// trigger_kill_switch() latcheaba la ejecución PARA SIEMPRE por un burst
    /// de rate-limit, dejando posiciones vivas sin poder cerrarse (riesgo de
    /// liquidación) hasta reiniciar el proceso.
    pub(crate) rate_brake_until_ms: AtomicU64,
    /// El MISMO Arc que `OrderExecutor::arena`: un solo reloj sincronizado.
    reloj: std::sync::Arc<arc_swap::ArcSwapOption<quantum_arena::GlobalArena>>,
}

impl FrenoExchange {
    fn new(reloj: std::sync::Arc<arc_swap::ArcSwapOption<quantum_arena::GlobalArena>>) -> Self {
        Self {
            cooldown_until_ms: AtomicU64::new(0),
            consecutive_429: AtomicUsize::new(0),
            rate_brake_until_ms: AtomicU64::new(0),
            reloj,
        }
    }

    /// La hora sincronizada del ejecutor (la de `get_synced_timestamp`).
    #[inline(always)]
    pub(crate) fn ahora_ms(&self) -> u64 {
        crate::executor::current_synced_timestamp_ms(self.reloj.load_full().as_deref())
    }

    /// N20 (CL-189): el freno con el reloj del ejecutor, para quien no lo trae
    /// (las rutas sin ejecutor y la cancelación del stream). Hasta el primer
    /// 429/418 del ejecutor no lee el reloj: los dos sellos nunca vuelven a 0.
    #[inline(always)]
    pub(crate) fn comprobar_ahora(&self) -> Result<(), String> {
        if self.rate_brake_until_ms.load(Ordering::Relaxed) == 0
            && self.cooldown_until_ms.load(Ordering::Relaxed) == 0
        {
            return Ok(());
        }
        self.comprobar(self.ahora_ms())
    }

    /// Los dos frenos que ya puso el exchange (`OrderExecutor::check_backoff`).
    pub(crate) fn comprobar(&self, timestamp_ms: u64) -> Result<(), String> {
        // FIX M4-C02: freno TEMPORAL por rate-limit (3×429 escalado / HTTP 418).
        // A diferencia del kill_switch permanente (incidentes reales), este
        // auto-expira: cuando timestamp_ms >= rate_brake_until_ms la ejecución
        // se libera sola, sin reinicio de proceso.
        let brake_until = self.rate_brake_until_ms.load(Ordering::Relaxed);
        if timestamp_ms < brake_until {
            return Err(format!(
                "RATE_BRAKE: freno temporal activo hasta {} (ahora {})",
                brake_until, timestamp_ms
            ));
        }

        // F1.8: cooldown post-429 — Binance ya nos frenó; respetar la ventana.
        if timestamp_ms < self.cooldown_until_ms.load(Ordering::Relaxed) {
            return Err(format!(
                "RATE_LIMIT_COOLDOWN: activo hasta {} (ahora {})",
                self.cooldown_until_ms.load(Ordering::Relaxed),
                timestamp_ms
            ));
        }
        Ok(())
    }

    /// F1.8: la política ante un 429/418 (`OrderExecutor::handle_rate_limit_error`).
    pub(crate) fn registrar(&self, e: &str) -> String {
        if e.starts_with("HTTP_429_RATE_LIMITED") {
            let retry_after_s: u64 = e
                .split("retry_after=")
                .nth(1)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(60);
            let now = self.ahora_ms();
            let until = now.saturating_add(retry_after_s.saturating_mul(1000));
            // Máximo atómico: conservar cooldown mayor si ya existía.
            let mut cur = self.cooldown_until_ms.load(Ordering::Relaxed);
            while until > cur {
                match self.cooldown_until_ms.compare_exchange(
                    cur,
                    until,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(v) => cur = v,
                }
            }
            let n = self.consecutive_429.fetch_add(1, Ordering::Relaxed) + 1;
            println!(
                "⏳ [RATE-LIMIT] 429 consecutivo #{}: cooldown {}s (hasta ms {})",
                n, retry_after_s, until
            );
            if n >= 3 {
                // FIX M4-C02: freno TEMPORAL escalado, NO kill-switch permanente.
                // Un burst de 429 es transitorio; latchear para siempre dejaba
                // posiciones vivas sin cierre (liquidación) hasta reiniciar.
                // Ventana = cooldown actual + 60s de margen de seguridad.
                let brake_until = until.saturating_add(60_000);
                self.extender_freno(brake_until);
                println!(
                    "🚨 [RATE BRAKE] 3+ rate limits consecutivos — freno temporal hasta ms {} (auto-rearme).",
                    brake_until
                );
            }
        } else if e.starts_with("HTTP_418_IP_BANNED") {
            // FIX M4-C02: el ban de IP es temporal (Binance lo levanta). Freno
            // por la ventana indicada (o 5 min por defecto) — si sigue banneado
            // al expirar, el próximo 418 re-latchea. Nunca permanente.
            let ban_s: u64 = e
                .split("retry_after=")
                .nth(1)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(300);
            let now = self.ahora_ms();
            let brake_until = now.saturating_add(ban_s.saturating_mul(1000));
            self.extender_freno(brake_until);
            println!(
                "🚨 [RATE BRAKE] HTTP 418: IP baneada — freno temporal {}s (auto-rearme hasta ms {}).",
                ban_s, brake_until
            );
        }
        e.to_string()
    }

    /// N20 (CL-193): `exito` para una respuesta sana de una ruta sin ejecutor
    /// (NTP, listenKey), que también cuenta para la racha desde CL-189. Sin
    /// racha no lee el reloj.
    #[inline(always)]
    pub(crate) fn exito_ahora(&self) {
        if self.consecutive_429.load(Ordering::Relaxed) != 0 {
            self.exito(self.ahora_ms());
        }
    }

    /// F1-005 (Ola Ω150): una respuesta sana sólo reinicia consecutive_429
    /// cuando el cooldown activo ya expiró. Antes, cualquier 2xx concurrente
    /// durante una ráfaga lo reiniciaba a 0 en medio del cooldown, impidiendo
    /// alcanzar el umbral de 3+ 429 consecutivos.
    #[inline(always)]
    pub(crate) fn exito(&self, now_ms: u64) {
        if now_ms >= self.cooldown_until_ms.load(Ordering::Relaxed) {
            self.consecutive_429.store(0, Ordering::Relaxed);
        }
    }

    /// FIX M4-C02: extiende el freno temporal (máximo atómico — nunca acorta
    /// una ventana ya activa). Auto-expira: `comprobar` lo libera cuando
    /// timestamp_ms >= rate_brake_until_ms.
    #[inline(always)]
    fn extender_freno(&self, until_ms: u64) {
        let mut cur = self.rate_brake_until_ms.load(Ordering::Relaxed);
        while until_ms > cur {
            match self.rate_brake_until_ms.compare_exchange(
                cur,
                until_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
    }
}

#[derive(Clone)]
pub struct BinanceClient {
    http: Client,
    /// Arc: ArcSwap no es Clone por diseño; los clones (la réplica del stream,
    /// F1.6) comparten la misma clave. RET-31: ya nada la cambia en caliente;
    /// el host cambia de entorno construyendo otro cliente (new_with_shared).
    pub api_key: std::sync::Arc<ArcSwap<HeaderValue>>,
    /// Arc interno: los clones comparten el mismo entorno (F1.6).
    pub is_testnet: std::sync::Arc<AtomicBool>,
    /// N20 (CL-188): el freno 429/418, compartido por los clones.
    freno: std::sync::Arc<FrenoExchange>,
}

impl BinanceClient {
    pub fn new(api_key: String, is_testnet: bool) -> Self {
        // Timeout HFT calibrado a 1200ms para failover rápido y evitar slippage tardío
        Self::with_timeout(api_key, is_testnet, Duration::from_millis(1200))
    }

    pub fn with_timeout(api_key: String, is_testnet: bool, timeout: Duration) -> Self {
        // FIX #1497: Construcción resiliente de cliente HTTP reqwest HFT sin expect
        let http = ClientBuilder::new()
            .pool_max_idle_per_host(25) // Maximize concurrent keep-alives for Binance
            .pool_idle_timeout(None) // NEVER drop idle connections to avoid TLS handshake latency
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(15))) // Latidos cada 15s para evitar drops fantasma de socket
            .hickory_dns(true) // Fast DNS
            .timeout(timeout) // Timeout optimizado para HFT
            .build()
            .unwrap_or_else(|_| Client::new());

        let header_val =
            HeaderValue::from_str(&api_key).unwrap_or_else(|_| HeaderValue::from_static(""));
        Self {
            http,
            api_key: std::sync::Arc::new(ArcSwap::from_pointee(header_val)),
            is_testnet: std::sync::Arc::new(AtomicBool::new(is_testnet)),
            freno: std::sync::Arc::new(FrenoExchange::new(std::sync::Arc::new(
                arc_swap::ArcSwapOption::empty(),
            ))),
        }
    }

    /// N20 (CL-188): un freno nuevo con el reloj del ejecutor (el mismo Arc
    /// que su campo `arena`); los clones posteriores lo comparten.
    pub(crate) fn con_reloj(
        mut self,
        reloj: std::sync::Arc<arc_swap::ArcSwapOption<quantum_arena::GlobalArena>>,
    ) -> Self {
        self.freno = std::sync::Arc::new(FrenoExchange::new(reloj));
        self
    }

    /// N20 (CL-188): el freno 429/418 de este cliente y sus clones.
    #[inline(always)]
    pub(crate) fn freno(&self) -> &FrenoExchange {
        &self.freno
    }

    /// N20 (CL-189): el 429/418 de una ruta sin ejecutor arma el freno con su
    /// Retry-After. Es su ÚNICO registro: el ejecutor no llama a estas rutas, y
    /// el error vuelve marcado «FRENO_ARMADO: …» (CL-194), que `registrar` no
    /// reconoce, así que quien lo registre otra vez no cuenta dos veces.
    #[inline]
    fn freno_de_respuesta(&self, estado: u16, cabeceras: &HeaderMap) -> Option<String> {
        classify_rate_limit_or_ban(estado, cabeceras)
            .map(|e| format!("FRENO_ARMADO: {}", self.freno.registrar(&e)))
    }

    pub fn get_base_url(&self) -> &str {
        if self.is_testnet.load(Ordering::Relaxed) {
            BINANCE_TESTNET_URL
        } else {
            BINANCE_BASE_URL
        }
    }

    /// F1.6: POST /fapi/v1/listenKey — crea la clave del user-data stream.
    /// NO requiere firma: solo el header X-MBX-APIKEY.
    pub async fn create_listen_key(&self) -> Result<String, String> {
        self.freno.comprobar_ahora()?;
        let api_key = self.api_key.load();
        let url = format!("{}/fapi/v1/listenKey", self.get_base_url());
        let response = self
            .http
            .post(&url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await
            .map_err(|e| format!("Network Error: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            if let Some(e) = self.freno_de_respuesta(status.as_u16(), response.headers()) {
                return Err(e);
            }
            let body = response.text().await.unwrap_or_default();
            return Err(crate::order_types::parse_reject_body(
                &body,
                status.as_u16(),
            ));
        }
        // CL-193: el exchange la sirvió; reinicia la racha como las del ejecutor.
        self.freno.exito_ahora();
        let body = response
            .text()
            .await
            .map_err(|e| format!("TEXT_ERR: {}", e))?;
        #[derive(serde::Deserialize)]
        struct ListenKeyResp {
            #[serde(rename = "listenKey")]
            listen_key: String,
        }
        let parsed: ListenKeyResp =
            serde_json::from_str(&body).map_err(|e| format!("LISTENKEY_PARSE: {}", e))?;
        Ok(parsed.listen_key)
    }

    /// F1.6: PUT /fapi/v1/listenKey — keepalive (expira a los 60 min; refrescar a los 30).
    pub async fn keep_alive_listen_key(&self) -> Result<(), String> {
        self.freno.comprobar_ahora()?;
        let api_key = self.api_key.load();
        let url = format!("{}/fapi/v1/listenKey", self.get_base_url());
        let response = self
            .http
            .put(&url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await
            .map_err(|e| format!("Network Error: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            if let Some(e) = self.freno_de_respuesta(status.as_u16(), response.headers()) {
                return Err(e);
            }
            let body = response.text().await.unwrap_or_default();
            return Err(crate::order_types::parse_reject_body(
                &body,
                status.as_u16(),
            ));
        }
        // CL-193: el exchange la sirvió; reinicia la racha como las del ejecutor.
        self.freno.exito_ahora();
        Ok(())
    }

    /// Fetches server time directly from Binance endpoint GET /fapi/v1/time
    pub async fn fetch_server_time(&self) -> Result<u64, String> {
        self.freno.comprobar_ahora()?;
        let url = format!("{}/fapi/v1/time", self.get_base_url());
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Network Error: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            if let Some(e) = self.freno_de_respuesta(status.as_u16(), response.headers()) {
                return Err(e);
            }
            let body = response.text().await.unwrap_or_default();
            return Err(crate::order_types::parse_reject_body(
                &body,
                status.as_u16(),
            ));
        }
        // CL-193: el exchange la sirvió; reinicia la racha como las del ejecutor.
        self.freno.exito_ahora();

        #[derive(serde::Deserialize)]
        struct ServerTimeResp {
            #[serde(rename = "serverTime")]
            server_time: u64,
        }

        let resp: ServerTimeResp = response
            .json()
            .await
            .map_err(|e| format!("JSON Parse Error: {}", e))?;

        Ok(resp.server_time)
    }

    /// Ejecuta una orden firmada enviando el payload HTTP de forma asíncrona O(1).
    #[inline(always)]
    pub async fn execute_order_payload(&self, full_url: &str) -> Result<BinanceRateLimits, String> {
        let api_key = self.api_key.load();
        let response = self
            .http
            .post(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let limits = extract_limits(resp.headers());
                    Ok(limits)
                } else if let Some(rate_err) =
                    classify_rate_limit_or_ban(resp.status().as_u16(), resp.headers())
                {
                    Err(rate_err)
                } else {
                    let text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    Err(format!("Binance API Error: {}", text))
                }
            }
            Err(e) => Err(format!("AMBIGUOUS: Network Error: {}", e)),
        }
    }

    /// F1.1: POST de orden con respuesta TIPADA — lee el body completo y lo
    /// parsea a OrderAck (orderId, status, executedQty, avgPrice, fills).
    /// Distingue tres clases de fallo para idempotencia (F1.2):
    ///   - Err empezando con "AMBIGUOUS:" → timeout/red: la orden PUEDE existir
    ///     en Binance; el caller DEBE consultar por clientOrderId antes de reintentar.
    ///   - ACK ilegible, HTTP408 y códigos -1006/-1007 → también ambiguos.
    ///   - Rechazos conocidos no prueban que sea seguro repetir otra intención.
    ///   - Ok(ack) → respuesta conocida.
    #[inline(always)]
    pub async fn execute_order_payload_typed(
        &self,
        full_url: &str,
    ) -> Result<(BinanceRateLimits, crate::order_types::OrderAck), String> {
        use crate::order_types::{parse_order_body, OrderAck};
        let (limits, body) = self.execute_order_payload_body(full_url).await?;
        // A successful submission can precede a truncated/invalid
        // response. A parse error is not proof of non-execution.
        let ack: OrderAck = parse_order_body(&body).map_err(|e| format!("AMBIGUOUS: {}", e))?;
        Ok((limits, ack))
    }

    /// CL-39: el POST de orden con la MISMA clasificación de fallos que
    /// `execute_order_payload_typed`, devolviendo el cuerpo 2xx sin parsear
    /// para que el llamador lo valide con su propio contrato (la IOC exige
    /// estado terminal y `executedQty` presentes, sin defaults a cero).
    pub async fn execute_order_payload_body(
        &self,
        full_url: &str,
    ) -> Result<(BinanceRateLimits, String), String> {
        use crate::order_types::{parse_reject_body, truncate};

        let api_key = self.api_key.load();
        let response = self
            .http
            .post(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await;

        match response {
            Ok(resp) => {
                let limits = extract_limits(resp.headers());
                let status = resp.status();
                // These transport-level controls are known from the headers,
                // even if reading the error body later would fail.
                if let Some(rate_err) =
                    classify_rate_limit_or_ban(status.as_u16(), resp.headers())
                {
                    return Err(rate_err);
                }
                let body = resp.text().await.map_err(|_| {
                    format!(
                        "AMBIGUOUS: HTTP {} unreadable response body",
                        status.as_u16()
                    )
                })?;
                if status.is_success() {
                    Ok((limits, body))
                } else if status.is_client_error() {
                    use crate::order_types::{error_codes, BinanceApiError};
                    let unknown_execution = status.as_u16() == 408
                        || serde_json::from_str::<BinanceApiError>(&body)
                            .map(|e| {
                                matches!(
                                    e.code,
                                    error_codes::EXECUTION_STATUS_UNKNOWN
                                        | error_codes::BACKEND_TIMEOUT
                                )
                            })
                            .unwrap_or(false);
                    if unknown_execution {
                        Err(format!(
                            "AMBIGUOUS: HTTP {} body={}",
                            status.as_u16(),
                            truncate(&body, 200)
                        ))
                    } else {
                        Err(parse_reject_body(&body, status.as_u16()))
                    }
                } else {
                    // 5xx: estado desconocido — tratar como ambiguo.
                    Err(format!(
                        "AMBIGUOUS: HTTP {} body={}",
                        status.as_u16(),
                        truncate(&body, 200)
                    ))
                }
            }
            Err(e) => Err(format!("AMBIGUOUS: Network Error: {}", e)),
        }
    }

    /// Cancela una orden enviando el payload HTTP (DELETE).
    #[inline(always)]
    pub async fn cancel_order_payload(&self, full_url: &str) -> Result<BinanceRateLimits, String> {
        // CL-190 (N20): la cancelación de la pierna hermana (user_data_stream.rs)
        // sale por un clon sin ejecutor. Sólo el freno: el 429 de un envío del
        // ejecutor lo registra su handle_rate_limit_error; el stream sólo lo
        // imprime (fila N27 del censo).
        self.freno.comprobar_ahora()?;
        let api_key = self.api_key.load();
        let response = self
            .http
            .delete(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone())
            .send()
            .await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let limits = extract_limits(resp.headers());
                    Ok(limits)
                } else if let Some(rate_err) =
                    classify_rate_limit_or_ban(resp.status().as_u16(), resp.headers())
                {
                    Err(rate_err)
                } else {
                    let text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    Err(format!("Binance API Error: {}", text))
                }
            }
            Err(e) => Err(format!("Network Error: {}", e)),
        }
    }
    /// GET payload (used for fetching balances, etc)
    #[inline(always)]
    pub async fn get_payload(&self, full_url: &str) -> Result<(BinanceRateLimits, String), String> {
        self.get_payload_with_timeout(full_url, None).await
    }

    /// GET con presupuesto de tiempo POR REQUEST. El timeout global HFT
    /// (1.2s) incluye la lectura del body: exchangeInfo (~MBs) no cabe y
    /// falla intermitentemente según la latencia del momento. Los endpoints
    /// masivos usan esta variante con presupuesto explícito; los de orden
    /// siguen con el global (latencia es parte del contrato de ejecución).
    pub async fn get_payload_with_timeout(
        &self,
        full_url: &str,
        timeout: Option<std::time::Duration>,
    ) -> Result<(BinanceRateLimits, String), String> {
        let api_key = self.api_key.load();
        let mut req = self
            .http
            .get(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref().clone());
        if let Some(t) = timeout {
            req = req.timeout(t);
        }
        let response = req.send().await;

        match response {
            Ok(resp) => {
                let status = resp.status();
                if let Some(rate_err) =
                    classify_rate_limit_or_ban(status.as_u16(), resp.headers())
                {
                    return Err(rate_err);
                }
                let limits = extract_limits(resp.headers());
                let raw_text = resp
                    .text()
                    .await
                    .unwrap_or_else(|e| format!("TEXT_ERR: {}", e));
                if status.is_success() {
                    Ok((limits, raw_text))
                } else {
                    // Nunca loguear `full_url`: contiene la firma HMAC del request.
                    Err(format!(
                        "Binance API Error: status={} body={}",
                        status, raw_text
                    ))
                }
            }
            Err(e) => Err(format!("Network Error: {}", e)),
        }
    }

    /// POST payload (used for changing leverage, margin type, etc)
    #[inline(always)]
    pub async fn post_payload(
        &self,
        full_url: &str,
    ) -> Result<(BinanceRateLimits, String), String> {
        let api_key_arc = self.api_key.load();
        let api_key = (*api_key_arc).clone();
        let response = self
            .http
            .post(full_url)
            .header("X-MBX-APIKEY", api_key.as_ref())
            .send()
            .await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let limits = extract_limits(resp.headers());
                    let text = resp.text().await.unwrap_or_else(|_| "[]".to_string());
                    Ok((limits, text))
                } else if let Some(rate_err) =
                    classify_rate_limit_or_ban(resp.status().as_u16(), resp.headers())
                {
                    Err(rate_err)
                } else {
                    let text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    Err(format!("Binance API Error: {}", text))
                }
            }
            Err(e) => Err(format!("Network Error: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_alloc_buffer_push_str_f64_u64_and_clear() {
        let mut buf = ZeroAllocBuffer::new();
        assert_eq!(buf.as_str(), "");
        assert!(!buf.is_overflow());

        buf.push_str("symbol=BTCUSDT&price=");
        buf.push_f64(50000.5);
        buf.push_str("&timestamp=");
        buf.push_u64(1700000000000);

        assert!(!buf.is_overflow());
        let s = buf.as_str();
        assert!(s.starts_with("symbol=BTCUSDT&price=50000.5&timestamp=1700000000000"));

        buf.clear();
        assert_eq!(buf.as_str(), "");
        assert!(!buf.is_overflow());
    }

    #[test]
    fn test_zero_alloc_buffer_overflow_detection() {
        let mut buf = ZeroAllocBuffer::new();
        let large_str = "A".repeat(1025);
        buf.push_str(&large_str);
        assert!(buf.is_overflow());
    }

    #[test]
    fn test_binance_rate_limits_and_retry_after_extraction() {
        let mut headers = HeaderMap::new();
        headers.insert("X-MBX-USED-WEIGHT-1M", HeaderValue::from_static("350"));
        headers.insert("X-MBX-ORDER-COUNT-10S", HeaderValue::from_static("25"));
        headers.insert("X-MBX-ORDER-COUNT-1M", HeaderValue::from_static("120"));
        headers.insert("Retry-After", HeaderValue::from_static("45"));

        let limits = extract_limits(&headers);
        assert_eq!(limits.weight_1m, Some(350));
        assert_eq!(limits.orders_10s, Some(25));
        assert_eq!(limits.orders_1m, Some(120));

        let retry_after = extract_retry_after(&headers);
        assert_eq!(retry_after, 45);

        let empty_headers = HeaderMap::new();
        let default_retry = extract_retry_after(&empty_headers);
        assert_eq!(default_retry, 60);

        assert_eq!(
            classify_rate_limit_or_ban(429, &headers).as_deref(),
            Some("HTTP_429_RATE_LIMITED retry_after=45")
        );
        assert_eq!(
            classify_rate_limit_or_ban(418, &headers).as_deref(),
            Some("HTTP_418_IP_BANNED retry_after=45")
        );
        assert_eq!(
            classify_rate_limit_or_ban(418, &empty_headers).as_deref(),
            Some("HTTP_418_IP_BANNED retry_after=300")
        );
        assert_eq!(classify_rate_limit_or_ban(200, &headers), None);
    }

    #[test]
    fn test_binance_client_api_key_and_testnet() {
        let testnet = BinanceClient::new("key1".to_string(), true);
        assert_eq!(testnet.get_base_url(), BINANCE_TESTNET_URL);
        assert_eq!(testnet.api_key.load().to_str().ok(), Some("key1"));

        let mainnet = BinanceClient::new("key2".to_string(), false);
        assert_eq!(mainnet.get_base_url(), BINANCE_BASE_URL);
        assert_eq!(mainnet.api_key.load().to_str().ok(), Some("key2"));
    }

    /// CL-189: el 429/418 de una ruta sin ejecutor arma el freno compartido
    /// una sola vez y con su Retry-After; los tres 429 escalan como en el
    /// ejecutor y otro estado no toca nada. Sin 429 previo no frena.
    #[test]
    fn cl189_el_429_de_una_ruta_cruda_arma_el_freno_una_vez() {
        let c = BinanceClient::new("k".into(), true);
        assert!(c.freno().comprobar_ahora().is_ok(), "sin 429 no frena");
        let mut cab = HeaderMap::new();
        cab.insert("Retry-After", HeaderValue::from_static("7"));
        let antes = c.freno().ahora_ms();
        assert_eq!(
            c.freno_de_respuesta(429, &cab).as_deref(),
            Some("FRENO_ARMADO: HTTP_429_RATE_LIMITED retry_after=7")
        );
        let despues = c.freno().ahora_ms();
        let hasta = c.freno().cooldown_until_ms.load(Ordering::Relaxed);
        assert!(hasta >= antes + 7_000 && hasta <= despues + 7_000, "Retry-After de 7 s");
        assert!(matches!(c.freno().comprobar_ahora(), Err(ref e) if e.starts_with("RATE_LIMIT_COOLDOWN")));
        assert_eq!(c.freno().consecutive_429.load(Ordering::Relaxed), 1);
        assert_eq!(c.freno_de_respuesta(503, &cab), None);
        assert_eq!(c.freno().consecutive_429.load(Ordering::Relaxed), 1);
        let clon = c.clone();
        clon.freno_de_respuesta(429, &cab);
        clon.clone().freno_de_respuesta(429, &cab);
        assert_eq!(c.freno().consecutive_429.load(Ordering::Relaxed), 3);
        assert!(matches!(c.freno().comprobar_ahora(), Err(ref e) if e.starts_with("RATE_BRAKE")));
        assert!(c.freno().rate_brake_until_ms.load(Ordering::Relaxed) >= antes + 67_000);

        let c = BinanceClient::new("k".into(), true);
        let mut cab = HeaderMap::new();
        cab.insert("Retry-After", HeaderValue::from_static("120"));
        let antes = c.freno().ahora_ms();
        assert_eq!(
            c.freno_de_respuesta(418, &cab).as_deref(),
            Some("FRENO_ARMADO: HTTP_418_IP_BANNED retry_after=120")
        );
        assert!(matches!(c.freno().comprobar_ahora(), Err(ref e) if e.starts_with("RATE_BRAKE")));
        assert!(c.freno().rate_brake_until_ms.load(Ordering::Relaxed) >= antes + 120_000);
        assert_eq!(c.freno().consecutive_429.load(Ordering::Relaxed), 0);
    }

    const CLIENTE: &str = include_str!("client.rs");
    const EJECUTOR: &str = include_str!("executor.rs");

    fn produccion(codigo: &'static str) -> &'static str {
        codigo.split("#[cfg(test)]").next().unwrap_or("")
    }

    /// Cuerpo de `fn {f}(` en el cliente de producción, hasta la siguiente función.
    fn cuerpo(f: &str) -> &'static str {
        let codigo = produccion(CLIENTE);
        let Some(i) = codigo.find(&format!("fn {f}(")) else {
            return "";
        };
        let fin = ["\n    pub ", "\n    fn ", "\n    async fn "]
            .iter()
            .filter_map(|m| codigo[i + 3..].find(m).map(|k| i + 3 + k))
            .min()
            .unwrap_or(codigo.len());
        &codigo[i..fin]
    }

    fn antes(c: &str, primero: &str, despues: &str) -> bool {
        matches!((c.find(primero), c.find(despues)), (Some(a), Some(b)) if a < b)
    }

    /// Los siete transportes de las peticiones firmadas del ejecutor: su
    /// 429/418 lo registra el ejecutor (`handle_rate_limit_error`), no ellos.
    const TRANSPORTES: [&str; 7] = [
        "execute_order_payload",
        "execute_order_payload_typed",
        "execute_order_payload_body",
        "cancel_order_payload",
        "get_payload",
        "get_payload_with_timeout",
        "post_payload",
    ];

    /// RED antes de CL-189: las tres rutas sin ejecutor (hora del NTP, alta
    /// y renovación del listenKey) no miraban el freno ni lo armaban con su
    /// propio 429/418 (lo leían como «REJECTED http=429»). Cada respuesta se
    /// registra una sola vez: los transportes del ejecutor no registran y el
    /// ejecutor no llama a las rutas crudas.
    #[test]
    fn cl189_las_rutas_crudas_frenan_y_registran_una_vez() {
        for f in ["create_listen_key", "keep_alive_listen_key", "fetch_server_time"] {
            let c = cuerpo(f);
            assert!(antes(c, "self.freno.comprobar_ahora()?;", ".send()"), "{f} sale sin freno");
            assert!(
                antes(c, "self.freno_de_respuesta(", "parse_reject_body("),
                "{f} no arma el freno con su 429/418"
            );
        }
        for f in TRANSPORTES {
            let c = cuerpo(f);
            assert!(!c.is_empty(), "{f} no encontrado");
            assert!(!c.contains("freno_de_respuesta(") && !c.contains(".registrar("), "{f} registra");
        }
        assert_eq!(produccion(CLIENTE).matches("consecutive_429.fetch_add").count(), 1);
        assert_eq!(produccion(EJECUTOR).matches("consecutive_429.fetch_add").count(), 0);
        for llamada in [".create_listen_key(", ".keep_alive_listen_key(", "client.fetch_server_time("] {
            assert!(!produccion(EJECUTOR).contains(llamada), "el ejecutor llama a {llamada}");
        }
    }

    /// RED antes de CL-190: la cancelación (la de la pierna hermana sale del
    /// stream por un clon, sin ejecutor) no miraba el freno. Sólo lo mira: su
    /// 429 lo registra el ejecutor cuando el envío es suyo; el stream sólo lo
    /// imprime (fila N27 del censo).
    #[test]
    fn cl190_la_cancelacion_frena_sin_registrar() {
        let c = cuerpo("cancel_order_payload");
        assert!(antes(c, "self.freno.comprobar_ahora()?;", ".send()"), "cancela sin freno");
        assert!(!c.contains("freno_de_respuesta(") && !c.contains(".registrar("), "registra");
    }

    /// RED antes de CL-193: el 429 de una ruta sin ejecutor cuenta para la
    /// racha (CL-189), pero su respuesta sana no la reiniciaba: en papel, con
    /// el ejecutor casi sin tráfico propio, tres 429 aislados del NTP a lo
    /// largo de un día armaban RATE_BRAKE, y desde ahí cada 429 lo rearmaba.
    #[test]
    fn cl193_las_rutas_crudas_reinician_la_racha() {
        for f in ["create_listen_key", "keep_alive_listen_key", "fetch_server_time"] {
            let c = cuerpo(f);
            // RV17-T: una sola vez y fuera de la rama de error (tras su último
            // `parse_reject_body(`): un 4xx/5xx no reinicia la racha.
            let exito = c.find("self.freno.exito_ahora();");
            let error = c.rfind("parse_reject_body(");
            assert!(
                matches!((error, exito), (Some(e), Some(x)) if e < x),
                "{f}: su 2xx no reinicia la racha"
            );
            assert_eq!(c.matches("exito_ahora").count(), 1, "{f}");
        }
        // Con el cooldown vencido reinicia; dentro de él no (F1-005).
        let c = BinanceClient::new("k".into(), true);
        c.freno().registrar("HTTP_429_RATE_LIMITED retry_after=0");
        assert_eq!(c.freno().consecutive_429.load(Ordering::Relaxed), 1);
        c.clone().freno().exito_ahora();
        assert_eq!(c.freno().consecutive_429.load(Ordering::Relaxed), 0);
        c.freno().registrar("HTTP_429_RATE_LIMITED retry_after=30");
        c.freno().exito_ahora();
        assert_eq!(c.freno().consecutive_429.load(Ordering::Relaxed), 1, "dentro del cooldown");
    }

    /// RED antes de CL-194: la ruta cruda devolvía su 429 ya registrado con el
    /// mismo texto que reconoce `registrar`; quien lo registrase otra vez (el
    /// stream o el NTP, que hoy sólo lo imprimen) lo contaba dos veces y el
    /// segundo 429 real ya armaba RATE_BRAKE.
    #[test]
    fn cl194_el_error_ya_registrado_no_cuenta_dos_veces() {
        let c = BinanceClient::new("k".into(), true);
        let mut cab = HeaderMap::new();
        cab.insert("Retry-After", HeaderValue::from_static("7"));
        let e = c.freno_de_respuesta(429, &cab).unwrap_or_default();
        assert!(e.contains("HTTP_429_RATE_LIMITED retry_after=7"), "{e}");
        c.clone().freno().registrar(&e);
        assert_eq!(c.freno().consecutive_429.load(Ordering::Relaxed), 1, "{e}");
    }
}
