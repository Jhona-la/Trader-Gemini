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

/// N23 (CL-196): el error de una petición firmada con la cabecera de uso de su
/// respuesta. Un 400, un -2019 o un 503 también gastan peso de la IP y traen
/// `X-MBX-USED-WEIGHT-1M`; antes el error era un `String` y esa cabecera se
/// perdía, así que una racha de rechazos subía el uso real sin que la franja
/// del 80 % lo viera. Se lee como `&str` (`Deref`) y se muestra igual que
/// antes; no hay `From<ErrorRest> for String`, para que ningún `?` tire la
/// cabecera sin pasar por `OrderExecutor::registrar_error`.
///
/// Sin cabecera: sin respuesta (red, timeout, rechazo local del freno) y en un
/// 429/418, cuya cabecera nunca se registra: manda su Retry-After. N24
/// (CL-201): una petición que salió y no tuvo respuesta (`sin_respuesta`) pudo
/// contar en el exchange sin que ninguna cabecera lo diga; el cupo de una
/// crítica anota entonces su coste como usado.
///
/// El prefijo AMBIGUOUS dice que la ejecución de una orden es desconocida, no
/// que faltara la respuesta: el 503 o el -1007 de un envío de orden es
/// AMBIGUOUS y trae su cabecera (N25, CL-199). Que hubo respuesta no se deduce
/// del prefijo.
#[derive(Debug, Clone)]
pub struct ErrorRest {
    texto: String,
    limites: BinanceRateLimits,
    sin_respuesta: bool,
}

impl ErrorRest {
    /// Un 429/418, o un rechazo local que no llegó a salir (el freno).
    pub fn sin_cabecera(texto: impl Into<String>) -> Self {
        Self {
            texto: texto.into(),
            limites: BinanceRateLimits::default(),
            sin_respuesta: false,
        }
    }

    fn con_cabecera(texto: String, limites: BinanceRateLimits) -> Self {
        Self {
            texto,
            limites,
            sin_respuesta: false,
        }
    }

    /// N24 (CL-201): un error de transporte, sin cabecera. `salio` dice si
    /// la petición pudo llegar al exchange (la función `salio` de abajo).
    pub(crate) fn de_red(texto: String, salio: bool) -> Self {
        Self {
            texto,
            limites: BinanceRateLimits::default(),
            sin_respuesta: salio,
        }
    }

    /// N24 (CL-201): la petición salió y no llegó respuesta (timeout,
    /// conexión cortada a mitad): el exchange pudo contarla. Es falso con
    /// cualquier respuesta, traiga o no cabecera, en un 429/418, en un
    /// rechazo local y si falló al conectar. Un plazo total que vence
    /// mientras conecta cuenta como salida, del lado seguro (`salio`). No se
    /// deduce del texto.
    #[inline]
    pub fn sin_respuesta(&self) -> bool {
        self.sin_respuesta
    }

    /// El uso que trajo la respuesta de error.
    #[inline]
    pub fn limites(&self) -> &BinanceRateLimits {
        &self.limites
    }
}

impl std::ops::Deref for ErrorRest {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        &self.texto
    }
}

impl std::fmt::Display for ErrorRest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.texto)
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
    /// Un 2xx es la aceptación (su cuerpo no se lee). Un no-2xx se clasifica
    /// como en `execute_order_payload_body` (N25, CL-199): la red, el 5xx, el
    /// 408, -1006/-1007 y un cuerpo ilegible son AMBIGUOUS (la orden o la
    /// pierna pudo quedar en el exchange); un rechazo 4xx firme conserva su
    /// texto, «Binance API Error: {cuerpo}».
    #[inline(always)]
    pub async fn execute_order_payload(&self, full_url: &str) -> Result<BinanceRateLimits, ErrorRest> {
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
                    Err(ErrorRest::sin_cabecera(rate_err))
                } else {
                    // N23 (CL-196): la cabecera antes que el cuerpo, que consume la respuesta.
                    let limits = extract_limits(resp.headers());
                    let estado = resp.status();
                    match resp.text().await {
                        Ok(cuerpo) => Err(error_de_orden(estado, &cuerpo, limits, |c| {
                            format!("Binance API Error: {}", c)
                        })),
                        Err(_) => Err(cuerpo_ilegible(estado, limits)),
                    }
                }
            }
            Err(e) => Err(error_de_red("AMBIGUOUS: ", e)),
        }
    }

    /// F1.1: POST de orden con respuesta TIPADA — lee el body completo y lo
    /// parsea a OrderAck (orderId, status, executedQty, avgPrice, fills).
    /// Distingue tres clases de fallo para idempotencia (F1.2):
    ///   - Err empezando con "AMBIGUOUS:" → timeout/red: la orden PUEDE existir
    ///     en Binance; el caller DEBE consultar por clientOrderId antes de reintentar.
    ///   - ACK ilegible, cuerpo ilegible, HTTP 408, 5xx y códigos -1006/-1007
    ///     → también ambiguos (`error_de_orden`, `cuerpo_ilegible`).
    ///   - Rechazos conocidos no prueban que sea seguro repetir otra intención.
    ///   - Ok(ack) → respuesta conocida.
    #[inline(always)]
    pub async fn execute_order_payload_typed(
        &self,
        full_url: &str,
    ) -> Result<(BinanceRateLimits, crate::order_types::OrderAck), ErrorRest> {
        use crate::order_types::{parse_order_body, OrderAck};
        let (limits, body) = self.execute_order_payload_body(full_url).await?;
        // A successful submission can precede a truncated/invalid
        // response. A parse error is not proof of non-execution.
        // N23 (CL-196): el 2xx sí respondió; su cabecera viaja con el error.
        let ack: OrderAck = parse_order_body(&body)
            .map_err(|e| ErrorRest::con_cabecera(format!("AMBIGUOUS: {}", e), limits))?;
        Ok((limits, ack))
    }

    /// CL-39: el POST de orden con la MISMA clasificación de fallos que
    /// `execute_order_payload_typed`, devolviendo el cuerpo 2xx sin parsear
    /// para que el llamador lo valide con su propio contrato (la IOC exige
    /// estado terminal y `executedQty` presentes, sin defaults a cero).
    pub async fn execute_order_payload_body(
        &self,
        full_url: &str,
    ) -> Result<(BinanceRateLimits, String), ErrorRest> {
        use crate::order_types::parse_reject_body;

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
                    return Err(ErrorRest::sin_cabecera(rate_err));
                }
                let body = resp.text().await.map_err(|_| cuerpo_ilegible(status, limits))?;
                if status.is_success() {
                    Ok((limits, body))
                } else {
                    Err(error_de_orden(status, &body, limits, |c| {
                        parse_reject_body(c, status.as_u16())
                    }))
                }
            }
            Err(e) => Err(error_de_red("AMBIGUOUS: ", e)),
        }
    }

    /// Cancela una orden enviando el payload HTTP (DELETE).
    #[inline(always)]
    pub async fn cancel_order_payload(&self, full_url: &str) -> Result<BinanceRateLimits, ErrorRest> {
        // CL-190 (N20): la cancelación de la pierna hermana (user_data_stream.rs)
        // sale por un clon sin ejecutor. Sólo el freno: el 429 de un envío del
        // ejecutor lo registra su registrar_error (CL-196); el stream sólo lo
        // imprime (fila N27 del censo).
        self.freno.comprobar_ahora().map_err(ErrorRest::sin_cabecera)?;
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
                    Err(ErrorRest::sin_cabecera(rate_err))
                } else {
                    let limits = extract_limits(resp.headers());
                    let text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    Err(ErrorRest::con_cabecera(format!("Binance API Error: {}", text), limits))
                }
            }
            Err(e) => Err(error_de_red("", e)),
        }
    }
    /// GET payload (used for fetching balances, etc)
    #[inline(always)]
    pub async fn get_payload(&self, full_url: &str) -> Result<(BinanceRateLimits, String), ErrorRest> {
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
    ) -> Result<(BinanceRateLimits, String), ErrorRest> {
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
                    return Err(ErrorRest::sin_cabecera(rate_err));
                }
                let limits = extract_limits(resp.headers());
                let raw_text = match resp.text().await {
                    Ok(texto) => texto,
                    // N40 (CL-203): un 2xx cuyo cuerpo no se pudo leer (cortado,
                    // o el plazo venció leyéndolo) no es una respuesta buena.
                    // Antes salía como Ok con «TEXT_ERR: …»: su lector fallaba
                    // después con un error de JSON que no decía que hubo
                    // respuesta, y ni get_payload_account ni exchangeInfo lo
                    // reintentaban. Es un error con su cabecera y su causa, sin
                    // la URL; no AMBIGUOUS, porque una lectura no ejecuta nada.
                    Err(e) if status.is_success() => {
                        return Err(ErrorRest::con_cabecera(
                            format!(
                                "HTTP {} unreadable response body: {}",
                                status.as_u16(),
                                e.without_url()
                            ),
                            limits,
                        ));
                    }
                    Err(e) => format!("TEXT_ERR: {}", e),
                };
                if status.is_success() {
                    Ok((limits, raw_text))
                } else {
                    // Nunca loguear `full_url`: contiene la firma HMAC del request.
                    Err(ErrorRest::con_cabecera(
                        format!("Binance API Error: status={} body={}", status, raw_text),
                        limits,
                    ))
                }
            }
            Err(e) => Err(error_de_red("", e)),
        }
    }

    /// POST payload (used for changing leverage, margin type, etc)
    #[inline(always)]
    pub async fn post_payload(
        &self,
        full_url: &str,
    ) -> Result<(BinanceRateLimits, String), ErrorRest> {
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
                    // N40: este «[]» no lo lee nadie como lista. El hedge
                    // (positionSide/dual) decide por el estado, y un 2xx es
                    // que el cambio entró; el apalancamiento lo rechaza como
                    // confirmación mal formada. Un error aquí haría que
                    // ensure_hedge_mode diera la cuenta por ONE-WAY tras un
                    // cambio aplicado.
                    let text = resp.text().await.unwrap_or_else(|_| "[]".to_string());
                    Ok((limits, text))
                } else if let Some(rate_err) =
                    classify_rate_limit_or_ban(resp.status().as_u16(), resp.headers())
                {
                    Err(ErrorRest::sin_cabecera(rate_err))
                } else {
                    let limits = extract_limits(resp.headers());
                    let text = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    Err(ErrorRest::con_cabecera(format!("Binance API Error: {}", text), limits))
                }
            }
            Err(e) => Err(error_de_red("", e)),
        }
    }
}

/// N25 (CL-199): si un no-2xx de un envío de orden deja su ejecución
/// desconocida. Lo es todo lo que no es un 4xx (5xx, y 1xx/3xx, que un envío
/// de orden no espera), el 408 y un 4xx con -1006/-1007: el exchange no dice
/// si la orden quedó, y quien lo lee debe consultarla antes de darla por
/// rechazada (contrato F1.2 de `execute_order_payload_typed`). Un 2xx no lo es.
fn ejecucion_desconocida(estado: reqwest::StatusCode, cuerpo: &str) -> bool {
    use crate::order_types::{error_codes, BinanceApiError};
    if estado.is_success() {
        return false;
    }
    !estado.is_client_error()
        || estado.as_u16() == 408
        || serde_json::from_str::<BinanceApiError>(cuerpo)
            .map(|e| {
                matches!(
                    e.code,
                    error_codes::EXECUTION_STATUS_UNKNOWN | error_codes::BACKEND_TIMEOUT
                )
            })
            .unwrap_or(false)
}

/// N25 (CL-199): el error de un no-2xx de un envío de orden (que no sea un
/// 429/418), con su cabecera. Si su ejecución es desconocida, AMBIGUOUS con
/// el estado y el cuerpo truncado; si no, el rechazo firme de cada
/// transporte (`firme`), con el texto que leen sus llamadores.
fn error_de_orden(
    estado: reqwest::StatusCode,
    cuerpo: &str,
    limites: BinanceRateLimits,
    firme: impl FnOnce(&str) -> String,
) -> ErrorRest {
    use crate::order_types::truncate;
    if ejecucion_desconocida(estado, cuerpo) {
        ErrorRest::con_cabecera(
            format!("AMBIGUOUS: HTTP {} body={}", estado.as_u16(), truncate(cuerpo, 200)),
            limites,
        )
    } else {
        ErrorRest::con_cabecera(firme(cuerpo), limites)
    }
}

/// N24 (CL-201): si una petición fallida pudo llegar al exchange. No llega si
/// no se pudo construir ni si falló al conectar (TCP, TLS o proxy, antes de
/// escribir nada); un timeout o una conexión cortada después sí pudo. El plazo
/// total que vence mientras conecta también cuenta como salida (reqwest 0.11
/// no dice en qué fase venció): en Windows una conexión rechazada tarda unos
/// 2 s en fallar, más que los 1,2 s de las órdenes, y cuenta así de más, del
/// lado seguro (fila N41 del censo).
fn salio(e: &reqwest::Error) -> bool {
    !(e.is_connect() || e.is_builder())
}

/// N24 (CL-202): el error de transporte de una petición firmada, sin su URL.
/// reqwest la escribe entera en el texto («for url (…)»), con la consulta y la
/// firma: el reintento de modo del aplanado, que busca «reduceOnly» en el
/// texto, la encontraba en la propia petición de un cierre sin respuesta, y
/// la firma acababa en los registros. Si salió se decide antes de quitarla.
fn error_de_red(prefijo: &str, e: reqwest::Error) -> ErrorRest {
    let salio = salio(&e);
    ErrorRest::de_red(format!("{}Network Error: {}", prefijo, e.without_url()), salio)
}

/// N25 (CL-199): un envío de orden que respondió pero cuyo cuerpo no se pudo
/// leer, con cualquier estado: la orden pudo quedar (AMBIGUOUS), con su
/// cabecera.
fn cuerpo_ilegible(estado: reqwest::StatusCode, limites: BinanceRateLimits) -> ErrorRest {
    ErrorRest::con_cabecera(
        format!("AMBIGUOUS: HTTP {} unreadable response body", estado.as_u16()),
        limites,
    )
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
    /// 429/418 lo registra el ejecutor (`registrar_error`, CL-196), no ellos.
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
        assert!(
            antes(c, "self.freno.comprobar_ahora().map_err(ErrorRest::sin_cabecera)?;", ".send()"),
            "cancela sin freno"
        );
        assert!(!c.contains("freno_de_respuesta(") && !c.contains(".registrar("), "registra");
    }

    /// RED antes de CL-199 (N25): los dos envíos de orden que leen la
    /// respuesta clasifican sus no-2xx con la misma regla (`error_de_orden` y
    /// `cuerpo_ilegible`), sin textos AMBIGUOUS propios.
    #[test]
    fn cl199_una_sola_clasificacion_de_los_envios_de_orden() {
        for f in ["execute_order_payload", "execute_order_payload_body"] {
            let c = cuerpo(f);
            assert!(c.contains("error_de_orden(") && c.contains("cuerpo_ilegible("), "{f}");
            assert!(!c.contains("\"AMBIGUOUS: HTTP "), "{f}: clasificación propia");
        }
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

/// N23 (CL-196): los transportes de las peticiones firmadas contra un servidor
/// HTTP en loopback (`tests_n20::respuestas` del ejecutor): sin exchange,
/// claves ni cuenta.
#[cfg(test)]
mod tests_n23 {
    use super::*;
    use crate::executor::tests_n20::{respuesta, respuestas};

    const TRANSPORTES: [&str; 6] = [
        "execute_order_payload",
        "execute_order_payload_body",
        "execute_order_payload_typed",
        "cancel_order_payload",
        "get_payload_with_timeout",
        "post_payload",
    ];

    async fn enviar(c: &BinanceClient, k: usize, url: &str) -> Result<(), ErrorRest> {
        match k {
            0 => c.execute_order_payload(url).await.map(|_| ()),
            1 => c.execute_order_payload_body(url).await.map(|_| ()),
            2 => c.execute_order_payload_typed(url).await.map(|_| ()),
            3 => c.cancel_order_payload(url).await.map(|_| ()),
            4 => c.get_payload_with_timeout(url, None).await.map(|_| ()),
            _ => c.post_payload(url).await.map(|_| ()),
        }
    }

    /// El error del transporte `k` ante esa respuesta, con un cliente nuevo.
    async fn error_ante(k: usize, wire: String) -> Option<ErrorRest> {
        let (url, _, servidor) = respuestas(vec![wire]).await?;
        let c = BinanceClient::with_timeout("k".into(), true, Duration::from_secs(5));
        let r = tokio::time::timeout(Duration::from_secs(20), enviar(&c, k, &url)).await;
        servidor.abort();
        r.ok()?.err()
    }

    /// Cada transporte lleva en su error la cabecera de uso de la respuesta:
    /// un 400, un 503 y un 2xx cuyo ACK no se lee (sólo `_typed` lo lee).
    #[tokio::test]
    async fn cl196_cada_transporte_lleva_la_cabecera_de_su_error() {
        let cab = "X-MBX-USED-WEIGHT-1M: 777\r\nX-MBX-ORDER-COUNT-10S: 3\r\n";
        let cuerpo = r#"{"code":-2019,"msg":"Margin is insufficient."}"#;
        for (k, nombre) in TRANSPORTES.iter().enumerate() {
            for estado in ["400 Bad Request", "503 Service Unavailable"] {
                let e = error_ante(k, respuesta(estado, cab, cuerpo)).await;
                assert!(e.is_some(), "{nombre} {estado}: es un error");
                let Some(e) = e else { continue };
                assert_eq!(e.limites().weight_1m, Some(777), "{nombre} {estado}: «{e}»");
                assert_eq!(e.limites().orders_10s, Some(3), "{nombre} {estado}: «{e}»");
            }
        }
        let e = error_ante(2, respuesta("200 OK", cab, "not json")).await;
        assert!(e.as_ref().is_some_and(|e| e.starts_with("AMBIGUOUS:")), "{e:?}");
        assert_eq!(e.and_then(|e| e.limites().weight_1m), Some(777));
    }

    /// El 429 y el 418 nunca llevan cabecera (manda su Retry-After) y su texto
    /// es el que reconoce `FrenoExchange::registrar`.
    #[tokio::test]
    async fn cl196_un_429_o_un_418_nunca_lleva_cabecera() {
        let cab = "X-MBX-USED-WEIGHT-1M: 2500\r\nRetry-After: 7\r\n";
        for (k, nombre) in TRANSPORTES.iter().enumerate() {
            for (estado, texto) in [
                ("429 Too Many Requests", "HTTP_429_RATE_LIMITED retry_after=7"),
                ("418 I'm a teapot", "HTTP_418_IP_BANNED retry_after=7"),
            ] {
                let e = error_ante(k, respuesta(estado, cab, "")).await;
                assert!(e.as_ref().is_some_and(|e| e.starts_with(texto)), "{nombre} {estado}: {e:?}");
                assert_eq!(e.and_then(|e| e.limites().weight_1m), None, "{nombre} {estado}");
            }
        }
    }

    /// Sin respuesta no hay cabecera. RV21-T (revisión de CL-196): el
    /// servidor acepta cada conexión y la cierra sin responder, en un puerto
    /// que sigue suyo hasta el final. Antes soltaba el puerto y apuntaba allí:
    /// otro test en paralelo podía recibirlo y responder.
    #[tokio::test]
    async fn cl196_sin_respuesta_no_hay_cabecera() {
        use tokio::io::AsyncReadExt;
        let oyente = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok();
        let direccion = oyente.as_ref().and_then(|o| o.local_addr().ok());
        assert!(direccion.is_some(), "puerto local");
        let (Some(oyente), Some(direccion)) = (oyente, direccion) else { return };
        let mudo = tokio::spawn(async move {
            while let Ok((mut s, _)) = oyente.accept().await {
                let mut peticion = [0u8; 4096];
                let _ = s.read(&mut peticion).await;
            }
        });
        let url = format!("http://{direccion}/fapi/v1/order");
        let c = BinanceClient::with_timeout("k".into(), true, Duration::from_secs(5));
        for (k, nombre) in TRANSPORTES.iter().enumerate() {
            let r = tokio::time::timeout(Duration::from_secs(20), enviar(&c, k, &url)).await;
            let e = r.ok().and_then(|r| r.err());
            assert!(e.as_ref().is_some_and(|e| e.contains("Network Error")), "{nombre}: {e:?}");
            assert_eq!(e.and_then(|e| e.limites().weight_1m), None, "{nombre}");
        }
        mudo.abort();
    }

    /// CL-201 (N24-a): `sin_respuesta` es el origen tipado de un error, no su
    /// texto. Cada transporte lo marca si la petición salió y la conexión se
    /// cerró sin respuesta, con el mismo texto de antes («Network Error»), y
    /// no lo marca si no llegó a conectar (un puerto ligado que no escucha),
    /// ante una respuesta (400, 503, 429, 418, un ACK ilegible) ni ante el
    /// rechazo local del freno.
    #[tokio::test]
    async fn cl201_sin_respuesta_es_el_origen_del_error() {
        for (k, nombre) in TRANSPORTES.iter().enumerate() {
            let e = error_ante(k, String::new()).await;
            assert!(e.as_ref().is_some_and(|e| e.contains("Network Error")), "{nombre}: {e:?}");
            assert!(e.as_ref().is_some_and(ErrorRest::sin_respuesta), "{nombre}: salió sin respuesta");
        }

        let enchufe = tokio::net::TcpSocket::new_v4().ok();
        let direccion = enchufe.as_ref().and_then(|e| {
            e.bind(std::net::SocketAddr::from(([127, 0, 0, 1], 0))).ok()?;
            e.local_addr().ok()
        });
        assert!(direccion.is_some(), "puerto ligado sin escuchar");
        if let Some(direccion) = direccion {
            let url = format!("http://{direccion}/fapi/v1/order");
            let c = BinanceClient::with_timeout("k".into(), true, Duration::from_secs(5));
            for (k, nombre) in TRANSPORTES.iter().enumerate() {
                let r = tokio::time::timeout(Duration::from_secs(20), enviar(&c, k, &url)).await;
                let e = r.ok().and_then(|r| r.err());
                assert!(e.as_ref().is_some_and(|e| e.contains("Network Error")), "{nombre}: {e:?}");
                assert!(e.as_ref().is_some_and(|e| !e.sin_respuesta()), "{nombre}: no llegó a conectar: {e:?}");
            }
        }
        drop(enchufe);

        let cab = "X-MBX-USED-WEIGHT-1M: 777\r\n";
        for (k, nombre) in TRANSPORTES.iter().enumerate() {
            for wire in [
                respuesta("400 Bad Request", cab, r#"{"code":-2019,"msg":"x"}"#),
                respuesta("503 Service Unavailable", "", ""),
                respuesta("429 Too Many Requests", "Retry-After: 7\r\n", ""),
                respuesta("418 I'm a teapot", "", ""),
            ] {
                let e = error_ante(k, wire).await;
                assert!(e.as_ref().is_some_and(|e| !e.sin_respuesta()), "{nombre}: hubo respuesta: {e:?}");
            }
        }
        let e = error_ante(2, respuesta("200 OK", cab, "not json")).await;
        assert!(e.as_ref().is_some_and(|e| e.starts_with("AMBIGUOUS:") && !e.sin_respuesta()), "{e:?}");

        let c = BinanceClient::with_timeout("k".into(), true, Duration::from_secs(5));
        let _ = c.freno().registrar("HTTP_429_RATE_LIMITED retry_after=30");
        let e = c.cancel_order_payload("http://127.0.0.1:9/fapi/v1/order").await.err();
        assert!(e.as_ref().is_some_and(|e| e.starts_with("RATE_LIMIT_COOLDOWN") && !e.sin_respuesta()), "{e:?}");
    }

    /// CL-202 (N24): el texto de un error de transporte no lleva la petición.
    /// reqwest escribe la URL entera («for url (…)», con la consulta y la
    /// firma), y el reintento de modo del aplanado busca «reduceOnly» en el
    /// texto. En los seis transportes, cerrada sin respuesta, sin respuesta a
    /// tiempo y sin conexión: ni la consulta, ni la firma, ni el destino; el
    /// prefijo de cada uno y `sin_respuesta` no cambian.
    #[tokio::test]
    async fn cl202_el_error_de_red_no_lleva_la_peticion() {
        use tokio::io::AsyncReadExt;
        const CONSULTA: &str = "?symbol=BTCUSDT&side=SELL&type=MARKET&reduceOnly=true\
                                &quantity=0.002&timestamp=1&signature=deadbeef";
        // Los tres envíos de orden marcan su red AMBIGUOUS; los demás no.
        let limpio = |k: usize, e: &ErrorRest| {
            let prefijo = if k < 3 { "AMBIGUOUS: Network Error: " } else { "Network Error: " };
            e.starts_with(prefijo)
                && !["reduceOnly", "signature", "deadbeef", "symbol=", "127.0.0.1", "/fapi/"]
                    .iter()
                    .any(|t| e.contains(t))
        };
        let c = BinanceClient::with_timeout("k".into(), true, Duration::from_millis(500));

        for (k, nombre) in TRANSPORTES.iter().enumerate() {
            let servido = respuestas(vec![String::new()]).await;
            assert!(servido.is_some(), "servidor local");
            let Some((url, _, servidor)) = servido else { return };
            let url = format!("{url}{CONSULTA}");
            let r = tokio::time::timeout(Duration::from_secs(20), enviar(&c, k, &url)).await;
            servidor.abort();
            let e = r.ok().and_then(|r| r.err());
            assert!(e.as_ref().is_some_and(|e| limpio(k, e) && e.sin_respuesta()), "{nombre} cerrada: {e:?}");
        }

        let oyente = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok();
        let direccion = oyente.as_ref().and_then(|o| o.local_addr().ok());
        assert!(direccion.is_some(), "puerto local");
        if let (Some(oyente), Some(direccion)) = (oyente, direccion) {
            let mudo = tokio::spawn(async move {
                let mut abiertas = Vec::new();
                while let Ok((mut s, _)) = oyente.accept().await {
                    let mut peticion = [0u8; 4096];
                    let _ = s.read(&mut peticion).await;
                    abiertas.push(s);
                }
            });
            let url = format!("http://{direccion}/fapi/v1/order{CONSULTA}");
            for (k, nombre) in TRANSPORTES.iter().enumerate() {
                let r = tokio::time::timeout(Duration::from_secs(20), enviar(&c, k, &url)).await;
                let e = r.ok().and_then(|r| r.err());
                assert!(e.as_ref().is_some_and(|e| limpio(k, e) && e.sin_respuesta()), "{nombre} muda: {e:?}");
            }
            mudo.abort();
        }

        let enchufe = tokio::net::TcpSocket::new_v4().ok();
        let direccion = enchufe.as_ref().and_then(|e| {
            e.bind(std::net::SocketAddr::from(([127, 0, 0, 1], 0))).ok()?;
            e.local_addr().ok()
        });
        assert!(direccion.is_some(), "puerto ligado sin escuchar");
        if let Some(direccion) = direccion {
            // Windows tarda unos 2 s en dar por rechazada la conexión: con el
            // plazo de 500 ms vencería el plazo total, que cuenta como salida
            // (`salio`). Con 5 s llega el rechazo, como en cl201.
            let c = BinanceClient::with_timeout("k".into(), true, Duration::from_secs(5));
            let url = format!("http://{direccion}/fapi/v1/order{CONSULTA}");
            for (k, nombre) in TRANSPORTES.iter().enumerate() {
                let r = tokio::time::timeout(Duration::from_secs(20), enviar(&c, k, &url)).await;
                let e = r.ok().and_then(|r| r.err());
                assert!(e.as_ref().is_some_and(|e| limpio(k, e) && !e.sin_respuesta()), "{nombre} sin conexión: {e:?}");
            }
        }
        drop(enchufe);
    }

    /// Las clases de ejecución desconocida de un envío de orden (5xx, un 3xx
    /// que reqwest no sigue, 408, -1006/-1007 y cuerpo ilegible), con la
    /// cabecera de uso de `con_peso`, y los rechazos firmes que no lo son. Los
    /// 1xx no llegan: hyper los descarta (salvo el 101).
    fn ambiguos() -> Vec<(&'static str, String)> {
        let cab = "X-MBX-USED-WEIGHT-1M: 777\r\n";
        let cortado = |estado: &str| {
            format!("HTTP/1.1 {estado}\r\n{cab}Content-Length: 100\r\nConnection: close\r\n\r\n{{\"code\"")
        };
        vec![
            ("503 -1001", respuesta("503 Service Unavailable", cab, r#"{"code":-1001,"msg":"Internal error; unable to process your request. Please try again."}"#)),
            ("502 html", respuesta("502 Bad Gateway", cab, "<html>bad gateway</html>")),
            ("500 vacío", respuesta("500 Internal Server Error", cab, "{}")),
            ("302 sin Location", respuesta("302 Found", cab, "")),
            ("408", respuesta("408 Request Timeout", cab, "")),
            ("400 -1006", respuesta("400 Bad Request", cab, r#"{"code":-1006,"msg":"x"}"#)),
            ("400 -1007", respuesta("400 Bad Request", cab, r#"{"code":-1007,"msg":"x"}"#)),
            ("400 cortado", cortado("400 Bad Request")),
            ("503 cortado", cortado("503 Service Unavailable")),
        ]
    }

    /// RED antes de CL-199 (N25): `execute_order_payload` (piernas del
    /// vigilante y de la OCO, GTX, trailing) daba «Binance API Error: …» ante
    /// un 5xx, un 408, un -1006/-1007 o un cuerpo ilegible, donde
    /// `execute_order_payload_body` (y `_typed`, que delega en él) marcan
    /// AMBIGUOUS: el mismo envío tenía dos clasificaciones. Ahora el texto de
    /// las dos es el mismo, con su cabecera.
    #[tokio::test]
    async fn cl199_los_envios_de_orden_marcan_igual() {
        for (caso, wire) in ambiguos() {
            let simple = error_ante(0, wire.clone()).await;
            let cuerpo = error_ante(1, wire.clone()).await;
            let tipado = error_ante(2, wire).await;
            for (nombre, e) in [("_payload", &simple), ("_body", &cuerpo), ("_typed", &tipado)] {
                assert!(e.as_ref().is_some_and(|e| e.starts_with("AMBIGUOUS: HTTP ")), "{caso} {nombre}: {e:?}");
                assert_eq!(e.as_ref().and_then(|e| e.limites().weight_1m), Some(777), "{caso} {nombre}");
            }
            assert_eq!(
                simple.as_ref().map(|e| e.to_string()),
                cuerpo.as_ref().map(|e| e.to_string()),
                "{caso}: una sola clasificación"
            );
        }
    }

    /// El rechazo firme de un envío de orden no cambia (verde antes y después
    /// de CL-199): lo leen `pierna_ya_existe` (-4116), el reintento -2021 del
    /// vigilante y `protection_health`.
    #[tokio::test]
    async fn cl199_el_rechazo_firme_del_envio_de_orden_no_cambia() {
        let cab = "X-MBX-USED-WEIGHT-1M: 777\r\n";
        for cuerpo in [
            r#"{"code":-2021,"msg":"Order would immediately trigger."}"#,
            r#"{"code":-4116,"msg":"ClientOrderId is duplicated."}"#,
            r#"{"code":-4164,"msg":"Order's notional must be no smaller than 5."}"#,
            "<html>forbidden</html>",
        ] {
            let estado = if cuerpo.starts_with('<') { "403 Forbidden" } else { "400 Bad Request" };
            let e = error_ante(0, respuesta(estado, cab, cuerpo)).await;
            assert_eq!(e.as_ref().map(|e| e.to_string()), Some(format!("Binance API Error: {cuerpo}")));
            assert_eq!(e.and_then(|e| e.limites().weight_1m), Some(777), "{cuerpo}");
        }
    }

    /// RED antes de CL-199 (N25): con el error REAL de `execute_order_payload`
    /// (el de `place_algo_leg`), un 5xx con código JSON (-1001, -1008) de una
    /// pierna del vigilante contaba como rechazo del exchange
    /// (`protection_health::note_rejection`) y alimentaba la escalada B3.5b a
    /// cierre de emergencia, contra su contrato («un fallo de red o estado
    /// transitorio no escala»); y `error_cierra_la_intencion` cerraba una
    /// intención que pudo ejecutarse. El rechazo firme sigue contando.
    #[tokio::test]
    async fn cl199_un_5xx_de_una_pierna_no_cuenta_como_rechazo() {
        for estado in ["500 Internal Server Error", "502 Bad Gateway", "503 Service Unavailable", "504 Gateway Timeout"] {
            for codigo in [-1001, -1008] {
                let cuerpo = format!(r#"{{"code":{codigo},"msg":"x"}}"#);
                let e = error_ante(0, respuesta(estado, "", &cuerpo)).await;
                assert!(e.is_some(), "{estado} {codigo}");
                let Some(e) = e else { continue };
                let simbolo = format!("CL199_{}_{}_USDT", &estado[..3], -codigo);
                let antes = quantum_arena::protection_health::rejections_of(&simbolo);
                assert!(!quantum_arena::protection_health::note_rejection(&simbolo, &e), "{estado} {codigo}: «{e}»");
                assert_eq!(quantum_arena::protection_health::rejections_of(&simbolo), antes);
                assert!(!crate::ioc_evidence::error_cierra_la_intencion(&e), "{estado} {codigo}: «{e}»");
            }
        }
        let e = error_ante(0, respuesta("400 Bad Request", "", r#"{"code":-4164,"msg":"x"}"#)).await;
        assert!(e.as_ref().is_some_and(|e| quantum_arena::protection_health::note_rejection("CL199_FIRME_USDT", e)), "{e:?}");
        assert!(e.is_some_and(|e| crate::ioc_evidence::error_cierra_la_intencion(&e)));
    }

    /// RV21-T (revisión de CL-196): las dos ramas AMBIGUOUS de
    /// `execute_order_payload_body` que no son 5xx también llevan la cabecera:
    /// un 4xx de ejecución desconocida (408, -1006, -1007) y un cuerpo que no
    /// se puede leer (se corta antes de su Content-Length). Ninguna prueba las
    /// recorría: devolverlas sin cabecera quedaba en verde.
    #[tokio::test]
    async fn rv21_las_ramas_ambiguas_del_cuerpo_llevan_la_cabecera() {
        let cab = "X-MBX-USED-WEIGHT-1M: 777\r\n";
        let cortado = format!(
            "HTTP/1.1 400 Bad Request\r\n{cab}Content-Length: 100\r\nConnection: close\r\n\r\n{{\"code\""
        );
        for (wire, texto) in [
            (respuesta("408 Request Timeout", cab, ""), "AMBIGUOUS: HTTP 408 body="),
            (respuesta("400 Bad Request", cab, r#"{"code":-1006,"msg":"x"}"#), "AMBIGUOUS: HTTP 400 body="),
            (respuesta("400 Bad Request", cab, r#"{"code":-1007,"msg":"x"}"#), "AMBIGUOUS: HTTP 400 body="),
            (cortado, "AMBIGUOUS: HTTP 400 unreadable response body"),
        ] {
            let e = error_ante(1, wire).await;
            assert!(e.as_ref().is_some_and(|e| e.starts_with(texto)), "{texto}: {e:?}");
            assert_eq!(e.and_then(|e| e.limites().weight_1m), Some(777), "{texto}");
        }
    }

    /// RED antes de CL-203 (N40): un 2xx cuyo cuerpo se corta antes de su
    /// Content-Length, o se queda a medias hasta que vence el plazo, salía de
    /// `get_payload_with_timeout` como Ok con «TEXT_ERR: …». Es un error con la
    /// cabecera de uso y la causa, sin la URL, que no es AMBIGUOUS (una
    /// lectura no ejecuta nada) ni cuenta como sin respuesta (la hubo). Un 2xx
    /// completo sigue siendo Ok con su cuerpo.
    #[tokio::test]
    async fn cl203_un_2xx_ilegible_de_lectura_es_un_error() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let cab = "X-MBX-USED-WEIGHT-1M: 777\r\n";
        let limpio = |e: &ErrorRest| {
            e.starts_with("HTTP 200 unreadable response body: ")
                && !e.sin_respuesta()
                && e.limites().weight_1m == Some(777)
                && !["127.0.0.1", "/fapi/", "signature", "deadbeef"].iter().any(|t| e.contains(t))
        };
        let cortado = format!("HTTP/1.1 200 OK\r\n{cab}Content-Length: 100\r\nConnection: close\r\n\r\n[{{\"x\"");
        let e = error_ante(4, cortado).await;
        assert!(e.as_ref().is_some_and(limpio), "cortado: {e:?}");

        let oyente = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok();
        let direccion = oyente.as_ref().and_then(|o| o.local_addr().ok());
        assert!(direccion.is_some(), "puerto local");
        if let (Some(oyente), Some(direccion)) = (oyente, direccion) {
            let a_medias = tokio::spawn(async move {
                let mut abiertas = Vec::new();
                while let Ok((mut s, _)) = oyente.accept().await {
                    let mut peticion = [0u8; 4096];
                    let _ = s.read(&mut peticion).await;
                    let _ = s.write_all(format!("HTTP/1.1 200 OK\r\n{cab}Content-Length: 100\r\n\r\n[").as_bytes()).await;
                    abiertas.push(s);
                }
            });
            let c = BinanceClient::with_timeout("k".into(), true, Duration::from_secs(5));
            let url = format!("http://{direccion}/fapi/v2/positionRisk?timestamp=1&signature=deadbeef");
            let r = tokio::time::timeout(
                Duration::from_secs(20),
                // RV24-T: 3 s para que la cabecera llegue dentro del plazo
                // aunque wine se pare; el cuerpo no termina nunca.
                c.get_payload_with_timeout(&url, Some(Duration::from_secs(3))),
            )
            .await;
            a_medias.abort();
            let e = r.ok().and_then(|r| r.err());
            assert!(e.as_ref().is_some_and(|e| limpio(e) && e.contains("timed out")), "a medias: {e:?}");
        }

        let servido = respuestas(vec![respuesta("200 OK", cab, "[]")]).await;
        assert!(servido.is_some(), "servidor local");
        let Some((url, _, servidor)) = servido else { return };
        let c = BinanceClient::with_timeout("k".into(), true, Duration::from_secs(5));
        let r = tokio::time::timeout(Duration::from_secs(20), c.get_payload_with_timeout(&url, None)).await;
        servidor.abort();
        assert!(matches!(r, Ok(Ok((ref l, ref cuerpo))) if cuerpo == "[]" && l.weight_1m == Some(777)), "{r:?}");
    }
}
