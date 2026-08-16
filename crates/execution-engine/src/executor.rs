use crate::binance_api::{
    sign_payload_to_buffer, ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL,
    TIME_IN_FORCE_IOC,
};
use crate::order_types::OrderAck;
use crate::ExecutionPayload;
use risk_engine::ValidatedOrder;
use signal_engine::SignalType;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::client::{BinanceClient, ZeroAllocBuffer};

#[derive(Debug, Clone)]
pub struct ActivePosition {
    pub symbol: String,
    pub qty: f64,
    pub entry_price: f64,
    pub is_long: bool,
}

#[allow(async_fn_in_trait)]
pub trait ExecutionProvider: Send + Sync {
    async fn execute_order(
        &self,
        order: &ValidatedOrder,
        symbol: &str,
        current_price: f64,
        step_size: f64,
    ) -> Result<(), String>;

    async fn execute_raw_qty(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        step_size: f64,
    ) -> Result<(), String>;

    async fn execute_limit_order(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String>;

    async fn execute_maker_chase(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String>;

    /// FASE 8: Immediate-Or-Cancel. Liquidity snipe without exposing to the book.
    async fn execute_ioc_order(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String>;

    /// FASE 22: Advanced API Exploitation - Iceberg Limit Orders
    /// Oculta volumen real dividiéndolo en icebergQty para evitar ser "cazado" por HFTs institucionales.
    async fn execute_iceberg_limit(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        iceberg_qty: f64,
        price: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String>;

    /// FASE 13: Reduce-Only Market. Perfectly safe position closing.
    async fn execute_reduce_only_market(
        &self,
        symbol: &str,
        is_long_close: bool, // true if closing a long (so side=SELL)
        quantity: f64,
        step_size: f64,
    ) -> Result<(), String>;

    /// FASE 8: Native Exchange Trailing Stop Market
    async fn execute_exchange_trailing_stop(
        &self,
        symbol: &str,
        is_long: bool, // is_long=true means Buy to close a Short
        quantity: f64,
        activation_price: f64,
        callback_rate: f64, // 0.1 to 5.0 (%)
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String>;

    /// FASE 12: Parallel OCO Tensor Execution
    /// En Binance Futuros, OCO no existe nativamente con un solo endpoint como en Spot.
    /// Se simula enviando simultáneamente un STOP_MARKET y un LIMIT (o TAKE_PROFIT_MARKET)
    /// ambos con reduceOnly=true. El motor interno cancela la otra al llenarse una.
    async fn execute_oco_order(
        &self,
        symbol: &str,
        is_long: bool, // true if closing a long
        quantity: f64,
        take_profit_price: f64,
        stop_loss_price: f64,
        step_size: f64,
        tick_size: f64,
        base_client_id: &str,
    ) -> Result<(), String>;

    async fn cancel_order(&self, symbol: &str, client_order_id: &str) -> Result<(), String>;

    /// F1.2/F1.3: consulta el estado REAL de una orden por su clientOrderId.
    /// Fuente de verdad para: resolver timeouts ambiguos, calcular el remanente
    /// tras un cancel en maker-chase, y reconciliación.
    async fn query_order(&self, symbol: &str, client_order_id: &str) -> Result<OrderAck, String> {
        let _ = (symbol, client_order_id);
        Err("query_order no implementado en este provider".to_string())
    }

    async fn fetch_open_positions(&self) -> Result<Vec<ActivePosition>, String>;

    async fn fetch_server_time(&self) -> Result<i64, String>;

    async fn fetch_account_balance(&self) -> Result<f64, String>;

    async fn set_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String>;

    async fn fetch_commission_rate(&self, symbol: &str) -> Result<(f64, f64), String>;

    async fn fetch_exchange_info(&self, symbol: &str) -> Result<f64, String>;

    fn trigger_kill_switch(&self);
}

pub struct OrderExecutor {
    api_secret: std::sync::RwLock<String>,
    client: BinanceClient,
    rate_limit_counter: AtomicUsize,
    last_reset_timestamp: AtomicU64,
    binance_weight_1m: AtomicUsize,
    binance_orders_10s: AtomicUsize,
    binance_orders_1m: AtomicUsize,
    max_weight_1m: AtomicUsize,
    max_orders_10s: AtomicUsize,
    max_orders_1m: AtomicUsize,
    kill_switch: AtomicBool,
    active_leverage: std::sync::RwLock<std::collections::HashMap<String, u32>>,
    is_paper_trading: bool,
    /// F1.5: memoria del ciclo de vida de órdenes — compartida con el
    /// user-data stream (F1.6) y la reconciliación (F1.7).
    order_registry: std::sync::Arc<crate::order_registry::OrderRegistry>,
    /// F1.8: no enviar órdenes hasta este timestamp (cooldown post-429).
    cooldown_until_ms: AtomicU64,
    /// F1.8: 429s consecutivos — >=3 sugiere ban inminente → kill-switch real.
    consecutive_429: AtomicUsize,
}

impl OrderExecutor {
    pub fn new(api_key: String, api_secret: String, is_testnet: bool) -> Self {
        Self {
            api_secret: std::sync::RwLock::new(api_secret),
            client: BinanceClient::new(api_key, is_testnet),
            rate_limit_counter: AtomicUsize::new(0),
            last_reset_timestamp: AtomicU64::new(0),
            binance_weight_1m: AtomicUsize::new(0),
            binance_orders_10s: AtomicUsize::new(0),
            binance_orders_1m: AtomicUsize::new(0),
            max_weight_1m: AtomicUsize::new(2200), // Default, but can be updated
            max_orders_10s: AtomicUsize::new(280),
            max_orders_1m: AtomicUsize::new(1100),
            kill_switch: AtomicBool::new(false),
            active_leverage: std::sync::RwLock::new(std::collections::HashMap::new()),
            is_paper_trading: is_testnet, // Initially mapped to is_testnet, will be overridden by PhaseOrchestrator if in PaperTrading mode
            order_registry: std::sync::Arc::new(crate::order_registry::OrderRegistry::new()),
            cooldown_until_ms: AtomicU64::new(0),
            consecutive_429: AtomicUsize::new(0),
        }
    }

    /// Registro de órdenes (F1.5) — para spawn del user-data stream y queries.
    pub fn registry(&self) -> std::sync::Arc<crate::order_registry::OrderRegistry> {
        self.order_registry.clone()
    }

    pub fn client(&self) -> &BinanceClient {
        &self.client
    }

    /// F1.7: GET /fapi/v2/positionRisk — posiciones abiertas según el EXCHANGE.
    /// Fuente de verdad para reconciliación al arranque y periódica.
    pub async fn fetch_position_risk(
        &self,
    ) -> Result<Vec<crate::reconciliation::PositionRiskEntry>, String> {
        if self.is_paper_trading {
            return Ok(Vec::new());
        }
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(self.client.get_base_url());
        buf.push_str("/fapi/v2/positionRisk?");
        let payload_start = buf.as_str().len();
        buf.push_str("timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.get_payload(buf.as_str()).await;
        match res {
            Ok((limits, body)) => {
                self.update_limits(&limits);
                serde_json::from_str(&body).map_err(|e| {
                    format!(
                        "POSITION_RISK_PARSE: {} body={}",
                        e,
                        crate::order_types::truncate(&body, 200)
                    )
                })
            }
            Err(e) => Err(e),
        }
    }

    pub fn set_rate_limit_thresholds(&self, weight_1m: usize, orders_10s: usize, orders_1m: usize) {
        self.max_weight_1m.store(weight_1m, Ordering::Relaxed);
        self.max_orders_10s.store(orders_10s, Ordering::Relaxed);
        self.max_orders_1m.store(orders_1m, Ordering::Relaxed);
    }

    pub fn set_paper_trading(&mut self, is_paper: bool) {
        self.is_paper_trading = is_paper;
    }

    pub fn hot_swap_credentials(&self, new_key: String, new_secret: String, is_testnet: bool) {
        if let Ok(mut secret) = self.api_secret.write() {
            *secret = new_secret;
        }
        self.client.hot_swap_credentials(new_key, is_testnet);
    }

    #[inline(always)]
    fn update_limits(&self, limits: &crate::client::BinanceRateLimits) {
        if let Some(w) = limits.weight_1m {
            self.binance_weight_1m.store(w, Ordering::Relaxed);
        }
        if let Some(o) = limits.orders_10s {
            self.binance_orders_10s.store(o, Ordering::Relaxed);
        }
        if let Some(o) = limits.orders_1m {
            self.binance_orders_1m.store(o, Ordering::Relaxed);
        }
        // Respuesta exitosa = la ventana de rate limit está sana.
        self.consecutive_429.store(0, Ordering::Relaxed);
    }

    /// F1.8: política centralizada ante rate limits de Binance.
    /// - 429: cooldown temporal (Retry-After o 60s default); kill-switch SOLO
    ///   si se acumulan 3+ consecutivos (patrón de ban inminente).
    /// - 418: IP baneada → kill-switch inmediato y legítimo.
    /// Antes: un SOLO 429 disparaba el kill-switch permanente — el sistema
    /// quedaba muerto hasta reinicio por un freno transitorio del exchange.
    fn handle_rate_limit_error(&self, e: &str) -> String {
        if e.starts_with("HTTP_429_RATE_LIMITED") {
            let retry_after_s: u64 = e
                .split("retry_after=")
                .nth(1)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(60);
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
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
                self.trigger_kill_switch();
                println!("🚨 [KILL SWITCH] 3+ rate limits consecutivos — freno total preventivo.");
            }
        } else if e.starts_with("HTTP_418_IP_BANNED") {
            self.trigger_kill_switch();
            println!("🚨 [KILL SWITCH] HTTP 418: IP baneada por Binance. Freno total.");
        }
        e.to_string()
    }

    #[inline(always)]
    fn check_rate_limits(&self, timestamp_ms: u64) -> Result<(), String> {
        if self.kill_switch.load(Ordering::Relaxed) {
            return Err("KILL SWITCH ACTIVE. Execution blocked.".to_string());
        }

        // F1.8: cooldown post-429 — Binance ya nos frenó; respetar la ventana.
        if timestamp_ms < self.cooldown_until_ms.load(Ordering::Relaxed) {
            return Err(format!(
                "RATE_LIMIT_COOLDOWN: activo hasta {} (ahora {})",
                self.cooldown_until_ms.load(Ordering::Relaxed),
                timestamp_ms
            ));
        }

        let bw1m = self.binance_weight_1m.load(Ordering::Relaxed);
        let bo10s = self.binance_orders_10s.load(Ordering::Relaxed);
        let bo1m = self.binance_orders_1m.load(Ordering::Relaxed);

        let max_bw1m = self.max_weight_1m.load(Ordering::Relaxed) as f64;
        let max_bo10s = self.max_orders_10s.load(Ordering::Relaxed) as f64;
        let max_bo1m = self.max_orders_1m.load(Ordering::Relaxed) as f64;

        // F1.8: HEADROOM 80% — los contadores de Binance llegan POR RESPUESTA;
        // frenar al 100% ya es tarde (requests en vuelo cruzan el límite).
        // Constante de infraestructura justificada, no parámetro de estrategia.
        const HEADROOM: f64 = 0.8;
        if bw1m as f64 > max_bw1m * HEADROOM {
            return Err(format!(
                "RATE_LIMIT_HEADROOM_WEIGHT: {}/{}",
                bw1m, max_bw1m as usize
            ));
        }
        if bo10s as f64 > max_bo10s * HEADROOM {
            return Err(format!(
                "RATE_LIMIT_HEADROOM_ORDERS_10S: {}/{}",
                bo10s, max_bo10s as usize
            ));
        }
        if bo1m as f64 > max_bo1m * HEADROOM {
            return Err(format!(
                "RATE_LIMIT_HEADROOM_ORDERS_1M: {}/{}",
                bo1m, max_bo1m as usize
            ));
        }

        let current_sec = timestamp_ms / 1000;
        let last_reset = self.last_reset_timestamp.load(Ordering::Relaxed);

        if current_sec > last_reset {
            self.last_reset_timestamp
                .store(current_sec, Ordering::Relaxed);
            self.rate_limit_counter.store(1, Ordering::Relaxed);
        } else {
            let ops = self.rate_limit_counter.fetch_add(1, Ordering::Relaxed);
            if ops > 20 {
                return Err("LOCAL RATE LIMIT EXCEEDED. Throttling execution.".to_string());
            }
        }
        Ok(())
    }

    /// Redondea la cantidad a los decimales permitidos (step_size).
    #[inline(always)]
    fn round_to_step_size(quantity: f64, step_size: f64) -> f64 {
        let inv = 1.0 / step_size;
        (quantity * inv).floor() / inv
    }

    /// Toma la orden validada por el Risk Engine, calcula el lote de cripto exacto
    /// basado en el precio actual, y construye el payload firmado para enviar a la API.
    pub fn build_payload(
        &self,
        order: &ValidatedOrder,
        symbol: &str,
        current_price: f64,
        step_size: f64,
        tick_size: f64,
    ) -> Option<ExecutionPayload> {
        if order.volume_usd <= 0.0 || current_price <= 0.0 || tick_size <= 0.0 {
            return None;
        }

        // FASE 21: Ensure volume accounts for fees margin safety buffer dynamically via Genome
        // We add the EV fee buffer multiplier to the required volume to ensure it never hits the $5 limit due to fees/slippage
        let raw_quantity =
            (order.volume_usd * order.leverage * order.fee_buffer_multiplier) / current_price;
        let final_quantity = Self::round_to_step_size(raw_quantity, step_size);

        if final_quantity == 0.0 {
            return None;
        }

        let side = match order.signal {
            SignalType::Long => SIDE_BUY,
            SignalType::Short => SIDE_SELL,
            SignalType::Flat => return None,
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // F1.2: identificador idempotente — toda orden lleva newClientOrderId.
        let client_order_id = uuid::Uuid::now_v7().simple().to_string();

        // FASE 19 + F1.4: Órdenes Institucionales y Evasión de Taker Fees.
        // El maker (POST_ONLY) exige timeInForce=GTX Y price en el MISMO query
        // que se firma y se envía. Precio redondeado al tickSize REAL del símbolo
        // (nunca un {:.4} fijo que viola el filtro PRICE_FILTER).
        let (order_type, time_in_force, extra_params) = if order.maker_only {
            let final_price = Self::round_to_step_size(current_price, tick_size);
            (
                ORDER_TYPE_LIMIT,
                crate::binance_api::TIME_IN_FORCE_GTX,
                format!("&timeInForce=GTX&price={}", final_price),
            )
        } else {
            (ORDER_TYPE_MARKET, TIME_IN_FORCE_IOC, String::new())
        };

        // F1.4: query EXACTA que se firma = query EXACTA que se envía.
        let signed_query = format!(
            "symbol={}&side={}&positionSide={}&type={}&quantity={}{}&newClientOrderId={}&timestamp={}",
            symbol,
            side,
            if side == SIDE_BUY { "LONG" } else { "SHORT" },
            order_type,
            final_quantity,
            extra_params,
            client_order_id,
            timestamp
        );

        // Firmar
        let mut sig_buf = [0u8; 64];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(&signed_query, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) }.to_string();

        Some(ExecutionPayload {
            symbol: symbol.to_string(),
            side: side.to_string(),
            quantity: final_quantity,
            order_type: order_type.to_string(),
            time_in_force: time_in_force.to_string(),
            position_side: if side == SIDE_BUY {
                "LONG".to_string()
            } else {
                "SHORT".to_string()
            },
            signed_query,
            client_order_id,
            signature,
            timestamp,
        })
    }
}

impl ExecutionProvider for OrderExecutor {
    fn trigger_kill_switch(&self) {
        self.kill_switch
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Despacha la orden a Binance usando el cliente HTTP hiper-optimizado.
    /// Retorna Ok si se ejecutó correctamente.
    #[inline(always)]
    async fn execute_order(
        &self,
        order: &ValidatedOrder,
        symbol: &str,
        current_price: f64,
        step_size: f64,
    ) -> Result<(), String> {
        if current_price <= 0.0 {
            return Err("SEGURIDAD: current_price inválido (<= 0.0). Orden abortada.".to_string());
        }

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Ejecutando orden de {:?} para {}. No se envió a Testnet API para evitar divergencia de latencia.", order.signal, symbol);
            return Ok(());
        }

        // Parametrizar dinámicamente el Leverage (Kelly Criterion)
        let target_leverage = order.leverage as u32;
        let needs_update = {
            let cache = self.active_leverage.read().unwrap();
            cache.get(symbol).copied().unwrap_or(0) != target_leverage
        };

        if needs_update {
            if let Err(e) = self.set_leverage(symbol, target_leverage).await {
                println!(
                    "⚠️ [EXECUTION] Failed to set dynamic leverage for {}: {}",
                    symbol, e
                );
            } else {
                let mut cache = self.active_leverage.write().unwrap();
                cache.insert(symbol.to_string(), target_leverage);
            }
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.check_rate_limits(timestamp)?;

        // F1.4: tickSize real del símbolo para el camino maker.
        let tick_size = self.fetch_exchange_info(symbol).await.unwrap_or(0.01);

        if let Some(payload) =
            self.build_payload(order, symbol, current_price, step_size, tick_size)
        {
            // F1.5: registrar la intención ANTES del envío (si la red muere, la
            // orden queda referenciada por clientOrderId para resolución).
            self.order_registry.register_intent(
                &payload.client_order_id,
                symbol,
                &payload.side,
                &payload.position_side,
                &payload.order_type,
                payload.quantity,
                payload.timestamp,
            );
            // F1.4: se envía EXACTAMENTE la query firmada. Sin reconstrucción.
            let mut buf = ZeroAllocBuffer::new();
            buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
                "https://testnet.binancefuture.com/fapi/v1/order?"
            } else {
                "https://fapi.binance.com/fapi/v1/order?"
            });
            buf.push_str(&payload.signed_query);
            buf.push_str("&signature=");
            buf.push_str(&payload.signature);

            if buf.is_overflow() {
                return Err(
                    "SEGURIDAD: query de orden excede el buffer (orden abortada)".to_string(),
                );
            }

            let res = self.client.execute_order_payload_typed(buf.as_str()).await;
            match &res {
                Ok((limits, ack)) => {
                    self.update_limits(limits);
                    self.order_registry.apply_ack(ack, payload.timestamp);
                    if !ack.is_filled() && !ack.is_active() && !ack.status.is_empty() {
                        println!(
                            "⚠️ [EXECUTION] Orden {} estado final inesperado: {} (executed={})",
                            payload.client_order_id, ack.status, ack.executed_qty
                        );
                    }
                    Ok(())
                }
                Err(e) => {
                    if e.starts_with("HTTP_429") || e.starts_with("HTTP_418") {
                        return Err(self.handle_rate_limit_error(e));
                    }
                    // F1.2: error ambiguo (timeout/5xx) — la orden PUEDE existir.
                    // Consultar por clientOrderId ANTES de reportar error: jamás duplicar.
                    if e.starts_with("AMBIGUOUS") {
                        println!(
                            "⏳ [EXECUTION] Timeout ambiguo para {}. Consultando estado real...",
                            payload.client_order_id
                        );
                        if let Ok(ack) = self.query_order(symbol, &payload.client_order_id).await {
                            if ack.is_active() {
                                let _ = self.cancel_order(symbol, &payload.client_order_id).await;
                                println!("🛡️ [EXECUTION] Orden ambigua {} estaba VIVA ({} ejecutado) — cancelada. Sin duplicación.", payload.client_order_id, ack.executed_qty);
                            } else {
                                println!(
                                    "🛡️ [EXECUTION] Orden ambigua {} resuelta: {} (executed={})",
                                    payload.client_order_id, ack.status, ack.executed_qty
                                );
                            }
                        }
                    }
                    Err(e.clone())
                }
            }
        } else {
            Err("No se pudo construir el payload (Volumen 0 o precio inválido)".to_string())
        }
    }

    /// Despacha una orden raw directamente con la cantidad final de crypto pre-calculada.
    /// Utilizado por el GodEngineCore unificado.
    #[inline(always)]
    async fn execute_raw_qty(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        step_size: f64,
    ) -> Result<(), String> {
        if quantity.is_infinite() || quantity.is_nan() || quantity <= 0.0 {
            return Err(
                "SEGURIDAD: quantity inválido (infinito, NaN o <= 0.0). Orden abortada."
                    .to_string(),
            );
        }
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if final_quantity == 0.0 {
            return Err("Volumen 0 despues de round_to_step_size".to_string());
        }

        if self.is_paper_trading {
            println!(
                "📝 [PAPER TRADING LOCAL] Ejecutando raw_qty {} para {}. Cero latencia simulada.",
                final_quantity, symbol
            );
            return Ok(());
        }

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        // F1.2: toda orden market lleva newClientOrderId (idempotencia).
        let client_order_id = uuid::Uuid::now_v7().simple().to_string();
        // F1.5: registrar la intención antes del envío.
        self.order_registry.register_intent(
            &client_order_id,
            symbol,
            side,
            if is_long { "LONG" } else { "SHORT" },
            ORDER_TYPE_MARKET,
            final_quantity,
            timestamp,
        );

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        });
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        buf.push_str("&positionSide=");
        buf.push_str(if is_long { "LONG" } else { "SHORT" });
        buf.push_str("&type=");
        buf.push_str(ORDER_TYPE_MARKET);
        buf.push_str("&quantity=");
        buf.push_f64(final_quantity);
        buf.push_str("&newClientOrderId=");
        buf.push_str(&client_order_id);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        if buf.is_overflow() {
            return Err("SEGURIDAD: query de orden excede el buffer (orden abortada)".to_string());
        }

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.execute_order_payload_typed(buf.as_str()).await;
        match &res {
            Ok((limits, ack)) => {
                self.update_limits(limits);
                self.order_registry.apply_ack(ack, timestamp);
                Ok(())
            }
            Err(e) => {
                if e.starts_with("HTTP_429") || e.starts_with("HTTP_418") {
                    return Err(self.handle_rate_limit_error(e));
                }
                Err(e.clone())
            }
        }
    }

    /// Despacha una orden LIMIT para Market Making.
    #[inline(always)]
    async fn execute_limit_order(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if final_quantity == 0.0 {
            return Err("Volumen 0 despues de round_to_step_size".to_string());
        }

        let final_price = Self::round_to_step_size(price, tick_size);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Ejecutando orden LIMIT de {} para {} @ {}. Cero latencia simulada.", final_quantity, symbol, final_price);
            return Ok(());
        }

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        });
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        buf.push_str("&positionSide=");
        buf.push_str(if is_long { "LONG" } else { "SHORT" });
        buf.push_str("&type=");
        buf.push_str(ORDER_TYPE_LIMIT);
        buf.push_str("&timeInForce=");
        buf.push_str(crate::binance_api::TIME_IN_FORCE_GTX);
        buf.push_str("&quantity=");
        buf.push_f64(final_quantity);
        buf.push_str("&price=");
        buf.push_f64(final_price);
        buf.push_str("&newClientOrderId=");
        buf.push_str(client_order_id);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        if buf.is_overflow() {
            return Err("SEGURIDAD: query de orden excede el buffer (orden abortada)".to_string());
        }

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.execute_order_payload(buf.as_str()).await;
        match &res {
            Ok(limits) => {
                self.update_limits(limits);
                Ok(())
            }
            Err(e) => {
                if e.starts_with("HTTP_429") || e.starts_with("HTTP_418") {
                    return Err(self.handle_rate_limit_error(e));
                }
                Err(e.clone())
            }
        }
    }

    /// Implementa el algoritmo Maker-Chase: Coloca orden Límite Maker (GTX), espera 50ms,
    /// si no se llena (o por simplicidad, la cancela preventivamente), y cae a orden Market (Taker).
    #[inline(always)]
    async fn execute_maker_chase(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        if self.is_paper_trading {
            println!(
                "📝 [PAPER TRADING LOCAL] Ejecutando MAKER_CHASE para {}. Cero latencia simulada.",
                symbol
            );
            return Ok(());
        }

        // 1. Intentar colocar la orden Límite Maker
        let maker_res = self
            .execute_limit_order(
                symbol,
                is_long,
                quantity,
                price,
                step_size,
                tick_size,
                client_order_id,
            )
            .await;
        if maker_res.is_err() {
            // Si la orden GTX es rechazada (ej. cruza el libro inmediatamente), caemos a Taker.
            return self
                .execute_raw_qty(symbol, is_long, quantity, step_size)
                .await;
        }

        // 2. Esperar 50ms (Tolerancia de Latencia Cuántica)
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // 3. Cancelar la orden límite. Si ya se llenó, Binance responde con
        //    error benigno — el fill YA ocurrió y no se puede deshacer.
        let _ = self.cancel_order(symbol, client_order_id).await;

        // F1.3 — FIX DOUBLE-FILL: consultar el estado REAL de la orden tras el
        // cancel y mercadear ÚNICAMENTE el remanente (origQty - executedQty).
        // Antes: se mercadeaba la cantidad completa aunque el límite se hubiera
        // llenado parcial o totalmente → posición duplicada (bug crítico de auditoría).
        let executed = match self.query_order(symbol, client_order_id).await {
            Ok(ack) => ack.executed_qty,
            Err(e) => {
                // Sin estado verificable NO se mercadea nada: a ciegas es el bug
                // original. El remanente se materializa vía reconciliación (F1.7).
                println!("🛑 [MAKER-CHASE] No se pudo verificar estado de {} ({}). Abortando chase SIN market de respaldo para evitar doble-fill.", client_order_id, e);
                return Err(format!("MAKER_CHASE_UNVERIFIED: {}", e));
            }
        };

        let remaining = Self::round_to_step_size((quantity - executed).max(0.0), step_size);
        if executed > 0.0 {
            println!(
                "🛡️ [MAKER-CHASE] {} maker ejecutado {} de {}. Remanente: {}",
                symbol, executed, quantity, remaining
            );
        }
        if remaining <= 0.0 || remaining < step_size {
            // Total o casi total lleno como maker: nada que completar.
            return Ok(());
        }

        // 4. Ejecutar como Taker SOLO el remanente (Market Order)
        self.execute_raw_qty(symbol, is_long, remaining, step_size)
            .await
    }

    /// FASE 8: Immediate-Or-Cancel. Intenta llenar limit; si no puede, se cancela automáticamente por Binance.
    #[inline(always)]
    async fn execute_ioc_order(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if final_quantity == 0.0 {
            return Err("Volumen 0".to_string());
        }
        let final_price = Self::round_to_step_size(price, tick_size);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Ejecutando orden IOC de {} para {} @ {}. Cero latencia simulada.", final_quantity, symbol, final_price);
            return Ok(());
        }

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        });
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        buf.push_str("&positionSide=");
        buf.push_str(if is_long { "LONG" } else { "SHORT" });
        buf.push_str("&positionSide=");
        buf.push_str(if is_long { "LONG" } else { "SHORT" });
        buf.push_str("&type=");
        buf.push_str(ORDER_TYPE_LIMIT);
        buf.push_str("&timeInForce=");
        buf.push_str(TIME_IN_FORCE_IOC);
        buf.push_str("&quantity=");
        buf.push_f64(final_quantity);
        buf.push_str("&price=");
        buf.push_f64(final_price);
        buf.push_str("&newClientOrderId=");
        buf.push_str(client_order_id);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.execute_order_payload(buf.as_str()).await;
        if let Ok(limits) = &res {
            self.update_limits(limits);
        }
        res.map(|_| ())
    }

    /// FASE 22: Advanced API Exploitation - Iceberg Limit Orders
    #[inline(always)]
    async fn execute_iceberg_limit(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        iceberg_qty: f64,
        price: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if final_quantity == 0.0 {
            return Err("Volumen 0".to_string());
        }

        let final_iceberg_qty = Self::round_to_step_size(iceberg_qty, step_size);
        if final_iceberg_qty == 0.0 {
            return Err("Iceberg Volumen 0".to_string());
        }

        let final_price = Self::round_to_step_size(price, tick_size);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Ejecutando orden ICEBERG_LIMIT de {} (Iceberg: {}) para {} @ {}. Cero latencia simulada.", final_quantity, final_iceberg_qty, symbol, final_price);
            return Ok(());
        }

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        });
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        buf.push_str("&positionSide=");
        buf.push_str(if is_long { "LONG" } else { "SHORT" });
        buf.push_str("&positionSide=");
        buf.push_str(if is_long { "LONG" } else { "SHORT" });
        buf.push_str("&type=");
        buf.push_str(ORDER_TYPE_LIMIT);
        buf.push_str("&timeInForce=");
        buf.push_str(crate::binance_api::TIME_IN_FORCE_GTC); // Iceberg requiere GTC
        buf.push_str("&quantity=");
        buf.push_f64(final_quantity);
        buf.push_str("&icebergQty=");
        buf.push_f64(final_iceberg_qty);
        buf.push_str("&price=");
        buf.push_f64(final_price);
        buf.push_str("&newClientOrderId=");
        buf.push_str(client_order_id);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.execute_order_payload(buf.as_str()).await;
        if let Ok(limits) = &res {
            self.update_limits(limits);
        }
        res.map(|_| ())
    }

    /// FASE 13: Reduce-Only Market. Guaranteed to only reduce position, never flip.
    #[inline(always)]
    async fn execute_reduce_only_market(
        &self,
        symbol: &str,
        is_long_close: bool,
        quantity: f64,
        step_size: f64,
    ) -> Result<(), String> {
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if final_quantity <= 0.0 {
            return Err("Volumen 0 despues de round_to_step_size".to_string());
        }

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Reduciendo posicion de {} para {}. Cero latencia simulada.", final_quantity, symbol);
            return Ok(());
        }

        let side = if is_long_close { SIDE_SELL } else { SIDE_BUY };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        });
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        buf.push_str("&positionSide=");
        buf.push_str(if is_long_close { "LONG" } else { "SHORT" });
        buf.push_str("&positionSide=");
        buf.push_str(if is_long_close { "LONG" } else { "SHORT" });
        buf.push_str("&type=");
        buf.push_str(ORDER_TYPE_MARKET);
        buf.push_str("&reduceOnly=true");
        buf.push_str("&quantity=");
        buf.push_f64(final_quantity);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.execute_order_payload(buf.as_str()).await;
        if let Ok(limits) = &res {
            self.update_limits(limits);
        }
        res.map(|_| ())
    }

    /// FASE 8: Native Exchange Trailing Stop Market
    #[inline(always)]
    async fn execute_exchange_trailing_stop(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        activation_price: f64,
        callback_rate: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if final_quantity == 0.0 {
            return Err("Volumen 0".to_string());
        }
        let final_price = Self::round_to_step_size(activation_price, tick_size);

        // El callbackRate en binance futures debe ser entre 0.1 y 5.
        let safe_callback = callback_rate.clamp(0.1, 5.0);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Trailing Stop interceptado para {} @ {}. Se maneja localmente.", symbol, final_price);
            return Ok(());
        }

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        });
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        buf.push_str("&positionSide=");
        buf.push_str(if is_long { "SHORT" } else { "LONG" });
        buf.push_str("&positionSide=");
        buf.push_str(if is_long { "SHORT" } else { "LONG" });
        buf.push_str("&type=TRAILING_STOP_MARKET");
        buf.push_str("&quantity=");
        buf.push_f64(final_quantity);
        buf.push_str("&activationPrice=");
        buf.push_f64(final_price);
        buf.push_str("&callbackRate=");
        buf.push_f64(safe_callback);
        buf.push_str("&newClientOrderId=");
        buf.push_str(client_order_id);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.execute_order_payload(buf.as_str()).await;
        if let Ok(limits) = &res {
            self.update_limits(limits);
        }
        res.map(|_| ())
    }

    /// FASE 12: Parallel OCO Tensor Execution
    #[inline(always)]
    async fn execute_oco_order(
        &self,
        symbol: &str,
        is_long_close: bool, // true if closing a long (Side = SELL)
        quantity: f64,
        take_profit_price: f64,
        stop_loss_price: f64,
        step_size: f64,
        tick_size: f64,
        base_client_id: &str,
    ) -> Result<(), String> {
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if final_quantity == 0.0 {
            return Err("Volumen 0".to_string());
        }

        let final_tp = Self::round_to_step_size(take_profit_price, tick_size);
        let final_sl = Self::round_to_step_size(stop_loss_price, tick_size);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] OCO Limit/Stop interceptada para {} @ TP: {} / SL: {}. Se maneja localmente.", symbol, final_tp, final_sl);
            return Ok(());
        }

        let side = if is_long_close { SIDE_SELL } else { SIDE_BUY };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.check_rate_limits(timestamp)?;

        let base_url = if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        };

        // 1. Build Stop Loss Order (STOP_MARKET)
        let mut sl_buf = ZeroAllocBuffer::new();
        sl_buf.push_str(base_url);
        let sl_payload_start = sl_buf.as_str().len();
        sl_buf.push_str("symbol=");
        sl_buf.push_str(symbol);
        sl_buf.push_str("&side=");
        sl_buf.push_str(side);
        sl_buf.push_str("&positionSide=");
        sl_buf.push_str(if is_long_close { "LONG" } else { "SHORT" });
        sl_buf.push_str("&type=STOP_MARKET");
        sl_buf.push_str("&reduceOnly=true");
        sl_buf.push_str("&quantity=");
        sl_buf.push_f64(final_quantity);
        sl_buf.push_str("&stopPrice=");
        sl_buf.push_f64(final_sl);
        sl_buf.push_str("&newClientOrderId=");
        sl_buf.push_str(&format!("{}_SL", base_client_id));
        sl_buf.push_str("&timestamp=");
        sl_buf.push_u64(timestamp);

        // 2. Build Take Profit Order (TAKE_PROFIT_MARKET)
        let mut tp_buf = ZeroAllocBuffer::new();
        tp_buf.push_str(base_url);
        let tp_payload_start = tp_buf.as_str().len();
        tp_buf.push_str("symbol=");
        tp_buf.push_str(symbol);
        tp_buf.push_str("&side=");
        tp_buf.push_str(side);
        tp_buf.push_str("&positionSide=");
        tp_buf.push_str(if is_long_close { "LONG" } else { "SHORT" });
        tp_buf.push_str("&type=TAKE_PROFIT_MARKET");
        tp_buf.push_str("&reduceOnly=true");
        tp_buf.push_str("&quantity=");
        tp_buf.push_f64(final_quantity);
        tp_buf.push_str("&stopPrice=");
        tp_buf.push_f64(final_tp);
        tp_buf.push_str("&newClientOrderId=");
        tp_buf.push_str(&format!("{}_TP", base_client_id));
        tp_buf.push_str("&timestamp=");
        tp_buf.push_u64(timestamp);

        let api_secret = self.api_secret.read().unwrap().clone();

        let mut sig_buf_sl = [0u8; 64];
        let sl_payload = &sl_buf.as_str()[sl_payload_start..];
        sign_payload_to_buffer(sl_payload, &api_secret, &mut sig_buf_sl);
        sl_buf.push_str("&signature=");
        sl_buf.push_str(unsafe { std::str::from_utf8_unchecked(&sig_buf_sl) });

        let mut sig_buf_tp = [0u8; 64];
        let tp_payload = &tp_buf.as_str()[tp_payload_start..];
        sign_payload_to_buffer(tp_payload, &api_secret, &mut sig_buf_tp);
        tp_buf.push_str("&signature=");
        tp_buf.push_str(unsafe { std::str::from_utf8_unchecked(&sig_buf_tp) });

        // Fire both simultaneously to minimize latency
        let (mut sl_res, mut tp_res) = tokio::join!(
            self.client.execute_order_payload(sl_buf.as_str()),
            self.client.execute_order_payload(tp_buf.as_str())
        );

        if let Ok(limits) = &sl_res {
            self.update_limits(limits);
        }
        if let Ok(limits) = &tp_res {
            self.update_limits(limits);
        }

        // F1.9 — FIX OCO PARCIAL: antes, una pierna fallida devolvía Ok(())
        // dejando la posición protegida por UN solo lado (naked al otro).
        // Política correcta: reintentar la pierna caída UNA vez; si sigue
        // caída, cancelar la pierna buena y devolver Err — el caller decide
        // aplanar o alertar. Nunca protección parcial silenciosa.
        if sl_res.is_err() || tp_res.is_err() {
            let sl_down = sl_res.is_err();
            let tp_down = tp_res.is_err();
            println!(
                "⚠️ [OCO] Pierna(s) fallida(s) (SL={}, TP={}). Reintentando...",
                sl_down, tp_down
            );
            if sl_down {
                sl_res = self.client.execute_order_payload(sl_buf.as_str()).await;
                if let Ok(limits) = &sl_res {
                    self.update_limits(limits);
                }
            }
            if tp_down {
                tp_res = self.client.execute_order_payload(tp_buf.as_str()).await;
                if let Ok(limits) = &tp_res {
                    self.update_limits(limits);
                }
            }
        }

        if sl_res.is_err() && tp_res.is_err() {
            return Err("Ambas órdenes OCO fallaron.".to_string());
        }
        if sl_res.is_err() || tp_res.is_err() {
            let good_id = if sl_res.is_ok() {
                format!("{}_SL", base_client_id)
            } else {
                format!("{}_TP", base_client_id)
            };
            let _ = self.cancel_order(symbol, &good_id).await;
            return Err(format!(
                "OCO PARCIAL: pierna fallida tras retry; pierna buena {} cancelada. Posición SIN protección — aplanar o alertar.",
                good_id
            ));
        }
        Ok(())
    }

    /// Cancela una orden activa usando el client_order_id
    #[inline(always)]
    async fn cancel_order(&self, symbol: &str, client_order_id: &str) -> Result<(), String> {
        if self.is_paper_trading {
            println!(
                "📝 [PAPER TRADING LOCAL] Orden Cancelada: {} en {}",
                client_order_id, symbol
            );
            return Ok(());
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        });
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&origClientOrderId=");
        buf.push_str(client_order_id);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.cancel_order_payload(buf.as_str()).await;
        if let Ok(limits) = &res {
            self.update_limits(limits);
        }
        res.map(|_| ())
    }

    /// F1.2/F1.3: GET /fapi/v1/order por origClientOrderId — estado REAL de la orden.
    /// Fuente de verdad para maker-chase y resolución de timeouts ambiguos.
    async fn query_order(&self, symbol: &str, client_order_id: &str) -> Result<OrderAck, String> {
        if self.is_paper_trading {
            // En paper no hay exchange: fingir orden nueva sin fills.
            return Ok(OrderAck {
                symbol: symbol.to_string(),
                client_order_id: client_order_id.to_string(),
                orig_qty: 0.0,
                executed_qty: 0.0,
                status: "NEW".to_string(),
                ..Default::default()
            });
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        });
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&origClientOrderId=");
        buf.push_str(client_order_id);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.get_payload(buf.as_str()).await;
        match res {
            Ok((limits, body)) => {
                self.update_limits(&limits);
                crate::order_types::parse_order_body(&body)
            }
            Err(e) => Err(e),
        }
    }

    async fn fetch_server_time(&self) -> Result<i64, String> {
        let base_url = self.client.get_base_url();
        let url = format!("{}/fapi/v1/time", base_url);

        match self.client.get_payload(&url).await {
            Ok((limits, text)) => {
                self.update_limits(&limits);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(time) = json["serverTime"].as_i64() {
                        return Ok(time);
                    }
                }
                Err(format!("serverTime not found in response. Text: {}", text))
            }
            Err(e) => Err(e),
        }
    }

    async fn fetch_open_positions(&self) -> Result<Vec<ActivePosition>, String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        if self.client.is_testnet.load(Ordering::Relaxed) {
            buf.push_str("https://testnet.binancefuture.com/fapi/v2/positionRisk?");
        } else {
            buf.push_str("https://fapi.binance.com/fapi/v2/positionRisk?");
        }
        let payload_start = buf.as_str().len();

        buf.push_str("timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.get_payload(buf.as_str()).await;
        match res {
            Ok((limits, text)) => {
                self.update_limits(&limits);
                let mut open_positions = Vec::new();
                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(arr) = json_val.as_array() {
                        for item in arr {
                            if let Some(amt_str) = item.get("positionAmt").and_then(|v| v.as_str())
                            {
                                if let Ok(amt) = amt_str.parse::<f64>() {
                                    if amt.abs() > 1e-8 {
                                        if let (Some(sym), Some(price_str)) = (
                                            item.get("symbol").and_then(|v| v.as_str()),
                                            item.get("entryPrice").and_then(|v| v.as_str()),
                                        ) {
                                            if let Ok(entry_price) = price_str.parse::<f64>() {
                                                open_positions.push(ActivePosition {
                                                    symbol: sym.to_string(),
                                                    qty: amt.abs(),
                                                    entry_price,
                                                    is_long: amt > 0.0,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(open_positions)
            }
            Err(e) => Err(e),
        }
    }

    async fn fetch_account_balance(&self) -> Result<f64, String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        if self.client.is_testnet.load(Ordering::Relaxed) {
            buf.push_str("https://testnet.binancefuture.com/fapi/v2/account?");
        } else {
            buf.push_str("https://fapi.binance.com/fapi/v2/account?");
        }
        let payload_start = buf.as_str().len();

        buf.push_str("timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        match self.client.get_payload(buf.as_str()).await {
            Ok((limits, text)) => {
                self.update_limits(&limits);

                // Parse the JSON object using serde_json to find availableBalance
                if let Ok(account_info) = serde_json::from_str::<serde_json::Value>(&text) {
                    // Try to fetch availableBalance first, fallback to totalWalletBalance
                    let bal_str = account_info
                        .get("availableBalance")
                        .or_else(|| account_info.get("totalWalletBalance"))
                        .and_then(|v| v.as_str());

                    if let Some(bal_str) = bal_str {
                        if let Ok(bal) = bal_str.parse::<f64>() {
                            return Ok(bal);
                        }
                    }
                }
                Err(format!(
                    "Failed to parse balance from Binance API /fapi/v2/account. Response: {}",
                    text
                ))
            }
            Err(e) => Err(e),
        }
    }

    async fn set_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        if self.client.is_testnet.load(Ordering::Relaxed) {
            buf.push_str("https://testnet.binancefuture.com/fapi/v1/leverage?");
        } else {
            buf.push_str("https://fapi.binance.com/fapi/v1/leverage?");
        }

        let payload_start = buf.as_str().len();
        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&leverage=");
        buf.push_u64(leverage as u64);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.read().unwrap().clone();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.post_payload(buf.as_str()).await;
        match res {
            Ok((limits, _text)) => {
                self.update_limits(&limits);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    async fn fetch_commission_rate(&self, symbol: &str) -> Result<(f64, f64), String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        if self.client.is_testnet.load(Ordering::Relaxed) {
            buf.push_str("https://testnet.binancefuture.com/fapi/v1/commissionRate?");
        } else {
            buf.push_str("https://fapi.binance.com/fapi/v1/commissionRate?");
        }
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        sign_payload_to_buffer(payload, &*self.api_secret.read().unwrap(), &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        match self.client.get_payload(buf.as_str()).await {
            Ok((limits, text)) => {
                self.update_limits(&limits);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let (Some(m_str), Some(t_str)) = (
                        json["makerCommissionRate"].as_str(),
                        json["takerCommissionRate"].as_str(),
                    ) {
                        let maker = m_str
                            .parse::<f64>()
                            .map_err(|e| format!("Parse error maker fee: {}", e))?;
                        let taker = t_str
                            .parse::<f64>()
                            .map_err(|e| format!("Parse error taker fee: {}", e))?;
                        // CORRECCIÓN: No usar unwrap_or con fallbacks hardcodeados
                        // Si Binance devuelve datos que no se pueden parsear, es un error real
                        return Ok((maker, taker));
                    }
                }
                Err(format!("Fallo al extraer comisiones reales de Binance. No se usarán tarifas hardcodeadas. Response: {}", text))
            }
            Err(e) => Err(e),
        }
    }

    async fn fetch_exchange_info(&self, symbol: &str) -> Result<f64, String> {
        let mut buf = ZeroAllocBuffer::new();
        if self.client.is_testnet.load(Ordering::Relaxed) {
            buf.push_str("https://testnet.binancefuture.com/fapi/v1/exchangeInfo");
        } else {
            buf.push_str("https://fapi.binance.com/fapi/v1/exchangeInfo");
        }

        match self.client.get_payload(buf.as_str()).await {
            Ok((limits, text)) => {
                self.update_limits(&limits);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(symbols) = json["symbols"].as_array() {
                        for sym_info in symbols {
                            if sym_info["symbol"] == symbol {
                                if let Some(filters) = sym_info["filters"].as_array() {
                                    for filter in filters {
                                        if filter["filterType"] == "MIN_NOTIONAL" {
                                            if let Some(min_notional_str) =
                                                filter["notional"].as_str()
                                            {
                                                if let Ok(min_notional) =
                                                    min_notional_str.parse::<f64>()
                                                {
                                                    return Ok(min_notional);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(format!(
                    "No se encontró MIN_NOTIONAL para el símbolo {} en la API.",
                    symbol
                ))
            }
            Err(e) => Err(e),
        }
    }
}
