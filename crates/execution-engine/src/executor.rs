use crate::binance_api::{
    sign_payload_to_buffer, ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL,
    TIME_IN_FORCE_IOC,
};
use crate::order_types::OrderAck;
use crate::ExecutionPayload;
use arc_swap::{ArcSwap, ArcSwapOption};
use risk_engine::ValidatedOrder;
use signal_engine::SignalType;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::client::{BinanceClient, ZeroAllocBuffer};

#[derive(Debug, Clone, Copy)]
pub struct SymbolFilter {
    pub step_size: f64,
    pub tick_size: f64,
    pub min_notional: f64,
}

#[inline(always)]
pub fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[inline(always)]
pub fn current_synced_timestamp_ms(arena: Option<&quantum_arena::GlobalArena>) -> u64 {
    let base = current_timestamp_ms() as i64;
    if let Some(a) = arena {
        (base + a.server_time_offset_ms.load(Ordering::Relaxed)) as u64
    } else {
        base as u64
    }
}

impl Default for SymbolFilter {
    fn default() -> Self {
        Self {
            step_size: 0.001,
            tick_size: 0.1,
            min_notional: 5.0,
        }
    }
}

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
    ) -> Result<(), String> {
        self.execute_raw_qty_with_client_id(symbol, is_long, quantity, step_size, "")
            .await
    }

    async fn execute_raw_qty_with_client_id(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        step_size: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        let _ = client_order_id;
        self.execute_raw_qty(symbol, is_long, quantity, step_size)
            .await
    }

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

    /// Cancela todas las órdenes activas para un símbolo en Binance (ej: OCO TP/SL huérfanos)
    async fn cancel_all_symbol_orders(&self, symbol: &str) -> Result<(), String> {
        let _ = symbol;
        Ok(())
    }

    /// D-371: Cancela únicamente órdenes OCO de la posición específica (LONG/SHORT)
    async fn cancel_position_oco_orders(
        &self,
        symbol: &str,
        is_long: bool,
    ) -> Result<usize, String> {
        let _ = (symbol, is_long);
        Ok(0)
    }

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
    api_secret: ArcSwap<String>,
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
    active_leverage: ArcSwap<std::collections::HashMap<String, u32>>,
    is_paper_trading: bool,
    /// F1.5: memoria del ciclo de vida de órdenes — compartida con el
    /// user-data stream (F1.6) y la reconciliación (F1.7).
    order_registry: std::sync::Arc<crate::order_registry::OrderRegistry>,
    /// F1.8: no enviar órdenes hasta este timestamp (cooldown post-429).
    cooldown_until_ms: AtomicU64,
    /// F1.8: 429s consecutivos — >=3 sugiere ban inminente → kill-switch real.
    consecutive_429: AtomicUsize,
    /// Cache en RAM de filtros de símbolos (tickSize, stepSize, minNotional) O(1) < 5ns
    symbol_filters: ArcSwap<std::collections::HashMap<String, SymbolFilter>>,
    pub arena: ArcSwapOption<quantum_arena::GlobalArena>,
    pub ws: std::sync::Arc<crate::ws_executor::WsExecutor>,
    pub is_hedge_mode: AtomicBool,
}

impl OrderExecutor {
    pub fn new(api_key: String, api_secret: String, is_testnet: bool) -> Self {
        Self {
            api_secret: ArcSwap::from_pointee(api_secret.clone()),
            client: BinanceClient::new(api_key.clone(), is_testnet),
            rate_limit_counter: AtomicUsize::new(0),
            last_reset_timestamp: AtomicU64::new(0),
            binance_weight_1m: AtomicUsize::new(0),
            binance_orders_10s: AtomicUsize::new(0),
            binance_orders_1m: AtomicUsize::new(0),
            max_weight_1m: AtomicUsize::new(2200), // Default, but can be updated
            max_orders_10s: AtomicUsize::new(280),
            max_orders_1m: AtomicUsize::new(1100),
            kill_switch: AtomicBool::new(false),
            active_leverage: ArcSwap::from_pointee(std::collections::HashMap::new()),
            is_paper_trading: false, // Desacoplado: false por defecto para permitir órdenes reales en Testnet/Mainnet; activar con set_paper_trading(true) si se desea simular
            order_registry: std::sync::Arc::new(crate::order_registry::OrderRegistry::new()),
            cooldown_until_ms: AtomicU64::new(0),
            consecutive_429: AtomicUsize::new(0),
            symbol_filters: ArcSwap::from_pointee(std::collections::HashMap::new()),
            arena: ArcSwapOption::empty(),
            ws: std::sync::Arc::new(crate::ws_executor::WsExecutor::new(
                api_key, api_secret, is_testnet,
            )),
            is_hedge_mode: AtomicBool::new(true),
        }
    }

    /// R3.2 / K-06: Constructor para hot-swap de executor (e.g. transición a mainnet).
    /// Preserva el OrderRegistry existente (para no romper la sincronización con el
    /// UserDataStreamer), la arena viva y los componentes compartidos.
    pub fn new_with_shared(
        api_key: String,
        api_secret: String,
        is_testnet: bool,
        order_registry: std::sync::Arc<crate::order_registry::OrderRegistry>,
        arena: Option<std::sync::Arc<quantum_arena::GlobalArena>>,
    ) -> Self {
        Self {
            api_secret: ArcSwap::from_pointee(api_secret.clone()),
            client: BinanceClient::new(api_key.clone(), is_testnet),
            rate_limit_counter: AtomicUsize::new(0),
            last_reset_timestamp: AtomicU64::new(0),
            binance_weight_1m: AtomicUsize::new(0),
            binance_orders_10s: AtomicUsize::new(0),
            binance_orders_1m: AtomicUsize::new(0),
            max_weight_1m: AtomicUsize::new(2200),
            max_orders_10s: AtomicUsize::new(280),
            max_orders_1m: AtomicUsize::new(1100),
            kill_switch: AtomicBool::new(false),
            active_leverage: ArcSwap::from_pointee(std::collections::HashMap::new()),
            is_paper_trading: false,
            order_registry,
            cooldown_until_ms: AtomicU64::new(0),
            consecutive_429: AtomicUsize::new(0),
            symbol_filters: ArcSwap::from_pointee(std::collections::HashMap::new()),
            arena: ArcSwapOption::new(arena),
            ws: std::sync::Arc::new(crate::ws_executor::WsExecutor::new(
                api_key, api_secret, is_testnet,
            )),
            is_hedge_mode: AtomicBool::new(true),
        }
    }

    /// Registro de órdenes (F1.5) — para spawn del user-data stream y queries.
    pub fn registry(&self) -> std::sync::Arc<crate::order_registry::OrderRegistry> {
        self.order_registry.clone()
    }

    pub fn client(&self) -> &BinanceClient {
        &self.client
    }

    pub fn api_secret(&self) -> String {
        self.api_secret.load().to_string()
    }

    #[inline(always)]
    pub fn get_synced_timestamp(&self) -> u64 {
        if let Some(guard) = self.arena.load_full() {
            current_synced_timestamp_ms(Some(guard.as_ref()))
        } else {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        }
    }

    /// F1.11: verifica que la cuenta esté en modo HEDGE (dualSidePosition).
    /// El motor SIEMPRE envía positionSide=LONG/SHORT: con la cuenta en modo
    /// one-way TODA orden falla con -4061 (hallazgo real del ciclo testnet).
    /// Si no está en hedge, la activa (POST firmado /fapi/v1/positionSide/dual).
    pub async fn ensure_hedge_mode(&self) -> Result<bool, String> {
        if self.is_paper_trading {
            return Ok(true);
        }
        let timestamp = self.get_synced_timestamp();
        let api_secret = self.api_secret.load();

        // 1) Modo actual (GET firmado)
        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(self.client.get_base_url());
        buf.push_str("/fapi/v1/positionSide/dual?");
        let payload_start = buf.as_str().len();
        buf.push_str("timestamp=");
        buf.push_u64(timestamp);
        let mut sig_buf = [0u8; 64];
        sign_payload_to_buffer(&buf.as_str()[payload_start..], &api_secret, &mut sig_buf);
        buf.push_str("&signature=");
        buf.push_str(unsafe { std::str::from_utf8_unchecked(&sig_buf) });

        let (_, body) = self.client.get_payload(buf.as_str()).await?;
        #[derive(serde::Deserialize)]
        struct ModeResp {
            #[serde(rename = "dualSidePosition")]
            dual: bool,
        }
        let mode: ModeResp = serde_json::from_str(&body)
            .map_err(|e| format!("POSITION_MODE_PARSE: {} body={}", e, body))?;
        if mode.dual {
            self.is_hedge_mode.store(true, Ordering::Relaxed);
            return Ok(false); // ya estaba en hedge: no se cambió nada
        }

        // 2) Activar hedge (POST firmado dualSidePosition=true)
        let timestamp = self.get_synced_timestamp();
        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(self.client.get_base_url());
        buf.push_str("/fapi/v1/positionSide/dual?");
        let payload_start = buf.as_str().len();
        buf.push_str("dualSidePosition=true&timestamp=");
        buf.push_u64(timestamp);
        let mut sig_buf = [0u8; 64];
        sign_payload_to_buffer(&buf.as_str()[payload_start..], &api_secret, &mut sig_buf);
        buf.push_str("&signature=");
        buf.push_str(unsafe { std::str::from_utf8_unchecked(&sig_buf) });

        match self.client.post_payload(buf.as_str()).await {
            Ok(_) => {
                self.is_hedge_mode.store(true, Ordering::Relaxed);
                println!("🔀 [POSITION-MODE] Cuenta migrada a HEDGE (dualSidePosition=true)");
                Ok(true)
            }
            Err(e) => {
                // N-05: Si falla activar hedge (ej. posiciones abiertas en one-way), reflejar la verdad local
                self.is_hedge_mode.store(false, Ordering::Relaxed);
                Err(format!(
                    "No se pudo activar modo hedge: {}. Cuenta permanece en ONE-WAY mode.",
                    e
                ))
            }
        }
    }

    /// F1.11/F5.2: APLANADO TOTAL — cancela todas las órdenes abiertas y cierra
    /// todas las posiciones con MARKET reduceOnly (side opuesto, qty exacta).
    /// Mode-aware: en one-way se OMITE positionSide (si no, -4061).
    /// Usos: pre-flight de demo (limpiar huérfanas de sesiones previas) y
    /// primitivo del kill-switch real. Devuelve (órdenes canceladas, posiciones cerradas).
    pub async fn flatten_all_positions(&self) -> Result<(usize, usize), String> {
        if self.is_paper_trading {
            return Ok((0, 0));
        }
        let timestamp = self.get_synced_timestamp();
        let api_secret = self.api_secret.load();

        // Modo de la cuenta (dual=hedge) — determina si se envía positionSide.
        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(self.client.get_base_url());
        buf.push_str("/fapi/v1/positionSide/dual?");
        let payload_start = buf.as_str().len();
        buf.push_str("timestamp=");
        buf.push_u64(timestamp);
        let mut sig_buf = [0u8; 64];
        sign_payload_to_buffer(&buf.as_str()[payload_start..], &api_secret, &mut sig_buf);
        buf.push_str("&signature=");
        buf.push_str(unsafe { std::str::from_utf8_unchecked(&sig_buf) });
        let dual_mode_opt = match self.client.get_payload(buf.as_str()).await {
            Ok((_, mode_body)) => {
                #[derive(serde::Deserialize)]
                struct ModeResp {
                    #[serde(rename = "dualSidePosition")]
                    dual: bool,
                }
                serde_json::from_str::<ModeResp>(&mode_body)
                    .map(|m| m.dual)
                    .ok()
            }
            Err(e) => {
                println!("⚠️ [FLATTEN] No se pudo leer modo de posición (/fapi/v1/positionSide/dual): {}. Se procederá con failover dinámico.", e);
                None
            }
        };
        let mut dual = dual_mode_opt.unwrap_or(true); // el motor ES hedge por diseño; failover a one-way si falla

        // 1) Cancelar TODAS las órdenes abiertas por símbolo con posiciones.
        let entries = self.fetch_position_risk().await?;
        let symbols: Vec<String> = entries
            .iter()
            .filter(|p| p.is_open())
            .map(|p| p.symbol.clone())
            .collect();
        let mut cancelled = 0usize;
        for sym in &symbols {
            let ts = self.get_synced_timestamp();
            let mut buf = ZeroAllocBuffer::new();
            buf.push_str(self.client.get_base_url());
            buf.push_str("/fapi/v1/allOpenOrders?");
            let payload_start = buf.as_str().len();
            buf.push_str("symbol=");
            buf.push_str(sym);
            buf.push_str("&timestamp=");
            buf.push_u64(ts);
            let mut sig_buf = [0u8; 64];
            sign_payload_to_buffer(&buf.as_str()[payload_start..], &api_secret, &mut sig_buf);
            buf.push_str("&signature=");
            buf.push_str(unsafe { std::str::from_utf8_unchecked(&sig_buf) });
            if self.client.cancel_order_payload(buf.as_str()).await.is_ok() {
                cancelled += 1;
            }
        }

        // 2) Cerrar cada posición con MARKET reduceOnly del lado opuesto.
        let mut closed = 0usize;
        for p in entries.iter().filter(|p| p.is_open()) {
            let is_long = p.is_long();
            let ts = self.get_synced_timestamp();
            let coid = uuid::Uuid::now_v7().simple().to_string();
            let mut buf = ZeroAllocBuffer::new();
            buf.push_str(self.client.get_base_url());
            buf.push_str("/fapi/v1/order?");
            let payload_start = buf.as_str().len();
            buf.push_str("symbol=");
            buf.push_str(&p.symbol);
            buf.push_str("&side=");
            buf.push_str(if is_long { "SELL" } else { "BUY" });
            if dual {
                buf.push_str("&positionSide=");
                buf.push_str(if is_long { "LONG" } else { "SHORT" });
                buf.push_str("&type=MARKET&quantity=");
            } else {
                buf.push_str("&type=MARKET&reduceOnly=true&quantity=");
            }
            buf.push_f64(p.position_amt.abs());
            buf.push_str("&newClientOrderId=");
            buf.push_str(&coid);
            buf.push_str("&timestamp=");
            buf.push_u64(ts);
            let mut sig_buf = [0u8; 64];

            // WS Execution Routing (Zero-TLS overhead)
            if self.ws.is_connected() {
                let api_key = self
                    .client
                    .api_key
                    .load()
                    .to_str()
                    .unwrap_or("")
                    .to_string();
                if let Err(e) = self.ws.send_order_payload(
                    &api_key,
                    &api_secret,
                    &p.symbol,
                    if is_long { "SELL" } else { "BUY" },
                    if dual {
                        Some(if is_long { "LONG" } else { "SHORT" })
                    } else {
                        None
                    },
                    ORDER_TYPE_MARKET,
                    p.position_amt.abs(),
                    None,
                    None,
                    !dual, // reduceOnly is true when NOT dual (One-Way Mode)
                    &coid,
                    ts,
                ) {
                    println!("⚠️ [WS-EXECUTOR] Failed to send close position order, falling back to REST: {}", e);
                } else {
                    println!("🧹 [FLATTEN] {} close request via WS dispatched", p.symbol);
                    closed += 1;
                    continue; // Skip REST fallback since WS sent successfully
                }
            }

            sign_payload_to_buffer(&buf.as_str()[payload_start..], &api_secret, &mut sig_buf);
            buf.push_str("&signature=");
            buf.push_str(unsafe { std::str::from_utf8_unchecked(&sig_buf) });

            let mut close_res = self.client.execute_order_payload_typed(buf.as_str()).await;
            // R3.3 / K-16: Fail-safe retry si el exchange rechaza por discrepancia de modo (-4061)
            if let Err(ref e) = close_res {
                if e.contains("-4061") || e.contains("position side") || e.contains("reduceOnly") {
                    println!("⚠️ [FLATTEN] Modo rechazado (-4061) para {}. Conmutando modo (dual={}) y reintentando cierre...", p.symbol, !dual);
                    dual = !dual;
                    self.is_hedge_mode.store(dual, Ordering::SeqCst);
                    let ts_retry = self.get_synced_timestamp();
                    let coid_retry = uuid::Uuid::now_v7().simple().to_string();
                    let mut retry_buf = ZeroAllocBuffer::new();
                    retry_buf.push_str(self.client.get_base_url());
                    retry_buf.push_str("/fapi/v1/order?");
                    let retry_payload_start = retry_buf.as_str().len();
                    retry_buf.push_str("symbol=");
                    retry_buf.push_str(&p.symbol);
                    retry_buf.push_str("&side=");
                    retry_buf.push_str(if is_long { "SELL" } else { "BUY" });
                    if dual {
                        retry_buf.push_str("&positionSide=");
                        retry_buf.push_str(if is_long { "LONG" } else { "SHORT" });
                        retry_buf.push_str("&type=MARKET&quantity=");
                    } else {
                        retry_buf.push_str("&type=MARKET&reduceOnly=true&quantity=");
                    }
                    retry_buf.push_f64(p.position_amt.abs());
                    retry_buf.push_str("&newClientOrderId=");
                    retry_buf.push_str(&coid_retry);
                    retry_buf.push_str("&timestamp=");
                    retry_buf.push_u64(ts_retry);
                    let mut retry_sig_buf = [0u8; 64];
                    sign_payload_to_buffer(
                        &retry_buf.as_str()[retry_payload_start..],
                        &api_secret,
                        &mut retry_sig_buf,
                    );
                    retry_buf.push_str("&signature=");
                    retry_buf.push_str(unsafe { std::str::from_utf8_unchecked(&retry_sig_buf) });

                    close_res = self
                        .client
                        .execute_order_payload_typed(retry_buf.as_str())
                        .await;
                }
            }

            match close_res {
                Ok((limits, ack)) => {
                    self.update_limits(&limits);
                    let now = self.get_synced_timestamp();
                    self.order_registry.apply_ack(&ack, now);
                    closed += 1;
                    println!(
                        "🧹 [FLATTEN] {} cerrada {} @ ~{} (status {})",
                        p.symbol,
                        p.position_amt.abs(),
                        ack.avg_price,
                        ack.status
                    );
                }
                Err(e) => println!("⚠️ [FLATTEN] {} NO cerrada tras reintento: {}", p.symbol, e),
            }
        }
        Ok((cancelled, closed))
    }

    /// F1.7: GET /fapi/v2/positionRisk — posiciones abiertas según el EXCHANGE.
    /// Fuente de verdad para reconciliación al arranque y periódica.
    pub async fn fetch_position_risk(
        &self,
    ) -> Result<Vec<crate::reconciliation::PositionRiskEntry>, String> {
        if self.is_paper_trading {
            return Ok(Vec::new());
        }
        let timestamp = self.get_synced_timestamp();

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(self.client.get_base_url());
        buf.push_str("/fapi/v2/positionRisk?");
        let payload_start = buf.as_str().len();
        buf.push_str("timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.load();
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
        self.api_secret.store(Arc::new(new_secret));
        if true {}
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
            let arena_guard = self.arena.load_full();
            let now = current_synced_timestamp_ms(arena_guard.as_deref());
            drop(arena_guard);
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
        if step_size <= 0.0 {
            return quantity;
        }
        let inv = 1.0 / step_size;
        ((quantity * inv) + 1e-9).floor() / inv
    }

    // FIX #738: Redondeo direccional de precio: ceil para SELL, floor para BUY
    #[inline(always)]
    fn round_price_to_tick(price: f64, tick_size: f64, is_sell: bool) -> f64 {
        if tick_size <= 0.0 {
            return price;
        }
        let inv = 1.0 / tick_size;
        if is_sell {
            ((price * inv) - 1e-9).ceil() / inv
        } else {
            ((price * inv) + 1e-9).floor() / inv
        }
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

        // Cantidad exacta calculada a partir del margen asignado y el apalancamiento aprobado
        let raw_quantity = (order.volume_usd * order.leverage) / current_price;
        let final_quantity = Self::round_to_step_size(raw_quantity, step_size);

        if final_quantity == 0.0 {
            return None;
        }

        let side = match order.signal {
            SignalType::Long => SIDE_BUY,
            SignalType::Short => SIDE_SELL,
            SignalType::Flat => return None,
        };

        let arena_guard = self.arena.load_full();
        let timestamp = current_synced_timestamp_ms(arena_guard.as_deref());
        drop(arena_guard);

        // F1.2: identificador idempotente — toda orden lleva newClientOrderId.
        let client_order_id = uuid::Uuid::now_v7().simple().to_string();

        // FASE 19 + F1.4: Órdenes Institucionales y Evasión de Taker Fees.
        // El maker (POST_ONLY) exige timeInForce=GTX Y price en el MISMO query
        // que se firma y se envía. Precio redondeado al tickSize REAL del símbolo
        // (nunca un {:.4} fijo que viola el filtro PRICE_FILTER).
        let (order_type, time_in_force, extra_params) = if order.maker_only {
            let is_sell = side == SIDE_SELL;
            // FIX #791: Offset pasivo seguro para Post-Only (GTX).
            // Si colocamos una orden al precio actual exacto, Binance la rechaza con -5022.
            // Para ser Maker pasivo: BUY debe estar al menos 1 tick por debajo del ask actual,
            // y SELL debe estar al menos 1 tick por encima del bid actual.
            let passive_price = if is_sell {
                current_price + tick_size
            } else {
                (current_price - tick_size).max(tick_size)
            };
            let final_price = Self::round_price_to_tick(passive_price, tick_size, is_sell);
            (
                ORDER_TYPE_LIMIT,
                crate::binance_api::TIME_IN_FORCE_GTX,
                format!("&timeInForce=GTX&price={}", final_price),
            )
        } else {
            (ORDER_TYPE_MARKET, TIME_IN_FORCE_IOC, String::new())
        };

        // D-171: Respetar is_hedge_mode. En One-Way mode, positionSide NO debe enviarse.
        let is_hedge = self.is_hedge_mode.load(Ordering::Relaxed);
        let pos_side_param = if is_hedge {
            if side == SIDE_BUY {
                "&positionSide=LONG"
            } else {
                "&positionSide=SHORT"
            }
        } else {
            ""
        };
        // F1.4: query EXACTA que se firma = query EXACTA que se envía.
        let signed_query = format!(
            "symbol={}&side={}{}&type={}&quantity={}{}&newClientOrderId={}&timestamp={}",
            symbol,
            side,
            pos_side_param,
            order_type,
            final_quantity,
            extra_params,
            client_order_id,
            timestamp
        );

        // Firmar
        let mut sig_buf = [0u8; 64];
        let api_secret = self.api_secret.load();
        sign_payload_to_buffer(&signed_query, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) }.to_string();

        Some(ExecutionPayload {
            symbol: symbol.to_string(),
            side: side.to_string(),
            quantity: final_quantity,
            order_type: order_type.to_string(),
            time_in_force: time_in_force.to_string(),
            position_side: if is_hedge {
                if side == SIDE_BUY {
                    "LONG".to_string()
                } else {
                    "SHORT".to_string()
                }
            } else {
                "BOTH".to_string()
            },
            price: if order.maker_only {
                let is_sell = side == SIDE_SELL;
                let passive_price = if is_sell {
                    current_price + tick_size
                } else {
                    (current_price - tick_size).max(tick_size)
                };
                Some(Self::round_price_to_tick(passive_price, tick_size, is_sell))
            } else {
                None
            },
            reduce_only: false,
            signed_query,
            client_order_id,
            signature,
            timestamp,
        })
    }

    pub async fn fetch_all_symbol_filters(
        &self,
    ) -> Result<std::collections::HashMap<String, SymbolFilter>, String> {
        let mut buf = ZeroAllocBuffer::new();
        if self.client.is_testnet.load(Ordering::Relaxed) {
            buf.push_str("https://testnet.binancefuture.com/fapi/v1/exchangeInfo");
        } else {
            buf.push_str("https://fapi.binance.com/fapi/v1/exchangeInfo");
        }

        match self.client.get_payload(buf.as_str()).await {
            Ok((limits, text)) => {
                self.update_limits(&limits);
                let mut map = std::collections::HashMap::new();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(symbols) = json["symbols"].as_array() {
                        for sym_info in symbols {
                            if let Some(sym) = sym_info["symbol"].as_str() {
                                let mut filter = SymbolFilter::default();
                                if let Some(filters) = sym_info["filters"].as_array() {
                                    for f in filters {
                                        match f["filterType"].as_str() {
                                            Some("LOT_SIZE") => {
                                                if let Some(s) = f["stepSize"].as_str() {
                                                    filter.step_size =
                                                        s.parse::<f64>().unwrap_or(0.001);
                                                }
                                            }
                                            Some("PRICE_FILTER") => {
                                                if let Some(t) = f["tickSize"].as_str() {
                                                    filter.tick_size =
                                                        t.parse::<f64>().unwrap_or(0.1);
                                                }
                                            }
                                            Some("MIN_NOTIONAL") => {
                                                if let Some(n) = f["notional"].as_str() {
                                                    filter.min_notional =
                                                        n.parse::<f64>().unwrap_or(5.0);
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                map.insert(sym.to_string(), filter);
                            }
                        }
                    }
                }
                Ok(map)
            }
            Err(e) => Err(e),
        }
    }

    pub async fn get_symbol_filter(&self, symbol: &str) -> SymbolFilter {
        {
            let cache = self.symbol_filters.load();
            if true {
                if let Some(f) = cache.get(symbol) {
                    return *f;
                }
            }
        }
        if let Ok(filters) = self.fetch_all_symbol_filters().await {
            let res = filters.get(symbol).copied();
            self.symbol_filters.store(Arc::new(filters));
            if let Some(f) = res {
                return f;
            }
        }
        if let Some(coin_id) = quantum_arena::symbol_registry::try_index(symbol) {
            if let Some(spec) = quantum_arena::symbol_registry::try_spec(coin_id) {
                return SymbolFilter {
                    step_size: spec.step_size,
                    tick_size: spec.tick_size,
                    min_notional: spec.min_notional,
                };
            }
        }
        SymbolFilter::default()
    }

    #[inline(always)]
    pub async fn cancel_all_symbol_orders(&self, symbol: &str) -> Result<(), String> {
        <Self as ExecutionProvider>::cancel_all_symbol_orders(self, symbol).await
    }

    #[inline(always)]
    pub async fn cancel_position_oco_orders(
        &self,
        symbol: &str,
        is_long: bool,
    ) -> Result<usize, String> {
        <Self as ExecutionProvider>::cancel_position_oco_orders(self, symbol, is_long).await
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
            let cache = self.active_leverage.load();
            cache.get(symbol).copied().unwrap_or(0) != target_leverage
        };

        if needs_update {
            if let Err(e) = self.set_leverage(symbol, target_leverage).await {
                println!(
                    "⚠️ [EXECUTION] Failed to set dynamic leverage for {}: {}",
                    symbol, e
                );
            } else {
                let mut cache = (**self.active_leverage.load()).clone();
                cache.insert(symbol.to_string(), target_leverage);
                self.active_leverage.store(Arc::new(cache));
            }
        }

        let timestamp = self.get_synced_timestamp();
        self.check_rate_limits(timestamp)?;

        // F1.4: tickSize real del símbolo consultado en cache O(1) (< 5 ns)
        let filter = self.get_symbol_filter(symbol).await;
        let tick_size = filter.tick_size;

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
            // WS Execution Routing (Zero-TLS overhead)
            if self.ws.is_connected() {
                let api_key = self
                    .client
                    .api_key
                    .load()
                    .to_str()
                    .unwrap_or("")
                    .to_string();
                let api_secret = self.api_secret.load();
                let time_in_force_opt = if payload.time_in_force.is_empty() {
                    None
                } else {
                    Some(payload.time_in_force.as_str())
                };

                if let Err(e) = self.ws.send_order_payload(
                    &api_key,
                    &api_secret,
                    &payload.symbol,
                    &payload.side,
                    Some(&payload.position_side),
                    &payload.order_type,
                    payload.quantity,
                    payload.price,
                    time_in_force_opt,
                    payload.reduce_only,
                    &payload.client_order_id,
                    payload.timestamp,
                ) {
                    println!(
                        "⚠️ [WS-EXECUTOR] Failed to send order, falling back to REST: {}",
                        e
                    );
                } else {
                    // Orden disparada con éxito vía WS. El UserDataStream procesará el ACK real.
                    return Ok(());
                }
            }

            // F1.4: se envía EXACTAMENTE la query firmada. Sin reconstrucción (REST Fallback).
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

    #[inline(always)]
    async fn execute_raw_qty(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        step_size: f64,
    ) -> Result<(), String> {
        self.execute_raw_qty_with_client_id(symbol, is_long, quantity, step_size, "")
            .await
    }

    #[inline(always)]
    async fn execute_raw_qty_with_client_id(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        step_size: f64,
        client_order_id_param: &str,
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
        let timestamp = self.get_synced_timestamp();

        self.check_rate_limits(timestamp)?;

        // F1.2: toda orden market lleva newClientOrderId (idempotencia y trazabilidad).
        let client_order_id = if client_order_id_param.is_empty() {
            uuid::Uuid::now_v7().simple().to_string()
        } else {
            client_order_id_param.to_string()
        };
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
        if self.is_hedge_mode.load(Ordering::Relaxed) {
            buf.push_str("&positionSide=");
            buf.push_str(if is_long { "LONG" } else { "SHORT" });
        }
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
        let api_secret = self.api_secret.load();

        // WS Execution Routing (Zero-TLS overhead)
        if self.ws.is_connected() {
            let api_key = self
                .client
                .api_key
                .load()
                .to_str()
                .unwrap_or("")
                .to_string();
            if let Err(e) = self.ws.send_order_payload(
                &api_key,
                &api_secret,
                symbol,
                side,
                if self.is_hedge_mode.load(Ordering::Relaxed) {
                    Some(if is_long { "LONG" } else { "SHORT" })
                } else {
                    None
                },
                ORDER_TYPE_MARKET,
                final_quantity,
                None,
                None,
                false,
                &client_order_id,
                timestamp,
            ) {
                println!(
                    "⚠️ [WS-EXECUTOR] Failed to send raw qty order, falling back to REST: {}",
                    e
                );
            } else {
                return Ok(());
            }
        }

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

        // FIX #1414: Redondeo direccional de precio (ceil para SELL, floor para BUY)
        let final_price = Self::round_price_to_tick(price, tick_size, !is_long);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Ejecutando orden LIMIT de {} para {} @ {}. Cero latencia simulada.", final_quantity, symbol, final_price);
            return Ok(());
        }

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };
        let timestamp = self.get_synced_timestamp();

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
        if self.is_hedge_mode.load(Ordering::Relaxed) {
            buf.push_str("&positionSide=");
            buf.push_str(if is_long { "LONG" } else { "SHORT" });
        }
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
        let api_secret = self.api_secret.load();

        // WS Execution Routing (Zero-TLS overhead)
        if self.ws.is_connected() {
            let api_key = self
                .client
                .api_key
                .load()
                .to_str()
                .unwrap_or("")
                .to_string();
            if let Err(e) = self.ws.send_order_payload(
                &api_key,
                &api_secret,
                symbol,
                side,
                if self.is_hedge_mode.load(Ordering::Relaxed) {
                    Some(if is_long { "LONG" } else { "SHORT" })
                } else {
                    None
                },
                ORDER_TYPE_LIMIT,
                final_quantity,
                Some(final_price),
                Some(crate::binance_api::TIME_IN_FORCE_GTX),
                false,
                client_order_id,
                timestamp,
            ) {
                println!(
                    "⚠️ [WS-EXECUTOR] Failed to send limit order, falling back to REST: {}",
                    e
                );
            } else {
                return Ok(());
            }
        }

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

        // 2. Espera adaptativa de baja latencia con sondeo de order_registry (D-361)
        for _ in 0..5 {
            tokio::time::sleep(std::time::Duration::from_millis(3)).await;
            if let Some(order) = self.order_registry.get(client_order_id) {
                if !order.status.is_active() {
                    break;
                }
            }
        }

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
        // FIX #1414: Redondeo direccional de precio en órdenes IOC
        let final_price = Self::round_price_to_tick(price, tick_size, !is_long);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Ejecutando orden IOC de {} para {} @ {}. Cero latencia simulada.", final_quantity, symbol, final_price);
            return Ok(());
        }

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };
        let timestamp = self.get_synced_timestamp();
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
        if self.is_hedge_mode.load(Ordering::Relaxed) {
            buf.push_str("&positionSide=");
            buf.push_str(if is_long { "LONG" } else { "SHORT" });
        }
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
        let api_secret = self.api_secret.load();
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

        let final_price = Self::round_price_to_tick(price, tick_size, !is_long);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Ejecutando orden ICEBERG_LIMIT de {} (Iceberg: {}) para {} @ {}. Cero latencia simulada.", final_quantity, final_iceberg_qty, symbol, final_price);
            return Ok(());
        }

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };
        let timestamp = self.get_synced_timestamp();
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
        if self.is_hedge_mode.load(Ordering::Relaxed) {
            buf.push_str("&positionSide=");
            buf.push_str(if is_long { "LONG" } else { "SHORT" });
        }
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
        let api_secret = self.api_secret.load();
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
        let timestamp = self.get_synced_timestamp();
        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        });
        let payload_start = buf.as_str().len();

        let is_hedge = self.is_hedge_mode.load(Ordering::Relaxed);
        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        if is_hedge {
            buf.push_str("&positionSide=");
            buf.push_str(if is_long_close { "LONG" } else { "SHORT" });
            buf.push_str("&type=");
            buf.push_str(ORDER_TYPE_MARKET);
        } else {
            buf.push_str("&type=");
            buf.push_str(ORDER_TYPE_MARKET);
            buf.push_str("&reduceOnly=true");
        }
        buf.push_str("&quantity=");
        buf.push_f64(final_quantity);
        let client_order_id = uuid::Uuid::now_v7().simple().to_string();
        buf.push_str("&newClientOrderId=");
        buf.push_str(&client_order_id);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.load();
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
        let final_price = Self::round_price_to_tick(activation_price, tick_size, is_long);

        // El callbackRate en binance futures debe ser entre 0.1 y 5.
        let safe_callback = callback_rate.clamp(0.1, 5.0);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Trailing Stop interceptado para {} @ {}. Se maneja localmente.", symbol, final_price);
            return Ok(());
        }

        // Para cerrar LONG: side = SELL, positionSide = LONG
        // Para cerrar SHORT: side = BUY, positionSide = SHORT
        let side = if is_long { SIDE_SELL } else { SIDE_BUY };
        let timestamp = self.get_synced_timestamp();
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
        if self.is_hedge_mode.load(Ordering::Relaxed) {
            buf.push_str("&positionSide=");
            buf.push_str(if is_long { "LONG" } else { "SHORT" });
        }
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
        let api_secret = self.api_secret.load();
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

        let final_tp = Self::round_price_to_tick(take_profit_price, tick_size, is_long_close);
        let final_sl = Self::round_price_to_tick(stop_loss_price, tick_size, is_long_close);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] OCO Limit/Stop interceptada para {} @ TP: {} / SL: {}. Se maneja localmente.", symbol, final_tp, final_sl);
            return Ok(());
        }

        let side = if is_long_close { SIDE_SELL } else { SIDE_BUY };
        let timestamp = self.get_synced_timestamp();
        self.check_rate_limits(timestamp)?;

        let base_url = if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        };

        // R3.1 — modo de posición condicional: en HEDGE, positionSide LONG/SHORT
        // hace la orden inherentemente reductora (Binance RECHAZA reduceOnly
        // combinado con positionSide). En ONE-WAY, positionSide está prohibido
        // (-4061) y la protección correcta es reduceOnly=true. Antes las
        // piernas enviaban positionSide incondicionalmente: toda cuenta one-way
        // recibía el bracket entero rechazado — posición desnuda.
        let is_hedge = self.is_hedge_mode.load(Ordering::Relaxed);
        let (position_side_q, reduce_only_q) = if is_hedge {
            (
                format!(
                    "&positionSide={}",
                    if is_long_close { "LONG" } else { "SHORT" }
                ),
                String::new(),
            )
        } else {
            (String::new(), "&reduceOnly=true".to_string())
        };

        // 1. Build Stop Loss Order (STOP_MARKET)
        let mut sl_buf = ZeroAllocBuffer::new();
        sl_buf.push_str(base_url);
        let sl_payload_start = sl_buf.as_str().len();
        sl_buf.push_str("symbol=");
        sl_buf.push_str(symbol);
        sl_buf.push_str("&side=");
        sl_buf.push_str(side);
        sl_buf.push_str(&position_side_q);
        sl_buf.push_str(&reduce_only_q);
        sl_buf.push_str("&type=STOP_MARKET");
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
        tp_buf.push_str(&position_side_q);
        tp_buf.push_str(&reduce_only_q);
        tp_buf.push_str("&type=TAKE_PROFIT_MARKET");
        tp_buf.push_str("&quantity=");
        tp_buf.push_f64(final_quantity);
        tp_buf.push_str("&stopPrice=");
        tp_buf.push_f64(final_tp);
        tp_buf.push_str("&newClientOrderId=");
        tp_buf.push_str(&format!("{}_TP", base_client_id));
        tp_buf.push_str("&timestamp=");
        tp_buf.push_u64(timestamp);

        let mut confirmed_sl_id: Option<String> = None;
        let mut confirmed_tp_id: Option<String> = None;

        let api_secret = self.api_secret.load();

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
            confirmed_sl_id = Some(format!("{}_SL", base_client_id));
        }
        if let Ok(limits) = &tp_res {
            self.update_limits(limits);
            confirmed_tp_id = Some(format!("{}_TP", base_client_id));
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
            // D-02 — RETRY CON REGENERACIÓN COMPLETA (5º informe): el retry
            // anterior reenviaba el MISMO buffer firmado — timestamp vencido
            // (-1021) o coid duplicado (-4116) garantizados. Ahora: nuevo
            // timestamp NTP, nuevo clientOrderId (sufijo _R), nueva firma
            // HMAC. Mismo patrón que el flatten retry (línea ~519).
            let ts_retry = self.get_synced_timestamp();
            let retry_secret = self.api_secret.load();

            if sl_down {
                let mut rb = ZeroAllocBuffer::new();
                rb.push_str(base_url);
                let rp = rb.as_str().len();
                rb.push_str("symbol=");
                rb.push_str(symbol);
                rb.push_str("&side=");
                rb.push_str(side);
                rb.push_str(&position_side_q);
                rb.push_str(&reduce_only_q);
                rb.push_str("&type=STOP_MARKET");
                rb.push_str("&quantity=");
                rb.push_f64(final_quantity);
                rb.push_str("&stopPrice=");
                rb.push_f64(final_sl);
                rb.push_str("&newClientOrderId=");
                rb.push_str(&format!("{}_SLR", base_client_id));
                rb.push_str("&timestamp=");
                rb.push_u64(ts_retry);
                let mut rsb = [0u8; 64];
                sign_payload_to_buffer(&rb.as_str()[rp..], &retry_secret, &mut rsb);
                rb.push_str("&signature=");
                rb.push_str(unsafe { std::str::from_utf8_unchecked(&rsb) });
                sl_res = self.client.execute_order_payload(rb.as_str()).await;
                if let Ok(limits) = &sl_res {
                    self.update_limits(limits);
                    confirmed_sl_id = Some(format!("{}_SLR", base_client_id));
                }
            }
            if tp_down {
                let mut rb = ZeroAllocBuffer::new();
                rb.push_str(base_url);
                let rp = rb.as_str().len();
                rb.push_str("symbol=");
                rb.push_str(symbol);
                rb.push_str("&side=");
                rb.push_str(side);
                rb.push_str(&position_side_q);
                rb.push_str(&reduce_only_q);
                rb.push_str("&type=TAKE_PROFIT_MARKET");
                rb.push_str("&quantity=");
                rb.push_f64(final_quantity);
                rb.push_str("&stopPrice=");
                rb.push_f64(final_tp);
                rb.push_str("&newClientOrderId=");
                rb.push_str(&format!("{}_TPR", base_client_id));
                rb.push_str("&timestamp=");
                rb.push_u64(ts_retry);
                let mut rsb = [0u8; 64];
                sign_payload_to_buffer(&rb.as_str()[rp..], &retry_secret, &mut rsb);
                rb.push_str("&signature=");
                rb.push_str(unsafe { std::str::from_utf8_unchecked(&rsb) });
                tp_res = self.client.execute_order_payload(rb.as_str()).await;
                if let Ok(limits) = &tp_res {
                    self.update_limits(limits);
                    confirmed_tp_id = Some(format!("{}_TPR", base_client_id));
                }
            }
        }

        if sl_res.is_err() && tp_res.is_err() {
            return Err("Ambas órdenes OCO fallaron.".to_string());
        }
        if sl_res.is_err() || tp_res.is_err() {
            // D-416: Cancelar el client_order_id exacto que fue confirmado en Binance (sea _SL, _SLR, _TP o _TPR)
            let mut cancelled_ids = Vec::new();
            if sl_res.is_ok() {
                if let Some(ref sl_id) = confirmed_sl_id {
                    let _ = self.cancel_order(symbol, sl_id).await;
                    cancelled_ids.push(sl_id.clone());
                }
            }
            if tp_res.is_ok() {
                if let Some(ref tp_id) = confirmed_tp_id {
                    let _ = self.cancel_order(symbol, tp_id).await;
                    cancelled_ids.push(tp_id.clone());
                }
            }
            return Err(format!(
                "OCO PARCIAL: pierna fallida tras retry; pierna(s) confirmada(s) [{}] cancelada(s). Posición SIN protección — aplanar o alertar.",
                cancelled_ids.join(", ")
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

        let timestamp = self.get_synced_timestamp();

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
        let api_secret = self.api_secret.load();
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

    /// Cancela todas las órdenes activas en Binance para un símbolo específico
    #[inline(always)]
    async fn cancel_all_symbol_orders(&self, symbol: &str) -> Result<(), String> {
        if self.is_paper_trading {
            println!(
                "📝 [PAPER TRADING LOCAL] Todas las órdenes canceladas en {}",
                symbol
            );
            return Ok(());
        }

        let timestamp = self.get_synced_timestamp();
        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/allOpenOrders?"
        } else {
            "https://fapi.binance.com/fapi/v1/allOpenOrders?"
        });
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.load();
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

    /// D-371: Cancela quirúrgicamente únicamente las órdenes OCO asociadas a la posición especificada
    #[inline(always)]
    async fn cancel_position_oco_orders(
        &self,
        symbol: &str,
        is_long: bool,
    ) -> Result<usize, String> {
        let pos_side = if is_long { "LONG" } else { "SHORT" };
        let active = self.order_registry.active_for_symbol(symbol);
        let mut canceled = 0;
        for o in active {
            let is_oco_bracket = o.client_order_id.contains("_SL")
                || o.client_order_id.contains("_TP")
                || o.client_order_id.contains("oco_");
            let matches_side = o.position_side == pos_side
                || o.position_side == "BOTH"
                || o.position_side.is_empty();
            if is_oco_bracket && matches_side {
                if self.cancel_order(symbol, &o.client_order_id).await.is_ok() {
                    canceled += 1;
                }
            }
        }
        Ok(canceled)
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

        let timestamp = self.get_synced_timestamp();

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
        let api_secret = self.api_secret.load();
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
        let timestamp = self.get_synced_timestamp();

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
        let api_secret = self.api_secret.load();
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
        let timestamp = self.get_synced_timestamp();

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
        let api_secret = self.api_secret.load();
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
        let timestamp = self.get_synced_timestamp();

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
        let api_secret = self.api_secret.load();
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
        let timestamp = self.get_synced_timestamp();

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
        sign_payload_to_buffer(payload, &self.api_secret.load(), &mut sig_buf);
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
        let filter = self.get_symbol_filter(symbol).await;
        Ok(filter.min_notional)
    }
}
