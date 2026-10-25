use crate::binance_api::{
    sign_payload_to_buffer, ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL,
    TIME_IN_FORCE_IOC,
};
use crate::order_types::{FillReport, OrderAck};
use crate::ExecutionPayload;
use arc_swap::{ArcSwap, ArcSwapOption};
use risk_engine::ValidatedOrder;
use signal_engine::SignalType;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::client::{BinanceClient, ZeroAllocBuffer};

/// Inverso de un paso de precio o cantidad, ajustado al entero más cercano
/// cuando el paso es una potencia de diez fraccionaria (0,1 … 1e-12).
///
/// D-629 (DÉCIMA OLA): `1/paso` no es exacto para pasos como 1e-5 (da
/// 99999,99999999999). Como `ZeroAllocBuffer::push_f64` imprime la
/// representación mínima de `ryu`, el residuo de `k / inverso` llegaba
/// literalmente al query de la orden y Binance la rechazaba por precisión.
#[inline(always)]
fn exact_inverse(step: f64) -> f64 {
    let inv = 1.0 / step;
    let r = inv.round();
    if r >= 1.0 && (inv - r).abs() <= 1e-6 * r {
        r
    } else {
        inv
    }
}

/// Tolerancia RELATIVA para considerar que un valor escalado ya es múltiplo
/// exacto del paso.
///
/// D-629 (DÉCIMA OLA): el épsilon anterior era ABSOLUTO (1e-9) y se sumaba al
/// valor escalado. Su efecto relativo dependía de la magnitud: con 1.500.000
/// unidades quedaba por debajo de la resolución de f64 y dejaba de corregir
/// nada, de modo que un 2999,9999999 se truncaba a 2999 en algunos símbolos.
const STEP_SNAP_REL_TOL: f64 = 1e-9;

#[inline(always)]
fn snap_floor(scaled: f64) -> f64 {
    let r = scaled.round();
    if (scaled - r).abs() <= STEP_SNAP_REL_TOL * r.abs().max(1.0) {
        r
    } else {
        scaled.floor()
    }
}

#[inline(always)]
fn snap_ceil(scaled: f64) -> f64 {
    let r = scaled.round();
    if (scaled - r).abs() <= STEP_SNAP_REL_TOL * r.abs().max(1.0) {
        r
    } else {
        scaled.ceil()
    }
}

/// D-628 (DÉCIMA OLA): precio límite pasivo que se UNE al mejor nivel del
/// propio lado a partir del precio medio.
///
/// El cálculo anterior desplazaba un tick COMPLETO desde el medio. Con spread
/// de un tick —el régimen dominante en los pares líquidos— eso colocaba la
/// orden un nivel por DETRÁS del mejor bid (compra) o del mejor ask (venta):
/// peor prioridad de cola, menos ejecución y selección adversa en las que sí
/// se llenaban. Desplazar MEDIO tick y redondear hacia el propio lado da:
///
/// ```text
/// spread 1 tick: compra = bid,        venta = ask
/// spread 2 ticks: compra = bid,       venta = ask
/// spread 3 ticks: compra = bid + 1t,  venta = ask − 1t   (dentro del spread)
/// ```
///
/// Sin bid/ask contemporáneos no se puede certificar pasividad localmente.
/// GTX deja al exchange rechazar una orden que tomaría liquidez; ese rechazo
/// no debe eludirse convirtiendo automáticamente la orden a taker.
#[inline(always)]
pub(crate) fn passive_join_price(mid: f64, tick: f64, is_sell: bool) -> f64 {
    if !mid.is_finite() || mid <= 0.0 || !tick.is_finite() || tick <= 0.0 {
        return 0.0;
    }
    let half = 0.5 * tick;
    let target = if is_sell {
        mid + half
    } else {
        (mid - half).max(tick)
    };
    let inv = exact_inverse(tick);
    let scaled = target * inv;
    if is_sell {
        snap_ceil(scaled) / inv
    } else {
        snap_floor(scaled) / inv
    }
}

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
    /// D-726 (DÉCIMA OLA · auditoría integral): apalancamiento REAL de la
    /// posición, tal y como lo devuelve `/fapi/v2/positionRisk`. Se descartaba
    /// al parsear y el restaurador lo sustituía por el literal 10, de modo que
    /// el margen reconstruido de una posición a 20x era el doble del real.
    /// 0 = el exchange no lo informó.
    pub leverage: f64,
    /// #595 — última modificación reportada por el exchange (`updateTime`):
    /// cota inferior de la edad de la posición. 0 = no informado. La
    /// adopción (FASE 5) la usa como reloj de entrada en vez del arranque
    /// del proceso.
    pub update_time: u64,
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
    /// CL-39: `Ok(())` sólo si la orden ejecutó algo (total o parcial).
    /// Terminal sin ejecución ⇒ `Err("IOC_UNFILLED…")` (no ambiguo); sin
    /// evidencia terminal ⇒ `Err("AMBIGUOUS…")`.
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
    async fn cancel_all_symbol_orders(&self, symbol: &str) -> Result<(), String>;

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
    /// N5 (CL-177): uso que reportan las cabeceras del exchange, por contador y
    /// con su propio sello (LG-3: cada uno vale dentro de su ventana).
    uso_exchange: std::sync::Mutex<UsoExchange>,
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
    /// FIX M4-C02 (quinta ola, CRITICAL): freno TEMPORAL por rate-limit
    /// (3×429 escalado o HTTP 418). A diferencia de `kill_switch` —permanente y
    /// reservado para incidentes reales (immune/X-009/insolvencia)— este freno
    /// AUTO-EXPIRA en este timestamp, porque un 429/418 es transitorio (Binance
    /// indica retry_after). Antes: trigger_kill_switch() latcheaba la ejecución
    /// PARA SIEMPRE por un burst de rate-limit, dejando posiciones vivas sin
    /// poder cerrarse (riesgo de liquidación) hasta reiniciar el proceso.
    rate_brake_until_ms: AtomicU64,
    /// Cache en RAM de filtros de símbolos (tickSize, stepSize, minNotional) O(1) < 5ns
    symbol_filters: ArcSwap<std::collections::HashMap<String, SymbolFilter>>,
    /// F1-105: Caché negativa de símbolos desconocidos/no parseables con TTL (60s)
    negative_symbol_cache: std::sync::RwLock<std::collections::HashMap<String, std::time::Instant>>,
    pub arena: ArcSwapOption<quantum_arena::GlobalArena>,
    pub ws: std::sync::Arc<crate::ws_executor::WsExecutor>,
    pub is_hedge_mode: AtomicBool,
}

/// FIX M4-C01 (quinta ola, CRITICAL): resultado del lazo de confirmación de
/// órdenes. En la ruta WS, `send_order_payload` retorna Ok(()) cuando el
/// kernel acepta los bytes — SIN ack del exchange. La confirmación real llega
/// por ORDER_TRADE_UPDATE (user-data stream → OrderRegistry) o por consulta
/// REST acotada (NUNCA re-envío: riesgo de duplicación).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderResolution {
    /// El exchange aceptó la orden (NEW/PARTIALLY_FILLED/FILLED con orderId)
    /// o hubo fill real antes de un estado terminal.
    Accepted,
    /// Terminal con cero fills y evidencia de identidad del exchange.
    /// No incluye errores de consulta como -2013 (ausencia no es rechazo).
    Rejected,
    /// Sin verificación concluyente (red caída en ambos planos). El llamador
    /// conserva el estado local; posición agregada no acredita la intención.
    Timeout,
}

impl OrderExecutor {
    pub fn new(api_key: String, api_secret: String, is_testnet: bool) -> Self {
        Self {
            api_secret: ArcSwap::from_pointee(api_secret.clone()),
            client: BinanceClient::new(api_key.clone(), is_testnet),
            rate_limit_counter: AtomicUsize::new(0),
            last_reset_timestamp: AtomicU64::new(0),
            uso_exchange: std::sync::Mutex::new(UsoExchange::default()),
            // LG-12: los límites documentados del exchange.
            max_weight_1m: AtomicUsize::new(crate::binance_api::LIMITE_PESO_1M),
            max_orders_10s: AtomicUsize::new(crate::binance_api::LIMITE_ORDENES_10S),
            max_orders_1m: AtomicUsize::new(crate::binance_api::LIMITE_ORDENES_1M),
            kill_switch: AtomicBool::new(false),
            active_leverage: ArcSwap::from_pointee(std::collections::HashMap::new()),
            is_paper_trading: false, // Desacoplado: false por defecto para permitir órdenes reales en Testnet/Mainnet; activar con set_paper_trading(true) si se desea simular
            order_registry: std::sync::Arc::new(crate::order_registry::OrderRegistry::new()),
            cooldown_until_ms: AtomicU64::new(0),
            consecutive_429: AtomicUsize::new(0),
            rate_brake_until_ms: AtomicU64::new(0),
            symbol_filters: ArcSwap::from_pointee(std::collections::HashMap::new()),
            negative_symbol_cache: std::sync::RwLock::new(std::collections::HashMap::new()),
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
            uso_exchange: std::sync::Mutex::new(UsoExchange::default()),
            // LG-12: los límites documentados del exchange.
            max_weight_1m: AtomicUsize::new(crate::binance_api::LIMITE_PESO_1M),
            max_orders_10s: AtomicUsize::new(crate::binance_api::LIMITE_ORDENES_10S),
            max_orders_1m: AtomicUsize::new(crate::binance_api::LIMITE_ORDENES_1M),
            kill_switch: AtomicBool::new(false),
            active_leverage: ArcSwap::from_pointee(std::collections::HashMap::new()),
            is_paper_trading: false,
            order_registry,
            cooldown_until_ms: AtomicU64::new(0),
            consecutive_429: AtomicUsize::new(0),
            rate_brake_until_ms: AtomicU64::new(0),
            symbol_filters: ArcSwap::from_pointee(std::collections::HashMap::new()),
            negative_symbol_cache: std::sync::RwLock::new(std::collections::HashMap::new()),
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

    /// FIX M4-C01: espera la resolución de una orden por clientOrderId.
    /// Invariante: `register_intent` escribe status=New y order_id=0 ANTES del
    /// envío — solo cuenta como confirmación lo que llega DEL exchange:
    /// order_id > 0 (asignado por apply_ack / apply_trade_update) o fills
    /// reales (executed_qty > 0). Intents sin confirmar siguen esperando
    /// hasta el deadline.
    pub async fn await_resolution(
        &self,
        client_order_id: &str,
        timeout_ms: u64,
    ) -> OrderResolution {
        if self.is_paper_trading {
            return OrderResolution::Accepted;
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        loop {
            if let Some(o) = self.order_registry.get(client_order_id) {
                if o.executed_qty > 0.0 {
                    return OrderResolution::Accepted;
                }
                let exchange_confirmed = o.order_id > 0;
                match o.status {
                    crate::order_registry::OrderStatus::New
                    | crate::order_registry::OrderStatus::PartiallyFilled
                    | crate::order_registry::OrderStatus::Filled => {
                        if exchange_confirmed {
                            return OrderResolution::Accepted;
                        }
                    }
                    crate::order_registry::OrderStatus::Rejected
                    | crate::order_registry::OrderStatus::Expired
                    | crate::order_registry::OrderStatus::Canceled => {
                        if exchange_confirmed {
                            return OrderResolution::Rejected;
                        }
                    }
                    crate::order_registry::OrderStatus::Unknown => {}
                }
            }
            if std::time::Instant::now() >= deadline {
                return OrderResolution::Timeout;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    }

    /// Consulta REST sin reenvío. Un error de consulta (incluido -2013) no
    /// prueba rechazo; la evidencia se valida ANTES de escribir el registro.
    pub async fn resolve_via_rest(&self, symbol: &str, client_order_id: &str) -> OrderResolution {
        if self.is_paper_trading {
            return OrderResolution::Accepted;
        }
        let result = self.query_order(symbol, client_order_id).await;
        self.record_query_resolution(symbol, client_order_id, result.as_ref().map_err(String::as_str))
    }

    fn record_query_resolution(
        &self,
        symbol: &str,
        client_order_id: &str,
        result: Result<&OrderAck, &str>,
    ) -> OrderResolution {
        let resolution = crate::execution_evidence::classify_query_result(symbol, client_order_id, result);
        if resolution == OrderResolution::Timeout {
            return resolution;
        }
        let ack = result.expect("validated query evidence");
        if let Some(prior) = self.order_registry.get(client_order_id) {
            if prior.symbol != symbol || prior.side != ack.side
                || (prior.order_id != 0 && prior.order_id != ack.order_id)
            {
                return OrderResolution::Timeout;
            }
        }
        self.order_registry.apply_ack(ack, self.get_synced_timestamp());
        // A late terminal snapshot with zero fills cannot erase positive
        // cumulative fill evidence already merged from WS/REST.
        if let Some(merged) = self.order_registry.get(client_order_id) {
            if merged.executed_qty.is_finite() && merged.executed_qty > 0.0 {
                return OrderResolution::Accepted;
            }
        }
        resolution
    }

    /// FIX M4-C01: gate de confirmación para envíos WS. Retorna Ok(()) SOLO
    /// con confirmación del exchange. Err clasificado para el host:
    /// · "WS_ORDER_REJECTED…" → terminal observado sin fills conocidos.
    /// · "AMBIGUOUS…" → inconcluso: conservar reserva, no certificar posición.
    /// El rollback por reserva/intención en el host sigue siendo necesario.
    /// Latencia happy-path: el ORDER_TRADE_UPDATE del fill llega en <500ms
    /// típicos; el timeout de 8s + 3 queries REST solo corre cuando el
    /// user-data stream falló.
    async fn confirm_ws_dispatch(
        &self,
        symbol: &str,
        client_order_id: &str,
    ) -> Result<(), String> {
        let mut resolution = self.await_resolution(client_order_id, 8_000).await;
        if resolution == OrderResolution::Timeout {
            for attempt in 1..=3u64 {
                tokio::time::sleep(std::time::Duration::from_millis(400 * attempt)).await;
                resolution = self.resolve_via_rest(symbol, client_order_id).await;
                if resolution != OrderResolution::Timeout {
                    break;
                }
            }
        }
        match resolution {
            OrderResolution::Accepted => Ok(()),
            OrderResolution::Rejected => Err(format!(
                "WS_ORDER_REJECTED: {} terminal confirmado sin fills conocidos",
                client_order_id
            )),
            OrderResolution::Timeout => Err(format!(
                "AMBIGUOUS: {} sin confirmación concluyente tras WS + 3 consultas REST",
                client_order_id
            )),
        }
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

    /// Consulta firmada de sólo lectura: true=hedge, false=one-way.
    /// No modifica la cuenta ni publica un estado operativo local.
    /// En simulación no existe evidencia remota: devolver error, no asumir modo.
    pub async fn fetch_hedge_mode(&self) -> Result<bool, String> {
        if self.is_paper_trading {
            return Err("POSITION_MODE_UNAVAILABLE_IN_PAPER".to_string());
        }
        let timestamp = self.get_synced_timestamp();
        let api_secret = self.api_secret.load();
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
        let (_, body) = self.get_payload_account(buf.as_str()).await?;
        #[derive(serde::Deserialize)]
        struct ModeResp {
            #[serde(rename = "dualSidePosition")]
            dual: bool,
        }
        serde_json::from_str::<ModeResp>(&body)
            .map(|mode| mode.dual)
            .map_err(|e| format!("POSITION_MODE_PARSE: {}", e))
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

        let (_, body) = self.get_payload_account(buf.as_str()).await?;
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
            Ok((limits, _)) => {
                self.update_limits(&limits);
                self.is_hedge_mode.store(true, Ordering::Relaxed);
                println!("🔀 [POSITION-MODE] Cuenta migrada a HEDGE (dualSidePosition=true)");
                Ok(true)
            }
            Err(e) => {
                let e = self.handle_rate_limit_error(&e);
                // N-05: Si falla activar hedge (ej. posiciones abiertas en one-way), reflejar la verdad local
                self.is_hedge_mode.store(false, Ordering::Relaxed);
                Err(format!(
                    "No se pudo activar modo hedge: {}. Cuenta permanece en ONE-WAY mode.",
                    e
                ))
            }
        }
    }

    /// F1-027: Cancela órdenes normales y algo brackets de un símbolo específico.
    async fn cancel_symbol_orders_and_brackets(&self, sym: &str, api_secret: &str) -> (usize, bool) {
        let mut cancelled = 0usize;
        let mut all_purged = true;
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
        sign_payload_to_buffer(&buf.as_str()[payload_start..], api_secret, &mut sig_buf);
        buf.push_str("&signature=");
        buf.push_str(unsafe { std::str::from_utf8_unchecked(&sig_buf) });
        // N5 (CL-178): durante un 429/418 el DELETE no sale; la purga queda
        // incompleta y el llamador marca la protección sucia. CL-179: cupo
        // crítico, soltado tras la respuesta. CL-182: su cabecera se registra
        // con el cupo vivo, para que el coste no salga del libro antes de
        // entrar en el uso.
        let borrado = match self.check_exit_rate_limits(
            ts,
            Tramo::Critica(crate::binance_api::COSTE_CANCELACION),
        ) {
            Ok(_cupo) => self
                .client
                .cancel_order_payload(buf.as_str())
                .await
                .inspect(|limits| self.update_limits(limits)),
            Err(e) => Err(e),
        };
        match borrado {
            Ok(_) => {
                cancelled += 1;
            }
            Err(e) => {
                let _ = self.handle_rate_limit_error(&e);
                // Si el error no es "no open orders" (-2011), la purga de órdenes falló
                if !e.contains("-2011") && !e.contains("Unknown order") {
                    all_purged = false;
                }
            }
        }
        match self.cancel_all_algo_open_orders(sym).await {
            Ok(()) => {
                cancelled += 1;
            }
            Err(e) => {
                if !e.contains("no open algo orders") && !e.contains("-2011") {
                    all_purged = false;
                }
            }
        }
        (cancelled, all_purged)
    }

    /// FASE 13: Emergency Flatten All.
    /// F1-027 (Protocolo X-009): Cierra PRIMERO cada posición abierta para que la protección
    /// siga viva hasta el aterrizaje del cierre; sólo tras confirmación purga los brackets de ese símbolo.
    /// F1-029: Sondeo WS ampliado a 200 ms y COID fresco en fallback REST para evitar -4116 falso.
    pub async fn flatten_all_positions(&self) -> Result<(usize, usize), String> {
        if self.is_paper_trading {
            return Ok((0, 0));
        }
        let timestamp = self.get_synced_timestamp();
        // N5 (RV14): durante un 429/418 no sale nada del aplanado (tampoco
        // positionRisk, que lo mira); se devuelve el freno aquí, antes de
        // firmar nada y sin anunciar un respaldo de modo que no se usaría.
        self.check_backoff(timestamp)?;
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
        // N5 (CL-179): es una lectura. Por encima del 80 % no se pregunta el
        // modo y se usa el registrado (el mismo respaldo que ante un fallo);
        // positionRisk es crítica y sigue.
        let modo = match self.check_exit_rate_limits(timestamp, Tramo::Lectura) {
            Ok(_cupo) => self.get_payload_account(buf.as_str()).await,
            Err(e) => Err(e),
        };
        let dual_mode_opt = match modo {
            Ok((limits, mode_body)) => {
                self.update_limits(&limits);
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
        let mut dual = dual_mode_opt.unwrap_or(self.is_hedge_mode.load(Ordering::Relaxed));

        let entries = self.fetch_position_risk().await?;
        let open_positions: Vec<_> = entries.iter().filter(|p| p.is_open()).cloned().collect();
        let sin_posicion = simbolos_sin_posicion(&entries);

        let mut closed = 0usize;
        let mut cancelled = 0usize;
        let mut unclosed_symbols = Vec::new();
        let mut unpurged_symbols = Vec::new();

        // 1) Cerrar PRIMERO cada posición abierta.
        for p in &open_positions {
            let is_long = p.is_long();
            let mut position_closed = false;

            // WS Execution Routing (Zero-TLS overhead) si está conectado
            if self.ws.is_connected() {
                let api_key = self
                    .client
                    .api_key
                    .load()
                    .to_str()
                    .unwrap_or("")
                    .to_string();
                let coid_ws = uuid::Uuid::now_v7().simple().to_string();
                let ts_ws = self.get_synced_timestamp();
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
                    !dual,
                    &coid_ws,
                    ts_ws,
                ) {
                    println!(
                        "⚠️ [WS-EXECUTOR] Failed to send close position order, falling back to REST: {}",
                        e
                    );
                } else {
                    // F1-029: Ventana de confirmación ampliada a 200ms (20×10ms) para acomodar RTT real
                    for _ in 0..20 {
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        if let Some(order) = self.order_registry.get(&coid_ws) {
                            if !order.status.is_active() && order.executed_qty > 0.0 {
                                position_closed = true;
                                println!(
                                    "🧹 [FLATTEN] {} cerrada por WS (ejecución confirmada)",
                                    p.symbol
                                );
                                break;
                            }
                        }
                    }
                    if !position_closed {
                        println!(
                            "⚠️ [FLATTEN] {} despachada por WS SIN confirmación de ejecución: se reintenta por REST con los fail-safes",
                            p.symbol
                        );
                    }
                }
            }

            if !position_closed {
                // Fallback a REST tipado con COID fresco independiente del WS (F1-029: evita duplicado -4116)
                let ts = self.get_synced_timestamp();
                let coid_rest = uuid::Uuid::now_v7().simple().to_string();
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
                    buf.push_str("&type=MARKET&newOrderRespType=RESULT&quantity=");
                } else {
                    buf.push_str("&type=MARKET&reduceOnly=true&newOrderRespType=RESULT&quantity=");
                }
                buf.push_f64(p.position_amt.abs());
                buf.push_str("&newClientOrderId=");
                buf.push_str(&coid_rest);
                buf.push_str("&timestamp=");
                buf.push_u64(ts);
                let mut sig_buf = [0u8; 64];

                sign_payload_to_buffer(&buf.as_str()[payload_start..], &api_secret, &mut sig_buf);
                buf.push_str("&signature=");
                buf.push_str(unsafe { std::str::from_utf8_unchecked(&sig_buf) });

                // N5 (CL-178): el cierre no sale durante un 429/418, y cada
                // respuesta pasa UNA vez por la política de rate-limit (antes
                // sus 429 no armaban el cooldown y el barrido seguía enviando).
                // Un rechazo local no lleva -4061 ni -2022: no reintenta.
                // CL-179: cupo crítico (sale mientras quepa en el límite).
                // CL-182: la cabecera se registra con el cupo vivo.
                let mut close_res = match self.check_exit_rate_limits(
                    self.get_synced_timestamp(),
                    Tramo::Critica(crate::binance_api::COSTE_ORDEN),
                ) {
                    Ok(_cupo) => self
                        .client
                        .execute_order_payload_typed(buf.as_str())
                        .await
                        .map_err(|e| self.handle_rate_limit_error(&e))
                        .inspect(|(limits, _)| self.update_limits(limits)),
                    Err(e) => Err(e),
                };
                // R3.3 / K-16: Fail-safe retry si el exchange rechaza por discrepancia de modo (-4061)
                if let Err(ref e) = close_res {
                    if e.contains("-4061") || e.contains("position side") || e.contains("reduceOnly") {
                        println!(
                            "⚠️ [FLATTEN] Modo rechazado (-4061) para {}. Conmutando modo (dual={}) y reintentando cierre...",
                            p.symbol, !dual
                        );
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
                            retry_buf.push_str("&type=MARKET&newOrderRespType=RESULT&quantity=");
                        } else {
                            retry_buf.push_str("&type=MARKET&reduceOnly=true&newOrderRespType=RESULT&quantity=");
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

                        close_res = match self.check_exit_rate_limits(
                            self.get_synced_timestamp(),
                            Tramo::Critica(crate::binance_api::COSTE_ORDEN),
                        ) {
                            Ok(_cupo) => self
                                .client
                                .execute_order_payload_typed(retry_buf.as_str())
                                .await
                                .map_err(|e| self.handle_rate_limit_error(&e))
                                .inspect(|(limits, _)| self.update_limits(limits)),
                            Err(e) => Err(e),
                        };
                    }
                }

                match close_res {
                    Ok((_, ack)) => {
                        let now = self.get_synced_timestamp();
                        self.order_registry.apply_ack(&ack, now);
                        position_closed = true;
                        println!(
                            "🧹 [FLATTEN] {} cerrada {} @ ~{} (status {})",
                            p.symbol,
                            p.position_amt.abs(),
                            ack.avg_price,
                            ack.status
                        );
                    }
                    Err(e) if e.contains("-2022") || e.contains("-4061") => {
                        // X-026 (REHAB-3): ReduceOnly rechazado — un TP llenó (o el
                        // modo cambió) ENTRE el snapshot y el close.
                        println!(
                            "♻️ [X-026] {} -2022: re-snapshot fresco (TP pudo llenar entre snapshot y close)…",
                            p.symbol
                        );
                        // CL-183: un re-snapshot que no respondió (freno 429/418,
                        // cuota del tramo crítico o red) no es «posición plana»:
                        // antes el Err caía a None, se daba por cerrada y se
                        // purgaban los brackets de una posición quizá viva.
                        let fresh = self.fetch_position_risk().await.map(|entries| {
                            entries
                                .into_iter()
                                .find(|e2| e2.symbol == p.symbol && e2.is_open())
                        });
                        match fresh {
                            Ok(Some(f)) => {
                                let fq = f.position_amt.abs();
                                let step = match self.get_symbol_filter(&p.symbol).await {
                                    Ok(flt) => flt.step_size,
                                    Err(e) => {
                                        println!(
                                            "⚠️ [X-026] {} sin filtro de precisión ({}): qty fresca sin re-cuantizar (paso 1e-9)",
                                            p.symbol, e
                                        );
                                        1e-9
                                    }
                                };
                                let retry = self
                                    .execute_reduce_only_market(&p.symbol, f.is_long(), fq, step)
                                    .await;
                                match retry {
                                    Ok(()) => {
                                        position_closed = true;
                                        println!("🧹 [X-026] {} cerrada con qty fresca {}", p.symbol, fq);
                                    }
                                    Err(e3) => println!(
                                        "🚨 [X-026] {} sigue ABIERTA (qty {}) tras re-snapshot+retry: {} — INTERVENCIÓN MANUAL",
                                        p.symbol, fq, e3
                                    ),
                                }
                            }
                            Ok(None) => {
                                position_closed = true; // ya no existe con qty>0: estaba plana
                                println!(
                                    "🧹 [X-026] {} ya plana en el exchange (snapshot fresco)",
                                    p.symbol
                                );
                            }
                            Err(e2) => println!(
                                "🚨 [X-026] {} sin re-snapshot ({}): se trata como ABIERTA y sus brackets no se purgan — INTERVENCIÓN MANUAL",
                                p.symbol, e2
                            ),
                        }
                    }
                    Err(e) => println!(
                        "🚨 [FLATTEN] {} NO cerrada tras reintento: {} — INTERVENCIÓN MANUAL",
                        p.symbol, e
                    ),
                }
            }

            if position_closed {
                closed += 1;
                // F1-027: Sólo purgar brackets de ESTE símbolo una vez que el cierre aterrizó
                let (sym_canc, purged) = self.cancel_symbol_orders_and_brackets(&p.symbol, &api_secret).await;
                cancelled += sym_canc;
                if !purged {
                    unpurged_symbols.push(p.symbol.clone());
                    quantum_arena::protection_health::mark_dirty();
                }
            } else {
                // F1-027: Si el cierre falló, NO purgar protección (evitar posición desnuda)
                unclosed_symbols.push(p.symbol.clone());
            }
        }

        // Purgar brackets de símbolos sin posiciones abiertas (órdenes huérfanas)
        // N5 (CL-179): el barrido es limpieza, no protección: cada símbolo pasa
        // por el tramo de lectura antes de sus dos cancelaciones críticas, para
        // que no gaste la franja reservada a cierres y protección. Lo que no
        // entra queda sin purgar y marca la protección sucia.
        for sym in &sin_posicion {
            if let Err(e) = self.check_exit_rate_limits(self.get_synced_timestamp(), Tramo::Lectura) {
                println!("⚠️ [FLATTEN] barrido de huérfanas sin cuota en {}: {}", sym, e);
                unpurged_symbols.push(sym.clone());
                quantum_arena::protection_health::mark_dirty();
                continue;
            }
            let (sym_canc, purged) = self.cancel_symbol_orders_and_brackets(sym, &api_secret).await;
            cancelled += sym_canc;
            if !purged {
                unpurged_symbols.push(sym.clone());
                quantum_arena::protection_health::mark_dirty();
            }
        }

        if !unpurged_symbols.is_empty() {
            println!(
                "⚠️ [FLATTEN] Purga incompleta en {} símbolos (se marcó protection_health dirty): {:?}",
                unpurged_symbols.len(),
                unpurged_symbols
            );
        }

        if !unclosed_symbols.is_empty() {
            Err(format!(
                "FLATTEN_INCOMPLETE: {} posiciones no pudieron cerrarse: {:?}",
                unclosed_symbols.len(),
                unclosed_symbols
            ))
        } else {
            Ok((cancelled, closed))
        }
    }

    /// F3.5: one reported income page (PnL, commission, funding, other flows).
    /// This is not complete accounting or a trade-outcome certificate.
    /// `income_types`: empty or exactly one type; the endpoint filter is scalar.
    /// `limit`: 1..=1000, validated rather than silently clamped.
    pub async fn fetch_income(
        &self,
        income_types: &[&str],
        start_ms: u64,
        limit: u32,
    ) -> Result<Vec<crate::order_types::IncomeEntry>, String> {
        // A single page is not a coverage certificate. Keep this API for callers
        // requesting one page; decision consumers use fetch_income_paged/window.
        self.fetch_income_page(income_types, start_ms, self.get_synced_timestamp(), 1, limit).await
    }

    async fn fetch_income_page(
        &self,
        income_types: &[&str],
        start_ms: u64,
        end_ms: u64,
        page: u32,
        limit: u32,
    ) -> Result<Vec<crate::order_types::IncomeEntry>, String> {
        let timestamp = self.get_synced_timestamp();
        let query = crate::income_evidence::income_page_query(
            start_ms, end_ms, page, limit, income_types, timestamp,
        ).map_err(|e| format!("INCOME_QUERY: {e:?}"))?;
        if self.is_paper_trading {
            return Ok(Vec::new());
        }
        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Lectura)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(self.client.get_base_url());
        buf.push_str("/fapi/v1/income?");
        let payload_start = buf.as_str().len();
        buf.push_str(&query);

        let mut sig_buf = [0u8; 64];
        sign_payload_to_buffer(
            &buf.as_str()[payload_start..],
            &self.api_secret.load(),
            &mut sig_buf,
        );
        buf.push_str("&signature=");
        buf.push_str(unsafe { std::str::from_utf8_unchecked(&sig_buf) });

        match self.get_payload_account(buf.as_str()).await {
            Ok((limits, body)) => {
                self.update_limits(&limits);
                serde_json::from_str(&body).map_err(|e| {
                    format!(
                        "INCOME_PARSE: {} body={}",
                        e,
                        crate::order_types::truncate(&body, 200)
                    )
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Bounded numbered traversal with a fixed inclusive interval. A short/empty
    /// page establishes observed exhaustion, not an immutable provider snapshot
    /// or completeness beyond provider retention. Paper is explicitly simulated.
    pub async fn fetch_income_window(
        &self,
        income_types: &[&str],
        start_ms: u64,
        end_ms: u64,
        max_pages: u32,
    ) -> Result<crate::income_evidence::IncomeWindow, String> {
        crate::income_evidence::income_page_query(
            start_ms, end_ms, 1, 1000, income_types, self.get_synced_timestamp(),
        ).map_err(|e| format!("INCOME_QUERY: {e:?}"))?;
        if max_pages == 0 {
            return Err("INCOME_QUERY: InvalidPageBudget".into());
        }
        if self.is_paper_trading {
            return Ok(crate::income_evidence::IncomeWindow {
                start_ms, end_ms, pages_read: 0,
                coverage: crate::income_evidence::IncomeCoverage::Simulated,
                entries: Vec::new(),
                quarantined: Vec::new(),
                exact_duplicates_dropped: 0,
                transport_error: None,
            });
        }
        crate::income_evidence::collect_income_window(
            start_ms, end_ms, max_pages, 1000,
            |page| self.fetch_income_page(income_types, start_ms, end_ms, page, 1000),
        ).await.map_err(|e| format!("INCOME_EVIDENCE: {e:?}"))
    }

    /// Compatibility API: never expose budget-truncated, stalled or simulated
    /// evidence as an exhausted real window. Existing suspensions are not cleared.
    pub async fn fetch_income_paged(
        &self,
        income_types: &[&str],
        start_ms: u64,
        max_pages: u32,
    ) -> Result<Vec<crate::order_types::IncomeEntry>, String> {
        self.fetch_income_window(income_types, start_ms, self.get_synced_timestamp(), max_pages)
            .await?
            .into_exhausted_entries()
            .map_err(|e| format!("INCOME_EVIDENCE: {e:?}"))
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

        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_POSITION_RISK))?;

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

        let res = self.get_payload_account(buf.as_str()).await;
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

    pub fn set_paper_trading(&mut self, is_paper: bool) {
        self.is_paper_trading = is_paper;
    }

    /// B1.2-fix: discriminador REAL de papel-vs-exchange. `is_demo_mode` del
    /// orquestador agrupa papel-local y testnet-live; solo el primero debe
    /// saltarse reconciliación/adopción (en testnet las posiciones son
    /// exposición real con brackets reales que el motor DEBE adoptar).
    #[inline(always)]
    pub fn is_paper_trading(&self) -> bool {
        self.is_paper_trading
    }

    #[inline(always)]
    fn update_limits(&self, limits: &crate::client::BinanceRateLimits) {
        let now_ms = self.get_synced_timestamp();
        self.observar_limites(limits, now_ms);
        // F1-005 (Ola Ω150): Solo reiniciar consecutive_429 cuando el cooldown activo ya expiró.
        // Antes, cualquier 2xx concurrente durante una ráfaga reiniciaba consecutive_429 a 0
        // en medio del cooldown, impidiendo alcanzar el umbral de 3+ 429 consecutivos.
        if now_ms >= self.cooldown_until_ms.load(Ordering::Relaxed) {
            self.consecutive_429.store(0, Ordering::Relaxed);
        }
    }

    /// N5 (CL-177): registra el uso de una respuesta. Sólo los contadores cuya
    /// cabecera llegó, cada uno con su sello: antes un único sello compartido
    /// renovaba la edad de los tres y la última respuesta en llegar pisaba el
    /// valor aunque fuera más vieja (peticiones concurrentes).
    fn observar_limites(&self, limits: &crate::client::BinanceRateLimits, ahora_ms: u64) {
        use crate::binance_api::{VENTANA_ORDENES_10S_MS, VENTANA_ORDENES_1M_MS, VENTANA_PESO_MS};
        let mut uso = self.uso_exchange_bloqueado();
        if let Some(w) = limits.weight_1m {
            uso.peso_1m.observar(w, ahora_ms, VENTANA_PESO_MS);
        }
        if let Some(o) = limits.orders_10s {
            uso.ordenes_10s.observar(o, ahora_ms, VENTANA_ORDENES_10S_MS);
        }
        if let Some(o) = limits.orders_1m {
            uso.ordenes_1m.observar(o, ahora_ms, VENTANA_ORDENES_1M_MS);
        }
    }

    /// Un pánico con el candado tomado no deja estado a medias (cada
    /// `observar` escribe muestras enteras), así que se sigue con el valor.
    fn uso_exchange_bloqueado(&self) -> std::sync::MutexGuard<'_, UsoExchange> {
        self.uso_exchange
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// F1.8: política centralizada ante rate limits de Binance.
    /// - 429: cooldown temporal (Retry-After o 60 s por defecto); al tercer 429
    ///   consecutivo, freno temporal RATE_BRAKE (cooldown + 60 s, M4-C02),
    ///   nunca el kill-switch.
    /// - 418: IP baneada → freno temporal RATE_BRAKE por la ventana del ban
    ///   (300 s por defecto), que caduca solo (M4-C02).
    ///
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
                // FIX M4-C02: freno TEMPORAL escalado, NO kill-switch permanente.
                // Un burst de 429 es transitorio; latchear para siempre dejaba
                // posiciones vivas sin cierre (liquidación) hasta reiniciar.
                // Ventana = cooldown actual + 60s de margen de seguridad.
                let brake_until = until.saturating_add(60_000);
                self.extend_rate_brake(brake_until);
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
            let arena_guard = self.arena.load_full();
            let now = current_synced_timestamp_ms(arena_guard.as_deref());
            drop(arena_guard);
            let brake_until = now.saturating_add(ban_s.saturating_mul(1000));
            self.extend_rate_brake(brake_until);
            println!(
                "🚨 [RATE BRAKE] HTTP 418: IP baneada — freno temporal {}s (auto-rearme hasta ms {}).",
                ban_s, brake_until
            );
        }
        e.to_string()
    }

    /// FIX M4-C02: extiende el freno temporal por rate-limit (máximo atómico —
    /// nunca acorta una ventana ya activa). Auto-expira: check_rate_limits lo
    /// libera cuando timestamp_ms >= rate_brake_until_ms.
    #[inline(always)]
    fn extend_rate_brake(&self, until_ms: u64) {
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

    /// B1.3: lector público del kill-switch para el watchdog de protección
    /// (no colocar brackets nuevos cuando el freno total está activo).
    #[inline(always)]
    pub fn is_kill_switch_active(&self) -> bool {
        self.kill_switch.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn check_rate_limits(&self, timestamp_ms: u64) -> Result<(), String> {
        if self.kill_switch.load(Ordering::Relaxed) {
            return Err("KILL SWITCH ACTIVE. Execution blocked.".to_string());
        }
        let _cupo = self.check_exit_rate_limits(timestamp_ms, Tramo::Lectura)?;

        // F1-030: Límite local de aceleración aplicado ÚNICAMENTE a la entrada
        // de órdenes nuevas (riesgo creciente), derivado de la capacidad de órdenes
        // del exchange (300 órdenes/10s ≈ 30 ops/s). Las cancelaciones, lecturas y
        // cierres reduce-only nunca son vetadas por este literal local.
        let current_sec = timestamp_ms / 1000;
        let last_reset = self.last_reset_timestamp.load(Ordering::Relaxed);

        if current_sec > last_reset {
            self.last_reset_timestamp
                .store(current_sec, Ordering::Relaxed);
            self.rate_limit_counter.store(1, Ordering::Relaxed);
        } else {
            let ops = self.rate_limit_counter.fetch_add(1, Ordering::Relaxed);
            if ops > 25 {
                return Err("LOCAL RATE LIMIT EXCEEDED. Throttling execution.".to_string());
            }
        }
        Ok(())
    }

    /// N5 (CL-178): los dos frenos que ya puso el exchange, sin cuota. Ninguna
    /// petición que lo consulte (directamente o por `check_exit_rate_limits`)
    /// sale mientras duran, salidas incluidas: Binance pide no enviar durante
    /// el Retry-After de un 429, y seguir enviando acumula 429 hasta el freno
    /// escalado (3 consecutivos) o el ban de IP (418). Las que aún no lo
    /// consultan (modo hedge, hora, NTP, listenKey, la cancelación de la
    /// pierna hermana del user-data stream y la primera lectura de
    /// `get_payload_account`) son la fila N20 del censo C-F7b.
    fn check_backoff(&self, timestamp_ms: u64) -> Result<(), String> {
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

    /// CL-3 — EL KILL-SWITCH BLOQUEA LO QUE AUMENTA EL RIESGO, NO LAS SALIDAS.
    ///
    /// F1-030: Presupuestos separados. Las cancelaciones, lecturas y cierres reduce-only
    /// NUNCA se bloquean por un literal local de ops/s. Se rigen por los frenos reales
    /// del exchange (RATE_BRAKE, 429 cooldown) y por la cuota vigente que reporta Binance
    /// (2400 de peso/min, 300 órdenes/10 s y 1200/min), según el tramo (N5, CL-179):
    /// las lecturas frenan al 80 % y las críticas al límite, contando las críticas en vuelo.
    /// Devuelve el cupo de la petición, que el llamador suelta tras la respuesta.
    fn check_exit_rate_limits(&self, timestamp_ms: u64, tramo: Tramo) -> Result<Cupo<'_>, String> {
        use crate::binance_api::{VENTANA_ORDENES_10S_MS, VENTANA_ORDENES_1M_MS, VENTANA_PESO_MS};
        self.check_backoff(timestamp_ms)?;

        let limite = [
            self.max_weight_1m.load(Ordering::Relaxed),
            self.max_orders_10s.load(Ordering::Relaxed),
            self.max_orders_1m.load(Ordering::Relaxed),
        ];
        const CONTADOR: [&str; 3] = ["WEIGHT", "ORDERS_10S", "ORDERS_1M"];

        // LG-3: el uso que reporta una cabecera vale sólo dentro de su ventana
        // del exchange (1 min / 10 s). Sin caducidad, una cabecera alta
        // bloqueaba toda petición nueva —también salidas y cancelaciones—
        // antes de enviarla, nunca llegaba la respuesta que la refrescara y el
        // bloqueo era permanente. N5 (CL-177): cada contador con su sello y sin
        // bajar dentro de su ventana por una respuesta tardía.
        let mut uso_exchange = self.uso_exchange_bloqueado();
        let uso = [
            uso_exchange.peso_1m.vigente(timestamp_ms, VENTANA_PESO_MS),
            uso_exchange.ordenes_10s.vigente(timestamp_ms, VENTANA_ORDENES_10S_MS),
            uso_exchange.ordenes_1m.vigente(timestamp_ms, VENTANA_ORDENES_1M_MS),
        ];

        match tramo {
            Tramo::Lectura => {
                drop(uso_exchange);
                // F1.8: HEADROOM 80% — los contadores de Binance llegan POR RESPUESTA;
                // frenar al 100% ya es tarde (requests en vuelo cruzan el límite).
                // N5 (CL-179): es el techo de entradas y lecturas, y la franja que deja
                // es la reserva del tramo crítico. El 0,8 es el valor heredado, no
                // derivado: la cota del tráfico crítico por ventana está en el censo (N5).
                const HEADROOM: f64 = 0.8;
                for k in 0..3 {
                    if uso[k] as f64 > limite[k] as f64 * HEADROOM {
                        return Err(format!(
                            "RATE_LIMIT_HEADROOM_{}: {}/{}",
                            CONTADOR[k], uso[k], limite[k]
                        ));
                    }
                }
                Ok(Cupo {
                    ex: self,
                    coste: crate::binance_api::CosteBinance::NULO,
                })
            }
            Tramo::Critica(coste) => {
                // La regla del exchange (rechaza si usado + coste > límite), con
                // las críticas admitidas y sin respuesta contadas como usadas.
                // CL-181: en un contador donde la petición no gasta (c = 0) sólo
                // la rechaza un uso ya por encima del límite; lo que está en
                // vuelo no le quita un hueco que no ocupa.
                let c = [coste.peso, coste.ordenes, coste.ordenes];
                for k in 0..3 {
                    let pendiente = if c[k] == 0 { 0 } else { uso_exchange.en_vuelo[k] };
                    let total = uso[k].saturating_add(pendiente).saturating_add(c[k]);
                    if total > limite[k] {
                        return Err(format!(
                            "RATE_LIMIT_EXHAUSTED_{}: {} usado + {} en vuelo + {} > {}",
                            CONTADOR[k], uso[k], pendiente, c[k], limite[k]
                        ));
                    }
                }
                for (pendiente, coste) in uso_exchange.en_vuelo.iter_mut().zip(c) {
                    *pendiente = pendiente.saturating_add(coste);
                }
                drop(uso_exchange);
                Ok(Cupo { ex: self, coste })
            }
        }
    }

    /// Redondea la cantidad a los decimales permitidos (step_size).
    #[inline(always)]
    fn round_to_step_size(quantity: f64, step_size: f64) -> f64 {
        // D-629: un paso inválido devuelve 0, que todos los llamadores tratan
        // como «volumen 0» y abortan. Antes devolvía la cantidad sin redondear
        // (paso ≤ 0) o NaN (paso NaN), y ambas llegaban al exchange.
        if !step_size.is_finite() || step_size <= 0.0 || !quantity.is_finite() {
            return 0.0;
        }
        let inv = exact_inverse(step_size);
        snap_floor(quantity * inv) / inv
    }

    // FIX #738: Redondeo direccional de precio: ceil para SELL, floor para BUY
    #[inline(always)]
    pub fn round_price_to_tick(price: f64, tick_size: f64, is_sell: bool) -> f64 {
        if !tick_size.is_finite() || tick_size <= 0.0 || !price.is_finite() {
            return 0.0;
        }
        let inv = exact_inverse(tick_size);
        let scaled = price * inv;
        if is_sell {
            snap_ceil(scaled) / inv
        } else {
            snap_floor(scaled) / inv
        }
    }

    /// D-630 (DÉCIMA OLA): redondeo de un STOP de protección.
    ///
    /// Un stop redondeado hacia la entrada queda más ajustado que la distancia
    /// mínima que calcula `risk-engine::tp_sl` —el piso anti-difusivo— y la
    /// posición sale por ruido. Se redondea ALEJÁNDOSE de la entrada: hacia
    /// abajo en un largo, hacia arriba en un corto.
    #[inline(always)]
    fn round_stop_away_from_entry(price: f64, tick_size: f64, position_is_long: bool) -> f64 {
        Self::round_price_to_tick(price, tick_size, !position_is_long)
    }

    /// Redondeo de un OBJETIVO, también alejándose de la entrada (hacia arriba
    /// en un largo, hacia abajo en un corto): así el ratio beneficio/riesgo
    /// aplicado nunca queda por debajo del que exige la fricción (D-636).
    #[inline(always)]
    fn round_target_away_from_entry(price: f64, tick_size: f64, position_is_long: bool) -> f64 {
        Self::round_price_to_tick(price, tick_size, position_is_long)
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
        if !order.volume_usd.is_finite()
            || order.volume_usd <= 0.0
            || !current_price.is_finite()
            || current_price <= 0.0
            || (order.maker_only && (!tick_size.is_finite() || tick_size <= 0.0))
            || !step_size.is_finite()
            || step_size <= 0.0
            || order
                .integer_leverage()
                .filter(|&l| l <= crate::binance_api::MAX_INITIAL_LEVERAGE)
                .is_none()
        {
            return None;
        }

        // Cantidad exacta calculada a partir del margen asignado y el apalancamiento aprobado
        let notional = order.volume_usd * order.leverage;
        if !notional.is_finite() || notional <= 0.0 {
            return None;
        }
        let raw_quantity = notional / current_price;
        if !raw_quantity.is_finite() || raw_quantity <= 0.0 {
            return None;
        }
        let final_quantity = Self::round_to_step_size(raw_quantity, step_size);

        if !final_quantity.is_finite() || final_quantity <= 0.0 {
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
            // D-628: unirse al mejor nivel propio (medio tick desde el mid), no
            // quedar un nivel por detrás.
            let final_price = passive_join_price(current_price, tick_size, is_sell);
            // The minimum-tick fallback must not move a BUY above its reference.
            // This checks quote geometry, not passivity against a live book.
            if !final_price.is_finite()
                || final_price <= 0.0
                || (!is_sell && final_price > current_price)
                || (is_sell && final_price < current_price)
            {
                return None;
            }
            let price_dec = crate::fixed_filter::try_count_decimals(tick_size);
            let price_str = price_dec.and_then(|dec| {
                let s = crate::fixed_filter::format_decimal(final_price, dec);
                if !s.is_empty() { Some(s) } else { None }
            });
            let price_param = if let Some(ref s) = price_str {
                format!("&timeInForce=GTX&price={}", s.as_str())
            } else {
                format!("&timeInForce=GTX&price={}", final_price)
            };
            (
                ORDER_TYPE_LIMIT,
                crate::binance_api::TIME_IN_FORCE_GTX,
                price_param,
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

        // Cuantización de cantidad exacta en stack ASCII sin residuos float (-1111)
        let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
        let qty_str = qty_dec.and_then(|dec| {
            let s = crate::fixed_filter::format_decimal(final_quantity, dec);
            if !s.is_empty() { Some(s) } else { None }
        });
        let qty_param = if let Some(ref s) = qty_str {
            s.as_str().to_string()
        } else {
            final_quantity.to_string()
        };

        // F1.4: query EXACTA que se firma = query EXACTA que se envía.
        let signed_query = format!(
            "symbol={}&side={}{}&type={}&quantity={}{}&newClientOrderId={}&timestamp={}",
            symbol,
            side,
            pos_side_param,
            order_type,
            qty_param,
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
                Some(passive_join_price(current_price, tick_size, is_sell))
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

    /// GET de clase CUENTA (positionRisk, account, income, commission):
    /// respuestas que escalan con el tamaño de la cuenta y NO pertenecen a
    /// la ruta de órdenes. El timeout HFT global (1.2s) las mataba a
    /// intermitencia (TEXT_ERR: operation timed out → body vacío → parse
    /// error). Presupuesto de 5s + 1 reintento. La ruta de órdenes sigue
    /// con el presupuesto global: ahí la latencia es parte del contrato.
    async fn get_payload_account(
        &self,
        url: &str,
    ) -> Result<(crate::client::BinanceRateLimits, String), String> {
        let mut res = self
            .client
            .get_payload_with_timeout(url, Some(std::time::Duration::from_secs(5)))
            .await;
        if let Err(ref e) = res {
            if !e.starts_with("HTTP_429_RATE_LIMITED") && !e.starts_with("HTTP_418_IP_BANNED") {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                // N5 (CL-178): otra petición pudo recibir un 429/418 durante
                // la espera; entonces el reintento no sale y vuelve el error
                // de la primera.
                if self.check_backoff(self.get_synced_timestamp()).is_ok() {
                    res = self
                        .client
                        .get_payload_with_timeout(url, Some(std::time::Duration::from_secs(5)))
                        .await;
                }
            }
        }
        res.map_err(|e| self.handle_rate_limit_error(&e))
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

        // exchangeInfo pesa MBs: el timeout HFT global (1.2s) lo mata
        // intermitentemente. Presupuesto dedicado + 1 reintento.
        // N20 (CL-184): ni la descarga ni su reintento salen dentro del
        // Retry-After de un 429 ni del freno (3×429 o 418). Sólo el freno,
        // sin el 80 %: los cierres necesitan los filtros.
        self.check_backoff(self.get_synced_timestamp())?;
        let mut res = self
            .client
            .get_payload_with_timeout(buf.as_str(), Some(std::time::Duration::from_secs(10)))
            .await;
        if let Err(ref e) = res {
            if !e.starts_with("HTTP_429_RATE_LIMITED") && !e.starts_with("HTTP_418_IP_BANNED") {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                // Como el reintento de get_payload_account (CL-178).
                if self.check_backoff(self.get_synced_timestamp()).is_ok() {
                    res = self
                        .client
                        .get_payload_with_timeout(buf.as_str(), Some(std::time::Duration::from_secs(10)))
                        .await;
                }
            }
        }
        match res {
            Ok((limits, text)) => {
                self.update_limits(&limits);
                Self::filtros_de_exchange_info(&text)
            }
            Err(e) => Err(self.handle_rate_limit_error(&e)),
        }
    }

    /// N20 (CL-185): los filtros de una respuesta de exchangeInfo. Una que no
    /// trae ninguno utilizable (cuerpo ilegible —el cliente devuelve Ok con
    /// «TEXT_ERR…» si un 2xx no se pudo leer—, JSON roto, sin `symbols` o sin
    /// un solo símbolo con paso y tick) es un error: antes era un mapa vacío
    /// que get_symbol_filter guardaba encima de la caché buena, y cada símbolo
    /// volvía a descargar exchangeInfo.
    fn filtros_de_exchange_info(
        texto: &str,
    ) -> Result<std::collections::HashMap<String, SymbolFilter>, String> {
        let mut map = std::collections::HashMap::new();
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(texto) {
            if let Some(symbols) = json["symbols"].as_array() {
                for sym_info in symbols {
                    if let Some(sym) = sym_info["symbol"].as_str() {
                        let mut filter = SymbolFilter::default();
                        // D-631 (DÉCIMA OLA): un paso o tick que no parsea ya
                        // no se sustituye por los de BTCUSDT; el símbolo sólo
                        // entra en el mapa si ambos se leyeron de verdad.
                        let mut step_ok = false;
                        let mut tick_ok = false;
                        if let Some(filters) = sym_info["filters"].as_array() {
                            for f in filters {
                                match f["filterType"].as_str() {
                                    Some("LOT_SIZE") => {
                                        if let Some(v) = f["stepSize"]
                                            .as_str()
                                            .and_then(|s| s.parse::<f64>().ok())
                                            .filter(|v| v.is_finite() && *v > 0.0)
                                        {
                                            filter.step_size = v;
                                            step_ok = true;
                                        }
                                    }
                                    Some("PRICE_FILTER") => {
                                        if let Some(v) = f["tickSize"]
                                            .as_str()
                                            .and_then(|t| t.parse::<f64>().ok())
                                            .filter(|v| v.is_finite() && *v > 0.0)
                                        {
                                            filter.tick_size = v;
                                            tick_ok = true;
                                        }
                                    }
                                    Some("MIN_NOTIONAL") => {
                                        if let Some(n) = f["notional"].as_str() {
                                            if let Ok(v) = n.parse::<f64>() {
                                                if v.is_finite() && v > 0.0 {
                                                    filter.min_notional = v;
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        if step_ok && tick_ok {
                            // F1-105: Si min_notional no vino o no parseó de exchangeInfo,
                            // consultar el registro de símbolos local si tiene spec válido
                            if filter.min_notional == 5.0 {
                                if let Some(cid) = quantum_arena::symbol_registry::try_index(sym) {
                                    if let Some(spec) = quantum_arena::symbol_registry::try_spec(cid) {
                                        if spec.min_notional.is_finite() && spec.min_notional > 0.0 {
                                            filter.min_notional = spec.min_notional;
                                        }
                                    }
                                }
                            }
                            map.insert(sym.to_string(), filter);
                        }
                    }
                }
            }
        }
        if map.is_empty() {
            return Err(format!(
                "EXCHANGE_INFO_SIN_FILTROS: {}",
                crate::order_types::truncate(texto, 200)
            ));
        }
        Ok(map)
    }

    /// Filtros reales de precisión de un símbolo.
    ///
    /// D-631 (DÉCIMA OLA): el último recurso era `SymbolFilter::default()` —paso
    /// 0,001 y tick 0,1, los de BTCUSDT— devuelto en silencio. Para un símbolo con
    /// tick 0,00001 eso produce precios cuatro órdenes de magnitud fuera de
    /// escala: el fallo aparecía como un rechazo del exchange en lugar de como el
    /// error de configuración que realmente era, y un cierre de emergencia podía
    /// enviarse con la precisión de otro activo. Ahora un símbolo sin filtro real
    /// es un ERROR que el llamador debe manejar.
    pub async fn get_symbol_filter(&self, symbol: &str) -> Result<SymbolFilter, String> {
        // 1. Caché positiva ArcSwap O(1) < 5ns
        {
            let cache = self.symbol_filters.load();
            if let Some(f) = cache.get(symbol) {
                return Ok(*f);
            }
        }

        // F1-105: 2. Caché negativa con TTL (60s) para evitar que símbolos ausentes o deslistados
        // bloqueen el hilo ~20s re-descargando y re-parseando megabytes de exchangeInfo en cada intento.
        let now = std::time::Instant::now();
        if let Ok(neg_cache) = self.negative_symbol_cache.read() {
            if let Some(ts) = neg_cache.get(symbol) {
                if now.duration_since(*ts).as_secs() < 60 {
                    // Fallback rápido al registro de símbolos local si fue añadido dinámicamente
                    if let Some(coin_id) = quantum_arena::symbol_registry::try_index(symbol) {
                        if let Some(spec) = quantum_arena::symbol_registry::try_spec(coin_id) {
                            if spec.step_size.is_finite()
                                && spec.step_size > 0.0
                                && spec.tick_size.is_finite()
                                && spec.tick_size > 0.0
                            {
                                return Ok(SymbolFilter {
                                    step_size: spec.step_size,
                                    tick_size: spec.tick_size,
                                    min_notional: spec.min_notional,
                                });
                            }
                        }
                    }
                    return Err(format!(
                        "filtro de precisión desconocido para {symbol} (en caché negativa TTL 60s): ni exchangeInfo ni el registro de símbolos lo aportan; no se opera con los pasos de otro símbolo"
                    ));
                }
            }
        }

        // 3. Descarga de exchangeInfo y actualización de caché positiva
        if let Ok(filters) = self.fetch_all_symbol_filters().await {
            let res = filters.get(symbol).copied();
            self.symbol_filters.store(Arc::new(filters));
            if let Some(f) = res {
                return Ok(f);
            }
        }

        // 4. Fallback al registro local de símbolos
        if let Some(coin_id) = quantum_arena::symbol_registry::try_index(symbol) {
            if let Some(spec) = quantum_arena::symbol_registry::try_spec(coin_id) {
                if spec.step_size.is_finite()
                    && spec.step_size > 0.0
                    && spec.tick_size.is_finite()
                    && spec.tick_size > 0.0
                {
                    return Ok(SymbolFilter {
                        step_size: spec.step_size,
                        tick_size: spec.tick_size,
                        min_notional: spec.min_notional,
                    });
                }
            }
        }

        // 5. Símbolo definitivamente ausente: almacenar en caché negativa con TTL 60s
        if let Ok(mut neg_cache) = self.negative_symbol_cache.write() {
            neg_cache.insert(symbol.to_string(), now);
        }

        Err(format!(
            "filtro de precisión desconocido para {symbol}: ni exchangeInfo ni el \
             registro de símbolos lo aportan; no se opera con los pasos de otro símbolo"
        ))
    }

    /// F1-105: Limpia la caché negativa de filtros de símbolos.
    pub fn clear_negative_symbol_cache(&self) {
        if let Ok(mut neg_cache) = self.negative_symbol_cache.write() {
            neg_cache.clear();
        }
    }

    /// F1-105: Retorna el tamaño actual de la caché negativa de símbolos.
    pub fn negative_symbol_cache_len(&self) -> usize {
        self.negative_symbol_cache.read().map(|c| c.len()).unwrap_or(0)
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
    /// OCO-F5: cancela una orden ALGO (condicional: TP/SL/trailing). Las piernas
    /// del bracket ya NO viven en /fapi/v1/order: cancelarlas por la ruta legacy
    /// devuelve "order does not exist" y deja el trigger vivo en el exchange.
    ///
    /// D-698 (DÉCIMA OLA · auditoría integral): SE CANCELA POR `algoId`.
    /// `fetch_open_algo_orders` ya parsea el identificador numérico que el
    /// exchange asigna (`OpenAlgoOrder::algo_id`) y ningún sitio lo leía: todas
    /// las cancelaciones iban por `clientAlgoId`, el identificador que fija el
    /// cliente. Cuando se conoce el `algoId` (el caso de las purgas, que parten
    /// de la lista abierta del exchange) se usa ése; el `clientAlgoId` queda
    /// como respaldo para las rutas que sólo tienen el identificador propio
    /// —el rollback de un bracket a medio colocar, donde el algoId aún no se
    /// ha leído—.
    #[inline(always)]
    pub async fn cancel_algo_order(&self, symbol: &str, client_algo_id: &str) -> Result<(), String> {
        self.cancel_algo_order_ids(symbol, 0, client_algo_id).await
    }

    /// D-698: cancela una pierna algo por `algoId` (preferente) o por
    /// `clientAlgoId` cuando el numérico no se conoce (`algo_id == 0`).
    pub async fn cancel_algo_order_ids(
        &self,
        symbol: &str,
        algo_id: u64,
        client_algo_id: &str,
    ) -> Result<(), String> {
        if algo_id == 0 && client_algo_id.is_empty() {
            return Err(format!(
                "{}: pierna algo sin algoId ni clientAlgoId — no se puede cancelar",
                symbol
            ));
        }
        if self.is_paper_trading {
            println!(
                "📝 [PAPER TRADING LOCAL] Algo Orden Cancelada: {} en {}",
                if algo_id > 0 { algo_id.to_string() } else { client_algo_id.to_string() },
                symbol
            );
            return Ok(());
        }

        let timestamp = self.get_synced_timestamp();
        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_CANCELACION))?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/algoOrder?"
        } else {
            "https://fapi.binance.com/fapi/v1/algoOrder?"
        });
        let payload_start = buf.as_str().len();
        buf.push_str("symbol=");
        buf.push_str(symbol);
        if algo_id > 0 {
            buf.push_str("&algoId=");
            buf.push_u64(algo_id);
        } else {
            buf.push_str("&clientAlgoId=");
            buf.push_str(client_algo_id);
        }
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
        match res {
            Ok(limits) => {
                self.update_limits(&limits);
                Ok(())
            }
            Err(e) => Err(self.handle_rate_limit_error(&e)),
        }
    }

    /// OCO-F5: purga TODAS las órdenes algo abiertas de un símbolo.
    /// DELETE /fapi/v1/algoOpenOrders — el /fapi/v1/allOpenOrders legacy NO
    /// las toca: un bracket TP/SL sobreviviría a la purga y dispararía un
    /// close fantasma sobre una posición ya aplanada.
    async fn cancel_all_algo_open_orders(&self, symbol: &str) -> Result<(), String> {
        if self.is_paper_trading {
            return Ok(());
        }
        let timestamp = self.get_synced_timestamp();
        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_CANCELACION))?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/algoOpenOrders?"
        } else {
            "https://fapi.binance.com/fapi/v1/algoOpenOrders?"
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
        match res {
            Ok(limits) => {
                self.update_limits(&limits);
                Ok(())
            }
            Err(e) => Err(self.handle_rate_limit_error(&e)),
        }
    }

    /// OCO-F5: lista las órdenes algo (TP/SL/trailing) abiertas de un símbolo.
    /// GET /fapi/v1/openAlgoOrders — los brackets NO aparecen en
    /// /fapi/v1/openOrders; sin esto, el motor cree que la posición está
    /// desnuda y re-bracketea infinito.
    /// B2.6: TODAS las algo orders abiertas de la cuenta (sin filtro de
    /// símbolo) — para purgar piernas huérfanas de posiciones ya cerradas.
    pub async fn fetch_all_open_algo_orders(
        &self,
    ) -> Result<Vec<crate::order_types::OpenAlgoOrder>, String> {
        self.fetch_open_algo_orders_inner(None).await
    }

    pub async fn fetch_open_algo_orders(
        &self,
        symbol: &str,
    ) -> Result<Vec<crate::order_types::OpenAlgoOrder>, String> {
        self.fetch_open_algo_orders_inner(Some(symbol)).await
    }

    async fn fetch_open_algo_orders_inner(
        &self,
        symbol: Option<&str>,
    ) -> Result<Vec<crate::order_types::OpenAlgoOrder>, String> {
        if self.is_paper_trading {
            return Ok(Vec::new());
        }
        let timestamp = self.get_synced_timestamp();
        // N5 (CL-179): con símbolo sostiene la protección de una posición
        // (crítica); sin él es el listado de huérfanas (lectura).
        let tramo = match symbol {
            Some(_) => Tramo::Critica(crate::binance_api::COSTE_ALGO_ABIERTAS_SIMBOLO),
            None => Tramo::Lectura,
        };
        let _cupo = self.check_exit_rate_limits(timestamp, tramo)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/openAlgoOrders?"
        } else {
            "https://fapi.binance.com/fapi/v1/openAlgoOrders?"
        });
        let payload_start = buf.as_str().len();
        if let Some(s) = symbol {
            buf.push_str("symbol=");
            buf.push_str(s);
            buf.push_str("&");
        }
        buf.push_str("timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.load();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.get_payload_account(buf.as_str()).await;
        match res {
            Ok((limits, body)) => {
                self.update_limits(&limits);
                serde_json::from_str(&body).map_err(|e| {
                    format!(
                        "OPEN_ALGO_PARSE: {} body={}",
                        e,
                        crate::order_types::truncate(&body, 200)
                    )
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Átomo 2.6 (Ola Ω217): consulta órdenes abiertas regulares (GET /fapi/v1/openOrders).
    /// Permite al reconciliador OMS detectar órdenes huérfanas en Binance tras reinicio.
    pub async fn fetch_all_open_orders(
        &self,
    ) -> Result<Vec<crate::order_types::OrderAck>, String> {
        self.fetch_open_orders_inner(None).await
    }

    pub async fn fetch_open_orders(
        &self,
        symbol: &str,
    ) -> Result<Vec<crate::order_types::OrderAck>, String> {
        self.fetch_open_orders_inner(Some(symbol)).await
    }

    async fn fetch_open_orders_inner(
        &self,
        symbol: Option<&str>,
    ) -> Result<Vec<crate::order_types::OrderAck>, String> {
        if self.is_paper_trading {
            return Ok(Vec::new());
        }
        let timestamp = self.get_synced_timestamp();
        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Lectura)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/openOrders?"
        } else {
            "https://fapi.binance.com/fapi/v1/openOrders?"
        });
        let payload_start = buf.as_str().len();
        if let Some(s) = symbol {
            buf.push_str("symbol=");
            buf.push_str(s);
            buf.push_str("&");
        }
        buf.push_str("timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.load();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.get_payload_account(buf.as_str()).await;
        match res {
            Ok((limits, body)) => {
                self.update_limits(&limits);
                serde_json::from_str(&body).map_err(|e| {
                    format!(
                        "OPEN_ORDERS_PARSE: {} body={}",
                        e,
                        crate::order_types::truncate(&body, 200)
                    )
                })
            }
            Err(e) => Err(e),
        }
    }

    /// B1.3: pierna ALGO individual (TP o SL) con el esquema validado OCO-F5.
    /// Para top-ups quirúrgicos del watchdog de protección: colocar SOLO el
    /// lado con shortfall — un OCO completo sobre-protegería el lado sano y
    /// el exchange lo rechaza (qty combinada > posición ⇒ -2022).
    pub async fn place_algo_leg(
        &self,
        symbol: &str,
        is_long_close: bool,
        order_type: &str, // "STOP_MARKET" | "TAKE_PROFIT_MARKET"
        quantity: f64,
        trigger_price: f64,
        step_size: f64,
        tick_size: f64,
        base_client_id: &str,
    ) -> Result<(), String> {
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if final_quantity == 0.0 {
            return Err("Volumen 0".to_string());
        }
        // SL se aleja de la entrada redondeando hacia el ruido; TP hacia el
        // objetivo (mismo criterio anti-difusivo D-630/D-636 del OCO).
        let final_trigger = if order_type == "STOP_MARKET" {
            Self::round_stop_away_from_entry(trigger_price, tick_size, is_long_close)
        } else {
            Self::round_target_away_from_entry(trigger_price, tick_size, is_long_close)
        };

        if self.is_paper_trading {
            println!(
                "📝 [PAPER TRADING LOCAL] Pierna algo {} para {} @ {} — local.",
                order_type, symbol, final_trigger
            );
            return Ok(());
        }

        let side = if is_long_close { "SELL" } else { "BUY" };
        let timestamp = self.get_synced_timestamp();
        // CL-20: pierna PROTECTORA, inherentemente reductora (positionSide en
        // hedge, reduceOnly en one-way): el kill-switch no la bloquea (mismo
        // principio que CL-3). Si el aplanado falla, la posición conserva
        // su protección en vez de quedar desnuda. La cuota sigue aplicando.
        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_PIERNA_ALGO))?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/algoOrder?"
        } else {
            "https://fapi.binance.com/fapi/v1/algoOrder?"
        });
        let payload_start = buf.as_str().len();
        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        let is_hedge = self.is_hedge_mode.load(Ordering::Relaxed);
        if is_hedge {
            buf.push_str("&positionSide=");
            buf.push_str(if is_long_close { "LONG" } else { "SHORT" });
        } else {
            buf.push_str("&reduceOnly=true");
        }
        let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
        let tick_dec = crate::fixed_filter::try_count_decimals(tick_size);
        buf.push_str("&algoType=CONDITIONAL");
        buf.push_str("&type=");
        buf.push_str(order_type);
        buf.push_str("&quantity=");
        buf.push_fixed_f64(final_quantity, qty_dec);
        buf.push_str("&triggerPrice=");
        buf.push_fixed_f64(final_trigger, tick_dec);
        buf.push_str("&clientAlgoId=");
        buf.push_str(base_client_id);
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
        match res {
            Ok(limits) => {
                self.update_limits(&limits);
                Ok(())
            }
            Err(e) => Err(self.handle_rate_limit_error(&e)),
        }
    }

    /// F1-031: Ejecución MARKET que retorna FillReport tipado con precios y comisiones reales.
    pub async fn execute_raw_qty_with_report(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        step_size: f64,
        client_order_id_param: &str,
    ) -> Result<FillReport, String> {
        let timestamp = self.get_synced_timestamp();
        self.check_rate_limits(timestamp)?;
        self.execute_raw_qty_internal(
            symbol,
            is_long,
            quantity,
            step_size,
            client_order_id_param,
            timestamp,
        )
        .await
    }

    async fn execute_raw_qty_internal(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        step_size: f64,
        client_order_id_param: &str,
        timestamp: u64,
    ) -> Result<FillReport, String> {
        if quantity.is_infinite() || quantity.is_nan() || quantity <= 0.0 {
            return Err(
                "SEGURIDAD: quantity inválido (infinito, NaN o <= 0.0). Orden abortada."
                    .to_string(),
            );
        }
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if !final_quantity.is_finite() || final_quantity <= 0.0 {
            return Err("Volumen 0 despues de round_to_step_size".to_string());
        }

        if self.is_paper_trading {
            println!(
                "📝 [PAPER TRADING LOCAL] Ejecutando raw_qty {} para {}. Cero latencia simulada.",
                final_quantity, symbol
            );
            return Ok(FillReport {
                client_order_id: client_order_id_param.to_string(),
                symbol: symbol.to_string(),
                executed_qty: final_quantity,
                avg_price: 0.0,
                commission: 0.0,
                is_filled: true,
                status: "FILLED".to_string(),
            });
        }

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };

        // F1.2: toda orden market lleva newClientOrderId (idempotencia y trazabilidad).
        let client_order_id = if client_order_id_param.is_empty() {
            uuid::Uuid::now_v7().simple().to_string()
        } else {
            client_order_id_param.to_string()
        };
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
        let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
        buf.push_fixed_f64(final_quantity, qty_dec);
        buf.push_str("&newClientOrderId=");
        buf.push_str(&client_order_id);
        // F1-031: Exigir RESULT para conocer avgPrice, executedQty y comisiones reales
        buf.push_str("&newOrderRespType=RESULT");
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        if buf.is_overflow() {
            return Err("SEGURIDAD: query de orden excede el buffer (orden abortada)".to_string());
        }

        // F1.5: registrar la intención antes del envío. CL-47: después de la
        // última salida previa al envío, como la IOC (CL-39b): un buffer
        // desbordado no deja una intención huérfana.
        self.order_registry.register_intent(
            &client_order_id,
            symbol,
            side,
            if is_long { "LONG" } else { "SHORT" },
            ORDER_TYPE_MARKET,
            final_quantity,
            timestamp,
        );

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
                // FIX M4-C01: confirmar con el exchange antes de retornar Ok —
                // el write WS no es un ack (ver confirm_ws_dispatch).
                self.confirm_ws_dispatch(symbol, &client_order_id).await?;
                return self
                    .order_registry
                    .get_fill_report(&client_order_id)
                    .ok_or_else(|| "FILL_REPORT_NOT_FOUND_AFTER_WS_CONFIRM".to_string());
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
                Ok(FillReport::from_ack(ack))
            }
            Err(e) => {
                // F4-H4: rechazo firme cierra la intención en el OrderRegistry
                if crate::ioc_evidence::error_cierra_la_intencion(e) {
                    self.order_registry.mark_local_reject(&client_order_id, timestamp);
                }
                if e.starts_with("HTTP_429") || e.starts_with("HTTP_418") {
                    return Err(self.handle_rate_limit_error(e));
                }
                Err(e.clone())
            }
        }
    }

    /// F1-031: Reduce-Only Market que retorna FillReport para retroalimentación de slippage y comisiones.
    pub async fn execute_reduce_only_market_with_report(
        &self,
        symbol: &str,
        is_long_close: bool,
        quantity: f64,
        step_size: f64,
    ) -> Result<FillReport, String> {
        let timestamp = self.get_synced_timestamp();
        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_ORDEN))?;
        self.execute_reduce_only_market_internal(
            symbol,
            is_long_close,
            quantity,
            step_size,
            timestamp,
        )
        .await
    }

    async fn execute_reduce_only_market_internal(
        &self,
        symbol: &str,
        is_long_close: bool,
        quantity: f64,
        step_size: f64,
        timestamp: u64,
    ) -> Result<FillReport, String> {
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if final_quantity <= 0.0 {
            return Err("Volumen 0 despues de round_to_step_size".to_string());
        }

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Reduciendo posicion de {} para {}. Cero latencia simulada.", final_quantity, symbol);
            return Ok(FillReport {
                client_order_id: String::new(),
                symbol: symbol.to_string(),
                executed_qty: final_quantity,
                avg_price: 0.0,
                commission: 0.0,
                is_filled: true,
                status: "FILLED".to_string(),
            });
        }

        let side = if is_long_close { SIDE_SELL } else { SIDE_BUY };

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/order?"
        } else {
            "https://fapi.binance.com/fapi/v1/order?"
        });
        let payload_start = buf.as_str().len();

        let is_hedge = self.is_hedge_mode.load(Ordering::Relaxed);
        let pos_side_str = if is_hedge {
            if is_long_close { "LONG" } else { "SHORT" }
        } else {
            "BOTH"
        };

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        if is_hedge {
            buf.push_str("&positionSide=");
            buf.push_str(pos_side_str);
            buf.push_str("&type=");
            buf.push_str(ORDER_TYPE_MARKET);
        } else {
            buf.push_str("&type=");
            buf.push_str(ORDER_TYPE_MARKET);
            buf.push_str("&reduceOnly=true");
        }
        buf.push_str("&quantity=");
        let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
        buf.push_fixed_f64(final_quantity, qty_dec);

        let client_order_id = uuid::Uuid::now_v7().simple().to_string();

        buf.push_str("&newClientOrderId=");
        buf.push_str(&client_order_id);
        // F1-028 / F1-031: Exigir RESULT para conocer executedQty y estado terminal
        buf.push_str("&newOrderRespType=RESULT");
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        if buf.is_overflow() {
            return Err("SEGURIDAD: query de orden excede el buffer (orden abortada)".to_string());
        }

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.load();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        // F1-028: Registrar intención antes del despacho
        self.order_registry.register_intent(
            &client_order_id,
            symbol,
            side,
            pos_side_str,
            ORDER_TYPE_MARKET,
            final_quantity,
            timestamp,
        );

        let res = self.client.execute_order_payload_typed(buf.as_str()).await;
        match res {
            Ok((limits, ack)) => {
                self.update_limits(&limits);
                let now = self.get_synced_timestamp();
                self.order_registry.apply_ack(&ack, now);
                let report = FillReport::from_ack(&ack);
                if ack.executed_qty.is_finite() && ack.executed_qty > 0.0 {
                    Ok(report)
                } else if ack.is_filled()
                    || ack.status.trim().eq_ignore_ascii_case("PARTIALLY_FILLED")
                {
                    Ok(report)
                } else {
                    Err(format!(
                        "CLOSE_UNFILLED: status={}, executed_qty={}",
                        ack.status, ack.executed_qty
                    ))
                }
            }
            Err(e) => {
                if crate::ioc_evidence::error_cierra_la_intencion(&e) {
                    self.order_registry.mark_local_reject(&client_order_id, timestamp);
                }
                Err(self.handle_rate_limit_error(&e))
            }
        }
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
        if !current_price.is_finite()
            || current_price <= 0.0
            || !order.volume_usd.is_finite()
            || order.volume_usd <= 0.0
            || !step_size.is_finite()
            || step_size <= 0.0
            || !(order.volume_usd * order.leverage).is_finite()
            || order.signal == SignalType::Flat
        {
            return Err("SEGURIDAD: precio, margen o paso inválido. Orden abortada.".to_string());
        }
        // FMT-214/218: validate before paper success, leverage mutation or IO.
        // Casting fractional L changes collateral semantics without changing N.
        let target_leverage = order
            .integer_leverage()
            .filter(|&l| l <= crate::binance_api::MAX_INITIAL_LEVERAGE)
            .ok_or_else(|| {
                "SEGURIDAD: apalancamiento no entero o fuera del dominio de Binance.".to_string()
            })?;

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] Ejecutando orden de {:?} para {}. No se envió a Testnet API para evitar divergencia de latencia.", order.signal, symbol);
            return Ok(());
        }

        // Parametrizar dinámicamente el Leverage (Kelly Criterion)
        let needs_update = {
            let cache = self.active_leverage.load();
            cache.get(symbol).copied().unwrap_or(0) != target_leverage
        };

        if needs_update {
            if let Err(e) = self.set_leverage(symbol, target_leverage).await {
                // D-732 (DÉCIMA OLA · auditoría integral): un fallo al fijar el
                // apalancamiento sólo se imprimía y la orden salía IGUAL, con el
                // apalancamiento que la cuenta tuviera de antes —el de otra
                // operación, u otro símbolo—: el nocional enviado se calculó con
                // el apalancamiento que el risk-engine decidió, así que el margen
                // exigido y el riesgo real son otros. Una orden cuya premisa de
                // apalancamiento no se cumplió no debe enviarse.
                return Err(format!(
                    "no se pudo fijar apalancamiento {}x en {}: {} — orden abortada (el nocional se dimensionó con ese apalancamiento)",
                    target_leverage, symbol, e
                ));
            } else {
                let mut cache = (**self.active_leverage.load()).clone();
                cache.insert(symbol.to_string(), target_leverage);
                self.active_leverage.store(Arc::new(cache));
            }
        }

        let timestamp = self.get_synced_timestamp();
        self.check_rate_limits(timestamp)?;

        // F1.4: tickSize real del símbolo consultado en cache O(1) (< 5 ns)
        let filter = self.get_symbol_filter(symbol).await?;
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
                    // FIX M4-C01: el Ok(()) del write WS sólo significaba
                    // "bytes al buffer del kernel" — SIN ack del exchange.
                    // Confirmar la resolución real antes de prometer éxito
                    // (ORDER_TRADE_UPDATE vía registry; respaldo REST acotado;
                    // NUNCA re-envío — riesgo de duplicación).
                    return self
                        .confirm_ws_dispatch(symbol, &payload.client_order_id)
                        .await;
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
                // CL-47: la intención ya estaba registrada (la ruta WS la
                // necesita) y esta orden nunca sale: se cierra.
                self.order_registry
                    .mark_local_reject(&payload.client_order_id, payload.timestamp);
                return Err(
                    "SEGURIDAD: query de orden excede el buffer (orden abortada)".to_string(),
                );
            }

            let res = self.client.execute_order_payload_typed(buf.as_str()).await;
            match &res {
                Ok((limits, ack)) => {
                    self.update_limits(limits);
                    self.order_registry.apply_ack(ack, payload.timestamp);
                    if !ack.is_filled() && !ack.is_active() {
                        let status_tag = if ack.status.is_empty() {
                            "UNEXPECTED_TERMINAL"
                        } else {
                            &ack.status
                        };
                        println!(
                            "⚠️ [EXECUTION] Orden {} estado final no ejecutable: {} (executed={})",
                            payload.client_order_id, status_tag, ack.executed_qty
                        );
                        return Err(format!(
                            "ORDER_{}: status={}, executed_qty={}",
                            status_tag, status_tag, ack.executed_qty
                        ));
                    }
                    Ok(())
                }
                Err(e) => {
                    // F4-H4: rechazo firme cierra la intención en el OrderRegistry
                    if crate::ioc_evidence::error_cierra_la_intencion(e) {
                        self.order_registry.mark_local_reject(&payload.client_order_id, payload.timestamp);
                    }
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
        let timestamp = self.get_synced_timestamp();
        self.check_rate_limits(timestamp)?;
        self.execute_raw_qty_internal(
            symbol,
            is_long,
            quantity,
            step_size,
            client_order_id_param,
            timestamp,
        )
        .await
        .map(|_| ())
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
        let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
        buf.push_fixed_f64(final_quantity, qty_dec);
        buf.push_str("&price=");
        let tick_dec = crate::fixed_filter::try_count_decimals(tick_size);
        buf.push_fixed_f64(final_price, tick_dec);
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
                // F4-H4: rechazo firme cierra la intención en el OrderRegistry
                if crate::ioc_evidence::error_cierra_la_intencion(e) {
                    self.order_registry.mark_local_reject(client_order_id, timestamp);
                }
                if e.starts_with("HTTP_429") || e.starts_with("HTTP_418") {
                    return Err(self.handle_rate_limit_error(e));
                }
                Err(e.clone())
            }
        }
    }

    /// GTX, sondeo acotado, cancelación y consulta terminal antes de completar
    /// el remanente como taker. Un resultado incierto no permite otra entrada.
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
        if let Err(error) = maker_res {
            // The legacy LIMIT transport does not prove non-execution on all
            // error paths. No unconditional full-size MARKET replacement.
            // Even a known rejection needs a typed fallback policy, not Err.
            return Err(format!("MAKER_CHASE_UNVERIFIED: initial maker submission: {error}"));
        }

        // 2. Presupuesto fijo heredado: 100 sondeos de 4ms más scheduling.
        // No es adaptación al espectro temporal ni acredita que 400ms sean
        // inocuos para cualquier intención. La salida temprana es sólo una
        // observación local: todavía se consulta el terminal remoto abajo.
        for _ in 0..100 {
            tokio::time::sleep(std::time::Duration::from_millis(4)).await;
            if let Some(order) = self.order_registry.get(client_order_id) {
                if !order.status.is_active() {
                    break;
                }
            }
        }

        // 3. Solicitar cancelación. Las etiquetas de error legacy sólo ayudan
        // al diagnóstico: ninguna prueba por sí sola ausencia o cancelación.
        // Incluso un DELETE exitoso exige identidad, cantidades válidas y
        // estado terminal en la consulta posterior antes de reemplazar.
        let mut cancel_net_failed = false;
        match self.cancel_order(symbol, client_order_id).await {
            Ok(()) => {}
            Err(e) => {
                let benign = e.contains("-2011")
                    || e.contains("-4002")
                    || e.contains("Unknown order")
                    || e.contains("CANCELLED")
                    || e.contains("CANCELED");
                if benign {
                    println!(
                        "🛡️ [MAKER-CHASE] cancel de {} devolvió etiqueta legacy ({}): no acredita terminal; se exige consulta antes del remanente.",
                        client_order_id, e
                    );
                } else {
                    cancel_net_failed = true;
                    println!(
                        "⚠️ [MAKER-CHASE] cancel de {} falló por RED ({}): la GTX puede seguir VIVA — verificando estado real ANTES de mercadoar el remnant.",
                        client_order_id, e
                    );
                }
            }
        }

        // F1.3 — FIX DOUBLE-FILL: consultar el estado REAL de la orden tras el
        // cancel y mercadear ÚNICAMENTE el remanente (origQty - executedQty).
        // Antes: se mercadeaba la cantidad completa aunque el límite se hubiera
        // llenado parcial o totalmente → posición duplicada (bug crítico de auditoría).
        let executed = match self.query_order(symbol, client_order_id).await {
            Ok(ack) => crate::execution_evidence::terminal_maker_executed_quantity(
                symbol, client_order_id, &ack,
            )?,
            Err(e) => {
                // Sin estado verificable NO se mercadea nada: a ciegas es el bug
                // original. El remanente se materializa vía reconciliación (F1.7).
                println!("🛑 [MAKER-CHASE] No se pudo verificar estado de {} ({}). Abortando chase SIN market de respaldo para evitar doble-fill.", client_order_id, e);
                if cancel_net_failed {
                    // C-03: el cancel TAMBIÉN falló por red — doble incertidumbre.
                    // El prefijo MAKER_CHASE_UNVERIFIED es el que el host trata en
                    // la rama X-007 (AMBIGUA, god_engine.rs); el token
                    // CANCEL_UNVERIFIED distingue este caso en el forense.
                    return Err(format!(
                        "MAKER_CHASE_UNVERIFIED (CANCEL_UNVERIFIED): {}",
                        e
                    ));
                }
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

        // 4. Ejecutar como Taker SOLO el remanente (Market Order). B3.10
        // (auditoría): ID firmado "mcT_" — el remanente taker del maker-chase
        // antes usaba un ID autogenerado y quedaba fuera del diario de fills,
        // sesgando la medición de selección adversa (sólo veíamos los "mc_").
        let mut id_buf = [0u8; 32];
        id_buf[0..4].copy_from_slice(b"mcT_");
        let micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        let mut itoa_buf = itoa::Buffer::new();
        let micros_str = itoa_buf.format(micros);
        id_buf[4..4 + micros_str.len()].copy_from_slice(micros_str.as_bytes());
        let remnant_id = std::str::from_utf8(&id_buf[..4 + micros_str.len()]).unwrap_or("mcT_0");
        // CL-46b: el host no resuelve esta orden hija consultando la padre.
        self.execute_raw_qty_with_client_id(symbol, is_long, remaining, step_size, remnant_id)
            .await
            .map_err(|e| crate::ioc_evidence::error_del_remanente(executed, remnant_id, e))
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
        if !quantity.is_finite() || quantity <= 0.0 {
            return Err(
                "SEGURIDAD: quantity inválido (infinito, NaN o <= 0.0). Orden abortada."
                    .to_string(),
            );
        }
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
        let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
        buf.push_fixed_f64(final_quantity, qty_dec);
        buf.push_str("&price=");
        let tick_dec = crate::fixed_filter::try_count_decimals(tick_size);
        buf.push_fixed_f64(final_price, tick_dec);
        buf.push_str("&newClientOrderId=");
        buf.push_str(client_order_id);
        // CL-39: con el ACK por defecto Binance responde `NEW`; con RESULT,
        // el estado TERMINAL de la IOC y su executedQty.
        buf.push_str("&newOrderRespType=RESULT");
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        if buf.is_overflow() {
            return Err("SEGURIDAD: query de orden excede el buffer (orden abortada)".to_string());
        }

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        let api_secret = self.api_secret.load();
        sign_payload_to_buffer(payload, &api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        // CL-39: la intención queda referenciada por clientOrderId justo antes
        // del envío, como en MARKET (F1.5): si la respuesta se pierde, la
        // reconciliación la resuelve por consulta.
        self.order_registry.register_intent(
            client_order_id,
            symbol,
            side,
            if is_long { "LONG" } else { "SHORT" },
            ORDER_TYPE_LIMIT,
            final_quantity,
            timestamp,
        );

        // CL-39: una IOC aceptada NO es un llenado. Antes se devolvía Ok(())
        // ante cualquier 2xx sin leer el cuerpo: una IOC expirada sin
        // ejecución (HTTP 200, EXPIRED, executedQty=0) se confirmaba y dejaba
        // una posición fantasma en el arena; un 5xx o -1007 salía como
        // rechazo y el host revertía una orden que pudo ejecutarse. Ahora
        // decide la evidencia de la propia orden (ver `ioc_evidence`):
        // ejecución > 0 ⇒ Ok; terminal sin ejecución ⇒ IOC_UNFILLED (el host
        // revierte); sin evidencia terminal ⇒ AMBIGUOUS (el host conserva la
        // reserva y reconcilia).
        match self.client.execute_order_payload_body(buf.as_str()).await {
            Ok((limits, body)) => {
                self.update_limits(&limits);
                let resultado =
                    crate::ioc_evidence::clasificar_respuesta_ioc(&body, symbol, client_order_id)?;
                self.order_registry.apply_ack(resultado.ack(), timestamp);
                // Igual que `record_query_resolution`: una instantánea
                // terminal sin ejecución no borra fills ya recibidos por WS.
                if let Some(merged) = self.order_registry.get(client_order_id) {
                    if merged.executed_qty.is_finite() && merged.executed_qty > 0.0 {
                        return Ok(());
                    }
                }
                crate::ioc_evidence::resultado_para_el_host(&resultado, client_order_id)
            }
            Err(e) => {
                // CL-39b: un rechazo firme (la orden nunca existió) cierra la
                // intención; un error ambiguo la deja viva para la consulta.
                if crate::ioc_evidence::error_cierra_la_intencion(&e) {
                    self.order_registry.mark_local_reject(client_order_id, timestamp);
                }
                if e.starts_with("HTTP_429") || e.starts_with("HTTP_418") {
                    return Err(self.handle_rate_limit_error(&e));
                }
                Err(e)
            }
        }
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
        // FMT-220: preserve the API, not an undocumented USD-M capability.
        // This gate precedes paper success, signing, and account/network effects.
        if !crate::binance_api::SUPPORTS_NATIVE_ICEBERG {
            return Err("UNSUPPORTED_NATIVE_ICEBERG: USD-M /fapi/v1/order has no documented icebergQty support".into());
        }
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
        let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
        buf.push_fixed_f64(final_quantity, qty_dec);
        buf.push_str("&icebergQty=");
        buf.push_fixed_f64(final_iceberg_qty, qty_dec);
        buf.push_str("&price=");
        let tick_dec = crate::fixed_filter::try_count_decimals(tick_size);
        buf.push_fixed_f64(final_price, tick_dec);
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
        match res {
            Ok(limits) => {
                self.update_limits(&limits);
                Ok(())
            }
            Err(e) => Err(self.handle_rate_limit_error(&e)),
        }
    }

    /// FASE 13: Reduce-Only Market. Guaranteed to only reduce position, never flip.
    /// F1-028: Ejecución tipada con newOrderRespType=RESULT, register_intent y validación
    /// estricta de executed_qty para prevenir cierres incompletos o ambiguos.
    #[inline(always)]
    async fn execute_reduce_only_market(
        &self,
        symbol: &str,
        is_long_close: bool,
        quantity: f64,
        step_size: f64,
    ) -> Result<(), String> {
        let timestamp = self.get_synced_timestamp();
        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_ORDEN))?;
        self.execute_reduce_only_market_internal(
            symbol,
            is_long_close,
            quantity,
            step_size,
            timestamp,
        )
        .await
        .map(|_| ())
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
        // CL-20: pierna PROTECTORA, inherentemente reductora (positionSide en
        // hedge, reduceOnly en one-way): el kill-switch no la bloquea (mismo
        // principio que CL-3). Si el aplanado falla, la posición conserva
        // su protección en vez de quedar desnuda. La cuota sigue aplicando.
        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_PIERNA_ALGO))?;

        let mut buf = ZeroAllocBuffer::new();
        // OCO-F5: TRAILING_STOP_MARKET es tipo condicional — bloqueado en
        // /fapi/v1/order (-4120) desde la migración al servicio de Algo
        // (2025-12-09). activationPrice/callbackRate mantienen sus nombres.
        buf.push_str(if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/algoOrder?"
        } else {
            "https://fapi.binance.com/fapi/v1/algoOrder?"
        });
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(&side);
        // R3.1 (consistencia con place_algo_leg/execute_oco_order): en HEDGE,
        // positionSide hace la orden inherentemente reductora; en ONE-WAY,
        // positionSide está prohibido (-4061) y la protección correcta es
        // reduceOnly=true — sin él, un trailing sobre qty desincronizada
        // podría ABRIR/invertir posición en cuentas one-way.
        if self.is_hedge_mode.load(Ordering::Relaxed) {
            buf.push_str("&positionSide=");
            buf.push_str(if is_long { "LONG" } else { "SHORT" });
        } else {
            buf.push_str("&reduceOnly=true");
        }
        buf.push_str("&algoType=CONDITIONAL");
        buf.push_str("&type=TRAILING_STOP_MARKET");
        buf.push_str("&quantity=");
        let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
        buf.push_fixed_f64(final_quantity, qty_dec);
        buf.push_str("&activationPrice=");
        let tick_dec = crate::fixed_filter::try_count_decimals(tick_size);
        buf.push_fixed_f64(final_price, tick_dec);
        buf.push_str("&callbackRate=");
        buf.push_f64(safe_callback);
        buf.push_str("&clientAlgoId=");
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
        match res {
            Ok(limits) => {
                self.update_limits(&limits);
                Ok(())
            }
            Err(e) => Err(self.handle_rate_limit_error(&e)),
        }
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

        // D-630 (DÉCIMA OLA): ambas piernas usaban la misma dirección de
        // redondeo. Eso dejaba el TP alejándose de la entrada (correcto) y el SL
        // ACERCÁNDOSE a ella (incorrecto: más ajustado que el piso
        // anti-difusivo). La ruta de reintento reutiliza estos dos valores.
        let final_tp =
            Self::round_target_away_from_entry(take_profit_price, tick_size, is_long_close);
        let final_sl = Self::round_stop_away_from_entry(stop_loss_price, tick_size, is_long_close);

        if self.is_paper_trading {
            println!("📝 [PAPER TRADING LOCAL] OCO Limit/Stop interceptada para {} @ TP: {} / SL: {}. Se maneja localmente.", symbol, final_tp, final_sl);
            return Ok(());
        }

        let side = if is_long_close { SIDE_SELL } else { SIDE_BUY };
        let timestamp = self.get_synced_timestamp();
        // CL-20: pierna PROTECTORA, inherentemente reductora (positionSide en
        // hedge, reduceOnly en one-way): el kill-switch no la bloquea (mismo
        // principio que CL-3). Si el aplanado falla, la posición conserva
        // su protección en vez de quedar desnuda. La cuota sigue aplicando.
        let cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_PIERNA_ALGO.por(2)))?;

        let base_url = if self.client.is_testnet.load(Ordering::Relaxed) {
            "https://testnet.binancefuture.com/fapi/v1/algoOrder?"
        } else {
            "https://fapi.binance.com/fapi/v1/algoOrder?"
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
        //
        // OCO-F5 (migración Algo Orders, efectiva 2025-12-09): Binance
        // bloqueó los tipos condicionales en /fapi/v1/order con -4120
        // ("Order type not supported for this endpoint"). STOP_MARKET,
        // TAKE_PROFIT_MARKET, STOP, TAKE_PROFIT y TRAILING_STOP_MARKET
        // viven ahora en el servicio de Algo: POST /fapi/v1/algoOrder con
        // algoType=CONDITIONAL, triggerPrice (antes stopPrice) y
        // clientAlgoId (antes newClientOrderId). Esquema validado
        // empíricamente contra testnet — ver src/bin/oco_probe.rs.
        let mut sl_buf = ZeroAllocBuffer::new();
        sl_buf.push_str(base_url);
        let sl_payload_start = sl_buf.as_str().len();
        sl_buf.push_str("symbol=");
        sl_buf.push_str(symbol);
        sl_buf.push_str("&side=");
        sl_buf.push_str(side);
        sl_buf.push_str(&position_side_q);
        sl_buf.push_str(&reduce_only_q);
        sl_buf.push_str("&algoType=CONDITIONAL");
        sl_buf.push_str("&type=STOP_MARKET");
        let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
        let tick_dec = crate::fixed_filter::try_count_decimals(tick_size);
        sl_buf.push_str("&quantity=");
        sl_buf.push_fixed_f64(final_quantity, qty_dec);
        sl_buf.push_str("&triggerPrice=");
        sl_buf.push_fixed_f64(final_sl, tick_dec);
        sl_buf.push_str("&clientAlgoId=");
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
        tp_buf.push_str("&algoType=CONDITIONAL");
        tp_buf.push_str("&type=TAKE_PROFIT_MARKET");
        tp_buf.push_str("&quantity=");
        tp_buf.push_fixed_f64(final_quantity, qty_dec);
        tp_buf.push_str("&triggerPrice=");
        tp_buf.push_fixed_f64(final_tp, tick_dec);
        tp_buf.push_str("&clientAlgoId=");
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

        match &sl_res {
            Ok(limits) => {
                self.update_limits(limits);
                confirmed_sl_id = Some(format!("{}_SL", base_client_id));
            }
            Err(e) => {
                let _ = self.handle_rate_limit_error(e);
            }
        }
        match &tp_res {
            Ok(limits) => {
                self.update_limits(limits);
                confirmed_tp_id = Some(format!("{}_TP", base_client_id));
            }
            Err(e) => {
                let _ = self.handle_rate_limit_error(e);
            }
        }

        // N5 (CL-179): las dos piernas ya respondieron y salen del libro en
        // vuelo antes de los reintentos y cancelaciones, que piden su propio
        // cupo. Una con 2xx deja su uso en la cabecera; una con error contó
        // en el exchange y el cliente tira su cabecera (client.rs), y un
        // timeout AMBIGUOUS pudo contar sin respuesta: hasta la próxima
        // cabecera 2xx su coste no se cuenta (censo C-F7b, fila N21).
        drop(cupo);

        // F1.9 — FIX OCO PARCIAL: antes, una pierna fallida devolvía Ok(())
        // dejando la posición protegida por UN solo lado (naked al otro).
        // Política correcta: reintentar la pierna caída UNA vez; si sigue
        // caída, cancelar la pierna buena y devolver Err — el caller decide
        // aplanar o alertar. Nunca protección parcial silenciosa.
        if sl_res.is_err() || tp_res.is_err() {
            let sl_down = sl_res.is_err();
            let tp_down = tp_res.is_err();
            // OCO-F1: antes solo se imprimían los booleanos — el cuerpo real
            // del rechazo de Binance (código -2022/-1111/-4061...) se perdía
            // y el diagnóstico forense era imposible.
            println!(
                "⚠️ [OCO] Pierna(s) fallida(s) (SL={}, TP={}). Reintentando... SL_err={:?} TP_err={:?}",
                sl_down,
                tp_down,
                sl_res.as_ref().err(),
                tp_res.as_ref().err()
            );
            // D-02 — RETRY CON REGENERACIÓN COMPLETA (5º informe): el retry
            // anterior reenviaba el MISMO buffer firmado — timestamp vencido
            // (-1021). Ahora: nuevo timestamp NTP y nueva firma HMAC.
            //
            // LG-6 (Línea G, G040): el reintento conserva el MISMO clientAlgoId.
            // Un error AMBIGUOUS (timeout de red) no prueba que la pierna no
            // exista: con un id nuevo (_SLR/_TPR) el reintento podía dejar dos
            // SL o dos TP de cantidad completa. Con el mismo id, si la primera
            // llegó al exchange el reintento devuelve «id duplicado» (-4116) y
            // eso CONFIRMA la pierna (`pierna_ya_existe`).
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
                let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
                let tick_dec = crate::fixed_filter::try_count_decimals(tick_size);
                rb.push_str("&algoType=CONDITIONAL");
                rb.push_str("&type=STOP_MARKET");
                rb.push_str("&quantity=");
                rb.push_fixed_f64(final_quantity, qty_dec);
                rb.push_str("&triggerPrice=");
                rb.push_fixed_f64(final_sl, tick_dec);
                rb.push_str("&clientAlgoId=");
                rb.push_str(&format!("{}_SL", base_client_id));
                rb.push_str("&timestamp=");
                rb.push_u64(ts_retry);
                let mut rsb = [0u8; 64];
                sign_payload_to_buffer(&rb.as_str()[rp..], &retry_secret, &mut rsb);
                rb.push_str("&signature=");
                rb.push_str(unsafe { std::str::from_utf8_unchecked(&rsb) });
                // N5 (CL-178): si la primera pierna recibió un 429/418, el
                // reintento no sale (seguía enviando dentro del cooldown).
                // CL-179: pide su cupo crítico, soltado tras la respuesta.
                // CL-182: la cabecera se registra con el cupo vivo.
                sl_res = match self.check_exit_rate_limits(
                    self.get_synced_timestamp(),
                    Tramo::Critica(crate::binance_api::COSTE_PIERNA_ALGO),
                ) {
                    Ok(_cupo) => self
                        .client
                        .execute_order_payload(rb.as_str())
                        .await
                        .inspect(|limits| self.update_limits(limits)),
                    Err(e) => Err(e),
                };
                if let Err(e) = &sl_res {
                    if pierna_ya_existe(e) {
                        // Sin cabecera; sólo el reinicio F1-005 de los 429, como antes.
                        let limits = crate::client::BinanceRateLimits::default();
                        self.update_limits(&limits);
                        sl_res = Ok(limits);
                    } else {
                        let _ = self.handle_rate_limit_error(e);
                    }
                }
                if sl_res.is_ok() {
                    confirmed_sl_id = Some(format!("{}_SL", base_client_id));
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
                let qty_dec = crate::fixed_filter::try_count_decimals(step_size);
                let tick_dec = crate::fixed_filter::try_count_decimals(tick_size);
                rb.push_str("&algoType=CONDITIONAL");
                rb.push_str("&type=TAKE_PROFIT_MARKET");
                rb.push_str("&quantity=");
                rb.push_fixed_f64(final_quantity, qty_dec);
                rb.push_str("&triggerPrice=");
                rb.push_fixed_f64(final_tp, tick_dec);
                rb.push_str("&clientAlgoId=");
                rb.push_str(&format!("{}_TP", base_client_id));
                rb.push_str("&timestamp=");
                rb.push_u64(ts_retry);
                let mut rsb = [0u8; 64];
                sign_payload_to_buffer(&rb.as_str()[rp..], &retry_secret, &mut rsb);
                rb.push_str("&signature=");
                rb.push_str(unsafe { std::str::from_utf8_unchecked(&rsb) });
                // N5 (CL-178): si la primera pierna recibió un 429/418, el
                // reintento no sale (seguía enviando dentro del cooldown).
                // CL-179: pide su cupo crítico, soltado tras la respuesta.
                // CL-182: la cabecera se registra con el cupo vivo.
                tp_res = match self.check_exit_rate_limits(
                    self.get_synced_timestamp(),
                    Tramo::Critica(crate::binance_api::COSTE_PIERNA_ALGO),
                ) {
                    Ok(_cupo) => self
                        .client
                        .execute_order_payload(rb.as_str())
                        .await
                        .inspect(|limits| self.update_limits(limits)),
                    Err(e) => Err(e),
                };
                if let Err(e) = &tp_res {
                    if pierna_ya_existe(e) {
                        // Sin cabecera; sólo el reinicio F1-005 de los 429, como antes.
                        let limits = crate::client::BinanceRateLimits::default();
                        self.update_limits(&limits);
                        tp_res = Ok(limits);
                    } else {
                        let _ = self.handle_rate_limit_error(e);
                    }
                }
                if tp_res.is_ok() {
                    confirmed_tp_id = Some(format!("{}_TP", base_client_id));
                }
            }
        }

        if sl_res.is_err() && tp_res.is_err() {
            // OCO-F1: propagar el rechazo EXACTO de cada pierna al caller
            // (el retry del motor y el emergency-close necesitan el código
            // para decidir; el log necesita el cuerpo para el forense).
            let sl_e = sl_res.as_ref().err().cloned().unwrap_or_default();
            let tp_e = tp_res.as_ref().err().cloned().unwrap_or_default();
            return Err(format!(
                "Ambas órdenes OCO fallaron. SL_err: {} | TP_err: {}",
                sl_e, tp_e
            ));
        }
        if sl_res.is_err() || tp_res.is_err() {
            // D-416: Cancelar el client_order_id exacto que fue confirmado en Binance.
            // LG-6 (G040): un fallo de cancelación ya no se informa como
            // cancelado: queda en `pendientes` y marca la protección como sucia
            // para que el vigilante la audite enseguida.
            let mut cancelled_ids = Vec::new();
            let mut pendientes = Vec::new();
            let confirmadas = [
                (sl_res.is_ok(), confirmed_sl_id.as_ref()),
                (tp_res.is_ok(), confirmed_tp_id.as_ref()),
            ];
            for (ok, id) in confirmadas {
                if let (true, Some(id)) = (ok, id) {
                    match self.cancel_algo_order(symbol, id).await {
                        Ok(()) => cancelled_ids.push(id.clone()),
                        Err(e) => pendientes.push(format!("{id}: {e}")),
                    }
                }
            }
            if !pendientes.is_empty() {
                quantum_arena::protection_health::mark_dirty();
            }
            let failed_e = if sl_res.is_err() {
                sl_res.as_ref().err().cloned().unwrap_or_default()
            } else {
                tp_res.as_ref().err().cloned().unwrap_or_default()
            };
            return Err(format!(
                "OCO PARCIAL: pierna fallida tras retry ({}); pierna(s) confirmada(s) [{}] cancelada(s); cancelaciones pendientes [{}]. Posición SIN protección — aplanar o alertar.",
                failed_e,
                cancelled_ids.join(", "),
                pendientes.join(" | ")
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

        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_CANCELACION))?;

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
        match res {
            Ok(limits) => {
                self.update_limits(&limits);
                Ok(())
            }
            Err(e) => Err(self.handle_rate_limit_error(&e)),
        }
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
        let cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_CANCELACION))?;

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
        match res {
            Ok(limits) => {
                self.update_limits(&limits);
            }
            Err(e) => {
                return Err(self.handle_rate_limit_error(&e));
            }
        }
        // N5 (CL-179): el DELETE ya respondió; la purga algo pide su cupo.
        drop(cupo);
        // OCO-F5: las órdenes TP/SL condicionales viven en el servicio de
        // Algo y SOBREVIVEN a /fapi/v1/allOpenOrders. Sin esta segunda purga,
        // todo cierre de posición dejaba triggers vivos que disparaban
        // closes fantasmas sobre posiciones ya aplanadas.
        self.cancel_all_algo_open_orders(symbol).await
    }

    /// D-371: Cancela quirúrgicamente únicamente las órdenes OCO asociadas a la posición especificada
    ///
    /// OCO-F5: los brackets son órdenes ALGO que (a) NO aparecen en
    /// /fapi/v1/openOrders ni en el order_registry alimentado por
    /// ORDER_TRADE_UPDATE, y (b) NO se cancelan por DELETE /fapi/v1/order.
    /// La fuente de verdad es /fapi/v1/openAlgoOrders y el cancel es
    /// DELETE /fapi/v1/algoOrder por algoId, filtrando por positionSide.
    #[inline(always)]
    async fn cancel_position_oco_orders(
        &self,
        symbol: &str,
        is_long: bool,
    ) -> Result<usize, String> {
        let pos_side = if is_long { "LONG" } else { "SHORT" };
        let open_algo = self.fetch_open_algo_orders(symbol).await?;
        let hedge = self.is_hedge_mode.load(Ordering::SeqCst);
        let mut canceled = 0;
        let mut fallidas: Vec<String> = Vec::new();
        for a in open_algo {
            // D-698: `positionSide` vacío no es «coincide». En modo one-way el
            // exchange no lo informa y sólo existe una posición por símbolo, así
            // que la pierna es nuestra; en modo hedge un vacío es DESCONOCIDO y
            // cancelarlo podía dejar desnuda la posición del lado contrario.
            let matches_side = a.position_side == pos_side
                || a.position_side == "BOTH"
                || (a.position_side.is_empty() && !hedge);
            if a.position_side.is_empty() && hedge {
                fallidas.push(format!(
                    "{} sin positionSide en modo hedge (algoId {}) — no se cancela a ciegas",
                    symbol, a.algo_id
                ));
                continue;
            }
            if matches_side {
                // D-698: el fallo YA NO se traga. Antes `is_ok()` descartaba el
                // error y la función devolvía `Ok(0)`: los tres llamadores
                // (rollback del bracket, purga tras cierre y purga de huérfanas)
                // informaban «brackets purgados» con las piernas vivas, y una
                // pierna vieja disparaba sobre la posición SIGUIENTE.
                match self
                    .cancel_algo_order_ids(symbol, a.algo_id, &a.client_algo_id)
                    .await
                {
                    Ok(()) => canceled += 1,
                    Err(e) => fallidas.push(format!(
                        "{} algoId {} clientAlgoId {}: {}",
                        symbol, a.algo_id, a.client_algo_id, e
                    )),
                }
            }
        }
        if !fallidas.is_empty() {
            return Err(format!(
                "cancelaciones de piernas fallidas ({} ok, {} pendientes): {}",
                canceled,
                fallidas.len(),
                fallidas.join(" · ")
            ));
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

        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Critica(crate::binance_api::COSTE_CONSULTA_ORDEN))?;

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
                crate::execution_evidence::parse_query_order_response(&body, symbol, client_order_id)
            }
            Err(e) => Err(self.handle_rate_limit_error(&e)),
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
            Err(e) => Err(self.handle_rate_limit_error(&e)),
        }
    }

    async fn fetch_open_positions(&self) -> Result<Vec<ActivePosition>, String> {
        let timestamp = self.get_synced_timestamp();

        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Lectura)?;

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

        let res = self.get_payload_account(buf.as_str()).await;
        match res {
            Ok((limits, text)) => {
                self.update_limits(&limits);
                crate::execution_evidence::parse_active_positions(&text)
            }
            Err(e) => Err(e),
        }
    }

    async fn fetch_account_balance(&self) -> Result<f64, String> {
        let timestamp = self.get_synced_timestamp();

        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Lectura)?;

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

        match self.get_payload_account(buf.as_str()).await {
            Ok((limits, text)) => {
                self.update_limits(&limits);

                // CL-8: patrimonio (`totalWalletBalance`), no `availableBalance`
                // (ya neto del margen que la adopción vuelve a reservar).
                crate::execution_evidence::parse_account_equity(&text).map_err(|e| {
                    format!("{e}. Response /fapi/v2/account: {text}")
                })
            }
            Err(e) => Err(e),
        }
    }

    async fn set_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String> {
        if symbol.is_empty() || !(1..=crate::binance_api::MAX_INITIAL_LEVERAGE).contains(&leverage) {
            return Err("SEGURIDAD: símbolo vacío o leverage fuera del dominio USD-M".into());
        }
        if self.is_paper_trading {
            return Ok(());
        }
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
            Ok((limits, text)) => {
                self.update_limits(&limits);
                crate::binance_api::validate_leverage_confirmation(&text, symbol, leverage)
            }
            Err(e) => Err(self.handle_rate_limit_error(&e)),
        }
    }

    async fn fetch_commission_rate(&self, symbol: &str) -> Result<(f64, f64), String> {
        let timestamp = self.get_synced_timestamp();

        let _cupo = self.check_exit_rate_limits(timestamp, Tramo::Lectura)?;

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

        match self.get_payload_account(buf.as_str()).await {
            Ok((limits, text)) => {
                self.update_limits(&limits);
                // A parseable float can still be NaN/Inf. Only a complete,
                // finite observation for this symbol may reach live fee state.
                crate::execution_evidence::parse_commission_rates(&text, symbol)
            }
            Err(e) => Err(e),
        }
    }

    async fn fetch_exchange_info(&self, symbol: &str) -> Result<f64, String> {
        let filter = self.get_symbol_filter(symbol).await?;
        Ok(filter.min_notional)
    }
}

/// LG-6 (Línea G, G040) — ¿EL ERROR DEL REINTENTO PRUEBA QUE LA PIERNA EXISTE?
///
/// El reintento de una pierna OCO reenvía el MISMO `clientAlgoId`. Si el
/// primer envío llegó al exchange (aunque la respuesta se perdiera), el
/// exchange rechaza el duplicado con -4116 («ClientOrderId is duplicated»):
/// eso confirma la pierna. Cualquier otro error, incluido un AMBIGUOUS, no la
/// confirma.
#[inline]
fn pierna_ya_existe(err: &str) -> bool {
    let codigo = crate::order_types::error_codes::DUPLICATED_ORDER;
    let compacto = format!("\"code\":{codigo}");
    let espaciado = format!("\"code\": {codigo}");
    let rechazo = format!("code={codigo}");
    let termina_codigo = |marca: &str| {
        err.match_indices(marca).any(|(i, _)| {
            !err[i + marca.len()..].starts_with(|c: char| c.is_ascii_digit())
        })
    };
    termina_codigo(&compacto) || termina_codigo(&espaciado) || termina_codigo(&rechazo)
}

/// N5 (CL-178): símbolos de positionRisk sin ninguna fila abierta, cada uno
/// UNA vez y en el orden en que aparecen. En hedge la respuesta trae una fila
/// LONG y otra SHORT por símbolo: antes el barrido de huérfanas purgaba dos
/// veces cada símbolo plano (dos DELETE y dos cancelaciones algo de más).
fn simbolos_sin_posicion(
    entries: &[crate::reconciliation::PositionRiskEntry],
) -> Vec<String> {
    let abiertos: std::collections::HashSet<&str> = entries
        .iter()
        .filter(|p| p.is_open())
        .map(|p| p.symbol.as_str())
        .collect();
    let mut vistos = std::collections::HashSet::new();
    entries
        .iter()
        .map(|p| p.symbol.as_str())
        .filter(|s| !abiertos.contains(s) && vistos.insert(*s))
        .map(str::to_owned)
        .collect()
}

/// LG-3 (Línea G) — USO DE RATE-LIMIT VIGENTE.
///
/// Las cabeceras `X-MBX-USED-WEIGHT-1M` y `X-MBX-ORDER-COUNT-10S/1M` miden el
/// uso dentro de una ventana fija del exchange. Pasada esa ventana desde la
/// cabecera, el contador del exchange se ha reiniciado al menos una vez: el
/// uso conocido es 0. Un sello ausente (0) o un reloj que retrocede conservan
/// el valor (conservador).
#[inline]
fn uso_vigente(uso: usize, sello_ms: u64, ahora_ms: u64, ventana_ms: u64) -> usize {
    if sello_ms == 0 || ahora_ms < sello_ms {
        return uso;
    }
    if ahora_ms - sello_ms >= ventana_ms {
        0
    } else {
        uso
    }
}

/// N5 (CL-177): una cabecera de uso y el instante (reloj sincronizado, ms) en
/// que llegó. Sello 0 = nunca llegó.
#[derive(Clone, Copy, Debug, Default)]
struct MuestraUso {
    uso: usize,
    sello_ms: u64,
}

impl MuestraUso {
    fn vigente(self, ahora_ms: u64, ventana_ms: u64) -> usize {
        uso_vigente(self.uso, self.sello_ms, ahora_ms, ventana_ms)
    }
}

/// N5 (CL-177): uso de UN contador del exchange.
///
/// Dentro de una ventana el contador del exchange sólo sube, así que una
/// respuesta que llega con menos uso que otra aún vigente puede ser más vieja
/// (las peticiones concurrentes responden en cualquier orden) y no lo rebaja:
/// el uso es el máximo de las muestras vigentes. CL-180: se guardan todas las
/// que aún pueden serlo, con sellos no decrecientes y usos estrictamente
/// decrecientes de delante hacia atrás. Una muestra nueva tapa a las de atrás
/// con uso menor o igual (llegó después y caduca después), y la primera no
/// caducada es la mayor vigente. Antes se guardaban sólo la mayor y la última,
/// y al caducar la mayor se perdía una intermedia más alta que la última.
///
/// Tras un reinicio del contador del exchange la mayor sobreestima hasta su
/// caducidad. Es la elección conservadora, no una imposibilidad: una muestra
/// menor que llega más tarde que la vida máxima de una petición tras la mayor
/// es posterior y prueba el reinicio. No se usa porque esa vida depende de los
/// timeouts del cliente (client.rs) y de un sello de reloj de pared, y la
/// alineación de las ventanas del exchange no está comprobada (censo C-F7b,
/// fila N22).
#[derive(Clone, Debug, Default)]
struct ContadorExchange {
    muestras: std::collections::VecDeque<MuestraUso>,
}

impl ContadorExchange {
    /// CL-180: red de seguridad de la cola. Los usos guardados son enteros
    /// estrictamente decrecientes y una cabecera 2xx no pasa del límite L de
    /// su contador (el exchange rechaza si usado + coste > L), así que caben a
    /// lo sumo L + 1 por contador (2401, 301 y 1201). La cota es la suma de los
    /// límites por minuto ×2, por encima de cualquiera de ellas: sólo se
    /// alcanza si una cabecera superase su límite.
    const MAX_MUESTRAS: usize = 2
        * (crate::binance_api::LIMITE_PESO_1M + crate::binance_api::LIMITE_ORDENES_1M);

    fn observar(&mut self, uso: usize, ahora_ms: u64, ventana_ms: u64) {
        // CL-180: una muestra llegó a lo sumo ahora. Con un reloj que
        // retrocede, un sello futuro alargaría su vida más allá de su ventana
        // y ninguna respuesta la corregiría; se acota a ahora (conserva el
        // valor, lo conservador). Los sellos futuros son un sufijo.
        for m in self.muestras.iter_mut().rev() {
            if m.sello_ms <= ahora_ms {
                break;
            }
            m.sello_ms = ahora_ms;
        }
        // Las caducadas no aportan nada y son un prefijo (sellos no
        // decrecientes). Una de uso 0 sólo puede ser la última; la saca el
        // pop_back de abajo y vigente() la salta mientras tanto.
        while self
            .muestras
            .front()
            .is_some_and(|m| m.vigente(ahora_ms, ventana_ms) == 0)
        {
            self.muestras.pop_front();
        }
        while self.muestras.back().is_some_and(|m| m.uso <= uso) {
            self.muestras.pop_back();
        }
        if self.muestras.len() >= Self::MAX_MUESTRAS {
            // La de atrás tiene más uso: toma el sello nuevo (sobrestima, lo
            // conservador) en lugar de crecer.
            if let Some(m) = self.muestras.back_mut() {
                m.sello_ms = ahora_ms;
            }
            return;
        }
        self.muestras.push_back(MuestraUso { uso, sello_ms: ahora_ms });
    }

    fn vigente(&self, ahora_ms: u64, ventana_ms: u64) -> usize {
        self.muestras
            .iter()
            .map(|m| m.vigente(ahora_ms, ventana_ms))
            .find(|&u| u > 0)
            .unwrap_or(0)
    }
}

/// N5 (CL-177): los tres contadores de `exchangeInfo.rateLimits`.
#[derive(Debug, Default)]
struct UsoExchange {
    peso_1m: ContadorExchange,
    ordenes_10s: ContadorExchange,
    ordenes_1m: ContadorExchange,
    /// N5 (CL-179): coste de las peticiones críticas admitidas cuya respuesta
    /// aún no llegó (peso, órdenes 10 s, órdenes 1 min). Su uso no está en
    /// ninguna cabecera todavía.
    en_vuelo: [usize; 3],
}

/// N5 (CL-179): tramo de cuota de una petición.
///
/// - `Lectura`: entradas y lecturas. Frenan cuando el uso supera 0,8·L, la
///   regla de siempre (F1.8). El 0,8 es la reserva para el tramo crítico.
/// - `Critica(c)`: salidas, protección y lo que las sostiene (positionRisk,
///   cancelaciones, piernas, cierres reduce-only, consulta de una orden
///   ambigua). Pasan mientras uso + en vuelo + c ≤ L en cada contador en el
///   que gastan (c > 0), y uso ≤ L en los demás: la regla del exchange, sin
///   literal. Antes frenaban al 80 % como una
///   lectura, y en la franja (0,8·L, L] el host no podía cerrar ni proteger
///   aunque el exchange lo admitía.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tramo {
    Lectura,
    Critica(crate::binance_api::CosteBinance),
}

/// N5 (CL-179): plaza de una petición crítica en el libro de las que están en
/// vuelo. Al soltarla (respuesta recibida o error) su coste sale del libro;
/// entre medias lo cuentan las demás admisiones críticas, así que N críticas
/// concurrentes no caben juntas en el hueco de una. La cabecera de su
/// respuesta se registra antes de soltarla (CL-182). De una respuesta de
/// error el cliente tira la cabecera (client.rs): su coste sale del libro sin
/// entrar en el uso (censo C-F7b, fila N21). Las de `Lectura` llevan coste nulo: sólo se admiten por
/// debajo de 0,8·L.
#[must_use = "soltar el cupo antes de la respuesta saca la petición del libro de las que están en vuelo"]
struct Cupo<'a> {
    ex: &'a OrderExecutor,
    coste: crate::binance_api::CosteBinance,
}

impl std::fmt::Debug for Cupo<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cupo").field("coste", &self.coste).finish()
    }
}

impl Drop for Cupo<'_> {
    fn drop(&mut self) {
        if self.coste == crate::binance_api::CosteBinance::NULO {
            return;
        }
        let c = [self.coste.peso, self.coste.ordenes, self.coste.ordenes];
        let mut uso = self.ex.uso_exchange_bloqueado();
        for (pendiente, coste) in uso.en_vuelo.iter_mut().zip(c) {
            *pendiente = pendiente.saturating_sub(coste);
        }
    }
}

// Historical merge fixture only: retains old tests, NOT an operational identity
// contract. New traversal validates IDs and includes asset/trade_id; see
// income_evidence. These legacy tests do not prove pagination completeness.
#[cfg(test)]
type IncomeKey = (u64, u64, u64, String, String);

#[inline]
#[cfg(test)]
fn income_key(e: &crate::order_types::IncomeEntry) -> IncomeKey {
    (
        e.tran_id,
        e.time,
        e.income.to_bits(),
        e.symbol.clone(),
        e.income_type.clone(),
    )
}

/// Fusiona una página de income en el acumulador deduplicando por clave.
/// Devuelve cuántas entradas NUEVAS agregó (0 = puro solapamiento de cursor).
#[cfg(test)]
fn merge_income_page(
    acc: &mut Vec<crate::order_types::IncomeEntry>,
    seen: &mut std::collections::HashSet<IncomeKey>,
    page: Vec<crate::order_types::IncomeEntry>,
) -> usize {
    let mut added = 0;
    for e in page {
        if seen.insert(income_key(&e)) {
            acc.push(e);
            added += 1;
        }
    }
    added
}

#[cfg(test)]
mod tests_lg12 {
    use super::*;

    #[test]
    fn lg12_los_limites_de_peticiones_son_los_del_exchange() {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        assert_eq!(ex.max_weight_1m.load(Ordering::Relaxed), crate::binance_api::LIMITE_PESO_1M);
        assert_eq!(ex.max_orders_10s.load(Ordering::Relaxed), crate::binance_api::LIMITE_ORDENES_10S);
        assert_eq!(ex.max_orders_1m.load(Ordering::Relaxed), crate::binance_api::LIMITE_ORDENES_1M);
    }

    /// El host ya no reescribe los límites con genes.
    #[test]
    fn lg12_el_host_no_toma_los_limites_del_genoma() {
        let host = include_str!("../../../src/bin/god_engine.rs");
        let gen_peso = ["executor_max_", "weight_1m"].concat();
        let gen_ordenes = ["executor_max_", "orders_10s"].concat();
        assert!(!host.contains(&gen_peso) && !host.contains(&gen_ordenes));
    }
}

#[cfg(test)]
mod tests_lg6 {
    use super::*;

    #[test]
    fn lg6_el_duplicado_confirma_la_pierna_y_lo_ambiguo_no() {
        assert!(pierna_ya_existe(r#"Binance API Error: {"code":-4116,"msg":"ClientOrderId is duplicated."}"#));
        assert!(pierna_ya_existe(r#"Binance API Error: {"code": -4116, "msg": "x"}"#));
        assert!(pierna_ya_existe("REJECTED code=-4116 msg=Duplicate client order id"));
        assert!(!pierna_ya_existe("AMBIGUOUS: Network Error: operation timed out"));
        assert!(!pierna_ya_existe(r#"Binance API Error: {"code":-2021,"msg":"Order would immediately trigger."}"#));
        assert!(!pierna_ya_existe(r#"Binance API Error: {"code":-41160,"msg":"otro código"}"#));
    }

    /// El reintento de una pierna OCO conserva el clientAlgoId (no crea
    /// variantes _SLR/_TPR que dupliquen piernas) y la reversión parcial no
    /// descarta el resultado de la cancelación.
    #[test]
    fn lg6_el_reintento_oco_no_crea_ids_nuevos_ni_se_traga_cancelaciones() {
        let src = include_str!("executor.rs");
        let inicio = src
            .find("is_long_close: bool, // true if closing a long (Side = SELL)")
            .expect("execute_oco_order de la implementación");
        let cuerpo = &src[inicio..];
        let fin = cuerpo.find("/// Cancela una orden activa usando el client_order_id").expect("fin");
        let cuerpo = &cuerpo[..fin];
        let slr = ["{}_", "SLR"].concat();
        let tpr = ["{}_", "TPR"].concat();
        let descarte = ["let _ = self.", "cancel_algo_order("].concat();
        assert!(!cuerpo.contains(&slr), "el reintento no debe usar un id _SLR nuevo");
        assert!(!cuerpo.contains(&tpr), "el reintento no debe usar un id _TPR nuevo");
        assert!(!cuerpo.contains(&descarte), "la reversión parcial no debe descartar el resultado");
    }
}

#[cfg(test)]
mod tests_lg3 {
    use super::*;

    #[test]
    fn lg3_el_uso_caduca_con_su_ventana() {
        assert_eq!(uso_vigente(2000, 1_000, 1_500, 60_000), 2000);
        assert_eq!(uso_vigente(2000, 1_000, 61_000, 60_000), 0);
        assert_eq!(uso_vigente(250, 1_000, 10_999, 10_000), 250);
        assert_eq!(uso_vigente(250, 1_000, 11_000, 10_000), 0);
        // Sin sello o con reloj regresivo: conservador.
        assert_eq!(uso_vigente(2000, 0, 999_999, 60_000), 2000);
        assert_eq!(uso_vigente(2000, 5_000, 4_000, 60_000), 2000);
    }

    /// Antes: una cabecera de uso alto bloqueaba para siempre las salidas
    /// (headroom 80 %), porque ninguna petición salía y ninguna respuesta
    /// refrescaba el contador. Ahora el bloqueo dura lo que su ventana.
    #[test]
    fn lg3_una_cabecera_alta_no_bloquea_las_salidas_para_siempre() {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        let t0 = 1_700_000_000_000u64;
        ex.observar_limites(&limites(Some(2_100), None, None), t0); // > 0,8·2400
        assert!(ex.check_exit_rate_limits(t0 + 1_000, Tramo::Lectura).is_err(), "dentro de la ventana frena");
        assert!(ex.check_exit_rate_limits(t0 + 60_000, Tramo::Lectura).is_ok(), "pasada la ventana libera");
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        ex.observar_limites(&limites(None, Some(250), None), t0); // > 0,8·300
        assert!(ex.check_exit_rate_limits(t0 + 5_000, Tramo::Lectura).is_err());
        assert!(ex.check_exit_rate_limits(t0 + 10_000, Tramo::Lectura).is_ok());
    }

    pub(super) fn limites(
        weight_1m: Option<usize>,
        orders_10s: Option<usize>,
        orders_1m: Option<usize>,
    ) -> crate::client::BinanceRateLimits {
        crate::client::BinanceRateLimits {
            weight_1m,
            orders_10s,
            orders_1m,
        }
    }
}

#[cfg(test)]
mod tests_n5a {
    use super::tests_lg3::limites;
    use super::*;

    const T0: u64 = 1_700_000_000_000;

    /// RED antes de CL-177: con el último valor en llegar, una respuesta
    /// tardía (100) pisaba el uso de otra posterior (2000) en la misma ventana
    /// y el freno del 80 % se abría con el exchange a 2000 de 2400.
    #[test]
    fn n5a_una_respuesta_tardia_no_rebaja_el_uso() {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        ex.observar_limites(&limites(Some(2_000), None, None), T0);
        ex.observar_limites(&limites(Some(100), None, None), T0 + 1_000);
        let r = ex.check_exit_rate_limits(T0 + 2_000, Tramo::Lectura);
        assert!(
            matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_HEADROOM_WEIGHT: 2000/2400")),
            "{r:?}"
        );
        // La mayor caduca con su ventana; la última (100) no frena.
        assert!(ex.check_exit_rate_limits(T0 + 60_000, Tramo::Lectura).is_ok());
    }

    /// RED antes de CL-177: con un sello compartido, una cabecera de peso
    /// renovaba la edad de las órdenes de 10 s, que seguían frenando pasada su
    /// ventana.
    #[test]
    fn n5a_cada_contador_caduca_en_su_ventana() {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        ex.observar_limites(&limites(None, Some(250), None), T0);
        ex.observar_limites(&limites(Some(10), None, None), T0 + 9_000);
        let r = ex.check_exit_rate_limits(T0 + 5_000, Tramo::Lectura);
        assert!(
            matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_HEADROOM_ORDERS_10S: 250/300")),
            "{r:?}"
        );
        let r = ex.check_exit_rate_limits(T0 + 12_000, Tramo::Lectura);
        assert!(r.is_ok(), "las órdenes de 10 s caducan con su ventana: {r:?}");
    }

    /// La última muestra sigue valiendo cuando caduca la mayor, un contador que
    /// sube se registra, una cabecera ausente no toca su contador y, pasada la
    /// ventana de la mayor, una cabecera menor vuelve a mandar.
    #[test]
    fn n5a_la_ultima_vale_tras_la_mayor_y_la_cabecera_ausente_no_toca() {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        ex.observar_limites(&limites(Some(2_000), None, None), T0);
        ex.observar_limites(&limites(Some(1_950), None, None), T0 + 1_000);
        let r = ex.check_exit_rate_limits(T0 + 60_000, Tramo::Lectura);
        assert!(
            matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_HEADROOM_WEIGHT: 1950/2400")),
            "{r:?}"
        );
        assert!(ex.check_exit_rate_limits(T0 + 61_000, Tramo::Lectura).is_ok());

        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        ex.observar_limites(&limites(Some(100), Some(5), Some(7)), T0);
        ex.observar_limites(&limites(Some(2_100), None, None), T0 + 1_000);
        let r = ex.check_exit_rate_limits(T0 + 1_500, Tramo::Lectura);
        assert!(
            matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_HEADROOM_WEIGHT: 2100/2400")),
            "{r:?}"
        );
        {
            let uso = ex.uso_exchange_bloqueado();
            assert_eq!(uso.ordenes_10s.vigente(T0 + 1_500, 10_000), 5);
            assert_eq!(uso.ordenes_1m.vigente(T0 + 1_500, 60_000), 7);
        }
        // Pasada la ventana de la mayor, una cabecera menor vuelve a mandar.
        ex.observar_limites(&limites(Some(50), None, None), T0 + 61_000);
        assert!(ex.check_exit_rate_limits(T0 + 61_500, Tramo::Lectura).is_ok());
        assert_eq!(ex.uso_exchange_bloqueado().peso_1m.vigente(T0 + 61_500, 60_000), 50);
    }

    /// RED antes de CL-180: con sólo la mayor y la última, al caducar la mayor
    /// (2000) quedaba la última (1900 ≤ 1920) y se perdía la intermedia (1950),
    /// aún vigente: la respuesta tardía sí rebajaba el uso.
    #[test]
    fn n5a_la_mayor_vigente_sobrevive_a_otra_mayor_que_caduca() {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        ex.observar_limites(&limites(Some(2_000), None, None), T0);
        ex.observar_limites(&limites(Some(1_950), None, None), T0 + 59_000);
        ex.observar_limites(&limites(Some(1_900), None, None), T0 + 59_500);
        let r = ex.check_exit_rate_limits(T0 + 60_000, Tramo::Lectura);
        assert!(
            matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_HEADROOM_WEIGHT: 1950/2400")),
            "{r:?}"
        );
        // Caducada la intermedia, manda la última (1900 ≤ 0,8·2400).
        assert!(ex.check_exit_rate_limits(T0 + 119_000, Tramo::Lectura).is_ok());
        assert_eq!(ex.uso_exchange_bloqueado().peso_1m.vigente(T0 + 119_000, 60_000), 1900);
    }

    /// RED antes de CL-180: con el reloj adelantado 30 s, la mayor quedaba con
    /// un sello futuro y, corregido el reloj, ninguna respuesta menor la
    /// sustituía: frenaba hasta 30 s más allá de su ventana. La siguiente
    /// cabecera de ese contador acota el sello a ahora; sin ella, la muestra
    /// sigue vigente (lo conservador).
    #[test]
    fn n5a_un_sello_futuro_no_alarga_la_ventana() {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        ex.observar_limites(&limites(Some(2_100), None, None), T0 + 30_000);
        ex.observar_limites(&limites(Some(50), None, None), T0);
        let r = ex.check_exit_rate_limits(T0 + 1_000, Tramo::Lectura);
        assert!(
            matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_HEADROOM_WEIGHT: 2100/2400")),
            "la mayor conserva su valor: {r:?}"
        );
        let r = ex.check_exit_rate_limits(T0 + 60_000, Tramo::Lectura);
        assert!(r.is_ok(), "y caduca una ventana después del reloj corregido: {r:?}");
    }

    /// La cola guarda sólo las muestras que aún pueden mandar.
    #[test]
    fn n5a_la_cola_guarda_solo_las_que_pueden_mandar() {
        let mut c = ContadorExchange::default();
        for (i, uso) in [10, 20, 30, 25, 5, 40].into_iter().enumerate() {
            c.observar(uso, T0 + i as u64, 60_000);
        }
        // 40 llegó la última y es la mayor: tapa a todas.
        assert_eq!(c.muestras.len(), 1);
        assert_eq!(c.vigente(T0 + 10, 60_000), 40);
        c.observar(30, T0 + 10, 60_000);
        c.observar(20, T0 + 20, 60_000);
        assert_eq!(c.muestras.len(), 3);
        assert_eq!(c.vigente(T0 + 59_999, 60_000), 40);
        assert_eq!(c.vigente(T0 + 60_005, 60_000), 30);
        assert_eq!(c.vigente(T0 + 60_015, 60_000), 20);
        assert_eq!(c.vigente(T0 + 60_025, 60_000), 0);
        // Caducadas, salen al observar.
        c.observar(1, T0 + 200_000, 60_000);
        assert_eq!(c.muestras.len(), 1);
    }
}

#[cfg(test)]
mod tests_n5b {
    use super::*;

    const CODIGO: &str = include_str!("executor.rs");

    /// Cuerpo de la última `fn {f}(` del archivo (la del impl), sin `expect`.
    fn cuerpo(f: &str) -> &'static str {
        let Some(i) = CODIGO.rfind(&format!("fn {f}(")) else {
            return "";
        };
        let fin = ["\n    pub ", "\n    fn ", "\n    async fn "]
            .iter()
            .filter_map(|m| CODIGO[i + 3..].find(m).map(|k| i + 3 + k))
            .min()
            .unwrap_or(CODIGO.len());
        &CODIGO[i..fin]
    }

    /// Los `n` caracteres anteriores al byte `i` de `c`.
    fn antes(c: &str, i: usize, n: usize) -> &str {
        let ini = c[..i].char_indices().rev().nth(n - 1).map_or(0, |(k, _)| k);
        &c[ini..i]
    }

    /// Los `n` caracteres desde el byte `i` de `c`.
    fn despues(c: &str, i: usize, n: usize) -> &str {
        let fin = c[i..].char_indices().nth(n).map_or(c.len(), |(k, _)| i + k);
        &c[i..fin]
    }

    fn envios(c: &str, llamada: &str) -> Vec<usize> {
        c.match_indices(llamada).map(|(i, _)| i).collect()
    }

    /// `check_backoff` o la admisión por tramo, que lo llama primero (CL-179).
    fn frena(texto: &str) -> bool {
        texto.contains("check_backoff(") || texto.contains("check_exit_rate_limits(")
    }

    /// RED antes de CL-178: ningún envío de estas salidas consultaba el
    /// freno 429/418, y los cierres del aplanado no pasaban sus 429 a la
    /// política de rate-limit (el barrido seguía enviando dentro del cooldown).
    #[test]
    fn n5b_las_salidas_respetan_el_freno() {
        let aplanado = cuerpo("flatten_all_positions");
        let cierres = envios(aplanado, "execute_order_payload_typed(");
        assert_eq!(cierres.len(), 2, "cierre y reintento -4061");
        for i in cierres {
            assert!(frena(antes(aplanado, i, 400)), "cierre sin freno");
            assert!(
                despues(aplanado, i, 200).contains(".map_err(|e| self.handle_rate_limit_error(&e))"),
                "el 429 del cierre no arma el cooldown"
            );
        }
        let modos = envios(aplanado, "self.get_payload_account(");
        assert_eq!(modos.len(), 1, "GET de modo");
        for i in modos {
            assert!(frena(antes(aplanado, i, 400)), "modo sin freno");
        }
        // RV14: el freno se devuelve antes de firmar el GET de modo.
        assert!(
            aplanado.find("self.check_backoff(timestamp)?;").unwrap_or(usize::MAX)
                < aplanado.find("/fapi/v1/positionSide/dual").unwrap_or(0),
            "el aplanado frenado sale antes de preguntar el modo"
        );

        let purga = cuerpo("cancel_symbol_orders_and_brackets");
        let borrados = envios(purga, ".cancel_order_payload(");
        assert_eq!(borrados.len(), 1);
        assert!(frena(antes(purga, borrados[0], 400)), "DELETE sin freno");

        let oco = cuerpo("execute_oco_order");
        let piernas = envios(oco, ".execute_order_payload(");
        assert_eq!(piernas.len(), 4, "dos piernas y dos reintentos D-02");
        for &i in &piernas[2..] {
            assert!(frena(antes(oco, i, 400)), "reintento D-02 sin freno");
        }

        let cuenta = cuerpo("get_payload_account");
        let lecturas = envios(cuenta, ".get_payload_with_timeout(");
        assert_eq!(lecturas.len(), 2, "lectura y reintento");
        assert!(frena(antes(cuenta, lecturas[1], 400)), "reintento sin freno");
    }

    /// RED antes de CL-183: en X-026, un re-snapshot que no respondió
    /// (`.ok()`) se leía como «posición plana» y el aplanado purgaba los
    /// brackets de una posición quizá viva.
    #[test]
    fn n5b_un_re_snapshot_sin_respuesta_no_es_posicion_plana() {
        let aplanado = cuerpo("flatten_all_positions");
        assert!(!aplanado.contains("fetch_position_risk().await.ok()"));
        let x026 = aplanado.split("// X-026 (REHAB-3)").nth(1).unwrap_or("");
        let brazo = x026.split("Err(e2) =>").nth(1).unwrap_or("");
        let brazo = brazo.split("Err(e) =>").next().unwrap_or("");
        assert!(brazo.contains("se trata como ABIERTA"), "sin brazo de error en X-026");
        assert!(!brazo.contains("position_closed = true"), "el error no cierra la posición");
    }

    fn fila(simbolo: &str, lado: &str, cantidad: &str) -> String {
        format!(r#"{{"symbol":"{simbolo}","positionSide":"{lado}","positionAmt":"{cantidad}"}}"#)
    }

    /// RED antes de CL-178: en hedge positionRisk trae una fila por lado y el
    /// barrido de huérfanas purgaba dos veces cada símbolo plano.
    #[test]
    fn n5b_barrido_sin_duplicados() {
        let json = format!(
            "[{},{},{},{},{},{}]",
            fila("BTCUSDT", "LONG", "0"),
            fila("BTCUSDT", "SHORT", "0"),
            fila("ETHUSDT", "LONG", "0.5"),
            fila("ETHUSDT", "SHORT", "0"),
            fila("SOLUSDT", "BOTH", "0"),
            fila("ADAUSDT", "BOTH", "-3"),
        );
        let entries: Vec<crate::reconciliation::PositionRiskEntry> =
            serde_json::from_str(&json).unwrap_or_default();
        assert_eq!(entries.len(), 6);
        assert_eq!(
            simbolos_sin_posicion(&entries),
            vec!["BTCUSDT".to_string(), "SOLUSDT".to_string()]
        );
    }

    /// El freno de CL-178 es el mismo que ya aplicaban las salidas: el
    /// cooldown de un 429 y el freno escalado, sin cuota ni kill-switch.
    #[test]
    fn n5b_backoff_es_el_cooldown_y_el_freno() {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        let ahora = current_synced_timestamp_ms(None);
        assert!(ex.check_backoff(ahora).is_ok());
        ex.trigger_kill_switch();
        assert!(ex.check_backoff(ahora).is_ok(), "el kill-switch no frena las salidas");
        ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=30");
        let r = ex.check_backoff(ahora);
        assert!(matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_COOLDOWN")), "{r:?}");
        assert!(ex.check_backoff(ahora + 31_000).is_ok());
        ex.handle_rate_limit_error("HTTP_418_IP_BANNED retry_after=120");
        let r = ex.check_backoff(ahora + 31_000);
        assert!(matches!(&r, Err(e) if e.starts_with("RATE_BRAKE")), "{r:?}");
        assert!(ex.check_backoff(ahora + 121_000).is_ok());
    }
}

#[cfg(test)]
mod tests_n5c {
    use super::tests_lg3::limites;
    use super::*;
    use crate::binance_api::{
        COSTE_CANCELACION, COSTE_ORDEN, COSTE_PIERNA_ALGO, COSTE_POSITION_RISK,
    };

    const T0: u64 = 1_700_000_000_000;

    fn ejecutor_con(
        peso: Option<usize>,
        o10: Option<usize>,
        o1m: Option<usize>,
        t: u64,
    ) -> OrderExecutor {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        ex.observar_limites(&limites(peso, o10, o1m), t);
        ex
    }

    fn critica(ex: &OrderExecutor, t: u64, c: crate::binance_api::CosteBinance) -> Result<(), String> {
        ex.check_exit_rate_limits(t, Tramo::Critica(c)).map(|_cupo| ())
    }

    /// RED antes de CL-179: en la franja (0,8·L, L] los cierres reduce-only
    /// frenaban como una lectura (RATE_LIMIT_HEADROOM_*) aunque el exchange
    /// los admitía. Modo papel: la cuota se consulta antes de la rama de papel
    /// y no se toca la red.
    #[tokio::test]
    async fn n5c_cierre_reduce_only_no_se_frena_en_la_franja() {
        let cabeceras = [
            (Some(2_100), None, None),
            (None, Some(250), None),
            (None, None, Some(1_000)),
        ];
        for (peso, o10, o1m) in cabeceras {
            let mut ex = OrderExecutor::new("k".into(), "s".into(), true);
            ex.set_paper_trading(true);
            ex.observar_limites(&limites(peso, o10, o1m), ex.get_synced_timestamp());
            let r = ExecutionProvider::execute_reduce_only_market(&ex, "BTCUSDT", true, 0.001, 0.001).await;
            assert!(r.is_ok(), "{peso:?}/{o10:?}/{o1m:?}: {r:?}");
            let r = ex
                .execute_reduce_only_market_with_report("BTCUSDT", true, 0.001, 0.001)
                .await;
            assert!(r.is_ok(), "{peso:?}/{o10:?}/{o1m:?}: {r:?}");
        }
    }

    /// El tope crítico es el del exchange: pasa mientras usado + coste ≤ L, en
    /// cada contador con su coste; las lecturas siguen en 0,8·L.
    #[test]
    fn n5c_el_tope_critico_es_el_limite_del_exchange() {
        let ex = ejecutor_con(Some(2_395), None, None, T0);
        assert!(critica(&ex, T0 + 1, COSTE_POSITION_RISK).is_ok(), "2395 + 5 = 2400");
        let ex = ejecutor_con(Some(2_396), None, None, T0);
        let r = critica(&ex, T0 + 1, COSTE_POSITION_RISK);
        assert!(
            matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_EXHAUSTED_WEIGHT: 2396 usado + 0 en vuelo + 5 > 2400")),
            "{r:?}"
        );
        // Una orden no pesa: con el peso en el límite pasa; una cancelación no.
        let ex = ejecutor_con(Some(2_400), None, None, T0);
        assert!(critica(&ex, T0 + 1, COSTE_ORDEN).is_ok());
        assert!(critica(&ex, T0 + 1, COSTE_CANCELACION).is_err());

        let ex = ejecutor_con(None, Some(299), None, T0);
        assert!(critica(&ex, T0 + 1, COSTE_ORDEN).is_ok(), "299 + 1 = 300");
        let ex = ejecutor_con(None, Some(300), None, T0);
        let r = critica(&ex, T0 + 1, COSTE_ORDEN);
        assert!(matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_EXHAUSTED_ORDERS_10S")), "{r:?}");
        assert!(critica(&ex, T0 + 1, COSTE_POSITION_RISK).is_ok(), "una lectura no cuenta órdenes");
        assert!(critica(&ex, T0 + 10_000, COSTE_ORDEN).is_ok(), "caduca con su ventana");
        let ex = ejecutor_con(None, None, Some(1_200), T0);
        let r = critica(&ex, T0 + 1, COSTE_PIERNA_ALGO);
        assert!(matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_EXHAUSTED_ORDERS_1M")), "{r:?}");
        assert!(critica(&ex, T0 + 60_000, COSTE_PIERNA_ALGO).is_ok());

        // La franja: la lectura frena, la crítica pasa.
        let ex = ejecutor_con(Some(2_100), None, None, T0);
        let r = ex.check_exit_rate_limits(T0 + 1, Tramo::Lectura);
        assert!(matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_HEADROOM_WEIGHT: 2100/2400")), "{r:?}");
        assert!(ex.check_rate_limits(T0 + 1).is_err(), "las entradas tampoco");
        assert!(critica(&ex, T0 + 1, COSTE_ORDEN).is_ok());
    }

    /// Las críticas admitidas y sin respuesta cuentan como usadas: N
    /// concurrentes no caben juntas en el hueco de una. Soltar el cupo las
    /// saca del libro; las lecturas no entran en él.
    #[test]
    fn n5c_las_criticas_en_vuelo_cuentan() {
        let ex = ejecutor_con(Some(2_390), None, None, T0);
        let primera = ex.check_exit_rate_limits(T0 + 1, Tramo::Critica(COSTE_POSITION_RISK));
        assert!(primera.is_ok());
        let segunda = ex.check_exit_rate_limits(T0 + 1, Tramo::Critica(COSTE_POSITION_RISK));
        assert!(segunda.is_ok(), "2390 + 5 + 5 = 2400");
        let r = critica(&ex, T0 + 1, COSTE_POSITION_RISK);
        assert!(
            matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_EXHAUSTED_WEIGHT: 2390 usado + 10 en vuelo + 5 > 2400")),
            "{r:?}"
        );
        drop(primera);
        assert!(critica(&ex, T0 + 1, COSTE_POSITION_RISK).is_ok(), "la primera respondió");
        drop(segunda);
        assert_eq!(ex.uso_exchange_bloqueado().en_vuelo, [0, 0, 0]);

        let ex = ejecutor_con(None, Some(298), None, T0);
        let par = ex.check_exit_rate_limits(T0 + 1, Tramo::Critica(COSTE_PIERNA_ALGO.por(2)));
        assert!(par.is_ok(), "298 + 2 = 300");
        assert_eq!(ex.uso_exchange_bloqueado().en_vuelo, [2, 2, 2]);
        assert!(critica(&ex, T0 + 1, COSTE_ORDEN).is_err(), "el par OCO llena la ventana de 10 s");
        drop(par);
        assert_eq!(ex.uso_exchange_bloqueado().en_vuelo, [0, 0, 0]);

        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        let par = ex.check_exit_rate_limits(T0 + 1, Tramo::Critica(COSTE_PIERNA_ALGO.por(2)));
        let lectura = ex.check_exit_rate_limits(T0 + 1, Tramo::Lectura);
        assert!(par.is_ok() && lectura.is_ok());
        assert_eq!(ex.uso_exchange_bloqueado().en_vuelo, [2, 2, 2], "la lectura no entra en el libro");
        drop(lectura);
        assert_eq!(ex.uso_exchange_bloqueado().en_vuelo, [2, 2, 2]);
        drop(par);
        assert_eq!(ex.uso_exchange_bloqueado().en_vuelo, [0, 0, 0]);
    }

    /// RED antes de CL-181: un cierre (peso 0) se rechazaba por el peso que
    /// tenía en vuelo un positionRisk, y un positionRisk (0 órdenes) por el par
    /// OCO en vuelo, aunque el exchange admite los dos: no gastan en ese
    /// contador.
    #[test]
    fn n5c_un_contador_sin_coste_no_frena_por_lo_que_esta_en_vuelo() {
        let ex = ejecutor_con(Some(2_390), None, None, T0);
        let pr = ex.check_exit_rate_limits(T0 + 1, Tramo::Critica(COSTE_POSITION_RISK));
        assert!(pr.is_ok());
        ex.observar_limites(&limites(Some(2_400), None, None), T0 + 2);
        let r = critica(&ex, T0 + 2, COSTE_ORDEN);
        assert!(r.is_ok(), "un cierre no pesa: {r:?}");
        let r = critica(&ex, T0 + 2, COSTE_CANCELACION);
        assert!(
            matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_EXHAUSTED_WEIGHT: 2400 usado + 5 en vuelo + 1 > 2400")),
            "una cancelación sí: {r:?}"
        );
        drop(pr);

        let ex = ejecutor_con(None, Some(298), None, T0);
        let par = ex.check_exit_rate_limits(T0 + 1, Tramo::Critica(COSTE_PIERNA_ALGO.por(2)));
        assert!(par.is_ok());
        ex.observar_limites(&limites(None, Some(300), None), T0 + 2);
        let r = critica(&ex, T0 + 2, COSTE_POSITION_RISK);
        assert!(r.is_ok(), "positionRisk no cuenta órdenes: {r:?}");
        assert!(critica(&ex, T0 + 2, COSTE_ORDEN).is_err());
        drop(par);
    }

    /// El tramo crítico respeta los frenos del exchange y no el kill-switch
    /// (CL-3); las entradas siguen bloqueadas por él.
    #[test]
    fn n5c_frenos_y_kill_switch_en_el_tramo_critico() {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        let ahora = current_synced_timestamp_ms(None);
        ex.trigger_kill_switch();
        assert!(critica(&ex, ahora, COSTE_ORDEN).is_ok());
        assert!(ex.check_exit_rate_limits(ahora, Tramo::Lectura).is_ok());
        assert!(matches!(ex.check_rate_limits(ahora), Err(e) if e.contains("KILL SWITCH")));
        ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=30");
        let r = critica(&ex, ahora, COSTE_ORDEN);
        assert!(matches!(&r, Err(e) if e.starts_with("RATE_LIMIT_COOLDOWN")), "{r:?}");
        ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=30");
        ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=30");
        let r = critica(&ex, ahora + 31_000, COSTE_ORDEN);
        assert!(matches!(&r, Err(e) if e.starts_with("RATE_BRAKE")), "{r:?}");
    }

    /// Guarda de fuente: cada ruta lleva su tramo. Críticas: lo que cierra,
    /// protege o sostiene una posición. Lecturas: el resto. El listado algo
    /// es crítico con símbolo y lectura sin él; el barrido de huérfanas pasa
    /// por el de lectura; y los cupos que preceden a otra admisión se sueltan
    /// antes de ella.
    #[test]
    fn n5c_tramos_por_ruta() {
        let codigo = include_str!("executor.rs");
        let cuerpo = |f: &str| -> &str {
            let Some(i) = codigo.rfind(&format!("fn {f}(")) else {
                return "";
            };
            let fin = ["\n    pub ", "\n    fn ", "\n    async fn "]
                .iter()
                .filter_map(|m| codigo[i + 3..].find(m).map(|k| i + 3 + k))
                .min()
                .unwrap_or(codigo.len());
            &codigo[i..fin]
        };
        for f in [
            "fetch_position_risk",
            "cancel_algo_order_ids",
            "cancel_all_algo_open_orders",
            "place_algo_leg",
            "execute_reduce_only_market_with_report",
            "execute_reduce_only_market",
            "execute_exchange_trailing_stop",
            "execute_oco_order",
            "cancel_order",
            "cancel_all_symbol_orders",
            "query_order",
            "cancel_symbol_orders_and_brackets",
        ] {
            let c = cuerpo(f);
            assert!(c.contains("Tramo::Critica("), "{f} es crítica");
            assert!(!c.contains("Tramo::Lectura"), "{f} no lee en el tramo de lectura");
        }
        for f in [
            "fetch_income_page",
            "fetch_open_orders_inner",
            "fetch_open_positions",
            "fetch_account_balance",
            "fetch_commission_rate",
            "check_rate_limits",
        ] {
            let c = cuerpo(f);
            assert!(c.contains("Tramo::Lectura"), "{f} es lectura");
            assert!(!c.contains("Tramo::Critica("), "{f} no es crítica");
        }
        let algo = cuerpo("fetch_open_algo_orders_inner");
        assert!(
            algo.contains("Some(_) => Tramo::Critica(crate::binance_api::COSTE_ALGO_ABIERTAS_SIMBOLO)"),
            "el listado algo de un símbolo es crítico"
        );
        assert!(algo.contains("None => Tramo::Lectura"), "el listado algo sin símbolo es lectura");
        assert_eq!(algo.matches("check_exit_rate_limits(").count(), 1, "una sola admisión");
        assert!(
            algo.contains("self.check_exit_rate_limits(timestamp, tramo)?"),
            "el tramo elegido es el que se admite"
        );
        assert!(cuerpo("fetch_open_algo_orders").contains("inner(Some(symbol))"));
        assert!(cuerpo("fetch_all_open_algo_orders").contains("inner(None)"));

        let aplanado = cuerpo("flatten_all_positions");
        assert_eq!(aplanado.matches("Tramo::Critica(crate::binance_api::COSTE_ORDEN)").count(), 2);
        let barrido = aplanado.split("for sym in &sin_posicion {").nth(1).unwrap_or("");
        assert!(
            barrido
                .split("self.cancel_symbol_orders_and_brackets(")
                .next()
                .is_some_and(|antes| antes.contains("Tramo::Lectura")),
            "el barrido pasa por el tramo de lectura"
        );

        let oco = cuerpo("execute_oco_order");
        let suelta = oco.find("drop(cupo);").unwrap_or(usize::MAX);
        assert!(suelta < oco.find("if sl_down {").unwrap_or(0), "el par OCO sale del libro antes de reintentar");
        let todas = cuerpo("cancel_all_symbol_orders");
        assert!(
            todas.find("drop(cupo);").unwrap_or(usize::MAX)
                < todas.find("self.cancel_all_algo_open_orders(").unwrap_or(0),
            "el DELETE sale del libro antes de la purga algo"
        );
    }

    /// RED antes de CL-182: en los brazos `Ok(_cupo) => envío.await` el cupo
    /// caía al acabar el brazo y la cabecera se registraba después; entre
    /// medias el coste no estaba ni en vuelo ni en el uso y otra admisión
    /// crítica podía ocupar su hueco.
    #[test]
    fn n5c_la_cabecera_se_registra_con_el_cupo_vivo() {
        let codigo = include_str!("executor.rs");
        let cuerpo = |f: &str| -> &str {
            let Some(i) = codigo.rfind(&format!("fn {f}(")) else {
                return "";
            };
            let fin = ["\n    pub ", "\n    fn ", "\n    async fn "]
                .iter()
                .filter_map(|m| codigo[i + 3..].find(m).map(|k| i + 3 + k))
                .min()
                .unwrap_or(codigo.len());
            &codigo[i..fin]
        };
        for (f, n) in [
            ("cancel_symbol_orders_and_brackets", 1),
            ("flatten_all_positions", 2),
            ("execute_oco_order", 2),
        ] {
            let c = cuerpo(f);
            let brazos: Vec<&str> = c
                .match_indices("Ok(_cupo) =>")
                .filter(|&(i, _)| c[i.saturating_sub(200)..i].contains("Tramo::Critica("))
                .map(|(i, _)| c[i..].split("Err(e) => Err(e)").next().unwrap_or(""))
                .collect();
            assert_eq!(brazos.len(), n, "{f}: brazos críticos");
            for b in brazos {
                assert!(b.contains("self.update_limits("), "{f}: cabecera tras soltar el cupo: {b}");
            }
        }
    }
}

#[cfg(test)]
mod tests_decima_ola {
    use super::*;

    /// D-628: con spread de 1 y 2 ticks el precio pasivo se une al mejor nivel
    /// propio; con 3 ticks mejora dentro del spread sin cruzarlo.
    #[test]
    fn d628_precio_pasivo_se_une_al_mejor_nivel() {
        let t = 0.1;
        // spread 1 tick
        let (bid, ask) = (60_000.0, 60_000.1);
        let mid = (bid + ask) / 2.0;
        assert!((passive_join_price(mid, t, false) - bid).abs() < 1e-9);
        assert!((passive_join_price(mid, t, true) - ask).abs() < 1e-9);
        // spread 2 ticks
        let (bid, ask) = (60_000.0, 60_000.2);
        let mid = (bid + ask) / 2.0;
        assert!((passive_join_price(mid, t, false) - bid).abs() < 1e-9);
        assert!((passive_join_price(mid, t, true) - ask).abs() < 1e-9);
        // spread 3 ticks: dentro del spread y sin cruzar
        let (bid, ask) = (60_000.0, 60_000.3);
        let mid = (bid + ask) / 2.0;
        let buy = passive_join_price(mid, t, false);
        let sell = passive_join_price(mid, t, true);
        assert!(buy >= bid - 1e-9 && buy < ask - 1e-9);
        assert!(sell <= ask + 1e-9 && sell > bid + 1e-9);
    }

    /// D-629: el redondeo no deja residuo flotante que `ryu` imprima.
    #[test]
    fn d629_redondeo_sin_residuo_flotante() {
        let q = OrderExecutor::round_to_step_size(0.123456, 0.00001);
        assert_eq!(ryu::Buffer::new().format(q), "0.12345");
        let p = OrderExecutor::round_price_to_tick(0.30000000000000004, 0.1, false);
        assert_eq!(ryu::Buffer::new().format(p), "0.3");
    }

    /// D-629: el ajuste a múltiplo exacto es relativo, no absoluto.
    #[test]
    fn d629_redondeo_invariante_de_escala() {
        let casi = 3_000.0 - 3_000.0 * 1e-13;
        assert_eq!(OrderExecutor::round_to_step_size(casi, 1.0), 3_000.0);
        let grande = 1_500_000.0 - 1_500_000.0 * 1e-13;
        assert_eq!(OrderExecutor::round_to_step_size(grande, 1.0), 1_500_000.0);
        // un valor genuinamente por debajo del múltiplo sí se trunca
        assert_eq!(OrderExecutor::round_to_step_size(2_999.5, 1.0), 2_999.0);
    }

    /// D-630: stops y objetivos se redondean ALEJÁNDOSE de la entrada.
    #[test]
    fn d630_stops_y_objetivos_se_alejan_de_la_entrada() {
        let t = 0.1;
        // largo: stop hacia abajo, objetivo hacia arriba
        assert!(
            (OrderExecutor::round_stop_away_from_entry(59_000.05, t, true) - 59_000.0).abs() < 1e-9
        );
        assert!(
            (OrderExecutor::round_target_away_from_entry(61_000.05, t, true) - 61_000.1).abs()
                < 1e-9
        );
        // corto: stop hacia arriba, objetivo hacia abajo
        assert!(
            (OrderExecutor::round_stop_away_from_entry(61_000.05, t, false) - 61_000.1).abs()
                < 1e-9
        );
        assert!(
            (OrderExecutor::round_target_away_from_entry(59_000.05, t, false) - 59_000.0).abs()
                < 1e-9
        );
    }

    /// D-629: pasos o precios inválidos no producen cantidades enviables.
    #[test]
    fn d629_entradas_invalidas_devuelven_cero() {
        assert_eq!(OrderExecutor::round_to_step_size(1.0, 0.0), 0.0);
        assert_eq!(OrderExecutor::round_to_step_size(1.0, f64::NAN), 0.0);
        assert_eq!(
            OrderExecutor::round_price_to_tick(60_000.0, f64::NAN, true),
            0.0
        );
        assert_eq!(passive_join_price(f64::NAN, 0.1, false), 0.0);    }
}

#[cfg(test)]
mod tests_b3_audit {
    use super::*;
    use crate::order_types::IncomeEntry;

    fn entry(tran: u64, time: u64, sym: &str, itype: &str, income: f64) -> IncomeEntry {
        IncomeEntry {
            symbol: sym.into(),
            income_type: itype.into(),
            income,
            asset: "USDT".into(),
            time,
            tran_id: tran,
            trade_id: String::new(),
        }
    }

    /// B3.6b-aud: el solapamiento del cursor (entradas re-servidas del ms
    /// frontera) se colapsa por tranId — ni se pierden entradas nuevas del
    /// mismo ms ni se duplican las frontera.
    #[test]
    fn b36b_merge_page_dedup_por_tranid() {
        let mut acc = Vec::new();
        let mut seen = std::collections::HashSet::new();
        // Página 1: tres entradas, dos comparten ms=2000.
        let p1 = vec![
            entry(1, 1000, "SOLUSDT", "COMMISSION", -0.01),
            entry(2, 2000, "SOLUSDT", "REALIZED_PNL", -1.2),
            entry(3, 2000, "SOLUSDT", "COMMISSION", -0.01),
        ];
        assert_eq!(merge_income_page(&mut acc, &mut seen, p1), 3);
        assert_eq!(acc.len(), 3);

        // Página 2 (cursor=2000): re-sirve las dos del ms 2000 (dupes) y
        // agrega una nueva del mismo ms 2000 y una posterior.
        let p2 = vec![
            entry(2, 2000, "SOLUSDT", "REALIZED_PNL", -1.2),
            entry(3, 2000, "SOLUSDT", "COMMISSION", -0.01),
            entry(4, 2000, "SOLUSDT", "COMMISSION", -0.02),
            entry(5, 3000, "SOLUSDT", "FUNDING_FEE", 0.004),
        ];
        assert_eq!(merge_income_page(&mut acc, &mut seen, p2), 2);
        assert_eq!(acc.len(), 5);
        // La entrada #4 (mismo ms que las dupes, tranId distinto) sobrevive.
        assert!(acc.iter().any(|e| e.tran_id == 4));
        assert!(acc.iter().any(|e| e.tran_id == 5));
    }

    /// tranId=0 (payload degradado): la tupla completa actúa de clave.
    #[test]
    fn b36b_merge_page_dedup_sin_tranid() {
        let mut acc = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let p1 = vec![entry(0, 1000, "BTCUSDT", "COMMISSION", -0.05)];
        merge_income_page(&mut acc, &mut seen, p1);
        // Idéntica re-servida: dup. Distinta income mismo ms: nueva.
        let p2 = vec![
            entry(0, 1000, "BTCUSDT", "COMMISSION", -0.05),
            entry(0, 1000, "BTCUSDT", "REALIZED_PNL", 3.0),
        ];
        assert_eq!(merge_income_page(&mut acc, &mut seen, p2), 1);
        assert_eq!(acc.len(), 2);
    }
}

#[cfg(test)]
mod tests_m4_c01 {
    use super::*;
    use crate::order_types::OrderAck;

    fn executor() -> OrderExecutor {
        OrderExecutor::new("test_key".to_string(), "test_secret".to_string(), true)
    }

    /// apply_ack se keyea por ack.client_order_id — DEBE coincidir con la clave
    /// de register_intent o el ack aterriza en una entrada separada del registry.
    fn ack(coid: &str, status: &str, order_id: u64, executed: f64) -> OrderAck {
        OrderAck {
            client_order_id: coid.to_string(),
            status: status.to_string(),
            order_id,
            executed_qty: executed,
            ..Default::default()
        }
    }

    /// M4-C01 núcleo: un intent registrado (status=New, order_id=0) NO es una
    /// confirmación del exchange. Antes del fix la ruta WS retornaba Ok(()) en
    /// este punto (fire-and-forget: la orden podía perderse en silencio si la
    /// conexión moría). Ahora debe agotar el deadline y reportar Timeout.
    #[tokio::test]
    async fn m4c01_intent_only_never_confirms() {
        let ex = executor();
        ex.registry()
            .register_intent("c1", "BTCUSDT", "BUY", "LONG", "MARKET", 1.0, 1000);
        assert_eq!(
            ex.await_resolution("c1", 200).await,
            OrderResolution::Timeout
        );
    }

    /// Solo un ack DEL exchange (order_id > 0) resuelve: NEW ⇒ Accepted,
    /// REJECTED ⇒ Rejected.
    #[tokio::test]
    async fn m4c01_exchange_ack_accepts_and_rejects() {
        let ex = executor();
        ex.registry()
            .register_intent("c2", "BTCUSDT", "BUY", "LONG", "MARKET", 1.0, 1000);
        ex.registry().apply_ack(&ack("c2", "NEW", 1234, 0.0), 1100);
        assert_eq!(
            ex.await_resolution("c2", 200).await,
            OrderResolution::Accepted
        );

        ex.registry()
            .register_intent("c3", "BTCUSDT", "BUY", "LONG", "MARKET", 1.0, 1000);
        ex.registry().apply_ack(&ack("c3", "REJECTED", 1235, 0.0), 1100);
        assert_eq!(
            ex.await_resolution("c3", 200).await,
            OrderResolution::Rejected
        );
    }

    /// Fill parcial antes de un terminal ⇒ existe posición real ⇒ Accepted
    /// (executed_qty > 0 domina sobre el estado CANCELED: el remanente cancelado
    /// no borra la qty ya ejecutada que el host debe bracketear).
    #[tokio::test]
    async fn m4c01_partial_fill_before_terminal_is_accepted() {
        let ex = executor();
        ex.registry()
            .register_intent("c4", "BTCUSDT", "BUY", "LONG", "MARKET", 1.0, 1000);
        ex.registry().apply_ack(&ack("c4", "CANCELED", 1236, 0.4), 1100);
        assert_eq!(
            ex.await_resolution("c4", 200).await,
            OrderResolution::Accepted
        );
    }

    /// Paper trading: sin exchange que consultar ⇒ el gate no debe bloquear ni
    /// falsear rechazos. await_resolution y resolve_via_rest cortocircuitan a
    /// Accepted, y confirm_ws_dispatch retorna Ok(()).
    #[tokio::test]
    async fn m4c01_paper_trading_short_circuits() {
        let mut ex = executor();
        ex.set_paper_trading(true);
        assert_eq!(
            ex.await_resolution("whatever", 60_000).await,
            OrderResolution::Accepted
        );
        assert_eq!(
            ex.resolve_via_rest("BTCUSDT", "whatever").await,
            OrderResolution::Accepted
        );
        assert!(ex.confirm_ws_dispatch("BTCUSDT", "whatever").await.is_ok());
    }
}

#[cfg(test)]
mod tests_query_evidence {
    use super::*;

    fn ack(status: &str, qty: f64) -> OrderAck {
        OrderAck {
            symbol: "AUDITUSDT".into(),
            client_order_id: "intent-1".into(),
            side: "BUY".into(),
            order_id: 123,
            orig_qty: 1.0,
            executed_qty: qty,
            status: status.into(),
            ..Default::default()
        }
    }

    #[test]
    fn invalid_or_missing_query_does_not_mutate_the_registry() {
        let ex = OrderExecutor::new("fixture".into(), "fixture".into(), true);
        let mut foreign = ack("NEW", 0.0);
        foreign.symbol = "OTHERUSDT".into();
        for result in [Ok(&foreign), Err("-2013"), Err("network timeout")] {
            assert_eq!(ex.record_query_resolution("AUDITUSDT", "intent-1", result), OrderResolution::Timeout);
        }
        assert_eq!(ex.registry().stats().total, 0);
    }

    #[test]
    fn known_intent_side_and_order_identity_cannot_be_replaced() {
        let ex = OrderExecutor::new("fixture".into(), "fixture".into(), true);
        ex.registry().register_intent("intent-1", "AUDITUSDT", "SELL", "SHORT", "LIMIT", 1.0, 1);
        assert_eq!(ex.record_query_resolution("AUDITUSDT", "intent-1", Ok(&ack("NEW", 0.0))), OrderResolution::Timeout);
        assert_eq!(ex.registry().get("intent-1").unwrap().side, "SELL");
        let mut original = ack("NEW", 0.0);
        original.side = "SELL".into();
        ex.registry().apply_ack(&original, 2);
        original.order_id += 1;
        assert_eq!(ex.record_query_resolution("AUDITUSDT", "intent-1", Ok(&original)), OrderResolution::Timeout);
        assert_eq!(ex.registry().get("intent-1").unwrap().order_id, 123);
    }

    #[test]
    fn stale_zero_fill_terminal_cannot_erase_a_known_partial_fill() {
        let ex = OrderExecutor::new("fixture".into(), "fixture".into(), true);
        let partial = ack("PARTIALLY_FILLED", 0.4);
        assert_eq!(ex.record_query_resolution("AUDITUSDT", "intent-1", Ok(&partial)), OrderResolution::Accepted);
        let stale_terminal = ack("CANCELED", 0.0);
        assert_eq!(ex.record_query_resolution("AUDITUSDT", "intent-1", Ok(&stale_terminal)), OrderResolution::Accepted);
        assert_eq!(ex.registry().get("intent-1").unwrap().executed_qty, 0.4);
    }

    #[test]
    fn valid_terminal_nofill_and_new_keep_different_outcomes() {
        let ex = OrderExecutor::new("fixture".into(), "fixture".into(), true);
        assert_eq!(ex.record_query_resolution("AUDITUSDT", "intent-1", Ok(&ack("NEW", 0.0))), OrderResolution::Accepted);
        assert_eq!(ex.record_query_resolution("AUDITUSDT", "intent-1", Ok(&ack("CANCELED", 0.0))), OrderResolution::Rejected);
    }
}

#[cfg(test)]
mod tests_m4_c02 {
    use super::*;

    fn executor() -> OrderExecutor {
        OrderExecutor::new("test_key".to_string(), "test_secret".to_string(), true)
    }

    /// M4-C02 path #3: el freno por rate-limit debe ser TEMPORAL, no permanente.
    /// Antes: 3×429 (transitorio, ya cubierto por cooldown) llamaba
    /// trigger_kill_switch() → kill_switch latcheado PARA SIEMPRE →
    /// check_rate_limits bloqueaba TODA ejecución (incluido el cierre de
    /// posiciones) hasta reiniciar el proceso → riesgo de liquidación.
    /// Ahora: freno con auto-expiración; el flag permanente NO se arma.
    #[test]
    fn m4c02_rate_brake_is_time_bound_not_permanent() {
        let ex = executor();
        let now = current_synced_timestamp_ms(None);
        for _ in 0..3 {
            ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=1");
        }
        // El kill-switch permanente (incidentes reales) NO se arma por rate-limit.
        assert!(!ex.is_kill_switch_active());
        // Dentro de la ventana (brake ≈ now + 1s + 60s): bloqueado por RATE_BRAKE.
        let blocked = ex.check_rate_limits(now);
        assert!(
            matches!(&blocked, Err(e) if e.starts_with("RATE_BRAKE")),
            "esperado RATE_BRAKE dentro de la ventana, got {:?}",
            blocked
        );
        // Tras la ventana (+120s > 61s): el freno AUTO-EXPIRA (antes: permanente).
        let cleared = ex.check_rate_limits(now + 120_000);
        assert!(
            !matches!(&cleared, Err(e) if e.starts_with("RATE_BRAKE")),
            "el freno no expiró tras la ventana: {:?}",
            cleared
        );
    }

    /// HTTP 418 (IP ban) tampoco debe latchear permanentemente: freno por la
    /// ventana de ban, auto-expira. Si el ban persiste, el próximo 418 re-latchea.
    #[test]
    fn m4c02_http_418_brake_expires() {
        let ex = executor();
        let now = current_synced_timestamp_ms(None);
        ex.handle_rate_limit_error("HTTP_418_IP_BANNED retry_after=1");
        assert!(!ex.is_kill_switch_active());
        // ban_s=1 → brake = now + 1s. Dentro: bloqueado.
        assert!(
            matches!(ex.check_rate_limits(now), Err(e) if e.starts_with("RATE_BRAKE"))
        );
        // +10s: expirado.
        assert!(
            !matches!(ex.check_rate_limits(now + 10_000), Err(e) if e.starts_with("RATE_BRAKE"))
        );
    }
    /// CL-3: tras el kill-switch, las salidas siguen disponibles. Antes
    /// `flatten_all_positions` abortaba en su primera lectura
    /// (`fetch_position_risk`) con «KILL SWITCH ACTIVE».
    #[test]
    fn cl3_el_kill_switch_no_bloquea_las_salidas() {
        let ex = executor();
        let now = current_synced_timestamp_ms(None);
        ex.trigger_kill_switch();
        assert!(ex.is_kill_switch_active());
        assert!(
            matches!(ex.check_rate_limits(now), Err(e) if e.contains("KILL SWITCH")),
            "las entradas deben seguir bloqueadas"
        );
        assert!(ex.check_exit_rate_limits(now, Tramo::Lectura).is_ok());
    }

    /// CL-3: guardia de fuente. Cada ruta de lectura, cancelación o cierre
    /// reduce-only (las que usa `flatten_all_positions` y los cierres del
    /// núcleo) usa la variante que ignora el kill-switch; las de entrada no.
    #[test]
    fn cl3_rutas_de_salida_y_de_entrada() {
        let codigo = include_str!("executor.rs");
        let cuerpo = |f: &str| {
            let i = codigo.rfind(&format!("fn {f}(")).expect(f);
            let fin = ["\n    pub ", "\n    fn ", "\n    async fn "]
                .iter()
                .filter_map(|m| codigo[i + 3..].find(m).map(|k| i + 3 + k))
                .min()
                .unwrap_or(codigo.len());
            &codigo[i..fin]
        };
        for f in [
            "fetch_position_risk",
            "cancel_all_algo_open_orders",
            "cancel_all_symbol_orders",
            "cancel_order",
            "execute_reduce_only_market",
            "fetch_open_positions",
            "fetch_account_balance",
            // CL-20: piernas protectoras (reductoras por construcción).
            "place_algo_leg",
            "execute_oco_order",
            "execute_exchange_trailing_stop",
        ] {
            assert!(cuerpo(f).contains("check_exit_rate_limits("), "{f} debe ignorar el kill-switch");
        }
        for f in ["execute_raw_qty_with_client_id", "execute_order", "set_leverage"] {
            let c = cuerpo(f);
            assert!(
                c.contains("check_rate_limits(") && !c.contains("check_exit_rate_limits("),
                "{f} debe respetar el kill-switch"
            );
        }
    }

    /// F1-005 (Ola Ω150): un 2xx concurrente durante una ventana de cooldown
    /// activa NO debe borrar `consecutive_429`. Solo cuando `now >= cooldown_until_ms`
    /// se restablece el contador a 0.
    #[test]
    fn f1_005_concurrent_2xx_during_cooldown_preserves_consecutive_429() {
        let ex = executor();
        ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=30");
        ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=30");
        assert_eq!(ex.consecutive_429.load(Ordering::Relaxed), 2);

        // Un 2xx concurrente llega mientras el cooldown de 30s sigue activo:
        ex.update_limits(&crate::client::BinanceRateLimits {
            weight_1m: Some(400),
            orders_10s: Some(5),
            orders_1m: Some(20),
        });
        assert_eq!(
            ex.consecutive_429.load(Ordering::Relaxed),
            2,
            "un 2xx concurrente durante cooldown activo no debe borrar consecutive_429"
        );

        // Un tercer 429 en la misma ráfaga activa RATE_BRAKE temporal:
        ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=30");
        assert_eq!(ex.consecutive_429.load(Ordering::Relaxed), 3);
        let now = current_synced_timestamp_ms(None);
        assert!(
            matches!(ex.check_rate_limits(now), Err(e) if e.starts_with("RATE_BRAKE")),
            "el 3er 429 debe activar RATE_BRAKE pese al 2xx concurrente intermedio"
        );

        // Una vez expirado el cooldown, un 2xx sano sí restablece consecutive_429 a 0:
        ex.cooldown_until_ms.store(0, Ordering::Relaxed);
        ex.update_limits(&crate::client::BinanceRateLimits {
            weight_1m: Some(100),
            orders_10s: Some(1),
            orders_1m: Some(5),
        });
        assert_eq!(ex.consecutive_429.load(Ordering::Relaxed), 0);
    }
}

#[cfg(test)]
mod tests_n20 {
    use super::*;
    use futures_util::FutureExt;

    const CODIGO: &str = include_str!("executor.rs");

    /// Cuerpo de la última `fn {f}(` del archivo (la del impl), como en tests_n5b.
    fn cuerpo(f: &str) -> &'static str {
        let Some(i) = CODIGO.rfind(&format!("fn {f}(")) else {
            return "";
        };
        let fin = ["\n    pub ", "\n    fn ", "\n    async fn "]
            .iter()
            .filter_map(|m| CODIGO[i + 3..].find(m).map(|k| i + 3 + k))
            .min()
            .unwrap_or(CODIGO.len());
        &CODIGO[i..fin]
    }

    fn ex() -> OrderExecutor {
        OrderExecutor::new("k".into(), "s".into(), true)
    }

    fn rechazo<T: std::fmt::Debug>(r: Option<Result<T, String>>, prefijo: &str) -> bool {
        matches!(r, Some(Err(ref e)) if e.starts_with(prefijo))
    }

    /// RED antes de CL-184: la descarga de exchangeInfo y su reintento no
    /// miraban el freno; con un 418 o dentro del Retry-After de un 429
    /// salían igual (el primer sondeo quedaba esperando a la red). El
    /// rechazo local no cuenta como 429: hacen falta tres reales.
    #[tokio::test]
    async fn cl184_exchange_info_no_sale_dentro_del_freno() {
        let ex = ex();
        ex.handle_rate_limit_error("HTTP_418_IP_BANNED retry_after=300");
        let r = ex.fetch_all_symbol_filters().now_or_never();
        assert!(rechazo(r, "RATE_BRAKE"), "con un 418 exchangeInfo no sale");

        let ex = self::ex();
        ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=30");
        let r = ex.fetch_all_symbol_filters().now_or_never();
        assert!(rechazo(r, "RATE_LIMIT_COOLDOWN"), "dentro del Retry-After no sale");
        ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=30");
        let ahora = ex.get_synced_timestamp();
        assert!(
            matches!(ex.check_backoff(ahora), Err(ref e) if e.starts_with("RATE_LIMIT_COOLDOWN")),
            "el rechazo local no contó como 429"
        );
        ex.handle_rate_limit_error("HTTP_429_RATE_LIMITED retry_after=30");
        assert!(matches!(ex.check_backoff(ahora), Err(ref e) if e.starts_with("RATE_BRAKE")));
    }

    /// RED antes de CL-184: ninguno de los dos envíos de exchangeInfo tenía
    /// freno. Sólo el freno, sin el 80 %: los cierres necesitan los filtros.
    #[test]
    fn cl184_los_dos_envios_de_exchange_info_frenan() {
        let c = cuerpo("fetch_all_symbol_filters");
        let envios: Vec<usize> = c.match_indices(".get_payload_with_timeout(").map(|(i, _)| i).collect();
        assert_eq!(envios.len(), 2, "descarga y reintento");
        assert!(c[..envios[0]].contains("check_backoff("), "freno antes de la descarga");
        assert!(c[envios[0]..envios[1]].contains("check_backoff("), "freno antes del reintento");
        assert!(!c.contains("check_exit_rate_limits("), "sin el tramo del 80 %");
    }

    /// RED antes de CL-185: una respuesta 2xx sin filtros utilizables (el
    /// cliente devuelve Ok con «TEXT_ERR…» si no pudo leer el cuerpo) era un
    /// mapa vacío, y get_symbol_filter lo guardaba encima de la caché buena.
    #[test]
    fn cl185_un_exchange_info_sin_filtros_no_es_un_mapa_vacio() {
        for texto in [
            "TEXT_ERR: operation timed out",
            "{}",
            r#"{"symbols":[]}"#,
            r#"{"symbols":[{"symbol":"CL185USDT","filters":[
                {"filterType":"LOT_SIZE","stepSize":"0"},
                {"filterType":"PRICE_FILTER","tickSize":"0.001"}]}]}"#,
        ] {
            let r = OrderExecutor::filtros_de_exchange_info(texto);
            assert!(
                matches!(r, Err(ref e) if e.starts_with("EXCHANGE_INFO_SIN_FILTROS")),
                "{texto} → {r:?}"
            );
        }
        let bueno = r#"{"symbols":[{"symbol":"CL185USDT","filters":[
            {"filterType":"LOT_SIZE","stepSize":"0.01"},
            {"filterType":"PRICE_FILTER","tickSize":"0.001"},
            {"filterType":"MIN_NOTIONAL","notional":"10"}]}]}"#;
        let mapa = OrderExecutor::filtros_de_exchange_info(bueno).unwrap_or_default();
        assert_eq!(mapa.len(), 1);
        let f = mapa.get("CL185USDT").copied().unwrap_or_default();
        assert_eq!((f.step_size, f.tick_size, f.min_notional), (0.01, 0.001, 10.0));
    }

    /// RED antes de CL-185: la descarga devolvía `Ok(map)` aunque el mapa
    /// estuviera vacío.
    #[test]
    fn cl185_la_descarga_no_devuelve_un_mapa_vacio() {
        let c = cuerpo("fetch_all_symbol_filters");
        assert!(c.contains("Self::filtros_de_exchange_info(&text)"));
        assert!(!c.contains("Ok(map)"));
    }
}
