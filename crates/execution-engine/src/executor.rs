use crate::binance_api::{sign_payload_to_buffer, ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT, SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_IOC, TIME_IN_FORCE_GTC};
use crate::ExecutionPayload;
use risk_engine::ValidatedOrder;
use signal_engine::SignalType;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};

use crate::client::{BinanceClient, ZeroAllocBuffer};

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

    async fn cancel_order(&self, symbol: &str, client_order_id: &str) -> Result<(), String>;

    async fn fetch_open_positions(&self) -> Result<Vec<String>, String>;

    fn trigger_kill_switch(&self);
}

pub struct OrderExecutor {
    api_secret: String,
    client: BinanceClient,
    rate_limit_counter: AtomicUsize,
    last_reset_timestamp: AtomicU64,
    binance_weight_1m: AtomicUsize,
    binance_orders_10s: AtomicUsize,
    binance_orders_1m: AtomicUsize,
    kill_switch: AtomicBool,
}

impl OrderExecutor {
    pub fn new(api_key: String, api_secret: String) -> Self {
        Self {
            api_secret,
            client: BinanceClient::new(api_key),
            rate_limit_counter: AtomicUsize::new(0),
            last_reset_timestamp: AtomicU64::new(0),
            binance_weight_1m: AtomicUsize::new(0),
            binance_orders_10s: AtomicUsize::new(0),
            binance_orders_1m: AtomicUsize::new(0),
            kill_switch: AtomicBool::new(false),
        }
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
    }

    #[inline(always)]
    fn check_rate_limits(&self, timestamp_ms: u64) -> Result<(), String> {
        if self.kill_switch.load(Ordering::Relaxed) {
            return Err("KILL SWITCH ACTIVE. Execution blocked.".to_string());
        }

        let bw1m = self.binance_weight_1m.load(Ordering::Relaxed);
        let bo10s = self.binance_orders_10s.load(Ordering::Relaxed);
        let bo1m = self.binance_orders_1m.load(Ordering::Relaxed);

        if bw1m > 2200 || bo10s > 280 || bo1m > 1100 {
            self.trigger_kill_switch();
            return Err("BINANCE GLOBAL RATE LIMIT APPROACHING. KILL SWITCH ACTIVATED.".to_string());
        }

        let current_sec = timestamp_ms / 1000;
        let last_reset = self.last_reset_timestamp.load(Ordering::Relaxed);
        
        if current_sec > last_reset {
            self.last_reset_timestamp.store(current_sec, Ordering::Relaxed);
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
    ) -> Option<ExecutionPayload> {
        if order.volume_usd <= 0.0 || current_price <= 0.0 {
            return None;
        }

        // Volumen real de la moneda
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

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Construir Query String (Formato URL Encoded para REST API)
        let query_string = format!(
            "symbol={}&side={}&type={}&quantity={}&timestamp={}",
            symbol, side, ORDER_TYPE_MARKET, final_quantity, timestamp
        );

        // Firmar
        let mut sig_buf = [0u8; 64];
        sign_payload_to_buffer(&query_string, &self.api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) }.to_string();

        Some(ExecutionPayload {
            symbol: symbol.to_string(),
            side: side.to_string(),
            quantity: final_quantity,
            order_type: ORDER_TYPE_MARKET.to_string(),
            time_in_force: TIME_IN_FORCE_IOC.to_string(),
            signature,
            timestamp,
        })
    }
}

impl ExecutionProvider for OrderExecutor {
    fn trigger_kill_switch(&self) {
        self.kill_switch.store(true, std::sync::atomic::Ordering::SeqCst);
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
        if let Some(payload) = self.build_payload(order, symbol, current_price, step_size) {
            let mut buf = ZeroAllocBuffer::new();
            buf.push_str("https://fapi.binance.com/fapi/v1/order?");
            buf.push_str("symbol=");
            buf.push_str(&payload.symbol);
            buf.push_str("&side=");
            buf.push_str(&payload.side);
            buf.push_str("&type=");
            buf.push_str(&payload.order_type);
            buf.push_str("&quantity=");
            buf.push_f64(payload.quantity);
            buf.push_str("&timestamp=");
            buf.push_u64(payload.timestamp);
            buf.push_str("&signature=");
            buf.push_str(&payload.signature);

            let res = self.client.execute_order_payload(buf.as_str()).await;
            if let Ok(limits) = &res {
                self.update_limits(limits);
            }
            res.map(|_| ())
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
            return Err("SEGURIDAD: quantity inválido (infinito, NaN o <= 0.0). Orden abortada.".to_string());
        }
        let final_quantity = Self::round_to_step_size(quantity, step_size);
        if final_quantity == 0.0 {
            return Err("Volumen 0 despues de round_to_step_size".to_string());
        }

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str("https://fapi.binance.com/fapi/v1/order?");
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        buf.push_str("&type=");
        buf.push_str(ORDER_TYPE_MARKET);
        buf.push_str("&quantity=");
        buf.push_f64(final_quantity);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        sign_payload_to_buffer(payload, &self.api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        // Despachar a Binance asíncronamente O(1)
        let res = self.client.execute_order_payload(buf.as_str()).await;
        if let Ok(limits) = &res {
            self.update_limits(limits);
        }
        res.map(|_| ())
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

        let side = if is_long { SIDE_BUY } else { SIDE_SELL };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str("https://fapi.binance.com/fapi/v1/order?");
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&side=");
        buf.push_str(side);
        buf.push_str("&type=");
        buf.push_str(ORDER_TYPE_LIMIT);
        buf.push_str("&timeInForce=");
        buf.push_str(TIME_IN_FORCE_GTC);
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
        sign_payload_to_buffer(payload, &self.api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.execute_order_payload(buf.as_str()).await;
        if let Ok(limits) = &res {
            self.update_limits(limits);
        }
        res.map(|_| ())
    }

    /// Cancela una orden activa usando el client_order_id
    #[inline(always)]
    async fn cancel_order(
        &self,
        symbol: &str,
        client_order_id: &str,
    ) -> Result<(), String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.check_rate_limits(timestamp)?;

        let mut buf = ZeroAllocBuffer::new();
        buf.push_str("https://fapi.binance.com/fapi/v1/order?");
        let payload_start = buf.as_str().len();

        buf.push_str("symbol=");
        buf.push_str(symbol);
        buf.push_str("&origClientOrderId=");
        buf.push_str(client_order_id);
        buf.push_str("&timestamp=");
        buf.push_u64(timestamp);

        let mut sig_buf = [0u8; 64];
        let payload = &buf.as_str()[payload_start..];
        sign_payload_to_buffer(payload, &self.api_secret, &mut sig_buf);
        let signature = unsafe { std::str::from_utf8_unchecked(&sig_buf) };
        buf.push_str("&signature=");
        buf.push_str(signature);

        let res = self.client.cancel_order_payload(buf.as_str()).await;
        if let Ok(limits) = &res {
            self.update_limits(limits);
        }
        res.map(|_| ())
    }

    async fn fetch_open_positions(&self) -> Result<Vec<String>, String> {
        Ok(vec![])
    }
}
