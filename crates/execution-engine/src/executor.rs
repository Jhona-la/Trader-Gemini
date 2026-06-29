use crate::binance_api::{sign_payload, ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL, TIME_IN_FORCE_IOC};
use crate::ExecutionPayload;
use risk_engine::ValidatedOrder;
use signal_engine::SignalType;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct OrderExecutor {
    api_secret: String,
}

impl OrderExecutor {
    pub fn new(api_secret: String) -> Self {
        Self { api_secret }
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
        let signature = sign_payload(&query_string, &self.api_secret);

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
