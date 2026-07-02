use crate::executor::ExecutionProvider;
use risk_engine::ValidatedOrder;

pub struct ShadowExecutor;

impl Default for ShadowExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ShadowExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl ExecutionProvider for ShadowExecutor {
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
        let is_long = order.signal == signal_engine::SignalType::Long;
        let side = if is_long { "BUY" } else { "SELL" };
        let mut raw_qty = order.volume_usd / current_price;
        if step_size > 0.0 {
            let mult = 1.0 / step_size;
            raw_qty = (raw_qty * mult).floor() / mult;
        }
        println!("👻 [SHADOW MODE] Executed {} {} {} @ {} [Simulated]", side, raw_qty, symbol, current_price);
        Ok(())
    }

    async fn execute_raw_qty(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        _step_size: f64,
    ) -> Result<(), String> {
        if quantity.is_infinite() || quantity.is_nan() || quantity <= 0.0 {
            return Err("SEGURIDAD: quantity inválido (infinito, NaN o <= 0.0). Orden abortada.".to_string());
        }
        let side = if is_long { "BUY" } else { "SELL" };
        println!("👻 [SHADOW MODE] Executed RAW {} {} {} [Simulated]", side, quantity, symbol);
        Ok(())
    }

    fn trigger_kill_switch(&self) {
        println!("👻 [SHADOW MODE] KILL SWITCH TRIGGERED");
    }

    async fn execute_limit_order(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        _step_size: f64,
        _tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        let side = if is_long { "BUY" } else { "SELL" };
        println!("👻 [SHADOW MODE] Limit {} {} {} @ {} (ID: {}) [Simulated]", side, quantity, symbol, price, client_order_id);
        Ok(())
    }

    async fn cancel_order(&self, symbol: &str, client_order_id: &str) -> Result<(), String> {
        println!("👻 [SHADOW MODE] Cancelled order {} on {} [Simulated]", client_order_id, symbol);
        Ok(())
    }

    async fn fetch_open_positions(&self) -> Result<Vec<String>, String> {
        println!("👻 [SHADOW MODE] Fetch open positions called [Simulated]");
        Ok(vec![])
    }
}
