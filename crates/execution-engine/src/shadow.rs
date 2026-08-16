use crate::executor::ExecutionProvider;
use risk_engine::ValidatedOrder;

pub struct ShadowExecutor {
    pub simulated_capital: f64,
}


impl ShadowExecutor {
    pub fn new(simulated_capital: f64) -> Self {
        Self { simulated_capital }
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

    async fn execute_maker_chase(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        _step_size: f64,
        _tick_size: f64,
        _client_order_id: &str,
    ) -> Result<(), String> {
        self.log_shadow_trade(symbol, "MAKER_CHASE", is_long, quantity, price);
        Ok(())
    }

    async fn execute_ioc_order(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        _step_size: f64,
        _tick_size: f64,
        _client_order_id: &str,
    ) -> Result<(), String> {
        self.log_shadow_trade(symbol, "LIMIT_IOC", is_long, quantity, price);
        Ok(())
    }

    async fn execute_exchange_trailing_stop(
        &self,
        symbol: &str,
        is_long: bool, 
        quantity: f64,
        activation_price: f64,
        callback_rate: f64, 
        _step_size: f64,
        _tick_size: f64,
        _client_order_id: &str,
    ) -> Result<(), String> {
        println!("[SHADOW] Trailing Stop {} {} Qty: {} Act: {} CB: {}%", 
            symbol, if is_long {"BUY"} else {"SELL"}, quantity, activation_price, callback_rate);
        Ok(())
    }

    async fn execute_oco_order(
        &self,
        symbol: &str,
        is_long_close: bool,
        quantity: f64,
        take_profit_price: f64,
        stop_loss_price: f64,
        _step_size: f64,
        _tick_size: f64,
        _base_client_id: &str,
    ) -> Result<(), String> {
        self.simulate_network_delay().await;
        let side = if is_long_close { "SELL" } else { "BUY" };
        println!("👻 [SHADOW MODE] OCO Order {} {} Qty: {} TP: {} SL: {}", symbol, side, quantity, take_profit_price, stop_loss_price);
        Ok(())
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

    async fn fetch_open_positions(&self) -> Result<Vec<crate::executor::ActivePosition>, String> {
        println!("👻 [SHADOW MODE] Fetch open positions called [Simulated]");
        Ok(vec![])
    }

    async fn fetch_account_balance(&self) -> Result<f64, String> {
        println!("👻 [SHADOW MODE] Fetch account balance called [Simulated]");
        Ok(self.simulated_capital)
    }

    async fn set_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String> {
        println!("👻 [SHADOW MODE] Set leverage to {} for {} [Simulated]", leverage, symbol);
        Ok(())
    }
}
