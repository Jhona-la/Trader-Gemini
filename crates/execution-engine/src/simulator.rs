use crate::executor::ExecutionProvider;
use risk_engine::ValidatedOrder;
use std::time::Duration;
use tokio::time::sleep;

pub struct SimulatedExecutor {
    pub average_latency_ms: u64,
    pub simulated_capital: f64,
}


impl SimulatedExecutor {
    pub fn new(simulated_capital: f64) -> Self {
        Self {
            average_latency_ms: 3, // Binance FAPI Latency (3ms)
            simulated_capital,
        }
    }
    
    async fn simulate_network_delay(&self) {
        // En una máquina de 16GB, tokio::time::sleep puede no ser ultra-preciso
        // pero es suficiente para forzar el path asíncrono y simular red.
        sleep(Duration::from_millis(self.average_latency_ms)).await;
    }
}

impl ExecutionProvider for SimulatedExecutor {
    fn trigger_kill_switch(&self) {
        println!("👻 [SHADOW MODE] KILL SWITCH TRIGGERED");
    }

    async fn fetch_account_balance(&self) -> Result<f64, String> {
        Ok(self.simulated_capital)
    }

    async fn set_leverage(&self, _symbol: &str, _leverage: u32) -> Result<(), String> {
        // Do nothing in simulator
        Ok(())
    }

    async fn fetch_exchange_info(&self, _symbol: &str) -> Result<f64, String> {
        Ok(5.0) // Mock value for simulator
    }

    #[inline(always)]
    async fn execute_order(
        &self,
        order: &ValidatedOrder,
        symbol: &str,
        current_price: f64,
        _step_size: f64,
    ) -> Result<(), String> {
        self.simulate_network_delay().await;
        let is_long = order.signal == signal_engine::SignalType::Long;
        let side = if is_long { "BUY" } else { "SELL" };
        println!("👻 [SHADOW MODE] Executed {} {} @ {} (Network delay: {}ms)", side, symbol, current_price, self.average_latency_ms);
        Ok(())
    }

    #[inline(always)]
    async fn execute_raw_qty(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        _step_size: f64,
    ) -> Result<(), String> {
        self.simulate_network_delay().await;
        let side = if is_long { "BUY" } else { "SELL" };
        // Asumimos slippage microscópico en log
        println!("👻 [SHADOW MODE] Executed RAW {} {:.4} {} (Network delay: {}ms)", side, quantity, symbol, self.average_latency_ms);
        Ok(())
    }

    #[inline(always)]
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
        self.simulate_network_delay().await;
        let side = if is_long { "BUY" } else { "SELL" };
        println!("👻 [SHADOW MODE] Limit {} {:.4} {} @ {} (ID: {})", side, quantity, symbol, price, client_order_id);
        Ok(())
    }

    #[inline(always)]
    async fn execute_maker_chase(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        price: f64,
        _step_size: f64,
        _tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        self.simulate_network_delay().await;
        let side = if is_long { "BUY" } else { "SELL" };
        println!("👻 [SHADOW MODE] Maker-Chase {} {:.4} {} @ {} (ID: {})", side, quantity, symbol, price, client_order_id);
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
        client_order_id: &str,
    ) -> Result<(), String> {
        self.simulate_network_delay().await;
        let side = if is_long { "BUY" } else { "SELL" };
        println!("👻 [SHADOW MODE] IOC {} {:.4} {} @ {} (ID: {})", side, quantity, symbol, price, client_order_id);
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
        client_order_id: &str,
    ) -> Result<(), String> {
        self.simulate_network_delay().await;
        let side = if is_long { "BUY" } else { "SELL" };
        println!("👻 [SHADOW MODE] Trailing {} {:.4} {} Act: {} Callback: {}% (ID: {})", side, quantity, symbol, activation_price, callback_rate * 100.0, client_order_id);
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
        let side = if is_long_close { "SELL" } else { "BUY" };
        println!("🎮 [SIMULATOR] OCO Order {} {} Qty: {} TP: {} SL: {}", symbol, side, quantity, take_profit_price, stop_loss_price);
        Ok(())
    }

    #[inline(always)]
    async fn execute_reduce_only_market(
        &self,
        symbol: &str,
        is_long_close: bool,
        quantity: f64,
        _step_size: f64,
    ) -> Result<(), String> {
        self.simulate_network_delay().await;
        let side = if is_long_close { "SELL" } else { "BUY" };
        println!("👻 [SHADOW MODE] Reduce-Only {} {:.4} {}", side, quantity, symbol);
        Ok(())
    }

    #[inline(always)]
    async fn cancel_order(&self, symbol: &str, client_order_id: &str) -> Result<(), String> {
        self.simulate_network_delay().await;
        println!("👻 [SHADOW MODE] Cancelled order {} on {}", client_order_id, symbol);
        Ok(())
    }

    #[inline(always)]
    async fn fetch_open_positions(&self) -> Result<Vec<crate::executor::ActivePosition>, String> {
        self.simulate_network_delay().await;
        Ok(vec![])
    }
    
    async fn fetch_server_time(&self) -> Result<i64, String> {
        Ok(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64)
    }

    async fn fetch_commission_rate(&self, _symbol: &str) -> Result<(f64, f64), String> {
        self.simulate_network_delay().await;
        Ok((0.0002, 0.0005)) // Default VIP 0
    }

    async fn execute_iceberg_limit(
        &self,
        symbol: &str,
        is_long: bool,
        _quantity: f64,
        _iceberg_qty: f64,
        _price: f64,
        _step_size: f64,
        _tick_size: f64,
        _client_order_id: &str,
    ) -> Result<(), String> {
        self.simulate_network_delay().await;
        let side = if is_long { "BUY" } else { "SELL" };
        println!("👻 [SHADOW MODE] ICEBERG {} on {}", side, symbol);
        Ok(())
    }
}
