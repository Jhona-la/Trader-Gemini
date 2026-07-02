use crate::executor::ExecutionProvider;
use risk_engine::ValidatedOrder;
use std::time::Duration;
use tokio::time::sleep;

pub struct SimulatedExecutor {
    pub average_latency_ms: u64,
}

impl Default for SimulatedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulatedExecutor {
    pub fn new() -> Self {
        Self {
            average_latency_ms: 3, // Binance FAPI Latency (3ms)
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
    async fn cancel_order(&self, symbol: &str, client_order_id: &str) -> Result<(), String> {
        self.simulate_network_delay().await;
        println!("👻 [SHADOW MODE] Cancelled order {} on {}", client_order_id, symbol);
        Ok(())
    }

    #[inline(always)]
    async fn fetch_open_positions(&self) -> Result<Vec<String>, String> {
        self.simulate_network_delay().await;
        Ok(vec![])
    }
}
