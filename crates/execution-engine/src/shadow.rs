use crate::executor::ExecutionProvider;
use risk_engine::ValidatedOrder;

pub struct ShadowExecutor {
    pub simulated_capital: f64,
    active_leverage: std::sync::RwLock<std::collections::HashMap<String, u32>>,
}

impl ShadowExecutor {
    pub fn new(simulated_capital: f64) -> Self {
        Self {
            simulated_capital,
            active_leverage: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    fn log_shadow_trade(
        &self,
        symbol: &str,
        order_type: &str,
        is_long: bool,
        qty: f64,
        price: f64,
    ) {
        println!(
            "👻 [SHADOW MODE] {} {} {} {} @ {} [Simulated]",
            order_type,
            if is_long { "BUY" } else { "SELL" },
            qty,
            symbol,
            price
        );
    }

    async fn simulate_network_delay(&self) {
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
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
        let effective_notional = order.volume_usd * order.leverage.max(1.0);
        let mut raw_qty = effective_notional / current_price;
        if step_size > 0.0 {
            let mult = 1.0 / step_size;
            raw_qty = (raw_qty * mult + 1e-9).floor() / mult;
        }
        println!(
            "👻 [SHADOW MODE] Executed {} {} {} @ {} [Simulated]",
            side, raw_qty, symbol, current_price
        );
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
            return Err(
                "SEGURIDAD: quantity inválido (infinito, NaN o <= 0.0). Orden abortada."
                    .to_string(),
            );
        }
        let side = if is_long { "BUY" } else { "SELL" };
        println!(
            "👻 [SHADOW MODE] Executed RAW {} {} {} [Simulated]",
            side, quantity, symbol
        );
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
        println!(
            "[SHADOW] Trailing Stop {} {} Qty: {} Act: {} CB: {}%",
            symbol,
            if is_long { "BUY" } else { "SELL" },
            quantity,
            activation_price,
            callback_rate
        );
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
        println!(
            "👻 [SHADOW MODE] OCO Order {} {} Qty: {} TP: {} SL: {}",
            symbol, side, quantity, take_profit_price, stop_loss_price
        );
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
        println!(
            "👻 [SHADOW MODE] Limit {} {} {} @ {} (ID: {}) [Simulated]",
            side, quantity, symbol, price, client_order_id
        );
        Ok(())
    }

    async fn cancel_order(&self, symbol: &str, client_order_id: &str) -> Result<(), String> {
        println!(
            "👻 [SHADOW MODE] Cancelled order {} on {} [Simulated]",
            client_order_id, symbol
        );
        Ok(())
    }

    async fn cancel_all_symbol_orders(&self, symbol: &str) -> Result<(), String> {
        println!(
            "👻 [SHADOW MODE] Cancelled all orders on {} [Simulated]",
            symbol
        );
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
        let mut map = self.active_leverage.write().unwrap();
        map.insert(symbol.to_string(), leverage);
        println!(
            "👻 [SHADOW MODE] Set leverage to {} for {} [Simulated]",
            leverage, symbol
        );
        Ok(())
    }

    async fn execute_iceberg_limit(
        &self,
        symbol: &str,
        is_long: bool,
        quantity: f64,
        _iceberg_qty: f64,
        price: f64,
        _step_size: f64,
        _tick_size: f64,
        _client_order_id: &str,
    ) -> Result<(), String> {
        self.log_shadow_trade(symbol, "ICEBERG_LIMIT", is_long, quantity, price);
        Ok(())
    }

    async fn execute_reduce_only_market(
        &self,
        symbol: &str,
        is_long_close: bool,
        quantity: f64,
        _step_size: f64,
    ) -> Result<(), String> {
        let side = if is_long_close { "SELL" } else { "BUY" };
        println!(
            "👻 [SHADOW MODE] Reduce-Only Market {} {} {}",
            side, quantity, symbol
        );
        Ok(())
    }

    async fn fetch_server_time(&self) -> Result<i64, String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Ok(now)
    }

    async fn fetch_commission_rate(&self, _symbol: &str) -> Result<(f64, f64), String> {
        Ok((0.0002, 0.0005))
    }

    async fn fetch_exchange_info(&self, _symbol: &str) -> Result<f64, String> {
        Ok(5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use risk_engine::ValidatedOrder;
    use signal_engine::SignalType;

    #[tokio::test]
    async fn test_shadow_executor_operations_and_nan_immunity() {
        let shadow = ShadowExecutor::new(13.0);
        let balance = shadow.fetch_account_balance().await.unwrap();
        assert_eq!(balance, 13.0);

        let order = ValidatedOrder {
            signal: SignalType::Long,
            volume_usd: 5.0,
            leverage: 5.0,
            maker_only: false,
            tp_target: 52000.0,
            sl_target: 49000.0,

            fee_buffer_multiplier: 1.01,
        };

        // Valid execution
        let res_valid = shadow
            .execute_order(&order, "BTCUSDT", 50000.0, 0.001)
            .await;
        assert!(res_valid.is_ok());

        // Invalid current price (<= 0.0) returns Err
        let res_zero_price = shadow.execute_order(&order, "BTCUSDT", 0.0, 0.001).await;
        assert!(res_zero_price.is_err());

        // Raw qty NaN returns Err
        let res_nan_qty = shadow
            .execute_raw_qty("BTCUSDT", true, f64::NAN, 0.001)
            .await;
        assert!(res_nan_qty.is_err());

        // Raw qty valid returns Ok
        let res_raw_ok = shadow.execute_raw_qty("BTCUSDT", true, 0.001, 0.001).await;
        assert!(res_raw_ok.is_ok());

        // Maker chase, IOC, Trailing stop, OCO, Iceberg
        assert!(shadow
            .execute_maker_chase("BTCUSDT", true, 0.001, 50000.0, 0.001, 0.1, "ID1")
            .await
            .is_ok());
        assert!(shadow
            .execute_ioc_order("BTCUSDT", true, 0.001, 50000.0, 0.001, 0.1, "ID2")
            .await
            .is_ok());
        assert!(shadow
            .execute_exchange_trailing_stop("BTCUSDT", true, 0.001, 50000.0, 1.0, 0.001, 0.1, "ID3")
            .await
            .is_ok());
        assert!(shadow
            .execute_oco_order("BTCUSDT", true, 0.001, 51000.0, 49000.0, 0.001, 0.1, "ID4")
            .await
            .is_ok());
        assert!(shadow
            .execute_iceberg_limit("BTCUSDT", true, 0.005, 0.001, 50000.0, 0.001, 0.1, "ID5")
            .await
            .is_ok());
        assert!(shadow
            .execute_reduce_only_market("BTCUSDT", true, 0.001, 0.001)
            .await
            .is_ok());
        assert!(shadow.cancel_order("BTCUSDT", "ID1").await.is_ok());
    }

    #[tokio::test]
    async fn test_shadow_executor_leverage_and_balance() {
        let shadow = ShadowExecutor::new(100.0);
        assert!(shadow.set_leverage("ETHUSDT", 10).await.is_ok());
        {
            let map = shadow.active_leverage.read().unwrap();
            assert_eq!(*map.get("ETHUSDT").unwrap(), 10);
        }

        let time = shadow.fetch_server_time().await.unwrap();
        assert!(time > 0);

        let rates = shadow.fetch_commission_rate("ETHUSDT").await.unwrap();
        assert_eq!(rates.0, 0.0002);
        assert_eq!(rates.1, 0.0005);
    }
}
