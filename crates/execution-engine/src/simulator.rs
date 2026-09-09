use crate::executor::{ActivePosition, ExecutionProvider};
use risk_engine::ValidatedOrder;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;
use tokio::time::sleep;

pub struct SimulatedExecutor {
    pub average_latency_ms: u64,
    pub simulated_capital: RwLock<f64>,
    pub open_positions: RwLock<HashMap<String, ActivePosition>>,
    /// D-04 — FILL MODEL: mid del último tick por símbolo (para distancia
    /// al mid en fills de limit/maker) y floor de slippage para market/IOC.
    pub last_mids: RwLock<HashMap<String, f64>>,
    pub base_slippage_bps: f64,
}

impl SimulatedExecutor {
    pub fn new(simulated_capital: f64) -> Self {
        Self {
            average_latency_ms: 3, // Binance FAPI Latency (3ms)
            simulated_capital: RwLock::new(simulated_capital),
            open_positions: RwLock::new(HashMap::new()),
            last_mids: RwLock::new(HashMap::new()),
            base_slippage_bps: 1.0, // floor conservador: 1 bps mínimo
        }
    }

    /// D-04 — Fill probability para limit/maker: decrece exponencialmente
    /// con la distancia agresiva al mid (0 bps = 95%, 10+ bps = <30%).
    fn fill_probability(&self, order_price: f64, symbol: &str) -> f64 {
        let mid = *self.last_mids.read().unwrap().get(symbol).unwrap_or(&0.0);
        if mid <= 0.0 || order_price <= 0.0 {
            return 0.5; // sin datos de mid: 50/50
        }
        let dist_bps = ((order_price - mid).abs() / mid) * 10_000.0;
        0.95 * (-dist_bps / 8.0).exp()
    }

    /// D-04 — Slippage para market/IOC: floor + impacto cuadrático por tamaño.
    fn market_slippage(&self, nominal_usd: f64) -> f64 {
        let impact = (nominal_usd / 1_000_000.0).powf(1.2) * 0.0005;
        (impact + self.base_slippage_bps / 10_000.0).clamp(0.0001, 0.01)
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
        if let Ok(c) = self.simulated_capital.read() {
            Ok(*c)
        } else {
            Ok(0.0)
        }
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
        // G-02 — INYECTAR MID: el fill probabilístico dependía de last_mids
        // que nadie escribía → fill_probability() devolvía 0.5 constante.
        // Ahora cada execute_order actualiza el mid del símbolo.
        if current_price.is_finite() && current_price > 0.0 {
            self.last_mids
                .write()
                .unwrap()
                .insert(symbol.to_string(), current_price);
        }
        let is_long = order.signal == signal_engine::SignalType::Long;
        let side = if is_long { "BUY" } else { "SELL" };
        let slippage = self.market_slippage(order.volume_usd);
        let exec_price = if is_long {
            current_price * (1.0 + slippage)
        } else {
            current_price * (1.0 - slippage)
        };
        if let Ok(mut pos_map) = self.open_positions.write() {
            let entry = pos_map
                .entry(symbol.to_string())
                .or_insert_with(|| ActivePosition {
                    symbol: symbol.to_string(),
                    qty: 0.0,
                    entry_price: exec_price,
                    is_long,
                });
            entry.qty += order.volume_usd / exec_price.max(1e-6);
            entry.is_long = is_long;
        }
        if let Ok(mut cap) = self.simulated_capital.write() {
            let fee = order.volume_usd * 0.0005; // 0.05% VIP0 taker fee
            *cap -= fee;
        }
        println!(
            "👻 [SHADOW MODE] Executed {} {} @ {:.4} (slip: {:.4}%, delay: {}ms)",
            side, symbol, exec_price, slippage * 100.0, self.average_latency_ms
        );
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
        if let Ok(mut pos_map) = self.open_positions.write() {
            let entry = pos_map
                .entry(symbol.to_string())
                .or_insert_with(|| ActivePosition {
                    symbol: symbol.to_string(),
                    qty: 0.0,
                    entry_price: 1.0,
                    is_long,
                });
            entry.qty += quantity;
            entry.is_long = is_long;
        }
        // Asumimos slippage microscópico en log
        println!(
            "👻 [SHADOW MODE] Executed RAW {} {:.4} {} (Network delay: {}ms)",
            side, quantity, symbol, self.average_latency_ms
        );
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
        // D-04: fill probabilístico basado en distancia al mid
        let fill_prob = self.fill_probability(price, symbol);
        let roll = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as f64)
            / 1_000_000_000.0;
        if roll > fill_prob {
            return Err(format!(
                "[SIM] Limit no llenado (fill_prob={:.2}, roll={:.2}) — como en live",
                fill_prob, roll
            ));
        }
        let side = if is_long { "BUY" } else { "SELL" };
        println!(
            "👻 [SHADOW MODE] Limit {} {:.4} {} @ {} (ID: {}, fill_prob={:.2})",
            side, quantity, symbol, price, client_order_id, fill_prob
        );
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
        println!(
            "👻 [SHADOW MODE] Maker-Chase {} {:.4} {} @ {} (ID: {})",
            side, quantity, symbol, price, client_order_id
        );
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
        println!(
            "👻 [SHADOW MODE] IOC {} {:.4} {} @ {} (ID: {})",
            side, quantity, symbol, price, client_order_id
        );
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
        println!(
            "👻 [SHADOW MODE] Trailing {} {:.4} {} Act: {} Callback: {}% (ID: {})",
            side,
            quantity,
            symbol,
            activation_price,
            callback_rate * 100.0,
            client_order_id
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
            "🎮 [SIMULATOR] OCO Order {} {} Qty: {} TP: {} SL: {}",
            symbol, side, quantity, take_profit_price, stop_loss_price
        );
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
        let mut pnl = 0.0;
        let mut fee = 0.0;

        if let Ok(mut pos_map) = self.open_positions.write() {
            if let Some(pos) = pos_map.remove(symbol) {
                let exit_price = if is_long_close { pos.entry_price * 1.002 } else { pos.entry_price * 0.998 };
                if exit_price.is_finite() && pos.entry_price.is_finite() && quantity.is_finite() {
                    pnl = if pos.is_long {
                        (exit_price - pos.entry_price) * quantity
                    } else {
                        (pos.entry_price - exit_price) * quantity
                    };
                    // FIX #699: Comisión en nocional USD (quantity * exit_price * 0.0004)
                    fee = quantity * exit_price * 0.0004;
                }
            }
        }
        if let Ok(mut cap) = self.simulated_capital.write() {
            if pnl.is_finite() && fee.is_finite() {
                *cap += pnl - fee;
            }
        }
        let side = if is_long_close { "SELL" } else { "BUY" };
        println!(
            "👻 [SHADOW MODE] Reduce-Only {} {:.4} {} (PnL: {:.4}, Fee: {:.4})",
            side, quantity, symbol, pnl, fee
        );
        Ok(())
    }

    #[inline(always)]
    async fn cancel_order(&self, symbol: &str, client_order_id: &str) -> Result<(), String> {
        self.simulate_network_delay().await;
        println!(
            "👻 [SHADOW MODE] Cancelled order {} on {}",
            client_order_id, symbol
        );
        Ok(())
    }

    #[inline(always)]
    async fn cancel_all_symbol_orders(&self, symbol: &str) -> Result<(), String> {
        self.simulate_network_delay().await;
        println!(
            "👻 [SHADOW MODE] Cancelled all orders on {}",
            symbol
        );
        Ok(())
    }

    #[inline(always)]
    async fn fetch_open_positions(&self) -> Result<Vec<crate::executor::ActivePosition>, String> {
        self.simulate_network_delay().await;
        if let Ok(pos_map) = self.open_positions.read() {
            Ok(pos_map.values().cloned().collect())
        } else {
            Ok(vec![])
        }
    }

    async fn fetch_server_time(&self) -> Result<i64, String> {
        Ok(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64)
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

#[cfg(test)]
mod tests {
    use super::*;
    use risk_engine::ValidatedOrder;
    use signal_engine::SignalType;

    #[tokio::test]
    async fn test_simulated_executor_open_and_reduce_only() {
        let sim = SimulatedExecutor::new(100.0);
        let balance = sim.fetch_account_balance().await.unwrap();
        assert_eq!(balance, 100.0);

        let order = ValidatedOrder {
            signal: SignalType::Long,
            volume_usd: 50.0,
            leverage: 1.0,
            maker_only: false,
            tp_target: 52000.0,
            sl_target: 49000.0,
            fee_buffer_multiplier: 1.01,
        };

        // Open Long position on BTCUSDT
        let res = sim.execute_order(&order, "BTCUSDT", 50000.0, 0.001).await;
        assert!(res.is_ok());

        let positions = sim.fetch_open_positions().await.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].symbol, "BTCUSDT");
        assert!(positions[0].is_long);
        assert!((positions[0].qty - 0.001).abs() < 1e-6);

        // Close position with reduce-only market
        let res_close = sim.execute_reduce_only_market("BTCUSDT", true, 0.001, 0.001).await;
        assert!(res_close.is_ok());

        let positions_after = sim.fetch_open_positions().await.unwrap();
        assert!(positions_after.is_empty());
    }

    #[tokio::test]
    async fn test_simulated_executor_methods_and_fees() {
        let sim = SimulatedExecutor::new(50.0);
        sim.trigger_kill_switch();

        assert!(sim.set_leverage("BTCUSDT", 20).await.is_ok());
        assert!(sim.fetch_exchange_info("BTCUSDT").await.is_ok());
        assert!(sim.execute_raw_qty("BTCUSDT", true, 0.002, 0.001).await.is_ok());
        assert!(sim.execute_limit_order("BTCUSDT", true, 0.001, 50000.0, 0.001, 0.1, "L1").await.is_ok());
        assert!(sim.execute_maker_chase("BTCUSDT", true, 0.001, 50000.0, 0.001, 0.1, "M1").await.is_ok());
        assert!(sim.execute_ioc_order("BTCUSDT", true, 0.001, 50000.0, 0.001, 0.1, "I1").await.is_ok());
        assert!(sim.execute_exchange_trailing_stop("BTCUSDT", true, 0.001, 50000.0, 1.0, 0.001, 0.1, "T1").await.is_ok());
        assert!(sim.execute_oco_order("BTCUSDT", true, 0.001, 52000.0, 48000.0, 0.001, 0.1, "O1").await.is_ok());
        assert!(sim.execute_iceberg_limit("BTCUSDT", true, 0.005, 0.001, 50000.0, 0.001, 0.1, "IC1").await.is_ok());
        assert!(sim.cancel_order("BTCUSDT", "L1").await.is_ok());

        let server_time = sim.fetch_server_time().await.unwrap();
        assert!(server_time > 0);

        let com = sim.fetch_commission_rate("BTCUSDT").await.unwrap();
        assert_eq!(com.0, 0.0002);
    }
}

