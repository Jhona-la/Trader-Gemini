use crate::executor::ExecutionProvider;
use crate::executor::OrderExecutor;
use arc_swap::ArcSwap;
use signal_engine::orchestrator::TensorDecision;
use signal_engine::SignalType;
use std::sync::Arc;

pub struct QuantumOrderRouter {
    executor: Arc<ArcSwap<OrderExecutor>>,
}

impl QuantumOrderRouter {
    pub fn new(executor: Arc<ArcSwap<OrderExecutor>>) -> Self {
        Self { executor }
    }

    /// Ruteo Cuántico: Decide dinámicamente si usar Limit, Market, IOC o Trailing Stop
    /// basado en la convicción matemática, la volatilidad y la latencia.
    pub async fn route_order(
        &self,
        symbol: &str,
        decision: &TensorDecision,
        current_price: f64,
        quantity: f64,
        step_size: f64,
        tick_size: f64,
        latency_ms: u64,
        spread_pct: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        // FIX #656: Validación previa de precio y cantidad
        if !current_price.is_finite()
            || current_price <= 0.0
            || !quantity.is_finite()
            || quantity <= 0.0
        {
            return Err(format!(
                "Invalid order price {} or quantity {} for symbol {}",
                current_price, quantity, symbol
            ));
        }

        let is_long = match decision.signal {
            SignalType::Long => true,
            SignalType::Short => false,
            SignalType::Flat => return Ok(()),
        };

        // Reglas de Ruteo Avanzadas (FASE 15)

        // 1. Condición de Pánico / Alta Latencia: Si la red es inestable (>500ms)
        // o el spread es enorme, USAR POST-ONLY O IOC para proteger el capital (Evita Slippage Mortal).
        if latency_ms > 500 || spread_pct > 0.005 {
            let offset_ticks = if is_long { -2.0 } else { 2.0 };
            let raw_limit_price = current_price + (offset_ticks * tick_size);
            // FIX #606 & #739: Cuantizar al múltiplo de tick_size y garantizar precio estrictamente positivo
            let min_price = tick_size.max(1e-8);
            let limit_price = if tick_size > 0.0 {
                ((raw_limit_price / tick_size).round() * tick_size).max(min_price)
            } else {
                raw_limit_price.max(min_price)
            };

            // Tratamos de inyectar liquidez sin cruzar el spread (Maker).
            println!("🛡️ [QUANTUM ROUTER] High Latency ({}ms) / Spread ({:.2}%). Routing to MAKER CHASE to avoid slippage.", latency_ms, spread_pct * 100.0);
            return self
                .executor
                .load()
                .execute_maker_chase(
                    symbol,
                    is_long,
                    quantity,
                    limit_price,
                    step_size,
                    tick_size,
                    client_order_id,
                )
                .await;
        }

        // 2. Extrema Convicción / Breakout (Volatilidad Esperada Alta)
        // FASE 3: umbrales leídos del GENOMA vía el arena del executor —
        // antes eran los literales 0.85/0.015. La confianza mínima usa el
        // gen min_confidence_btc; el gate de volatilidad se deriva de
        // scalp_sl_base: si el movimiento esperado supera la mitad del
        // stop, un limit arriesga quedarse fuera Y que el stop vuele.
        let (conf_gate, vol_gate) = {
            let exec = self.executor.load();
            match exec.arena.load_full() {
                Some(arena) => (
                    arena
                        .config
                        .min_confidence_btc
                        .load(std::sync::atomic::Ordering::Relaxed),
                    arena
                        .config
                        .scalp_sl_base
                        .load(std::sync::atomic::Ordering::Relaxed)
                        * 0.5,
                ),
                // Sin arena inyectada (tests/boot temprano): conserva el
                // umbral conservador histórico hasta que el engine lo provea.
                None => (0.85, 0.015),
            }
        };
        if decision.net_confidence > conf_gate && decision.expected_volatility > vol_gate {
            // El mercado se va a mover rapidísimo. Un Limit no se llenará.
            // Usar IOC (Immediate-Or-Cancel) a un precio ligeramente peor para garantizar la entrada
            // Tolerancia de slippage adaptativa a la volatilidad esperada (D-448)
            let dyn_slip = (decision.expected_volatility * 0.5).clamp(0.0008, 0.0035);
            let slippage_allowance = if is_long { dyn_slip } else { -dyn_slip };
            let raw_ioc_price = current_price * (1.0 + slippage_allowance);
            let min_price = tick_size.max(1e-8);
            let ioc_price = if tick_size > 0.0 {
                ((raw_ioc_price / tick_size).round() * tick_size).max(min_price)
            } else {
                raw_ioc_price.max(min_price)
            };

            println!("⚡ [QUANTUM ROUTER] Breakout Detected! Routing to IOC (Immediate-Or-Cancel) at {} max slippage (Price: {:.6}).", slippage_allowance, ioc_price);
            return self
                .executor
                .load()
                .execute_ioc_order(
                    symbol,
                    is_long,
                    quantity,
                    ioc_price,
                    step_size,
                    tick_size,
                    client_order_id,
                )
                .await;
        }

        // 3. Fallback a Market estándar pero filtrado por nuestro Risk Engine (execute_raw_qty)
        // Ya que la orden viene validada y la latencia es baja.
        println!("🌊 [QUANTUM ROUTER] Optimal Conditions. Routing to Standard Market Execution (ID: {}).", client_order_id);
        // FIX #630: Preservar trazabilidad pasando client_order_id a la ejecución Market
        self.executor
            .load()
            .execute_raw_qty_with_client_id(symbol, is_long, quantity, step_size, client_order_id)
            .await
    }

    /// Ruteo para asegurar ganancias (Take Profit dinámico de Exchange)
    pub async fn route_trailing_stop(
        &self,
        symbol: &str,
        is_long_position: bool,
        quantity: f64,
        activation_price: f64,
        callback_rate_pct: f64,
        step_size: f64,
        tick_size: f64,
        client_order_id: &str,
    ) -> Result<(), String> {
        // FIX #656 & #698: Validación previa y clamping estricto de callback rate [0.1, 5.0]
        if !activation_price.is_finite()
            || activation_price <= 0.0
            || !quantity.is_finite()
            || quantity <= 0.0
        {
            return Err(format!(
                "Invalid trailing stop activation price {} or quantity {} for symbol {}",
                activation_price, quantity, symbol
            ));
        }

        let safe_callback_rate = if callback_rate_pct.is_finite() {
            callback_rate_pct.clamp(0.1, 5.0)
        } else {
            1.0
        };

        // Enviar un Trailing Stop nativo de Binance (Se ejecuta del lado de sus servidores)
        // garantizando protección absoluta frente a caídas de internet local.
        println!(
            "🛡️ [QUANTUM ROUTER] Deploying Native Exchange Trailing Stop for {}.",
            symbol
        );
        self.executor
            .load()
            .execute_exchange_trailing_stop(
                symbol,
                is_long_position, // Passes position orientation: true for Long (generates SELL + positionSide=LONG)
                quantity,
                activation_price,
                safe_callback_rate,
                step_size,
                tick_size,
                client_order_id,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use signal_engine::orchestrator::TensorDecision;
    use signal_engine::{SignalType, TradeHorizon};

    #[tokio::test]
    async fn test_quantum_order_router_invalid_inputs_and_flat_decision() {
        let executor = Arc::new(ArcSwap::from_pointee(OrderExecutor::new(
            "dummy_key".to_string(),
            "dummy_secret".to_string(),
            true,
        )));
        let router = QuantumOrderRouter::new(executor);

        let flat_decision = TensorDecision {
            signal: SignalType::Flat,
            net_confidence: 0.0,
            expected_volatility: 0.0,
            expected_lifetime_ms: 1000,
            horizon: TradeHorizon::Scalp,
        };

        // Flat decision returns Ok(()) immediately without calling executor
        let res = router
            .route_order(
                "BTCUSDT",
                &flat_decision,
                50000.0,
                0.001,
                0.001,
                0.1,
                50,
                0.0001,
                "TEST_FLAT_ORDER",
            )
            .await;
        assert!(res.is_ok());

        let long_decision = TensorDecision {
            signal: SignalType::Long,
            net_confidence: 0.9,
            expected_volatility: 0.02,
            expected_lifetime_ms: 1000,
            horizon: TradeHorizon::Scalp,
        };

        // Invalid price (NaN) returns Err
        let res_nan_price = router
            .route_order(
                "BTCUSDT",
                &long_decision,
                f64::NAN,
                0.001,
                0.001,
                0.1,
                50,
                0.0001,
                "TEST_NAN_ORDER",
            )
            .await;
        assert!(res_nan_price.is_err());

        // Invalid quantity (0.0) returns Err
        let res_zero_qty = router
            .route_order(
                "BTCUSDT",
                &long_decision,
                50000.0,
                0.0,
                0.001,
                0.1,
                50,
                0.0001,
                "TEST_ZERO_QTY",
            )
            .await;
        assert!(res_zero_qty.is_err());
    }

    #[tokio::test]
    async fn test_quantum_order_router_trailing_stop_nan_validation() {
        let executor = Arc::new(ArcSwap::from_pointee(OrderExecutor::new(
            "dummy_key".to_string(),
            "dummy_secret".to_string(),
            true,
        )));
        let router = QuantumOrderRouter::new(executor);

        // Invalid activation price (NaN) returns Err
        let res_nan_act = router
            .route_trailing_stop(
                "BTCUSDT",
                true,
                0.001,
                f64::NAN,
                1.0,
                0.001,
                0.1,
                "TEST_TRAILING_NAN",
            )
            .await;
        assert!(res_nan_act.is_err());

        // Invalid quantity (negative) returns Err
        let res_neg_qty = router
            .route_trailing_stop(
                "BTCUSDT",
                true,
                -0.001,
                50000.0,
                1.0,
                0.001,
                0.1,
                "TEST_TRAILING_NEG",
            )
            .await;
        assert!(res_neg_qty.is_err());
    }
}
