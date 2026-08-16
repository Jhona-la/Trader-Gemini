use crate::executor::OrderExecutor;
use crate::executor::ExecutionProvider;
use signal_engine::orchestrator::TensorDecision;
use signal_engine::SignalType;
use std::sync::Arc;
use arc_swap::ArcSwap;

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
            let limit_price = current_price + (offset_ticks * tick_size);
            
            // Tratamos de inyectar liquidez sin cruzar el spread (Maker).
            println!("🛡️ [QUANTUM ROUTER] High Latency ({}ms) / Spread ({:.2}%). Routing to MAKER CHASE to avoid slippage.", latency_ms, spread_pct * 100.0);
            return self.executor.load().execute_maker_chase(
                symbol, is_long, quantity, limit_price, step_size, tick_size, client_order_id
            ).await;
        }

        // 2. Extrema Convicción / Breakout (Volatilidad Esperada Alta)
        if decision.net_confidence > 0.85 && decision.expected_volatility > 0.015 {
            // El mercado se va a mover rapidísimo. Un Limit no se llenará.
            // Usar IOC (Immediate-Or-Cancel) a un precio ligeramente peor para garantizar la entrada
            // pero con un techo (Slippage protection).
            let slippage_allowance = if is_long { 0.001 } else { -0.001 }; // 10 bps max slippage
            let ioc_price = current_price * (1.0 + slippage_allowance);
            
            println!("⚡ [QUANTUM ROUTER] Breakout Detected! Routing to IOC (Immediate-Or-Cancel) at {} max slippage.", slippage_allowance);
            return self.executor.load().execute_ioc_order(
                symbol, is_long, quantity, ioc_price, step_size, tick_size, client_order_id
            ).await;
        }

        // 3. Fallback a Market estándar pero filtrado por nuestro Risk Engine (execute_raw_qty)
        // Ya que la orden viene validada y la latencia es baja.
        println!("🌊 [QUANTUM ROUTER] Optimal Conditions. Routing to Standard Market Execution.");
        self.executor.load().execute_raw_qty(
            symbol, is_long, quantity, step_size
        ).await
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
        // Enviar un Trailing Stop nativo de Binance (Se ejecuta del lado de sus servidores)
        // garantizando protección absoluta frente a caídas de internet local.
        println!("🛡️ [QUANTUM ROUTER] Deploying Native Exchange Trailing Stop for {}.", symbol);
        self.executor.load().execute_exchange_trailing_stop(
            symbol,
            !is_long_position, // To close a long, we sell (is_long = false for the order)
            quantity,
            activation_price,
            callback_rate_pct,
            step_size,
            tick_size,
            client_order_id
        ).await
    }
}
