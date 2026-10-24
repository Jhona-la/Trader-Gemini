use crate::executor::ExecutionProvider;
use crate::executor::OrderExecutor;
use arc_swap::ArcSwap;
use signal_engine::orchestrator::TensorDecision;
use signal_engine::SignalType;
use std::sync::Arc;

/// Métricas de posición en cola FIFO y selección adversa (Fase 3, Ola Ω210)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FifoQueueMetrics {
    /// Profundidad nocional estimada por delante en la cola al mismo precio nivel L2 [USD]
    pub queue_ahead_usd: f64,
    /// Probabilidad de llenado pasivo sin selección adversa (P_clean_fill ∈ [0.0, 1.0])
    pub clean_fill_probability: f64,
    /// Costo esperado de selección adversa (deslizamiento post-fill por toxicidad de flujo agresor) [bps]
    pub adverse_selection_bps: f64,
    /// Fricción combinada total (comisión maker/taker + funding + medio spread + selección adversa) [fracción]
    pub total_friction: f64,
    /// Ganancia bruta esperada de la señal [fracción]
    pub gross_expected_edge: f64,
    /// Esperanza matemática neta: EV_gross - total_friction
    pub net_expected_edge: f64,
    /// Ratio de viabilidad matemática: EV_gross / total_friction (debe ser >= 3.0)
    pub viability_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderViability {
    /// Viable: EV_gross >= 3.0 * total_friction
    Viable,
    /// No viable: la fricción de comisiones, funding y selección adversa supera el tercio del edge
    InsufficientEdge,
}

pub struct AdverseSelectionGuard;

impl AdverseSelectionGuard {
    /// Evalúa la posición en cola FIFO y el costo de selección adversa para una orden.
    ///
    /// Condición estricta de viabilidad financiera (Axioma de Viabilidad):
    ///   E[ΔW_bruto] >= 3.0 * (fees + funding + spread + adverse_selection)
    pub fn evaluate_viability(
        expected_edge_fraction: f64,
        spread_pct: f64,
        expected_volatility: f64,
        confidence: f64,
        maker_fee_pct: f64,
        funding_rate_pct: f64,
        queue_depth_ahead_usd: f64,
        passive_depth_level_usd: f64,
    ) -> (OrderViability, FifoQueueMetrics) {
        let safe_spread = if spread_pct.is_finite() && spread_pct >= 0.0 { spread_pct } else { 0.0002 };
        let safe_vol = if expected_volatility.is_finite() && expected_volatility >= 0.0 { expected_volatility } else { 0.001 };
        let safe_conf = if confidence.is_finite() { confidence.clamp(0.0, 1.0) } else { 0.5 };
        let safe_maker_fee = if maker_fee_pct.is_finite() && maker_fee_pct >= 0.0 { maker_fee_pct } else { 0.0002 }; // 2 bps default
        let safe_funding = if funding_rate_pct.is_finite() { funding_rate_pct.abs() } else { 0.0001 }; // 1 bp default

        // 1. Toxicidad del flujo y probabilidad de llenado limpio (Glosten-Milgrom / VPIN proxy)
        let vpin_proxy = ((safe_vol * 10.0) / (safe_conf + 0.1)).clamp(0.05, 0.95);
        let q_ahead = if queue_depth_ahead_usd.is_finite() && queue_depth_ahead_usd >= 0.0 { queue_depth_ahead_usd } else { 1000.0 };
        let p_level = if passive_depth_level_usd.is_finite() && passive_depth_level_usd > 0.0 { passive_depth_level_usd } else { 5000.0 };
        let queue_ratio = (q_ahead / p_level).clamp(0.1, 10.0);

        let clean_fill_prob = (1.0 / (1.0 + vpin_proxy * queue_ratio)).clamp(0.01, 0.99);

        // 2. Costo de selección adversa: penetración del nivel por flujo agresor tóxico
        let adverse_cost_fraction = safe_vol * (1.0 - clean_fill_prob) * 0.5;
        let adverse_selection_bps = adverse_cost_fraction * 10_000.0;

        // 3. Fricciones combinadas totales
        let half_spread = safe_spread * 0.5;
        let total_friction = (safe_maker_fee + safe_funding + half_spread + adverse_cost_fraction).max(1e-6);

        // 4. Edge bruto esperado
        let gross_edge = if expected_edge_fraction.is_finite() && expected_edge_fraction > 0.0 {
            expected_edge_fraction
        } else {
            safe_conf * safe_vol * 1.5
        };

        let net_edge = gross_edge - total_friction;
        let viability_ratio = gross_edge / total_friction;

        // Viabilidad estricta: EV_gross >= 3.0 * total_friction
        let status = if viability_ratio >= 3.0 && net_edge > 0.0 {
            OrderViability::Viable
        } else {
            OrderViability::InsufficientEdge
        };

        (
            status,
            FifoQueueMetrics {
                queue_ahead_usd: q_ahead,
                clean_fill_probability: clean_fill_prob,
                adverse_selection_bps,
                total_friction,
                gross_expected_edge: gross_edge,
                net_expected_edge: net_edge,
                viability_ratio,
            },
        )
    }
}

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

        // Reglas de Ruteo Avanzadas (FASE 15, Ola Ω106)
        // Se resuelven umbrales dinámicos de latencia, spread y volatilidad continua desde la arena.
        let (panic_latency_ms, panic_spread_pct, conf_gate, vol_gate) = {
            let exec = self.executor.load();
            match exec.arena.load_full() {
                Some(arena) => {
                    let tau_ms = if decision.expected_lifetime_ms > 0 {
                        decision.expected_lifetime_ms as f64
                    } else {
                        quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS
                    };
                    let panic_lat = arena
                        .config
                        .latency_ms_panic_threshold
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let panic_lat = if panic_lat.is_finite() && panic_lat > 0.0 {
                        panic_lat
                    } else {
                        500.0
                    };
                    let max_spread = arena
                        .registry
                        .get_value_or("router_max_spread_pct", 0.005);
                    let max_spread = if max_spread.is_finite() && max_spread > 0.0 {
                        max_spread
                    } else {
                        0.005
                    };
                    (
                        panic_lat,
                        max_spread,
                        arena
                            .config
                            .min_confidence_btc
                            .load(std::sync::atomic::Ordering::Relaxed),
                        arena.config.sl_at_tau(tau_ms) * 0.5,
                    )
                }
                None => (500.0, 0.005, 0.85, 0.015),
            }
        };

        // 1. Condición de Pánico / Alta Latencia: Si la red es inestable (>panic_latency_ms)
        // o el spread es enorme, ir directo a MARKET — la velocidad de
        // aterrizaje es la protección.
        // Ola Ω106: Se preserva client_order_id (FIX #630) para garantizar idempotencia en ruteo de emergencia.
        if (latency_ms as f64) > panic_latency_ms || spread_pct > panic_spread_pct {
            println!(
                "🛡️ [QUANTUM ROUTER] High Latency ({}ms vs {:.0}ms) / Spread ({:.2}% vs {:.2}%). MARKET directo (ID: {}).",
                latency_ms,
                panic_latency_ms,
                spread_pct * 100.0,
                panic_spread_pct * 100.0,
                client_order_id
            );
            return self
                .executor
                .load()
                .execute_raw_qty_with_client_id(symbol, is_long, quantity, step_size, client_order_id)
                .await;
        }

        // 2. Extrema Convicción / Breakout (Volatilidad Esperada Alta)
        // Continuo Espectral: umbrales leídos del GENOMA y evaluados en la escala
        // temporal continua del horizonte tau de la orden vía sl_at_tau — no
        // con scalp_sl_base discreto. Si la volatilidad esperada supera la
        // mitad del stop continuo a esa tau, un limit arriesga quedarse fuera.
        if decision.net_confidence > conf_gate && decision.expected_volatility > vol_gate {
            // El mercado se va a mover rapidísimo. Un Limit no se llenará.
            // Usar IOC (Immediate-Or-Cancel) a un precio ligeramente peor para garantizar la entrada
            // Tolerancia de slippage adaptativa a la volatilidad esperada (D-448, Ola Ω106)
            // Se desacopla el techo estático de 35 bps expandiéndolo según el presupuesto de stop continuo:
            let max_slip_cap = vol_gate.clamp(0.0035, 0.0150);
            let dyn_slip = (decision.expected_volatility * 0.5).clamp(0.0008, max_slip_cap);
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

        // 3. Verificación de Viabilidad Financiera y Selección Adversa (Axioma: E[ΔW] >= 3x fricciones)
        let (live_maker_fee, live_funding_cost) = {
            let exec = self.executor.load();
            match exec.arena.load_full() {
                Some(arena) => {
                    let coin_id_opt = quantum_arena::symbols::get_coin_id(symbol);
                    let maker = coin_id_opt
                        .and_then(|id| arena.config.fee_rates_for(id))
                        .map(|(m, _t)| m)
                        .unwrap_or(0.0002);
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    let tau_ms = if decision.expected_lifetime_ms > 0 {
                        decision.expected_lifetime_ms as f64
                    } else {
                        quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS
                    };
                    let funding = coin_id_opt
                        .and_then(|id| arena.coins.get(id))
                        .map(|coin| coin.expected_funding_cost(is_long, tau_ms, now_ms))
                        .unwrap_or(0.0001);
                    (maker, funding)
                }
                None => (0.0002, 0.0001),
            }
        };

        let (viability, metrics) = AdverseSelectionGuard::evaluate_viability(
            decision.net_confidence * decision.expected_volatility * 2.0,
            spread_pct,
            decision.expected_volatility,
            decision.net_confidence,
            live_maker_fee,
            live_funding_cost,
            quantity * current_price * 0.5,
            quantity * current_price * 2.0,
        );

        if viability == OrderViability::InsufficientEdge {
            println!(
                "⚠️ [QUANTUM ROUTER] Rejection: Edge insuficiente para vencer fricciones y selección adversa. Viability: {:.2}x (< 3.0x), Friction: {:.2} bps (ID: {}).",
                metrics.viability_ratio,
                metrics.total_friction * 10_000.0,
                client_order_id
            );
            return Err(format!(
                "REJ_INSUFFICIENT_VIABILITY: EV_gross ({:.2} bps) < 3x friction ({:.2} bps). Viability ratio: {:.2}",
                metrics.gross_expected_edge * 10_000.0,
                metrics.total_friction * 10_000.0,
                metrics.viability_ratio
            ));
        }

        // 4. Fallback a Market estándar pero filtrado por nuestro Risk Engine (execute_raw_qty)
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
            horizon: TradeHorizon::Continuous,
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
            horizon: TradeHorizon::Continuous,
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

    #[test]
    fn test_adverse_selection_guard_viability_and_queue_modeling() {
        // Caso 1: Alto edge, baja fricción -> Viable
        let (status, metrics) = AdverseSelectionGuard::evaluate_viability(
            0.015,  // 150 bps edge
            0.0001, // 1 bp spread
            0.002,  // 20 bps vol
            0.90,   // 90% conf
            0.0002, // 2 bps fee
            0.0001, // 1 bp funding
            500.0,
            10000.0,
        );
        assert_eq!(status, OrderViability::Viable);
        assert!(metrics.viability_ratio >= 3.0);
        assert!(metrics.net_expected_edge > 0.0);
        assert!(metrics.clean_fill_probability > 0.50);

        // Caso 2: Micro-edge devorado por fees y toxicidad -> InsufficientEdge
        let (status_insuf, metrics_insuf) = AdverseSelectionGuard::evaluate_viability(
            0.0003, // 3 bps edge bruto
            0.0005, // 5 bps spread (2.5 bps half spread)
            0.010,  // 100 bps vol
            0.30,   // 30% conf
            0.0004, // 4 bps fee
            0.0002, // 2 bps funding
            20000.0,
            1000.0,
        );
        assert_eq!(status_insuf, OrderViability::InsufficientEdge);
        assert!(metrics_insuf.viability_ratio < 3.0);
    }
}
