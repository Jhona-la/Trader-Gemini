use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineMode {
    Optimistic,    // Solo Fees fijos, sin impacto de latencia ni de libro de órdenes. (Ideal para IA training inicial)
    HyperRealistic, // Fricción exponencial basada en nominal size y demoras estocásticas.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealityPhysics {
    pub mode: EngineMode,
    pub base_maker_fee: f64,
    pub base_taker_fee: f64,
    pub latency_penalty_ms: u64, // Simula latencia de RTT a Binance Tokyo/AWS AP-Northeast.
}

impl Default for RealityPhysics {
    fn default() -> Self {
        Self {
            mode: EngineMode::HyperRealistic,
            base_maker_fee: 0.0002, // 0.02% (Binance VIP 0 Maker)
            base_taker_fee: 0.0005, // 0.05% (Binance VIP 0 Taker)
            latency_penalty_ms: 15, // 15ms Round-Trip-Time (muy agresivo).
        }
    }
}

impl RealityPhysics {
    pub fn new(mode: EngineMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Calcula el costo real de entrar a mercado cruzando el spread (Taker)
    /// Devuelve: (precio_ejecutado, fee_total_usd)
    pub fn calculate_market_entry(
        &self,
        base_price: f64,
        is_long: bool,
        nominal_usd_size: f64,
        tick_volatility: f64,
    ) -> (f64, f64) {
        if self.mode == EngineMode::Optimistic {
            let fee = nominal_usd_size * self.base_taker_fee;
            return (base_price, fee);
        }

        // --- HYPER REALISTIC PHYSICS ---
        // 1. Orderbook impact: Asumimos que 1 Millón de dólares mueve el precio 0.05% en activos ultra-líquidos.
        // Pero el impacto es cuadrático para castigar tamaños absurdos (ej: si mete 10M, el impacto no es 10x, sino 100x).
        let impact_multiplier = (nominal_usd_size / 1_000_000.0).powf(1.2); 
        let slippage_impact_pct = impact_multiplier * 0.0005;

        // 2. Latency slippage: Durante 15ms el precio pudo haberse movido a nuestro favor o en contra. 
        // Asumiremos el peor caso (movimiento adverso igual a la volatilidad del tick * 10%).
        let latency_slippage = tick_volatility * 0.10; 

        let total_slippage_pct = slippage_impact_pct + latency_slippage;

        let executed_price = if is_long {
            base_price * (1.0 + total_slippage_pct) // Compramos más caro
        } else {
            base_price * (1.0 - total_slippage_pct) // Vendemos más barato
        };

        let fee_usd = nominal_usd_size * self.base_taker_fee;

        (executed_price, fee_usd)
    }

    /// Calcula el costo real de salir (Taker o Maker según trailing stop)
    pub fn calculate_exit(
        &self,
        base_price: f64,
        is_long: bool,
        nominal_usd_size: f64,
        is_maker: bool,
        tick_volatility: f64,
    ) -> (f64, f64) {
        if self.mode == EngineMode::Optimistic {
            let fee_rate = if is_maker { self.base_maker_fee } else { self.base_taker_fee };
            return (base_price, nominal_usd_size * fee_rate);
        }

        let fee_rate = if is_maker { self.base_maker_fee } else { self.base_taker_fee };
        let fee_usd = nominal_usd_size * fee_rate;

        // Si somos Maker, proveemos liquidez. Teóricamente ejecutamos AL precio límite exacto sin slippage de libro.
        if is_maker {
            return (base_price, fee_usd);
        }

        // Si somos Taker, cruzamos el libro al salir.
        let impact_multiplier = (nominal_usd_size / 1_000_000.0).powf(1.2);
        let slippage_impact_pct = impact_multiplier * 0.0005;
        let latency_slippage = tick_volatility * 0.10;
        let total_slippage_pct = slippage_impact_pct + latency_slippage;

        let executed_price = if is_long {
            // Cerramos LONG vendiendo al BID (cruzando hacia abajo)
            base_price * (1.0 - total_slippage_pct)
        } else {
            // Cerramos SHORT comprando al ASK (cruzando hacia arriba)
            base_price * (1.0 + total_slippage_pct)
        };

        (executed_price, fee_usd)
    }
}
