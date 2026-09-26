use serde::{Deserialize, Serialize};

// XXXIV audit: deterministic local cost heuristic, not an exchange fill model.
// Maker results are conditional price/fee estimates: no queue/fill probability.
// Fixed reference notional, 150ms time reference, caps and coefficients are not
// a cross-asset calibration. Invalid latency and output-overflow contracts remain
// open; callers must not infer that finite input price implies a valid fill.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineMode {
    Optimistic, // Solo Fees fijos, sin impacto de latencia ni de libro de órdenes. (Ideal para IA training inicial)
    HyperRealistic, // Nombre legacy: fórmula determinista, no demora estocástica simulada.
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

    /// Estima precio/coste taker con la heurística local; no confirma una ejecución.
    /// Devuelve: (precio_ejecutado, fee_total_usd)
    pub fn calculate_market_entry(
        &self,
        base_price: f64,
        is_long: bool,
        nominal_usd_size: f64,
        tick_volatility: f64,
        base_slippage_floor: f64,
        latency_penalty_ms: f64,
    ) -> (f64, f64) {
        if base_price <= 0.0 || !base_price.is_finite() {
            return (0.0, 0.0);
        }
        let safe_nominal = if nominal_usd_size.is_finite() {
            nominal_usd_size.max(0.0)
        } else {
            0.0
        };
        let safe_vol = if tick_volatility.is_finite() {
            tick_volatility.max(0.0)
        } else {
            0.0
        };
        // FIX #686: Sanitizar comisiones
        let safe_taker_fee = if self.base_taker_fee.is_finite() && self.base_taker_fee >= 0.0 {
            self.base_taker_fee
        } else {
            0.0005
        };

        if self.mode == EngineMode::Optimistic {
            let fee = safe_nominal * safe_taker_fee;
            return (base_price, fee);
        }

        // --- HYPER REALISTIC PHYSICS ---
        // 1. Orderbook impact: ley raíz cuadrada EMPÍRICA del mercado (la
        // forma funcional que miden los datos institucionales): impacto ∝
        // √Q. La versión anterior usaba potencia 1.2 CONTRADICIENDO su
        // propio comentario "cuadrático 100×" (1.2 ⇒ 10M→15.8×, no 100×).
        // Calibración: $1M mueve ~5 bps en ultra-líquidos ⇒ constante 5bps.
        let impact_multiplier = (safe_nominal / 1_000_000.0).sqrt();
        let slippage_impact_pct = impact_multiplier * 0.0005;

        // 2. Latency slippage: el precio es BROWNIANO — la dispersión
        // crece con la RAÍZ del tiempo, no linealmente. La versión lineal
        // SOBRESTIMABA el costo de latencias cortas y SUBRESTIMaba el de
        // largas: sesgo sistemático en todo fill simulado. τ=150ms es la
        // escala de referencia de la calibración.
        // D-753 — UNA SOLA FÍSICA PARA EL MISMO EVENTO. La raíz era correcta
        // (el precio difunde: la dispersión crece con √t), pero la referencia
        // NO: 150 ms es una escala inventada, mientras `safe_vol` es el ATR
        // relativo medido sobre la ventana de un minuto. Escalar un ATR de 60 s
        // como si fuera de 150 ms multiplica el deslizamiento por √(60 000/150)
        // = 20. Con 13,6 ms de latencia y un ATR del 0,2 %, la física cobraba
        // 6,0 pb donde la difusión da 0,30 pb: el simulador penalizaba cada
        // fill con veinte veces el coste real de la latencia. Ahora llama a la
        // MISMA función pura que el gate de expectativa y el constructor de la
        // orden (`tp_sl::latency_slippage_pct`), cuya referencia temporal es la
        // de la propia medida de volatilidad.
        let latency_slippage =
            risk_engine::tp_sl::latency_slippage_pct(safe_vol, latency_penalty_ms);

        let total_slippage_pct = (slippage_impact_pct + latency_slippage)
            .max(base_slippage_floor)
            .clamp(0.0, 0.05);

        let executed_price = if is_long {
            base_price * (1.0 + total_slippage_pct) // Compramos más caro
        } else {
            base_price * (1.0 - total_slippage_pct) // Vendemos más barato
        };

        let fee_usd = safe_nominal * safe_taker_fee;

        (executed_price, fee_usd)
    }

    /// Estimación condicional maker (Post-Only); no modela si/cuándo se llena.
    /// Para órdenes de horizonte más pausado (tau >= 60s), se coloca en el mejor bid/ask
    /// ejecutando sin slippage adverso y pagando tarifa Maker VIP0 (0.0002).
    pub fn calculate_maker_entry(
        &self,
        base_price: f64,
        _is_long: bool,
        nominal_usd_size: f64,
    ) -> (f64, f64) {
        if base_price <= 0.0 || !base_price.is_finite() {
            return (0.0, 0.0);
        }
        let safe_nominal = if nominal_usd_size.is_finite() {
            nominal_usd_size.max(0.0)
        } else {
            0.0
        };
        let safe_maker_fee = if self.base_maker_fee.is_finite() && self.base_maker_fee >= 0.0 {
            self.base_maker_fee
        } else {
            0.0002
        };
        let fee_usd = safe_nominal * safe_maker_fee;
        (base_price, fee_usd)
    }

    /// Estimación local de salida taker/maker; no resultado reconciliado del exchange.
    pub fn calculate_exit(
        &self,
        base_price: f64,
        is_long: bool,
        nominal_usd_size: f64,
        is_maker: bool,
        tick_volatility: f64,
        base_slippage_floor: f64,
        latency_penalty_ms: f64,
    ) -> (f64, f64) {
        if base_price <= 0.0 || !base_price.is_finite() {
            return (0.0, 0.0);
        }
        let safe_nominal = if nominal_usd_size.is_finite() {
            nominal_usd_size.max(0.0)
        } else {
            0.0
        };
        let safe_vol = if tick_volatility.is_finite() {
            tick_volatility.max(0.0)
        } else {
            0.0
        };
        // FIX #686: Sanitizar comisiones
        let safe_maker_fee = if self.base_maker_fee.is_finite() && self.base_maker_fee >= 0.0 {
            self.base_maker_fee
        } else {
            0.0002
        };
        let safe_taker_fee = if self.base_taker_fee.is_finite() && self.base_taker_fee >= 0.0 {
            self.base_taker_fee
        } else {
            0.0005
        };

        if self.mode == EngineMode::Optimistic {
            let fee_rate = if is_maker {
                safe_maker_fee
            } else {
                safe_taker_fee
            };
            return (base_price, safe_nominal * fee_rate);
        }

        let fee_rate = if is_maker {
            safe_maker_fee
        } else {
            safe_taker_fee
        };
        let fee_usd = safe_nominal * fee_rate;

        // Si somos Maker, proveemos liquidez. Teóricamente ejecutamos AL precio límite exacto sin slippage de libro.
        if is_maker {
            return (base_price, fee_usd);
        }

        // Si somos Taker, cruzamos el libro al salir.
        // CERT-M3-H02: la SALIDA retiene potencia-1.2 e impacto LINEAL en
        // latencia — exactamente los dos defectos que M0.6 corrigió en la
        // ENTRADA. Cada fill de cierre simulado pagaba física diferente a
        // su apertura: asimetría sistemática que favorecía stop-heavy
        // genomes en backtest. Unificado al mismo kernel: √Q impacto y
        // √latency dispersión browniana.
        let impact_multiplier = (safe_nominal / 1_000_000.0).sqrt();
        let slippage_impact_pct = impact_multiplier * 0.0005;
        // D-753 — UNA SOLA FÍSICA PARA EL MISMO EVENTO. La raíz era correcta
        // (el precio difunde: la dispersión crece con √t), pero la referencia
        // NO: 150 ms es una escala inventada, mientras `safe_vol` es el ATR
        // relativo medido sobre la ventana de un minuto. Escalar un ATR de 60 s
        // como si fuera de 150 ms multiplica el deslizamiento por √(60 000/150)
        // = 20. Con 13,6 ms de latencia y un ATR del 0,2 %, la física cobraba
        // 6,0 pb donde la difusión da 0,30 pb: el simulador penalizaba cada
        // fill con veinte veces el coste real de la latencia. Ahora llama a la
        // MISMA función pura que el gate de expectativa y el constructor de la
        // orden (`tp_sl::latency_slippage_pct`), cuya referencia temporal es la
        // de la propia medida de volatilidad.
        let latency_slippage =
            risk_engine::tp_sl::latency_slippage_pct(safe_vol, latency_penalty_ms);
        let total_slippage_pct = (slippage_impact_pct + latency_slippage)
            .max(base_slippage_floor)
            .clamp(0.0, 0.05);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reality_physics_optimistic_entry_and_exit() {
        let phys = RealityPhysics::new(EngineMode::Optimistic);
        let (entry_price, fee) =
            phys.calculate_market_entry(60000.0, true, 13.0, 0.001, 0.0005, 15.0);
        assert_eq!(entry_price, 60000.0);
        assert_eq!(fee, 13.0 * 0.0005);

        let (exit_price, exit_fee) =
            phys.calculate_exit(60000.0, true, 13.0, true, 0.001, 0.0005, 15.0);
        assert_eq!(exit_price, 60000.0);
        assert_eq!(exit_fee, 13.0 * 0.0002);
    }

    #[test]
    fn test_reality_physics_hyper_realistic_long_and_short_slippage() {
        let phys = RealityPhysics::default();
        let (entry_long_price, _) =
            phys.calculate_market_entry(60000.0, true, 100.0, 0.002, 0.0005, 15.0);
        assert!(entry_long_price > 60000.0); // Slippage increases long buy price

        let (entry_short_price, _) =
            phys.calculate_market_entry(60000.0, false, 100.0, 0.002, 0.0005, 15.0);
        assert!(entry_short_price < 60000.0); // Slippage decreases short sell price

        let (exit_long_price, _) =
            phys.calculate_exit(60000.0, true, 100.0, false, 0.002, 0.0005, 15.0);
        assert!(exit_long_price < 60000.0); // Taker exit long sells lower

        let (exit_short_price, _) =
            phys.calculate_exit(60000.0, false, 100.0, false, 0.002, 0.0005, 15.0);
        assert!(exit_short_price > 60000.0); // Taker exit short buys higher
    }

    #[test]
    fn test_reality_physics_nan_and_negative_immunity() {
        let phys = RealityPhysics::default();
        let (p1, f1) = phys.calculate_market_entry(f64::NAN, true, -10.0, f64::NAN, 0.0005, 15.0);
        assert_eq!(p1, 0.0);
        assert_eq!(f1, 0.0);

        let (p2, f2) = phys.calculate_exit(-50.0, false, f64::NAN, true, f64::NAN, 0.0005, 15.0);
        assert_eq!(p2, 0.0);
        assert_eq!(f2, 0.0);
    }
}
