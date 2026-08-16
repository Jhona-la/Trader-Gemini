use std::f64;

/// 🔒 PREDICTOR DE IMPACTO DE MERCADO Y SLIPPAGE POR PROFUNDIDAD DE LIBRO (BOOK LEVEL SLIPPAGE PREDICTOR)
/// Estima el deslizamiento probabilístico (slippage) antes de la ejecución de una orden.
/// Selecciona automáticamente la ruta óptima de orden (Maker Post-Only vs Taker Limit).
#[derive(Debug, Clone, Copy, Default)]
pub struct BookDepthSlippagePredictor;

impl BookDepthSlippagePredictor {
    /// Calcula el impacto de mercado esperado (en bps) usando la ley de raíz cuadrada de Kyle expandida por OBI
    #[inline(always)]
    pub fn predict_slippage_bps(
        order_notional_usd: f64,
        book_depth_usd: f64,
        volatility_atr: f64,
        obi: f64,
        is_long: bool,
    ) -> f64 {
        if book_depth_usd <= 0.0 || order_notional_usd <= 0.0 {
            return 1.5; // Fallback predeterminado de 1.5 bps
        }

        let gamma = 0.50; // Constante de impacto microestructural de Kyle
        let volume_ratio = (order_notional_usd / book_depth_usd.max(100.0)).sqrt();
        let mut expected_impact_bps = gamma * (volatility_atr * 10000.0) * volume_ratio;

        // FASE 11: Penalización por Asimetría Estructural (Order Book Imbalance)
        // Si compramos y el OBI es fuertemente negativo (presión de venta), el slippage real será mayor.
        let directional_obi = if is_long { obi } else { -obi };

        // Si directional_obi es negativo, vamos contra la corriente -> slippage exponencial
        // Si directional_obi es positivo, el impacto de mercado es mitigado por la liquidez a favor
        let pressure_multiplier = if directional_obi < 0.0 {
            (1.0 + directional_obi.abs() * 2.0).exp() // Multiplicador exponencial (ej: OBI -0.8 -> exp(1.6) -> ~4.95x)
        } else {
            (1.0 - directional_obi * 0.5).max(0.5) // Reducción de hasta un 50% del slippage
        };

        expected_impact_bps *= pressure_multiplier;
        expected_impact_bps.clamp(0.1, 35.0) // Límite incrementado a 35 bps para reflejar desastres reales
    }

    /// Recomienda el tipo de orden óptimo (true para Maker Post-Only, false para Taker Market)
    #[inline(always)]
    pub fn recommend_maker_execution(
        expected_slippage_bps: f64,
        taker_fee_bps: f64,
        maker_fee_bps: f64,
    ) -> bool {
        // Si el costo Taker (fee + slippage) supera la economía Maker, usar Maker Post-Only
        let total_taker_cost = taker_fee_bps + expected_slippage_bps;
        total_taker_cost > (maker_fee_bps + 0.5)
    }
}
