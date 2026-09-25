use std::f64;

/// 🔒 PREDICTOR DE IMPACTO DE MERCADO Y SLIPPAGE POR PROFUNDIDAD DE LIBRO (BOOK LEVEL SLIPPAGE PREDICTOR)
/// Heurística determinista de coste y recomendación; no devuelve una distribución
/// probabilística ni demuestra una ruta óptima. Sin caller operativo localizado.
#[derive(Debug, Clone, Copy, Default)]
pub struct BookDepthSlippagePredictor;

impl BookDepthSlippagePredictor {
    /// Score en bps con forma raíz-cuadrada y modulador OBI. Coeficientes/cotas
    /// heredados no calibrados aquí; esta forma no acredita un modelo de Kyle.
    #[inline(always)]
    pub fn predict_slippage_bps(
        order_notional_usd: f64,
        book_depth_usd: f64,
        volatility_atr: f64,
        obi: f64,
        is_long: bool,
    ) -> f64 {
        if book_depth_usd <= 0.0
            || order_notional_usd <= 0.0
            || !order_notional_usd.is_finite()
            || !book_depth_usd.is_finite()
            || !volatility_atr.is_finite()
            || !obi.is_finite()
        {
            return 1.5; // Fallback predeterminado de 1.5 bps
        }

        let gamma = 0.50; // Coeficiente heredado; no constante universal.
        // FIX #714: Clampear ratio de volumen y ATR para evitar explosión de slippage en libros desiertos
        let safe_vol_ratio = (order_notional_usd / book_depth_usd.max(100.0))
            .sqrt()
            .clamp(0.001, 10.0);
        let safe_atr = volatility_atr.clamp(0.00001, 0.50);
        let mut expected_impact_bps = gamma * (safe_atr * 10000.0) * safe_vol_ratio;

        // FASE 11: Penalización por Asimetría Estructural (Order Book Imbalance)
        // Si compramos y el OBI es fuertemente negativo (presión de venta), el slippage real será mayor.
        // FIX #635: Clampear directional_obi a [-1.0, 1.0] para evitar desbordamiento en exp()
        let raw_directional_obi = if is_long { obi } else { -obi };
        let directional_obi = raw_directional_obi.clamp(-1.0, 1.0);

        // Si directional_obi es negativo, vamos contra la corriente -> slippage exponencial
        // Si directional_obi es positivo, el impacto de mercado es mitigado por la liquidez a favor
        let pressure_multiplier = if directional_obi < 0.0 {
            (directional_obi.abs() * 2.0).exp() // exp(1.6) ≈ 4.95; límite en OBI=0 es 1.
        } else {
            (1.0 - directional_obi * 0.5).max(0.5) // Reducción de hasta un 50% del slippage
        };

        expected_impact_bps *= pressure_multiplier;
        if expected_impact_bps.is_finite() {
            expected_impact_bps.clamp(0.1, 35.0) // Límite incrementado a 35 bps para reflejar desastres reales
        } else {
            1.5
        }
    }

    /// Recomienda el tipo de orden óptimo (true para Maker Post-Only, false para Taker Market/IOC)
    #[inline(always)]
    pub fn recommend_maker_execution(
        expected_slippage_bps: f64,
        taker_fee_bps: f64,
        maker_fee_bps: f64,
        urgency_score: f64, // 0.0 (paciente) a 1.0 (inmediata/stop/breakout)
    ) -> bool {
        // FIX #667: Sanitizar parámetros de decisión Maker/Taker
        if !expected_slippage_bps.is_finite()
            || !taker_fee_bps.is_finite()
            || !maker_fee_bps.is_finite()
            || !urgency_score.is_finite()
        {
            return false;
        }

        if urgency_score > 0.65 {
            return false; // Alta urgencia: cruzar el libro (Taker IOC)
        }
        let total_taker_cost = taker_fee_bps + expected_slippage_bps;
        let maker_cost_with_adverse_selection = maker_fee_bps + (urgency_score * 4.0);
        total_taker_cost > (maker_cost_with_adverse_selection + 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_predictor_and_recommendation() {
        let slip =
            BookDepthSlippagePredictor::predict_slippage_bps(100.0, 50000.0, 0.001, 0.2, true);
        assert!(slip >= 0.1 && slip <= 35.0);

        // Baja urgencia -> Recomienda Maker
        let maker = BookDepthSlippagePredictor::recommend_maker_execution(slip, 4.0, 2.0, 0.1);
        assert!(maker);

        // Alta urgencia -> Recomienda Taker IOC para no perder rompimiento
        let taker = BookDepthSlippagePredictor::recommend_maker_execution(slip, 4.0, 2.0, 0.85);
        assert!(!taker);
    }

    #[test]
    fn test_slippage_predictor_nan_immunity_and_zero_depth() {
        let slip_nan = BookDepthSlippagePredictor::predict_slippage_bps(
            f64::NAN,
            0.0,
            f64::NAN,
            f64::NAN,
            false,
        );
        assert!(slip_nan.is_finite());
        assert!(slip_nan >= 0.1 && slip_nan <= 35.0);

        let maker_nan =
            BookDepthSlippagePredictor::recommend_maker_execution(f64::NAN, 4.0, 2.0, 0.1);
        assert!(!maker_nan);
    }
}
