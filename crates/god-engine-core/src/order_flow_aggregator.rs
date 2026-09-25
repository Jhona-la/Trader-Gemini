/// ⚡ ALGORITMO #63: AGREGADOR DE FLUJO DE ÓRDENES CERO-COPIA NANO HFT (ORDER FLOW AGGREGATOR ENGINE)
/// Agrega volúmenes agresores de compra y venta en microsegundos sin asignaciones en Heap.
/// Pure arithmetic helper; this function alone does not establish system latency.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct OrderFlowAggregatorEngine;

impl OrderFlowAggregatorEngine {
    /// Agrega el volumen agresor instantáneo en O(1)
    #[inline(always)]
    pub fn aggregate_order_flow(buy_volume: f64, sell_volume: f64) -> (f64, f64) {
        // FIX #713: Sanitización de positividad y finitud estricta con guarda de volumen nulo
        let safe_buy = if buy_volume.is_finite() && buy_volume >= 0.0 {
            buy_volume
        } else {
            0.0
        };
        let safe_sell = if sell_volume.is_finite() && sell_volume >= 0.0 {
            sell_volume
        } else {
            0.0
        };
        let scale = safe_buy.max(safe_sell);
        if scale == 0.0 {
            return (0.5, 0.0);
        }
        // Homogeneous degree zero: changing volume units cannot change the
        // ratio. Scale first to avoid overflowing B+S for finite inputs.
        // A nonzero tiny volume is not missing data or a neutral market.
        let buy = safe_buy / scale;
        let sell = safe_sell / scale;
        let buy_ratio = buy / (buy + sell);
        let delta = safe_buy - safe_sell;
        (buy_ratio, delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_flow_aggregator_balanced_and_skewed() {
        let (ratio, delta) = OrderFlowAggregatorEngine::aggregate_order_flow(100.0, 100.0);
        assert!((ratio - 0.5).abs() < 1e-6);
        assert_eq!(delta, 0.0);

        let (ratio2, delta2) = OrderFlowAggregatorEngine::aggregate_order_flow(300.0, 100.0);
        assert!((ratio2 - 0.75).abs() < 1e-6);
        assert_eq!(delta2, 200.0);
    }

    #[test]
    fn test_order_flow_aggregator_nan_and_negative_immunity() {
        let (ratio, delta) = OrderFlowAggregatorEngine::aggregate_order_flow(f64::NAN, -50.0);
        assert_eq!(ratio, 0.5);
        assert_eq!(delta, 0.0);
    }
}
