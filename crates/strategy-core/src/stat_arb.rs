use crate::{SignalIntent, SignalType};

pub struct StatArbEngine {
    window_size: usize,
    history: Vec<f64>,
    index: usize,
    count: usize,
    sum: f64,
    z_score_threshold: f64,
}

impl StatArbEngine {
    pub fn new(window_size: usize, z_score_threshold: f64) -> Self {
        let safe_window = window_size.max(2);
        let safe_thresh = if z_score_threshold.is_finite() && z_score_threshold > 0.0 {
            z_score_threshold
        } else {
            1.5
        };
        Self {
            window_size: safe_window,
            history: vec![0.0; safe_window],
            index: 0,
            count: 0,
            sum: 0.0,
            z_score_threshold: safe_thresh,
        }
    }

    /// Toma los precios de dos activos correlacionados y devuelve la intención de arbitraje sobre el Activo A.
    /// (El Activo B debe operar en la dirección contraria).
    #[inline(always)]
    pub fn update(&mut self, price_a: f64, price_b: f64) -> SignalIntent {
        if price_a <= 0.0 || price_b <= 0.0 || !price_a.is_finite() || !price_b.is_finite() {
            return SignalIntent::flat();
        }

        let spread = price_a.ln() - price_b.ln();

        let old_val = self.history[self.index];
        self.history[self.index] = spread;

        if self.count < self.window_size {
            self.count += 1;
            self.sum += spread;
        } else {
            self.sum += spread - old_val;
        }

        self.index = (self.index + 1) % self.window_size;

        if self.count < self.window_size {
            return SignalIntent::flat();
        }

        let n = self.count as f64;
        let mean = self.sum / n;
        let var_sum: f64 = self.history[..self.count]
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum();
        let variance = (var_sum / n).max(0.0);
        let stdev = if variance > 1e-12 {
            variance.sqrt()
        } else {
            1e-6
        };

        let z_score = if stdev > 0.0 {
            (spread - mean) / stdev
        } else {
            0.0
        };
        // FIX #664: Guarda de finitud estricta en Z-Score
        if !z_score.is_finite() {
            return SignalIntent::flat();
        }
        let expected_spread_edge = (spread - mean).abs();
        // FIX #576: Exigir un borde mínimo de 20 bps para cubrir 4 comisiones taker (16 bps) + slippage
        let min_spread_profit_bps = 0.0020;

        // FIX #784: StatArb emite señales de forma puramente funcional/stateless
        // Previene estados fantasma si RiskEngine o Consejo vetan la orden downstream
        if z_score > self.z_score_threshold && expected_spread_edge > min_spread_profit_bps {
            let norm_conf = (z_score.abs() / self.z_score_threshold.max(0.1)).tanh();
            return SignalIntent {
                signal: SignalType::Short,
                confidence: norm_conf,
                horizon: crate::TradeHorizon::Continuous,
                ..Default::default()
            };
        } else if z_score < -self.z_score_threshold && expected_spread_edge > min_spread_profit_bps
        {
            let norm_conf = (z_score.abs() / self.z_score_threshold.max(0.1)).tanh();
            return SignalIntent {
                signal: SignalType::Long,
                confidence: norm_conf,
                horizon: crate::TradeHorizon::Continuous,
                ..Default::default()
            };
        } else if z_score.abs() < 0.1 {
            // Regresión a la media alcanzada (Señal de cierre / Neutral)
            return SignalIntent {
                signal: SignalType::Flat,
                confidence: 1.0,
                horizon: crate::TradeHorizon::Continuous,
                ..Default::default()
            };
        }

        SignalIntent::flat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stat_arb_spread_and_signal() {
        let mut engine = StatArbEngine::new(10, 1.5);
        for i in 0..9 {
            let res = engine.update(100.0 + (i as f64 * 0.1), 100.0);
            assert_eq!(res.signal, SignalType::Flat);
        }

        // Divergencia grande de precio A respecto a B
        let signal = engine.update(110.0, 100.0);
        assert_eq!(signal.signal, SignalType::Short);
    }

    #[test]
    fn test_stat_arb_zero_window_and_nan_immunity() {
        let mut engine = StatArbEngine::new(0, f64::NAN);
        assert_eq!(engine.window_size, 2);
        assert_eq!(engine.z_score_threshold, 1.5);

        let sig_nan = engine.update(f64::NAN, 100.0);
        assert_eq!(sig_nan.signal, SignalType::Flat);

        let sig_neg = engine.update(-10.0, 100.0);
        assert_eq!(sig_neg.signal, SignalType::Flat);
    }

    #[test]
    fn test_stat_arb_symmetric_long_and_flat_reversion() {
        let mut engine = StatArbEngine::new(10, 1.5);
        for i in 0..9 {
            let res = engine.update(100.0, 100.0 + (i as f64 * 0.1));
            assert_eq!(res.signal, SignalType::Flat);
        }

        // Divergencia donde A cae fuertemente respecto a B
        let long_signal = engine.update(90.0, 100.0);
        assert_eq!(long_signal.signal, SignalType::Long);
        assert_eq!(long_signal.horizon, crate::TradeHorizon::Continuous);

        // Reversión a la media
        for _ in 0..10 {
            let _ = engine.update(100.0, 100.0);
        }
        let flat_signal = engine.update(100.0, 100.0);
        assert_eq!(flat_signal.signal, SignalType::Flat);
    }
}
