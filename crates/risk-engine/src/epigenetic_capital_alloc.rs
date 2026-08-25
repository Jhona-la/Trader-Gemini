/// 🧬 ALGORITMO #103: REBALANCEADOR DE CAPITAL EPIGENÉTICO PARA EL TOP 10 (EPIGENETIC CAPITAL ALLOCATOR ENGINE)
/// Modula las máscaras de metilación epigenéticas para canalizar dinámicamente el margen ($50.00capital base)
/// exclusivamente hacia las 10 mejores oportunidades con mayor payoff esperado.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct EpigeneticCapitalAllocEngine;

impl EpigeneticCapitalAllocEngine {
    /// Asigna las fracciones óptimas de capital para los Top 10 activos bajo expresión epigenética en O(1).
    /// FIX #580: Si available_capital es bajo (< $30 USD), concentra el margen en los top-K activos para cumplir Binance MIN_NOTIONAL ($5.00).
    #[inline(always)]
    pub fn allocate_epigenetic_top10_margin_with_capital(
        top10_scores: &[f64; 10],
        methylation_tensor: &[f64; 10],
        available_capital: f64,
    ) -> [f64; 10] {
        let mut raw_weights = [0.0f64; 10];
        for i in 0..10 {
            // FIX #615: Sanitización estricta de finitud en scores y tensores de metilación
            let safe_methyl = if methylation_tensor[i].is_finite() { methylation_tensor[i].clamp(0.0, 1.0) } else { 0.5 };
            let safe_score = if top10_scores[i].is_finite() { top10_scores[i].max(0.0) } else { 0.0 };
            let epigenetic_multiplier = 0.5 + (safe_methyl * 1.0);
            raw_weights[i] = safe_score * epigenetic_multiplier;
        }

        // Determinar K máximo de activos viables según capital (ej: $13 USD -> máx 2 monedas para notional >= $5)
        let safe_cap = if available_capital.is_finite() && available_capital > 0.0 { available_capital } else { 13.0 };
        let max_k = if safe_cap < 30.0 {
            2
        } else if safe_cap < 100.0 {
            5
        } else {
            10
        };

        // Encontrar los índices del top-K
        let mut indices: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        indices.sort_by(|&a, &b| {
            raw_weights[b]
                .partial_cmp(&raw_weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut weights = [0.0f64; 10];
        let mut total_w = 0.0;
        for i in 0..max_k {
            let idx = indices[i];
            weights[idx] = raw_weights[idx].max(0.01);
            total_w += weights[idx];
        }

        let inv_total = if total_w > 0.0 { 1.0 / total_w } else { 0.0 };
        for i in 0..10 {
            weights[i] *= inv_total;
        }
        weights
    }

    /// Asigna las fracciones óptimas de capital para los Top 10 activos bajo expresión epigenética en O(1)
    #[inline(always)]
    pub fn allocate_epigenetic_top10_margin(top10_scores: &[f64; 10], methylation_tensor: &[f64; 10]) -> [f64; 10] {
        Self::allocate_epigenetic_top10_margin_with_capital(top10_scores, methylation_tensor, 13.0)
    }

    /// SISTEMA SUPREMO: Genera el Tensor de Plasticidad (0.5 a 1.5) para acelerar el PPO
    /// en los activos que el modelo de Staking Epigenético considera más valiosos.
    #[inline(always)]
    pub fn calculate_plasticity_tensor(methylation_tensor: &[f64; 10]) -> [f64; 10] {
        let mut plasticity = [1.0f64; 10];
        for i in 0..10 {
            // FIX #1431: Sanitización estricta del tensor de metilación
            let safe_methyl = if methylation_tensor[i].is_finite() { methylation_tensor[i].clamp(0.0, 1.0) } else { 0.5 };
            // Si la metilación es alta (1.0), el PPO aprenderá a 1.5x (Mayor plasticidad).
            // Si es baja (0.0), el PPO aprenderá a 0.5x (Menor plasticidad).
            plasticity[i] = 0.5 + (safe_methyl * 1.0);
        }
        plasticity
    }

    /// Asignación Epigenética Universal para N activos (hasta 30 monedas del universo) (Punto #190)
    pub fn allocate_epigenetic_universe_margin(
        scores: &[f64],
        methylation: &[f64],
        available_capital: f64,
    ) -> Vec<f64> {
        let n = scores.len().min(methylation.len());
        if n == 0 {
            return Vec::new();
        }

        let mut raw_weights = Vec::with_capacity(n);
        for i in 0..n {
            let safe_methyl = if methylation[i].is_finite() { methylation[i].clamp(0.0, 1.0) } else { 0.5 };
            let safe_score = if scores[i].is_finite() { scores[i].max(0.0) } else { 0.0 };
            let mult = 0.5 + safe_methyl;
            raw_weights.push((i, safe_score * mult));
        }

        let safe_cap = if available_capital.is_finite() && available_capital > 0.0 { available_capital } else { 13.0 };
        let max_k = if safe_cap < 30.0 {
            2
        } else if safe_cap < 100.0 {
            5
        } else if safe_cap < 300.0 {
            10
        } else {
            n
        }.min(n);

        raw_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut weights = vec![0.0f64; n];
        let mut total_w = 0.0;
        for i in 0..max_k {
            let (idx, w) = raw_weights[i];
            let val = w.max(0.01);
            if idx < n {
                weights[idx] = val;
                total_w += val;
            }
        }

        let inv_total = if total_w > 0.0 { 1.0 / total_w } else { 0.0 };
        for w in &mut weights {
            *w *= inv_total;
        }
        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epigenetic_capital_allocation_micro_account() {
        let scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
        let methylation = [1.0; 10];

        // En microcuenta ($13 USD), concentra en máx 2 activos para cumplir MIN_NOTIONAL
        let weights = EpigeneticCapitalAllocEngine::allocate_epigenetic_top10_margin_with_capital(&scores, &methylation, 13.0);
        let active_count = weights.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(active_count, 2);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);

        // En cuenta grande ($1000 USD), asigna a todos los activos
        let weights_large = EpigeneticCapitalAllocEngine::allocate_epigenetic_top10_margin_with_capital(&scores, &methylation, 1000.0);
        let active_count_large = weights_large.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(active_count_large, 10);
    }

    #[test]
    fn test_epigenetic_capital_allocation_30_universe() {
        let scores: Vec<f64> = (0..30).map(|i| (30 - i) as f64 / 30.0).collect();
        let methylation = vec![0.8; 30];

        let weights_micro = EpigeneticCapitalAllocEngine::allocate_epigenetic_universe_margin(&scores, &methylation, 13.0);
        assert_eq!(weights_micro.len(), 30);
        let active_micro = weights_micro.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(active_micro, 2, "Micro-account of $13 must only activate top 2 coins in 30-coin universe");
        assert!((weights_micro.iter().sum::<f64>() - 1.0).abs() < 1e-6);

        let weights_large = EpigeneticCapitalAllocEngine::allocate_epigenetic_universe_margin(&scores, &methylation, 500.0);
        let active_large = weights_large.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(active_large, 30, "Large account of $500 can diversify across all 30 coins");
    }

    #[test]
    fn test_epigenetic_capital_allocation_nan_capital_immunity() {
        let scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
        let methylation = [1.0; 10];

        let weights_nan = EpigeneticCapitalAllocEngine::allocate_epigenetic_top10_margin_with_capital(&scores, &methylation, f64::NAN);
        let active_nan = weights_nan.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(active_nan, 2);

        let weights_neg = EpigeneticCapitalAllocEngine::allocate_epigenetic_top10_margin_with_capital(&scores, &methylation, -50.0);
        let active_neg = weights_neg.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(active_neg, 2);
    }
}
