/// Legacy score-based allocator with an epigenetic multiplier. This is not a
/// covariance-aware portfolio optimizer or an exchange notional validator.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct EpigeneticCapitalAllocEngine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationError {
    InvalidCapital,
    LengthMismatch,
    NonFiniteScore { index: usize },
    NonFiniteMethylation { index: usize },
}

impl EpigeneticCapitalAllocEngine {
    /// Compatibility wrapper: invalid input leaves capital unallocated. Use the
    /// checked API to distinguish invalid data from absence of positive evidence.
    #[inline(always)]
    pub fn allocate_epigenetic_top10_margin_with_capital(
        top10_scores: &[f64; 10],
        methylation_tensor: &[f64; 10],
        available_capital: f64,
    ) -> [f64; 10] {
        Self::try_allocate_epigenetic_top10_margin_with_capital(
            top10_scores,
            methylation_tensor,
            available_capital,
        )
        .unwrap_or([0.0; 10])
    }

    pub fn try_allocate_epigenetic_top10_margin_with_capital(
        scores: &[f64; 10],
        methylation: &[f64; 10],
        capital: f64,
    ) -> Result<[f64; 10], AllocationError> {
        let mut ranked = [(0, 0.0); 10];
        let mut weights = [0.0; 10];
        Self::allocate_checked(scores, methylation, capital, &mut ranked, &mut weights)?;
        Ok(weights)
    }

    /// Historical no-capital API explicitly assumes USD 13; new callers should
    /// provide observed capital through the checked API.
    #[inline(always)]
    pub fn allocate_epigenetic_top10_margin(
        top10_scores: &[f64; 10],
        methylation_tensor: &[f64; 10],
    ) -> [f64; 10] {
        Self::allocate_epigenetic_top10_margin_with_capital(top10_scores, methylation_tensor, 13.0)
    }

    /// SISTEMA SUPREMO: Genera el Tensor de Plasticidad (0.5 a 1.5) para acelerar el PPO
    /// en los activos que el modelo de Staking Epigenético considera más valiosos.
    #[inline(always)]
    pub fn calculate_plasticity_tensor(methylation_tensor: &[f64; 10]) -> [f64; 10] {
        let mut plasticity = [1.0f64; 10];
        for i in 0..10 {
            // FIX #1431: Sanitización estricta del tensor de metilación
            let safe_methyl = if methylation_tensor[i].is_finite() {
                methylation_tensor[i].clamp(0.0, 1.0)
            } else {
                0.5
            };
            // Si la metilación es alta (1.0), el PPO aprenderá a 1.5x (Mayor plasticidad).
            // Si es baja (0.0), el PPO aprenderá a 0.5x (Menor plasticidad).
            plasticity[i] = 0.5 + (safe_methyl * 1.0);
        }
        plasticity
    }

    /// Compatibility API; preserves score cardinality on errors, with zero weights.
    pub fn allocate_epigenetic_universe_margin(
        scores: &[f64],
        methylation: &[f64],
        available_capital: f64,
    ) -> Vec<f64> {
        Self::try_allocate_epigenetic_universe_margin(scores, methylation, available_capital)
            .unwrap_or_else(|_| vec![0.0; scores.len()])
    }

    /// Checked heuristic allocation, O(n log n). Positive scores are eligibility
    /// evidence, not calibrated expected returns. No floor manufactures exposure.
    pub fn try_allocate_epigenetic_universe_margin(
        scores: &[f64],
        methylation: &[f64],
        capital: f64,
    ) -> Result<Vec<f64>, AllocationError> {
        let mut ranked = vec![(0, 0.0); scores.len()];
        let mut weights = vec![0.0; scores.len()];
        Self::allocate_checked(scores, methylation, capital, &mut ranked, &mut weights)?;
        Ok(weights)
    }

    fn allocate_checked(
        scores: &[f64],
        methylation: &[f64],
        capital: f64,
        ranked: &mut [(usize, f64)],
        weights: &mut [f64],
    ) -> Result<(), AllocationError> {
        if !capital.is_finite() || capital <= 0.0 {
            return Err(AllocationError::InvalidCapital);
        }
        if scores.len() != methylation.len() {
            return Err(AllocationError::LengthMismatch);
        }
        let mut scale = 0.0_f64;
        for (index, (&score, &methyl)) in scores.iter().zip(methylation).enumerate() {
            if !score.is_finite() {
                return Err(AllocationError::NonFiniteScore { index });
            }
            if !methyl.is_finite() {
                return Err(AllocationError::NonFiniteMethylation { index });
            }
            scale = scale.max(score);
        }
        if scale == 0.0 {
            return Ok(());
        }
        for (i, entry) in ranked.iter_mut().enumerate() {
            // Common scaling cancels in normalization, bounding each raw weight
            // by one even when finite scores are near f64::MAX.
            *entry = (
                i,
                (scores[i].max(0.0) / scale) * ((0.5 + methylation[i].clamp(0.0, 1.0)) / 1.5),
            );
        }
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        // Retained compatibility policy, NOT a proof of exchange MIN_NOTIONAL
        // compliance: unequal weights can still produce sub-minimum notionals.
        let max_k = if capital < 30.0 {
            2
        } else if capital < 100.0 {
            5
        } else if capital < 300.0 {
            10
        } else {
            scores.len()
        };
        let selected = &ranked[..max_k.min(ranked.len())];
        let total: f64 = selected.iter().map(|entry| entry.1).sum();
        for &(i, weight) in selected {
            weights[i] = weight / total;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epigenetic_capital_allocation_micro_account() {
        let scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
        let methylation = [1.0; 10];

        // Legacy concentration policy; this does not verify MIN_NOTIONAL.
        let weights = EpigeneticCapitalAllocEngine::allocate_epigenetic_top10_margin_with_capital(
            &scores,
            &methylation,
            13.0,
        );
        let active_count = weights.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(active_count, 2);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);

        // En cuenta grande ($1000 USD), asigna a todos los activos
        let weights_large =
            EpigeneticCapitalAllocEngine::allocate_epigenetic_top10_margin_with_capital(
                &scores,
                &methylation,
                1000.0,
            );
        let active_count_large = weights_large.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(active_count_large, 10);
    }

    #[test]
    fn test_epigenetic_capital_allocation_30_universe() {
        let scores: Vec<f64> = (0..30).map(|i| (30 - i) as f64 / 30.0).collect();
        let methylation = vec![0.8; 30];

        let weights_micro = EpigeneticCapitalAllocEngine::allocate_epigenetic_universe_margin(
            &scores,
            &methylation,
            13.0,
        );
        assert_eq!(weights_micro.len(), 30);
        let active_micro = weights_micro.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(
            active_micro, 2,
            "Micro-account of $13 must only activate top 2 coins in 30-coin universe"
        );
        assert!((weights_micro.iter().sum::<f64>() - 1.0).abs() < 1e-6);

        let weights_large = EpigeneticCapitalAllocEngine::allocate_epigenetic_universe_margin(
            &scores,
            &methylation,
            500.0,
        );
        let active_large = weights_large.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(
            active_large, 30,
            "Large account of $500 can diversify across all 30 coins"
        );
    }

    #[test]
    fn test_epigenetic_capital_allocation_invalid_capital_abstains() {
        let scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
        let methylation = [1.0; 10];

        let weights_nan =
            EpigeneticCapitalAllocEngine::allocate_epigenetic_top10_margin_with_capital(
                &scores,
                &methylation,
                f64::NAN,
            );
        let active_nan = weights_nan.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(active_nan, 0);

        let weights_neg =
            EpigeneticCapitalAllocEngine::allocate_epigenetic_top10_margin_with_capital(
                &scores,
                &methylation,
                -50.0,
            );
        let active_neg = weights_neg.iter().filter(|&&w| w > 0.0).count();
        assert_eq!(active_neg, 0);
    }
}
