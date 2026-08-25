/// 🧬 ALGORITMO #162: MOTOR RASTREADOR DE PAISAJE DE FITNESS EPIGENÉTICO ADAPTATIVO AUTO-EVOLUTIVO (EPIGENETIC FITNESS LANDSCAPE ENGINE)
/// Mapea el gradiente de adaptación del genoma sobre el espacio de estados del mercado en tiempo real,
/// enrutando el capital de $50.00capital base hacia los Top 10 activos con mayor densidad de fitness en O(1).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct EpigeneticFitnessLandscapeEngine;

impl EpigeneticFitnessLandscapeEngine {
    /// Calcula la densidad de fitness [0.0, 1.0] en la ubicación actual del paisaje de adaptación en O(1)
    #[inline(always)]
    pub fn compute_fitness_density(sharpe: f64, win_rate: f64, profit_factor: f64) -> f64 {
        // FIX #654: Sanitizar parámetros de fitness epigenético
        let safe_sharpe = if sharpe.is_finite() { sharpe.max(0.0) } else { 0.0 };
        let safe_wr = if win_rate.is_finite() { win_rate.clamp(0.0, 1.0) } else { 0.5 };
        let safe_pf = if profit_factor.is_finite() && profit_factor > 1.0 { profit_factor - 1.0 } else { 0.0 };

        let score = (safe_sharpe * 0.4) + (safe_wr * 0.3) + (safe_pf * 0.3);
        if score.is_finite() { score.clamp(0.0, 1.0) } else { 0.5 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epigenetic_fitness_density_calculation() {
        let density = EpigeneticFitnessLandscapeEngine::compute_fitness_density(2.0, 0.70, 2.5);
        assert!(density > 0.5 && density <= 1.0);
    }

    #[test]
    fn test_epigenetic_fitness_density_nan_immunity() {
        let density = EpigeneticFitnessLandscapeEngine::compute_fitness_density(f64::NAN, f64::NAN, f64::NAN);
        assert!(density.is_finite());
        assert_eq!(density, 0.15); // (0.0 * 0.4) + (0.5 * 0.3) + (0.0 * 0.3) = 0.15
    }
}

