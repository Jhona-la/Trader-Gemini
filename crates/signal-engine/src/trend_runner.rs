use std::f64;

/// 🚀 ALGORITMO #27: MOTOR DE EXPANSIÓN DE TENDENCIA DE ALTA GANANCIA (HIGH-PAYOFF TREND-RUNNER)
/// Amplifica los objetivos de Take-Profit en Swing (+3.5% a +8.0%) cuando se confirma inercia estocástica.
/// Al lograr un payoff de +500 bps frente a 4.5 bps de fee, el usuario retiene más del 99.1% de la ganancia.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct HighPayoffTrendRunner;

impl HighPayoffTrendRunner {
    /// Calcula el objetivo de extensión de ganancia alta usando funciones continuas tensoriales (O(1))
    #[inline(always)]
    pub fn calculate_expanded_tp(
        base_tp_magnitude: f64,
        hurst: f64,
        vpin: f64,
        atr_pct: f64,
        upper_bound: f64, // Extracted from SuperGenotype
    ) -> f64 {
        // Activación continua: Hurst (>0.5 indica tendencia) y VPIN (fuerza de volumen)
        let trend_score = ((hurst - 0.5) * 4.0).tanh().max(0.0);
        let volume_score = (vpin * 2.0).tanh().max(0.0);

        // Factor de expansión tensorial (0.0 a 1.0)
        let expansion_factor = trend_score * volume_score;

        // Cálculo del TP expandido, integrando directamente la volatilidad (atr_pct)
        let expanded_tp = base_tp_magnitude * (1.0 + (expansion_factor * 8.0));
        let vol_boost = atr_pct * expansion_factor * 10.0;

        let final_tp = (expanded_tp + vol_boost).max(0.0120);

        // Límite superior suave continuo usando tanh() en lugar de un clamp() brusco.
        // Asymptotically approaches upper_bound sin romper derivabilidad.
        let safe_bound = upper_bound.max(0.01); // Prevent div by zero
        safe_bound * (final_tp / safe_bound).tanh()
    }
}
