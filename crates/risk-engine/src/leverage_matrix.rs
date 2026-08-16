use quantum_arena::GlobalArena;
use signal_engine::SignalIntent;
use std::sync::atomic::Ordering;

/// 🚀 ALGORITMO #77 V11: QUANTUM LEVERAGE MATRIX (Adaptativo-Autoevolutivo)
///
/// FASE 10: Erradicación de hardcodes. Todo parámetro es un gen del genoma
/// o derivado de métricas reales del arena (WR, PF, Hurst, ATR).
///
/// QUÉ: Algoritmo de apalancamiento que combina 5 tensores matemáticos para
///      producir un leverage óptimo por operación.
///
/// CÓMO: 5 Tensores multiplicativos:
///   T1 = Fractional Kelly (con profit_factor REAL, no proxy)
///   T2 = Conviction³ (Bayesian Proxy, escala del genoma)
///   T3 = Volatility Brake (tanh friction, SIEMPRE activo incluso en scalp)
///   T4 = Growth Pressure (relativo a base_capital, no a $50)
///   T5 = Hurst Predictability Bonus
pub struct QuantumLeverageMatrix;

impl QuantumLeverageMatrix {
    /// Retorna el Leverage recomendado (entre 1.0 y genome_max_leverage)
    ///
    /// Fórmula: L = T1(kelly) × T2(conviction) × T3(vol_brake) × T4(growth) × T5(hurst)
    pub fn calculate_dynamic_leverage(
        signal: &SignalIntent,
        is_scalp: bool,
        current_capital: f64,
        base_capital: f64, // Capital base real (extraído de API)
        tick_volatility: f64,
        volatility_multiplier: f64, // Extracted from SuperGenotype
        hurst_exponent: f64,        // TENSOR 5: Predictabilidad de la serie
        real_profit_factor: f64, // PF REAL del coin (de arena.coins[id].scalp/swing.profit_factor)
        genome_max_leverage: f64, // Límite del genoma (de config.global_leverage)
        arena: &GlobalArena,
    ) -> f64 {
        // ═══════════════════════════════════════════════════════
        // TENSOR 1: Full Fractional Kelly (con profit_factor REAL)
        // ═══════════════════════════════════════════════════════
        // Kelly: K = p - (1-p)/R donde R es el profit factor REAL
        let prob_win = signal.confidence; // Remove .clamp(0.5, 0.99)
        let pf = real_profit_factor.max(0.1); // PF real, fallback si no hay historial
        let kelly = (prob_win - (1.0 - prob_win) / pf).max(0.01);

        // Fracción adaptativa: Hurst × confidence determinan agresividad
        // Usamos fraction multiplier dictado por el genoma o derivado de la escala predictiva.
        let fraction_multiplier =
            if prob_win > arena.config.veto_threshold_btc.load(Ordering::Relaxed) {
                1.0
            } else {
                (prob_win - 0.5).max(0.1) * 2.0 // Minimum 0.1 to prevent near-zero fractions
            };

        let dynamic_kelly = kelly * fraction_multiplier;

        // ═══════════════════════════════════════════════════════
        // TENSOR 2: Convicción de la Señal (Bayesian Proxy)
        // ═══════════════════════════════════════════════════════
        let conviction_scale = volatility_multiplier.max(1.0); // Minimum 1.0, let genome control scale
        let conviction = signal.confidence.powi(3) * conviction_scale;

        // ═══════════════════════════════════════════════════════
        // TENSOR 3: Freno de Volatilidad (SIEMPRE activo)
        // ═══════════════════════════════════════════════════════
        let vol_sensitivity = volatility_multiplier.max(1.0); // Minimum 1.0, genome dictates
        let vol_brake = 1.0 - (tick_volatility * vol_sensitivity).tanh();

        // ═══════════════════════════════════════════════════════
        // TENSOR 4: Micro-Capital Acceleration (Curva Logarítmica)
        // ═══════════════════════════════════════════════════════
        // A menor capital, mayor multiplicador para permitir interés compuesto rápido.
        // A mayor capital, amortiguación logarítmica para proteger patrimonio.
        let log_divisor = arena
            .config
            .lev_matrix_log_cap_divisor
            .load(Ordering::Relaxed);
        let capital_ratio = current_capital / base_capital.max(1.0);
        let log_cap = current_capital.max(2.0).log10();
        let logarithmic_dampener = (2.0 / log_cap).max(0.2); // Remove strict upper bound

        let growth_scalar = arena
            .config
            .lev_matrix_growth_scalar
            .load(Ordering::Relaxed);
        let growth_factor = (1.0 + growth_scalar / (1.0 + capital_ratio)) * logarithmic_dampener;

        // Techo dinámico logarítmico: capitales bajos permiten leverages altos guiados por EV
        let dynamic_ceiling = (100.0 * (1.0 - (log_cap / log_divisor))).max(3.0); // Remove strict 100 max bound
        let effective_max_leverage = genome_max_leverage.min(dynamic_ceiling);

        // ═══════════════════════════════════════════════════════
        // TENSOR 5: Hurst Predictability Bonus
        // ═══════════════════════════════════════════════════════
        let hurst_bonus = if hurst_exponent > 0.5 {
            1.0 + (hurst_exponent - 0.5) * 2.0
        } else {
            1.0 - (0.5 - hurst_exponent)
        };

        let vol_clamp_min = arena
            .config
            .lev_matrix_vol_clamp_min
            .load(Ordering::Relaxed);

        let (final_vol_factor, final_dynamic_kelly) = if is_scalp {
            // SCALP (Milisegundos/Segundos): Extreme Volatility Dependency
            // Requires huge predictability to lever up safely.
            (
                vol_brake.max(vol_clamp_min),
                kelly * fraction_multiplier.max(1.0),
            )
        } else {
            // SWING (Horas/Días): Macro Trend Dependency
            // Volatility is smoothed; max leverage is strictly constrained by base risk.
            (vol_brake.max(vol_clamp_min + 0.2), dynamic_kelly * 0.7) // Swing has a natural 30% reduction in base leverage capability
        };

        // Capital factor: Kelly escala el leverage. Sqrt para suavizar.
        let capital_factor = 1.0 + (final_dynamic_kelly * effective_max_leverage.sqrt()).sqrt();

        let final_leverage =
            capital_factor * conviction * final_vol_factor * growth_factor * hurst_bonus;

        let clamped = final_leverage.clamp(1.0, effective_max_leverage);

        // telemetry_engine::telemetry!(
        //     "Leverage Matrix: {:?} [Scalp={}] (Hurst: {:.2}, VolFactor: {:.2}, PF: {:.2}) -> {:.2}x",
        //     signal.signal, is_scalp, hurst_bonus, final_vol_factor, real_profit_factor, clamped
        // );

        clamped
    }
}
