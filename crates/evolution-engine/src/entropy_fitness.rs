use strategy_core::SignalType;

/// 🧬 ALGORITMO #77: SHANNON ENTROPY FITNESS
/// Calcula la entropía de información de las decisiones de un modelo evolutivo.
/// Si un modelo siempre dice "Long" o siempre "Flat", su entropía es 0 (Sobreajuste extremo).
/// Un modelo robusto debe tener alta entropía en sus decisiones, reflejando adaptación.
pub struct EntropyFitness;

impl EntropyFitness {
    /// Calcula la Entropía de Shannon (en bits) de un conjunto de decisiones.
    pub fn calculate_shannon_entropy(decisions: &[SignalType]) -> f64 {
        if decisions.is_empty() {
            return 0.0;
        }

        let mut counts = [0usize; 3];
        for &decision in decisions {
            match decision {
                SignalType::Long => counts[0] += 1,
                SignalType::Short => counts[1] += 1,
                SignalType::Flat => counts[2] += 1,
            }
        }

        let total = decisions.len() as f64;
        let mut entropy = 0.0;

        for &count in &counts {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Penalización Bayesiana Epigenética
    /// Combina la probabilidad Priori (Backtest) y la Evidencia (Live Demo) para generar
    /// una probabilidad Posterior de que la estrategia NO esté sobreajustada.
    /// Retorna un multiplicador [0.0, 1.0] para castigar el fitness.
    pub fn bayesian_posterior_collapse_penalty(
        prior_wr: f64, // Winrate en Backtest
        live_wr: f64,  // Winrate en Producción/Demo
        live_trades: usize,
    ) -> f64 {
        // FIX #672: Validar finitud de winrates
        if !prior_wr.is_finite() || !live_wr.is_finite() {
            return 1.0;
        }

        if live_trades < 5 {
            return 1.0; // Poca evidencia, mantener beneficio de la duda
        }

        // Si la realidad (live_wr) es mucho peor que la teoría (prior_wr),
        // el teorema de bayes colapsa rápidamente la confianza.
        let divergence = (prior_wr - live_wr).max(0.0);

        // Función de decaimiento exponencial basada en la divergencia y la cantidad de evidencia
        let evidence_weight = (live_trades as f64 / 30.0).min(1.0);
        let penalty = f64::exp(-divergence * 10.0 * evidence_weight);

        // Retornamos el multiplicador de penalización (1.0 = Perfecto, 0.1 = Colapso total)
        penalty.clamp(0.01, 1.0)
    }

    /// FASE 7: REALITY GAP ADVERSARIAL PENALTY (Autoevolutivo)
    /// Castiga las estrategias que tienen un Sharpe de Backtest altísimo pero un Sharpe de Demo mediocre.
    /// Esto elimina las estrategias "Sobreajustadas" que encontraron anomalías en datos pasados pero fallan en el futuro.
    pub fn reality_gap_adversarial_score(
        backtest_sharpe: f64,
        live_sharpe: f64,
        live_trades: usize,
    ) -> f64 {
        // FIX #672: Validar finitud de Sharpes
        if !backtest_sharpe.is_finite() || !live_sharpe.is_finite() {
            return 1.0;
        }

        if live_trades < 10 {
            return 1.0; // Muy poca evidencia en vivo, no penalizar todavía.
        }

        let gap = (backtest_sharpe - live_sharpe).max(0.0);
        let ratio = if backtest_sharpe > 0.0 {
            gap / backtest_sharpe
        } else {
            0.0
        };

        // Si el gap es > 50% (ej. Sharpe 4 en backtest, 1.5 en live), penalización masiva (fitness mutante)
        let decay = 1.0 - (ratio * 1.5).tanh();

        decay.clamp(0.05, 1.0)
    }

    /// 🚀 V13+FASE22: PENALIZACIÓN DE SLIPPAGE DE REALIDAD (Distribución de Poisson)
    /// Castiga severamente las estrategias genéticas que sobreoperan asumiendo 0 fricción.
    /// Simula el impacto real del bid/ask spread y latencia (50-200ms) de Binance.
    /// `actual_fee_rate`: Fee real por trade (ej: taker_fee del CoinArena). NUNCA hardcodear.
    pub fn reality_slippage_penalty(
        gross_pnl: f64,
        total_trades: usize,
        actual_fee_rate: f64,
    ) -> f64 {
        Self::reality_slippage_penalty_with_notional(gross_pnl, total_trades, actual_fee_rate, 35.0)
    }

    /// Penalización de slippage con volumen nocional dinámico parametrizado por trade
    pub fn reality_slippage_penalty_with_notional(
        gross_pnl: f64,
        total_trades: usize,
        actual_fee_rate: f64,
        avg_notional_usd: f64,
    ) -> f64 {
        // FIX #672: Validar finitud de inputs de slippage
        if !gross_pnl.is_finite() || !actual_fee_rate.is_finite() || !avg_notional_usd.is_finite() {
            return 0.0;
        }

        if total_trades == 0 {
            return gross_pnl;
        }
        let notional = if avg_notional_usd > 0.0 { avg_notional_usd } else { 35.0 };
        let total_volume_usd = total_trades as f64 * notional;

        // Fricción base: comisión roundtrip real de Binance VIP0 (0.06% o tasa de cuenta) sobre el volumen
        let fee_rate = if actual_fee_rate > 0.0 { actual_fee_rate } else { 0.0006 };
        let deterministic_friction = total_volume_usd * fee_rate;

        // Fricción estocástica por slippage (distribución de Poisson sobre trades afectados)
        let lambda = total_trades as f64 * 0.05; // 5% de trades sufren slippage
        let std_dev = lambda.sqrt();
        let stochastic_friction = (lambda + std_dev * 1.96) * (notional * 0.0005); // ~5 bps

        gross_pnl - deterministic_friction - stochastic_friction
    }

    /// FASE 17: PENALIZACIÓN DE CAÍDA LIBRE (MAX DRAWDOWN ADVERSARIAL PENALTY)
    /// Castiga exponencialmente a las estrategias que acumulen un Max Drawdown
    /// superior a la tolerancia dictada por la gestión de micro-capitales.
    /// Devuelve un multiplicador entre [0.0, 1.0].
    pub fn drawdown_adversarial_penalty(max_drawdown_pct: f64, threshold_pct: f64) -> f64 {
        // FIX #672: Validar finitud de drawdown
        if !max_drawdown_pct.is_finite() || !threshold_pct.is_finite() {
            return 1.0;
        }

        if max_drawdown_pct <= threshold_pct {
            return 1.0; // Drawdown aceptable, 100% retención del fitness
        }

        // Exceso sobre el umbral
        let excess = max_drawdown_pct - threshold_pct;

        // Decaimiento exponencial abrupto. Un DD de 20% multiplicaría por ~0.04 (aniquilado)
        let decay = f64::exp(-excess * 20.0);

        decay.clamp(0.01, 1.0)
    }

    /// Función de Fitness Multi-Objetivo Hiperdimensional NSGA-III (Punto #202)
    /// Combina Sharpe/PnL real, penalización de Drawdown, Entropía de Shannon y Coherencia Bayesiana.
    pub fn compute_nsga3_hyper_fitness(
        gross_pnl: f64,
        total_trades: usize,
        actual_fee_rate: f64,
        max_drawdown_pct: f64,
        max_dd_threshold: f64,
        decisions: &[SignalType],
        backtest_sharpe: f64,
        live_sharpe: f64,
    ) -> f64 {
        let real_pnl = Self::reality_slippage_penalty(gross_pnl, total_trades, actual_fee_rate);
        let dd_penalty = Self::drawdown_adversarial_penalty(max_drawdown_pct, max_dd_threshold);
        let reality_gap = Self::reality_gap_adversarial_score(backtest_sharpe, live_sharpe, total_trades);
        let entropy = Self::calculate_shannon_entropy(decisions);
        let entropy_boost = (1.0 + entropy * 0.1).clamp(0.5, 1.5);

        let base_fitness = if real_pnl >= 0.0 {
            real_pnl * dd_penalty * reality_gap * entropy_boost
        } else {
            // FIX #861: Dividir por entropy_boost (>1.0) para que mayor entropía reduzca la magnitud negativa de pérdida
            (real_pnl / (dd_penalty * reality_gap).clamp(0.05, 1.0)) / entropy_boost
        };

        if base_fitness.is_finite() {
            base_fitness
        } else {
            -1e6
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy() {
        let decisions = vec![SignalType::Long, SignalType::Long, SignalType::Long];
        let entropy = EntropyFitness::calculate_shannon_entropy(&decisions);
        assert_eq!(entropy, 0.0); // 100% predecible = 0 bits de entropía

        let decisions = vec![SignalType::Long, SignalType::Short, SignalType::Flat];
        let entropy = EntropyFitness::calculate_shannon_entropy(&decisions);
        assert!(entropy > 1.5); // Máxima entropía para 3 estados
    }

    #[test]
    fn test_bayesian_collapse() {
        // Un backtest del 90% WR, pero en vivo da 40% WR tras 20 trades. Debe colapsar.
        let penalty = EntropyFitness::bayesian_posterior_collapse_penalty(0.90, 0.40, 20);
        assert!(penalty < 0.10);
    }

    #[test]
    fn test_nsga3_hyper_fitness() {
        let decisions = vec![SignalType::Long, SignalType::Short, SignalType::Flat, SignalType::Long];
        let fitness = EntropyFitness::compute_nsga3_hyper_fitness(
            15.0,
            10,
            0.0004,
            0.02,
            0.05,
            &decisions,
            2.5,
            2.3,
        );
        assert!(fitness > 0.0, "Fitness should be positive for profitable low-drawdown run");
    }

    #[test]
    fn test_entropy_fitness_nan_and_infinite_immunity() {
        let decisions = vec![SignalType::Long, SignalType::Short];
        let fitness_nan = EntropyFitness::compute_nsga3_hyper_fitness(
            f64::NAN,
            10,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            &decisions,
            f64::NAN,
            f64::NAN,
        );
        assert!(fitness_nan.is_finite());

        let collapse_nan = EntropyFitness::bayesian_posterior_collapse_penalty(f64::NAN, f64::INFINITY, 50);
        assert_eq!(collapse_nan, 1.0);

        let slippage_nan = EntropyFitness::reality_slippage_penalty(f64::NAN, 10, f64::NAN);
        assert_eq!(slippage_nan, 0.0);
    }

    #[test]
    fn test_entropy_fitness_extreme_drawdown_penalty() {
        let decisions = vec![SignalType::Long, SignalType::Long];
        let fitness_dd = EntropyFitness::compute_nsga3_hyper_fitness(
            10.0,
            50,
            0.0004,
            0.50, // 50% max drawdown
            0.05,
            &decisions,
            1.5,
            0.5,
        );
        assert!(fitness_dd.is_finite());
    }
}

