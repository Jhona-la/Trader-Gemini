use std::collections::HashMap;
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

        let mut frequencies = HashMap::new();
        for &decision in decisions {
            *frequencies.entry(decision).or_insert(0) += 1;
        }

        let total = decisions.len() as f64;
        let mut entropy = 0.0;

        for count in frequencies.values() {
            let p = *count as f64 / total;
            if p > 0.0 {
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
        if total_trades == 0 {
            return gross_pnl;
        }
        // Asumimos una distribución de fricción donde cada trade tiene
        // una probabilidad Poisson de sufrir latencia/slippage de ~5bps.
        let lambda = total_trades as f64 * 0.05; // 5% of trades suffer heavy slippage
        let std_dev = lambda.sqrt();

        // FASE 22: Fricción base inevitable usa la comisión REAL del tier VIP de Binance
        let deterministic_friction = total_trades as f64 * actual_fee_rate;
        let stochastic_friction = (lambda + std_dev * 1.96) * 0.001; // Worst case 95% CI

        let real_net_pnl = gross_pnl - deterministic_friction - stochastic_friction;

        real_net_pnl
    }

    /// FASE 17: PENALIZACIÓN DE CAÍDA LIBRE (MAX DRAWDOWN ADVERSARIAL PENALTY)
    /// Castiga exponencialmente a las estrategias que acumulen un Max Drawdown
    /// superior a la tolerancia dictada por la gestión de micro-capitales.
    /// Devuelve un multiplicador entre [0.0, 1.0].
    pub fn drawdown_adversarial_penalty(max_drawdown_pct: f64, threshold_pct: f64) -> f64 {
        if max_drawdown_pct <= threshold_pct {
            return 1.0; // Drawdown aceptable, 100% retención del fitness
        }

        // Exceso sobre el umbral
        let excess = max_drawdown_pct - threshold_pct;

        // Decaimiento exponencial abrupto. Un DD de 20% multiplicaría por ~0.04 (aniquilado)
        let decay = f64::exp(-excess * 20.0);

        decay.clamp(0.01, 1.0)
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
}
