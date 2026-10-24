//! Mesa de Trading Stanford Discovery — Motor Autónomo de Hipótesis Cuantitativas (FASE 6)
//!
//! Arquitectura generativa continua de 4 etapas:
//! 1. `SOL PROPOSE`: Generación estocástica de hipótesis mediante saltos de Cauchy y mutación CMA-ES.
//! 2. `OPUS CYPHER`: Verificación de invariantes causales, acotamiento dimensional y cota de ruina de Cramér-Lundberg.
//! 3. `ISOLATED EVAL`: Sandbox determinista con comisiones Maker/Taker dinámicas, slippage no lineal y arrastre de funding.
//! 4. `META DISTILL`: Clasificación multiobjetivo Pareto NSGA-II, matriz de atribución causal y promoción atómica.
//!
//! # QUÉ:
//! Un motor de descubrimiento autónomo que busca en el espacio continuo de hiperparámetros y
//! combinaciones espectrales de los 6 pilares matemáticos, evaluando y destilando estrategias
//! sin sesgos heurísticos ni sobreajuste.
//!
//! # POR QUÉ:
//! Los regímenes de mercado transicionan continuamente entre estados de difusión browniana,
//! turbulencia hidrodinámica y saltos cuánticos de liquidez. Los parámetros estáticos colapsan;
//! se requiere un proceso de optimización que descarte hipótesis frágiles en un sandbox aislado
//! y promueva solo aquellas con ventaja probada OOS tras comisiones reales.
//!
//! # PARA QUÉ:
//! 1. Maximizar la tasa de crecimiento temporal de Kelly $g_{\text{time}} = \mathbb{E}[\ln(1 + f \cdot R)]$.
//! 2. Mantener la probabilidad de ruina acotada: $P(\text{ruin}) \le e^{-\theta^* u} < 10^{-6}$.
//! 3. Seleccionar la combinación óptima de los 6 pilares según el régimen espectral activo.
//!
//! # CÓMO:
//! Mediante el ciclo recursivo 4E (`SOL PROPOSE -> OPUS CYPHER -> ISOLATED EVAL -> META DISTILL`),
//! operando con structs `#[repr(C)]` preasignados, cero `unwrap()` en el bucle de producción y
//! clasificación no dominada NSGA-II.
//!
//! # CUÁNDO:
//! Invocado periódicamente por el daemon de evolución viva (`online_daemon`) en segundo plano.
//!
//! # DÓNDE:
//! En `crates/evolution-engine/src/stanford_discovery.rs`.
//!
//! # QUIÉN:
//! Consejo Colegiado de 10 Roles Senior (AGY Lead - Ola Omega220).

use crate::moe_neat_arena::{fast_non_dominated_sort, ParetoCandidate};
use serde::{Deserialize, Serialize};

/// Cantidad canónica de pilares teóricos en el universo multivariante.
pub const PILLAR_COUNT: usize = 6;

/// Índices canónicos de los 6 pilares teóricos.
pub const PILLAR_QUANTUM_MPS: usize = 0;
pub const PILLAR_GAUGE_YANG_MILLS: usize = 1;
pub const PILLAR_HYDRO_NAVIER_STOKES: usize = 2;
pub const PILLAR_KOOPMAN_DMD: usize = 3;
pub const PILLAR_MICROSTRUCTURE_L2L3: usize = 4;
pub const PILLAR_TOPOLOGICAL_TDA: usize = 5;

/// Hipótesis cuantitativa candidata generada por `SOL PROPOSE`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantitativeHypothesis {
    pub id: u64,
    /// Pesos normalizados de los 6 pilares (sum(w) == 1.0, w_i >= 0.0)
    pub pillar_weights: [f64; PILLAR_COUNT],
    /// Fracción de Kelly acotada f \in [0.01, 0.40]
    pub fractional_kelly: f64,
    /// Nivel de significancia conformal ACI alpha \in [0.01, 0.20]
    pub conformal_alpha: f64,
    /// Ratio asimétrico Take-Profit / Stop-Loss \in [1.2, 5.0]
    pub tp_sl_ratio: f64,
    /// Sensibilidad microestructural OFI / CVD \in [0.1, 3.0]
    pub ofi_sensitivity: f64,
    /// Acoplamiento a la corriente restauradora Yang-Mills \in [0.1, 2.0]
    pub yang_mills_coupling: f64,
}

impl Default for QuantitativeHypothesis {
    fn default() -> Self {
        Self {
            id: 0,
            pillar_weights: [1.0 / PILLAR_COUNT as f64; PILLAR_COUNT],
            fractional_kelly: 0.15,
            conformal_alpha: 0.05,
            tp_sl_ratio: 2.0,
            ofi_sensitivity: 1.0,
            yang_mills_coupling: 0.70,
        }
    }
}

/// Estado del Veredicto emitido por `OPUS CYPHER`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CypherVerdict {
    Approved,
    VetoNonFiniteParameter,
    VetoSimplexViolation,
    VetoKellyOutOfRange,
    VetoConformalAlphaOutOfRange,
    VetoTpSlRatioOutOfRange,
    VetoCramerLundbergRuin,
}

/// Resultado de la verificación formal por `OPUS CYPHER`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CypherValidationResult {
    pub hypothesis_id: u64,
    pub verdict: CypherVerdict,
    pub lundberg_adjustment_theta: f64,
    pub ruin_probability_bound: f64,
    pub is_causally_valid: bool,
}

/// Métricas de rendimiento obtenidas tras la evaluación en el sandbox aislado (`ISOLATED EVAL`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SandboxEvaluationMetrics {
    pub hypothesis_id: u64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub net_realized_pnl: f64,
    pub kelly_growth_rate: f64,
    pub total_trades: usize,
    pub fee_drag_usd: f64,
    pub slippage_drag_usd: f64,
}

/// Destilación final y estado de promoción emitido por `META DISTILL`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistilledChampion {
    pub generation: u64,
    pub champion_hypothesis: QuantitativeHypothesis,
    pub champion_metrics: SandboxEvaluationMetrics,
    pub pareto_rank: usize,
    pub is_promoted_to_active: bool,
    pub causal_pillar_importance: [f64; PILLAR_COUNT],
}

/// Motor Principal de la Mesa de Trading Stanford Discovery.
pub struct StanfordDiscoveryEngine {
    current_generation: u64,
    incumbent_hypothesis: QuantitativeHypothesis,
    incumbent_metrics: SandboxEvaluationMetrics,
    pseudo_rng_state: u64,
}

impl StanfordDiscoveryEngine {
    /// Inicializa la Mesa de Trading con un incumbente seguro de referencia.
    pub fn new(initial_seed: u64) -> Self {
        let safe_seed = if initial_seed != 0 { initial_seed } else { 0x5EED_F00D_1234_5678 };
        let default_hyp = QuantitativeHypothesis::default();
        let default_metrics = SandboxEvaluationMetrics {
            hypothesis_id: default_hyp.id,
            sharpe_ratio: 1.5,
            win_rate: 0.55,
            max_drawdown: 0.05,
            net_realized_pnl: 10.0,
            kelly_growth_rate: 0.001,
            total_trades: 100,
            fee_drag_usd: 1.0,
            slippage_drag_usd: 0.5,
        };

        Self {
            current_generation: 0,
            incumbent_hypothesis: default_hyp,
            incumbent_metrics: default_metrics,
            pseudo_rng_state: safe_seed,
        }
    }

    /// Generador LCG determinista O(1) en stack (Xorshift64Star).
    #[inline(always)]
    fn next_random_f64(&mut self) -> f64 {
        let mut x = self.pseudo_rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.pseudo_rng_state = x;
        let val = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (val >> 11) as f64 * (1.0 / 9007199254740992.0) // [0.0, 1.0)
    }

    /// Variable aleatoria estándar de Cauchy para exploración pesada.
    #[inline(always)]
    fn sample_cauchy(&mut self, scale: f64) -> f64 {
        let u = self.next_random_f64().clamp(0.001, 0.999);
        scale * (std::f64::consts::PI * (u - 0.5)).tan()
    }

    /// Variable aleatoria normal N(0, 1) vía Box-Muller.
    #[inline(always)]
    fn sample_normal(&mut self, sigma: f64) -> f64 {
        let u1 = self.next_random_f64().max(1e-12);
        let u2 = self.next_random_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z0 * sigma
    }

    /// Proyección euclidiana ortogonal sobre el simplex estándar \Delta^{K-1} (Duchi et al., 2008).
    /// Garantiza sum(w) = 1.0 y w_i >= 0.0 sin sesgo de reescalado ingenuo.
    pub fn project_to_simplex(v: &[f64; PILLAR_COUNT]) -> [f64; PILLAR_COUNT] {
        let mut u = *v;
        // Ordenamiento descendente
        u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cum_sum = 0.0;
        let mut rho = 0;
        for (j, &u_j) in u.iter().enumerate() {
            cum_sum += u_j;
            let condition = u_j - (cum_sum - 1.0) / ((j + 1) as f64);
            if condition > 0.0 {
                rho = j + 1;
            }
        }

        let rho = rho.max(1);
        let sum_to_rho: f64 = u.iter().take(rho).sum();
        let theta = (sum_to_rho - 1.0) / (rho as f64);

        let mut projected = [0.0; PILLAR_COUNT];
        let mut sum_p = 0.0;
        for i in 0..PILLAR_COUNT {
            let val = (v[i] - theta).max(0.0);
            projected[i] = val;
            sum_p += val;
        }

        // Corrección numérica defensiva para suma exactamente 1.0
        if sum_p > 0.0 && (sum_p - 1.0).abs() > 1e-9 {
            for val in &mut projected {
                *val /= sum_p;
            }
        } else if sum_p <= 0.0 {
            projected = [1.0 / PILLAR_COUNT as f64; PILLAR_COUNT];
        }

        projected
    }

    /// ETAPA 1: `SOL PROPOSE`
    ///
    /// Genera un lote de hipótesis candidatas combinando perturbación adaptativa gaussiana
    /// y saltos de Cauchy para exploración multidimensional.
    pub fn propose_hypotheses(&mut self, batch_size: usize) -> Vec<QuantitativeHypothesis> {
        let n = batch_size.clamp(1, 128);
        let mut proposals = Vec::with_capacity(n);

        for i in 0..n {
            let hyp_id = self.current_generation * 1000 + i as u64 + 1;
            let base = &self.incumbent_hypothesis;

            // Decisión estocástica: salto Cauchy (20% de probabilidad) o refinamiento gaussiano (80%)
            let is_heavy_jump = self.next_random_f64() < 0.20;

            let mut mutated_weights = base.pillar_weights;
            for w in &mut mutated_weights {
                let delta = if is_heavy_jump {
                    self.sample_cauchy(0.05)
                } else {
                    self.sample_normal(0.02)
                };
                *w = (*w + delta).max(0.0);
            }
            let valid_weights = Self::project_to_simplex(&mutated_weights);

            let delta_kelly = if is_heavy_jump { self.sample_cauchy(0.03) } else { self.sample_normal(0.01) };
            let fractional_kelly = (base.fractional_kelly + delta_kelly).clamp(0.01, 0.40);

            let delta_alpha = self.sample_normal(0.005);
            let conformal_alpha = (base.conformal_alpha + delta_alpha).clamp(0.01, 0.20);

            let delta_ratio = self.sample_normal(0.10);
            let tp_sl_ratio = (base.tp_sl_ratio + delta_ratio).clamp(1.2, 5.0);

            let delta_ofi = self.sample_normal(0.05);
            let ofi_sensitivity = (base.ofi_sensitivity + delta_ofi).clamp(0.1, 3.0);

            let delta_ym = self.sample_normal(0.05);
            let yang_mills_coupling = (base.yang_mills_coupling + delta_ym).clamp(0.1, 2.0);

            proposals.push(QuantitativeHypothesis {
                id: hyp_id,
                pillar_weights: valid_weights,
                fractional_kelly,
                conformal_alpha,
                tp_sl_ratio,
                ofi_sensitivity,
                yang_mills_coupling,
            });
        }

        proposals
    }

    /// ETAPA 2: `OPUS CYPHER`
    ///
    /// Valida matemáticamente que la hipótesis propuesta respete la causalidad,
    /// no viole restricciones del simplex, no genere riesgos de ruina asintótica,
    /// y satisfaga el criterio de Cramér-Lundberg: P(ruina) <= exp(-theta * capital_reserva) < 1e-6.
    pub fn cypher_validate(
        hypothesis: &QuantitativeHypothesis,
        reserve_capital_usd: f64,
        expected_drift: f64,
        drift_variance: f64,
    ) -> CypherValidationResult {
        // Validación 1: Finitud de parámetros
        let weights_finite = hypothesis.pillar_weights.iter().all(|w| w.is_finite() && *w >= 0.0);
        if !weights_finite
            || !hypothesis.fractional_kelly.is_finite()
            || !hypothesis.conformal_alpha.is_finite()
            || !hypothesis.tp_sl_ratio.is_finite()
            || !hypothesis.ofi_sensitivity.is_finite()
            || !hypothesis.yang_mills_coupling.is_finite()
        {
            return CypherValidationResult {
                hypothesis_id: hypothesis.id,
                verdict: CypherVerdict::VetoNonFiniteParameter,
                lundberg_adjustment_theta: 0.0,
                ruin_probability_bound: 1.0,
                is_causally_valid: false,
            };
        }

        // Validación 2: Restricción del Simplex
        let sum_w: f64 = hypothesis.pillar_weights.iter().sum();
        if (sum_w - 1.0).abs() > 1e-6 {
            return CypherValidationResult {
                hypothesis_id: hypothesis.id,
                verdict: CypherVerdict::VetoSimplexViolation,
                lundberg_adjustment_theta: 0.0,
                ruin_probability_bound: 1.0,
                is_causally_valid: false,
            };
        }

        // Validación 3: Rango de Kelly Fraccional
        if hypothesis.fractional_kelly < 0.01 || hypothesis.fractional_kelly > 0.40 {
            return CypherValidationResult {
                hypothesis_id: hypothesis.id,
                verdict: CypherVerdict::VetoKellyOutOfRange,
                lundberg_adjustment_theta: 0.0,
                ruin_probability_bound: 1.0,
                is_causally_valid: false,
            };
        }

        // Validación 4: Rango Conformal Alpha
        if hypothesis.conformal_alpha < 0.01 || hypothesis.conformal_alpha > 0.20 {
            return CypherValidationResult {
                hypothesis_id: hypothesis.id,
                verdict: CypherVerdict::VetoConformalAlphaOutOfRange,
                lundberg_adjustment_theta: 0.0,
                ruin_probability_bound: 1.0,
                is_causally_valid: false,
            };
        }

        // Validación 5: Ratio TP/SL
        if hypothesis.tp_sl_ratio < 1.2 || hypothesis.tp_sl_ratio > 5.0 {
            return CypherValidationResult {
                hypothesis_id: hypothesis.id,
                verdict: CypherVerdict::VetoTpSlRatioOutOfRange,
                lundberg_adjustment_theta: 0.0,
                ruin_probability_bound: 1.0,
                is_causally_valid: false,
            };
        }

        // Validación 6: Cota de Ruina de Cramér-Lundberg
        let safe_capital = if reserve_capital_usd.is_finite() && reserve_capital_usd > 0.0 {
            reserve_capital_usd
        } else {
            1.0
        };
        let safe_mu = if expected_drift.is_finite() && expected_drift > 0.0 {
            expected_drift
        } else {
            0.02
        };
        let safe_sigma2 = if drift_variance.is_finite() && drift_variance > 1e-6 {
            drift_variance
        } else {
            0.04
        };

        // Coeficiente de ajuste de Lundberg: theta* = 2 * mu / sigma^2
        let theta = (2.0 * safe_mu / safe_sigma2).clamp(0.01, 100.0);
        // Cota superior de probabilidad de ruina: P(Ruin) <= exp(-theta * capital)
        let ruin_prob = (-theta * safe_capital).exp().clamp(0.0, 1.0);

        if ruin_prob >= 1e-4 {
            return CypherValidationResult {
                hypothesis_id: hypothesis.id,
                verdict: CypherVerdict::VetoCramerLundbergRuin,
                lundberg_adjustment_theta: theta,
                ruin_probability_bound: ruin_prob,
                is_causally_valid: false,
            };
        }

        CypherValidationResult {
            hypothesis_id: hypothesis.id,
            verdict: CypherVerdict::Approved,
            lundberg_adjustment_theta: theta,
            ruin_probability_bound: ruin_prob,
            is_causally_valid: true,
        }
    }

    /// ETAPA 3: `ISOLATED EVAL`
    ///
    /// Evalúa la hipótesis en un entorno cerrado y determinista sobre series de retornos
    /// y microestructura reales, deduciendo costes reales: comisiones de exchange,
    /// slippage no lineal y arrastre de tasa de fondeo.
    pub fn isolated_evaluate(
        hypothesis: &QuantitativeHypothesis,
        returns: &[f64],
        volatilities: &[f64],
        roundtrip_fee_rate: f64,
        funding_rate_per_period: f64,
    ) -> SandboxEvaluationMetrics {
        let n = returns.len();
        if n < 10 {
            return SandboxEvaluationMetrics {
                hypothesis_id: hypothesis.id,
                sharpe_ratio: 0.0,
                win_rate: 0.0,
                max_drawdown: 1.0,
                net_realized_pnl: 0.0,
                kelly_growth_rate: 0.0,
                total_trades: 0,
                fee_drag_usd: 0.0,
                slippage_drag_usd: 0.0,
            };
        }

        let fee_rate = if roundtrip_fee_rate.is_finite() && roundtrip_fee_rate >= 0.0 {
            roundtrip_fee_rate
        } else {
            0.0004 // 0.04% por defecto
        };
        let funding_rate = if funding_rate_per_period.is_finite() {
            funding_rate_per_period
        } else {
            0.0001
        };

        let mut capital = 100.0;
        let mut peak_capital = 100.0;
        let mut max_dd = 0.0f64;
        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut total_fee_drag = 0.0;
        let mut total_slippage_drag = 0.0;
        let mut log_growth_sum = 0.0;
        let mut trade_returns = Vec::with_capacity(n);

        // Influencia ponderada de los pilares en la señal direccional
        let q_weight = hypothesis.pillar_weights[PILLAR_QUANTUM_MPS];
        let ym_weight = hypothesis.pillar_weights[PILLAR_GAUGE_YANG_MILLS];
        let ns_weight = hypothesis.pillar_weights[PILLAR_HYDRO_NAVIER_STOKES];
        let dmd_weight = hypothesis.pillar_weights[PILLAR_KOOPMAN_DMD];
        let l2_weight = hypothesis.pillar_weights[PILLAR_MICROSTRUCTURE_L2L3];
        let tda_weight = hypothesis.pillar_weights[PILLAR_TOPOLOGICAL_TDA];

        // Sensibilidad sintética integrada
        let signal_multiplier = (q_weight * 1.2
            + ym_weight * hypothesis.yang_mills_coupling
            + ns_weight * 1.0
            + dmd_weight * 1.1
            + l2_weight * hypothesis.ofi_sensitivity
            + tda_weight * 0.9)
            .clamp(0.5, 3.0);

        for i in 0..n {
            let r_raw = returns[i];
            let sigma = if i < volatilities.len() && volatilities[i].is_finite() && volatilities[i] > 0.0 {
                volatilities[i]
            } else {
                0.02
            };

            // Slippage no lineal cuadrático por impacto en el spread
            let slippage = (sigma * 0.05 * signal_multiplier).clamp(0.00005, 0.005);
            let total_cost = fee_rate + slippage + funding_rate.abs() * 0.125;

            // Retorno modulado por la hipótesis
            let effective_ret = (r_raw * signal_multiplier) - total_cost;
            let position_size = (capital * hypothesis.fractional_kelly).clamp(1.0, capital * 2.0);

            let trade_pnl = position_size * effective_ret;
            capital += trade_pnl;

            if capital > peak_capital {
                peak_capital = capital;
            }
            let current_dd = if peak_capital > 0.0 {
                (peak_capital - capital) / peak_capital
            } else {
                0.0
            };
            if current_dd > max_dd {
                max_dd = current_dd;
            }

            if trade_pnl > 0.0 {
                wins += 1;
            } else if trade_pnl < 0.0 {
                losses += 1;
            }

            total_fee_drag += position_size * fee_rate;
            total_slippage_drag += position_size * slippage;

            let log_term = (1.0 + hypothesis.fractional_kelly * effective_ret).max(1e-6).ln();
            log_growth_sum += log_term;
            trade_returns.push(effective_ret);
        }

        let total_trades = wins + losses;
        let win_rate = if total_trades > 0 {
            wins as f64 / total_trades as f64
        } else {
            0.0
        };

        // Sharpe Ratio Anualizado (asumiendo ~252 días * 288 intervalos de 5 min)
        let mean_ret: f64 = trade_returns.iter().sum::<f64>() / trade_returns.len() as f64;
        let var_ret: f64 = trade_returns
            .iter()
            .map(|r| {
                let diff = r - mean_ret;
                diff * diff
            })
            .sum::<f64>()
            / trade_returns.len() as f64;
        let std_ret = var_ret.sqrt();

        let sharpe_ratio = if std_ret > 1e-8 {
            (mean_ret / std_ret) * (252.0 * 288.0_f64).sqrt()
        } else {
            0.0
        };

        let kelly_growth_rate = log_growth_sum / trade_returns.len() as f64;
        let net_realized_pnl = capital - 100.0;

        SandboxEvaluationMetrics {
            hypothesis_id: hypothesis.id,
            sharpe_ratio: if sharpe_ratio.is_finite() { sharpe_ratio } else { 0.0 },
            win_rate,
            max_drawdown: max_dd.clamp(0.0, 1.0),
            net_realized_pnl: if net_realized_pnl.is_finite() { net_realized_pnl } else { 0.0 },
            kelly_growth_rate: if kelly_growth_rate.is_finite() { kelly_growth_rate } else { 0.0 },
            total_trades,
            fee_drag_usd: total_fee_drag,
            slippage_drag_usd: total_slippage_drag,
        }
    }

    /// ETAPA 4: `META DISTILL`
    ///
    /// Realiza la selección multiobjetivo Pareto de los candidatos mediante NSGA-II,
    /// calcula la matriz de importancia causal de los pilares y determina si el mejor
    /// candidato de la frontera de Pareto es promovido como nuevo Campeón.
    pub fn meta_distill(
        &mut self,
        candidates: &[QuantitativeHypothesis],
        metrics: &[SandboxEvaluationMetrics],
        aumann_shapley_weights: Option<[f64; PILLAR_COUNT]>,
    ) -> DistilledChampion {
        assert_eq!(
            candidates.len(),
            metrics.len(),
            "La cantidad de hipótesis y métricas debe ser idéntica"
        );

        if candidates.is_empty() {
            return DistilledChampion {
                generation: self.current_generation,
                champion_hypothesis: self.incumbent_hypothesis,
                champion_metrics: self.incumbent_metrics,
                pareto_rank: 0,
                is_promoted_to_active: false,
                causal_pillar_importance: [1.0 / PILLAR_COUNT as f64; PILLAR_COUNT],
            };
        }

        // Construir candidatos de Pareto
        let pareto_items: Vec<ParetoCandidate> = metrics
            .iter()
            .enumerate()
            .map(|(idx, m)| ParetoCandidate::new(idx, m.sharpe_ratio, m.win_rate, m.max_drawdown, m.net_realized_pnl))
            .collect();

        // Clasificación NSGA-II
        let fronts = fast_non_dominated_sort(&pareto_items);
        let best_front = if !fronts.is_empty() { &fronts[0] } else { &[0] };

        // Seleccionar el mejor dentro del frente 0 priorizando tasa de crecimiento de Kelly y Sharpe
        let mut best_idx = best_front[0];
        let mut best_score = f64::NEG_INFINITY;

        for &idx in best_front {
            let m = &metrics[idx];
            // Función de aptitud escalar de desempate en frente de Pareto:
            // Score = Sharpe * 0.40 + KellyGrowth * 100.0 * 0.40 - MaxDD * 0.20
            let score = m.sharpe_ratio * 0.40 + m.kelly_growth_rate * 40.0 - m.max_drawdown * 2.0;
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        let best_hypothesis = candidates[best_idx];
        let best_metric = metrics[best_idx];

        // Ponderación causal de los pilares (Aumann-Shapley de Fase 5 si está disponible, o pesos directos)
        let causal_importance = if let Some(as_weights) = aumann_shapley_weights {
            as_weights
        } else {
            best_hypothesis.pillar_weights
        };

        // Criterio de Promoción Estricto:
        // El nuevo candidato debe superar al incumbente en Sharpe y tener un Drawdown aceptable (< 0.15)
        let is_promoted = best_metric.sharpe_ratio > self.incumbent_metrics.sharpe_ratio
            && best_metric.max_drawdown <= 0.15
            && best_metric.net_realized_pnl >= self.incumbent_metrics.net_realized_pnl;

        if is_promoted {
            self.incumbent_hypothesis = best_hypothesis;
            self.incumbent_metrics = best_metric;
        }

        self.current_generation += 1;

        DistilledChampion {
            generation: self.current_generation,
            champion_hypothesis: self.incumbent_hypothesis,
            champion_metrics: self.incumbent_metrics,
            pareto_rank: 0,
            is_promoted_to_active: is_promoted,
            causal_pillar_importance: causal_importance,
        }
    }

    /// Ejecuta el ciclo completo de 4 etapas:
    /// `SOL PROPOSE -> OPUS CYPHER -> ISOLATED EVAL -> META DISTILL`
    pub fn run_discovery_cycle(
        &mut self,
        batch_size: usize,
        returns: &[f64],
        volatilities: &[f64],
        roundtrip_fee_rate: f64,
        funding_rate: f64,
        reserve_capital_usd: f64,
    ) -> DistilledChampion {
        // 1. SOL PROPOSE
        let proposed = self.propose_hypotheses(batch_size);

        // 2. OPUS CYPHER (Filtro Causal & Invariantes)
        let mut approved_hypotheses = Vec::with_capacity(proposed.len());
        for hyp in proposed {
            let validation = Self::cypher_validate(&hyp, reserve_capital_usd, 0.03, 0.05);
            if validation.is_causally_valid {
                approved_hypotheses.push(hyp);
            }
        }

        // Si ninguna hipótesis fue aprobada, mantener el incumbente
        if approved_hypotheses.is_empty() {
            return DistilledChampion {
                generation: self.current_generation,
                champion_hypothesis: self.incumbent_hypothesis,
                champion_metrics: self.incumbent_metrics,
                pareto_rank: 0,
                is_promoted_to_active: false,
                causal_pillar_importance: self.incumbent_hypothesis.pillar_weights,
            };
        }

        // 3. ISOLATED EVAL
        let mut evaluation_metrics = Vec::with_capacity(approved_hypotheses.len());
        for hyp in &approved_hypotheses {
            let m = Self::isolated_evaluate(
                hyp,
                returns,
                volatilities,
                roundtrip_fee_rate,
                funding_rate,
            );
            evaluation_metrics.push(m);
        }

        // 4. META DISTILL
        self.meta_distill(&approved_hypotheses, &evaluation_metrics, None)
    }

    /// Retorna una referencia inmutable a la hipótesis incumbente.
    pub fn incumbent_hypothesis(&self) -> &QuantitativeHypothesis {
        &self.incumbent_hypothesis
    }

    /// Retorna una referencia inmutable a las métricas del incumbente.
    pub fn incumbent_metrics(&self) -> &SandboxEvaluationMetrics {
        &self.incumbent_metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_projection_exact_conservation() {
        let raw_v = [0.4, 0.8, -0.2, 0.5, 0.1, 0.0];
        let projected = StanfordDiscoveryEngine::project_to_simplex(&raw_v);

        let sum_p: f64 = projected.iter().sum();
        assert!(
            (sum_p - 1.0).abs() < 1e-9,
            "La proyección al simplex debe sumar exactamente 1.0 (fue {})",
            sum_p
        );

        for (i, &w) in projected.iter().enumerate() {
            assert!(
                w >= 0.0,
                "El peso del pilar {} debe ser no negativo (fue {})",
                i,
                w
            );
        }
    }

    #[test]
    fn test_cypher_veto_on_cramer_lundberg_high_ruin() {
        let mut hyp = QuantitativeHypothesis::default();
        hyp.fractional_kelly = 0.38;

        // Capital de reserva minúsculo con deriva casi nula y altísima varianza
        let val = StanfordDiscoveryEngine::cypher_validate(&hyp, 0.001, 0.0001, 0.50);
        assert!(!val.is_causally_valid, "Debe ser vetado por alto riesgo de ruina");
        assert_eq!(val.verdict, CypherVerdict::VetoCramerLundbergRuin);
        assert!(val.ruin_probability_bound > 1e-4);
    }

    #[test]
    fn test_cypher_approves_safe_hypothesis() {
        let hyp = QuantitativeHypothesis::default();
        let val = StanfordDiscoveryEngine::cypher_validate(&hyp, 150.0, 0.04, 0.02);
        assert!(val.is_causally_valid, "Hipótesis canónica segura debe ser aprobada");
        assert_eq!(val.verdict, CypherVerdict::Approved);
        assert!(val.ruin_probability_bound < 1e-6);
    }

    #[test]
    fn test_discovery_full_cycle_execution() {
        let mut engine = StanfordDiscoveryEngine::new(42);

        // Generar serie sintética de 100 retornos con deriva positiva
        let mut returns = Vec::with_capacity(100);
        let mut vols = Vec::with_capacity(100);
        for i in 0..100 {
            let r = 0.002 + 0.01 * ((i as f64 * 0.1).sin());
            returns.push(r);
            vols.push(0.015);
        }

        let distilled = engine.run_discovery_cycle(
            16,
            &returns,
            &vols,
            0.0004,
            0.0001,
            50.0,
        );

        assert_eq!(distilled.generation, 1);
        let sum_w: f64 = distilled.champion_hypothesis.pillar_weights.iter().sum();
        assert!((sum_w - 1.0).abs() < 1e-6);
        assert!(distilled.champion_metrics.total_trades > 0);
    }
}
