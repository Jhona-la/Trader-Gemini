use crate::entropy_fitness::EntropyFitness;
use rand::RngExt;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmaInputError {
    InvalidDimension,
    InvalidPopulationSize,
    InvalidStepSize,
    InvalidOptimizerShape,
    PopulationSizeMismatch,
    GenomeDimensionMismatch,
    InvalidGene,
    EvaluationCountMismatch,
    CandidateIndexOutOfRange,
    DuplicateCandidateIndex,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateOutcome {
    Updated,
    InsufficientValidParents,
    Rejected(CmaInputError),
}

/// Supervisor policy level and CMA search dispersion are distinct quantities.
/// This trace explains their composition; it is not evidence of profitability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepSizeAdjustment {
    pub previous_sigma: f64,
    pub sigma: f64,
    pub previous_level: f64,
    pub level: f64,
}

/// Optimizador clásico híbrido CMA/PSO sobre coordenadas normalizadas.
/// No implementa computación cuántica. Sus clamps y blanqueamiento diagonal
/// requieren validación propia; no hereda todas las invariancias de CMA canónico.
#[derive(Clone, Debug)]
pub struct CmaEsOptimizer {
    pub dimension: usize,
    pub lambda: usize,
    pub mu: usize,
    pub weights: Vec<f64>,
    pub mueff: f64,
    pub sigma: f64,
    supervisor_level: f64,
    pub mean: Vec<f64>,
    pub cov_matrix: Vec<Vec<f64>>,
    pub p_c: Vec<f64>,
    pub p_sigma: Vec<f64>,
    pub c_c: f64,
    pub c_1: f64,
    pub c_mu: f64,
    pub d_sigma: f64,
    pub generation: usize,
    // --- PSO Additions ---
    pub global_best: Vec<f64>,
    pub global_best_fitness: f64,
    pub velocities: Vec<Vec<f64>>,
    pub personal_bests: Vec<Vec<f64>>,
    pub personal_best_fitnesses: Vec<f64>,
}

/// D-387: Proyección de barrera reflectiva para variables acotadas a [lower, upper].
/// Previene el Boundary Drift del centroide de CMA-ES hacia el infinito o valores negativos no físicos.
#[inline(always)]
fn reflective_boundary(mut val: f64, lower: f64, upper: f64) -> f64 {
    if !val.is_finite() {
        return (lower + upper) * 0.5;
    }
    let range = upper - lower;
    if range <= 1e-12 {
        return lower;
    }
    let mut iterations = 0;
    while (val < lower || val > upper) && iterations < 10 {
        if val < lower {
            val = lower + (lower - val);
        }
        if val > upper {
            val = upper - (val - upper);
        }
        iterations += 1;
    }
    val.clamp(lower, upper)
}

impl CmaEsOptimizer {
    /// Compatibility constructor for statically valid parameters. Runtime
    /// configuration should use `try_new` and handle invalid input explicitly.
    pub fn new(dimension: usize, initial_sigma: f64, override_lambda: Option<usize>) -> Self {
        Self::try_new(dimension, initial_sigma, override_lambda)
            .expect("CMA requires dimension > 0, lambda >= 2 and finite positive sigma")
    }

    pub fn try_new(
        dimension: usize,
        initial_sigma: f64,
        override_lambda: Option<usize>,
    ) -> Result<Self, CmaInputError> {
        if dimension == 0 {
            return Err(CmaInputError::InvalidDimension);
        }
        if !initial_sigma.is_finite() || initial_sigma <= 0.0 {
            return Err(CmaInputError::InvalidStepSize);
        }
        let lambda =
            override_lambda.unwrap_or_else(|| 4 + (3.0 * (dimension as f64).ln()) as usize);
        if lambda < 2 {
            return Err(CmaInputError::InvalidPopulationSize);
        }
        let mu = lambda / 2;

        let mut weights = Vec::with_capacity(mu);
        let mut sum_weights = 0.0;
        let mut sum_sq_weights = 0.0;

        for i in 0..mu {
            let w = ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0);
            weights.push(w);
            sum_weights += w;
        }

        for i in 0..mu {
            weights[i] /= sum_weights;
            sum_sq_weights += weights[i] * weights[i];
        }

        let mueff = 1.0 / sum_sq_weights;

        // CMA-ES Hyperparameters
        let c_c = (4.0 + mueff / dimension as f64)
            / (dimension as f64 + 4.0 + 2.0 * mueff / dimension as f64);
        let c_sigma = (mueff + 2.0) / (dimension as f64 + mueff + 5.0);
        let c_1 = 2.0 / ((dimension as f64 + 1.3).powi(2) + mueff);
        // FIX #611: Cálculo canónico de c_mu acotado por (1.0 - c_1) sobre toda la fracción
        let c_mu = ((2.0 * (mueff - 2.0 + 1.0 / mueff).max(0.0))
            / ((dimension as f64 + 2.0).powi(2) + mueff))
            .min(1.0 - c_1)
            .max(0.0);
        // Hansen, arXiv:1604.00772, Appendix C: damping of log(sigma).
        // Missing sqrt and -1 overdamped adaptation, especially at large mu_eff/n.
        let d_sigma = 1.0
            + 2.0 * (((mueff - 1.0) / (dimension as f64 + 1.0)).sqrt() - 1.0).max(0.0)
            + c_sigma;

        let mut cov_matrix = vec![vec![0.0; dimension]; dimension];
        for i in 0..dimension {
            cov_matrix[i][i] = 1.0;
        }

        Ok(Self {
            dimension,
            lambda,
            mu,
            weights,
            mueff,
            sigma: initial_sigma,
            supervisor_level: initial_sigma,
            mean: vec![0.0; dimension], // Centered at 0 initially
            cov_matrix,
            p_c: vec![0.0; dimension],
            p_sigma: vec![0.0; dimension],
            c_c,
            c_1,
            c_mu,
            d_sigma,
            generation: 0,
            global_best: vec![0.0; dimension],
            global_best_fitness: f64::MIN,
            velocities: vec![vec![0.0; dimension]; lambda],
            personal_bests: vec![vec![0.0; dimension]; lambda],
            personal_best_fitnesses: vec![f64::MIN; lambda],
        })
    }

    /// Compose external exploration policy with learned dispersion:
    /// sigma_new = sigma_learned * level_new / level_previous.
    /// A repeated level is an exact no-op, not a restart. The supervisor's
    /// rationale (currently heuristic) must be validated separately from CSA.
    /// Reject invalid/overflowing rescaling without mutating either state.
    pub fn apply_exploration_level(
        &mut self,
        level: f64,
    ) -> Result<StepSizeAdjustment, CmaInputError> {
        if !level.is_finite() || level <= 0.0 || !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err(CmaInputError::InvalidStepSize);
        }
        let sigma = if level == self.supervisor_level {
            self.sigma
        } else {
            let scaled = self.sigma * (level / self.supervisor_level);
            if scaled.is_finite() && scaled > 0.0 {
                scaled
            } else {
                // Avoid intermediate overflow/underflow when the final value
                // is representable. True overflow/underflow is rejected below.
                (self.sigma.ln() + level.ln() - self.supervisor_level.ln()).exp()
            }
        };
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(CmaInputError::InvalidStepSize);
        }
        let adjustment = StepSizeAdjustment {
            previous_sigma: self.sigma,
            sigma,
            previous_level: self.supervisor_level,
            level,
        };
        self.sigma = sigma;
        self.supervisor_level = level;
        Ok(adjustment)
    }

    /// Samplea una población usando una técnica híbrida CMA-ES + PSO (Particle Swarm)
    pub fn sample_population(&mut self, w: f64, c1: f64, c2: f64) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        self.sample_population_with_rng(w, c1, c2, &mut rng)
    }

    /// RNG inyectable para reproducir y comparar propuestas con ruido común.
    /// La semilla no convierte la evaluación económica en evidencia independiente.
    pub fn sample_population_with_rng<R: rand::Rng + ?Sized>(
        &mut self,
        w: f64,
        c1: f64,
        c2: f64,
        rng: &mut R,
    ) -> Vec<Vec<f64>> {
        let mut population = Vec::with_capacity(self.lambda);

        // Cholesky decomposition L of covariance matrix C (L * L^T = C)
        // FIX #715: Regularización y sanitización de Cholesky contra matrices singulares
        let mut l_mat = vec![vec![0.0; self.dimension]; self.dimension];
        for i in 0..self.dimension {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l_mat[i][k] * l_mat[j][k];
                }
                if i == j {
                    let raw_val = self.cov_matrix[i][i] - sum;
                    let val = if raw_val.is_finite() { raw_val } else { 1.0 };
                    l_mat[i][j] = if val > 1e-8 { val.sqrt() } else { 1e-4 };
                } else {
                    let diag = l_mat[j][j].max(1e-8);
                    let raw_cov = self.cov_matrix[i][j];
                    let cov_val = if raw_cov.is_finite() { raw_cov } else { 0.0 };
                    let l_val = (cov_val - sum) / diag;
                    l_mat[i][j] = if l_val.is_finite() {
                        l_val.clamp(-100.0, 100.0)
                    } else {
                        0.0
                    };
                }
            }
        }

        for i in 0..self.lambda {
            // Generate standard normal vector z via Box-Muller
            let mut z_vec = Vec::with_capacity(self.dimension);
            for _ in 0..self.dimension {
                let r1 = rng.random::<f64>();
                let r2 = rng.random::<f64>();
                let val = -2.0_f64 * r1.max(1e-10).ln();
                let z = (r2 * 2.0 * PI).cos() * val.sqrt();
                z_vec.push(if z.is_finite() {
                    z.clamp(-5.0, 5.0)
                } else {
                    0.0
                });
            }

            let mut ind = Vec::with_capacity(self.dimension);
            for d in 0..self.dimension {
                // Correlated multivariate step: (L * z)[d]
                let mut cov_step = 0.0;
                for k in 0..=d {
                    cov_step += l_mat[d][k] * z_vec[k];
                }
                let safe_cov_step = if cov_step.is_finite() {
                    cov_step.clamp(-10.0, 10.0)
                } else {
                    0.0
                };
                let raw_cma = self.mean[d] + self.sigma * safe_cov_step;
                let safe_cma = if raw_cma.is_finite() {
                    raw_cma
                } else {
                    self.mean[d]
                };

                // No measured fitness means no personal/social attractor.
                // Zero-filled constructor vectors are storage, not evidence.
                let r1_pso: f64 = rng.random();
                let r2_pso: f64 = rng.random();
                let cognitive = if self.personal_best_fitnesses[i].is_finite()
                    && self.personal_best_fitnesses[i] > f64::MIN
                {
                    c1 * r1_pso * (self.personal_bests[i][d] - safe_cma)
                } else {
                    0.0
                };
                let social = if self.global_best_fitness.is_finite()
                    && self.global_best_fitness > f64::MIN
                {
                    c2 * r2_pso * (self.global_best[d] - safe_cma)
                } else {
                    0.0
                };
                let vel = w * self.velocities[i][d] + cognitive + social;
                let safe_vel = if vel.is_finite() {
                    vel.clamp(-100000.0, 100000.0)
                } else {
                    0.0
                };
                self.velocities[i][d] = safe_vel;

                // Hybridize: Base CMA sample + PSO momentum con barrera reflectiva [0.0, 1.0] (D-386 / D-387)
                let hybrid_val = reflective_boundary(safe_cma + safe_vel * 0.1, 0.0, 1.0);
                ind.push(hybrid_val);
            }
            population.push(ind);
        }
        population
    }

    /// API histórica con comparación backtest/live: el sexto componente debe
    /// ser Sharpe live comparable, nunca crecimiento de capital. El conteo de
    /// trades compartido es una limitación de esta API; no certifica evidencia.
    /// El fitness canónico ya debe incluir costes. `actual_fee_rate` se
    /// conserva por compatibilidad: el antiguo PnL auxiliar no cambiaba el
    /// factor efectivo [0.05,1], por lo que no descontaba fees del fitness.
    pub fn update(
        &mut self,
        population: &[Vec<f64>],
        fitness_scores: &mut [(usize, f64, f64, usize, f64, f64)],
        _actual_fee_rate: f64,
    ) -> UpdateOutcome {
        self.update_impl(population, fitness_scores, true)
    }

    /// Actualiza candidatos que sólo tienen evaluación de backtest. No
    /// inventa una referencia live: el sexto componente es metadata opaca
    /// del caller y se preserva, pero no interviene en la selección.
    /// Esto NO valida fuera de muestra ni autoriza promoción/despliegue.
    pub fn update_backtest_only(
        &mut self,
        population: &[Vec<f64>],
        fitness_scores: &mut [(usize, f64, f64, usize, f64, f64)],
        _actual_fee_rate: f64,
    ) -> UpdateOutcome {
        self.update_impl(population, fitness_scores, false)
    }

    /// Each generation is a complete indexed batch: exactly one evaluation
    /// per candidate. Preflight is transactional for both optimizer and scores.
    /// This validates shape/identity, NOT the fitness's economic provenance.
    fn validate_batch(
        &self,
        population: &[Vec<f64>],
        scores: &[(usize, f64, f64, usize, f64, f64)],
    ) -> Result<(), CmaInputError> {
        let n = self.dimension;
        if n == 0
            || self.lambda < 2
            || self.mu == 0
            || self.mu > self.lambda
            || self.weights.len() != self.mu
            || self.mean.len() != n
            || self.p_c.len() != n
            || self.p_sigma.len() != n
            || self.global_best.len() != n
            || self.cov_matrix.len() != n
            || self.cov_matrix.iter().any(|row| row.len() != n)
            || self.velocities.len() != self.lambda
            || self.velocities.iter().any(|row| row.len() != n)
            || self.personal_bests.len() != self.lambda
            || self.personal_bests.iter().any(|row| row.len() != n)
            || self.personal_best_fitnesses.len() != self.lambda
        {
            return Err(CmaInputError::InvalidOptimizerShape);
        }
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err(CmaInputError::InvalidStepSize);
        }
        if population.len() != self.lambda {
            return Err(CmaInputError::PopulationSizeMismatch);
        }
        for genome in population {
            if genome.len() != n {
                return Err(CmaInputError::GenomeDimensionMismatch);
            }
            if genome
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            {
                return Err(CmaInputError::InvalidGene);
            }
        }
        if scores.len() != self.lambda {
            return Err(CmaInputError::EvaluationCountMismatch);
        }
        let mut seen = vec![false; self.lambda];
        for stat in scores {
            let Some(already_seen) = seen.get_mut(stat.0) else {
                return Err(CmaInputError::CandidateIndexOutOfRange);
            };
            if *already_seen {
                return Err(CmaInputError::DuplicateCandidateIndex);
            }
            *already_seen = true;
        }
        Ok(())
    }

    fn update_impl(
        &mut self,
        population: &[Vec<f64>],
        fitness_scores: &mut [(usize, f64, f64, usize, f64, f64)],
        has_live_reference: bool,
    ) -> UpdateOutcome {
        if let Err(error) = self.validate_batch(population, fitness_scores) {
            return UpdateOutcome::Rejected(error);
        }
        // fitness_scores tuples: (index, raw_fitness, gross_pnl, num_trades, backtest_sharpe, live_sharpe)

        // Apply only an identified comparison, on the SIGNED canonical score.
        for stat in fitness_scores.iter_mut() {
            if !stat.1.is_finite()
                || !stat.2.is_finite()
                || (has_live_reference && (!stat.4.is_finite() || !stat.5.is_finite()))
            {
                stat.1 = f64::MIN;
                continue;
            }
            let retention = if has_live_reference {
                EntropyFitness::reality_gap_adversarial_score(stat.4, stat.5, stat.3)
            } else {
                1.0
            };
            // FMT-011: for 0<r<=1, f*r<=f if f>=0, and f/r<=f if f<0.
            // Use the score's sign, not the sign of a differently defined PnL.
            if stat.3 > 0 {
                stat.1 = if stat.1 >= 0.0 {
                    stat.1 * retention
                } else {
                    stat.1 / retention
                };
            } else {
                // Penalidad suave: la distancia a la media ayuda a evitar que todos tengan la misma puntuación (planicie)
                let norm: f64 = population[stat.0].iter().map(|v| v.abs()).sum();
                stat.1 = -10.0 - norm * 0.001; // Incentiva reducir la magnitud de los tensores (como umbrales altos)
            }

            if !stat.1.is_finite() {
                // A fixed -1e9 could outrank a finite score below -1e9.
                stat.1 = f64::MIN;
            }
        }

        fitness_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        // No empirical parents: leave optimizer memory and geometry untouched.
        if self.mu == 0
            || fitness_scores.len() < self.mu
            || fitness_scores.iter().take(self.mu).any(|s| s.1 == f64::MIN)
        {
            return UpdateOutcome::InsufficientValidParents;
        }

        for stat in fitness_scores.iter().filter(|s| s.1 != f64::MIN) {
            // --- PSO: Update Personal Best ---
            let idx = stat.0;
            if stat.1 > self.personal_best_fitnesses[idx] {
                self.personal_best_fitnesses[idx] = stat.1;
                self.personal_bests[idx] = population[idx].clone();
            }
        }

        // --- PSO: Update Global Best ---
        if fitness_scores[0].1 > self.global_best_fitness {
            self.global_best_fitness = fitness_scores[0].1;
            self.global_best = population[fitness_scores[0].0].clone();
        }

        let old_mean = self.mean.clone();

        // 1. Update Mean con proyección de barrera reflectiva (D-387)
        for d in 0..self.dimension {
            let mut sum_mean = 0.0;
            for i in 0..self.mu {
                let pop_idx = fitness_scores[i].0;
                sum_mean += self.weights[i] * population[pop_idx][d];
            }
            self.mean[d] = reflective_boundary(sum_mean, 0.0, 1.0);
        }

        // 2. Update Evolution Paths (Simplified for diagonal/independent variables)
        let c_sigma = (self.mueff + 2.0) / (self.dimension as f64 + self.mueff + 5.0);
        let expected_norm = (self.dimension as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * self.dimension as f64)
                + 1.0 / (21.0 * (self.dimension as f64).powi(2)));

        // FIX #670: Sanitizar y acotar sigma para evitar divergencias numéricas
        let safe_sigma = if self.sigma.is_finite() && self.sigma > 0.0 {
            self.sigma.clamp(1e-6, 100.0)
        } else {
            0.1
        };

        let mut ps_norm = 0.0;
        for d in 0..self.dimension {
            let raw_step = (self.mean[d] - old_mean[d]) / safe_sigma;
            let step = if raw_step.is_finite() {
                raw_step.clamp(-10.0, 10.0)
            } else {
                0.0
            };
            let std_dev = self.cov_matrix[d][d].sqrt().max(1e-6);

            self.p_sigma[d] = (1.0 - c_sigma) * self.p_sigma[d]
                + (c_sigma * (2.0 - c_sigma) * self.mueff).sqrt() * (step / std_dev);
            ps_norm += self.p_sigma[d] * self.p_sigma[d];
        }
        ps_norm = ps_norm.sqrt();

        self.generation += 1;
        let gen_exp = (2 * self.generation) as i32;
        let denom_sq = ((1.0 - (1.0 - c_sigma).powi(gen_exp)) * self.dimension as f64).max(1e-12);
        let h_sigma = if ps_norm / denom_sq.sqrt() < 1.4 + 2.0 / (self.dimension as f64 + 1.0) {
            1.0
        } else {
            0.0
        };

        for d in 0..self.dimension {
            let raw_step = (self.mean[d] - old_mean[d]) / safe_sigma;
            let step = if raw_step.is_finite() {
                raw_step.clamp(-10.0, 10.0)
            } else {
                0.0
            };
            self.p_c[d] = (1.0 - self.c_c) * self.p_c[d]
                + h_sigma * (self.c_c * (2.0 - self.c_c) * self.mueff).sqrt() * step;
        }

        // 3. Update Covariance Matrix C (Rank-1 + Rank-mu update across all pairs i, j)
        for i in 0..self.dimension {
            for j in 0..=i {
                let mut artmp = 0.0;
                for k in 0..self.mu {
                    let pop_idx = fitness_scores[k].0;
                    let raw_zi = (population[pop_idx][i] - old_mean[i]) / safe_sigma;
                    let raw_zj = (population[pop_idx][j] - old_mean[j]) / safe_sigma;
                    let z_i = if raw_zi.is_finite() {
                        raw_zi.clamp(-50.0, 50.0)
                    } else {
                        0.0
                    };
                    let z_j = if raw_zj.is_finite() {
                        raw_zj.clamp(-50.0, 50.0)
                    } else {
                        0.0
                    };
                    artmp += self.weights[k] * z_i * z_j;
                }
                let old_c = self.cov_matrix[i][j];
                let rank1 = self.p_c[i] * self.p_c[j]
                    + (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c) * old_c;
                let new_c =
                    (1.0 - self.c_1 - self.c_mu) * old_c + self.c_1 * rank1 + self.c_mu * artmp;

                let final_c = if i == j {
                    if new_c.is_finite() && new_c > 1e-8 {
                        new_c.clamp(1e-8, 1000.0)
                    } else {
                        1e-8
                    }
                } else {
                    if new_c.is_finite() {
                        new_c.clamp(-1000.0, 1000.0)
                    } else {
                        0.0
                    }
                };
                self.cov_matrix[i][j] = final_c;
                self.cov_matrix[j][i] = final_c;
            }
        }

        // 4. Update Step Size con clamping numérico seguro en el exponente
        let exp_term =
            ((c_sigma / self.d_sigma) * (ps_norm / expected_norm - 1.0)).clamp(-20.0, 20.0);
        self.sigma *= exp_term.exp();

        // FASE 3: Permitir expansión evolutiva libre. Solo proteger el límite bajo computacional (división por 0).
        if !self.sigma.is_finite() {
            self.sigma = 1e-6;
        } else {
            self.sigma = self.sigma.max(1e-9);
        }
        UpdateOutcome::Updated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cma_es_optimizer_sample_and_update() {
        let mut optimizer = CmaEsOptimizer::new(10, 0.2, Some(10));
        let population = optimizer.sample_population(0.7, 1.4, 1.4);
        assert_eq!(population.len(), 10);
        assert_eq!(population[0].len(), 10);

        let mut fitness_scores: Vec<(usize, f64, f64, usize, f64, f64)> = (0..10)
            .map(|i| (i, 1.0 + i as f64, 10.0 * (i + 1) as f64, 5, 2.0, 2.0))
            .collect();

        optimizer.update(&population, &mut fitness_scores, 0.0005);
        assert!(optimizer.sigma > 0.0);
        assert!(!optimizer.sigma.is_nan() && !optimizer.sigma.is_infinite());
    }

    #[test]
    fn test_cma_es_singular_matrix_recovery() {
        let mut optimizer = CmaEsOptimizer::new(5, 0.1, Some(6));
        // Forzar matriz de covarianza casi singular con ceros en la diagonal
        for i in 0..5 {
            for j in 0..5 {
                optimizer.cov_matrix[i][j] = 0.0;
            }
        }
        let pop = optimizer.sample_population(0.5, 1.0, 1.0);
        assert_eq!(pop.len(), 6);
        for ind in &pop {
            for &val in ind {
                assert!(
                    val.is_finite(),
                    "Sampled value must be finite even with singular cov matrix"
                );
            }
        }
    }

    #[test]
    fn test_cma_es_nan_fitness_recovery() {
        let mut optimizer = CmaEsOptimizer::new(4, 0.2, Some(6));
        let pop = optimizer.sample_population(0.5, 1.0, 1.0);
        let mut scores = vec![
            (0, f64::NAN, 10.0, 5, 1.0, 1.0),
            (1, f64::INFINITY, 10.0, 5, 1.0, 1.0),
            (2, 5.0, 10.0, 5, 1.0, 1.0),
            (3, -2.0, 10.0, 5, 1.0, 1.0),
            (4, 1.0, 10.0, 5, 1.0, 1.0),
            (5, 0.0, 10.0, 5, 1.0, 1.0),
        ];

        optimizer.update(&pop, &mut scores, 0.0005);
        assert!(optimizer.sigma.is_finite());
        for &m in &optimizer.mean {
            assert!(m.is_finite());
        }
    }
}
