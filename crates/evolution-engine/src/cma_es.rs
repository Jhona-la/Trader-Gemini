use std::f64::consts::PI;
use rand::Rng;
use rand::RngExt;
use crate::entropy_fitness::EntropyFitness;

/// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) con Realidad Cuántica
/// Optimiza tensores de parámetros HFT usando CMA-ES penalizando fuertemente 
/// las estrategias irrealistas mediante EntropyFitness::reality_slippage_penalty.
pub struct CmaEsOptimizer {
    pub dimension: usize,
    pub lambda: usize,
    pub mu: usize,
    pub weights: Vec<f64>,
    pub mueff: f64,
    pub sigma: f64,
    pub mean: Vec<f64>,
    pub cov_matrix: Vec<Vec<f64>>,
    pub p_c: Vec<f64>,
    pub p_sigma: Vec<f64>,
    pub c_c: f64,
    pub c_1: f64,
    pub c_mu: f64,
    pub d_sigma: f64,
    // --- PSO Additions ---
    pub global_best: Vec<f64>,
    pub global_best_fitness: f64,
    pub velocities: Vec<Vec<f64>>,
    pub personal_bests: Vec<Vec<f64>>,
    pub personal_best_fitnesses: Vec<f64>,
}

impl CmaEsOptimizer {
    pub fn new(dimension: usize, initial_sigma: f64, override_lambda: Option<usize>) -> Self {
        let lambda = override_lambda.unwrap_or_else(|| 4 + (3.0 * (dimension as f64).ln()) as usize);
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
        let c_c = (4.0 + mueff / dimension as f64) / (dimension as f64 + 4.0 + 2.0 * mueff / dimension as f64);
        let c_sigma = (mueff + 2.0) / (dimension as f64 + mueff + 5.0);
        let c_1 = 2.0 / ((dimension as f64 + 1.3).powi(2) + mueff);
        let c_mu = (2.0 * (mueff - 2.0 + 1.0 / mueff)) / ((dimension as f64 + 2.0).powi(2) + mueff).min(1.0 - c_1);
        let d_sigma = 1.0 + 2.0 * ((mueff - 1.0) / (dimension as f64 + 1.0)).max(0.0) + c_sigma;

        let mut cov_matrix = vec![vec![0.0; dimension]; dimension];
        for i in 0..dimension { cov_matrix[i][i] = 1.0; }
        
        Self {
            dimension,
            lambda,
            mu,
            weights,
            mueff,
            sigma: initial_sigma,
            mean: vec![0.0; dimension], // Centered at 0 initially
            cov_matrix,
            p_c: vec![0.0; dimension],
            p_sigma: vec![0.0; dimension],
            c_c,
            c_1,
            c_mu,
            d_sigma,
            global_best: vec![0.0; dimension],
            global_best_fitness: f64::MIN,
            velocities: vec![vec![0.0; dimension]; lambda],
            personal_bests: vec![vec![0.0; dimension]; lambda],
            personal_best_fitnesses: vec![f64::MIN; lambda],
        }
    }

    /// Samplea una población usando una técnica híbrida CMA-ES + PSO (Particle Swarm)
    pub fn sample_population(&mut self, w: f64, c1: f64, c2: f64) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        let mut population = Vec::with_capacity(self.lambda);

        for i in 0..self.lambda {
            let mut ind = Vec::with_capacity(self.dimension);
            for d in 0..self.dimension {
                // CMA-ES Sampling (Gaussian)
                let std_dev = self.cov_matrix[d][d].sqrt().max(1e-6);
                let r1 = rng.random::<f64>();
                let r2 = rng.random::<f64>();
                let val = -2.0_f64 * r1.max(1e-10).ln();
                let z = (r2 * 2.0 * PI).cos() * val.sqrt(); // Box-Muller
                let cma_sample = self.mean[d] + self.sigma * std_dev * z;

                // PSO Velocity Update
                let r1_pso: f64 = rng.random();
                let r2_pso: f64 = rng.random();
                self.velocities[i][d] = w * self.velocities[i][d] 
                    + c1 * r1_pso * (self.personal_bests[i][d] - cma_sample)
                    + c2 * r2_pso * (self.global_best[d] - cma_sample);
                
                // FASE 3: Removiendo el clamp de -5 a 5 impuesto por humanos.
                // Usamos `clamp` solo a nivel extremo computacional (-1e5, 1e5) para evitar el colapso a Infinity/NaN
                self.velocities[i][d] = self.velocities[i][d].clamp(-100000.0, 100000.0);

                // Hybridize: Base CMA sample + PSO momentum
                let hybrid_val = cma_sample + self.velocities[i][d] * 0.1; // 10% PSO influence
                ind.push(hybrid_val);
            }
            population.push(ind);
        }
        population
    }

    /// Actualiza la matriz de covarianza, media y step-size basado en la población evaluada.
    /// Incorpora penalizaciones de realidad (Slippage/Fees y Reality Gap) internamente.
    /// FASE 22: `actual_fee_rate` inyectado desde el Arena (no hardcodeado).
    pub fn update(&mut self, population: &[Vec<f64>], fitness_scores: &mut [(usize, f64, f64, usize, f64, f64)], actual_fee_rate: f64) {
        // fitness_scores tuples: (index, raw_fitness, gross_pnl, num_trades, backtest_sharpe, live_sharpe)
        
        // Aplicar Reality Slippage Penalty y Reality Gap Penalty a los PnLs
        for stat in fitness_scores.iter_mut() {
            let real_pnl = EntropyFitness::reality_slippage_penalty(stat.2, stat.3, actual_fee_rate);
            
            // FASE VII: Penalización bayesiana si el sistema colapsa en producción respecto al backtest
            let reality_gap = EntropyFitness::reality_gap_adversarial_score(stat.4, stat.5, stat.3);
            
            // FASE 17 (Tuning Genómico Profundo): No dar fitness "0" exacto si hay 0 trades.
            // Si hay 0 trades, damos una puntuación negativa que empeora a medida que
            // los umbrales del genoma (asumimos que están codificados en la población) son más restrictivos.
            // Para simplificar y no romper dependencias, si hay 0 trades penalizamos levemente
            // para que los CMA-ES bounds no colapsen a una planicie.
            if stat.3 > 0 {
                stat.1 = real_pnl * reality_gap;
            } else {
                // Penalidad suave: la distancia a la media ayuda a evitar que todos tengan la misma puntuación (planicie)
                let norm: f64 = population[stat.0].iter().map(|v| v.abs()).sum();
                stat.1 = -10.0 - norm * 0.001; // Incentiva reducir la magnitud de los tensores (como umbrales altos)
            }
            
            // --- PSO: Update Personal Best ---
            let idx = stat.0;
            if stat.1 > self.personal_best_fitnesses[idx] {
                self.personal_best_fitnesses[idx] = stat.1;
                self.personal_bests[idx] = population[idx].clone();
            }
        }
        
        // Sort population by reality-adjusted fitness (descending, higher is better)
        fitness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // --- PSO: Update Global Best ---
        if fitness_scores[0].1 > self.global_best_fitness {
            self.global_best_fitness = fitness_scores[0].1;
            self.global_best = population[fitness_scores[0].0].clone();
        }
        
        let old_mean = self.mean.clone();
        
        // 1. Update Mean
        for d in 0..self.dimension {
            self.mean[d] = 0.0;
            for i in 0..self.mu {
                let pop_idx = fitness_scores[i].0;
                self.mean[d] += self.weights[i] * population[pop_idx][d];
            }
        }
        
        // 2. Update Evolution Paths (Simplified for diagonal/independent variables)
        let c_sigma = (self.mueff + 2.0) / (self.dimension as f64 + self.mueff + 5.0);
        let expected_norm = (self.dimension as f64).sqrt() * (1.0 - 1.0 / (4.0 * self.dimension as f64) + 1.0 / (21.0 * (self.dimension as f64).powi(2)));
        
        let mut ps_norm = 0.0;
        for d in 0..self.dimension {
            let step = (self.mean[d] - old_mean[d]) / self.sigma;
            let std_dev = self.cov_matrix[d][d].sqrt().max(1e-6);
            
            self.p_sigma[d] = (1.0 - c_sigma) * self.p_sigma[d] + (c_sigma * (2.0 - c_sigma) * self.mueff).sqrt() * (step / std_dev);
            ps_norm += self.p_sigma[d] * self.p_sigma[d];
        }
        ps_norm = ps_norm.sqrt();
        
        let h_sigma = if ps_norm / ((1.0 - (1.0 - c_sigma).powi(2)) * self.dimension as f64).sqrt() < 1.4 + 2.0 / (self.dimension as f64 + 1.0) { 1.0 } else { 0.0 };
        
        for d in 0..self.dimension {
            let step = (self.mean[d] - old_mean[d]) / self.sigma;
            self.p_c[d] = (1.0 - self.c_c) * self.p_c[d] + h_sigma * (self.c_c * (2.0 - self.c_c) * self.mueff).sqrt() * step;
        }
        
        // 3. Update Covariance Matrix
        for d in 0..self.dimension {
            let mut artmp = 0.0;
            for i in 0..self.mu {
                let pop_idx = fitness_scores[i].0;
                let z = (population[pop_idx][d] - old_mean[d]) / self.sigma;
                artmp += self.weights[i] * z * z;
            }
            self.cov_matrix[d][d] = (1.0 - self.c_1 - self.c_mu) * self.cov_matrix[d][d] 
                + self.c_1 * (self.p_c[d] * self.p_c[d] + (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c) * self.cov_matrix[d][d])
                + self.c_mu * artmp * self.cov_matrix[d][d];
        }
        
        // 4. Update Step Size
        self.sigma *= ((c_sigma / self.d_sigma) * (ps_norm / expected_norm - 1.0)).exp();
        
        // FASE 3: Permitir expansión evolutiva libre. Solo proteger el límite bajo computacional (división por 0).
        if !self.sigma.is_finite() {
            self.sigma = 1e-6;
        } else {
            self.sigma = self.sigma.max(1e-9);
        }
    }
}
