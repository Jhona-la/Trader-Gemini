/// 🧬 OPERADORES GENÉTICOS AVANZADOS: CROSSOVER BLX-ALPHA Y MUTACIÓN DE CAUCHY
/// Operadores de exploración estocástica no local para optimización de genomas (#186-#190).

#[derive(Debug, Clone)]
pub struct EvolutionaryOperators {
    pub alpha_blx: f64,
    pub cauchy_scale: f64,
}

impl EvolutionaryOperators {
    pub fn new(alpha_blx: f64, cauchy_scale: f64) -> Self {
        Self {
            alpha_blx: if alpha_blx.is_finite() && alpha_blx >= 0.0 {
                alpha_blx
            } else {
                0.5
            },
            cauchy_scale: if cauchy_scale.is_finite() && cauchy_scale > 0.0 {
                cauchy_scale
            } else {
                0.05
            },
        }
    }

    /// Realiza Crossover BLX-alpha entre dos genomas continuos de igual longitud
    /// Retorna un nuevo genoma hijo explorando el hipercubo expandido por $\alpha$
    pub fn blx_alpha_crossover(&self, parent1: &[f64], parent2: &[f64], seed: u64) -> Vec<f64> {
        let len = parent1.len().min(parent2.len());
        let mut child = Vec::with_capacity(len);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;

        for i in 0..len {
            // FIX #640: Sanitizar finitud de valores de los padres
            let p1 = if parent1[i].is_finite() {
                parent1[i]
            } else {
                0.0
            };
            let p2 = if parent2[i].is_finite() {
                parent2[i]
            } else {
                0.0
            };

            let min_val = p1.min(p2);
            let max_val = p1.max(p2);
            let diff = max_val - min_val;

            let lower = min_val - self.alpha_blx * diff;
            let upper = max_val + self.alpha_blx * diff;

            // Pseudo-random number u in [0, 1) using SplitMix64
            rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z = z ^ (z >> 31);
            let u = (z as f64) / (u64::MAX as f64);

            let val = lower + u * (upper - lower);
            let safe_val = if val.is_finite() { val } else { 0.0 };
            child.push(safe_val.clamp(-10.0, 10.0));
        }

        child
    }

    /// Aplica Mutación Adaptativa de Cauchy a un genoma continuo
    /// La distribución de Cauchy genera saltos de colas pesadas para escapar de mínimos locales
    pub fn cauchy_mutate(&self, genome: &mut [f64], mutation_prob: f64, seed: u64) {
        let mut rng_state = seed ^ 0xD1B54A32D192ED03;

        for val in genome.iter_mut() {
            // FIX #717: Sanitizar valor inicial del gen
            if !val.is_finite() {
                *val = 0.0;
            }

            // Generar u1 para decidir si muta
            rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z1 = rng_state;
            z1 = (z1 ^ (z1 >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z1 = (z1 ^ (z1 >> 27)).wrapping_mul(0x94D049BB133111EB);
            z1 = z1 ^ (z1 >> 31);
            let u1 = (z1 as f64) / (u64::MAX as f64);

            if u1 < mutation_prob {
                // Generar u2 para la función inversa acumulativa de Cauchy: $X = x_0 + \gamma \tan(\pi (u - 0.5))$
                rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
                let mut z2 = rng_state;
                z2 = (z2 ^ (z2 >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z2 = (z2 ^ (z2 >> 27)).wrapping_mul(0x94D049BB133111EB);
                z2 = z2 ^ (z2 >> 31);
                // FIX #589: Acotar u2 a [0.02, 0.98] para prevenir singularidades de tangente (pasos infinitos)
                let u2 = ((z2 as f64) / (u64::MAX as f64)).clamp(0.02, 0.98);

                let raw_cauchy_step = self.cauchy_scale * (std::f64::consts::PI * (u2 - 0.5)).tan();
                let cauchy_step =
                    raw_cauchy_step.clamp(-5.0 * self.cauchy_scale, 5.0 * self.cauchy_scale);
                if cauchy_step.is_finite() {
                    let new_val = *val + cauchy_step;
                    *val = if new_val.is_finite() {
                        new_val.clamp(-10.0, 10.0)
                    } else {
                        *val
                    };
                }
            }
        }
    }
}

impl Default for EvolutionaryOperators {
    fn default() -> Self {
        Self::new(0.5, 0.05)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blx_alpha_and_cauchy_mutation() {
        let ops = EvolutionaryOperators::new(0.5, 0.1);

        let parent1 = vec![1.0, 2.0, 3.0, 4.0];
        let parent2 = vec![2.0, 3.0, 4.0, 5.0];

        let child = ops.blx_alpha_crossover(&parent1, &parent2, 42);
        assert_eq!(child.len(), 4);

        // Cada gen i debe estar dentro de [p_min - 0.5*diff, p_max + 0.5*diff] = [0.5, 2.5] etc.
        assert!(child[0] >= 0.5 && child[0] <= 2.5);
        assert!(child[3] >= 3.5 && child[3] <= 5.5);

        let mut mutated = child.clone();
        ops.cauchy_mutate(&mut mutated, 1.0, 12345); // 100% prob

        // Todos los genes mutados deben seguir siendo válidos finitos
        for g in mutated {
            assert!(g.is_finite());
            assert!(g >= -10.0 && g <= 10.0);
        }
    }

    #[test]
    fn test_crossover_mismatched_parent_lengths_and_nan_immunity() {
        let ops = EvolutionaryOperators::new(0.5, 0.1);

        let parent1 = vec![1.0, f64::NAN, 3.0];
        let parent2 = vec![2.0, 3.0]; // Mismatched length (2 vs 3)

        let child = ops.blx_alpha_crossover(&parent1, &parent2, 100);
        assert_eq!(child.len(), 2);
        for g in child {
            assert!(g.is_finite());
        }
    }
}
