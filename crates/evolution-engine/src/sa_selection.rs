//! Score-only SA selection state; no genome, replay or model side effects.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaDecision {
    pub accepted: bool,
    pub improved_best: bool,
}

pub struct SaScoreState {
    current: Option<f64>,
    best: Option<f64>,
}

impl SaScoreState {
    pub fn new() -> Self {
        Self {
            current: None,
            best: None,
        }
    }

    pub fn current_score(&self) -> Option<f64> {
        self.current
    }
    pub fn best_score(&self) -> Option<f64> {
        self.best
    }

    /// Preserve the CLI's Metropolis rule for an evaluated finite incumbent.
    /// Consume a random sample only when the candidate is not better.
    pub fn consider(
        &mut self,
        score: f64,
        temperature: f64,
        mut random: impl FnMut() -> f64,
    ) -> SaDecision {
        if !score.is_finite() {
            return SaDecision {
                accepted: false,
                improved_best: false,
            };
        }
        let accepted = match self.current {
            None => true,
            Some(current) if score > current => true,
            Some(current) => {
                let probability = std::f64::consts::E.powf((score - current) / temperature);
                random() < probability
            }
        };
        let improved_best = self.best.is_none_or(|best| score > best);
        if accepted {
            self.current = Some(score);
        }
        if improved_best {
            self.best = Some(score);
        }
        SaDecision {
            accepted,
            improved_best,
        }
    }
}

impl Default for SaScoreState {
    fn default() -> Self {
        Self::new()
    }
}
