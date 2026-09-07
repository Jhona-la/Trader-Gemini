//! R4.3 — Conformal real (split-conformal sobre ventana deslizante).
//!
//! ANTES: `conformal_p_value` era la constante 0.95 escrita cada tick — el
//! "filtro conformal" era una tautología (0.95 >= 1-alpha siempre) y el gen
//! `conformal_alpha` no tenía efecto sobre nada.
//!
//! NOW: p-valor conformal estándar. No-conformidad de una observación con
//! probabilidad de modelo `p_hat` (convención: prob de que el trade resulte
//! ganador en la dirección operada) y realización `won`:
//!     score = 1 - p_hat   si won        (el modelo falló su confianza)
//!     score = p_hat       si !won
//! (score bajo = el modelo asignó alta probabilidad a lo que ocurrió).
//!
//! p-valor de una nueva predicción `p_hat_new` (clase predicha: win):
//!     p = (1 + #{score_i >= score_new}) / (n + 1)
//! Es la garantía clásica de cobertura: con umbral p >= 1-alpha, la cobertura
//! empírica converge a 1-alpha bajo intercambiabilidad.
//!
//! FUENTE DEL DATO: cada cierre de trade alimenta el calibrador con el
//! `ml_prediction` ALMACENADO EN LA POSICIÓN al abrir (no el prob actual) y
//! su resultado neto de fees. Warmup: con <30 observaciones devuelve 1.0
//! (fail-open documentado — el filtro no bloquea sin evidencia, pero deja de
//! mentir con una constante).

use std::collections::VecDeque;

pub struct ConformalCalibrator {
    scores: VecDeque<f64>,
    window: usize,
    min_calibration: usize,
}

impl ConformalCalibrator {
    pub fn new() -> Self {
        Self {
            scores: VecDeque::with_capacity(256),
            window: 200,
            min_calibration: 30,
        }
    }

    #[inline]
    fn nonconformity(p_hat: f64, won: bool) -> f64 {
        let p = if p_hat.is_finite() { p_hat.clamp(0.0, 1.0) } else { 0.5 };
        if won {
            1.0 - p
        } else {
            p
        }
    }

    /// Alimenta la ventana de calibración con un par predicción/realización.
    pub fn update(&mut self, p_hat: f64, won: bool) {
        let s = Self::nonconformity(p_hat, won);
        if self.scores.len() == self.window {
            self.scores.pop_front();
        }
        self.scores.push_back(s);
    }

    /// p-valor conformal para una nueva predicción (clase predicha: win).
    /// Fail-open (1.0) durante el warmup — documentado, no un bug.
    pub fn p_value(&self, p_hat_new: f64) -> f64 {
        if self.scores.len() < self.min_calibration {
            return 1.0;
        }
        let s_new = Self::nonconformity(p_hat_new, true);
        let ge = self.scores.iter().filter(|&&s| s >= s_new).count();
        (1.0 + ge as f64) / (self.scores.len() as f64 + 1.0)
    }

    pub fn observations(&self) -> usize {
        self.scores.len()
    }
}

impl Default for ConformalCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r43_warmup_fail_open() {
        let c = ConformalCalibrator::new();
        assert_eq!(c.p_value(0.9), 1.0, "sin calibración suficiente: fail-open");
    }

    #[test]
    fn test_r43_calibrated_p_value_bounds_and_sensitivity() {
        let mut c = ConformalCalibrator::new();
        // 40 observaciones donde el modelo fue confiable y acertó,
        // 10 donde fue confiable y falló.
        for _ in 0..40 {
            c.update(0.9, true);
        }
        for _ in 0..10 {
            c.update(0.9, false);
        }
        let p_high = c.p_value(0.9);
        let p_low = c.p_value(0.3);
        assert!(p_high > 0.0 && p_high <= 1.0);
        // Predicción poco confiable (score alto 0.7) vs ventana dominada por
        // scores 0.1: pocos scores >= 0.7 -> p-valor BAJO (el filtro rechaza).
        assert!(p_low < p_high, "menos confianza -> p-valor menor (más rechazo)");
        // Cobertura aproximada: p(0.9) ~ (1 + #{s>=0.1})/51 ~ 1.0 (todos),
        // p(0.3) ~ (1 + 0)/51 ~ 0.02.
        assert!(p_high > 0.9, "confianza calibrada debe pasar el filtro");
        assert!(p_low < 0.25, "desconfianza debe ser rechazada (p ~ 0.216 = 11/51)");
    }

    #[test]
    fn test_r43_window_sliding() {
        let mut c = ConformalCalibrator::new();
        for i in 0..300 {
            c.update(0.5, i % 2 == 0);
        }
        assert_eq!(c.observations(), 200, "ventana deslizante acotada");
    }
}
