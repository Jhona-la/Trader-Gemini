//! R4.3 — Conformal real (split-conformal sobre ventana deslizante).
//!
//! No-conformidad de una observación con probabilidad de modelo `p_hat`
//! (convención: probabilidad de que el trade resulte ganador en la dirección
//! operada) y realización `won`:
//!
//! ```text
//! score = 1 − p_hat   si ganó     (el modelo falló su confianza)
//! score = p_hat       si perdió
//! ```
//!
//! p-valor de la etiqueta `y` para una nueva predicción:
//!
//! ```text
//! p_y = (1 + #{score_i ≥ score_nuevo(y)}) / (n + 1)
//! ```
//!
//! FUENTE DEL DATO: cada cierre de trade alimenta el calibrador con el
//! `ml_prediction` almacenado en la posición AL ABRIR y su resultado neto de
//! comisiones. Con menos de `min_calibration` observaciones el calibrador
//! acepta (fail-open documentado: sin evidencia no bloquea, pero no miente con
//! una constante).
//!
//! # D-618 (DÉCIMA OLA) — la regla de decisión estaba invertida
//!
//! El filtro exigía `p_gana ≥ 1 − α`, y este mismo docstring afirmaba que esa
//! regla garantiza cobertura 1 − α. No es así. La cobertura 1 − α la tiene el
//! CONJUNTO DE PREDICCIÓN que incluye cada etiqueta con `p > α`. Exigir
//! `p ≥ 1 − α` pide que la no-conformidad de la predicción esté en el α-cuantil
//! inferior de la historia —«operar sólo en el decil más confiado»—: un filtro
//! heurístico sin ninguna garantía.
//!
//! Regla adoptada, predicción selectiva conformal: se opera cuando el conjunto
//! al nivel α es exactamente {gana}.
//!
//! ```text
//! p_gana > α   y   p_pierde ≤ α
//! ```
//!
//! «Ganar» es plausible y «perder» no lo es, con la tasa de error controlada
//! por α. Si ambas etiquetas son plausibles el calibrador se abstiene.
//!
//! # D-617 (DÉCIMA OLA) — intercambiabilidad rota bajo cambio de régimen
//!
//! La garantía exige que calibración y predicción sean intercambiables. Con un
//! modelo que se reentrena y se sustituye en caliente y un mercado
//! heterocedástico no lo son. Se adopta ACI (Adaptive Conformal Inference,
//! Gibbs & Candès, 2021): el nivel efectivo se corrige online
//!
//! ```text
//! α_{t+1} = α_t + γ · (α_objetivo − err_t)
//! ```
//!
//! con `err_t = 1` si el resultado realizado quedó fuera del conjunto. Si el
//! calibrador se equivoca más de lo prometido, α baja, los conjuntos se
//! ensanchan y se opera menos. El teorema de ACI acota la tasa media de error
//! frente al objetivo por `(max(α_1, 1 − α_1) + γ)/(γ·T)` para CUALQUIER
//! secuencia, sin suponer intercambiabilidad. `γ = 1/ventana`: la corrección
//! tiene la misma memoria que la calibración.

use std::collections::VecDeque;

pub struct ConformalCalibrator {
    scores: VecDeque<f64>,
    window: usize,
    min_calibration: usize,
    /// Nivel objetivo, procedente del gen `conformal_alpha`.
    target_alpha: f64,
    /// Nivel efectivo corregido por ACI.
    effective_alpha: f64,
    /// Pasos de adaptación realizados y errores observados en ellos.
    adapt_steps: u64,
    adapt_errors: u64,
}

impl ConformalCalibrator {
    pub fn new() -> Self {
        Self {
            scores: VecDeque::with_capacity(256),
            window: 200,
            min_calibration: 30,
            target_alpha: 0.10,
            effective_alpha: 0.10,
            adapt_steps: 0,
            adapt_errors: 0,
        }
    }

    #[inline]
    fn nonconformity(p_hat: f64, won: bool) -> f64 {
        let p = if p_hat.is_finite() {
            p_hat.clamp(0.0, 1.0)
        } else {
            0.5
        };
        if won { 1.0 - p } else { p }
    }

    #[inline]
    fn is_warm(&self) -> bool {
        self.scores.len() >= self.min_calibration
    }

    /// Nivel objetivo del genoma. Mientras no haya habido adaptación, el nivel
    /// efectivo lo sigue; después, ACI lo corrige a partir de ese objetivo.
    pub fn set_target_alpha(&mut self, alpha: f64) {
        let a = if alpha.is_finite() {
            alpha.clamp(1e-4, 0.5)
        } else {
            0.10
        };
        self.target_alpha = a;
        if self.adapt_steps == 0 {
            self.effective_alpha = a;
        }
    }

    pub fn effective_alpha(&self) -> f64 {
        self.effective_alpha
    }

    /// Tasa media de error de los pasos adaptativos (conjunto que no contuvo
    /// el resultado realizado).
    pub fn adaptive_error_rate(&self) -> f64 {
        if self.adapt_steps == 0 {
            0.0
        } else {
            self.adapt_errors as f64 / self.adapt_steps as f64
        }
    }

    /// p-valor conformal de la etiqueta indicada. 1,0 durante el warmup.
    pub fn p_value_for(&self, p_hat: f64, won_label: bool) -> f64 {
        if !self.is_warm() {
            return 1.0;
        }
        let s_new = Self::nonconformity(p_hat, won_label);
        let ge = self.scores.iter().filter(|&&s| s >= s_new).count();
        (1.0 + ge as f64) / (self.scores.len() as f64 + 1.0)
    }

    /// p-valor de la etiqueta «gana» (compatibilidad y telemetría).
    pub fn p_value(&self, p_hat_new: f64) -> f64 {
        self.p_value_for(p_hat_new, true)
    }

    /// Regla selectiva: se acepta operar si el conjunto de predicción al nivel
    /// efectivo es exactamente {gana}. Fail-open durante el warmup.
    pub fn accepts(&self, p_hat: f64) -> bool {
        if !self.is_warm() {
            return true;
        }
        let a = self.effective_alpha.clamp(1e-4, 0.5);
        self.p_value_for(p_hat, true) > a && self.p_value_for(p_hat, false) <= a
    }

    /// Alimenta la ventana con un par predicción/realización y, si ya hay
    /// calibración, adapta el nivel efectivo (ACI) ANTES de añadir el dato: el
    /// error se mide contra el conjunto que existía cuando se decidió.
    pub fn update(&mut self, p_hat: f64, won: bool) {
        if self.is_warm() {
            let a = self.effective_alpha.clamp(1e-4, 0.5);
            let realized_in_set = self.p_value_for(p_hat, won) > a;
            let err = if realized_in_set { 0.0 } else { 1.0 };
            let gamma = 1.0 / self.window as f64;
            self.effective_alpha =
                (self.effective_alpha + gamma * (self.target_alpha - err)).clamp(1e-4, 0.5);
            self.adapt_steps += 1;
            if err > 0.0 {
                self.adapt_errors += 1;
            }
        }
        let s = Self::nonconformity(p_hat, won);
        if self.scores.len() == self.window {
            self.scores.pop_front();
        }
        self.scores.push_back(s);
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
        assert!(c.accepts(0.9), "sin calibración suficiente: no bloquea");
    }

    #[test]
    fn test_r43_calibrated_p_value_bounds_and_sensitivity() {
        let mut c = ConformalCalibrator::new();
        for _ in 0..40 {
            c.update(0.9, true);
        }
        for _ in 0..10 {
            c.update(0.9, false);
        }
        let p_high = c.p_value(0.9);
        let p_low = c.p_value(0.3);
        assert!(p_high > 0.0 && p_high <= 1.0);
        assert!(p_low < p_high, "menos confianza -> p-valor menor");
    }

    /// D-618: con un historial limpio, una predicción confiada produce el
    /// conjunto {gana} y se acepta.
    #[test]
    fn d618_historial_limpio_acepta_prediccion_confiada() {
        let mut c = ConformalCalibrator::new();
        c.set_target_alpha(0.10);
        for _ in 0..50 {
            c.update(0.9, true);
        }
        c.effective_alpha = 0.10;
        assert!(c.p_value_for(0.9, true) > 0.10);
        assert!(c.p_value_for(0.9, false) <= 0.10);
        assert!(c.accepts(0.9));
    }

    /// D-618: si el modelo falló el 20 % de las veces con esa confianza, al
    /// nivel 0,10 «perder» sigue siendo plausible y el calibrador se abstiene;
    /// con una tolerancia de 0,25 ya puede excluirse.
    #[test]
    fn d618_si_perder_es_plausible_se_abstiene() {
        let mut c = ConformalCalibrator::new();
        for _ in 0..40 {
            c.update(0.9, true);
        }
        for _ in 0..10 {
            c.update(0.9, false);
        }
        c.effective_alpha = 0.10;
        assert!(!c.accepts(0.9), "p_pierde = 11/51 > 0,10: ambas etiquetas plausibles");
        c.effective_alpha = 0.25;
        assert!(c.accepts(0.9), "p_pierde = 11/51 ≤ 0,25: se excluye perder");
        c.effective_alpha = 0.10;
        assert!(!c.accepts(0.3), "predicción poco confiada: abstención");
    }

    /// D-617: tras un cambio de régimen en el que el modelo sigue confiado
    /// pero pierde, el calibrador deja de aceptar y el nivel efectivo llega a
    /// bajar del objetivo.
    #[test]
    fn d617_aci_reacciona_al_cambio_de_regimen() {
        let mut c = ConformalCalibrator::new();
        c.set_target_alpha(0.10);
        for _ in 0..50 {
            c.update(0.9, true);
        }
        assert!(c.accepts(0.9));
        let mut min_alpha = c.effective_alpha();
        for _ in 0..60 {
            c.update(0.9, false);
            min_alpha = min_alpha.min(c.effective_alpha());
        }
        assert!(min_alpha < 0.10, "ACI debe reducir α ante errores: mínimo {min_alpha}");
        assert!(!c.accepts(0.9), "tras el cambio de régimen no debe aceptar");
    }

    /// D-617: cota de ACI para una secuencia arbitraria. Aquí un modelo que
    /// predice 0,7 y acierta con esa frecuencia; la tasa media de error de los
    /// conjuntos debe quedar dentro de la cota teórica alrededor del objetivo.
    #[test]
    fn d617_tasa_de_error_de_largo_plazo_respeta_la_cota_de_aci() {
        let mut c = ConformalCalibrator::new();
        let target = 0.10;
        c.set_target_alpha(target);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..5_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            c.update(0.7, u < 0.7);
        }
        let steps = c.adapt_steps as f64;
        let gamma = 1.0 / 200.0;
        let bound = (0.9f64.max(0.1) + gamma) / (gamma * steps);
        let err = (c.adaptive_error_rate() - target).abs();
        assert!(
            err <= bound + 1e-9,
            "|error medio − α| = {err} excede la cota de ACI {bound}"
        );
    }
}
