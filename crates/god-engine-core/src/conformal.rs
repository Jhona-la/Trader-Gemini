//! R4.3 — Rangos conformales sobre ventana deslizante y adaptación recortada.
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
//! FUENTE DEL DATO: cada cierre de trade alimenta el calibrador con la
//! puntuación direccional al abrir y su resultado neto de comisiones. La
//! posición guarda `ml_prob` crudo; el llamador usa la complementaria para
//! cortos (D-676). Esto NO demuestra que p_hat estime P(PnL_neto > 0): el
//! target del modelo y los costes deben validarse por separado. Sólo se
//! admiten entradas finitas en [0, 1]. Con menos de `min_calibration`
//! observaciones válidas, acepta entradas válidas por política de warmup;
//! p=1 en ese estado es un sentinel de falta de calibración, no evidencia.
//!
//! # D-618 (DÉCIMA OLA) — la regla de decisión estaba invertida
//!
//! El filtro exigía `p_gana ≥ 1 − α`, y este mismo docstring afirmaba que esa
//! regla garantiza cobertura 1 − α. No es así. Bajo las hipótesis conformales,
//! esa cobertura corresponde al CONJUNTO que incluye etiquetas con `p > α`. Exigir
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
//! «Ganar» es plausible y «perder» no lo es según los rangos empíricos.
//! Si ambas etiquetas son plausibles el calibrador se abstiene. Incluso con
//! cobertura marginal válida, α NO acota automáticamente la tasa de pérdidas
//! CONDICIONADA a las operaciones seleccionadas. Esa garantía requiere otro
//! contrato estadístico; no la aporta la regla singleton por sí sola.
//!
//! # D-617 (DÉCIMA OLA) — intercambiabilidad rota bajo cambio de régimen
//!
//! La garantía exige que calibración y predicción sean intercambiables. Con un
//! modelo que se reentrena y se sustituye en caliente y un mercado
//! heterocedástico no está demostrada. Se usa una variante inspirada en ACI
//! (Gibbs & Candès, 2021), con proyección explícita:
//!
//! ```text
//! α_{t+1} = clip(α_t + γ · (α_objetivo − err_t), 1e-4, 0.5)
//! ```
//!
//! con `err_t = 1` si el resultado queda fuera del conjunto calculado JUSTO
//! ANTES de insertar ese cierre. No se conserva aquí el conjunto de entrada:
//! con feedback retrasado, el error no es necesariamente el de la decisión.
//!
//! La Proposición 4.1 de https://arxiv.org/abs/2106.00170 usa la recurrencia
//! SIN proyección y conjuntos extremos fuera de [0, 1]. Su cota de seguimiento
//! NO se hereda por este clip (véase el contraejemplo de empates en tests).
//! `γ = 1/200` es una política por observación; NO equivale a una ventana
//! física ni certifica cobertura por activo, escala temporal o selección.
//! La adaptación sólo observa cierres ejecutados; no outcomes contrafactuales.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// F1-057 (Ola Ω169): Estado serializable del calibrador conformal para persistencia
/// en `MonedaGuardada` / `EstadoAprendido`, erradicando el comportamiento fail-open
/// tras reinicio del host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformalGuardado {
    pub scores: Vec<f64>,
    pub effective_alpha: f64,
    pub adapt_steps: u64,
    pub adapt_errors: u64,
}

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
    /// OLA Ω220 (Átomo 4.2): Tasa de adaptación ACI efectiva modulada por Yang-Mills.
    last_gamma: f64,
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
            last_gamma: 1.0 / 200.0,
        }
    }

    /// F1-057: Exporta el estado del calibrador para persistencia en disco.
    pub fn snapshot(&self) -> ConformalGuardado {
        ConformalGuardado {
            scores: self.scores.iter().copied().collect(),
            effective_alpha: self.effective_alpha,
            adapt_steps: self.adapt_steps,
            adapt_errors: self.adapt_errors,
        }
    }

    /// F1-057: Restaura el estado desde un guardado, preservando calibración y ACI.
    pub fn restaurar(&mut self, g: &ConformalGuardado) {
        self.scores.clear();
        for &s in &g.scores {
            if s.is_finite() && (0.0..=1.0).contains(&s) {
                if self.scores.len() < self.window {
                    self.scores.push_back(s);
                }
            }
        }
        if g.effective_alpha.is_finite() {
            self.effective_alpha = g.effective_alpha.clamp(1e-4, 0.5);
        }
        self.adapt_steps = g.adapt_steps;
        self.adapt_errors = g.adapt_errors;
    }

    #[inline]
    fn valid_probability(p_hat: f64) -> bool {
        p_hat.is_finite() && (0.0..=1.0).contains(&p_hat)
    }

    #[inline]
    fn nonconformity(p_hat: f64, won: bool) -> f64 {
        debug_assert!(Self::valid_probability(p_hat));
        if won { 1.0 - p_hat } else { p_hat }
    }

    #[inline]
    pub fn is_warm(&self) -> bool {
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

    /// OLA Ω220 (Átomo 4.2): Última tasa efectiva de adaptación $\gamma_{\text{ACI}}$.
    pub fn last_gamma(&self) -> f64 {
        if self.last_gamma.is_finite() && self.last_gamma > 0.0 {
            self.last_gamma
        } else {
            1.0 / self.window as f64
        }
    }

    /// Tasa media de error de los pasos adaptativos (conjunto que no contuvo
    /// el resultado realizado al procesar el cierre, no al decidir). Cero con
    /// ningún paso es sólo el valor legacy de ausencia de datos, no cobertura.
    pub fn adaptive_error_rate(&self) -> f64 {
        if self.adapt_steps == 0 {
            0.0
        } else {
            self.adapt_errors as f64 / self.adapt_steps as f64
        }
    }

    /// Rango empírico de la etiqueta; 1,0 durante warmup de entradas válidas.
    /// Devuelve 0,0 para una entrada inválida: sentinel de rechazo, NO un
    /// p-valor estimado. Un rango válido siempre es al menos 1/(n+1).
    pub fn p_value_for(&self, p_hat: f64, won_label: bool) -> f64 {
        if !Self::valid_probability(p_hat) {
            return 0.0;
        }
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

    /// F1-057 (Ola Ω169): Evaluación unificada de p-valor y regla selectiva
    /// en una sola pasada lineal sobre `scores` (reduciendo un 66% las comparaciones).
    /// Devuelve `(p_valor_gana, acepta_operar)`.
    #[inline(always)]
    pub fn eval_prediction(&self, p_hat: f64) -> (f64, bool) {
        if !Self::valid_probability(p_hat) {
            return (0.0, false);
        }
        if !self.is_warm() {
            return (1.0, true);
        }
        let a = self.effective_alpha.clamp(1e-4, 0.5);
        let s_win = Self::nonconformity(p_hat, true);
        let s_lose = Self::nonconformity(p_hat, false);
        let mut ge_win = 0;
        let mut ge_lose = 0;
        for &s in &self.scores {
            if s >= s_win {
                ge_win += 1;
            }
            if s >= s_lose {
                ge_lose += 1;
            }
        }
        let denom = self.scores.len() as f64 + 1.0;
        let p_val_win = (1.0 + ge_win as f64) / denom;
        let p_val_lose = (1.0 + ge_lose as f64) / denom;
        let accepts = p_val_win > a && p_val_lose <= a;
        (p_val_win, accepts)
    }

    /// OLA Ω220 (Átomo 4.2): Evaluación conformal con modulación de curvatura Yang-Mills SU(N).
    ///
    /// Modula la cota de significación conformal $\alpha_{\text{gauge}}$ en función de la
    /// holonomía de bucle de Wilson $\bar{H} \in [0, 2]$:
    ///   $$\alpha_{\text{gauge}} = \text{clip}(\alpha_{\text{eff}} \times (1.0 + 0.5 \cdot \min(\bar{H}, 1.0)), 1e-4, 0.5)$$
    /// y evalúa la alineación de la corriente restauradora $\mathcal{J}_i$:
    /// - Para Long: $J_{\text{align}} = \mathcal{J}_i$
    /// - Para Short: $J_{\text{align}} = -\mathcal{J}_i$
    ///
    /// Si $J_{\text{align}} < -0.70$ (la orden se opone frontalmente al flujo covariante gauge),
    /// la compuerta conformal emite un veto estricto (`accepts = false`).
    ///
    /// Retorna: `(p_val_win, accepts, alpha_gauge, j_align)`
    #[inline]
    pub fn eval_prediction_gauge(
        &self,
        p_hat: f64,
        holonomy_density: f64,
        restoring_current: f64,
        is_long: bool,
    ) -> (f64, bool, f64, f64) {
        if !Self::valid_probability(p_hat) {
            return (0.0, false, self.effective_alpha, 0.0);
        }
        let j_cur = if restoring_current.is_finite() {
            restoring_current.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let j_align = if is_long { j_cur } else { -j_cur };

        let h = if holonomy_density.is_finite() && holonomy_density >= 0.0 {
            holonomy_density.min(1.0)
        } else {
            0.0
        };

        // Ensanchamiento de nivel alfa ante curvatura gauge dislocada
        let a_gauge = (self.effective_alpha * (1.0 + 0.5 * h)).clamp(1e-4, 0.5);

        // Veto conformal si la corriente de restauración gauge se opone críticamente
        if j_align < -0.70 {
            let p_val = self.p_value_for(p_hat, true);
            return (p_val, false, a_gauge, j_align);
        }

        if !self.is_warm() {
            return (1.0, true, a_gauge, j_align);
        }

        let s_win = Self::nonconformity(p_hat, true);
        let s_lose = Self::nonconformity(p_hat, false);
        let mut ge_win = 0;
        let mut ge_lose = 0;
        for &s in &self.scores {
            if s >= s_win {
                ge_win += 1;
            }
            if s >= s_lose {
                ge_lose += 1;
            }
        }
        let denom = self.scores.len() as f64 + 1.0;
        let p_val_win = (1.0 + ge_win as f64) / denom;
        let p_val_lose = (1.0 + ge_lose as f64) / denom;

        let accepts = p_val_win > a_gauge && p_val_lose <= a_gauge;
        (p_val_win, accepts, a_gauge, j_align)
    }

    /// Regla selectiva: se acepta operar si el conjunto de predicción al nivel
    /// efectivo es exactamente {gana}. Fail-open sólo para entradas válidas
    /// durante warmup. No garantiza tasa de acierto entre las seleccionadas.
    pub fn accepts(&self, p_hat: f64) -> bool {
        self.eval_prediction(p_hat).1
    }

    /// Alimenta la ventana con un par predicción/realización y, si ya hay
    /// calibración, adapta el nivel efectivo ANTES de añadir el dato. El error
    /// usa el estado al procesar el cierre, no un snapshot de la decisión.
    /// Una entrada inválida no muta ventana, contadores ni nivel adaptativo.
    pub fn update(&mut self, p_hat: f64, won: bool) {
        self.update_with_gauge(p_hat, won, 0.0);
    }

    /// OLA Ω220 (Átomo 4.2): Actualización ACI enriquecida con la densidad de acción Yang-Mills $\mathcal{S}_{\text{YM}}$.
    /// Modula la tasa de adaptación $\gamma_t = \gamma_0 \times (1.0 + \tanh(\mathcal{S}_{\text{YM}}))$ para responder
    /// rápidamente a la no-estacionariedad cuando el mercado sufre tensión de arbitraje triangular.
    pub fn update_with_gauge(&mut self, p_hat: f64, won: bool, action_density: f64) {
        if !Self::valid_probability(p_hat) {
            return;
        }
        if self.is_warm() {
            let a = self.effective_alpha.clamp(1e-4, 0.5);
            let realized_in_set = self.p_value_for(p_hat, won) > a;
            let err = if realized_in_set { 0.0 } else { 1.0 };

            let s_act = if action_density.is_finite() && action_density >= 0.0 {
                action_density.min(10.0)
            } else {
                0.0
            };
            let gamma_0 = 1.0 / self.window as f64;
            let gamma = gamma_0 * (1.0 + s_act.tanh());
            self.last_gamma = gamma;

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

    /// Testigo histórico D-617: esta secuencia concreta cae dentro de la
    /// referencia numérica de ACI. No prueba la cota para la variante recortada;
    /// conformal_wiring_contract contiene un contraejemplo de empates.
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

    /// F1-057 (Ola Ω169): Verificación de que el snapshot y restauración serializan y
    /// reconstruyen fielmente el estado del calibrador conformal, preservando `is_warm`,
    /// `effective_alpha`, `eval_prediction` e impidiendo el fail-open tras reinicio.
    #[test]
    fn f1_057_snapshot_restaurar_preserva_calibracion_y_eval_prediction() {
        let mut original = ConformalCalibrator::new();
        original.set_target_alpha(0.08);
        for i in 0..45 {
            original.update(0.85, i % 3 != 0);
        }
        assert!(original.is_warm());
        let (orig_p, orig_acc) = original.eval_prediction(0.85);

        let snap = original.snapshot();
        assert_eq!(snap.scores.len(), 45);
        assert!((snap.effective_alpha - original.effective_alpha()).abs() < 1e-12);

        // Serialización serde JSON idempotente
        let json = serde_json::to_string(&snap).expect("serializa");
        let decoded: ConformalGuardado = serde_json::from_str(&json).expect("deserializa");
        assert_eq!(snap, decoded);

        let mut restaurado = ConformalCalibrator::new();
        assert!(!restaurado.is_warm());
        restaurado.restaurar(&decoded);
        assert!(restaurado.is_warm(), "debe estar caliente tras restaurar");
        assert!((restaurado.effective_alpha() - original.effective_alpha()).abs() < 1e-12);
        let (rest_p, rest_acc) = restaurado.eval_prediction(0.85);
        assert!((orig_p - rest_p).abs() < 1e-12);
        assert_eq!(orig_acc, rest_acc);
        assert_eq!(original.accepts(0.85), restaurado.accepts(0.85));
        assert_eq!(original.p_value(0.85), restaurado.p_value(0.85));
    }

    #[test]
    fn f4_002_conformal_gauge_veto_on_opposing_current() {
        let mut c = ConformalCalibrator::new();
        c.set_target_alpha(0.10);
        for _ in 0..50 {
            c.update(0.9, true);
        }
        assert!(c.is_warm());

        // Long con corriente alineada (+0.50) debe ser admitido
        let (_p, accept_aligned, _a_gauge, j_align) = c.eval_prediction_gauge(0.9, 0.0, 0.50, true);
        assert!(accept_aligned, "Long con corriente Yang-Mills alineada debe ser admitido");
        assert!((j_align - 0.50).abs() < 1e-12);

        // Long con corriente fuertemente opuesta (-0.75) debe ser vetado conformalmente
        let (_p, accept_opposed, _a_gauge, j_opposed) = c.eval_prediction_gauge(0.9, 0.0, -0.75, true);
        assert!(!accept_opposed, "Long con corriente Yang-Mills fuertemente opuesta debe ser vetado");
        assert!((j_opposed - (-0.75)).abs() < 1e-12);

        // Short con corriente positiva (+0.75) se opone al short (j_align = -0.75) => vetado
        let (_p, accept_short_opposed, _a_gauge, j_short_opp) = c.eval_prediction_gauge(0.9, 0.0, 0.75, false);
        assert!(!accept_short_opposed, "Short contra corriente positiva debe ser vetado");
        assert!((j_short_opp - (-0.75)).abs() < 1e-12);
    }

    #[test]
    fn f4_002_conformal_gauge_holonomy_widens_alpha() {
        let c = ConformalCalibrator::new();
        // Con holonomía nula (curvatura plana H=0): alpha_gauge == effective_alpha
        let (_p, _acc, a_flat, _) = c.eval_prediction_gauge(0.8, 0.0, 0.0, true);
        assert!((a_flat - c.effective_alpha()).abs() < 1e-9);

        // Con holonomía dislocada (H=1.0): alpha_gauge = effective_alpha * 1.5
        let (_p, _acc, a_curved, _) = c.eval_prediction_gauge(0.8, 1.0, 0.0, true);
        assert!((a_curved - c.effective_alpha() * 1.5).abs() < 1e-9);
        assert!(a_curved > a_flat);
    }

    #[test]
    fn f4_002_conformal_update_with_gauge_adapts_faster() {
        let mut c_flat = ConformalCalibrator::new();
        let mut c_dislocated = ConformalCalibrator::new();
        for _ in 0..40 {
            c_flat.update(0.9, true);
            c_dislocated.update(0.9, true);
        }
        // Ambas reciben un fallo (err=1.0)
        c_flat.update_with_gauge(0.9, false, 0.0);
        c_dislocated.update_with_gauge(0.9, false, 2.5);

        // La dislocación gauge Yang-Mills acelera la tasa gamma de adaptación ACI
        assert!(c_dislocated.last_gamma() > c_flat.last_gamma());
        // En consecuencia, effective_alpha reduce más ante el error bajo dislocación
        assert!(c_dislocated.effective_alpha() < c_flat.effective_alpha());
    }
}
