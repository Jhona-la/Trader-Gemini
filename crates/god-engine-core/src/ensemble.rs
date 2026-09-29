//! ENSAMBLE ONLINE (F4.7) — predicción combinada con pesos aprendidos.
//!
//! QUÉ: combina predictores (NanoForest y DarkAlphaNN) en
//!      una sola probabilidad con pesos que EVOLUCIONAN con el desempeño real.
//! POR QUÉ: el código anterior era una cadena if-else de fallback (forest O NN
//!      O 0.5) — el segundo modelo solo opinaba si el primero no existía. Un
//!      ensamble real promedia diversidad: modelos que ven features distintas
//!      pueden aportar diversidad; su beneficio debe medirse fuera de muestra.
//! MATEMÁTICA (Hedge/Exponential Weighting — Cesa-Bianchi & Lugosi):
//!      w_i *= exp(-η · Brier_i(p_i, y))
//!   donde Brier = (p − y)² es una puntuación estrictamente apropiada: el
//!   peso decae con la pérdida predictiva, no sólo con descalibración. Los
//!   pesos viven en escala log para no sub/flotear en f64; se normalizan al
//!   combinar. El factor shrink descuenta log-evidencia hacia pesos uniformes;
//!   no constituye por sí solo una posterior bayesiana ni una penalización L2.
//! CONTRATO CAUSAL: barras califican su primera opinión; cierres demorados
//!   requieren un snapshot de apertura ligado por el llamador a la posición.
//!   La tasa por trade depende del retorno observado: repondera la pérdida
//!   y no hereda automáticamente garantías de calibración o regret de Hedge.

/// Identificador estable de cada predictor del ensamble.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    /// NanoForest del motor continuo (features universales 48D).
    MotorForest,
    /// DarkAlpha NN (features macro+micro, BTC-only por diseño).
    DarkAlphaNN,
}

/// D-695 (DÉCIMA OLA) — HABILIDAD MEDIDA DEL ENSAMBLE.
///
/// El escudo neuronal vetaba intenciones con la opinión del ensamble sin
/// comprobar que esa opinión predijera algo. Tras D-693 seguía vetando más del
/// 95 % de los largos con un ensamble sesgado (bosque bajista, red saturada).
/// Un veto sólo está justificado si el modelo supera a la referencia trivial:
/// la tasa base de subidas.
///
/// Se mide con la puntuación de Brier, estrictamente apropiada, sobre la misma
/// opinión que califica el Hedge (la del arranque de cada vela):
/// `d = (tasa_base − y)² − (p − y)²`, positiva cuando el modelo gana. La tasa
/// base es causal (la estimada antes de ver el resultado). Media y varianza
/// exponenciales de `d` con la escala de la EMA macro del motor; la habilidad
/// se resume mediante `z = media / (σ / √n_eff)` frente a z95, con
/// `n_eff = (2 − α)/α`, el tamaño efectivo de una media exponencial.
/// Ese tamaño presupone observaciones independientes; autocorrelación,
/// selección por gates y consultas secuenciales invalidan una garantía al 95 %.
pub const SKILL_SPAN_BARS: f64 = crate::diffusion::EMA_MACRO_BARS;

#[derive(Debug, Clone, Default)]
pub struct SkillTracker {
    base_rate: f64,
    base_n: u64,
    mean_d: f64,
    var_d: f64,
    n: u64,
}

impl SkillTracker {
    fn alpha() -> f64 {
        2.0 / (SKILL_SPAN_BARS + 1.0)
    }

    /// Registra una vela: `p` es la probabilidad combinada del arranque y `y`
    /// el resultado (1 subió, 0 no).
    pub fn record(&mut self, p: f64, y: f64) {
        if !valid_probability(p) || !valid_label(y) {
            return;
        }
        let a = Self::alpha();
        if self.base_n > 0 {
            let reference = self.base_rate;
            let d = (reference - y).powi(2) - (p - y).powi(2);
            if self.n == 0 {
                self.mean_d = d;
                self.var_d = 0.0;
            } else {
                let delta = d - self.mean_d;
                self.mean_d += a * delta;
                self.var_d = (1.0 - a) * (self.var_d + a * delta * delta);
            }
            self.n += 1;
            self.base_rate += a * (y - self.base_rate);
        } else {
            self.base_rate = y;
        }
        self.base_n += 1;
    }

    /// Estadístico z de la ventaja de Brier sobre la tasa base, o `None` hasta
    /// haber visto una ventana completa.
    pub fn z(&self) -> Option<f64> {
        if (self.n as f64) < SKILL_SPAN_BARS {
            return None;
        }
        let a = Self::alpha();
        let n_eff = (2.0 - a) / a;
        let sd = self.var_d.max(0.0).sqrt();
        if sd <= 0.0 {
            return None;
        }
        Some(self.mean_d / (sd / n_eff.sqrt()))
    }

    /// Umbral heurístico de habilidad; no es una certificación estadística al 95 %.
    pub fn is_significant(&self) -> bool {
        self.z().is_some_and(|z| z > crate::diffusion::Z95)
    }
}

fn valid_probability(p: f64) -> bool {
    p.is_finite() && (0.0..=1.0).contains(&p)
}

fn valid_label(y: f64) -> bool {
    y == 0.0 || y == 1.0
}

pub struct ModelEnsemble {
    /// Pesos en escala logarítmica (log w_i).
    log_weights: [f64; 2],
    /// Tasa de aprendizaje del Hedge (η). 0.05 = ajuste gradual.
    eta: f64,
    /// Regularización hacia uniforme (regresión a la media del peso).
    shrink: f64,
    predictions: [Option<f64>; 2],
    /// X-023: primera opinión de cada modelo dentro del bar — la calificada
    /// al cierre (lead real sobre el desenlace).
    bar_open_predictions: [Option<f64>; 2],
    /// Última predicción registrada por cada modelo (inmune a reset entre velas)
    last_predictions: [Option<f64>; 2],
    /// D-695: habilidad medida frente a la tasa base.
    skill: SkillTracker,
}

impl Default for ModelEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelEnsemble {
    pub fn new() -> Self {
        Self {
            log_weights: [0.0, 0.0], // w=1 ambos: sin sesgo de arranque
            eta: 0.05,
            shrink: 0.995,
            predictions: [None, None],
            bar_open_predictions: [None, None],
            last_predictions: [None, None],
            skill: SkillTracker::default(),
        }
    }

    /// Registra la predicción de un modelo para el evento actual.
    /// Fuera de [0,1] o no finita: ausencia actual, nunca una probabilidad
    /// imputada. Conserva la primera opinión válida del bar para calificarla.
    #[inline(always)]
    pub fn submit(&mut self, id: ModelId, prob: f64) {
        let slot = id as usize;
        if !valid_probability(prob) {
            self.predictions[slot] = None;
            self.last_predictions[slot] = None;
            return;
        }
        // B3.38b — CLAMP en vez de neutralizar. El neutralizador (p≥0.9999
        // ⇒ 0.5) mataba señal LEGÍTIMA: con el etiquetado honesto HOST-010
        // (barrera asimétrica RR 2:1, base ~20-30%) un GBDT fuerte satura a
        // 1.0/0.0 en estados extremos — NEAR medido en vivo prediciendo 1.0
        // y el ensamble sirviendo 0.5 neutral. Un modelo DEGENERADO de
        // verdad (constante en el extremo) lo castiga el propio Hedge:
        // update_with_outcome gradúa sus predicciones cada vela y su peso
        // decae exponencialmente — autocorrectivo, sin código que adivine
        // la intención del modelo.
        let p = prob.clamp(0.001, 0.999);
        self.predictions[slot] = Some(p);
        self.last_predictions[slot] = Some(p);
        // X-023: si es la PRIMERA opinión del bar, es la del arranque — la
        // única con lead real sobre el desenlace del bar.
        if self.bar_open_predictions[slot].is_none() {
            self.bar_open_predictions[slot] = Some(p);
        }
    }

    /// Probabilidad combinada de los modelos que opinaron (None si ninguno).
    /// p = Σ w_i·p_i / Σ w_i con w_i = softmax(log_weights).
    #[inline(always)]
    pub fn combined(&self) -> Option<f64> {
        self.combine(&self.predictions)
    }

    /// Opiniones del evento actual, sin recuperar cachés de eventos anteriores.
    /// El dueño de la posición debe congelarlas al abrir, no al cerrar.
    pub fn prediction_snapshot(&self) -> [Option<f64>; 2] {
        self.predictions
    }

    /// Combinación ponderada de un juego de opiniones con los pesos actuales.
    fn combine(&self, preds: &[Option<f64>; 2]) -> Option<f64> {
        // Normalizar sólo sobre participantes. Un modelo ausente con mayor
        // peso no debe subdesbordar a cero todos los votos disponibles.
        let nn_penalty = self.skill.z().unwrap_or(0.0).clamp(-4.0, 0.0);
        let effective: [Option<f64>; 2] = std::array::from_fn(|i| {
            preds[i].filter(|p| valid_probability(*p)).map(|_| {
                self.log_weights[i]
                    + if i == ModelId::DarkAlphaNN as usize {
                        nn_penalty
                    } else {
                        0.0
                    }
            })
        });
        let max_lw = effective
            .iter()
            .flatten()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum_w = 0.0;
        let mut sum_wp = 0.0;
        for (i, pred) in preds.iter().enumerate() {
            if let (Some(p), Some(lw)) = (pred, effective[i]) {
                let w = (lw - max_lw).exp();
                sum_w += w;
                sum_wp += w * p;
            }
        }
        if sum_w <= 0.0 {
            None
        } else {
            Some(sum_wp / sum_w)
        }
    }

    /// Aprende del resultado REAL (y ∈ {0.0, 1.0} — dirección efectiva).
    /// X-023: califica la opinión del ARRANQUE del bar (bar_open_predictions)
    /// — la predicción que tenía lead real sobre el desenlace. Antes se
    /// calificaba la del tick inmediatamente previo al cierre (lead ~0): los
    /// pesos Hedge aprendían la cantidad equivocada.
    pub fn update_with_outcome(&mut self, y: f64) {
        if !valid_label(y) {
            return; // no mutar pesos, habilidad ni opiniones pendientes
        }
        let graded = self.bar_open_predictions;
        // D-695: la habilidad se mide con la opinión combinada del arranque y
        // los pesos vigentes antes de actualizarlos.
        if let Some(p_open) = self.combine(&graded) {
            self.skill.record(p_open, y);
        }
        for (i, pred) in graded.iter().enumerate() {
            if let Some(p) = pred.as_ref() {
                let p = *p;
                let brier = (p - y) * (p - y);
                // Hedge: peso decae con el error; shrink: tira hacia uniforme.
                self.log_weights[i] = self.log_weights[i] * self.shrink - self.eta * brier;
            }
        }
        // Reset: nuevo bar — nueva primera opinión por aprender.
        self.predictions = [None, None];
        self.bar_open_predictions = [None, None];
    }

    /// Adaptador legado para feedback síncrono. No identifica una operación:
    /// NO usar para cierres demorados; usar `update_with_trade_snapshot` con
    /// evidencia congelada y verificada por el propietario de la posición.
    pub fn update_with_trade_outcome(&mut self, subio: bool, pnl_pct: f64) {
        let preds = if self.predictions.iter().any(|p| p.is_some()) {
            self.predictions
        } else {
            self.last_predictions
        };
        self.update_with_trade_snapshot(preds, subio, pnl_pct);
    }

    /// Califica opiniones de apertura, no las disponibles al recibir el cierre.
    /// `subio`: dirección de precio, no signo del PnL neto. `pnl_pct`: retorno
    /// fraccional que modula la tasa existente; no prueba liquidación económica.
    /// Retorno inválido: no-op. Opinión inválida: se omite sólo ese predictor.
    pub fn update_with_trade_snapshot(
        &mut self,
        preds: [Option<f64>; 2],
        subio: bool,
        pnl_pct: f64,
    ) {
        if !pnl_pct.is_finite() {
            return;
        }
        let y = if subio { 1.0 } else { 0.0 };
        let eta_trade = (0.25 * (pnl_pct.abs() / 0.001).clamp(0.5, 3.0)).clamp(0.08, 0.75);
        for (i, pred) in preds.iter().enumerate() {
            if let Some(p) = pred.filter(|p| valid_probability(*p)) {
                let brier = (p - y) * (p - y);
                // Sin shrink aquí: la evidencia del PnL real es persistente
                self.log_weights[i] -= eta_trade * brier;
            }
        }
    }

    /// D-695: estadístico z de la ventaja de Brier sobre la tasa base.
    pub fn skill_z(&self) -> Option<f64> {
        self.skill.z()
    }

    /// D-695: ¿supera el umbral heurístico de habilidad frente a la tasa base?
    pub fn has_significant_skill(&self) -> bool {
        self.skill.is_significant()
    }

    /// Pesos normalizados actuales (para telemetría/calibración F6).
    pub fn weights(&self) -> [f64; 2] {
        let max_lw = self
            .log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = self
            .log_weights
            .iter()
            .map(|lw| (lw - max_lw).exp())
            .collect();
        let s: f64 = w.iter().sum();
        if s > 0.0 {
            [w[0] / s, w[1] / s]
        } else {
            [0.5, 0.5]
        }
    }

    /// Calibración viva (Brier acumulado por modelo, para reportes F6).
    pub fn log_briers(&self) -> [f64; 2] {
        // Expuesto indirecto: el lector externo trackea outcomes; aquí
        // devolvemos la brecha de pesos (proxy de desempeño relativo).
        self.log_weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// D-695: un predictor informativo supera a la tasa base con significación.
    #[test]
    fn d695_predictor_informativo_tiene_habilidad() {
        let mut t = SkillTracker::default();
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..(SKILL_SPAN_BARS as usize * 4) {
            let signal = xorshift(&mut s) < 0.5;
            let p = if signal { 0.8 } else { 0.2 };
            let y = if xorshift(&mut s) < p { 1.0 } else { 0.0 };
            t.record(p, y);
        }
        assert!(t.is_significant(), "z = {:?}", t.z());
    }

    /// D-695: un modelo sesgado sin información no gana el derecho a vetar.
    #[test]
    fn d695_modelo_sesgado_sin_informacion_no_tiene_habilidad() {
        let mut t = SkillTracker::default();
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..(SKILL_SPAN_BARS as usize * 4) {
            let y = if xorshift(&mut s) < 0.5 { 1.0 } else { 0.0 };
            t.record(0.3, y);
        }
        assert!(!t.is_significant(), "z = {:?}", t.z());
        assert!(t.z().is_some_and(|z| z < 0.0));
    }

    /// D-695: sin una ventana completa no se publica habilidad.
    #[test]
    fn d695_sin_ventana_completa_no_hay_veredicto() {
        let mut t = SkillTracker::default();
        for i in 0..(SKILL_SPAN_BARS as usize / 2) {
            t.record(0.9, if i % 2 == 0 { 1.0 } else { 0.0 });
        }
        assert_eq!(t.z(), None);
        assert!(!t.is_significant());
    }

    /// D-695: el ensamble mide su habilidad con la opinión del arranque de la vela.
    #[test]
    fn d695_el_ensamble_registra_su_habilidad_al_calificar() {
        let mut e = ModelEnsemble::new();
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        for _ in 0..(SKILL_SPAN_BARS as usize * 4) {
            let signal = xorshift(&mut s) < 0.5;
            let p = if signal { 0.85 } else { 0.15 };
            e.submit(ModelId::MotorForest, p);
            e.submit(ModelId::DarkAlphaNN, p);
            let y = if xorshift(&mut s) < p { 1.0 } else { 0.0 };
            e.update_with_outcome(y);
        }
        assert!(e.has_significant_skill(), "z = {:?}", e.skill_z());
    }

    /// XLIV-9b: un largo cuyo mid sube 3 pb y pierde por 10 pb de comisiones.
    /// El mercado SUBIÓ: el modelo que dijo «sube» (0,9) debe ganar peso
    /// frente al que dijo «baja» (0,1). Con la etiqueta vieja
    /// `(is_long, is_win)` = (true, false) ⇒ y = 0 y ocurría lo contrario.
    #[test]
    fn xliv_el_ensamble_aprende_la_direccion_del_mercado_no_el_pnl_neto() {
        let mut e = ModelEnsemble::new();
        for _ in 0..50 {
            e.submit(ModelId::MotorForest, 0.9);
            e.submit(ModelId::DarkAlphaNN, 0.1);
            let subio = crate::direccion_realizada(true, 0.0003).expect("hubo movimiento");
            e.update_with_trade_outcome(subio, 0.0003);
        }
        let w = e.weights();
        assert!(w[0] > w[1], "el que acertó «sube» debe pesar más: {w:?}");
    }

    #[test]
    fn sin_opiniones_no_hay_combinacion() {
        let mut e = ModelEnsemble::new();
        assert_eq!(e.combined(), None);
        e.submit(ModelId::MotorForest, 0.7);
        assert!((e.combined().unwrap() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn promedio_ponderado_inicial_es_uniforme() {
        let mut e = ModelEnsemble::new();
        e.submit(ModelId::MotorForest, 0.8);
        e.submit(ModelId::DarkAlphaNN, 0.4);
        // Pesos iguales al inicio: media simple.
        assert!((e.combined().unwrap() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn hedge_premia_al_calibrado_y_castiga_al_descalibrado() {
        let mut e = ModelEnsemble::new();
        // 200 eventos: forest siempre acierta (p=0.9 cuando y=1), NN siempre
        // en contra (p=0.2 cuando y=1).
        for _ in 0..200 {
            e.submit(ModelId::MotorForest, 0.9);
            e.submit(ModelId::DarkAlphaNN, 0.2);
            e.update_with_outcome(1.0);
        }
        let w = e.weights();
        assert!(
            w[ModelId::MotorForest as usize] > 0.9,
            "forest calibrado debe dominar: {:?}",
            w
        );
        // La combinación debe acercarse a la opinión del bueno.
        e.submit(ModelId::MotorForest, 0.9);
        e.submit(ModelId::DarkAlphaNN, 0.2);
        assert!(e.combined().unwrap() > 0.7);
    }

    #[test]
    fn shrink_evita_muerte_eterna() {
        let mut e = ModelEnsemble::new();
        // El forest arraca pésimo y el NN perfecto...
        for _ in 0..50 {
            e.submit(ModelId::MotorForest, 0.9);
            e.submit(ModelId::DarkAlphaNN, 0.6);
            e.update_with_outcome(0.0); // siempre cae: forest muy mal, NN mal
        }
        let w_bad_start = e.weights();
        // ...luego el forest se calibra de verdad por 300 eventos.
        for _ in 0..300 {
            e.submit(ModelId::MotorForest, 0.1);
            e.submit(ModelId::DarkAlphaNN, 0.6);
            e.update_with_outcome(0.0);
        }
        let w_recovered = e.weights();
        assert!(
            w_recovered[ModelId::MotorForest as usize] > w_bad_start[ModelId::MotorForest as usize],
            "shrink permite recuperación: {:?} → {:?}",
            w_bad_start,
            w_recovered
        );
    }

    #[test]
    fn outcome_resetea_predicciones_del_evento() {
        let mut e = ModelEnsemble::new();
        e.submit(ModelId::MotorForest, 0.7);
        e.update_with_outcome(1.0);
        assert_eq!(
            e.combined(),
            None,
            "predicciones no contaminan el próximo evento"
        );
    }
}
