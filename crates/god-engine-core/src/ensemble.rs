//! ENSAMBLE BAYESIANO ONLINE (F4.7) — predicción combinada con pesos aprendidos.
//!
//! QUÉ: combina N predictores (NanoForest scalp, NN swing, forest online) en
//!      una sola probabilidad con pesos que EVOLUCIONAN con el desempeño real.
//! POR QUÉ: el código anterior era una cadena if-else de fallback (forest O NN
//!      O 0.5) — el segundo modelo solo opinaba si el primero no existía. Un
//!      ensamble real promedia diversidad: modelos que ven features distintas
//!      se corrigen mutuamente (varianza ↓, robustez ↑).
//! MATEMÁTICA (Hedge/Exponential Weighting — Cesa-Bianchi & Lugosi):
//!      w_i *= exp(-η · Brier_i(p_i, y))
//!   donde Brier = (p − y)² es una puntuación estrictamente apropiada: el
//!   peso decae exponencialmente con el error de calibración ACUMULADO. Los
//!   pesos viven en escala log para no sub/flotear en f64; se normalizan al
//!   combinar. Regresión a la media del peso (regularización L2 suave) evita
//!   que un modelo quede muerto eternamente tras un mal arranque.
//! SIN CRISTAL BALLA: update_with_outcome se alimenta del CIERRE real
//!      (dirección efectiva del trade o del bar siguiente) — jamás de labels
//!      sintéticos.

/// Identificador estable de cada predictor del ensamble.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    /// NanoForest de scalping (features microestructura).
    ScalpForest,
    /// DarkAlpha NN de swing (features macro+micro).
    SwingNN,
}

pub struct ModelEnsemble {
    /// Pesos en escala logarítmica (log w_i).
    log_weights: [f64; 2],
    /// Tasa de aprendizaje del Hedge (η). 0.05 = ajuste gradual.
    eta: f64,
    /// Regularización hacia uniforme (regresión a la media del peso).
    shrink: f64,
    predictions: [Option<f64>; 2],
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
        }
    }

    /// Registra la predicción de un modelo para el evento actual.
    #[inline(always)]
    pub fn submit(&mut self, id: ModelId, prob: f64) {
        let p = prob.clamp(0.0, 1.0);
        self.predictions[id as usize] = Some(p);
    }

    /// Probabilidad combinada de los modelos que opinaron (None si ninguno).
    /// p = Σ w_i·p_i / Σ w_i con w_i = softmax(log_weights).
    #[inline(always)]
    pub fn combined(&self) -> Option<f64> {
        // Softmax estable: restar el máximo antes de exp.
        let max_lw = self
            .log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum_w = 0.0;
        let mut sum_wp = 0.0;
        for (i, pred) in self.predictions.iter().enumerate() {
            if let Some(p) = pred.as_ref() {
                let p = *p;
                let w = (self.log_weights[i] - max_lw).exp();
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
    /// Debe llamarse tras conocer el desenlace del evento predicho.
    pub fn update_with_outcome(&mut self, y: f64) {
        for (i, pred) in self.predictions.iter().enumerate() {
            if let Some(p) = pred.as_ref() {
                let p = *p;
                let brier = (p - y) * (p - y);
                // Hedge: peso decae con el error; shrink: tira hacia uniforme.
                self.log_weights[i] = self.log_weights[i] * self.shrink - self.eta * brier;
            }
        }
        // Reset: las predicciones eran de ESTE evento — no contaminar el siguiente.
        self.predictions = [None, None];
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

    #[test]
    fn sin_opiniones_no_hay_combinacion() {
        let mut e = ModelEnsemble::new();
        assert_eq!(e.combined(), None);
        e.submit(ModelId::ScalpForest, 0.7);
        assert!((e.combined().unwrap() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn promedio_ponderado_inicial_es_uniforme() {
        let mut e = ModelEnsemble::new();
        e.submit(ModelId::ScalpForest, 0.8);
        e.submit(ModelId::SwingNN, 0.4);
        // Pesos iguales al inicio: media simple.
        assert!((e.combined().unwrap() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn hedge_premia_al_calibrado_y_castiga_al_descalibrado() {
        let mut e = ModelEnsemble::new();
        // 200 eventos: forest siempre acierta (p=0.9 cuando y=1), NN siempre
        // en contra (p=0.2 cuando y=1).
        for _ in 0..200 {
            e.submit(ModelId::ScalpForest, 0.9);
            e.submit(ModelId::SwingNN, 0.2);
            e.update_with_outcome(1.0);
        }
        let w = e.weights();
        assert!(
            w[ModelId::ScalpForest as usize] > 0.9,
            "forest calibrado debe dominar: {:?}",
            w
        );
        // La combinación debe acercarse a la opinión del bueno.
        e.submit(ModelId::ScalpForest, 0.9);
        e.submit(ModelId::SwingNN, 0.2);
        assert!(e.combined().unwrap() > 0.7);
    }

    #[test]
    fn shrink_evita_muerte_eterna() {
        let mut e = ModelEnsemble::new();
        // El forest arraca pésimo y el NN perfecto...
        for _ in 0..50 {
            e.submit(ModelId::ScalpForest, 0.9);
            e.submit(ModelId::SwingNN, 0.6);
            e.update_with_outcome(0.0); // siempre cae: forest muy mal, NN mal
        }
        let w_bad_start = e.weights();
        // ...luego el forest se calibra de verdad por 300 eventos.
        for _ in 0..300 {
            e.submit(ModelId::ScalpForest, 0.1);
            e.submit(ModelId::SwingNN, 0.6);
            e.update_with_outcome(0.0);
        }
        let w_recovered = e.weights();
        assert!(
            w_recovered[ModelId::ScalpForest as usize] > w_bad_start[ModelId::ScalpForest as usize],
            "shrink permite recuperación: {:?} → {:?}",
            w_bad_start,
            w_recovered
        );
    }

    #[test]
    fn outcome_resetea_predicciones_del_evento() {
        let mut e = ModelEnsemble::new();
        e.submit(ModelId::ScalpForest, 0.7);
        e.update_with_outcome(1.0);
        assert_eq!(
            e.combined(),
            None,
            "predicciones no contaminan el próximo evento"
        );
    }
}
