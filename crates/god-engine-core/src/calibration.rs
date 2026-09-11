//! CALIBRACIÓN EN LÍNEA DE LA CONFIANZA (D-619 — DÉCIMA OLA).
//!
//! # El problema
//!
//! `SignalIntent::confidence` entra en el Kelly y en el valor esperado como si
//! fuera una probabilidad de ganar, pero se fabrica con mapeos heurísticos
//! —`tanh(·)` con suelo 0,55, `0,5 + 0,4·|score|`, fuerza del consenso
//! tensorial— que nunca se contrastaron con resultados. Quitar esos suelos a
//! secas no la convierte en probabilidad: sólo cambia qué número arbitrario
//! llega a los gates. Lo que falta es un mapa aprendido de la puntuación a la
//! frecuencia real de acierto.
//!
//! # El modelo
//!
//! Escalado de Platt: `p = σ(a·logit(s) + b)`, con `s` la puntuación cruda,
//! estimado por máximo a posteriori sobre una ventana acotada de pares
//! (puntuación, resultado).
//!
//! * **Sin datos, identidad exacta.** El sistema arranca con el comportamiento
//!   que ya tenía; la calibración sólo se aparta de él con evidencia.
//! * **Prior en forma de pseudo-observaciones.** El mapa identidad equivale a
//!   observar, en cada puntuación `s`, una frecuencia de acierto `s`. El prior
//!   son `z² ≈ 3,84` pseudo-observaciones en total —el mismo peso que el prior
//!   del win rate (D-680)—, con etiqueta suave igual a la puntuación, repartidas
//!   por igual sobre las puntuaciones observadas. Cuando todas las
//!   observaciones comparten puntuación, el resultado coincide exactamente con
//!   la media posterior Beta de D-680: tres pérdidas a 0,8 dan 0,449, no 0,27.
//!   Un prior sobre los coeficientes `(a, b)` no tiene esa propiedad: su peso
//!   efectivo en la escala de probabilidad depende de dónde caen los datos.
//! * **Pendiente no negativa.** Si la evidencia dice que la puntuación no
//!   informa, el mapa se aplana hacia la tasa base (`a = 0`); nunca invierte
//!   el orden de las señales.
//!
//! Cada ajuste son unas pocas iteraciones de Newton sobre una matriz 2×2, y
//! sólo ocurre al cerrar una operación: fuera del camino caliente.

use std::collections::VecDeque;

/// Pseudo-observaciones del prior: `z²` con `z` = 1,959964.
pub const PRIOR_PSEUDO_OBSERVATIONS: f64 = 1.959_963_984_540_054 * 1.959_963_984_540_054;

/// Pares (puntuación, resultado) conservados. Una ventana acotada deja que el
/// mapa siga cambios de régimen sin olvidar de golpe.
pub const WINDOW: usize = 512;

/// Margen que aleja la puntuación de 0 y 1 antes del logit.
const EPS: f64 = 1e-4;

/// Regularización numérica hacia la identidad. Sólo actúa en la dirección que
/// los datos no identifican —por ejemplo, con todas las observaciones en la
/// misma puntuación la hessiana 2×2 es singular—; no altera ninguna estimación
/// que los datos sí determinen.
const RIDGE: f64 = 1e-4;

#[inline]
fn logit(p: f64) -> f64 {
    let p = p.clamp(EPS, 1.0 - EPS);
    (p / (1.0 - p)).ln()
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
pub struct PlattCalibrator {
    obs: VecDeque<(f64, bool)>,
    a: f64,
    b: f64,
}

impl Default for PlattCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl PlattCalibrator {
    pub fn new() -> Self {
        Self {
            obs: VecDeque::with_capacity(WINDOW),
            a: 1.0,
            b: 0.0,
        }
    }

    pub fn observations(&self) -> usize {
        self.obs.len()
    }

    /// Coeficientes actuales `(a, b)`.
    pub fn coefficients(&self) -> (f64, f64) {
        (self.a, self.b)
    }

    /// Probabilidad calibrada para una puntuación cruda.
    #[inline]
    pub fn calibrate(&self, score: f64) -> f64 {
        if !score.is_finite() {
            return 0.5;
        }
        sigmoid(self.a * logit(score) + self.b)
    }

    /// Añade un resultado y reajusta el mapa.
    pub fn update(&mut self, score: f64, won: bool) {
        if !score.is_finite() {
            return;
        }
        if self.obs.len() == WINDOW {
            self.obs.pop_front();
        }
        self.obs.push_back((score, won));
        self.fit();
    }

    fn fit(&mut self) {
        let n = self.obs.len();
        if n == 0 {
            return;
        }
        let pseudo_weight = PRIOR_PSEUDO_OBSERVATIONS / n as f64;
        let (mut a, mut b) = (self.a, self.b);
        for _ in 0..50 {
            // Gradiente y hessiana del negativo del log-posterior.
            let (mut ga, mut gb) = (RIDGE * (a - 1.0), RIDGE * b);
            let (mut haa, mut hab, mut hbb) = (RIDGE, 0.0, RIDGE);
            for &(s, won) in &self.obs {
                let x = logit(s);
                let p = sigmoid(a * x + b);
                let w = p * (1.0 - p);
                let observed = if won { 1.0 } else { 0.0 };
                // Observación real (peso 1) y pseudo-observación del prior
                // (peso z²/n, etiqueta suave = la propia puntuación).
                let r = (p - observed) + pseudo_weight * (p - s);
                let wt = 1.0 + pseudo_weight;
                ga += r * x;
                gb += r;
                haa += wt * w * x * x;
                hab += wt * w * x;
                hbb += wt * w;
            }
            let det = haa * hbb - hab * hab;
            if !det.is_finite() || det <= 0.0 {
                break;
            }
            let da = (hbb * ga - hab * gb) / det;
            let db = (haa * gb - hab * ga) / det;
            a = (a - da).max(0.0);
            b -= db;
            if da.abs() < 1e-12 && db.abs() < 1e-12 {
                break;
            }
        }
        if a.is_finite() && b.is_finite() {
            self.a = a;
            self.b = b;
        }
    }
}

/// D-693 (DÉCIMA OLA) — PROBABILIDAD DE SUBIDA QUE CONSUMEN LAS DECISIONES.
///
/// Es la probabilidad del ensamble (bosque y red, ponderados por Brier) más el
/// sesgo spot, acotada a [0, 1]. El residuo online del `OnlineLearningModule`
/// queda fuera:
///
/// - Se entrenaba con `realized_ret − ml_at_entry`, un retorno fraccional
///   (≈ ±0,005) menos una probabilidad (≈ 0,3–0,5). El error fue negativo en el
///   100 % de los cierres medidos y el residuo pasó el 99 % del tiempo en su
///   suelo de −0,15, restando 0,15 a toda predicción: `ml_prob < ½` en el 99,5 %
///   de las evaluaciones y el escudo neuronal vetando casi todos los largos
///   (D-692).
/// - Con el error en unidades de probabilidad tampoco converge: la ganancia de
///   Kalman por rasgo, con momentum 0,9 y rasgos correlacionados, oscila con
///   amplitud muy superior a su cota, así que sólo añadiría ruido de ±0,15.
///
/// Reincorporarlo exige un estimador que converja (mínimos cuadrados recursivos
/// sobre el vector completo, con el error del predictor completo) y su propia
/// validación walk-forward. La calibración con resultados reales ya la hace
/// `PlattCalibrator` (D-619).
pub fn compose_ml_prob(ensemble_prob: f64, spot_bias: f64) -> f64 {
    let p = if ensemble_prob.is_finite() { ensemble_prob } else { 0.5 };
    let b = if spot_bias.is_finite() { spot_bias } else { 0.0 };
    (p + b).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d693_ml_prob_es_el_ensamble_mas_el_sesgo_spot() {
        assert_eq!(compose_ml_prob(0.45, 0.0), 0.45);
        assert_eq!(compose_ml_prob(0.95, 0.15), 1.0);
        assert_eq!(compose_ml_prob(0.05, -0.15), 0.0);
        assert_eq!(compose_ml_prob(f64::NAN, 0.0), 0.5);
        assert_eq!(compose_ml_prob(0.6, f64::NAN), 0.6);
    }

    /// D-693: por qué el residuo online no puede entrar en la probabilidad. Con
    /// el error que usaba el núcleo (retorno − probabilidad), y aunque las
    /// operaciones ganen y pierdan por igual, su predicción se satura en negativo.
    #[test]
    fn d693_el_error_en_unidades_mezcladas_satura_el_residuo() {
        use metacortex_engine::online_learning::OnlineLearningModule;
        let mut features = [0.0f32; 64];
        for f in features.iter_mut().take(12) {
            *f = 0.5;
        }
        let mut learner = OnlineLearningModule::new(0.001, 0.9);
        for i in 0..400 {
            let realized_ret = if i % 2 == 0 { 0.005 } else { -0.005 };
            let ml_at_entry = 0.5 + (learner.predict(&features) as f64).clamp(-0.15, 0.15);
            learner.update_weights_with_kalman_adaptive_vol(
                &features,
                (realized_ret - ml_at_entry) as f32,
                0.1,
                0.001,
            );
        }
        assert!(
            learner.predict(&features) < -0.15,
            "predicción del residuo {}",
            learner.predict(&features)
        );
    }

    /// Media posterior Beta de D-680 para `n` observaciones en una única
    /// puntuación `s` con `wins` aciertos.
    fn beta_mean(s: f64, wins: f64, n: f64) -> f64 {
        (PRIOR_PSEUDO_OBSERVATIONS * s + wins) / (PRIOR_PSEUDO_OBSERVATIONS + n)
    }

    #[test]
    fn sin_datos_es_la_identidad() {
        let c = PlattCalibrator::new();
        for &s in &[0.2, 0.55, 0.7, 0.9] {
            assert!((c.calibrate(s) - s).abs() < 1e-9, "s = {s}");
        }
    }

    #[test]
    fn pocas_operaciones_mueven_el_mapa_como_el_prior_beta() {
        let mut c = PlattCalibrator::new();
        for _ in 0..3 {
            c.update(0.8, false);
        }
        let p = c.calibrate(0.8);
        let esperado = beta_mean(0.8, 0.0, 3.0);
        assert!((p - esperado).abs() < 1e-3, "tres pérdidas a 0,8: {p} frente a {esperado}");
    }

    #[test]
    fn aprende_una_puntuacion_sobreconfiada() {
        // La puntuación dice 0,8, pero sólo se gana la mitad de las veces.
        let mut c = PlattCalibrator::new();
        for i in 0..400 {
            c.update(0.8, i % 2 == 0);
        }
        let p = c.calibrate(0.8);
        let esperado = beta_mean(0.8, 200.0, 400.0);
        assert!((p - esperado).abs() < 1e-3, "400 observaciones: {p} frente a {esperado}");
    }

    #[test]
    fn conserva_el_orden_de_las_senales() {
        let mut c = PlattCalibrator::new();
        for i in 0..300 {
            // Puntuaciones altas ganan más a menudo que las bajas.
            let s = if i % 3 == 0 { 0.9 } else { 0.6 };
            let won = if s > 0.8 { i % 4 != 0 } else { i % 3 == 1 };
            c.update(s, won);
        }
        assert!(c.calibrate(0.9) > c.calibrate(0.6));
        assert!(c.coefficients().0 >= 0.0);
    }

    #[test]
    fn una_puntuacion_sin_informacion_se_aplana_hacia_la_tasa_base() {
        let mut c = PlattCalibrator::new();
        for i in 0..500 {
            // Alta o baja, se gana el 40 % de las veces.
            let s = if i % 2 == 0 { 0.9 } else { 0.55 };
            c.update(s, i % 5 < 2);
        }
        let (hi, lo) = (c.calibrate(0.9), c.calibrate(0.55));
        assert!((hi - lo).abs() < 0.02, "sin información no debe separar: {hi} vs {lo}");
        assert!((hi - 0.4).abs() < 0.02 && (lo - 0.4).abs() < 0.02);
    }
}
