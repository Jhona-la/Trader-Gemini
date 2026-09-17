//! Diagnóstico por dirección del embudo de entrada (sólo telemetría).
//!
//! En las corridas forenses de la Décima Ola el motor abrió un único largo en
//! unas 140 operaciones, también en la ventana de validación, en la que BTC
//! sube un 4 %. Las compuertas del núcleo están escritas en espejo, así que el
//! sesgo tiene que venir de sus entradas. Este módulo cuenta, por dirección, qué
//! propone el consenso tensorial, qué condición del gate de la rama 11 falla,
//! qué escudo del embudo unificado veta la intención, qué rechaza el risk-engine
//! o el consejo y qué se abre. También registra la distribución de
//! `composite_score` y `ml_prob`, la contribución media de cada término, la
//! descomposición de `ml_prob` (base del ensamble, residuo online, bosque y red)
//! y el error con el que se entrena el residuo online. No participa en ninguna
//! decisión.

use strategy_core::SignalType;

/// Número de condiciones del gate de la rama 11.
pub const N_BRANCH11_CONDITIONS: usize = 9;

/// Condiciones del gate de la rama 11, en el mismo orden que en el núcleo.
pub const BRANCH11_CONDITIONS: [&str; N_BRANCH11_CONDITIONS] = [
    "racha direccional < 2",
    "sin tendencia confirmada en contra",
    "sin momentum adverso",
    "tendencia superior y secular no en contra",
    "sin estiramiento extremo en contra",
    "tendencia superior dentro de su cota",
    "composite_score supera el umbral",
    "OBI supera el umbral de rango",
    "sin sobreextensión",
];

/// Número de etapas del embudo unificado entre la intención y la orden.
pub const N_FUNNEL_STAGES: usize = 7;

/// Etapas del embudo unificado, en el orden en que el núcleo las aplica.
pub const FUNNEL_STAGES: [&str; N_FUNNEL_STAGES] = [
    "escudo macro (D-624)",
    "acondicionamiento espectral (X-016)",
    "anti-whiplash (D-463)",
    "convicción bayesiana (D-472)",
    "racha direccional (D-499)",
    "escudo de libro L2 (D-475/D-688)",
    "escudo neuronal (D-473/D-688)",
];

pub const STAGE_MACRO: usize = 0;
pub const STAGE_SPECTRAL: usize = 1;
pub const STAGE_WHIPLASH: usize = 2;
pub const STAGE_BAYES: usize = 3;
pub const STAGE_STREAK: usize = 4;
pub const STAGE_L2: usize = 5;
pub const STAGE_NEURAL: usize = 6;

const COMPOSITE_BINS: usize = 20;
const ML_BINS: usize = 10;
/// Celdas del histograma del residuo online, sobre la cota con la que el núcleo
/// lo suma a `ml_prob` (±0,15).
const RESIDUAL_BINS: usize = 10;
const RESIDUAL_BOUND: f64 = 0.15;
/// Celdas del histograma del error de entrenamiento del residuo, sobre [−1, 1].
const UPDATE_BINS: usize = 10;
/// Celdas del histograma de confianza en la compuerta del risk-engine, sobre [0, 1].
const RISK_CONF_BINS: usize = 20;
const LONG: usize = 0;
const SHORT: usize = 1;

#[derive(Debug, Clone, Default)]
pub struct DirectionDiag {
    /// Evaluaciones de la fusión bayesiana con valores finitos.
    pub evaluations: u64,
    /// Histograma de `composite_score` en 20 celdas iguales de [−1, 1].
    pub composite_hist: [u64; COMPOSITE_BINS],
    /// Histograma de `ml_prob` en 10 celdas iguales de [0, 1].
    pub ml_prob_hist: [u64; ML_BINS],
    /// Suma de las contribuciones ponderadas a `composite_score`: micro, red y tensor.
    pub contribution_sum: [f64; 3],
    /// Propuestas del consenso tensorial: largo, corto y plano.
    pub tensor_signal: [u64; 3],
    /// Evaluaciones del gate de la rama 11 por dirección.
    pub gate_reached: [u64; 2],
    pub gate_passed: [u64; 2],
    pub gate_failed: [[u64; N_BRANCH11_CONDITIONS]; 2],
    /// Veces que la condición fue la única en fallar.
    pub gate_sole_failure: [[u64; N_BRANCH11_CONDITIONS]; 2],
    /// Intenciones que llegan al embudo unificado sin posición abierta.
    pub funnel_entered: [u64; 2],
    /// Vetos de cada etapa del embudo unificado.
    pub funnel_veto: [[u64; N_FUNNEL_STAGES]; 2],
    /// Intenciones que atraviesan todo el embudo unificado.
    pub funnel_survived: [u64; 2],
    /// Rechazos del risk-engine y del consejo.
    pub risk_rejected: [u64; 2],
    pub council_vetoed: [u64; 2],
    /// Confianza con la que la intención llega al risk-engine:
    /// [dirección][0 rechazada, 1 aceptada][celda].
    pub risk_confidence_hist: [[[u64; RISK_CONF_BINS]; 2]; 2],
    /// Aperturas por dirección.
    pub opened: [u64; 2],
    /// D-695: evaluaciones en que el modelo iba contra la intención, y cuántas
    /// con habilidad demostrada (las únicas en que veta).
    pub neural_against: [u64; 2],
    pub neural_against_skilled: [u64; 2],
    /// Descomposición de `ml_prob`: base del ensamble y residuo online acotado.
    pub ml_components_n: u64,
    pub ml_base_hist: [u64; ML_BINS],
    pub ml_residual_hist: [u64; RESIDUAL_BINS],
    pub ml_residual_sum: f64,
    /// Predicciones de cada modelo del ensamble cuando opinan.
    pub forest_n: u64,
    pub forest_hist: [u64; ML_BINS],
    pub nn_n: u64,
    pub nn_hist: [u64; ML_BINS],
    /// Error con el que se entrena el residuo online en cada cierre.
    pub online_update_n: u64,
    pub online_update_sum: f64,
    pub online_update_hist: [u64; UPDATE_BINS],
    /// Estado de la evaluación en curso del embudo unificado.
    funnel_active: bool,
    funnel_last: SignalType,
}

fn dir_index(signal: SignalType) -> Option<usize> {
    match signal {
        SignalType::Long => Some(LONG),
        SignalType::Short => Some(SHORT),
        SignalType::Flat => None,
    }
}

/// Celda de un histograma de `bins` celdas iguales sobre [lo, hi].
fn bin(value: f64, lo: f64, hi: f64, bins: usize) -> usize {
    let x = ((value.clamp(lo, hi) - lo) / (hi - lo) * bins as f64) as usize;
    x.min(bins - 1)
}

fn share(hist: &[u64], from: usize, n: u64) -> f64 {
    100.0 * hist[from..].iter().sum::<u64>() as f64 / n.max(1) as f64
}

impl DirectionDiag {
    #[inline]
    pub fn record_evaluation(
        &mut self,
        composite: f64,
        ml_prob: f64,
        contributions: [f64; 3],
        tensor: SignalType,
    ) {
        if !composite.is_finite() || !ml_prob.is_finite() {
            return;
        }
        self.evaluations += 1;
        self.composite_hist[bin(composite, -1.0, 1.0, COMPOSITE_BINS)] += 1;
        self.ml_prob_hist[bin(ml_prob, 0.0, 1.0, ML_BINS)] += 1;
        for (sum, value) in self.contribution_sum.iter_mut().zip(contributions) {
            if value.is_finite() {
                *sum += value;
            }
        }
        let idx = match tensor {
            SignalType::Long => 0,
            SignalType::Short => 1,
            SignalType::Flat => 2,
        };
        self.tensor_signal[idx] += 1;
    }

    /// Descompone `ml_prob = base + residuo acotado + sesgo spot`.
    #[inline]
    pub fn record_ml_components(
        &mut self,
        base: f64,
        residual: f64,
        forest: Option<f64>,
        nn: Option<f64>,
    ) {
        if base.is_finite() && residual.is_finite() {
            self.ml_components_n += 1;
            self.ml_base_hist[bin(base, 0.0, 1.0, ML_BINS)] += 1;
            self.ml_residual_hist[bin(residual, -RESIDUAL_BOUND, RESIDUAL_BOUND, RESIDUAL_BINS)] += 1;
            self.ml_residual_sum += residual;
        }
        if let Some(p) = forest.filter(|p| p.is_finite()) {
            self.forest_n += 1;
            self.forest_hist[bin(p, 0.0, 1.0, ML_BINS)] += 1;
        }
        if let Some(p) = nn.filter(|p| p.is_finite()) {
            self.nn_n += 1;
            self.nn_hist[bin(p, 0.0, 1.0, ML_BINS)] += 1;
        }
    }

    /// Error con el que se actualiza el residuo online al cerrar una operación.
    #[inline]
    pub fn record_online_update(&mut self, td_error: f64) {
        if !td_error.is_finite() {
            return;
        }
        self.online_update_n += 1;
        self.online_update_sum += td_error;
        self.online_update_hist[bin(td_error, -1.0, 1.0, UPDATE_BINS)] += 1;
    }

    #[inline]
    pub fn record_gate(&mut self, is_long: bool, conditions: &[bool; N_BRANCH11_CONDITIONS]) {
        let d = if is_long { LONG } else { SHORT };
        self.gate_reached[d] += 1;
        let failures = conditions.iter().filter(|&&ok| !ok).count();
        if failures == 0 {
            self.gate_passed[d] += 1;
            return;
        }
        for (i, &ok) in conditions.iter().enumerate() {
            if !ok {
                self.gate_failed[d][i] += 1;
                if failures == 1 {
                    self.gate_sole_failure[d][i] += 1;
                }
            }
        }
    }

    /// Abre la evaluación del embudo unificado. Sólo cuenta si hay intención y
    /// no hay posición abierta: con posición abierta nada podría abrirse.
    #[inline]
    pub fn funnel_begin(&mut self, signal: SignalType, book_flat: bool) {
        self.funnel_active = book_flat && dir_index(signal).is_some();
        self.funnel_last = signal;
        if let (true, Some(d)) = (self.funnel_active, dir_index(signal)) {
            self.funnel_entered[d] += 1;
        }
    }

    /// Registra el estado tras la etapa `stage`: si la intención pasó de tener
    /// dirección a plana, esa etapa la vetó.
    #[inline]
    pub fn funnel_checkpoint(&mut self, signal: SignalType, stage: usize) {
        if !self.funnel_active || stage >= N_FUNNEL_STAGES {
            return;
        }
        if let Some(d) = dir_index(self.funnel_last) {
            if signal == SignalType::Flat {
                self.funnel_veto[d][stage] += 1;
                self.funnel_active = false;
            }
        }
        self.funnel_last = signal;
        if stage == N_FUNNEL_STAGES - 1 {
            if let (true, Some(d)) = (self.funnel_active, dir_index(signal)) {
                self.funnel_survived[d] += 1;
            }
            self.funnel_active = false;
        }
    }

    #[inline]
    pub fn record_risk(&mut self, is_long: bool, confidence: f64, accepted: bool) {
        let d = if is_long { LONG } else { SHORT };
        if !accepted {
            self.risk_rejected[d] += 1;
        }
        if confidence.is_finite() {
            self.risk_confidence_hist[d][accepted as usize][bin(confidence, 0.0, 1.0, RISK_CONF_BINS)] += 1;
        }
    }

    #[inline]
    pub fn record_council(&mut self, is_long: bool, approved: bool) {
        if !approved {
            self.council_vetoed[if is_long { LONG } else { SHORT }] += 1;
        }
    }

    #[inline]
    pub fn record_neural_gate(&mut self, is_long: bool, skilled: bool) {
        let d = if is_long { LONG } else { SHORT };
        self.neural_against[d] += 1;
        if skilled {
            self.neural_against_skilled[d] += 1;
        }
    }

    #[inline]
    pub fn record_open(&mut self, is_long: bool) {
        self.opened[if is_long { LONG } else { SHORT }] += 1;
    }

    /// Informe legible, una línea por hecho, con el prefijo `DIRECTION_DIAG`.
    pub fn report(&self) -> String {
        let n = self.evaluations.max(1) as f64;
        let mut out = String::new();
        out.push_str(&format!(
            "DIRECTION_DIAG evaluaciones={} · composite≥0 {:.1} % · ml_prob≥0,5 {:.1} % · contribución media micro={:+.4} red={:+.4} tensor={:+.4}\n",
            self.evaluations,
            share(&self.composite_hist, COMPOSITE_BINS / 2, self.evaluations),
            share(&self.ml_prob_hist, ML_BINS / 2, self.evaluations),
            self.contribution_sum[0] / n,
            self.contribution_sum[1] / n,
            self.contribution_sum[2] / n,
        ));
        out.push_str(&format!(
            "DIRECTION_DIAG tensor largo={} corto={} plano={}\n",
            self.tensor_signal[0], self.tensor_signal[1], self.tensor_signal[2]
        ));
        out.push_str(&format!("DIRECTION_DIAG composite_hist={:?}\n", self.composite_hist));
        out.push_str(&format!("DIRECTION_DIAG ml_prob_hist={:?}\n", self.ml_prob_hist));
        out.push_str(&format!(
            "DIRECTION_DIAG ml_componentes n={} · base≥0,5 {:.1} % base_hist={:?} · residuo medio={:+.4} en el suelo −0,15 {:.1} % residuo_hist={:?}\n",
            self.ml_components_n,
            share(&self.ml_base_hist, ML_BINS / 2, self.ml_components_n),
            self.ml_base_hist,
            self.ml_residual_sum / self.ml_components_n.max(1) as f64,
            100.0 * self.ml_residual_hist[0] as f64 / self.ml_components_n.max(1) as f64,
            self.ml_residual_hist,
        ));
        out.push_str(&format!(
            "DIRECTION_DIAG ml_modelos bosque n={} ≥0,5 {:.1} % hist={:?} · red n={} ≥0,5 {:.1} % hist={:?}\n",
            self.forest_n,
            share(&self.forest_hist, ML_BINS / 2, self.forest_n),
            self.forest_hist,
            self.nn_n,
            share(&self.nn_hist, ML_BINS / 2, self.nn_n),
            self.nn_hist,
        ));
        out.push_str(&format!(
            "DIRECTION_DIAG residuo_online actualizaciones={} · error medio={:+.4} · error<0 {:.1} % · hist[−1,1]={:?}\n",
            self.online_update_n,
            self.online_update_sum / self.online_update_n.max(1) as f64,
            100.0 * self.online_update_hist[..UPDATE_BINS / 2].iter().sum::<u64>() as f64
                / self.online_update_n.max(1) as f64,
            self.online_update_hist,
        ));
        out.push_str(&format!(
            "DIRECTION_DIAG gate_rama11 largo alcanzado={} pasado={} · corto alcanzado={} pasado={}\n",
            self.gate_reached[LONG], self.gate_passed[LONG], self.gate_reached[SHORT], self.gate_passed[SHORT]
        ));
        for (i, name) in BRANCH11_CONDITIONS.iter().enumerate() {
            out.push_str(&format!(
                "DIRECTION_DIAG falla[{}] {} · largo={} (única {}) · corto={} (única {})\n",
                i,
                name,
                self.gate_failed[LONG][i],
                self.gate_sole_failure[LONG][i],
                self.gate_failed[SHORT][i],
                self.gate_sole_failure[SHORT][i],
            ));
        }
        out.push_str(&format!(
            "DIRECTION_DIAG embudo entradas sin posición · largo={} corto={}\n",
            self.funnel_entered[LONG], self.funnel_entered[SHORT]
        ));
        for (i, name) in FUNNEL_STAGES.iter().enumerate() {
            out.push_str(&format!(
                "DIRECTION_DIAG veto[{}] {} · largo={} corto={}\n",
                i, name, self.funnel_veto[LONG][i], self.funnel_veto[SHORT][i],
            ));
        }
        out.push_str(&format!(
            "DIRECTION_DIAG embudo superado · largo={} corto={} · rechazo risk-engine largo={} corto={} · veto consejo largo={} corto={}\n",
            self.funnel_survived[LONG],
            self.funnel_survived[SHORT],
            self.risk_rejected[LONG],
            self.risk_rejected[SHORT],
            self.council_vetoed[LONG],
            self.council_vetoed[SHORT],
        ));
        for (d, name) in [(LONG, "largo"), (SHORT, "corto")] {
            out.push_str(&format!(
                "DIRECTION_DIAG risk_confianza {} celdas de 0,05 · rechazadas={:?} · aceptadas={:?}\n",
                name, self.risk_confidence_hist[d][0], self.risk_confidence_hist[d][1],
            ));
        }
        out.push_str(&format!(
            "DIRECTION_DIAG escudo_neuronal en contra · largo={} (con habilidad {}) · corto={} (con habilidad {})\n",
            self.neural_against[LONG],
            self.neural_against_skilled[LONG],
            self.neural_against[SHORT],
            self.neural_against_skilled[SHORT],
        ));
        out.push_str(&format!(
            "DIRECTION_DIAG aperturas largo={} corto={}",
            self.opened[LONG], self.opened[SHORT]
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d695_cuenta_el_escudo_con_y_sin_habilidad() {
        let mut d = DirectionDiag::default();
        d.record_neural_gate(true, false);
        d.record_neural_gate(true, true);
        d.record_neural_gate(false, false);
        assert_eq!(d.neural_against, [2, 1]);
        assert_eq!(d.neural_against_skilled, [1, 0]);
        assert!(d.report().contains("escudo_neuronal en contra · largo=2 (con habilidad 1)"));
    }

    #[test]
    fn diag_cuenta_histogramas_y_fallos_por_direccion() {
        let mut d = DirectionDiag::default();
        d.record_evaluation(-1.0, 0.0, [0.1, -0.2, 0.3], SignalType::Short);
        d.record_evaluation(1.0, 1.0, [0.0; 3], SignalType::Long);
        d.record_evaluation(f64::NAN, 0.5, [0.0; 3], SignalType::Flat);
        assert_eq!(d.evaluations, 2);
        assert_eq!(d.composite_hist[0], 1);
        assert_eq!(d.composite_hist[COMPOSITE_BINS - 1], 1);
        assert_eq!(d.ml_prob_hist[0], 1);
        assert_eq!(d.ml_prob_hist[ML_BINS - 1], 1);
        assert_eq!(d.tensor_signal, [1, 1, 0]);

        let mut c = [true; N_BRANCH11_CONDITIONS];
        d.record_gate(true, &c);
        c[6] = false;
        d.record_gate(true, &c);
        c[7] = false;
        d.record_gate(false, &c);
        assert_eq!(d.gate_reached, [2, 1]);
        assert_eq!(d.gate_passed, [1, 0]);
        assert_eq!(d.gate_sole_failure[LONG][6], 1);
        assert_eq!(d.gate_failed[SHORT][6], 1);
        assert_eq!(d.gate_failed[SHORT][7], 1);
        assert_eq!(d.gate_sole_failure[SHORT][6], 0);

        d.record_open(false);
        assert_eq!(d.opened, [0, 1]);
        assert!(d.report().contains("DIRECTION_DIAG aperturas largo=0 corto=1"));
    }

    #[test]
    fn diag_atribuye_cada_veto_a_la_etapa_que_lo_produce() {
        let mut d = DirectionDiag::default();

        // Largo vetado por el escudo neuronal.
        d.funnel_begin(SignalType::Long, true);
        for stage in 0..STAGE_NEURAL {
            d.funnel_checkpoint(SignalType::Long, stage);
        }
        d.funnel_checkpoint(SignalType::Flat, STAGE_NEURAL);

        // Corto vetado por el escudo macro: las etapas siguientes no cuentan.
        d.funnel_begin(SignalType::Short, true);
        d.funnel_checkpoint(SignalType::Flat, STAGE_MACRO);
        for stage in 1..N_FUNNEL_STAGES {
            d.funnel_checkpoint(SignalType::Flat, stage);
        }

        // Corto que supera todo el embudo.
        d.funnel_begin(SignalType::Short, true);
        for stage in 0..N_FUNNEL_STAGES {
            d.funnel_checkpoint(SignalType::Short, stage);
        }

        // Con posición abierta no se cuenta nada.
        d.funnel_begin(SignalType::Long, false);
        d.funnel_checkpoint(SignalType::Flat, STAGE_MACRO);

        assert_eq!(d.funnel_entered, [1, 2]);
        assert_eq!(d.funnel_veto[LONG][STAGE_NEURAL], 1);
        assert_eq!(d.funnel_veto[SHORT][STAGE_MACRO], 1);
        assert_eq!(d.funnel_veto[LONG][STAGE_MACRO], 0);
        assert_eq!(d.funnel_survived, [0, 1]);

        d.record_risk(true, 0.72, false);
        d.record_risk(true, 0.80, true);
        d.record_council(false, false);
        assert_eq!(d.risk_rejected, [1, 0]);
        assert_eq!(d.risk_confidence_hist[LONG][0][14], 1);
        assert_eq!(d.risk_confidence_hist[LONG][1][16], 1);
        assert_eq!(d.council_vetoed, [0, 1]);
        assert!(d.report().contains("veto[6] escudo neuronal"));
    }

    #[test]
    fn diag_descompone_ml_prob_y_el_error_del_residuo() {
        let mut d = DirectionDiag::default();
        d.record_ml_components(0.45, -0.15, Some(0.6), None);
        d.record_ml_components(0.55, 0.15, None, Some(0.2));
        d.record_ml_components(f64::NAN, 0.0, None, None);
        assert_eq!(d.ml_components_n, 2);
        assert_eq!(d.ml_base_hist[4], 1);
        assert_eq!(d.ml_base_hist[5], 1);
        assert_eq!(d.ml_residual_hist[0], 1);
        assert_eq!(d.ml_residual_hist[RESIDUAL_BINS - 1], 1);
        assert_eq!((d.forest_n, d.nn_n), (1, 1));
        assert_eq!(d.forest_hist[6], 1);
        assert_eq!(d.nn_hist[2], 1);

        d.record_online_update(-0.4);
        d.record_online_update(0.01);
        d.record_online_update(f64::INFINITY);
        assert_eq!(d.online_update_n, 2);
        assert_eq!(d.online_update_hist[3], 1);
        assert_eq!(d.online_update_hist[5], 1);
        assert!(d.report().contains("DIRECTION_DIAG residuo_online actualizaciones=2"));
    }
}
