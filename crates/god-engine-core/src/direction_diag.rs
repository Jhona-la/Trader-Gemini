//! Diagnóstico por dirección del embudo de entrada (sólo telemetría).
//!
//! En las corridas forenses de la Décima Ola el motor abrió un único largo en
//! unas 140 operaciones, también en la ventana de validación, en la que BTC
//! sube un 4 %. Las compuertas del núcleo están escritas en espejo, así que el
//! sesgo tiene que venir de sus entradas. Este módulo cuenta, por dirección, qué
//! propone el consenso tensorial, qué condición del gate de la rama 11 falla y
//! qué se abre, junto con la distribución de `composite_score` y `ml_prob` y la
//! contribución media de cada término. No participa en ninguna decisión.

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

const COMPOSITE_BINS: usize = 20;
const ML_BINS: usize = 10;
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
    /// Aperturas por dirección.
    pub opened: [u64; 2],
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
        let c = ((composite.clamp(-1.0, 1.0) + 1.0) * 0.5 * COMPOSITE_BINS as f64) as usize;
        self.composite_hist[c.min(COMPOSITE_BINS - 1)] += 1;
        let m = (ml_prob.clamp(0.0, 1.0) * ML_BINS as f64) as usize;
        self.ml_prob_hist[m.min(ML_BINS - 1)] += 1;
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

    #[inline]
    pub fn record_open(&mut self, is_long: bool) {
        self.opened[if is_long { LONG } else { SHORT }] += 1;
    }

    /// Informe legible, una línea por hecho, con el prefijo `DIRECTION_DIAG`.
    pub fn report(&self) -> String {
        let n = self.evaluations.max(1) as f64;
        let positive: u64 = self.composite_hist[COMPOSITE_BINS / 2..].iter().sum();
        let ml_up: u64 = self.ml_prob_hist[ML_BINS / 2..].iter().sum();
        let mut out = String::new();
        out.push_str(&format!(
            "DIRECTION_DIAG evaluaciones={} · composite≥0 {:.1} % · ml_prob≥0,5 {:.1} % · contribución media micro={:+.4} red={:+.4} tensor={:+.4}\n",
            self.evaluations,
            100.0 * positive as f64 / n,
            100.0 * ml_up as f64 / n,
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
}
