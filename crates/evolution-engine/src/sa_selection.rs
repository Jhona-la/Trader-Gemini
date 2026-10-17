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
        // SOL-F9-2026-10-16: la temperatura es el denominador de la
        // razón de Metropolis. T inválida (negativa, -infinito, NaN)
        // invierte el signo de la divergencia — un candidato PEOR se
        // acepta con p = exp(+|Δ|/T) > 1 (siempre) — o congela la
        // búsqueda (NaN ⇒ comparación falsa). Dominio físico de la
        // temperatura: T > 0. T = 0 (greedy) y T = +inf (aleatorio)
        // son límites correctos que la condición `temperature > 0.0`
        // preserva: 0 > 0 es falso (degradan a greedy, que es lo que
        // T→0+ hacía ya por exp(-inf) = 0) y +inf > 0 es verdadero
        // (conserva Metropolis con p→1). Con T ≤ 0 o NaN se degrada
        // a GREEDY: sólo mejoras estrictas, que es el límite honesto.
        let accepted = match self.current {
            None => true,
            Some(current) if score > current => true,
            Some(_) if !(temperature > 0.0) => false, // SOL-F9: T inválida ⇒ greedy
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

// ── SOL-F9-2026-10-16: la temperatura del SA debe tener un límite con
// sentido ─────────────────────────────────────────────────────────────
//
// Hallazgo: `consider` divide por `temperature` sin validarla. Con
// T <= 0 el cociente (score - current)/T invierte el signo de la
// divergencia: un candid PEOR se acepta con probabilidad exp(+|Δ|/T)
// > 1 (siempre), o con T = 0 exacto el cociente es NaN y la comparación
// `random() < NaN` es falsa — la búsqueda se congela en greedy. Con los
// constantes actuales del bin (initial_temp=100, cooling 0.9^50) T
// nunca llega a 0, pero el día que alguien parametrice el cooling (los
// valores están hardcodeados en evolution.rs:250-253) el límite se
// activa. Contrato: T no finita o <= 0 degrada a GREEDY (sólo mejoras
// estrictas), que es el límite correcto de T -> 0+.
#[cfg(test)]
mod sol_f9_temp_tests {
    use super::*;

    fn peor_score() -> f64 {
        1.0
    }

    #[test]
    fn sol_f9_temperatura_invalida_degrada_a_greedy_no_invierte_la_aceptacion() {
        // Candidato PEOR que el incumbent, con temperaturas INVÁLIDAS
        // (negativas, -infinito, NaN): debe RECHAZARSE siempre (greedy),
        // nunca aceptarse. Con T<0 el cociente (score-current)/T invierte
        // el signo: exp(+|Δ|/T) > 1 ⇒ el peor se acepta SIEMPRE.
        for temp in [f64::NAN, f64::NEG_INFINITY, -1.0, -100.0] {
            let mut st = SaScoreState::new();
            assert!(st.consider(10.0, 10.0, || 0.0).accepted, "incumbent inicial");
            let d = st.consider(peor_score(), temp, || 0.0); // peor, con T inválida
            assert!(
                !d.accepted,
                "T={temp}: un candidato peor NO puede aceptarse (greedy es el limite de T->0+)"
            );
            assert_eq!(st.current_score(), Some(10.0), "T={temp}: el incumbent no cambia");
        }
    }

    #[test]
    fn sol_f9_temperatura_cero_y_mas_infinito_son_limites_no_defectos() {
        // T = 0 es el límite greedy correcto (rechaza el peor); T = +inf
        // es el límite aleatorio correcto (acepta con p -> 1). NO se
        // degradan: son el dominio físico de la temperatura, no basura.
        let mut st0 = SaScoreState::new();
        assert!(st0.consider(10.0, 10.0, || 0.0).accepted);
        let d0 = st0.consider(1.0, 0.0, || 0.0); // peor con T=0
        assert!(!d0.accepted, "T=0: greedy, el peor se rechaza");

        let mut sti = SaScoreState::new();
        assert!(sti.consider(10.0, 10.0, || 0.0).accepted);
        let di = sti.consider(1.0, f64::INFINITY, || 0.0); // peor con T=+inf
        assert!(di.accepted, "T=+inf: p->1, el peor se acepta (limite aleatorio)");
    }

    #[test]
    fn sol_f9_temperatura_valida_conserva_metropolis() {
        // Con T válida, la regla de Metropolis sigue viva: un candidato
        // peor se acepta si la muestra cae bajo exp(Δ/T).
        let mut st = SaScoreState::new();
        assert!(st.consider(10.0, 10.0, || 0.0).accepted);
        // Δ = 1 - 10 = -9, T = 1 ⇒ p = e^-9 ≈ 1.23e-4.
        let d_acepta = st.consider(1.0, 1.0, || 0.0); // muestra 0 < p
        assert!(d_acepta.accepted, "muestra 0 < e^-9 debe aceptar");
        let mut st2 = SaScoreState::new();
        assert!(st2.consider(10.0, 10.0, || 0.0).accepted);
        let d_rechaza = st2.consider(1.0, 1.0, || 0.5); // muestra 0.5 > p
        assert!(!d_rechaza.accepted, "muestra 0.5 > e^-9 debe rechazar");
    }

    #[test]
    fn sol_f9_mejora_estricta_siempre_acepta_con_cualquier_temperatura() {
        // Una mejora estricta se acepta con ANY T (válida o inválida):
        // el greedy del incumbent no depende del límite.
        for temp in [0.0_f64, -1.0, f64::NAN, 1.0] {
            let mut st = SaScoreState::new();
            assert!(st.consider(10.0, 10.0, || 0.0).accepted);
            let d = st.consider(11.0, temp, || 1.0);
            assert!(d.accepted, "T={temp}: la mejora estricta debe aceptarse");
            assert!(d.improved_best);
        }
    }
}
