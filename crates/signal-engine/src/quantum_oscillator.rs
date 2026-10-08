use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;
use crate::voto_espectral::VotoEspectral;

pub const ESCALAS_OSCILADOR: usize = 32;

/// ⚛️ ALGORITMO #80: SUAVIZADOR POR OSCILADOR ANARMÓNICO CUÁNTICO (QUANTUM OSCILLATOR ENGINE)
/// Simula el comportamiento del estado fundamental del precio en un pozo de potencial anarmónico V(x) = 1/2 k x^2 + lambda x^4,
/// filtrando ruido estocástico no-lineal con cero desfase de fase.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct QuantumOscillatorEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for QuantumOscillatorEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumOscillatorEngine").finish()
    }
}

impl QuantumOscillatorEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Calcula la fuerza del pozo de potencial anarmónico en O(1)
    #[inline(always)]
    pub fn compute_quantum_restoring_force(
        position: f64,
        k_spring: f64,
        lambda_anharmonic: f64,
    ) -> f64 {
        // FIX #648: Sanitizar y acotar posición para evitar desbordamiento cúbico x^3
        let safe_x = if position.is_finite() {
            position.clamp(-10.0, 10.0)
        } else {
            0.0
        };
        let safe_k = if k_spring.is_finite() && k_spring >= 0.0 {
            k_spring
        } else {
            1.0
        };
        let safe_l = if lambda_anharmonic.is_finite() && lambda_anharmonic >= 0.0 {
            lambda_anharmonic
        } else {
            0.1
        };

        let res = -(safe_k * safe_x + 4.0 * safe_l * safe_x * safe_x * safe_x);
        if res.is_finite() {
            res
        } else {
            0.0
        }
    }

    /// Calcula la probabilidad de colapso en estado de superposición cuántica $|\psi(x)|^2$ (Punto #260)
    #[inline(always)]
    pub fn compute_superposition_probability(position: f64, mass_freq_alpha: f64) -> f64 {
        let safe_x = if position.is_finite() {
            position.clamp(-10.0, 10.0)
        } else {
            0.0
        };
        let safe_alpha = if mass_freq_alpha.is_finite() && mass_freq_alpha > 0.0 {
            mass_freq_alpha.clamp(0.01, 10.0)
        } else {
            1.0
        };

        let norm = (safe_alpha / std::f64::consts::PI).sqrt();
        let prob = norm * (-safe_alpha * safe_x * safe_x).exp();
        if prob.is_finite() {
            prob.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// #609 (Ola 31) — VOTO ESPECTRAL del pozo: la fuerza restauradora con
    /// el confinamiento de AGY-P14, evaluada en el DESPLAZAMIENTO DE CADA
    /// ESCALA x(τ) de la malla del TemporalSpectrum (momentum_z por banda).
    /// Es la refactorización espectral del motor: el voto deja de ser el
    /// escalar del libro actual (`quantum_position_deviation`) y pasa a ser
    /// un espectro que τ* del consejo puede rebanar. Puro y sin estado —
    /// la sombra observacional del core lo computa AL LADO del voto vivo
    /// (bit a bit intacto hasta el cambio ordenado con oráculo).
    pub fn voto_espectral(
        desplazamientos: &[f64; ESCALAS_OSCILADOR],
    ) -> VotoEspectral {
        // G2-11 (F3-A4): los knobs del registro (quantum_k_spring /
        // lambda_anharmonic / alpha) no tenían ESCRITOR ni gen — lecturas
        // decorativas congeladas a los defaults. Retiradas: consts internas
        // bit-idénticas, la ilusión de configurabilidad eliminada.
        const K_SPRING: f64 = 1.0;
        const LAMBDA_ANHARMONIC: f64 = 0.1;
        const ALPHA_CONFINAMIENTO: f64 = 0.5;
        let alpha = ALPHA_CONFINAMIENTO;
        let safe_alpha = if alpha.is_finite() && alpha > 0.0 {
            alpha.clamp(0.01, 10.0)
        } else {
            0.5
        };
        VotoEspectral::desde_espectro(desplazamientos, |x| {
            let safe_x = if x.is_finite() { x.clamp(-10.0, 10.0) } else { 0.0 };
            let force = Self::compute_quantum_restoring_force(safe_x, K_SPRING, LAMBDA_ANHARMONIC);
            // #650 (Ola 50) — PARIDAD con el vivo (AGY-P14): la MISMA
            // envolvente C(x) = e^{−α·x²} ∈ (0,1] que modula la fuerza del
            // motor vivo. La sombra usaba √|ψ|² = (α/π)^{1/4}·e^{−αx²/2}:
            // exponente a la MITAD (supresión de ruptura 3-4 órdenes más
            // débil que el vivo, p.ej. x=6, α=0.5: e^{−9} vs e^{−18}) y
            // una constante (α/π)^{1/4} que supera 1 para α > π (rompía la
            // cota antes del clamp).
            let confinement = (-safe_alpha * safe_x * safe_x).exp().clamp(0.0, 1.0);
            (force * confinement).clamp(-1.0, 1.0)
        })
    }
}

impl QuantumStrategy for QuantumOscillatorEngine {
    fn name(&self) -> &str {
        "QuantumOscillatorEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() {
            None
        } else {
            Some(symbol)
        };
        let cid_opt = if symbol.is_empty() {
            None
        } else {
            Some(coin_id)
        };
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };
        let pos = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "quantum_position_deviation",
                "QuantumOscillatorEngine",
            )
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_book_imbalance",
                    "QuantumOscillatorEngine",
                )
            })
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_imbalance",
                    "QuantumOscillatorEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        // G2-11 (F3-A4): knobs sin escritor NI gen — retiradas del registro;
        // consts bit-idénticas a los defaults congelados (paridad por
        // construcción entre vivo y sombra, ahora evidente).
        const K_SPRING: f64 = 1.0;
        const LAMBDA_ANHARMONIC: f64 = 0.1;
        const ALPHA_CONFINAMIENTO: f64 = 0.5;
        let k_spring = K_SPRING;
        let lambda = LAMBDA_ANHARMONIC;
        let alpha = ALPHA_CONFINAMIENTO;

        if !pos.is_finite() {
            return 0.0;
        }
        let force = Self::compute_quantum_restoring_force(pos, k_spring, lambda);
        // AGY-AUD-P14: Modulación por envolvente de confinamiento cuántico:
        // C(x) = exp(-alpha * x^2). En el pozo confinado (x moderado), la fuerza restauradora
        // rige la reversión a la media. En estados del continuo / ruptura cuántica (|x| extremo),
        // C(x) tiende a 0, evitando que el oscilador luche suicidamente contra rupturas supersónicas.
        let confinement = (-alpha * pos * pos).exp().clamp(0.0, 1.0);
        (force * confinement).clamp(-1.0, 1.0)
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_restoring_force() {
        let force = QuantumOscillatorEngine::compute_quantum_restoring_force(1.0, 1.0, 0.5);
        assert_eq!(force, -(1.0 + 4.0 * 0.5));
    }

    #[test]
    fn test_quantum_superposition_probability() {
        let prob_center = QuantumOscillatorEngine::compute_superposition_probability(0.0, 1.0);
        assert!(prob_center > 0.0 && prob_center <= 1.0);

        let prob_tail = QuantumOscillatorEngine::compute_superposition_probability(3.0, 1.0);
        assert!(prob_tail < prob_center);
        assert!(prob_tail >= 0.0);

        let prob_nan = QuantumOscillatorEngine::compute_superposition_probability(f64::NAN, 1.0);
        assert!(prob_nan.is_finite());
    }

    #[test]
    fn test_quantum_oscillator_engine_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("quantum_position_deviation", 0.5);
        registry.set("quantum_k_spring", 1.0);
        registry.set("quantum_lambda_anharmonic", 0.1);

        let mut engine = QuantumOscillatorEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        assert!(
            eval < 0.0,
            "Desviación positiva debe generar fuerza restauradora negativa"
        );
        assert!(eval >= -1.0 && eval <= 1.0);
    }

    #[test]
    fn test_quantum_oscillator_breakout_suppression() {
        let registry = Arc::new(OmniscientRegistry::new());
        // Desviación extrema de breakout (pos = 6.0)
        registry.set("quantum_position_deviation", 6.0);
        registry.set("quantum_k_spring", 1.0);
        registry.set("quantum_lambda_anharmonic", 0.1);
        registry.set("quantum_alpha", 0.5);

        let mut engine = QuantumOscillatorEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        // Confinamiento exp(-0.5 * 36) = exp(-18) < 1e-7: la fuerza restauradora debe amortiguarse a ~0
        assert!(
            eval.abs() < 1e-4,
            "Breakout cuántico extremo debe tener voto amortiguado, no luchar contra la tendencia: {eval}"
        );
    }
}

#[cfg(test)]
mod qo_609_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;

    #[test]
    fn qo_609_voto_espectral_antisimetrico_y_confinado() {
        let mut x_pos = [0.0; ESCALAS_VOTO];
        let mut x_neg = [0.0; ESCALAS_VOTO];
        let mut x_extremo = [0.0; ESCALAS_VOTO];
        for (k, v) in x_pos.iter_mut().enumerate() {
            *v = 0.1 * (k as f64 + 1.0); // desplazamientos crecientes
            x_neg[k] = -*v;
            x_extremo[k] = 10.0; // estado del continuo (ruptura)
        }
        let voto_pos = QuantumOscillatorEngine::voto_espectral(&x_pos);
        let voto_neg = QuantumOscillatorEngine::voto_espectral(&x_neg);
        let voto_extremo = QuantumOscillatorEngine::voto_espectral(&x_extremo);
        // Antisimetría: x → −x ⇒ voto → −voto (el pozo es impar).
        for kk in 0..ESCALAS_VOTO {
            let vp = voto_pos.en_escala(kk);
            let vn = voto_neg.en_escala(kk);
            assert!((vp + vn).abs() < 1e-12, "escala {}: {} vs {}", kk, vp, vn);
            assert!(vp.abs() <= 1.0);
        }
        // Confinamiento AGY-P14: en |x|=10 el voto se amortigua a ~0.
        for kk in 0..ESCALAS_VOTO {
            assert!(voto_extremo.en_escala(kk).abs() < 1e-3);
        }
        // Desplazamientos nulos ⇒ voto nulo.
        let cero = QuantumOscillatorEngine::voto_espectral(&[0.0; ESCALAS_VOTO]);
        assert_eq!(cero.dominante(), None, "sin convicción no hay dominante");
    }

    #[test]
    fn qo_609_confinamiento_amortigua_la_escala_extrema() {
        // AGY-P14 por escala: la misma desviación extrema (estado del
        // continuo) vota MÁS DEBIL que la moderada — el oscilador no lucha
        // contra rupturas, escala a escala.
        let mut x = [0.0; ESCALAS_VOTO];
        x[5] = 1.0; // desplazamiento moderado (confinado)
        x[25] = 6.0; // desplazamiento extremo (ruptura)
        let voto = QuantumOscillatorEngine::voto_espectral(&x);
        assert!(
            voto.en_escala(25).abs() < voto.en_escala(5).abs(),
            "la escala de ruptura debe votar más débil: moderada={} extrema={}",
            voto.en_escala(5).abs(),
            voto.en_escala(25).abs()
        );
    }
}
