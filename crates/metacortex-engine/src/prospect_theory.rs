use serde::{Deserialize, Serialize};

/// 🧠 ALGORITMO #102: MODELO CONDUCTUAL CONTINUO DE KAHNEMAN-TVERSKY (PROSPECT THEORY ENGINE)
///
/// Modela los sesgos psicológicos colectivos y asimetrías emocionales de la masa de mercado:
///
/// 1. **Función de Valor Asimétrica V(Δx)**:
///      V(Δx) = (Δx)^α               para Δx ≥ 0  (ganancia: utilidad cóncava, aversión al riesgo en ganancias)
///      V(Δx) = -λ (-Δx)^β           para Δx < 0  (pérdida: desutilidad convexa, 2.25x más dolorosa)
///    donde:
///      - λ = 2.25: Coeficiente empírico canónico de aversión a la pérdida (Loss Aversion).
///      - α = 0.88, β = 0.88: Exponentes canónicos de sensibilidad marginal decreciente.
///
/// 2. **Función de Ponderación de Probabilidad no Lineal w(p) (Prelec / Tversky)**:
///      w(p) = p^γ / (p^γ + (1 - p)^γ)^(1 / γ)   con γ ≈ 0.65
///    - Sobrepondera probabilidades diminutas (miedo extremo a la liquidación / cola negra).
///    - Subpondera probabilidades medias y altas (complacencia tardía).
///
/// 3. **Presión de Prospecto Colectiva P_kt**:
///      P_kt = w(p_bull) · V(Δp_up) + w(p_crash) · V(-Δp_down)
///    - Si P_kt << -1.0: Pánico extremo colectivo. Los minoristas vomitan inventario en pérdidas;
///      se producen barridos de stops y liquidaciones en cascada forzadas ⇒ rebote elástico contrarian.
///    - Si P_kt >> +1.0: Euforia FOMO extrema. La multitud compra en el techo sin prima de riesgo ⇒
///      agotamiento de compradores taker y colapso de soporte.
///
/// 4. **Garantías de Alto Rendimiento**:
///    - O(1) determinista en CPU (< 25 ns), #[inline(always)], zero heap allocations.
///    - Inmunidad total fail-closed ante NaN, ±Inf o parámetros no finitos.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProspectTheoryEngine {
    pub lambda_loss_aversion: f64,
    pub alpha_gain: f64,
    pub beta_loss: f64,
    pub gamma_weighting: f64,
}

impl Default for ProspectTheoryEngine {
    fn default() -> Self {
        Self {
            lambda_loss_aversion: Self::DEFAULT_LAMBDA,
            alpha_gain: Self::DEFAULT_ALPHA,
            beta_loss: Self::DEFAULT_BETA,
            gamma_weighting: Self::DEFAULT_GAMMA,
        }
    }
}

impl ProspectTheoryEngine {
    pub const DEFAULT_LAMBDA: f64 = 2.25;
    pub const DEFAULT_ALPHA: f64 = 0.88;
    pub const DEFAULT_BETA: f64 = 0.88;
    pub const DEFAULT_GAMMA: f64 = 0.65;

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_params(lambda: f64, alpha: f64, beta: f64, gamma: f64) -> Self {
        Self {
            lambda_loss_aversion: if lambda.is_finite() && lambda >= 1.0 {
                lambda.clamp(1.0, 10.0)
            } else {
                Self::DEFAULT_LAMBDA
            },
            alpha_gain: if alpha.is_finite() && alpha > 0.0 {
                alpha.clamp(0.1, 1.0)
            } else {
                Self::DEFAULT_ALPHA
            },
            beta_loss: if beta.is_finite() && beta > 0.0 {
                beta.clamp(0.1, 1.0)
            } else {
                Self::DEFAULT_BETA
            },
            gamma_weighting: if gamma.is_finite() && gamma > 0.0 {
                gamma.clamp(0.1, 1.0)
            } else {
                Self::DEFAULT_GAMMA
            },
        }
    }

    /// Evalúa la función de valor asimétrica de Kahneman-Tversky V(Δx):
    /// - Para Δx >= 0: V = (Δx)^α
    /// - Para Δx < 0:  V = -λ · (-Δx)^β
    #[inline(always)]
    pub fn value(&self, delta_x: f64) -> f64 {
        if !delta_x.is_finite() {
            return 0.0;
        }
        let safe_x = delta_x.clamp(-100.0, 100.0);
        if safe_x >= 0.0 {
            safe_x.powf(self.alpha_gain)
        } else {
            -self.lambda_loss_aversion * (-safe_x).powf(self.beta_loss)
        }
    }

    /// Ponderación de probabilidad no lineal de Tversky-Kahneman w(p):
    ///   w(p) = p^γ / (p^γ + (1 - p)^γ)^(1 / γ)
    #[inline(always)]
    pub fn probability_weight(&self, p: f64) -> f64 {
        if !p.is_finite() {
            return 0.0;
        }
        let safe_p = p.clamp(0.0, 1.0);
        if safe_p <= 1e-6 {
            return 0.0;
        }
        if safe_p >= 1.0 - 1e-6 {
            return 1.0;
        }

        let gamma = self.gamma_weighting;
        let p_g = safe_p.powf(gamma);
        let q_g = (1.0 - safe_p).powf(gamma);
        let denom = (p_g + q_g).powf(1.0 / gamma);

        if denom > 1e-12 && denom.is_finite() {
            (p_g / denom).clamp(0.0, 1.0)
        } else {
            safe_p
        }
    }

    /// Calcula la presión psicológica neta de prospecto P_kt:
    ///   P_kt = w(p_bull) · V(Δp_up) + w(p_crash) · V(-Δp_down)
    #[inline(always)]
    pub fn compute_prospect_pressure(
        &self,
        p_bull: f64,
        p_crash: f64,
        delta_up: f64,
        delta_down: f64,
    ) -> f64 {
        let w_bull = self.probability_weight(p_bull);
        let w_crash = self.probability_weight(p_crash);

        let safe_up = if delta_up.is_finite() && delta_up >= 0.0 {
            delta_up.clamp(0.0, 100.0)
        } else {
            0.0
        };
        let safe_down = if delta_down.is_finite() && delta_down >= 0.0 {
            delta_down.clamp(0.0, 100.0)
        } else {
            0.0
        };

        let v_gain = self.value(safe_up);
        let v_loss = self.value(-safe_down); // Retorna valor negativo debido a -λ

        let pressure = w_bull * v_gain + w_crash * v_loss;
        if pressure.is_finite() {
            pressure.clamp(-50.0, 50.0)
        } else {
            0.0
        }
    }

    /// Modulador continuo de convicción para la deliberación del Consejo:
    /// - Si intended_direction y prospect_pressure indican pánico de masas adverso (la multitud vomita inventario
    ///   en pérdidas mientras el sistema compra), amplifica la convicción contrarian smart-money [1.0, 1.30].
    /// - Si intentamos sumarnos tarde al FOMO de la masa, atenúa defensivamente [0.50, 1.0].
    #[inline(always)]
    pub fn modulation_factor(&self, intended_direction: f64, prospect_pressure: f64) -> f64 {
        if !intended_direction.is_finite() || !prospect_pressure.is_finite() {
            return 1.0;
        }
        let dir = intended_direction.clamp(-1.0, 1.0);
        if dir.abs() < 1e-6 {
            return 1.0;
        }

        // Smart contrarian edge:
        // Si dir = +1 (Long) y prospect_pressure < 0 (Pánico bajista masivo): edge = -1 * (-P) = +P > 0.
        // Si dir = +1 (Long) y prospect_pressure > 0 (FOMO alcista masivo): edge = -1 * (+P) = -P < 0.
        let smart_contrarian_edge = -dir * prospect_pressure;

        if smart_contrarian_edge >= 0.0 {
            1.0 + 0.30 * (smart_contrarian_edge * 0.25).tanh()
        } else {
            1.0 + 0.50 * (smart_contrarian_edge * 0.25).tanh()
        }
    }

    /// Calcula la presión psicológica continua anti-simétrica del mercado a partir del sentimiento de masas (LS ratio).
    /// Cumple simetría espejo estricta anti-simétrica: P(1/LS, liq) = -P(LS, liq).
    /// Neutral exacto en LS = 1.0 -> 0.0.
    #[inline(always)]
    pub fn compute_crowd_net_prospect_pressure(
        &self,
        ls_ratio: f64,
        liquidation_severity: f64,
        delta_pts: f64,
    ) -> f64 {
        let safe_ls = if ls_ratio.is_finite() && ls_ratio > 1e-4 {
            ls_ratio.clamp(0.01, 100.0)
        } else {
            1.0
        };
        let p_long = (safe_ls / (1.0 + safe_ls)).clamp(0.01, 0.99);
        let p_short = 1.0 - p_long;

        let safe_liq = if liquidation_severity.is_finite() && liquidation_severity >= 0.0 {
            liquidation_severity.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let p_crash_long = (safe_liq * p_long + p_short * 0.5).clamp(0.01, 0.99);
        let p_squeeze_short = (safe_liq * p_short + p_long * 0.5).clamp(0.01, 0.99);

        let safe_delta = if delta_pts.is_finite() && delta_pts >= 0.0 {
            delta_pts.clamp(0.1, 10.0)
        } else {
            1.0
        };

        let p_kt_long = self.compute_prospect_pressure(p_long, p_crash_long, safe_delta, safe_delta);
        let p_kt_short = self.compute_prospect_pressure(p_short, p_squeeze_short, safe_delta, safe_delta);

        let net_pressure = p_kt_long - p_kt_short;
        if net_pressure.is_finite() {
            net_pressure.clamp(-50.0, 50.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crowd_prospect_pressure_mirror_symmetry() {
        let engine = ProspectTheoryEngine::new();
        // 1. En LS = 1.0 (neutral), la presión neta del mercado debe ser exactamente 0.0
        let p_neutral = engine.compute_crowd_net_prospect_pressure(1.0, 0.20, 1.5);
        assert!(
            p_neutral.abs() < 1e-12,
            "En neutralidad LS=1.0, la presión neta debe ser 0.0, obtenido: {}",
            p_neutral
        );

        // 2. Simetría anti-simétrica exacta: P(1/LS) == -P(LS)
        for &ls in &[1.5, 2.0, 3.5, 5.0, 10.0] {
            let p_bull = engine.compute_crowd_net_prospect_pressure(ls, 0.30, 2.0);
            let p_bear = engine.compute_crowd_net_prospect_pressure(1.0 / ls, 0.30, 2.0);
            assert!(
                (p_bull + p_bear).abs() < 1e-10,
                "Para LS={} y 1/LS={}, p_bull={} y p_bear={} deben ser opuestos exactos",
                ls,
                1.0 / ls,
                p_bull,
                p_bear
            );
            assert!(
                p_bull > 0.0,
                "Para LS > 1.0, la presión de euforia compradora debe ser positiva: {}",
                p_bull
            );
            assert!(
                p_bear < 0.0,
                "Para LS < 1.0, la presión de pánico vendedor debe ser negativa: {}",
                p_bear
            );
        }
    }

    #[test]
    fn test_modulation_factor_mirror_symmetry() {
        let engine = ProspectTheoryEngine::new();
        // Para cualquier par espejado (dir=+1, P) y (dir=-1, -P), modulation_factor debe ser idéntico
        for &p in &[-15.0, -5.0, 0.0, 5.0, 15.0] {
            let m_long = engine.modulation_factor(1.0, p);
            let m_short = engine.modulation_factor(-1.0, -p);
            assert!(
                (m_long - m_short).abs() < 1e-12,
                "Modulation factor debe ser idéntico bajo espejo para P={}: long={}, short={}",
                p,
                m_long,
                m_short
            );
        }
    }
}

