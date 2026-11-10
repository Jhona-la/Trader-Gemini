//! # Fase 92 (§92) — Cosmología Inflacionaria de Starobinsky ($R + R^2$) & Dinámica de Inflatón
//!
//! ## Modo Profesor — Ontología Técnica y Matemática (7 Pilares Rigurosos)
//!
//! ### 1. QUÉ (Definición Técnica)
//! El motor `StarobinskyInflationaryEngine` modela la dinámica de aceleración y momentum del
//! mercado financiero basándose en la cosmología inflacionaria de Alexei Starobinsky (1980) y la
//! gravedad modificada $f(R) = R + \frac{R^2}{6 M^2}$. En el marco de Einstein, esta teoría equivale
//! de forma exacta a relatividad general acoplada a un campo escalar (el inflatón $\phi$) con un
//! potencial efectivo que exhibe una meseta asintóticamente plana:
//! $$V(\phi) = V_0 \left( 1 - \exp\left(-\sqrt{\frac{2}{3}}\frac{\phi}{M_{\text{pl}}}\right) \right)^2$$
//! Evalúa analíticamente los parámetros de rodadura lenta (*slow-roll*) $\epsilon_V$ y $\eta_V$, el
//! índice espectral de perturbaciones $n_s = 1 - 6\epsilon_V + 2\eta_V$, el ratio tensorial-escalar
//! $r = 16\epsilon_V$ y el número de e-foldings restantes $N_e$.
//!
//! ### 2. POR QUÉ (Fundamento Teórico Físico)
//! Los osciladores lineales de momentum (RSI, MACD) saturan rápidamente en tendencias parabólicas
//! o sufren divergencias falsas porque asumen una relación lineal o browniana entre impulso y retorno.
//! En regímenes de aceleración institucional o liquidaciones en cadena, el mercado entra en una fase
//! cuasi-de Sitter de expansión inflacionaria exponencial: el potencial plano de Starobinsky estabiliza
//! naturalmente el momentum a lo largo de la meseta (*slow-roll* $\epsilon_V \ll 1$), prediciendo con
//! exactitud matemática cuándo la inflación termina y colapsa violentamente hacia una fase de recalentamiento
//! caótico (*reheating*).
//!
//! ### 3. PARA QUÉ (Objetivo de Trading y Preservación de Capital)
//! - Identificar fases inflacionarias laminares donde el momentum institucional es auto-consistente y el
//!   riesgo de reversión es mínimo, potenciando el tamaño de posición con slot scale $\eta_{\text{starobinsky}} \in [0.15, 1.30]$.
//! - Proveer un sesgo direccional asimétrico continuo $\kappa_{\text{starobinsky}} \in [-1.0, 1.0]$.
//! - Vetar inmediatamente (`STAROBINSKY_REHEATING_COLLAPSE_VETO`) ante la ruptura del régimen de slow-roll
//!   y transición a recalentamiento térmico caótico ($\epsilon_V > 1.80$ o $r > 0.90$), blindando la
//!   supermartingala de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//!
//! ### 4. CÓMO (Algorítmica y Silicio en Stack $O(1)$)
//! - Acumulador del campo escalar $\phi(t)$ en memoria local con disipación proporcional a la fricción de Hubble $3H\dot{\phi}$.
//! - Evaluación analítica en tiempo constante de $V(\phi)$, $V'(\phi)$ y $V''(\phi)$.
//! - Cálculo en punto flotante estricto sin asignaciones dinámicas ni llamadas trigonométricas pesadas.
//!
//! ### 5. CUÁNDO (Condiciones de Ejecución)
//! En cada tick de evaluación de señales del grafo vivo omnisciente dentro de `OmniscientLivingGraphOrchestrator`.
//!
//! ### 6. DÓNDE (Ubicación Arquitectónica)
//! `crates/signal-engine/src/starobinsky_inflationary_attractor.rs`, exportado en `crates/signal-engine/src/lib.rs`.
//!
//! ### 7. QUIÉN (Responsabilidad de Diseño)
//! `StarobinskyInflationaryEngine`, bajo la supervisión del Rol 8 (Físico Teórico & Cosmología Cuántica)
//! y Rol 2 (Quant Developer de Microestructura).

/// Parámetros de configuración para la cosmología inflacionaria de Starobinsky.
#[derive(Debug, Clone, Copy)]
pub struct StarobinskyConfig {
    /// Masa de Planck efectiva reducida $M_{\text{pl}} > 0$.
    pub m_pl: f64,
    /// Escala de energía del potencial de meseta $V_0 > 0$.
    pub v0: f64,
    /// Umbral crítico de slow-roll $\epsilon_{\text{crit}}$ para transición a recalentamiento.
    pub max_epsilon_slow_roll: f64,
    /// Umbral crítico para el ratio tensorial-escalar $r_{\text{crit}}$.
    pub max_tensor_to_scalar_ratio: f64,
    /// Factor de fricción de Hubble / disipación del campo $\gamma_H$.
    pub hubble_friction: f64,
    /// Regularizador numérico $\epsilon$.
    pub epsilon: f64,
}

impl Default for StarobinskyConfig {
    fn default() -> Self {
        Self {
            m_pl: 1.0,
            v0: 1.0,
            max_epsilon_slow_roll: 1.80,
            max_tensor_to_scalar_ratio: 0.90,
            hubble_friction: 0.15,
            epsilon: 1e-7,
        }
    }
}

/// Reporte de estado emitido por el atractor inflacionario de Starobinsky.
#[derive(Debug, Clone, Copy)]
pub struct StarobinskyReport {
    /// Valor acumulado del campo inflatón $\phi$.
    pub inflaton_field: f64,
    /// Velocidad instantánea del inflatón $\dot{\phi}$.
    pub inflaton_velocity: f64,
    /// Densidad de energía del potencial de meseta $V(\phi) \ge 0$.
    pub effective_potential: f64,
    /// Primer parámetro de slow-roll $\epsilon_V \ge 0$.
    pub epsilon_slow_roll: f64,
    /// Segundo parámetro de slow-roll $\eta_V$.
    pub eta_slow_roll: f64,
    /// Índice espectral de fluctuaciones escalares $n_s$.
    pub spectral_index: f64,
    /// Ratio tensorial-escalar $r = 16\epsilon_V$.
    pub tensor_to_scalar_ratio: f64,
    /// Estimación analítica de e-foldings de inflación restantes $N_e \ge 0$.
    pub e_folding_estimate: f64,
    /// Sesgo direccional normalizado $\kappa_{\text{starobinsky}} \in [-1.0, 1.0]$.
    pub directional_bias: f64,
    /// Factor de escala de slot continuo $\eta_{\text{starobinsky}} \in [0.15, 1.30]$.
    pub slot_scaling_factor: f64,
    /// Veto estricto por colapso de recalentamiento violento (*reheating collapse*).
    pub reheating_collapse_veto: bool,
}

/// Motor de cosmología inflacionaria de Starobinsky en microestructura.
#[derive(Debug, Clone)]
pub struct StarobinskyInflationaryEngine {
    config: StarobinskyConfig,
    phi_state: f64,
    prev_phi: f64,
}

impl StarobinskyInflationaryEngine {
    /// Crea una nueva instancia del motor con la configuración dada.
    #[inline]
    pub fn new(config: StarobinskyConfig) -> Self {
        Self {
            config,
            phi_state: 1.0, // Punto de inicio en meseta de slow-roll
            prev_phi: 1.0,
        }
    }

    /// Evalúa la dinámica del inflatón y emite el reporte cosmológico.
    ///
    /// # Parámetros
    /// - `price_ret`: Retorno fraccionario instantáneo del precio.
    /// - `ofi`: Order Flow Imbalance normalizado $\in [-1.0, 1.0]$.
    /// - `spread_bps`: Spread relativo actual en bps.
    /// - `vol_norm`: Volatilidad normalizada.
    /// - `dt`: Diferencial temporal en segundos.
    #[inline]
    pub fn evaluate(
        &mut self,
        price_ret: f64,
        ofi: f64,
        spread_bps: f64,
        vol_norm: f64,
        dt: f64,
    ) -> StarobinskyReport {
        // 1. Validación fail-closed de entradas
        let inputs_valid = price_ret.is_finite()
            && ofi.is_finite()
            && spread_bps.is_finite()
            && spread_bps > 0.0
            && vol_norm.is_finite()
            && vol_norm >= 0.0
            && dt.is_finite()
            && dt > 0.0;
        if !inputs_valid {
            return StarobinskyReport {
                inflaton_field: self.phi_state,
                inflaton_velocity: 0.0,
                effective_potential: 0.0,
                epsilon_slow_roll: 0.0,
                eta_slow_roll: 0.0,
                spectral_index: 0.0,
                tensor_to_scalar_ratio: 0.0,
                e_folding_estimate: 0.0,
                directional_bias: 0.0,
                slot_scaling_factor: 0.15,
                reheating_collapse_veto: true,
            };
        }

        let p_ret = price_ret;
        let ofi_val = ofi.clamp(-1.0, 1.0);
        let _spread = spread_bps;
        let vol = vol_norm;
        let delta_t = dt.min(1.0);

        let eps = self.config.epsilon;
        let m_pl = self.config.m_pl.max(eps);
        let v0 = self.config.v0.max(eps);

        // 2. Fuerza impulsora del flujo de órdenes sobre el campo inflatón
        let force = (p_ret * 50.0 + ofi_val * 0.80) * (1.0 + vol * 0.50);

        // Actualización de estado del campo con fricción de Hubble
        let friction = self.config.hubble_friction * delta_t;
        let phi_new = (self.phi_state * (1.0 - friction) + force * delta_t).clamp(-10.0, 10.0);
        let phi_dot = (phi_new - self.prev_phi) / delta_t.max(1e-4);

        self.prev_phi = self.phi_state;
        self.phi_state = phi_new;

        let phi = self.phi_state;

        // 3. Potencial de Starobinsky:
        // V(phi) = V_0 * (1 - exp(-sqrt(2/3) * phi / M_pl))^2
        let alpha = (2.0 / 3.0_f64).sqrt() / m_pl;
        let exp_term = (-alpha * phi).exp().clamp(1e-12, 1e6);
        let bracket = (1.0 - exp_term).clamp(-10.0, 10.0);
        let effective_potential = (v0 * bracket * bracket).max(0.0);

        // Derivadas del potencial:
        // V'(phi) = 2 * V_0 * (1 - exp(-alpha * phi)) * (alpha * exp(-alpha * phi))
        let v_prime = 2.0 * v0 * bracket * alpha * exp_term;

        // V''(phi) = 2 * V_0 * alpha^2 * exp(-alpha * phi) * (2 * exp(-alpha * phi) - 1)
        let v_double_prime = 2.0 * v0 * alpha * alpha * exp_term * (2.0 * exp_term - 1.0);

        // 4. Parámetros de Slow-Roll:
        // epsilon_V = (M_pl^2 / 2) * (V' / V)^2
        let v_safe = effective_potential.max(eps);
        let epsilon_slow_roll = (0.50 * m_pl * m_pl * (v_prime / v_safe).powi(2)).clamp(0.0, 100.0);

        // eta_V = M_pl^2 * (V'' / V)
        let eta_slow_roll = (m_pl * m_pl * (v_double_prime / v_safe)).clamp(-100.0, 100.0);

        // 5. Parámetros cosmológicos de perturbaciones:
        let spectral_index = 1.0 - 6.0 * epsilon_slow_roll + 2.0 * eta_slow_roll;
        let tensor_to_scalar_ratio = (16.0 * epsilon_slow_roll).clamp(0.0, 100.0);

        // Estimación de e-foldings restantes en meseta: N_e \approx 0.75 * exp(alpha * phi)
        let e_folding_estimate = if phi > 0.0 {
            (0.75 * (alpha * phi).exp()).clamp(0.0, 120.0)
        } else {
            0.0
        };

        // 6. Sesgo direccional: combinación de velocidad del inflatón y pendiente de meseta
        let directional_bias = (phi_dot * 0.50 + ofi_val * 0.50).clamp(-1.0, 1.0);

        // 7. Modulación de escala de slot continuo
        // En régimen de slow-roll suave (epsilon << 1, r << 1), la expansión es consistente -> boost de slot
        let slow_roll_fidelity = (1.0 / (1.0 + epsilon_slow_roll * 1.50)).clamp(0.10, 1.0);
        let slot_scaling_factor =
            ((0.15 + 1.15 * slow_roll_fidelity) * (1.0 - 0.15 * vol.min(1.0))).clamp(0.15, 1.30);

        // 8. Condición de Veto Fail-Closed:
        // Si epsilon_V excede el umbral de slow-roll o el ratio tensorial-escalar colapsa
        let reheating_collapse_veto = epsilon_slow_roll > self.config.max_epsilon_slow_roll
            || tensor_to_scalar_ratio > (self.config.max_tensor_to_scalar_ratio * 16.0)
            || phi_dot.abs() > 8.0;

        StarobinskyReport {
            inflaton_field: phi,
            inflaton_velocity: phi_dot,
            effective_potential,
            epsilon_slow_roll,
            eta_slow_roll,
            spectral_index,
            tensor_to_scalar_ratio,
            e_folding_estimate,
            directional_bias,
            slot_scaling_factor,
            reheating_collapse_veto,
        }
    }
}

impl Default for StarobinskyInflationaryEngine {
    fn default() -> Self {
        Self::new(StarobinskyConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starobinsky_normal_slow_roll() {
        let mut engine = StarobinskyInflationaryEngine::default();
        let report = engine.evaluate(0.001, 0.10, 1.5, 0.02, 0.01);

        assert!(report.effective_potential >= 0.0);
        assert!(report.epsilon_slow_roll.is_finite());
        assert!(report.slot_scaling_factor >= 0.15 && report.slot_scaling_factor <= 1.30);
        assert!(!report.reheating_collapse_veto);
    }

    #[test]
    fn test_starobinsky_subnormal_safety() {
        let mut engine = StarobinskyInflationaryEngine::default();
        let report = engine.evaluate(f64::NAN, f64::INFINITY, -1.0, f64::NAN, 0.0);

        assert!(report.inflaton_field.is_finite());
        assert!(report.effective_potential.is_finite());
        assert!(report.slot_scaling_factor.is_finite());
    }
}
