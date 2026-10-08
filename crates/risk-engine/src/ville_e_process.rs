//! SUPERMARTINGALAS DE VILLE Y E-VALORES ANYTIME-VALID (Ω30)
//!
//! # Fundamento Matemático
//!
//! En la inferencia estadística clásica, el valor p ($p$-value) asume un tamaño muestral $N$
//! predeterminado y fijo. El monitoreo continuo de métricas en tiempo real con parada reactiva
//! (*optional stopping*) infla el error Tipo I hasta tasas inaceptables (> 40%).
//!
//! Un **E-proceso** $(M_t)_{t \ge 0}$ es una supermartingala de prueba no negativa bajo la hipótesis
//! nula $H_0: \mathbb{E}[X] \le 0$ (ausencia de edge o ventaja matemática):
//!
//! 1. $M_0 = 1.0$ casi seguramente.
//! 2. $\forall t \ge s \ge 0: \quad \mathbb{E}_{H_0}[M_t \mid \mathcal{F}_s] \le M_s$.
//! 3. $M_t \ge 0$.
//!
//! Por la **Desigualdad Maximal de Ville (1939)**:
//!
//! $$\mathbb{P}_{H_0}\left( \sup_{t \ge 0} M_t \ge \frac{1}{\alpha} \right) \le \alpha$$
//!
//! Esto garantiza que el sistema puede inspeccionar el estado de la estrategia en cualquier tick,
//! en cualquier instante temporal o en cualquier tiempo de parada $\tau$, certificando edge genuino
//! cuando $M_t \ge 1/\alpha$ con cota estricta $(1 - \alpha)$ de confianza sin sesgo de parada opcional.

use std::f64;

/// Parámetro de aprendizaje conservador para adaptación de lambda (Online Newton Step).
const ONS_ETA: f64 = 0.50;
/// Suelo de varianza numérica para evitar divisiones por cero.
const EPS_VARIANCE: f64 = 1e-6;
/// Límite inferior de e-value para declarar agotamiento irreversible de ventaja.
pub const E_EXHAUSTION_FLOOR: f64 = 1e-4;

/// E-Proceso secuencial anytime-valid basado en supermartingalas de Ville.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VilleEProcess {
    /// Nivel de significancia alpha (ej. 0.05 para 95% de confianza; 0.01 para 99%).
    pub alpha: f64,
    /// Umbral de parada / certificación de edge: 1.0 / alpha (ej. 20.0 para alpha = 0.05).
    pub threshold: f64,
    /// Valor actual del e-proceso M_t (inicia en 1.0).
    pub e_value: f64,
    /// Logaritmo de la riqueza acumulada: ln(M_t).
    pub log_wealth: f64,
    /// Máximo e-value histórico alcanzado: sup_{s <= t} M_s.
    pub peak_e_value: f64,
    /// Conteo de observaciones válidas admitidas.
    pub count: usize,
    /// Estimador recursivo EWMA de la media condicional de innovaciones.
    pub running_mean: f64,
    /// Estimador recursivo EWMA del momento de segundo orden E[X^2].
    pub running_sq_mean: f64,
    /// Cota mínima permitida para la fracción de apuesta lambda (suelo de exploración).
    pub lambda_min: f64,
    /// Cota máxima permitida para la fracción de apuesta lambda.
    pub lambda_max: f64,
    /// Fracción de apuesta lambda calculada en el paso previo.
    pub last_lambda: f64,
}

impl VilleEProcess {
    /// Crea un nuevo E-proceso de Ville con significancia `alpha` (ej. 0.05) y cota por defecto `lambda_max = 0.50`.
    pub fn new(alpha: f64) -> Result<Self, &'static str> {
        Self::with_bounds(alpha, 0.0, 0.50)
    }

    /// Crea un nuevo E-proceso con cota de apuesta personalizada `lambda_max` en (0, 1) y `lambda_min = 0.0`.
    pub fn with_lambda_max(alpha: f64, lambda_max: f64) -> Result<Self, &'static str> {
        Self::with_bounds(alpha, 0.0, lambda_max)
    }

    /// Crea un nuevo E-proceso con límites explícitos de apuesta `lambda_min` y `lambda_max`.
    pub fn with_bounds(alpha: f64, lambda_min: f64, lambda_max: f64) -> Result<Self, &'static str> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err("alpha debe pertenecer al intervalo abierto (0.0, 1.0)");
        }
        if !lambda_min.is_finite() || lambda_min < 0.0 || lambda_min >= 1.0 {
            return Err("lambda_min debe pertenecer al intervalo semiatestado [0.0, 1.0)");
        }
        if !lambda_max.is_finite() || lambda_max <= lambda_min || lambda_max >= 1.0 {
            return Err("lambda_max debe ser estrictamente mayor que lambda_min y menor que 1.0");
        }
        Ok(Self {
            alpha,
            threshold: 1.0 / alpha,
            e_value: 1.0,
            log_wealth: 0.0,
            peak_e_value: 1.0,
            count: 0,
            running_mean: 0.0,
            running_sq_mean: 0.0,
            lambda_min,
            lambda_max,
            last_lambda: 0.0,
        })
    }

    /// Actualiza el E-proceso ante una nueva observación discreta de retorno o innovación de señal $X_t$.
    ///
    /// La observación se clampa a $[-1.0, 1.0]$ para asegurar que $1 + \lambda X > 0$ por construcción.
    /// Retorna el nuevo valor de $M_t$.
    pub fn update(&mut self, observation: f64) -> f64 {
        if !observation.is_finite() {
            return self.e_value;
        }

        let x = observation.clamp(-1.0, 1.0);

        // 1) Adaptación causal del parámetro de apuesta lambda basado en información PREVIA a x
        let lambda = self.compute_causal_lambda();
        self.last_lambda = lambda;

        // 2) Factor de multiplicación de la martingala: M_t = M_{t-1} * (1 + lambda * x)
        let wealth_mult = 1.0 + lambda * x;
        if wealth_mult > 0.0 {
            let log_increment = wealth_mult.ln();
            self.log_wealth += log_increment;
            self.e_value = self.log_wealth.exp();
        } else {
            // Protección contra pérdida terminal de capital
            self.e_value = 0.0;
            self.log_wealth = f64::NEG_INFINITY;
        }

        if self.e_value > self.peak_e_value {
            self.peak_e_value = self.e_value;
        }

        // 3) Actualización de estadísticos recursivos para las siguientes observaciones
        self.count += 1;
        let decay = if self.count <= 10 {
            1.0 / self.count as f64
        } else {
            ONS_ETA * (1.0 / (self.count as f64).sqrt())
        };

        self.running_mean = (1.0 - decay) * self.running_mean + decay * x;
        self.running_sq_mean = (1.0 - decay) * self.running_sq_mean + decay * (x * x);

        self.e_value
    }

    /// Actualiza el E-proceso en el caso de una difusión estocástica continua:
    ///
    /// $$dM_t = M_t \cdot \lambda_t \left( dX_t - \frac{1}{2} \lambda_t \sigma^2 dt \right)$$
    pub fn update_continuous_sde(&mut self, dx: f64, dt_seconds: f64, sigma: f64) -> f64 {
        if !dx.is_finite() || !dt_seconds.is_finite() || dt_seconds <= 0.0 || !sigma.is_finite() || sigma <= 0.0 {
            return self.e_value;
        }

        let lambda = self.compute_causal_lambda();
        self.last_lambda = lambda;

        // Integral de Ito exponencial: ln M_t = sum [ lambda * dx - 0.5 * lambda^2 * sigma^2 * dt ]
        let drift_penalty = 0.5 * lambda * lambda * sigma * sigma * dt_seconds;
        let increment = lambda * dx - drift_penalty;

        self.log_wealth += increment;
        self.e_value = self.log_wealth.exp();

        if self.e_value > self.peak_e_value {
            self.peak_e_value = self.e_value;
        }

        self.count += 1;
        let decay = 1.0 / (self.count.min(100) as f64);
        let normalized_rate = dx / dt_seconds.max(1e-4);
        self.running_mean = (1.0 - decay) * self.running_mean + decay * normalized_rate;
        self.running_sq_mean = (1.0 - decay) * self.running_sq_mean + decay * (normalized_rate * normalized_rate);

        self.e_value
    }

    /// Calcula la fracción de apuesta causal lambda adaptada mediante la ratio de Sharpe empírica acotada.
    #[inline]
    fn compute_causal_lambda(&self) -> f64 {
        if self.count < 3 || self.running_mean <= 0.0 {
            return self.lambda_min;
        }
        let variance = (self.running_sq_mean - self.running_mean * self.running_mean).max(EPS_VARIANCE);
        let empirical_kelly = self.running_mean / variance;
        empirical_kelly.clamp(self.lambda_min, self.lambda_max)
    }

    /// Indica si el proceso ha certificado presencia de ventaja estadística (*edge*)
    /// superando el umbral de Ville $M_t \ge 1 / \alpha$.
    #[inline]
    pub fn is_edge_certified(&self) -> bool {
        self.e_value >= self.threshold
    }

    /// Pérdida fraccional de evidencia estadística acumulada respecto a su máximo histórico:
    ///
    /// $$\text{DD}_t = \frac{\sup_{s \le t} M_s - M_t}{\sup_{s \le t} M_s} \in [0.0, 1.0]$$
    #[inline]
    pub fn evidence_drawdown(&self) -> f64 {
        if self.peak_e_value > 0.0 {
            ((self.peak_e_value - self.e_value) / self.peak_e_value).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Indica si la evidencia estadística ha sufrido un deterioro o degradación que supera `max_drawdown`.
    #[inline]
    pub fn is_evidence_decayed(&self, max_drawdown: f64) -> bool {
        self.evidence_drawdown() >= max_drawdown
    }

    /// Indica si el proceso ha agotado el capital de prueba ($M_t < 10^{-4}$),
    /// o si habiendo tenido evidencia previa, su riqueza cayó bajo 1.0 con deriva negativa persistente.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.e_value <= E_EXHAUSTION_FLOOR
            || (self.peak_e_value > 1.0 && self.e_value < 1.0 && self.running_mean <= 0.0)
    }

    /// Cota superior del p-valor en cualquier momento (*anytime p-value bound*):
    ///
    /// $$p_{\text{anytime}} \le \min\left(1.0, \, \frac{1}{\sup_{s \le t} M_s}\right)$$
    #[inline]
    pub fn anytime_p_value(&self) -> f64 {
        if self.peak_e_value <= 1.0 {
            1.0
        } else {
            (1.0 / self.peak_e_value).min(1.0)
        }
    }
}

impl Default for VilleEProcess {
    fn default() -> Self {
        Self::new(0.05).expect("default alpha 0.05 siempre es válido")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ville_parametros_invalidos_retornan_error() {
        assert!(VilleEProcess::new(0.0).is_err());
        assert!(VilleEProcess::new(1.0).is_err());
        assert!(VilleEProcess::new(-0.05).is_err());
        assert!(VilleEProcess::new(f64::NAN).is_err());
        assert!(VilleEProcess::with_lambda_max(0.05, 1.5).is_err());
    }

    #[test]
    fn ville_inicializacion_correcta() {
        let ep = VilleEProcess::new(0.05).unwrap();
        assert_eq!(ep.alpha, 0.05);
        assert!((ep.threshold - 20.0).abs() < 1e-12);
        assert_eq!(ep.e_value, 1.0);
        assert_eq!(ep.peak_e_value, 1.0);
        assert_eq!(ep.count, 0);
        assert!(!ep.is_edge_certified());
        assert!(!ep.is_exhausted());
        assert_eq!(ep.anytime_p_value(), 1.0);
    }

    #[test]
    fn ville_observaciones_no_finitas_no_mutan_estado() {
        let mut ep = VilleEProcess::new(0.05).unwrap();
        assert_eq!(ep.update(f64::NAN), 1.0);
        assert_eq!(ep.update(f64::INFINITY), 1.0);
        assert_eq!(ep.count, 0);
        assert_eq!(ep.e_value, 1.0);
    }

    #[test]
    fn ville_bajo_el_nulo_puro_la_martingala_no_explota() {
        // Bajo H0 pura (ruido blanco simétrico con media cero), el e-proceso no debe certificar edge
        let mut ep = VilleEProcess::new(0.05).unwrap();
        let noise = [0.02, -0.02, 0.01, -0.01, -0.03, 0.02, -0.01, 0.01, -0.02, 0.02];
        for &x in noise.iter().cycle().take(100) {
            ep.update(x);
        }
        // Con media cero, lambda se mantiene en 0 o muy bajo, y el E-valor no supera el umbral de 20
        assert!(!ep.is_edge_certified(), "ruido nulo no puede certificar edge");
        assert!(ep.e_value <= ep.threshold);
    }

    #[test]
    fn ville_senial_persistente_con_edge_certifica_superacion_de_umbral() {
        // Bajo H1 (señal genuina con retorno positivo consistente)
        let mut ep = VilleEProcess::new(0.05).unwrap(); // umbral = 20.0
        let positive_returns = [0.15, 0.12, 0.18, 0.14, 0.16, 0.20, 0.13, 0.17];
        for &x in positive_returns.iter().cycle().take(50) {
            ep.update(x);
            if ep.is_edge_certified() {
                break;
            }
        }
        assert!(ep.is_edge_certified(), "señal positiva debe certificar edge legítimo");
        assert!(ep.e_value >= 20.0);
        assert!(ep.anytime_p_value() <= 0.05);
    }

    #[test]
    fn ville_difusion_continua_sde_integra_fielmente() {
        let mut ep = VilleEProcess::new(0.05).unwrap();
        // Simular deriva positiva continua con volatilidad
        for _ in 0..50 {
            ep.update_continuous_sde(0.005, 0.1, 0.01);
        }
        assert!(ep.e_value > 1.0);
        assert!(ep.count == 50);
    }
}
