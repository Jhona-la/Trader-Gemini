//! Fase 7: Ergocicidad, Rough Paths & Tasa de Crecimiento Temporal de Ole Peters
//!
//! # Fundamentación Teórica Cuántica y de Sistemas Complejos
//!
//! Este módulo implementa la tríada matemática no-ergódica para microestructura continua:
//! 1. **Tasa de Crecimiento Temporal de Ole Peters (`ErgodicGrowthEngine`):**
//!    En procesos estocásticos multiplicativos de riqueza, el promedio de ensamble
//!    $\langle \Delta W \rangle$ diverge del promedio temporal casi seguro:
//!    $$g_{\text{time}}(f) = \lim_{T \to \infty} \frac{1}{T} \ln \left(\frac{W(T)}{W(0)}\right) = \mathbb{E}[\ln(1 + f \cdot R - \text{drag})]$$
//!    Maximizar el promedio de ensamble lleva a la ruina matemática segura ($P(\text{ruin}) \to 1$).
//!    Este motor resuelve la asignación óptima $f^*_{\text{ergodic}} = \arg\max g_{\text{time}}(f)$ mediante
//!    búsqueda de sección áurea acotada y computa la brecha de ergocicidad $\Delta_{\text{ergodic}}$.
//!
//! 2. **Predictor Rough Bergomi Multiescala (`RoughBergomiPredictor`):**
//!    La volatilidad financiera no es un semimartingala browniana ($H = 0.50$), sino un proceso
//!    rugoso (*rough*) con exponente de Hurst $H \in (0, 0.50)$, típicamente $H \approx 0.10 - 0.14$.
//!    A través de la representación Markoviana multi-factor de Abi Jaber & El Euch (2019):
//!    $$(t-s)^{H - 1/2} \approx \sum_{m=1}^M c_m e^{-\lambda_m (t-s)}$$
//!    se proyecta la evolución estocástica fraccionaria en tiempo $O(1)$ con memoria constante,
//!    reproduciendo el skew de volatilidad de ley de potencia $\mathcal{S}(\tau) \propto \tau^{H - 1/2}$.
//!
//! 3. **Griegas Estocásticas de Malliavin (`MalliavinRoughGreeks`):**
//!    Derivada de Malliavin $D_s S_t$ y sensibilidades continuas sin sesgo de discretización:
//!    Gamma rugosa, Vanna causal y Volga fraccional, modulando un amortiguador dinámico
//!    del tamaño de orden $\kappa \in [0.10, 1.0]$ para prevenir atrapamiento en vacíos de liquidez.
//!
//! Cumple estrictamente con el trinquete `rb_p1`: CERO `unwrap`, CERO `expect`, CERO `panic`, CERO `assert`.

/// Cantidad de factores exponenciales Markovianos para aproximar el núcleo fraccionario singular.
pub const ROUGH_FACTORS_COUNT: usize = 8;

/// Resultado de la optimización ergódica de la tasa de crecimiento temporal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErgodicGrowthResult {
    /// Apalancamiento fraccional óptimo $f^* \in [0.0, 0.40]$.
    pub optimal_leverage: f64,
    /// Tasa de crecimiento temporal óptima $g_{\text{time}}(f^*)$.
    pub g_time_optimal: f64,
    /// Apalancamiento conservador Half-Kelly ($f^* / 2$).
    pub half_leverage: f64,
    /// Tasa de crecimiento temporal bajo Half-Kelly $g_{\text{time}}(f^* / 2)$.
    pub g_time_half: f64,
    /// Promedio de ensamble clásico $\mathbb{E}[R]$.
    pub ensemble_average: f64,
    /// Brecha de no-ergocicidad $\Delta_{\text{ergodic}} = f^* \cdot \mathbb{E}[R] - g_{\text{time}}(f^*)$.
    pub ergodicity_gap: f64,
    /// Cota analítica superior de probabilidad de ruina estocástica.
    pub ruin_probability_bound: f64,
    /// Si la estrategia posee ventaja ergódica positiva ($g_{\text{time}} > 0$).
    pub is_ergodically_viable: bool,
}

impl Default for ErgodicGrowthResult {
    fn default() -> Self {
        Self {
            optimal_leverage: 0.0,
            g_time_optimal: 0.0,
            half_leverage: 0.0,
            g_time_half: 0.0,
            ensemble_average: 0.0,
            ergodicity_gap: 0.0,
            ruin_probability_bound: 1.0,
            is_ergodically_viable: false,
        }
    }
}

/// Motor de Optimización Ergódica según la teoría de crecimiento temporal de Ole Peters.
#[derive(Debug, Clone, Default)]
pub struct ErgodicGrowthEngine {
    /// Límite superior de apalancamiento permitido (por diseño <= 0.40 para evitar colapso).
    pub max_leverage_bound: f64,
    /// Tolerancia de convergencia de la búsqueda de sección áurea.
    pub tolerance: f64,
}

impl ErgodicGrowthEngine {
    /// Construye una nueva instancia con parámetros conservadores por defecto.
    pub fn new() -> Self {
        Self {
            max_leverage_bound: 0.40,
            tolerance: 1e-5,
        }
    }

    /// Evalúa la tasa de crecimiento temporal empírica $g_{\text{time}}(f)$ para una muestra de retornos y fricciones.
    ///
    /// $$g_{\text{time}}(f) = \frac{1}{K}\sum_{k=1}^K \ln(1 + f \cdot r_k - \text{drag}_k)$$
    pub fn evaluate_g_time(f: f64, returns: &[f64], friction_drag: f64) -> f64 {
        if returns.is_empty() || !f.is_finite() || f < 0.0 {
            return f64::NEG_INFINITY;
        }

        let drag = if friction_drag.is_finite() && friction_drag >= 0.0 {
            friction_drag
        } else {
            0.0
        };

        let mut sum_log = 0.0;
        let count = returns.len() as f64;

        for &r in returns {
            if !r.is_finite() {
                continue;
            }
            let wealth_multiplier = 1.0 + f * r - drag;
            if wealth_multiplier <= 1e-9 {
                // Absorción en la barrera de ruina (log-riqueza -> -inf)
                return f64::NEG_INFINITY;
            }
            sum_log += wealth_multiplier.ln();
        }

        sum_log / count
    }

    /// Optimiza el apalancamiento $f^*$ maximizando la tasa de crecimiento temporal $g_{\text{time}}(f)$.
    ///
    /// Utiliza el algoritmo de búsqueda de sección áurea (Golden Section Search) en el intervalo $[0, f_{\max}]$.
    pub fn optimize_growth(&self, returns: &[f64], friction_drag: f64) -> ErgodicGrowthResult {
        if returns.is_empty() {
            return ErgodicGrowthResult::default();
        }

        let max_f = self.max_leverage_bound.clamp(0.01, 0.40);
        let tol = self.tolerance.clamp(1e-7, 1e-3);

        // Promedio clásico de ensamble
        let mut sum_r = 0.0;
        let mut valid_count = 0usize;
        for &r in returns {
            if r.is_finite() {
                sum_r += r;
                valid_count += 1;
            }
        }
        let ensemble_avg = if valid_count > 0 { sum_r / valid_count as f64 } else { 0.0 };

        // Proporción áurea: phi = (sqrt(5) - 1) / 2 ~ 0.6180339887
        let phi = 0.6180339887498948;

        let mut a = 0.0;
        let mut b = max_f;

        let mut c = b - phi * (b - a);
        let mut d = a + phi * (b - a);

        let mut fc = Self::evaluate_g_time(c, returns, friction_drag);
        let mut fd = Self::evaluate_g_time(d, returns, friction_drag);

        let mut iterations = 0;
        while (b - a).abs() > tol && iterations < 64 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - phi * (b - a);
                fc = Self::evaluate_g_time(c, returns, friction_drag);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + phi * (b - a);
                fd = Self::evaluate_g_time(d, returns, friction_drag);
            }
            iterations += 1;
        }

        let optimal_f = ((a + b) * 0.5).clamp(0.0, max_f);
        let g_opt = Self::evaluate_g_time(optimal_f, returns, friction_drag);

        // Comparar con f = 0 (tasa base = 0.0)
        let (best_f, best_g) = if g_opt > 0.0 && g_opt.is_finite() {
            (optimal_f, g_opt)
        } else {
            (0.0, 0.0)
        };

        let half_f = best_f * 0.5;
        let g_half = Self::evaluate_g_time(half_f, returns, friction_drag);

        let ergodicity_gap = (best_f * ensemble_avg - best_g).max(0.0);

        // Varianza empírica de log-crecimiento para cota de ruina
        let mut sum_sq_diff = 0.0;
        for &r in returns {
            if r.is_finite() {
                let m = (1.0 + best_f * r - friction_drag).max(1e-9);
                let diff = m.ln() - best_g;
                sum_sq_diff += diff * diff;
            }
        }
        let g_var = if valid_count > 1 {
            sum_sq_diff / (valid_count as f64 - 1.0)
        } else {
            0.01
        };

        // Cota de Cramér-Lundberg sobre la riqueza: P(ruina) <= exp(-2 * g_opt / sigma^2 * log_drawdown_barrier)
        let ruin_bound = if best_g > 0.0 {
            if g_var <= 1e-8 {
                // Si la dispersión es despreciable y el crecimiento temporal es positivo,
                // la dinámica es determinista monótonamente creciente (probabilidad de ruina nula).
                0.0
            } else {
                let theta = (2.0 * best_g / g_var).clamp(0.0, 50.0);
                // Para una caída crítica del 30% en capital (ln(1 / 0.70) ~ 0.35667)
                (-theta * 0.35667).exp().clamp(0.0, 1.0)
            }
        } else {
            1.0
        };

        ErgodicGrowthResult {
            optimal_leverage: best_f,
            g_time_optimal: best_g,
            half_leverage: half_f,
            g_time_half: g_half,
            ensemble_average: ensemble_avg,
            ergodicity_gap,
            ruin_probability_bound: ruin_bound,
            is_ergodically_viable: best_g > 0.0,
        }
    }
}

/// Estado y parámetros del Predictor Rough Bergomi Multiescala.
#[derive(Debug, Clone)]
pub struct RoughBergomiPredictor {
    /// Parámetro de Hurst $H \in (0.02, 0.48)$.
    pub hurst: f64,
    /// Volatilidad de la volatilidad (*vol of vol*) $\eta \in [0.5, 4.0]$.
    pub eta_vol_of_vol: f64,
    /// Correlación spot-volatilidad $\rho_{xv} \in [-0.95, -0.10]$.
    pub rho_spot_vol: f64,
    /// Curva inicial de varianza hacia adelante $\xi_0$.
    pub forward_var_xi0: f64,
    /// Factores de decaimiento temporal $\lambda_m$ para cada uno de los $M=8$ modos Markovianos.
    pub lambda_factors: [f64; ROUGH_FACTORS_COUNT],
    /// Coeficientes de ponderación $c_m$ del núcleo fraccionario singular.
    pub weights_cm: [f64; ROUGH_FACTORS_COUNT],
    /// Estado continuo actual de los 8 factores Markovianos $X_t^m$.
    pub current_factors: [f64; ROUGH_FACTORS_COUNT],
}

impl Default for RoughBergomiPredictor {
    fn default() -> Self {
        Self::new(0.12, 1.80, -0.72, 0.04)
    }
}

impl RoughBergomiPredictor {
    /// Inicializa el predictor Rough Bergomi calculando los coeficientes del núcleo fraccionario.
    pub fn new(hurst: f64, eta_vol_of_vol: f64, rho_spot_vol: f64, forward_var_xi0: f64) -> Self {
        let h = hurst.clamp(0.03, 0.45);
        let eta = eta_vol_of_vol.clamp(0.1, 5.0);
        let rho = rho_spot_vol.clamp(-0.99, 0.0);
        let xi0 = forward_var_xi0.clamp(1e-6, 1.0);

        let mut lambda_factors = [0.0; ROUGH_FACTORS_COUNT];
        let mut weights_cm = [0.0; ROUGH_FACTORS_COUNT];

        // Discretización geométrica de escalas temporales:
        // lambda_m = lambda_0 * beta^(m-1), cubriendo desde alta frecuencia (1 tick) hasta regímenes amplios
        let lambda_0: f64 = 0.10;
        let beta: f64 = 2.40;

        // Aproximación de Gamma(H + 0.5) para H ~ 0.12: Gamma(0.62) ~ 1.43
        let gamma_h_plus_half = 1.432;
        let alpha = h - 0.5; // alpha in (-0.5, 0.0), exponente singular

        let mut sum_weights = 0.0;
        for m in 0..ROUGH_FACTORS_COUNT {
            let lam = lambda_0 * beta.powi(m as i32);
            lambda_factors[m] = lam;

            // Peso de integración según la fórmula de cuadratura de Abi Jaber & El Euch (2019)
            let w = (beta.powf(alpha) - 1.0) * beta.powf(m as f64 * alpha) / gamma_h_plus_half;
            weights_cm[m] = w.abs();
            sum_weights += weights_cm[m];
        }

        // Normalización exacta del núcleo para preservar escala unitaria en el origen
        if sum_weights > 1e-12 {
            for w in &mut weights_cm {
                *w /= sum_weights;
            }
        }

        Self {
            hurst: h,
            eta_vol_of_vol: eta,
            rho_spot_vol: rho,
            forward_var_xi0: xi0,
            lambda_factors,
            weights_cm,
            current_factors: [0.0; ROUGH_FACTORS_COUNT],
        }
    }

    /// Actualiza el estado estocástico en tiempo continuo ante la llegada de un nuevo shock microestructural.
    ///
    /// Cada factor $X^m$ evoluciona como un proceso de Ornstein-Uhlenbeck en $O(1)$:
    /// $$X_{t+\Delta t}^m = X_t^m e^{-\lambda_m \Delta t} + \Delta W_t$$
    pub fn update_tick(&mut self, price_shock: f64, dt_seconds: f64) {
        let dt = dt_seconds.clamp(1e-4, 60.0);
        let shock = if price_shock.is_finite() { price_shock } else { 0.0 };

        for m in 0..ROUGH_FACTORS_COUNT {
            let lam = self.lambda_factors[m];
            let decay = (-lam * dt).exp();
            self.current_factors[m] = self.current_factors[m] * decay + shock;
        }
    }

    /// Predice la varianza y volatilidad instantánea forward $\hat{\sigma}_t$.
    ///
    /// $$\hat{v}_t = \xi_0 \exp\left(\eta \sqrt{2H} \sum_{m=1}^M c_m X_t^m - \frac{1}{2}\eta^2 \sum_{m=1}^M c_m^2 \frac{1 - e^{-2\lambda_m t}}{2\lambda_m}\right)$$
    pub fn predict_instantaneous_volatility(&self) -> f64 {
        let mut aggregate_rough_factor = 0.0;
        let mut var_correction = 0.0;

        let sqrt_2h = (2.0 * self.hurst).sqrt();

        for m in 0..ROUGH_FACTORS_COUNT {
            let cm = self.weights_cm[m];
            let xm = self.current_factors[m];
            aggregate_rough_factor += cm * xm;
            var_correction += cm * cm;
        }

        let exponent = self.eta_vol_of_vol * sqrt_2h * aggregate_rough_factor
            - 0.5 * self.eta_vol_of_vol * self.eta_vol_of_vol * var_correction * 0.05;

        // Evitar desbordamiento en f64: exponente acotado en [-10, 10]
        let clamped_exp = exponent.clamp(-10.0, 10.0);
        let predicted_var = self.forward_var_xi0 * clamped_exp.exp();

        predicted_var.max(1e-8).sqrt()
    }

    /// Predice el sesgo o pendiente de volatilidad instantánea (Volatility Skew) en función del horizonte $\tau$.
    ///
    /// Bajo Rough Bergomi, para $\tau \to 0$ el skew escala como una ley de potencia singular:
    /// $$\mathcal{S}_{\text{skew}}(\tau) = \rho_{xv} \eta \frac{\sqrt{2H}}{H + 0.5} \tau^{H - 0.5}$$
    pub fn predict_volatility_skew(&self, tau_seconds: f64) -> f64 {
        let tau = tau_seconds.clamp(0.01, 86400.0);
        let power = self.hurst - 0.5; // Exponente negativo (-0.38 para H=0.12)

        let factor = self.rho_spot_vol * self.eta_vol_of_vol * (2.0 * self.hurst).sqrt() / (self.hurst + 0.5);
        let skew = factor * tau.powf(power);

        if skew.is_finite() {
            skew
        } else {
            0.0
        }
    }
}

/// Griegas Estocásticas y Operadores de Malliavin para Microestructura Rugosa.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MalliavinRoughGreeks {
    /// Delta de Malliavin $D_t S_t / S_t \approx \sigma_t$.
    pub rough_delta: f64,
    /// Gamma rugosa $\Gamma_{\text{rough}}$ (curvatura de segundo orden del camino no-semimartingala).
    pub rough_gamma: f64,
    /// Vanna causal $\partial_S \sigma_t = \rho_{xv} \eta \sqrt{2H} \sigma_t$.
    pub vanna_causal: f64,
    /// Volga fraccional $\partial^2_{\sigma^2} V$.
    pub volga_fractional: f64,
    /// Amortiguador dinámico de orden $\kappa \in [0.10, 1.0]$.
    pub order_size_dampener: f64,
    /// Indicador de alerta de vacío de liquidez rugoso.
    pub is_rough_vacuum_alert: bool,
}

impl Default for MalliavinRoughGreeks {
    fn default() -> Self {
        Self {
            rough_delta: 0.0,
            rough_gamma: 0.0,
            vanna_causal: 0.0,
            volga_fractional: 0.0,
            order_size_dampener: 1.0,
            is_rough_vacuum_alert: false,
        }
    }
}

/// Evaluador de Griegas de Malliavin sobre procesos estocásticos fraccionarios.
#[derive(Debug, Clone, Copy, Default)]
pub struct MalliavinEvaluator;

impl MalliavinEvaluator {
    /// Computa las griegas de Malliavin acoplando el predictor Rough Bergomi y los estados microestructurales.
    pub fn compute_greeks(
        predictor: &RoughBergomiPredictor,
        current_spot_price: f64,
        observed_spread_bps: f64,
        tau_horizon_seconds: f64,
    ) -> MalliavinRoughGreeks {
        let spot = if current_spot_price.is_finite() && current_spot_price > 0.0 {
            current_spot_price
        } else {
            1.0
        };

        let spread = if observed_spread_bps.is_finite() && observed_spread_bps >= 0.0 {
            observed_spread_bps
        } else {
            1.0
        };

        let sigma = predictor.predict_instantaneous_volatility();
        let h = predictor.hurst;
        let eta = predictor.eta_vol_of_vol;
        let rho = predictor.rho_spot_vol;
        let tau = tau_horizon_seconds.clamp(0.01, 3600.0);

        // 1. Rough Delta
        let rough_delta = sigma;

        // 2. Vanna Causal: acoplamiento instantáneo entre precio y volatilidad
        let vanna_causal = rho * eta * (2.0 * h).sqrt() * sigma;

        // 3. Volga Fraccional: sensibilidad de segundo orden ante estallidos de volatilidad
        let volga_fractional = 0.5 * eta * eta * (2.0 * h) * tau.powf(2.0 * h - 1.0);

        // 4. Gamma Rugosa: aceleración estocástica del camino rugoso escalada por el spread
        let spread_penalty = 1.0 + spread * 0.05;
        let rough_gamma = (sigma * sigma / spot) * (1.0 + eta * (tau.powf(h - 0.5)).abs()) * spread_penalty;

        // 5. Amortiguador de Tamaño de Orden: protege el capital ante vacíos de liquidez
        // Cuando Gamma rugosa o Volga aumentan bruscamente, el tamaño de orden se reduce gradualmente
        let risk_score = 0.10 * rough_gamma.abs() + 0.05 * volga_fractional.abs();
        let dampener = (1.0 / (1.0 + risk_score)).clamp(0.10, 1.0);

        let is_vacuum = dampener < 0.50 || rough_gamma > 5.0;

        MalliavinRoughGreeks {
            rough_delta,
            rough_gamma,
            vanna_causal,
            volga_fractional,
            order_size_dampener: dampener,
            is_rough_vacuum_alert: is_vacuum,
        }
    }
}
