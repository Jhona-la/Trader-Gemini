//! Fase 114 (§114): Censura Cósmica de Penrose & Superficies Atrapadas de Raychaudhuri
//!
//! # Fundamentación Matemática y Física
//! La conjetura de censura cósmica de Roger Penrose (Weak Cosmic Censorship Conjecture, WCCC)
//! establece que en el colapso gravitacional físico realista no pueden formarse singularidades desnudas:
//! toda singularidad de curvatura debe estar estrictamente confinada detrás de un horizonte de sucesos
//! aparente que proteja a los observadores externos de rupturas de predictibilidad causal.
//!
//! La dinámica de enfoque geodésico en el espacio de liquidez se rige por la ecuación fundamental de Raychaudhuri:
//!
//! $$\frac{d\theta}{d\lambda} = -\frac{1}{2}\theta^2 - \sigma_{ab}\sigma^{ab} + \omega_{ab}\omega^{ab} - R_{ab}k^a k^b$$
//!
//! donde:
//! - $\theta$ es el escalar de expansión de congruencias geodésicas nulas generadas por la profundidad del libro.
//! - $\sigma_{ab}\sigma^{ab} \ge 0$ es la intensidad de cizallamiento (shear) inducida por el flujo agresor.
//! - $R_{ab}k^a k^b \ge 0$ satisface la condición de energía nula (Null Energy Condition, NEC).
//!
//! Condiciones de Riesgo Crítico:
//! 1. Superficie Atrapada Cerrada (Trapped Surface): $\theta_+ < 0$ y $\theta_- < 0$. Ambas congruencias nulas
//!    (saliente y entrante) colapsan hacia adentro. Toda la liquidez se drena sin posibilidad de salida.
//! 2. Singularidad Desnuda (Naked Singularity): Ocurre cuando el parámetro de momento angular normalizado
//!    $a = J/M$ supera la masa efectiva $M$ ($a^2 > M^2$), destruyendo las soluciones reales del horizonte
//!    $r_\pm = M \pm \sqrt{M^2 - a^2}$. El vacío de liquidez se manifiesta directamente en el spread sin amortiguación.
//!
//! En una cuenta pequeña (\$13 USD), operar dentro de una superficie atrapada o ante una singularidad desnuda
//! destruye la cota de ruina de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$. El motor detona vetos deterministas fail-closed.
//!
//! Implementación en Rust puro, sin asignaciones dinámicas en el hot-path ($O(1)$ stack), libre de pánicos
//! y con modulación continua de slot $\eta_{\text{pcc}} \in [0.10, 1.25]$.

/// Capacidad del buffer circular histórico de congruencias nulas en stack.
pub const PCC_HISTORY_CAPACITY: usize = 8;

/// Configuración paramétrica para el motor de censura cósmica de Penrose.
#[derive(Debug, Clone, Copy)]
pub struct PenroseCosmicCensorshipConfig {
    /// Umbral crítico de riesgo de singularidad desnuda $(a/M)^2$.
    pub max_naked_singularity_risk: f64,
    /// Cota mínima admisible para la derivada de Raychaudhuri antes de veto de enfoque.
    pub min_raychaudhuri_derivative: f64,
    /// Umbral crítico de cizallamiento $\sigma^2$ en el libro de órdenes.
    pub max_shear_threshold: f64,
    /// Factor de regularización ultravioleta para división segura.
    pub epsilon: f64,
    /// Tasa de decaimiento temporal exponencial de la memoria de horizonte.
    pub decay_rate: f64,
}

impl Default for PenroseCosmicCensorshipConfig {
    fn default() -> Self {
        Self {
            max_naked_singularity_risk: 0.85,
            min_raychaudhuri_derivative: -4.0,
            max_shear_threshold: 2.5,
            epsilon: 1e-12,
            decay_rate: 0.15,
        }
    }
}

/// Reporte de diagnóstico del motor de Censura Cósmica de Penrose y Raychaudhuri.
#[derive(Debug, Clone, Copy)]
pub struct PenroseCosmicCensorshipReport {
    /// Escalar de expansión nula saliente $\theta_+$.
    pub expansion_theta_plus: f64,
    /// Escalar de expansión nula entrante $\theta_-$.
    pub expansion_theta_minus: f64,
    /// Intensidad de cizallamiento $\sigma^2 \ge 0$.
    pub shear_intensity_sigma2: f64,
    /// Derivada temporal de Raychaudhuri $d\theta/d\lambda$.
    pub raychaudhuri_derivative: f64,
    /// Riesgo normalizado de formación de singularidad desnuda $\in [0.0, 1.0]$.
    pub naked_singularity_risk: f64,
    /// Indicador booleano de superficie atrapada cerrada $(\theta_+ < 0 \land \theta_- < 0)$.
    pub trapped_surface_detected: bool,
    /// Factor de escala continuo de modulación de slot $\eta_{\text{pcc}} \in [0.10, 1.25]$.
    pub slot_scaling_factor: f64,
    /// Veto determinista fail-closed por violación de censura cósmica (singularidad desnuda o superficie atrapada).
    pub penrose_cosmic_censorship_veto: bool,
    /// Veto determinista fail-closed por enfoque catastrófico de Raychaudhuri.
    pub raychaudhuri_singularity_veto: bool,
}

/// Motor de evaluación de Censura Cósmica de Penrose y Ecuación de Raychaudhuri.
#[derive(Debug, Clone)]
pub struct PenroseCosmicCensorshipEngine {
    config: PenroseCosmicCensorshipConfig,
    theta_history: [f64; PCC_HISTORY_CAPACITY],
    head: usize,
    count: usize,
    smoothed_theta_plus: f64,
    smoothed_theta_minus: f64,
    smoothed_shear: f64,
    smoothed_raychaudhuri: f64,
    smoothed_risk: f64,
}

impl PenroseCosmicCensorshipEngine {
    /// Construye una nueva instancia con los parámetros especificados.
    #[inline]
    pub fn new(config: PenroseCosmicCensorshipConfig) -> Self {
        Self {
            config,
            theta_history: [0.0; PCC_HISTORY_CAPACITY],
            head: 0,
            count: 0,
            smoothed_theta_plus: 0.10,
            smoothed_theta_minus: -0.10,
            smoothed_shear: 0.05,
            smoothed_raychaudhuri: 0.0,
            smoothed_risk: 0.10,
        }
    }

    /// Evalúa el horizonte de liquidez y detecta singularidades de Penrose-Raychaudhuri.
    ///
    /// # Parámetros
    /// - `spread_bps`: Spread relativo bid/ask en puntos básicos.
    /// - `depth_imbalance`: Desbalance de profundidad bid/ask $\in [-1.0, 1.0]$.
    /// - `vol_norm`: Volatilidad continua normalizada.
    /// - `drawdown_pct`: Drawdown actual de la cuenta $\in [0.0, 1.0]$.
    pub fn update(
        &mut self,
        spread_bps: f64,
        depth_imbalance: f64,
        vol_norm: f64,
        drawdown_pct: f64,
    ) -> PenroseCosmicCensorshipReport {
        // 1. Sanitización estricta sin pánicos (rb_p1)
        let eps = self.config.epsilon.max(1e-12);
        let s_bps = if spread_bps.is_finite() && spread_bps > 0.0 {
            spread_bps.clamp(0.01, 1000.0)
        } else {
            1.0
        };
        let d_imb = if depth_imbalance.is_finite() {
            depth_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let v_val = if vol_norm.is_finite() && vol_norm >= 0.0 {
            vol_norm.clamp(0.0001, 100.0)
        } else {
            0.01
        };
        let dd = if drawdown_pct.is_finite() && drawdown_pct >= 0.0 {
            drawdown_pct.clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 2. Cálculo de las expansiones nulas saliente (\theta_+) y entrante (\theta_-):
        // \theta_+ representa la tasa de ensanchamiento del soporte de liquidez exterior.
        // \theta_- representa la tasa de contracción del túnel de absorción interior.
        let spread_damping = (1.0 / (1.0 + s_bps * 0.05)).clamp(0.05, 1.0);
        let raw_theta_plus = spread_damping * (1.0 - d_imb.abs() * 0.80) - v_val * 0.20 - dd * 2.0;
        let raw_theta_minus =
            -spread_damping * (1.0 + d_imb.abs() * 0.50) - v_val * 0.30 - dd * 2.0;

        self.smoothed_theta_plus = (1.0 - self.config.decay_rate) * self.smoothed_theta_plus
            + self.config.decay_rate * raw_theta_plus;
        self.smoothed_theta_minus = (1.0 - self.config.decay_rate) * self.smoothed_theta_minus
            + self.config.decay_rate * raw_theta_minus;

        let expansion_theta_plus = self.smoothed_theta_plus;
        let expansion_theta_minus = self.smoothed_theta_minus;

        // 3. Intensidad de Cizallamiento Geodésico \sigma^2:
        // Cizallamiento inducido por la asimetría de profundidad combinada con volatilidad
        let raw_shear = d_imb.powi(2) * 2.0 + (s_bps / 20.0).powi(2) * 0.50 + v_val * 0.10;
        self.smoothed_shear = (1.0 - self.config.decay_rate) * self.smoothed_shear
            + self.config.decay_rate * raw_shear;
        let shear_intensity_sigma2 = self.smoothed_shear.clamp(0.0, 10.0);

        // 4. Registro histórico de \theta y evaluación de Raychaudhuri:
        // d\theta/d\lambda = -1/2 \theta^2 - \sigma^2 - R_{ab}k^a k^b
        let mean_theta = 0.50 * (expansion_theta_plus + expansion_theta_minus);
        self.theta_history[self.head] = mean_theta;
        self.head = (self.head + 1) % PCC_HISTORY_CAPACITY;
        if self.count < PCC_HISTORY_CAPACITY {
            self.count += 1;
        }

        // Término de curvatura de Ricci de materia de órdenes R_{ab} k^a k^b \approx v_val^2 + dd * 10
        let ricci_curvature_energy = (v_val * v_val * 0.50 + dd * 5.0).clamp(0.0, 20.0);
        let instant_raychaudhuri =
            -0.50 * mean_theta * mean_theta - shear_intensity_sigma2 - ricci_curvature_energy;

        self.smoothed_raychaudhuri = (1.0 - self.config.decay_rate) * self.smoothed_raychaudhuri
            + self.config.decay_rate * instant_raychaudhuri;
        let raychaudhuri_derivative = self.smoothed_raychaudhuri;

        // 5. Superficie Atrapada Cerrada:
        // Ocurre cuando ambas congruencias nulas se vuelven estrictamente convergentes
        let trapped_surface_detected = expansion_theta_plus < 0.0 && expansion_theta_minus < 0.0;

        // 6. Masa efectiva de horizonte M y momento angular de spin a:
        // M = v_val * 2.0 + (1.0 - dd) * 5.0
        // a = d_imb.abs() * 4.0 + (s_bps * 0.10)
        let effective_mass_m = (v_val * 0.50 + (1.0 - dd) * 2.0).max(eps);
        let angular_spin_a = (d_imb.abs() * 2.0 + (s_bps / 50.0)).max(0.0);
        let raw_risk =
            (angular_spin_a * angular_spin_a) / (effective_mass_m * effective_mass_m + eps);

        self.smoothed_risk =
            (1.0 - self.config.decay_rate) * self.smoothed_risk + self.config.decay_rate * raw_risk;
        let naked_singularity_risk = self.smoothed_risk.clamp(0.0, 1.0);

        // 7. Vetos Fail-Closed deterministas:
        // Veto 1: Censura cósmica violada por singularidad desnuda o superficie atrapada crítica
        let severe_trapped_collapse = trapped_surface_detected && (expansion_theta_plus < -0.20);
        let naked_singularity_breach =
            naked_singularity_risk > self.config.max_naked_singularity_risk;
        let shear_overload = shear_intensity_sigma2 > self.config.max_shear_threshold;
        let penrose_cosmic_censorship_veto =
            severe_trapped_collapse || naked_singularity_breach || shear_overload;

        // Veto 2: Derivada de Raychaudhuri colapsando hacia singularidad incompleta
        let raychaudhuri_singularity_veto =
            raychaudhuri_derivative < self.config.min_raychaudhuri_derivative;

        // Sanitización final ante fallos numéricos
        let numerical_fault = !expansion_theta_plus.is_finite()
            || !expansion_theta_minus.is_finite()
            || !shear_intensity_sigma2.is_finite()
            || !raychaudhuri_derivative.is_finite()
            || !naked_singularity_risk.is_finite();

        let final_penrose_veto = penrose_cosmic_censorship_veto || numerical_fault;
        let final_raychaudhuri_veto = raychaudhuri_singularity_veto || numerical_fault;

        // 8. Factor continuo de modulación de slot \eta_{\text{pcc}} \in [0.10, 1.25]
        let slot_scaling_factor = if final_penrose_veto || final_raychaudhuri_veto {
            0.10
        } else {
            let risk_factor = (1.0 - naked_singularity_risk * 0.50).clamp(0.20, 1.0);
            let shear_factor = (1.0 / (1.0 + shear_intensity_sigma2 * 0.30)).clamp(0.20, 1.0);
            (0.20 + 0.95 * risk_factor * shear_factor).clamp(0.10, 1.25)
        };

        PenroseCosmicCensorshipReport {
            expansion_theta_plus,
            expansion_theta_minus,
            shear_intensity_sigma2,
            raychaudhuri_derivative,
            naked_singularity_risk,
            trapped_surface_detected,
            slot_scaling_factor,
            penrose_cosmic_censorship_veto: final_penrose_veto,
            raychaudhuri_singularity_veto: final_raychaudhuri_veto,
        }
    }
}

impl Default for PenroseCosmicCensorshipEngine {
    fn default() -> Self {
        Self::new(PenroseCosmicCensorshipConfig::default())
    }
}
