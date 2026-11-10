//! Fase 111 (§111): Teoría del Absorbedor de Wheeler-Feynman & Reacción de Radiación de Dirac
//!
//! Modela la interacción no local temporalmente simétrica entre la ejecución de órdenes y la
//! liquidez del mercado a través de funciones de Green retardadas y avanzadas:
//! $$G_{\text{sym}} = \frac{1}{2}(G_{\text{ret}} + G_{\text{adv}})$$
//! y la fuerza de reacción de radiación de Dirac $F_{\text{rad}} = \frac{2}{3}\frac{q^2}{c^3}\dddot{x}$ (jerk de liquidez).
//!
//! # Invariantes de Silicio:
//! - Cero asignaciones dinámicas en hot-path ($O(1)$ estricto).
//! - Invariante `rb_p1` (0 panics, 0 unwraps, 0 expects).
//! - Buffer circular de derivadas temporales hasta orden 3 (jerk) en stack.

/// Capacidad del buffer circular de precios para cálculo de jerk
pub const WFA_HISTORY_CAPACITY: usize = 8;

/// Configuración determinista para el motor de Absorbedor de Wheeler-Feynman
#[derive(Debug, Clone, Copy)]
pub struct WheelerFeynmanAbsorberConfig {
    /// Constante de acoplamiento de radiación de Dirac $\kappa = \frac{2}{3} \frac{q^2}{c^3}$
    pub radiation_damping_coupling: f64,
    /// Déficit máximo permitido del absorbedor $\Delta_{\text{absorber}}$
    pub max_absorber_deficit: f64,
    /// Umbral crítico de jerk para detección de modo auto-acelerado de fuga (runaway mode)
    pub max_runaway_jerk_threshold: f64,
    /// Factor de escala mínimo ante veto
    pub scale_min: f64,
    /// Factor de escala máximo en absorción completa
    pub scale_max: f64,
}

impl Default for WheelerFeynmanAbsorberConfig {
    fn default() -> Self {
        Self {
            radiation_damping_coupling: 0.6666666667, // 2/3 canónico
            max_absorber_deficit: 0.65,
            max_runaway_jerk_threshold: 15.0,
            scale_min: 0.10,
            scale_max: 1.25,
        }
    }
}

/// Informe de evaluación del motor de Absorbedor de Wheeler-Feynman
#[derive(Debug, Clone, Copy)]
pub struct WheelerFeynmanAbsorberReport {
    /// Jerk de mercado normalizado $\dddot{P}$ (derivada temporal de tercer orden)
    pub liquidity_jerk: f64,
    /// Fuerza de reacción de radiación de Dirac $F_{\text{rad}}$
    pub dirac_radiation_reaction_force: f64,
    /// Capacidad de absorción del medio de mercado $\rho_{\text{absorber}}$
    pub absorber_absorption_capacity: f64,
    /// Déficit del absorbedor $\Delta_{\text{absorber}} = |1 - \rho / \rho_{\text{ideal}}|$
    pub absorber_deficit: f64,
    /// Indicador de modo auto-acelerado de fuga (pre-acceleration runaway mode)
    pub pre_acceleration_runaway_detected: bool,
    /// Veto fail-closed de colapso del absorbedor de Wheeler-Feynman
    pub wheeler_feynman_absorber_veto: bool,
    /// Veto fail-closed de fuga de reacción de radiación de Dirac
    pub dirac_radiation_runaway_veto: bool,
    /// Factor continuo de modulación de slot $\eta_{\text{absorber}} \in [0.10, 1.25]$
    pub slot_scale: f64,
}

/// Motor de Absorbedor de Wheeler-Feynman en stack $O(1)$
#[derive(Debug, Clone)]
pub struct WheelerFeynmanAbsorberEngine {
    config: WheelerFeynmanAbsorberConfig,
    price_history: [f64; WFA_HISTORY_CAPACITY],
    velocity_history: [f64; WFA_HISTORY_CAPACITY],
    acceleration_history: [f64; WFA_HISTORY_CAPACITY],
    head: usize,
    count: usize,
}

impl WheelerFeynmanAbsorberEngine {
    /// Construye una nueva instancia con configuración dada
    pub fn new(config: WheelerFeynmanAbsorberConfig) -> Self {
        Self {
            config,
            price_history: [0.0; WFA_HISTORY_CAPACITY],
            velocity_history: [0.0; WFA_HISTORY_CAPACITY],
            acceleration_history: [0.0; WFA_HISTORY_CAPACITY],
            head: 0,
            count: 0,
        }
    }

    /// Evalúa en stack la reacción de radiación y absorción temporal simétrica.
    ///
    /// # Argumentos:
    /// - `mid_price`: Precio medio actual $\ge 0$
    /// - `ofi`: Desequilibrio de flujo de órdenes $\in [-1, 1]$
    /// - `vol_norm`: Volatilidad normalizada $\ge 0$
    /// - `spread_bps`: Spread relativo en bps $\ge 0$
    pub fn update(
        &mut self,
        mid_price: f64,
        ofi: f64,
        vol_norm: f64,
        spread_bps: f64,
    ) -> WheelerFeynmanAbsorberReport {
        let p = if mid_price.is_finite() && mid_price > 0.0 {
            mid_price
        } else {
            100.0
        };
        let flow = if ofi.is_finite() { ofi.clamp(-1.0, 1.0) } else { 0.0 };
        let vol = if vol_norm.is_finite() && vol_norm > 0.0 {
            vol_norm.clamp(0.01, 10.0)
        } else {
            1.0
        };
        let spr = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.1, 50.0)
        } else {
            1.0
        };

        // 1. Cálculo de derivadas sucesivas hasta orden 3 (jerk)
        let prev_idx = if self.head == 0 {
            WFA_HISTORY_CAPACITY - 1
        } else {
            self.head - 1
        };

        let v = if self.count > 0 {
            let p_prev = self.price_history[prev_idx];
            if p_prev > 0.0 {
                ((p - p_prev) / p_prev) * 1000.0 // Velocidad normalizada
            } else {
                0.0
            }
        } else {
            0.0
        };

        let a = if self.count > 1 {
            let v_prev = self.velocity_history[prev_idx];
            v - v_prev // Aceleración normalizada
        } else {
            0.0
        };

        let jerk = if self.count > 2 {
            let a_prev = self.acceleration_history[prev_idx];
            a - a_prev // Jerk normalizado (\dddot{P})
        } else {
            0.0
        };

        self.price_history[self.head] = p;
        self.velocity_history[self.head] = v;
        self.acceleration_history[self.head] = a;
        self.head = (self.head + 1) % WFA_HISTORY_CAPACITY;
        if self.count < WFA_HISTORY_CAPACITY {
            self.count += 1;
        }

        // 2. Fuerza de reacción de radiación de Dirac F_rad = \kappa * jerk
        let dirac_force = (self.config.radiation_damping_coupling * jerk).clamp(-50.0, 50.0);

        // 3. Capacidad de absorción del medio de mercado \rho_absorber
        // El medio actúa como absorbedor completo si la liquidez local (inversa de spread y vol)
        // puede disipar la radiación sin generar ondas avanzadas residuales
        let liquidity_density = (10.0 / (spr * 0.20 + vol * 0.80)).clamp(0.05, 20.0);
        let absorber_capacity = (liquidity_density * (1.0 - flow.abs() * 0.30)).clamp(0.01, 10.0);
        let ideal_capacity = 5.0f64;

        // 4. Déficit del absorbedor \Delta_absorber: déficit relativo cuando la absorción es insuficiente
        let absorber_deficit = ((ideal_capacity - absorber_capacity) / ideal_capacity)
            .max(0.0)
            .clamp(0.0, 1.0);

        // 5. Detección de modo de fuga de auto-aceleración (runaway pre-acceleration mode)
        // Ocurre cuando la fuerza de radiación se alinea con la velocidad (anti-amortiguamiento)
        // y el jerk excede el umbral de seguridad
        let is_runaway = jerk.abs() > self.config.max_runaway_jerk_threshold
            || (dirac_force * v > 5.0 && jerk.abs() > 5.0);

        // 6. Condiciones de veto:
        // - WHEELER_FEYNMAN_ABSORBER_COLLAPSE_VETO: déficit de absorción crítico
        let wheeler_feynman_absorber_veto = absorber_deficit > self.config.max_absorber_deficit;

        // - DIRAC_RADIATION_RUNAWAY_VETO: régimen de auto-aceleración no acotado
        let dirac_radiation_runaway_veto = is_runaway;

        // 7. Modulación continua de slot
        let raw_scale = if wheeler_feynman_absorber_veto || dirac_radiation_runaway_veto {
            self.config.scale_min
        } else {
            let absorption_margin = (self.config.max_absorber_deficit - absorber_deficit)
                / (self.config.max_absorber_deficit + 1e-6);
            let jerk_safety = (self.config.max_runaway_jerk_threshold - jerk.abs()).max(0.0)
                / (self.config.max_runaway_jerk_threshold + 1.0);
            1.0 + (absorption_margin * 0.12) + (jerk_safety * 0.08)
        };

        let slot_scale = raw_scale.clamp(self.config.scale_min, self.config.scale_max);

        WheelerFeynmanAbsorberReport {
            liquidity_jerk: jerk,
            dirac_radiation_reaction_force: dirac_force,
            absorber_absorption_capacity: absorber_capacity,
            absorber_deficit,
            pre_acceleration_runaway_detected: is_runaway,
            wheeler_feynman_absorber_veto,
            dirac_radiation_runaway_veto,
            slot_scale,
        }
    }
}
