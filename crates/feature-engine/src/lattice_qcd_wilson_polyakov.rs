//! # Fase 82 (§82): Teoría de Campos de Gauge en Retículo (Lattice QCD) & Lazos de Wilson-Polyakov
//!
//! ## 🏛️ Gobernanza y Mandato del Consejo Colegiado (10 Roles Senior)
//! Módulo formalmente integrado en cumplimiento estricto del Plan Maestro Unificado Omega-220
//! (`docs/PLAN_MAESTRO_UNIFICADO_OMEGA220_CUANTICO.md`). Modela el flujo multivariante continuo
//! de microestructura como una teoría de campos de gauge no abeliana en un retículo espaciotemporal
//! continuo $4 \times 4$ de variables de enlace unitarias $U_\mu(x) \in SU(2) \subset SU(3)$.
//!
//! ## 👨‍🏫 Modo Profesor: Explicación Integral de la Fase 82
//! - **QUÉ**: Motor analítico de Lattice QCD discreto en stack que calcula variables de enlace $U_\mu(x)$,
//!   plaquetas elementales de Wilson $U_{\mu\nu}(x)$, la acción euclidiana de Wilson $S[U]$,
//!   lazos de Wilson $W(C)$, lazos térmicos de Polyakov $L(\vec{x})$ y la tensión de cuerda de confinamiento $\sigma$.
//! - **POR QUÉ**: La liquidez en libros de órdenes de alta frecuencia exhibe fenómenos no lineales de
//!   hadronización y confinamiento cromodinámico. Cuando la liquidez se congela en racimos densos, la
//!   tensión de cuerda entre niveles de precios se intensifica; una rotura de cuerda gauge induce deslizamientos
//!   catastróficos que no son detectables con autocorrelaciones lineales ordinarias.
//! - **PARA QUÉ**: Computar el parámetro de orden de desconfinamiento $\langle |L| \rangle$, la acción media
//!   de plaqueta $\langle P \rangle \in [0.0, 1.0]$ y modular el factor de slot continuo $\eta_{\text{lattice}} \in [0.10, 1.25]$.
//!   Si se detecta colapso cromodinámico por singularidad de acción o confinamiento extremo sin flujo, detonar
//!   el veto determinista absoluto `LATTICE_GAUGE_CONFINEMENT_COLLAPSE_VETO` forzando slot 0.0.
//! - **CÓMO**:
//!   1. Retículo en stack $N_s \times N_\tau = 4 \times 4$ con 2 dimensiones relevantes (espacio de precios y tiempo de micro-ticks).
//!   2. Cada enlace porta una matriz $2 \times 2$ compleja $U_\mu(x) = a_0 I + i \vec{a} \cdot \vec{\sigma} \in SU(2)$
//!      con $a_0^2 + |\vec{a}|^2 = 1$.
//!   3. Plaqueta $U_{\mu\nu}(x) = U_\mu(x) U_\nu(x + \hat{\mu}) U_\mu^\dagger(x + \hat{\nu}) U_\nu^\dagger(x)$.
//!   4. Acción de Wilson: $S[U] = \beta \sum_{\square} (1 - \frac{1}{2} \operatorname{Re}\operatorname{Tr}(U_\square))$.
//!   5. Lazo de Polyakov térmico a lo largo de la dimensión temporal periódica: $L(\vec{x}) = \frac{1}{2}\operatorname{Tr}\prod_{t=1}^{N_\tau} U_0(\vec{x}, t)$.
//! - **CUÁNDO**: En cada evaluación del grafo vivo para activos del portafolio continuo.
//! - **DÓNDE**: `crates/feature-engine/src/lattice_qcd_wilson_polyakov.rs`.
//! - **QUIÉN**: `LatticeQcdWilsonPolyakovEngine`.
//!
//! ## 🔒 Invariantes y Garantías de Silicio
//! - Cero asignaciones en heap ($O(1)$ stack allocation estricto).
//! - Ratchet de cero pánicos (`rb_p1`): 0 unwrap, 0 expect, 0 panic.
//! - Inmunidad contra subnormales y NaN mediante `f64::clamp` y `is_finite`.

/// Dimensión espacial del retículo continuo en stack.
pub const LATTICE_DIM_SPATIAL: usize = 4;
/// Dimensión temporal (micro-ticks) del retículo continuo en stack.
pub const LATTICE_DIM_TEMPORAL: usize = 4;
/// Número total de plaquetas elementales en el retículo 2D proyectado.
pub const LATTICE_PLAQUETTE_COUNT: usize = LATTICE_DIM_SPATIAL * LATTICE_DIM_TEMPORAL;

/// Matriz compleja unitaria $2 \times 2$ en $SU(2)$ sin asignación dinámica.
#[derive(Debug, Clone, Copy)]
pub struct Su2Matrix {
    /// Parte real de la componente (0, 0).
    pub r00: f64,
    /// Parte imaginaria de la componente (0, 0).
    pub i00: f64,
    /// Parte real de la componente (0, 1).
    pub r01: f64,
    /// Parte imaginaria de la componente (0, 1).
    pub i01: f64,
    /// Parte real de la componente (1, 0): para $SU(2)$, es $-r_{01}$.
    pub r10: f64,
    /// Parte imaginaria de la componente (1, 0): para $SU(2)$, es $i_{01}$.
    pub i10: f64,
    /// Parte real de la componente (1, 1): para $SU(2)$, es $r_{00}$.
    pub r11: f64,
    /// Parte imaginaria de la componente (1, 1): para $SU(2)$, es $-i_{00}$.
    pub i11: f64,
}

impl Default for Su2Matrix {
    fn default() -> Self {
        Self::identity()
    }
}

impl Su2Matrix {
    /// Matriz identidad $I_{2 \times 2}$.
    #[inline(always)]
    pub const fn identity() -> Self {
        Self {
            r00: 1.0,
            i00: 0.0,
            r01: 0.0,
            i01: 0.0,
            r10: 0.0,
            i10: 0.0,
            r11: 1.0,
            i11: 0.0,
        }
    }

    /// Construye una matriz $U \in SU(2)$ a partir de los coeficientes de cuaternión unitario:
    /// $U = a_0 I + i (a_1 \sigma_1 + a_2 \sigma_2 + a_3 \sigma_3)$.
    #[inline]
    pub fn from_quaternion(a0: f64, a1: f64, a2: f64, a3: f64) -> Self {
        let norm_sq = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        let inv_norm = if norm_sq > 1e-12 && norm_sq.is_finite() {
            1.0 / norm_sq.sqrt()
        } else {
            1.0
        };
        let u0 = a0 * inv_norm;
        let u1 = a1 * inv_norm;
        let u2 = a2 * inv_norm;
        let u3 = a3 * inv_norm;

        Self {
            r00: u0,
            i00: u3,
            r01: u2,
            i01: u1,
            r10: -u2,
            i10: u1,
            r11: u0,
            i11: -u3,
        }
    }

    /// Multiplicación matricial $C = A \cdot B$ en $O(1)$ sin heap.
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        let c_r00 =
            self.r00 * rhs.r00 - self.i00 * rhs.i00 + self.r01 * rhs.r10 - self.i01 * rhs.i10;
        let c_i00 =
            self.r00 * rhs.i00 + self.i00 * rhs.r00 + self.r01 * rhs.i10 + self.i01 * rhs.r10;

        let c_r01 =
            self.r00 * rhs.r01 - self.i00 * rhs.i01 + self.r01 * rhs.r11 - self.i01 * rhs.i11;
        let c_i01 =
            self.r00 * rhs.i01 + self.i00 * rhs.r01 + self.r01 * rhs.i11 + self.i01 * rhs.r11;

        let c_r10 =
            self.r10 * rhs.r00 - self.i10 * rhs.i00 + self.r11 * rhs.r10 - self.i11 * rhs.i10;
        let c_i10 =
            self.r10 * rhs.i00 + self.i10 * rhs.r00 + self.r11 * rhs.i10 + self.i11 * rhs.r10;

        let c_r11 =
            self.r10 * rhs.r01 - self.i10 * rhs.i01 + self.r11 * rhs.r11 - self.i11 * rhs.i11;
        let c_i11 =
            self.r10 * rhs.i01 + self.i10 * rhs.r01 + self.r11 * rhs.i11 + self.i11 * rhs.r11;

        Self {
            r00: c_r00,
            i00: c_i00,
            r01: c_r01,
            i01: c_i01,
            r10: c_r10,
            i10: c_i10,
            r11: c_r11,
            i11: c_i11,
        }
    }

    /// Adjunta hermítica $U^\dagger$.
    #[inline(always)]
    pub fn dagger(&self) -> Self {
        Self {
            r00: self.r00,
            i00: -self.i00,
            r01: self.r10,
            i01: -self.i10,
            r10: self.r01,
            i10: -self.i01,
            r11: self.r11,
            i11: -self.i11,
        }
    }

    /// Traza normalizada $\frac{1}{2} \operatorname{Re}\operatorname{Tr}(U)$.
    #[inline(always)]
    pub fn half_real_trace(&self) -> f64 {
        0.5 * (self.r00 + self.r11)
    }

    /// Módulo de la traza $|\frac{1}{2} \operatorname{Tr}(U)|$.
    #[inline]
    pub fn trace_norm(&self) -> f64 {
        let half_re = 0.5 * (self.r00 + self.r11);
        let half_im = 0.5 * (self.i00 + self.i11);
        (half_re * half_re + half_im * half_im).sqrt()
    }
}

/// Configuración de la teoría de campos de gauge en retículo.
#[derive(Debug, Clone, Copy)]
pub struct LatticeQcdConfig {
    /// Constante de acoplamiento inverso $\beta = 4/g^2$.
    pub beta_coupling: f64,
    /// Umbral crítico de acción media de plaqueta para shock de confinamiento.
    pub critical_action_threshold: f64,
    /// Umbral mínimo de valor esperado de Polyakov para desconfinamiento.
    pub polyakov_deconfinement_floor: f64,
}

impl Default for LatticeQcdConfig {
    fn default() -> Self {
        Self {
            beta_coupling: 2.30,
            critical_action_threshold: 1.85,
            polyakov_deconfinement_floor: 0.05,
        }
    }
}

/// Informe analítico de Lattice QCD y lazos de Wilson-Polyakov.
#[derive(Debug, Clone, Copy)]
pub struct LatticeQcdReport {
    /// Acción media de plaqueta de Wilson $\langle P \rangle = \frac{1}{N_\square} \sum \frac{1}{2}\operatorname{Re}\operatorname{Tr}(U_\square) \in [-1.0, 1.0]$.
    pub average_plaquette: f64,
    /// Acción total de Wilson $S[U] = \beta \sum (1 - P_\square) \ge 0$.
    pub wilson_action: f64,
    /// Valor esperado del lazo térmico de Polyakov $\langle |L| \rangle \in [0.0, 1.0]$.
    pub polyakov_loop_expectation: f64,
    /// Tensión de cuerda efectiva de confinamiento $\sigma_{\text{string}} \ge 0$.
    pub string_tension_sigma: f64,
    /// Indicador booleano de veto determinista por colapso de confinamiento cromodinámico.
    pub lattice_confinement_collapse_veto: bool,
    /// Factor de modulación continuo de slot $\eta_{\text{lattice}} \in [0.10, 1.25]$.
    pub slot_scale_factor: f64,
}

/// Motor determinista de Lattice QCD y Lazos de Wilson-Polyakov.
#[derive(Debug, Clone)]
pub struct LatticeQcdWilsonPolyakovEngine {
    config: LatticeQcdConfig,
    /// Enlaces en dirección espacial $\mu = 1$: $U_1(x, t)$.
    links_spatial: [[Su2Matrix; LATTICE_DIM_TEMPORAL]; LATTICE_DIM_SPATIAL],
    /// Enlaces en dirección temporal $\mu = 0$: $U_0(x, t)$.
    links_temporal: [[Su2Matrix; LATTICE_DIM_TEMPORAL]; LATTICE_DIM_SPATIAL],
}

impl Default for LatticeQcdWilsonPolyakovEngine {
    fn default() -> Self {
        Self::new(LatticeQcdConfig::default())
    }
}

impl LatticeQcdWilsonPolyakovEngine {
    /// Instancia un nuevo motor con configuración dada.
    pub fn new(config: LatticeQcdConfig) -> Self {
        Self {
            config,
            links_spatial: [[Su2Matrix::identity(); LATTICE_DIM_TEMPORAL]; LATTICE_DIM_SPATIAL],
            links_temporal: [[Su2Matrix::identity(); LATTICE_DIM_TEMPORAL]; LATTICE_DIM_SPATIAL],
        }
    }

    /// Actualiza el retículo continuo a partir de observables multivariantes de microestructura:
    /// - `volatility`: Volatilidad instantánea normalizada.
    /// - `obi`: Order Book Imbalance $\in [-1.0, 1.0]$.
    /// - `micro_spread_bps`: Spread efectivo en puntos básicos.
    /// - `flow_divergence`: Divergencia del flujo de órdenes continuo.
    pub fn update(
        &mut self,
        volatility: f64,
        obi: f64,
        micro_spread_bps: f64,
        flow_divergence: f64,
    ) -> LatticeQcdReport {
        let vol_clean = if volatility.is_finite() && volatility > 0.0 {
            volatility
        } else {
            0.01
        };
        let obi_clean = if obi.is_finite() {
            obi.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let spread_clean = if micro_spread_bps.is_finite() && micro_spread_bps >= 0.0 {
            micro_spread_bps
        } else {
            1.0
        };
        let flow_clean = if flow_divergence.is_finite() {
            flow_divergence.clamp(-10.0, 10.0)
        } else {
            0.0
        };

        // Proyecta observables a enlaces espaciotemporales con rotaciones locales de gauge
        for x in 0..LATTICE_DIM_SPATIAL {
            for t in 0..LATTICE_DIM_TEMPORAL {
                let x_phase = (x as f64 + 1.0) * 0.5;
                let t_phase = (t as f64 + 1.0) * 0.5;

                // Enlace espacial U_1(x, t): gobernado por OBI y spread
                let a0_s = (1.0 - (obi_clean * 0.5).powi(2)).sqrt();
                let a1_s = obi_clean * 0.3 * x_phase.sin();
                let a2_s = (spread_clean * 0.01).tanh() * 0.2;
                let a3_s = vol_clean.sin() * 0.2 * t_phase.cos();
                self.links_spatial[x][t] = Su2Matrix::from_quaternion(a0_s, a1_s, a2_s, a3_s);

                // Enlace temporal U_0(x, t): gobernado por flujo y volatilidad
                let a0_t = (1.0 - (flow_clean * 0.1).tanh().powi(2)).sqrt();
                let a1_t = (flow_clean * 0.1).tanh() * 0.4;
                let a2_t = vol_clean.tanh() * 0.3 * t_phase.sin();
                let a3_t = obi_clean * 0.2 * x_phase.cos();
                self.links_temporal[x][t] = Su2Matrix::from_quaternion(a0_t, a1_t, a2_t, a3_t);
            }
        }

        // Computa plaquetas de Wilson elementales U_{\mu\nu}(x, t) con condiciones de frontera periódicas
        let mut total_plaquette_re_tr = 0.0;
        let mut total_wilson_action = 0.0;

        for x in 0..LATTICE_DIM_SPATIAL {
            let next_x = (x + 1) % LATTICE_DIM_SPATIAL;
            for t in 0..LATTICE_DIM_TEMPORAL {
                let next_t = (t + 1) % LATTICE_DIM_TEMPORAL;

                // Plaqueta: U_square = U_1(x, t) * U_0(next_x, t) * U_1^\dagger(x, next_t) * U_0^\dagger(x, t)
                let u_1_xt = self.links_spatial[x][t];
                let u_0_nxt = self.links_temporal[next_x][t];
                let u_1_xnt_dag = self.links_spatial[x][next_t].dagger();
                let u_0_xt_dag = self.links_temporal[x][t].dagger();

                let p1 = u_1_xt.mul(&u_0_nxt);
                let p2 = p1.mul(&u_1_xnt_dag);
                let plaquette = p2.mul(&u_0_xt_dag);

                let p_re_tr = plaquette.half_real_trace().clamp(-1.0, 1.0);
                total_plaquette_re_tr += p_re_tr;
                total_wilson_action += self.config.beta_coupling * (1.0 - p_re_tr);
            }
        }

        let num_plaquettes = LATTICE_PLAQUETTE_COUNT as f64;
        let avg_plaquette = total_plaquette_re_tr / num_plaquettes;
        let wilson_action = total_wilson_action;

        // Computa el lazo térmico de Polyakov para cada línea espacial:
        // L(x) = \frac{1}{2} \operatorname{Tr} \prod_{t=0}^{N_\tau-1} U_0(x, t)
        let mut sum_polyakov_norm = 0.0;
        for x in 0..LATTICE_DIM_SPATIAL {
            let mut polyakov_prod = Su2Matrix::identity();
            for t in 0..LATTICE_DIM_TEMPORAL {
                polyakov_prod = polyakov_prod.mul(&self.links_temporal[x][t]);
            }
            sum_polyakov_norm += polyakov_prod.trace_norm();
        }
        let polyakov_loop_expectation = sum_polyakov_norm / (LATTICE_DIM_SPATIAL as f64);

        // Tensión de cuerda analítica en retículo: \sigma_{\text{string}} \approx -\ln(\langle P \rangle)
        let string_tension_sigma = if avg_plaquette > 1e-4 {
            (-avg_plaquette.ln()).max(0.0)
        } else {
            10.0
        };

        // Detección de veto determinista: acción excesiva o colapso de desconfinamiento
        let lattice_confinement_collapse_veto = wilson_action
            >= (self.config.critical_action_threshold * num_plaquettes)
            || polyakov_loop_expectation < self.config.polyakov_deconfinement_floor;

        let slot_scale_factor = if lattice_confinement_collapse_veto {
            0.0
        } else {
            let plaquette_factor = 0.5 + 0.5 * avg_plaquette.max(0.0);
            let polyakov_factor = (polyakov_loop_expectation * 1.5).min(1.20);
            (plaquette_factor * polyakov_factor).clamp(0.10, 1.25)
        };

        LatticeQcdReport {
            average_plaquette: avg_plaquette,
            wilson_action,
            polyakov_loop_expectation,
            string_tension_sigma,
            lattice_confinement_collapse_veto,
            slot_scale_factor,
        }
    }
}
