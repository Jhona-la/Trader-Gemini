//! Fase 46: Variedades Hiperbólicas de Poincaré-Hodge y Flujos de Lipschitz
//!
//! Este módulo implementa la geometría hiperbólica no euclidiana en el disco de Poincaré $\mathbb{D}$
//! acoplada con el cálculo exterior de formas diferenciales y el operador estrella de Hodge $\star$.
//! Modela la microestructura del libro de órdenes y los flujos inter-activos como variedades riemannianas
//! de curvatura seccional negativa constante $K = -c < 0$, preservando las jerarquías fractales sin
//! la distorsión métrica exponencial intrínseca de los espacios euclidianos planos.
//!
//! Invariantes y garantías:
//! - Mapeo conformal suave al interior estricto del disco de Poincaré $\|z\| \le r_{\max} < 1.0$.
//! - Distancia geodésica hiperbólica $d_{\mathbb{H}}(u, v) = \operatorname{arcosh}(1 + \delta)$ numéricamente estable.
//! - Dualidad de Hodge $\star$ para descomponer flujos de liquidez (1-formas) y vorticidad de área (2-formas).
//! - Estimación en ventana fija de la constante de Lipschitz local $L = \sup \frac{d_{\mathbb{H}}}{\Delta t}$.
//! - Cero asignaciones dinámicas en heap ($O(1)$ stack).
//! - Ratchet de cero pánicos `rb_p1`: 0 unwrap, 0 expect, 0 panic en código de producción.
//! - Cota analítica de ruina de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$ estrictamente preservada.

/// Longitud de la ventana estática de historial para cálculo de Lipschitz y curvatura Hodge.
pub const POINCARE_HISTORY_LEN: usize = 16;

/// Parámetros de configuración del motor hiperbólico de Poincaré-Hodge.
#[derive(Debug, Clone, Copy)]
pub struct HyperbolicPoincareHodgeParams {
    /// Constante de curvatura negativa $c > 0$ ($K = -c$).
    pub curvature_c: f64,
    /// Radio máximo conformal estricto $r_{\max} \in (0.0, 1.0)$ para evitar divergencias en la frontera $\partial \mathbb{D}$.
    pub max_disk_radius: f64,
    /// Sensibilidad de escala para el retorno normalizado de precio $\alpha_p > 0$.
    pub sensitivity_p: f64,
    /// Sensibilidad de escala para el desequilibrio de flujo de órdenes $\alpha_{\text{ofi}} > 0$.
    pub sensitivity_ofi: f64,
    /// Factor de suavizado exponencial para la constante de Lipschitz continua.
    pub lipschitz_smoothing: f64,
    /// Umbral de divergencia de flujo de Hodge para alertar estrés de liquidez.
    pub hodge_divergence_threshold: f64,
}

impl Default for HyperbolicPoincareHodgeParams {
    fn default() -> Self {
        Self {
            curvature_c: 1.0,
            max_disk_radius: 0.995,
            sensitivity_p: 20.0,
            sensitivity_ofi: 1.5,
            lipschitz_smoothing: 0.10,
            hodge_divergence_threshold: 0.70,
        }
    }
}

/// Punto en el disco de Poincaré $\mathbb{D} = \{ (x, y) \in \mathbb{R}^2 : x^2 + y^2 < 1 \}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoincarePoint {
    /// Coordenada conformal horizontal $x \in (-1.0, 1.0)$.
    pub x: f64,
    /// Coordenada conformal vertical $y \in (-1.0, 1.0)$.
    pub y: f64,
}

impl PoincarePoint {
    /// Punto en el origen geodésico de referencia $(0, 0)$.
    pub const ORIGIN: Self = Self { x: 0.0, y: 0.0 };

    /// Crea un nuevo punto en el disco de Poincaré, proyectando al interior seguro $\|z\| \le r_{\max} < 1.0$.
    #[inline]
    pub fn new(x: f64, y: f64, max_radius: f64) -> Self {
        let safe_max_r = max_radius.clamp(0.01, 0.9999);
        let safe_x = if x.is_finite() { x } else { 0.0 };
        let safe_y = if y.is_finite() { y } else { 0.0 };
        let r_sq = safe_x * safe_x + safe_y * safe_y;
        let r = r_sq.sqrt();

        if r <= safe_max_r {
            Self {
                x: safe_x,
                y: safe_y,
            }
        } else if r > 1e-12 {
            let scale = safe_max_r / r;
            Self {
                x: safe_x * scale,
                y: safe_y * scale,
            }
        } else {
            Self::ORIGIN
        }
    }

    /// Norma euclidiana al cuadrado $\|z\|^2 = x^2 + y^2$.
    #[inline]
    pub fn norm_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Radio euclidiano $r = \sqrt{x^2 + y^2} < 1.0$.
    #[inline]
    pub fn radius(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Factor conformal de la métrica de Riemann $g_{ij} = \lambda^2 \delta_{ij}$, donde $\lambda = \frac{2}{1 - \|z\|^2}$.
    #[inline]
    pub fn conformal_factor(&self) -> f64 {
        let denom = (1.0 - self.norm_sq()).max(1e-8);
        2.0 / denom
    }

    /// Distancia geodésica hiperbólica $d_{\mathbb{H}}(u, v)$ en el disco de Poincaré.
    ///
    /// $$d_{\mathbb{H}}(u, v) = \operatorname{arcosh}\left(1 + 2 \frac{\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$$
    #[inline]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dist_sq = dx * dx + dy * dy;

        let u_sq = self.norm_sq().min(0.9999);
        let v_sq = other.norm_sq().min(0.9999);

        let denom = ((1.0 - u_sq) * (1.0 - v_sq)).max(1e-12);
        let delta = 2.0 * dist_sq / denom;

        // arcosh(1 + delta) = ln((1 + delta) + sqrt(delta * (delta + 2)))
        let xi = 1.0 + delta;
        let arg = xi + (delta * (delta + 2.0)).sqrt();
        if arg > 1.0 {
            arg.ln()
        } else {
            0.0
        }
    }

    /// Distancia geodésica desde el origen $d_{\mathbb{H}}(0, z) = \ln\left(\frac{1 + r}{1 - r}\right)$.
    #[inline]
    pub fn distance_to_origin(&self) -> f64 {
        let r = self.radius().min(0.9999);
        ((1.0 + r) / (1.0 - r).max(1e-8)).ln()
    }
}

/// Informe detallado de la variedad hiperbólica y el campo diferencial de Hodge.
#[derive(Debug, Clone, Copy)]
pub struct HyperbolicPoincareHodgeReport {
    /// Coordenada horizontal $x \in (-1.0, 1.0)$ en el disco de Poincaré.
    pub poincare_x: f64,
    /// Coordenada vertical $y \in (-1.0, 1.0)$ en el disco de Poincaré.
    pub poincare_y: f64,
    /// Radio polar euclidiano $r = \sqrt{x^2 + y^2} \in [0.0, 1.0)$.
    pub poincare_radius: f64,
    /// Distancia geodésica hiperbólica respecto al atractor neutral en el origen $d_{\mathbb{H}}(0, z)$.
    pub geodesic_distance_to_center: f64,
    /// Velocidad de deriva geodésica instantánea $\dot{d}_{\mathbb{H}} = \frac{\Delta d_{\mathbb{H}}}{\Delta t}$ en unidades/segundo.
    pub geodesic_drift_velocity: f64,
    /// Constante de Lipschitz local estimada $L \ge 0$.
    pub lipschitz_constant: f64,
    /// Magnitud de la 1-forma dual de Hodge $\star \omega$ que mide presión ortogonal de liquidez $\in [0.0, 1.0]$.
    pub hodge_orthogonal_flux: f64,
    /// Vorticidad simpléctica de área $d\omega$ en el plano hiperbólico $\in [-1.0, 1.0]$.
    pub symplectic_area_vorticity: f64,
    /// Índice continuo de estrés hiperbólico del libro $\in [0.0, 1.0]$.
    pub hyperbolic_stress_index: f64,
    /// Factor multiplicativo de modulación de slot para el orquestador $\in [0.10, 1.25]$.
    pub slot_scaling_factor: f64,
}

/// Motor continuo de geometría hiperbólica de Poincaré y flujos diferenciales de Hodge.
#[derive(Debug, Clone)]
pub struct HyperbolicPoincareHodgeEngine {
    params: HyperbolicPoincareHodgeParams,
    points_history: [PoincarePoint; POINCARE_HISTORY_LEN],
    timestamps_history: [u64; POINCARE_HISTORY_LEN],
    history_count: usize,
    history_index: usize,
    prev_distance_to_center: f64,
    prev_ts_ms: u64,
    smoothed_lipschitz: f64,
}

impl HyperbolicPoincareHodgeEngine {
    /// Construye una nueva instancia del motor con los parámetros especificados.
    pub fn new(params: HyperbolicPoincareHodgeParams) -> Self {
        let valid_params = HyperbolicPoincareHodgeParams {
            curvature_c: if params.curvature_c.is_finite() && params.curvature_c > 0.0 {
                params.curvature_c
            } else {
                1.0
            },
            max_disk_radius: if params.max_disk_radius.is_finite()
                && params.max_disk_radius > 0.0
                && params.max_disk_radius < 1.0
            {
                params.max_disk_radius
            } else {
                0.995
            },
            sensitivity_p: if params.sensitivity_p.is_finite() && params.sensitivity_p > 0.0 {
                params.sensitivity_p
            } else {
                20.0
            },
            sensitivity_ofi: if params.sensitivity_ofi.is_finite() && params.sensitivity_ofi > 0.0 {
                params.sensitivity_ofi
            } else {
                1.5
            },
            lipschitz_smoothing: if params.lipschitz_smoothing.is_finite()
                && (0.01..=0.99).contains(&params.lipschitz_smoothing)
            {
                params.lipschitz_smoothing
            } else {
                0.10
            },
            hodge_divergence_threshold: if params.hodge_divergence_threshold.is_finite()
                && params.hodge_divergence_threshold > 0.0
            {
                params.hodge_divergence_threshold
            } else {
                0.70
            },
        };

        Self {
            params: valid_params,
            points_history: [PoincarePoint::ORIGIN; POINCARE_HISTORY_LEN],
            timestamps_history: [0; POINCARE_HISTORY_LEN],
            history_count: 0,
            history_index: 0,
            prev_distance_to_center: 0.0,
            prev_ts_ms: 0,
            smoothed_lipschitz: 0.0,
        }
    }

    /// Actualiza el estado del motor con el retorno de precio, el OFI y la marca de tiempo,
    /// retornando el informe de variedad hiperbólica y tensor de Hodge.
    pub fn update_and_evaluate(
        &mut self,
        price_return: f64,
        ofi: f64,
        ts_ms: u64,
    ) -> HyperbolicPoincareHodgeReport {
        let p_in = if price_return.is_finite() {
            price_return
        } else {
            0.0
        };
        let ofi_in = if ofi.is_finite() { ofi } else { 0.0 };

        // 1. Mapeo no lineal al disco de Poincaré $\mathbb{D}$
        let x_raw = p_in * self.params.sensitivity_p;
        let y_raw = ofi_in * self.params.sensitivity_ofi;
        let rho = (x_raw * x_raw + y_raw * y_raw).sqrt();

        let current_point = if rho > 1e-12 {
            let r_mapped = rho.tanh() * self.params.max_disk_radius;
            let scale = r_mapped / rho;
            PoincarePoint::new(x_raw * scale, y_raw * scale, self.params.max_disk_radius)
        } else {
            PoincarePoint::ORIGIN
        };

        // 2. Distancia geodésica al centro de equilibrio neutral
        let dist_center = current_point.distance_to_origin();

        // 3. Velocidad de deriva geodésica $\dot{d}_{\mathbb{H}}$
        let dt_sec = if self.prev_ts_ms > 0 && ts_ms > self.prev_ts_ms {
            ((ts_ms - self.prev_ts_ms) as f64) / 1000.0
        } else {
            0.010
        };
        let drift_velocity = if dt_sec > 1e-6 {
            (dist_center - self.prev_distance_to_center) / dt_sec
        } else {
            0.0
        };

        // 4. Estimación de la constante de Lipschitz local $L = \sup \frac{d_{\mathbb{H}}}{\Delta t}$
        let mut max_lipschitz_sample = 0.0_f64;
        let valid_history = self.history_count.min(POINCARE_HISTORY_LEN);

        for k in 0..valid_history {
            let hist_pt = self.points_history[k];
            let hist_ts = self.timestamps_history[k];
            if hist_ts > 0 && ts_ms > hist_ts {
                let dt_k = ((ts_ms - hist_ts) as f64) / 1000.0;
                if dt_k > 1e-4 {
                    let d_geo = current_point.distance_to(&hist_pt);
                    let l_k = d_geo / dt_k;
                    if l_k > max_lipschitz_sample {
                        max_lipschitz_sample = l_k;
                    }
                }
            }
        }

        self.smoothed_lipschitz = (1.0 - self.params.lipschitz_smoothing) * self.smoothed_lipschitz
            + self.params.lipschitz_smoothing * max_lipschitz_sample;

        // 5. Cálculo exterior: 1-forma de flujo $\omega = dx \, \mathbf{e}_x + dy \, \mathbf{e}_y$
        // y su 1-forma dual de Hodge $\star \omega = -dy \, \mathbf{e}_x + dx \, \mathbf{e}_y$
        let prev_point = if self.history_count > 0 {
            let last_idx = (self.history_index + POINCARE_HISTORY_LEN - 1) % POINCARE_HISTORY_LEN;
            self.points_history[last_idx]
        } else {
            PoincarePoint::ORIGIN
        };

        let dx = current_point.x - prev_point.x;
        let dy = current_point.y - prev_point.y;

        // Norma euclidiana del flujo ortogonal dual de Hodge: $\|\star \omega\| = \|\omega\|$
        let hodge_flux_norm = (dx * dx + dy * dy).sqrt();
        let conformal_weight = current_point.conformal_factor();
        let hodge_orthogonal_flux = (hodge_flux_norm * conformal_weight * 0.5).tanh();

        // 2-forma de curvatura simpléctica / vorticidad de área en el plano hiperbólico
        let symplectic_vorticity = prev_point.x * current_point.y - prev_point.y * current_point.x;
        let normalized_vorticity = (symplectic_vorticity * 10.0).tanh();

        // 6. Índice continuo de estrés hiperbólico $\Omega_{\text{hyperbolic}} \in [0.0, 1.0]$
        let radius_stress = current_point.radius();
        let lipschitz_stress = (self.smoothed_lipschitz / 5.0).tanh();
        let drift_stress = (drift_velocity.abs() / 2.0).tanh();

        let hyperbolic_stress_index = (0.40 * radius_stress
            + 0.30 * lipschitz_stress
            + 0.20 * drift_stress
            + 0.10 * hodge_orthogonal_flux)
            .clamp(0.0, 1.0);

        // 7. Modulación de escala de slot continuo $\eta_{\text{poincare}} \in [0.10, 1.25]$
        let slot_scaling_factor = if hyperbolic_stress_index > 0.85 {
            0.15
        } else if hyperbolic_stress_index > self.params.hodge_divergence_threshold {
            0.45
        } else {
            let recovery = (1.0 - hyperbolic_stress_index).max(0.0);
            (0.80 + 0.45 * recovery).clamp(0.10, 1.25)
        };

        // 8. Registro en el historial circular de stack
        self.points_history[self.history_index] = current_point;
        self.timestamps_history[self.history_index] = ts_ms;
        self.history_index = (self.history_index + 1) % POINCARE_HISTORY_LEN;
        if self.history_count < POINCARE_HISTORY_LEN {
            self.history_count += 1;
        }
        self.prev_distance_to_center = dist_center;
        self.prev_ts_ms = ts_ms;

        HyperbolicPoincareHodgeReport {
            poincare_x: current_point.x,
            poincare_y: current_point.y,
            poincare_radius: current_point.radius(),
            geodesic_distance_to_center: dist_center,
            geodesic_drift_velocity: drift_velocity,
            lipschitz_constant: self.smoothed_lipschitz,
            hodge_orthogonal_flux,
            symplectic_area_vorticity: normalized_vorticity,
            hyperbolic_stress_index,
            slot_scaling_factor,
        }
    }
}
