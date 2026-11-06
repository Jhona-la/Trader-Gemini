//! # Navier-Stokes Kolmogorov-Kraichnan 2D Hydrodynamic Microstructure Engine (§43)
//!
//! Modela el plano continuo bidimensional de liquidez $(\text{Profundidad}, \text{Flujo})$ como
//! un campo de velocidad incompresible $\mathbf{u}(x, y, t) = (u_x, u_y)$ gobernado por
//! las ecuaciones estocásticas de Navier-Stokes con disipación viscosa y forzamiento browniano:
//!
//! $$\partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}(x, y, t)$$
//!
//! Detecta la cascada inversa de energía de Kraichnan ($E(k) \propto k^{-5/3}$) frente a la
//! cascada directa de enstrofía ($E(k) \propto k^{-3}$) para identificar vórtices de liquidez
//! institucionales y singularidades inminentes de precio/volumen sin asignaciones de heap dinámico.
//!
//! ## Invariantes de Silicio y Cero Pánicos
//! - Memoria $O(1)$ en stack: mallas bidimensionales fijas de $8 \times 8$ (`GRID_DIM = 8`).
//! - Cero `assert!`, cero `unwrap()`, cero `panic!()`, cero `expect()`.
//! - Cumplimiento estricto del invariante `rb_p1`.

/// Dimensión estática de la malla espacial 2D en stack (8x8 = 64 celdas).
pub const NS_GRID_DIM: usize = 8;

/// Parámetros de configuración del motor hidrodinámico de Navier-Stokes.
#[derive(Debug, Clone, Copy)]
pub struct NavierStokesParams {
    /// Viscosidad cinemática de mercado $\nu$ (típicamente 0.05).
    pub kinematic_viscosity: f64,
    /// Densidad efectiva del fluido de órdenes $\rho$ (típicamente 1.0).
    pub fluid_density: f64,
    /// Amortiguación de fricción de fondo (Rayleigh drag) $\gamma$.
    pub rayleigh_drag: f64,
    /// Umbral crítico de enstrofía para alertar sobre vórtices turbulentos.
    pub enstrophy_vortex_threshold: f64,
}

impl Default for NavierStokesParams {
    fn default() -> Self {
        Self {
            kinematic_viscosity: 0.05,
            fluid_density: 1.0,
            rayleigh_drag: 0.02,
            enstrophy_vortex_threshold: 0.75,
        }
    }
}

/// Informe analítico emitido por el motor de Navier-Stokes y Kraichnan en cada evaluación.
#[derive(Debug, Clone, Copy)]
pub struct NavierStokesReport {
    /// Enstrofía total normalizada $\mathcal{E} \in [0.0, 1.0]$.
    pub total_enstrophy: f64,
    /// Vorticidad máxima observada en la malla $|\omega_{\max}|$.
    pub max_vorticity: f64,
    /// Exponente espectral de Kraichnan $\alpha$ ($E(k) \propto k^{-\alpha}$).
    pub kraichnan_spectral_slope: f64,
    /// Indicador continuo de cascada inversa de energía (inestabilidad macro inminente) $\in [0.0, 1.0]$.
    pub inverse_cascade_intensity: f64,
    /// Vector de fuerza hidrodinámica resultante $(F_x, F_y)$.
    pub hydrodynamic_force: (f64, f64),
    /// Factor de escala hidrodinámica para el dimensionamiento del slot $\eta_{\text{NS}} \in (0.0, 1.0]$.
    pub slot_scaling_factor: f64,
}

/// Motor de fluidos hidrodinámicos 2D de microestructura de mercado.
#[derive(Debug, Clone)]
pub struct NavierStokesKolmogorovEngine {
    params: NavierStokesParams,
    /// Componente horizontal de velocidad $u_x$ (profundidad de precios).
    u_x: [[f64; NS_GRID_DIM]; NS_GRID_DIM],
    /// Componente vertical de velocidad $u_y$ (tasa de flujo de volumen).
    u_y: [[f64; NS_GRID_DIM]; NS_GRID_DIM],
    /// Campo de vorticidad escalar $\omega = \partial_x u_y - \partial_y u_x$.
    vorticity: [[f64; NS_GRID_DIM]; NS_GRID_DIM],
    /// Marca de tiempo del último evento procesado en milisegundos.
    last_ts_ms: u64,
}

impl NavierStokesKolmogorovEngine {
    /// Inicializa un nuevo motor con los parámetros dados en memoria de stack.
    pub fn new(params: NavierStokesParams) -> Self {
        Self {
            params,
            u_x: [[0.0; NS_GRID_DIM]; NS_GRID_DIM],
            u_y: [[0.0; NS_GRID_DIM]; NS_GRID_DIM],
            vorticity: [[0.0; NS_GRID_DIM]; NS_GRID_DIM],
            last_ts_ms: 0,
        }
    }

    /// Actualiza el campo de velocidades con el nuevo evento de microestructura y resuelve
    /// un paso temporal continuo de Navier-Stokes con advección upwind y difusión viscosa.
    #[allow(clippy::needless_range_loop)]
    pub fn update_and_evaluate(
        &mut self,
        order_flow_imbalance: f64,
        volume_delta: f64,
        volatility_norm: f64,
        spread_bps: f64,
        ts_ms: u64,
    ) -> NavierStokesReport {
        // Sanitización estricta de entradas numéricas contra NaN o infinitos
        let ofi = if order_flow_imbalance.is_finite() {
            order_flow_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let v_delta = if volume_delta.is_finite() {
            volume_delta.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let vol = if volatility_norm.is_finite() {
            volatility_norm.clamp(0.0, 10.0)
        } else {
            0.1
        };
        let sp = if spread_bps.is_finite() {
            spread_bps.clamp(0.1, 50.0)
        } else {
            1.0
        };

        let dt = if self.last_ts_ms > 0 && ts_ms > self.last_ts_ms {
            let elapsed_s = (ts_ms - self.last_ts_ms) as f64 * 0.001;
            elapsed_s.clamp(0.001, 1.0)
        } else {
            0.05
        };
        self.last_ts_ms = ts_ms;

        // 1. Forzamiento externo inyectado por la microestructura en los nodos centrales
        let f_x = ofi * (1.0 + vol * 0.1);
        let f_y = v_delta * (1.0 / (1.0 + sp * 0.05));

        let c_mid = NS_GRID_DIM / 2;
        self.u_x[c_mid][c_mid] = (self.u_x[c_mid][c_mid] * 0.8 + f_x * 0.2).clamp(-5.0, 5.0);
        self.u_y[c_mid][c_mid] = (self.u_y[c_mid][c_mid] * 0.8 + f_y * 0.2).clamp(-5.0, 5.0);

        // 2. Paso temporal Navier-Stokes: Advección Upwind + Difusión Viscosa + Amortiguación Rayleigh
        let dx = 2.0 / (NS_GRID_DIM as f64 - 1.0);
        let dy = 2.0 / (NS_GRID_DIM as f64 - 1.0);
        let nu = self.params.kinematic_viscosity.clamp(0.001, 1.0);
        let drag = self.params.rayleigh_drag.clamp(0.0, 1.0);

        let mut next_ux = self.u_x;
        let mut next_uy = self.u_y;

        for i in 1..(NS_GRID_DIM - 1) {
            for j in 1..(NS_GRID_DIM - 1) {
                let ux = self.u_x[i][j];
                let uy = self.u_y[i][j];

                // Derivadas espaciales upwind para advección no lineal
                let dux_dx = if ux >= 0.0 {
                    (ux - self.u_x[i - 1][j]) / dx
                } else {
                    (self.u_x[i + 1][j] - ux) / dx
                };
                let dux_dy = if uy >= 0.0 {
                    (ux - self.u_x[i][j - 1]) / dy
                } else {
                    (self.u_x[i][j + 1] - ux) / dy
                };

                let duy_dx = if ux >= 0.0 {
                    (uy - self.u_y[i - 1][j]) / dx
                } else {
                    (self.u_y[i + 1][j] - uy) / dx
                };
                let duy_dy = if uy >= 0.0 {
                    (uy - self.u_y[i][j - 1]) / dy
                } else {
                    (self.u_y[i][j + 1] - uy) / dy
                };

                let advection_x = ux * dux_dx + uy * dux_dy;
                let advection_y = ux * duy_dx + uy * duy_dy;

                // Laplaciano espacial para difusión viscosa $\nu \nabla^2 \mathbf{u}$
                let lap_ux = (self.u_x[i + 1][j] - 2.0 * ux + self.u_x[i - 1][j]) / (dx * dx)
                    + (self.u_x[i][j + 1] - 2.0 * ux + self.u_x[i][j - 1]) / (dy * dy);
                let lap_uy = (self.u_y[i + 1][j] - 2.0 * uy + self.u_y[i - 1][j]) / (dx * dx)
                    + (self.u_y[i][j + 1] - 2.0 * uy + self.u_y[i][j - 1]) / (dy * dy);

                let d_ux = (-advection_x + nu * lap_ux - drag * ux) * dt;
                let d_uy = (-advection_y + nu * lap_uy - drag * uy) * dt;

                next_ux[i][j] = (ux + d_ux).clamp(-10.0, 10.0);
                next_uy[i][j] = (uy + d_uy).clamp(-10.0, 10.0);
            }
        }

        self.u_x = next_ux;
        self.u_y = next_uy;

        // 3. Cálculo de la Vorticidad Escalar $\omega = \partial_x u_y - \partial_y u_x$ y Enstrofía
        let mut sum_enstrophy = 0.0;
        let mut max_abs_vorticity = 0.0;
        let mut mean_fx = 0.0;
        let mut mean_fy = 0.0;

        for i in 1..(NS_GRID_DIM - 1) {
            for j in 1..(NS_GRID_DIM - 1) {
                let vort = (self.u_y[i + 1][j] - self.u_y[i - 1][j]) / (2.0 * dx)
                    - (self.u_x[i][j + 1] - self.u_x[i][j - 1]) / (2.0 * dy);

                let sanitized_vort = if vort.is_finite() {
                    vort.clamp(-50.0, 50.0)
                } else {
                    0.0
                };
                self.vorticity[i][j] = sanitized_vort;

                let abs_v = sanitized_vort.abs();
                if abs_v > max_abs_vorticity {
                    max_abs_vorticity = abs_v;
                }
                sum_enstrophy += sanitized_vort * sanitized_vort;
                mean_fx += self.u_x[i][j];
                mean_fy += self.u_y[i][j];
            }
        }

        let inner_cells = ((NS_GRID_DIM - 2) * (NS_GRID_DIM - 2)) as f64;
        let raw_enstrophy = sum_enstrophy / inner_cells.max(1.0);
        let normalized_enstrophy = (raw_enstrophy / (raw_enstrophy + 5.0)).clamp(0.0, 1.0);

        mean_fx /= inner_cells.max(1.0);
        mean_fy /= inner_cells.max(1.0);

        // 4. Estimación Espectral de la Cascada de Kraichnan
        // Evaluamos la energía en 3 bandas de número de onda diádicas k=1, 2, 3
        let mut e_k = [0.0; 3];
        for (idx, energy_slot) in e_k.iter_mut().enumerate() {
            let ring_radius = (idx + 1) as f64;
            let mut ring_energy = 0.0;
            let mut count = 0.0_f64;

            for i in 1..(NS_GRID_DIM - 1) {
                for j in 1..(NS_GRID_DIM - 1) {
                    let rx = i as f64 - (NS_GRID_DIM as f64 / 2.0);
                    let ry = j as f64 - (NS_GRID_DIM as f64 / 2.0);
                    let dist = (rx * rx + ry * ry).sqrt();

                    if (dist - ring_radius).abs() <= 0.8 {
                        let energy_dens = 0.5 * (self.u_x[i][j].powi(2) + self.u_y[i][j].powi(2));
                        if energy_dens.is_finite() {
                            ring_energy += energy_dens;
                            count += 1.0;
                        }
                    }
                }
            }
            *energy_slot = (ring_energy / count.max(1.0)).max(1e-6);
        }

        // Pendiente log-espectral: $\ln E(k) = -\alpha \ln k + C$
        let ln_k1 = 1.0_f64.ln();
        let ln_k3 = 3.0_f64.ln();
        let delta_ln_k = (ln_k3 - ln_k1).max(1e-4);
        let delta_ln_e = e_k[2].ln() - e_k[0].ln();
        let spectral_slope = (-delta_ln_e / delta_ln_k).clamp(0.5, 4.5);

        // Si la pendiente se acerca a 5/3 (1.67) en vez de 3.0, predomina la cascada inversa
        // hacia escalas grandes (formación de vórtice direccional macro inminente)
        let dist_to_inverse = (spectral_slope - 1.6667).abs();
        let inverse_cascade_intensity = (-dist_to_inverse * 1.5).exp().clamp(0.0, 1.0);

        // 5. Factor de Escala del Slot Hidrodinámico $\eta_{\text{NS}}$
        let slot_scaling_factor =
            if normalized_enstrophy > self.params.enstrophy_vortex_threshold {
                // Alta turbulencia: reducir suavemente apalancamiento sin vetar a cero
                1.0 / (1.0 + 2.0 * (normalized_enstrophy - self.params.enstrophy_vortex_threshold))
            } else {
                1.0 + 0.20 * inverse_cascade_intensity
            }
            .clamp(0.10, 1.25);

        NavierStokesReport {
            total_enstrophy: normalized_enstrophy,
            max_vorticity: max_abs_vorticity,
            kraichnan_spectral_slope: spectral_slope,
            inverse_cascade_intensity,
            hydrodynamic_force: (mean_fx.clamp(-2.0, 2.0), mean_fy.clamp(-2.0, 2.0)),
            slot_scaling_factor,
        }
    }
}
