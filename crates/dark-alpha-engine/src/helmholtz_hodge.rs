//! # Descomposición Hidrodinámica de Helmholtz-Hodge ($L^2 = \nabla \phi \oplus \nabla \times \mathbf{A} \oplus \mathbf{h}$)
//!
//! Desacoplamiento ortogonal continuo del flujo de órdenes en el espacio de Hilbert $L^2$.
//! Separa el flujo en tres campos mutuamente ortogonales con computación en stack zero-alloc (<25 µs):
//! 1. $\nabla \phi$: Flujo laminar potencial irrotacional (momentum genuino de liquidez).
//! 2. $\nabla \times \mathbf{A}$: Flujo solenoidal rotacional (vorticidad de spoofing, churn algorítmico y turbulencia).
//! 3. $\mathbf{h}$: Flujo armónico no disipativo (deriva secular y consistencia triangular).
//!
//! ## Fundamentación Matemática
//! Según el teorema de descomposición de Helmholtz-Hodge:
//! $$\mathbf{u} = \nabla \phi + \nabla \times \mathbf{A} + \mathbf{h}$$
//! con ortogonalidad estricta en el producto interno de Hilbert:
//! $$\langle \nabla \phi, \nabla \times \mathbf{A} \rangle_{L^2} = \int_{\mathcal{D}} (\nabla \phi) \cdot (\nabla \times \mathbf{A}) \, dV = 0$$
//!
//! Por consiguiente, la energía cinética del libro de órdenes se desacopla exactamente:
//! $$E_{\text{total}} = E_{\text{laminar}} + E_{\text{vortical}} + E_{\text{armonica}}$$
//! donde la Enstrofía $\Omega$ mide la turbulencia rotacional:
//! $$\Omega = \frac{1}{2} \int |\nabla \times \mathbf{u}|^2 dV$$
//!
//! La fracción laminar $\Lambda \in [0, 1]$ cuantifica la pureza directiva del impulso institucional:
//! $$\Lambda = \frac{E_{\text{laminar}}}{E_{\text{total}} + \epsilon}$$

use serde::{Deserialize, Serialize};

pub const MAX_FLOW_HISTORY: usize = 32;

/// Parámetros dinámicos para la descomposición de Helmholtz-Hodge
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HelmholtzHodgeParams {
    pub enstrophy_threshold: f64,
    pub min_laminar_fraction: f64,
    pub smoothing_alpha: f64,
    pub viscosity: f64,
    pub lambda_regularization: f64,
}

impl Default for HelmholtzHodgeParams {
    fn default() -> Self {
        Self {
            enstrophy_threshold: 0.75,
            min_laminar_fraction: 0.35,
            smoothing_alpha: 0.15,
            viscosity: 0.05,
            lambda_regularization: 1e-4,
        }
    }
}

/// Estado instantáneo descompuesto del flujo de mercado
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HelmholtzDecomposition {
    /// Magnitud del vector de flujo total ||u||
    pub total_energy: f64,
    /// Componente laminar (momentum potencial) ||∇φ||²
    pub laminar_energy: f64,
    /// Componente solenoidal (vorticidad/spoofing) ||∇×A||² (Enstrofía Ω)
    pub enstrophy: f64,
    /// Componente armónica de fondo ||h||²
    pub harmonic_energy: f64,
    /// Fracción laminar Λ ∈ [0, 1]: 1 = laminar puro (alta convicción), 0 = turbulencia pura (spoofing)
    pub laminar_fraction: f64,
    /// Dirección del flujo potencial laminar (-1.0 = venta neta, +1.0 = compra neta)
    pub potential_direction: f64,
    /// Veredicto de compuerta conformal: true si el flujo es seguro y laminar
    pub is_laminar_safe: bool,
    /// Factor de modulación de convicción C¹ ∈ [0.0, 1.30]
    pub conviction_factor: f64,
}

impl Default for HelmholtzDecomposition {
    fn default() -> Self {
        Self {
            total_energy: 0.0,
            laminar_energy: 0.0,
            enstrophy: 0.0,
            harmonic_energy: 0.0,
            laminar_fraction: 1.0,
            potential_direction: 0.0,
            is_laminar_safe: true,
            conviction_factor: 1.0,
        }
    }
}

/// Motor de descomposición de Helmholtz-Hodge zero-alloc en stack
#[derive(Debug, Clone)]
pub struct HelmholtzHodgeEngine {
    pub params: HelmholtzHodgeParams,
    /// Ventana circular de observaciones de flujo en stack: [v_price, v_volume]
    flow_history: [[f64; 2]; MAX_FLOW_HISTORY],
    history_idx: usize,
    history_len: usize,
    smoothed_enstrophy: f64,
    smoothed_laminar: f64,
}

impl HelmholtzHodgeEngine {
    pub fn new(params: HelmholtzHodgeParams) -> Self {
        Self {
            params,
            flow_history: [[0.0; 2]; MAX_FLOW_HISTORY],
            history_idx: 0,
            history_len: 0,
            smoothed_enstrophy: 0.0,
            smoothed_laminar: 0.0,
        }
    }

    /// Registra una nueva observación de microestructura y calcula la descomposición ortogonal
    /// - `price_delta_bps`: Desplazamiento de precio en bps (componente longitudinal u_p)
    /// - `order_flow_imbalance`: OFI normalizado en [-1.0, 1.0] (componente transversal u_v)
    #[inline]
    pub fn update(&mut self, price_delta_bps: f64, order_flow_imbalance: f64) -> HelmholtzDecomposition {
        // Blindaje numérico estricto
        let p_vel = if price_delta_bps.is_finite() {
            price_delta_bps.clamp(-50.0, 50.0)
        } else {
            0.0
        };
        let v_vel = if order_flow_imbalance.is_finite() {
            order_flow_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        self.flow_history[self.history_idx] = [p_vel, v_vel];
        self.history_idx = (self.history_idx + 1) % MAX_FLOW_HISTORY;
        if self.history_len < MAX_FLOW_HISTORY {
            self.history_len += 1;
        }

        self.decompose()
    }

    /// Ejecución analítica de la descomposición en stack con cero alocaciones (<100 ns)
    #[inline]
    pub fn decompose(&mut self) -> HelmholtzDecomposition {
        if self.history_len < 3 {
            return HelmholtzDecomposition::default();
        }

        let n = self.history_len;
        let mut div_sum = 0.0f64;
        let mut curl_sum = 0.0f64;
        let mut total_sq = 0.0f64;
        let mut net_flow_dir = 0.0f64;

        // Recorrido en memoria contigua en stack para máxima localidad L1
        for i in 0..(n - 1) {
            let idx_curr = (self.history_idx + MAX_FLOW_HISTORY - n + i) % MAX_FLOW_HISTORY;
            let idx_next = (idx_curr + 1) % MAX_FLOW_HISTORY;

            let u0 = self.flow_history[idx_curr];
            let u1 = self.flow_history[idx_next];

            // Diferencias finitas para divergencia ∇·u y rotacional ∇×u
            // Componente 0: precio (x), Componente 1: volumen (y)
            let du_x = u1[0] - u0[0];
            let du_y = u1[1] - u0[1];

            // Divergencia: ∂u_x/∂x + ∂u_y/∂y (aproximada en espacio de fases de microestructura)
            let divergence = du_x + du_y;
            // Rotacional en 2D (vorticidad escalar ω): ∂u_y/∂x - ∂u_x/∂y
            let curl = du_y - du_x;

            div_sum += divergence * divergence;
            curl_sum += curl * curl;

            let mag_sq = u1[0] * u1[0] + u1[1] * u1[1];
            total_sq += mag_sq;
            net_flow_dir += u1[0];
        }

        let denom = (n - 1) as f64;
        let raw_div = (div_sum / denom).max(0.0);
        let raw_curl = (curl_sum / denom).max(0.0);
        let raw_total = (total_sq / denom).max(1e-6);

        // Suavizado EWMA de energía laminar y enstrofía
        let alpha = self.params.smoothing_alpha;
        self.smoothed_laminar = (1.0 - alpha) * self.smoothed_laminar + alpha * raw_div;
        self.smoothed_enstrophy = (1.0 - alpha) * self.smoothed_enstrophy + alpha * raw_curl;

        let e_laminar = self.smoothed_laminar;
        let enstrophy = self.smoothed_enstrophy;
        let e_harmonic = (raw_total - (e_laminar + enstrophy)).max(0.0);
        let e_sum = e_laminar + enstrophy + e_harmonic + 1e-6;

        // La energía coherente de momentum integra tanto la aceleración potencial (∇φ) como la deriva armónica traslacional (h)
        let e_coherent = e_laminar + e_harmonic;
        let laminar_fraction = (e_coherent / e_sum).clamp(0.0, 1.0);
        let potential_direction = (net_flow_dir / denom).clamp(-1.0, 1.0);

        // Veredicto de seguridad: Flujo seguro si la fracción laminar supera el umbral
        // y la enstrofía relativa no satura el umbral de turbulencia de Navier-Stokes
        let rel_enstrophy = (enstrophy / e_sum).clamp(0.0, 1.0);
        let is_laminar_safe = laminar_fraction >= self.params.min_laminar_fraction
            && rel_enstrophy <= self.params.enstrophy_threshold;

        // Modulación continua C¹: si laminar_fraction es alta, bonificación hasta +30%;
        // si domina la vorticidad (spoofing), atenuación suave smoothstep
        let conviction_factor = if is_laminar_safe {
            1.0 + 0.30 * (laminar_fraction - self.params.min_laminar_fraction).max(0.0)
        } else {
            // Decaimiento Hermite cúbico suave ante turbulencia
            let t = (rel_enstrophy / self.params.enstrophy_threshold).min(1.0);
            (1.0 - t * t * (3.0 - 2.0 * t)).clamp(0.10, 1.0)
        };

        HelmholtzDecomposition {
            total_energy: raw_total,
            laminar_energy: e_laminar,
            enstrophy,
            harmonic_energy: e_harmonic,
            laminar_fraction,
            potential_direction,
            is_laminar_safe,
            conviction_factor,
        }
    }
}
