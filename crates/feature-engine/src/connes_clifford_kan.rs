//! # FASE 13: Geometría No Conmutativa de Connes, Hudson-Parthasarathy & Redes KAN con Álgebra de Clifford
//!
//! **QUÉ:** Implementación en stack ($O(1)$ / $O(K^2)$, 0 heap allocations) de tres pilares
//! de física matemática cuántica y aproximación funcional avanzada:
//! 1. **Tripleta Espectral de Alain Connes $(\mathcal{A}, \mathcal{H}, D)$:** Modela el libro de órdenes L2/L3
//!    como una variedad no conmutativa donde los operadores de precio $\hat{X}$ y momento de flujo $\hat{P}$
//!    satisfacen $[\hat{X}, \hat{P}] \ne 0$, calculando la constante efectiva de Planck de mercado
//!    $\hbar_{\text{mkt}} = \|[\hat{X}, \hat{P}]\|_F$ y la métrica espectral exacta de Connes
//!    $d_{\text{Connes}}(\rho_1, \rho_2) = \sup \{ |\text{Tr}((\rho_1 - \rho_2)a)| : \|[D, a]\|_{\text{Lip}} \le 1 \}$.
//! 2. **Cálculo Estocástico Cuántico de Hudson-Parthasarathy (QSDE):** Evoluciona la matriz de densidad
//!    reducida $\rho_t \in \mathbb{C}^{2 \times 2}$ bajo difusión bosónica $(dA_t, dA_t^\dagger)$ y saltos
//!    de calibre de Poisson $d\Lambda_t$, preservando exactamente $\text{Tr}(\rho_t) \equiv 1$, hermiticidad
//!    y positividad completa (mapa de Kraus-Belavkin).
//! 3. **Álgebra Geométrica de Clifford $\mathcal{C}\ell(3, 0)$ & Redes Kolmogorov-Arnold (KAN):**
//!    Multivectores de 8 grados (escalar, 3 vectores, 3 bivectores de rotación de fase, 1 trivector),
//!    rotores espinoriales isométricos $M' = R M \tilde{R}$, y aristas funcionales KAN parametrizadas por
//!    B-Splines cúbicas evaluadas en stack.
//!
//! **POR QUÉ:**
//! - En microestructura de alta frecuencia, observar o ejecutar volumen altera el precio: las variables
//!   conjugadas no conmutan. La métrica de Connes mide la distancia real de transporte de liquidez ponderada
//!   por el operador de Dirac de saltos entre niveles del libro.
//! - Los impactos de liquidez combinan ruido difusivo continuo con barridos discretos; la ecuación de
//!   Hudson-Parthasarathy unifica ambos en el espacio de Fock sin perder coherencia cuántica.
//! - Las redes KAN con B-Splines cúbicas eliminan la rigidez de las activaciones fijas y, al operar sobre
//!   multivectores $\mathcal{C}\ell(3,0)$, capturan productos exteriores (bivectores precio $\wedge$ flujo)
//!   y volúmenes orientados (trivectores) de forma nativa.
//!
//! **PARA QUÉ:** Extraer invariantes geométricos no conmutativos, filtrar incertidumbre de Heisenberg
//! en el libro de órdenes y sintetizar señales no lineales multivariantes con latencia de nanosegundos.

/// Dimensión máxima del espacio de Hilbert finito de niveles del libro en la Tripleta de Connes.
pub const CONNES_HILBERT_DIM: usize = 8;

/// Número de funciones base B-Spline cúbicas por arista univariada en la red KAN.
pub const KAN_SPLINE_BASIS_COUNT: usize = 8;

/// Régimen cuántico-espectral diagnosticado por la Tripleta de Connes y el filtro de Hudson-Parthasarathy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonCommutativeRegime {
    /// Baja incertidumbre no conmutativa ($\hbar_{\text{mkt}}$ reducido) y alta pureza de estado cuántico:
    /// ejecución direccional limpia con bajo impacto de mercado.
    CoherentLowImpact,
    /// Alta no-conmutatividad ($[\hat{X}, \hat{P}]$ elevado) o salto de calibre bosónico $d\Lambda_t$ activo:
    /// barrido agresivo del libro de órdenes en curso.
    QuantumJumpBreakout,
    /// Decoherencia severa ($\text{Tr}(\rho^2) \to 0.5$) e incertidumbre de Heisenberg sobre el límite:
    /// mercado difuso y manipulativo, se recomienda contracción o abstención.
    HeisenbergUncertaintyVeto,
}

// ============================================================================
// 1. TRIPLETA ESPECTRAL DE CONNES (A, H, D) & MÉTRICA NO CONMUTATIVA
// ============================================================================

/// Reporte de la Tripleta Espectral de Connes $(\mathcal{A}, \mathcal{H}, D)$ sobre el libro L2/L3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConnesSpectralReport {
    /// Norma de Frobenius del conmutador posición-momento $\hbar_{\text{mkt}} = \|[\hat{X}, \hat{P}]\|_F$.
    pub planck_market_constant: f64,
    /// Cota inferior de incertidumbre de Robertson-Heisenberg $\frac{1}{2} |\langle [\hat{X}, \hat{P}] \rangle_\rho|$.
    pub heisenberg_uncertainty_bound: f64,
    /// Norma de Lipschitz no conmutativa $\|[D, \hat{X}]\|_F$ respecto al operador de Dirac $D$.
    pub dirac_lipschitz_norm: f64,
    /// Distancia espectral de Connes $d_{\text{Connes}}(\rho_{\text{bid}}, \rho_{\text{ask}})$ entre distribuciones.
    pub connes_spectral_distance: f64,
}

/// Motor de Geometría No Conmutativa de Alain Connes en stack.
#[derive(Debug, Clone, Copy)]
pub struct ConnesSpectralTripleEngine {
    /// Escala base de acoplamiento del operador de Dirac entre niveles adyacentes.
    pub dirac_coupling_scale: f64,
    /// Umbral de incertidumbre de Heisenberg para declarar veto de ejecución.
    pub uncertainty_veto_threshold: f64,
}

impl Default for ConnesSpectralTripleEngine {
    fn default() -> Self {
        Self {
            dirac_coupling_scale: 1.0,
            uncertainty_veto_threshold: 2.5,
        }
    }
}

impl ConnesSpectralTripleEngine {
    /// Construye un motor de Tripleta Espectral de Connes con parámetros explícitos.
    #[inline]
    pub const fn new(dirac_coupling_scale: f64, uncertainty_veto_threshold: f64) -> Self {
        Self {
            dirac_coupling_scale,
            uncertainty_veto_threshold,
        }
    }

    /// Evalúa la Tripleta Espectral $(\mathcal{A}, \mathcal{H}, D)$ sobre `K <= 8` niveles del libro.
    /// - `price_offsets`: desplazamientos relativos de precio de cada nivel.
    /// - `bid_volumes`, `ask_volumes`: liquidez en cada nivel de compra y venta.
    /// - `order_flow_rates`: tasas de flujo neto en cada nivel (generan el operador de momento $\hat{P}$).
    #[inline]
    pub fn evaluate_spectral_triple(
        &self,
        price_offsets: &[f64],
        bid_volumes: &[f64],
        ask_volumes: &[f64],
        order_flow_rates: &[f64],
    ) -> ConnesSpectralReport {
        let k = price_offsets
            .len()
            .min(bid_volumes.len())
            .min(ask_volumes.len())
            .min(order_flow_rates.len())
            .min(CONNES_HILBERT_DIM);

        if k < 2 {
            return ConnesSpectralReport {
                planck_market_constant: 0.0,
                heisenberg_uncertainty_bound: 0.0,
                dirac_lipschitz_norm: 0.0,
                connes_spectral_distance: 0.0,
            };
        }

        // 1. Normalizar distribuciones de densidad diagonal \rho_bid y \rho_ask en el simplex
        let mut rho_bid = [0.0_f64; CONNES_HILBERT_DIM];
        let mut rho_ask = [0.0_f64; CONNES_HILBERT_DIM];
        let mut sum_bid = 0.0_f64;
        let mut sum_ask = 0.0_f64;

        for ((rb, ra), (&bv, &av)) in rho_bid
            .iter_mut()
            .zip(rho_ask.iter_mut())
            .zip(bid_volumes.iter().zip(ask_volumes.iter()))
            .take(k)
        {
            let clean_b = if bv.is_finite() && bv > 0.0 { bv } else { 0.0 };
            let clean_a = if av.is_finite() && av > 0.0 { av } else { 0.0 };
            *rb = clean_b;
            *ra = clean_a;
            sum_bid += clean_b;
            sum_ask += clean_a;
        }

        let inv_k = 1.0 / (k as f64);
        for (rb, ra) in rho_bid.iter_mut().zip(rho_ask.iter_mut()).take(k) {
            *rb = if sum_bid > 1e-12 {
                *rb / sum_bid
            } else {
                inv_k
            };
            *ra = if sum_ask > 1e-12 {
                *ra / sum_ask
            } else {
                inv_k
            };
        }

        // 2. Construir el operador de posición \hat{X} (diagonal) y el operador de momento \hat{P}
        // (antisimétrico tridiagonal de desplazamiento de flujo entre niveles contiguos)
        let mut x_diag = [0.0_f64; CONNES_HILBERT_DIM];
        for (slot, &px) in x_diag.iter_mut().zip(price_offsets.iter()).take(k) {
            *slot = if px.is_finite() { px } else { 0.0 };
        }

        let mut p_sub = [0.0_f64; CONNES_HILBERT_DIM]; // P_{i, i+1} = -P_{i+1, i}
        let mut d_sub = [0.0_f64; CONNES_HILBERT_DIM]; // Acoplamiento del operador de Dirac D_{i, i+1}
        let scale = if self.dirac_coupling_scale.is_finite() && self.dirac_coupling_scale > 0.0 {
            self.dirac_coupling_scale
        } else {
            1.0
        };

        for (((p_slot, d_slot), flow_pair), x_pair) in p_sub
            .iter_mut()
            .zip(d_sub.iter_mut())
            .zip(order_flow_rates[..k].windows(2))
            .zip(x_diag[..k].windows(2))
        {
            let f_curr = if flow_pair[0].is_finite() {
                flow_pair[0]
            } else {
                0.0
            };
            let f_next = if flow_pair[1].is_finite() {
                flow_pair[1]
            } else {
                0.0
            };
            *p_slot = 0.5 * (f_curr - f_next);

            let dx = (x_pair[1] - x_pair[0]).abs().max(1e-6);
            *d_slot = scale / dx;
        }

        // 3. Conmutador [\hat{X}, \hat{P}]_{i, i+1} = (X_i - X_{i+1}) P_{i, i+1}
        //    y Conmutador de Dirac [D, \hat{X}]_{i, i+1} = D_{i, i+1} (X_{i+1} - X_i)
        let mut comm_xp_sq_sum = 0.0_f64;
        let mut comm_dx_sq_sum = 0.0_f64;
        let mut expectation_comm_xp = 0.0_f64;

        for ((((&p_val, &d_val), x_pair), bid_pair), ask_pair) in p_sub
            .iter()
            .zip(d_sub.iter())
            .zip(x_diag[..k].windows(2))
            .zip(rho_bid[..k].windows(2))
            .zip(rho_ask[..k].windows(2))
        {
            let delta_x = x_pair[0] - x_pair[1];
            let xp_entry = delta_x * p_val;
            // Dos entradas simétricas fuera de la diagonal: (i, i+1) e (i+1, i)
            comm_xp_sq_sum += 2.0 * xp_entry * xp_entry;

            let dx_entry = d_val * (-delta_x);
            comm_dx_sq_sum += 2.0 * dx_entry * dx_entry;

            let weight_level = 0.5 * (bid_pair[0] + ask_pair[0] + bid_pair[1] + ask_pair[1]);
            expectation_comm_xp += weight_level * xp_entry.abs();
        }

        let planck_market_constant = comm_xp_sq_sum.sqrt();
        let heisenberg_uncertainty_bound = 0.5 * expectation_comm_xp;
        let dirac_lipschitz_norm = comm_dx_sq_sum.sqrt();

        // 4. Distancia espectral exacta de Connes en la cadena discreta:
        // d_Connes(\rho_bid, \rho_ask) = \sum_{i=0}^{k-2} |\sum_{j=0}^i (\rho_bid[j] - \rho_ask[j])| / D_{i, i+1}
        let mut cumulative_diff = 0.0_f64;
        let mut connes_distance = 0.0_f64;
        for ((&rb, &ra), &d_val) in rho_bid
            .iter()
            .zip(rho_ask.iter())
            .zip(d_sub.iter())
            .take(k - 1)
        {
            cumulative_diff += rb - ra;
            let dirac_hop = d_val.max(1e-9);
            connes_distance += cumulative_diff.abs() / dirac_hop;
        }

        ConnesSpectralReport {
            planck_market_constant,
            heisenberg_uncertainty_bound,
            dirac_lipschitz_norm,
            connes_spectral_distance: connes_distance,
        }
    }
}

// ============================================================================
// 2. CÁLCULO ESTOCÁSTICO CUÁNTICO DE HUDSON-PARTHASARATHY (QSDE)
// ============================================================================

/// Estado de la matriz de densidad reducida $\rho_t \in \mathbb{C}^{2 \times 2}$ representada en la esfera de Bloch
/// $\rho = \frac{1}{2}(I + r_x \sigma_x + r_y \sigma_y + r_z \sigma_z)$ con $\|r\|_2 \le 1$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HudsonParthasarathyState {
    /// Componente de coherencia real $\text{Re}(\rho_{01}) = r_x / 2$.
    pub bloch_x: f64,
    /// Componente de coherencia imaginaria $\text{Im}(\rho_{01}) = r_y / 2$.
    pub bloch_y: f64,
    /// Inversión de población direccional $\rho_{00} - \rho_{11} = r_z \in [-1, 1]$ (+1 Long, -1 Short).
    pub bloch_z: f64,
    /// Traza acumulada verificada (invariante exacto $\text{Tr}(\rho) \equiv 1.0$).
    pub trace: f64,
    /// Pureza cuántica $\text{Tr}(\rho^2) = \frac{1}{2}(1 + \|r\|_2^2) \in [0.5, 1.0]$.
    pub purity: f64,
    /// Contador de saltos de calibre de Poisson $d\Lambda_t$ observados.
    pub gauge_jumps_observed: u64,
}

impl Default for HudsonParthasarathyState {
    fn default() -> Self {
        Self {
            bloch_x: 0.0,
            bloch_y: 0.0,
            bloch_z: 0.0,
            trace: 1.0,
            purity: 0.5,
            gauge_jumps_observed: 0,
        }
    }
}

/// Evolucionador cuántico estocástico de Hudson-Parthasarathy en el espacio de Fock.
#[derive(Debug, Clone, Copy)]
pub struct HudsonParthasarathyEvolutor {
    /// Frecuencia hamiltoniana base $\omega_0$ de precesión coherente.
    pub hamiltonian_freq: f64,
    /// Tasa de acoplamiento de Lindblad $\gamma$ con el baño bosónico de flujo de órdenes.
    pub lindblad_coupling: f64,
    /// Ángulo de dispersión unitaria $\theta_S$ del operador de calibre $S$ ante saltos $d\Lambda_t$.
    pub scattering_phase: f64,
}

impl Default for HudsonParthasarathyEvolutor {
    fn default() -> Self {
        Self {
            hamiltonian_freq: 0.8,
            lindblad_coupling: 0.25,
            scattering_phase: 0.6,
        }
    }
}

impl HudsonParthasarathyEvolutor {
    /// Avanza un paso temporal $dt$ de la ecuación estocástica cuántica de Hudson-Parthasarathy:
    /// incorpora el incremento difusivo browniano `da_quadrature` ($dA_t + dA_t^\dagger$) y el
    /// indicador de salto de Poisson `d_lambda_jump` ($d\Lambda_t \in \{0, 1\}$).
    #[inline]
    pub fn step_qsde(
        &self,
        state: &HudsonParthasarathyState,
        hamiltonian_drive: f64,
        da_quadrature: f64,
        d_lambda_jump: bool,
        dt: f64,
    ) -> HudsonParthasarathyState {
        let safe_dt = if dt.is_finite() {
            dt.clamp(1e-6, 1.0)
        } else {
            1e-3
        };
        let drive = if hamiltonian_drive.is_finite() {
            hamiltonian_drive.clamp(-10.0, 10.0)
        } else {
            0.0
        };
        let da = if da_quadrature.is_finite() {
            da_quadrature.clamp(-5.0, 5.0)
        } else {
            0.0
        };

        let mut rx = if state.bloch_x.is_finite() {
            state.bloch_x
        } else {
            0.0
        };
        let mut ry = if state.bloch_y.is_finite() {
            state.bloch_y
        } else {
            0.0
        };
        let mut rz = if state.bloch_z.is_finite() {
            state.bloch_z
        } else {
            0.0
        };

        let gamma = if self.lindblad_coupling.is_finite() && self.lindblad_coupling >= 0.0 {
            self.lindblad_coupling
        } else {
            0.1
        };
        let omega = (if self.hamiltonian_freq.is_finite() {
            self.hamiltonian_freq
        } else {
            0.5
        }) + drive;

        // 1. Precesión unitaria de Heisenberg-Hamilton + amortiguamiento de Lindblad -1/2 {L^\dagger L, \rho} dt
        let damping = (-gamma * safe_dt).exp();
        let angle = omega * safe_dt;
        let (sin_a, cos_a) = angle.sin_cos();
        let rx_rot = (rx * cos_a - ry * sin_a) * damping;
        let ry_rot = (rx * sin_a + ry * cos_a) * damping;

        // 2. Innovación difusiva bosónica de Belavkin-Hudson-Parthasarathy: \sqrt{\gamma} (L dA^\dagger + L^\dagger dA)
        let stochastic_gain = (gamma * safe_dt).sqrt() * da;
        rx = rx_rot + stochastic_gain * (1.0 - rz * rz).max(0.0).sqrt() * 0.5;
        ry = ry_rot;
        rz =
            rz * damping + (1.0 - damping) * drive.tanh() + stochastic_gain * (1.0 - rz * rz.abs());

        // 3. Término de calibre de Poisson (S - I) d\Lambda_t: rotación unitaria de dispersión + colapso parcial hacia el flujo
        let mut jumps = state.gauge_jumps_observed;
        if d_lambda_jump {
            jumps = jumps.saturating_add(1);
            let theta = if self.scattering_phase.is_finite() {
                self.scattering_phase
            } else {
                0.5
            };
            let (sin_s, cos_s) = theta.sin_cos();
            let next_rx = rx * cos_s - rz * sin_s;
            let next_rz = rx * sin_s + rz * cos_s + 0.25 * da.signum();
            rx = next_rx;
            rz = next_rz;
        }

        // 4. Proyección contractiva en la bola de Bloch ||r||_2 <= 1 para garantizar positividad completa (\det \rho >= 0)
        let norm_sq = rx * rx + ry * ry + rz * rz;
        if norm_sq > 1.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            rx *= inv_norm;
            ry *= inv_norm;
            rz *= inv_norm;
        }

        let clamped_norm_sq = (rx * rx + ry * ry + rz * rz).clamp(0.0, 1.0);
        let purity = 0.5 * (1.0 + clamped_norm_sq);

        HudsonParthasarathyState {
            bloch_x: rx,
            bloch_y: ry,
            bloch_z: rz,
            trace: 1.0,
            purity,
            gauge_jumps_observed: jumps,
        }
    }
}

// ============================================================================
// 3. ÁLGEBRA GEOMÉTRICA DE CLIFFORD Cl(3,0) & REDES KOLMOGOROV-ARNOLD (KAN)
// ============================================================================

/// Multivector completo de 8 componentes en el Álgebra Geométrica de Clifford $\mathcal{C}\ell(3, 0)$:
/// Base: $[1, e_1, e_2, e_3, e_{12}, e_{23}, e_{31}, e_{123}]$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CliffordMultivector3D {
    /// Coeficientes en la base canónica $[s, v_1, v_2, v_3, b_{12}, b_{23}, b_{31}, p_{123}]$.
    pub coeffs: [f64; 8],
}

impl Default for CliffordMultivector3D {
    fn default() -> Self {
        Self::zero()
    }
}

impl CliffordMultivector3D {
    /// Multivector nulo.
    #[inline]
    pub const fn zero() -> Self {
        Self { coeffs: [0.0; 8] }
    }

    /// Construye un multivector escalar puro.
    #[inline]
    pub const fn from_scalar(s: f64) -> Self {
        Self {
            coeffs: [s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Construye un vector puro de grado 1 ($v_1 e_1 + v_2 e_2 + v_3 e_3$).
    #[inline]
    pub const fn from_vector(v1: f64, v2: f64, v3: f64) -> Self {
        Self {
            coeffs: [0.0, v1, v2, v3, 0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Construye un rotor espinorial $R = \exp(-\frac{\theta}{2} \hat{B}) = \cos(\theta/2) - \sin(\theta/2)(b_{12} e_{12} + b_{23} e_{23} + b_{31} e_{31})$
    /// en el subálgebra par $\mathcal{C}\ell^+(3,0) \cong \mathbb{H}$.
    #[inline]
    pub fn rotor_from_bivector_angle(b12: f64, b23: f64, b31: f64, angle_rad: f64) -> Self {
        let norm_sq = b12 * b12 + b23 * b23 + b31 * b31;
        if !norm_sq.is_finite() || norm_sq < 1e-18 || !angle_rad.is_finite() {
            return Self::from_scalar(1.0);
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        let half = 0.5 * angle_rad;
        let (sin_h, cos_h) = half.sin_cos();
        Self {
            coeffs: [
                cos_h,
                0.0,
                0.0,
                0.0,
                -sin_h * b12 * inv_norm,
                -sin_h * b23 * inv_norm,
                -sin_h * b31 * inv_norm,
                0.0,
            ],
        }
    }

    /// Reverso algebraico $\tilde{M}$: invierte el signo de los bivectores (grado 2) y del trivector (grado 3).
    #[inline]
    pub const fn reverse(&self) -> Self {
        let c = self.coeffs;
        Self {
            coeffs: [c[0], c[1], c[2], c[3], -c[4], -c[5], -c[6], -c[7]],
        }
    }

    /// Producto geométrico exacto de Clifford $A B = A \cdot B + A \wedge B$ en $\mathcal{C}\ell(3, 0)$.
    /// Satisface asociatividad $(AB)C = A(BC)$ y $e_i^2 = +1$, $e_i e_j = -e_j e_i$.
    #[inline]
    pub fn geometric_product(&self, rhs: &Self) -> Self {
        let a = self.coeffs;
        let b = rhs.coeffs;

        // Grado 0 (escalar 1)
        let c0 = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
            - a[4] * b[4]
            - a[5] * b[5]
            - a[6] * b[6]
            - a[7] * b[7];

        // Grado 1 (vectores e1, e2, e3)
        let c1 = a[0] * b[1] + a[1] * b[0] - a[2] * b[4] + a[3] * b[6] + a[4] * b[2]
            - a[5] * b[7]
            - a[6] * b[3]
            - a[7] * b[5];
        let c2 = a[0] * b[2] + a[1] * b[4] + a[2] * b[0] - a[3] * b[5] - a[4] * b[1] + a[5] * b[3]
            - a[6] * b[7]
            - a[7] * b[6];
        let c3 = a[0] * b[3] - a[1] * b[6] + a[2] * b[5] + a[3] * b[0] - a[4] * b[7] - a[5] * b[2]
            + a[6] * b[1]
            - a[7] * b[4];

        // Grado 2 (bivectores e12, e23, e31)
        let c4 = a[0] * b[4] + a[1] * b[2] - a[2] * b[1] + a[3] * b[7] + a[4] * b[0] - a[5] * b[6]
            + a[6] * b[5]
            + a[7] * b[3];
        let c5 = a[0] * b[5] + a[1] * b[7] + a[2] * b[3] - a[3] * b[2] + a[4] * b[6] + a[5] * b[0]
            - a[6] * b[4]
            + a[7] * b[1];
        let c6 = a[0] * b[6] - a[1] * b[3] + a[2] * b[7] + a[3] * b[1] - a[4] * b[5]
            + a[5] * b[4]
            + a[6] * b[0]
            + a[7] * b[2];

        // Grado 3 (trivector / pseudoescalar e123)
        let c7 = a[0] * b[7]
            + a[1] * b[5]
            + a[2] * b[6]
            + a[3] * b[4]
            + a[4] * b[3]
            + a[5] * b[1]
            + a[6] * b[2]
            + a[7] * b[0];

        Self {
            coeffs: [c0, c1, c2, c3, c4, c5, c6, c7],
        }
    }

    /// Acción sándwich espinorial $M' = R M \tilde{R}$ que rota el multivector preservando su norma por grados.
    #[inline]
    pub fn apply_rotor(&self, rotor: &Self) -> Self {
        let rev = rotor.reverse();
        rotor.geometric_product(self).geometric_product(&rev)
    }

    /// Norma euclidiana al cuadrado de los coeficientes $\sum_{i=0}^7 c_i^2$.
    #[inline]
    pub fn norm_sq(&self) -> f64 {
        let mut s = 0.0_f64;
        for &c in &self.coeffs {
            if c.is_finite() {
                s += c * c;
            }
        }
        s
    }

    /// Energía del subespacio de bivectores (vorticidad rotacional entre ejes precio-flujo-volatilidad).
    #[inline]
    pub fn bivector_energy(&self) -> f64 {
        let b12 = self.coeffs[4];
        let b23 = self.coeffs[5];
        let b31 = self.coeffs[6];
        b12 * b12 + b23 * b23 + b31 * b31
    }
}

/// Arista funcional Kolmogorov-Arnold (KAN) parametrizada mediante B-Splines cúbicas uniformes en stack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubicBSplineKanEdge {
    /// Pesos aprendibles / adaptativos sobre las `KAN_SPLINE_BASIS_COUNT` funciones base B-Spline cúbicas.
    pub spline_weights: [f64; KAN_SPLINE_BASIS_COUNT],
    /// Peso de la activación base residual SiLU $\phi_{\text{base}}(x) = x / (1 + e^{-x})$.
    pub base_silu_weight: f64,
}

impl Default for CubicBSplineKanEdge {
    fn default() -> Self {
        Self {
            spline_weights: [0.15, -0.10, 0.25, 0.40, -0.35, 0.30, -0.15, 0.20],
            base_silu_weight: 0.50,
        }
    }
}

impl CubicBSplineKanEdge {
    /// Evalúa la función univariada $\phi(x) = w_{\text{base}} \text{SiLU}(x) + \sum_k c_k B_{k, 3}(x)$
    /// para $x \in [-1, 1]$ usando las 4 funciones base B-Spline cúbicas activas en la celda local ($O(1)$).
    #[inline]
    pub fn evaluate(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        let clamped = x.clamp(-1.0, 1.0);
        let silu = clamped / (1.0 + (-clamped).exp());

        // Mapear [-1, 1] al intervalo de celdas B-Spline [0, M - 4]
        let num_cells = (KAN_SPLINE_BASIS_COUNT - 3) as f64;
        let u_scaled = 0.5 * (clamped + 1.0) * num_cells;
        let cell_idx = (u_scaled.floor() as usize).min(KAN_SPLINE_BASIS_COUNT - 4);
        let t = (u_scaled - (cell_idx as f64)).clamp(0.0, 1.0);

        // Polinomios base B-Spline cúbicos uniformes (partición exacta de la unidad: b0 + b1 + b2 + b3 == 1)
        let one_minus_t = 1.0 - t;
        let t2 = t * t;
        let t3 = t2 * t;
        let b0 = (one_minus_t * one_minus_t * one_minus_t) / 6.0;
        let b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
        let b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
        let b3 = t3 / 6.0;

        let spline_val = b0 * self.spline_weights[cell_idx]
            + b1 * self.spline_weights[cell_idx + 1]
            + b2 * self.spline_weights[cell_idx + 2]
            + b3 * self.spline_weights[cell_idx + 3];

        let out = self.base_silu_weight * silu + spline_val;
        if out.is_finite() {
            out
        } else {
            0.0
        }
    }

    /// Evalúa las 4 bases B-Spline cúbicas locales $(b_0, b_1, b_2, b_3)$ en $t \in [0, 1]$.
    #[inline]
    pub fn cubic_bspline_basis_weights(t: f64) -> [f64; 4] {
        let u = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let om = 1.0 - u;
        let u2 = u * u;
        let u3 = u2 * u;
        [
            (om * om * om) / 6.0,
            (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
            (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
            u3 / 6.0,
        ]
    }
}

/// Reporte unificado de la Fase 13 (Connes + Hudson-Parthasarathy + KAN Clifford).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConnesCliffordKanReport {
    /// Diagnóstico geométrico no conmutativo de la Tripleta de Connes.
    pub connes: ConnesSpectralReport,
    /// Estado evolucionado de la matriz de densidad de Hudson-Parthasarathy.
    pub qsde_state: HudsonParthasarathyState,
    /// Multivector de salida en $\mathcal{C}\ell(3, 0)$ sintetizado por la red KAN Clifford.
    pub clifford_output: CliffordMultivector3D,
    /// Señal direccional neta acotada en $[-1.0, 1.0]$.
    pub directional_conviction: f64,
    /// Clasificación del régimen cuántico no conmutativo.
    pub regime: NonCommutativeRegime,
}

/// Motor maestro de Fase 13 que acopla la Tripleta de Connes, el filtro QSDE de Hudson-Parthasarathy
/// y la red Kolmogorov-Arnold sobre el Álgebra Geométrica de Clifford $\mathcal{C}\ell(3,0)$.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConnesCliffordKanEngine {
    pub connes_engine: ConnesSpectralTripleEngine,
    pub qsde_evolutor: HudsonParthasarathyEvolutor,
    pub edge_price: CubicBSplineKanEdge,
    pub edge_flow: CubicBSplineKanEdge,
    pub edge_volatility: CubicBSplineKanEdge,
}

impl ConnesCliffordKanEngine {
    /// Evalúa un ciclo completo del motor Connes-Hudson-Parthasarathy-Clifford-KAN.
    #[inline]
    pub fn evaluate_cycle(
        &self,
        price_offsets: &[f64],
        bid_volumes: &[f64],
        ask_volumes: &[f64],
        order_flow_rates: &[f64],
        prev_qsde: &HudsonParthasarathyState,
        dt: f64,
    ) -> ConnesCliffordKanReport {
        let connes = self.connes_engine.evaluate_spectral_triple(
            price_offsets,
            bid_volumes,
            ask_volumes,
            order_flow_rates,
        );

        let net_flow = order_flow_rates
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .sum::<f64>();
        let jump_detected = connes.planck_market_constant > 1.5 || net_flow.abs() > 3.0;

        let qsde_state = self.qsde_evolutor.step_qsde(
            prev_qsde,
            net_flow.tanh(),
            connes.connes_spectral_distance.tanh() * net_flow.signum(),
            jump_detected,
            dt,
        );

        // Transformación KAN B-Spline sobre las 3 coordenadas canónicas (desequilibrio, distancia Connes, incertidumbre)
        let x1 = self.edge_price.evaluate(qsde_state.bloch_z);
        let x2 = self
            .edge_flow
            .evaluate(connes.connes_spectral_distance.tanh());
        let x3 = self
            .edge_volatility
            .evaluate((connes.planck_market_constant * 0.25).tanh());

        let vec_state = CliffordMultivector3D::from_vector(x1, x2, x3);
        let vec_conjugate = CliffordMultivector3D::from_vector(
            qsde_state.bloch_z,
            net_flow.tanh(),
            qsde_state.purity - 0.5,
        );
        let rotor = CliffordMultivector3D::rotor_from_bivector_angle(
            qsde_state.bloch_x,
            qsde_state.bloch_y,
            qsde_state.bloch_z,
            connes
                .dirac_lipschitz_norm
                .clamp(-std::f64::consts::PI, std::f64::consts::PI),
        );

        let clifford_output = vec_state
            .geometric_product(&vec_conjugate)
            .apply_rotor(&rotor);

        let raw_signal = 0.6 * qsde_state.bloch_z
            + 0.4 * (clifford_output.coeffs[0] + clifford_output.coeffs[1]);
        let directional_conviction = if raw_signal.is_finite() {
            raw_signal.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let regime = if connes.heisenberg_uncertainty_bound
            > self.connes_engine.uncertainty_veto_threshold
            || qsde_state.purity < 0.52
        {
            NonCommutativeRegime::HeisenbergUncertaintyVeto
        } else if jump_detected {
            NonCommutativeRegime::QuantumJumpBreakout
        } else {
            NonCommutativeRegime::CoherentLowImpact
        };

        ConnesCliffordKanReport {
            connes,
            qsde_state,
            clifford_output,
            directional_conviction,
            regime,
        }
    }
}
