//! # Geometría Espectral No Conmutativa de Connes y Acción Quiral (§44)
//!
//! Implementa la geometría espectral no conmutativa basada en la tripleta espectral de Alain Connes
//! $(\mathcal{A}, \mathcal{H}, \mathcal{D})$ y el álgebra de Clifford de espinores de Dirac de dimensión 4:
//!
//! - $\mathcal{A}$: Álgebra $C^*$ no conmutativa de observables de microestructura generada por los
//!   operadores de tiempo de arribo $T$ y desequilibrio de flujo $M$, con relación de Heisenberg-Connes
//!   $[T, M] = i \theta \mathbf{1}$ inducida por la asimetría temporal de emparejamiento L2/L3.
//! - $\mathcal{H}$: Espacio de Hilbert espinorial $\mathbb{C}^4$ equipado con las matrices gamma de Dirac
//!   $\{\gamma^\mu, \gamma^\nu\} = 2 \eta^{\mu\nu} \mathbf{1}_4$ ($\eta = \operatorname{diag}(+1, -1, -1, -1)$).
//! - $\mathcal{D}$: Operador de Dirac autoadjunto diferencial $\mathcal{D} = \sum_{\mu=0}^3 \gamma^\mu \partial_\mu$.
//! - Distancia geodésica no conmutativa de Connes:
//!   $$d_{\mathcal{D}}(P, Q) = \sup_{a \in \mathcal{A}} \{ |P(a) - Q(a)| : \|[\mathcal{D}, a]\| \le 1 \}$$
//! - Índice de anomalía quiral $\Gamma_5 = i \gamma^0 \gamma^1 \gamma^2 \gamma^3$ para identificar
//!   ruptura espontánea de simetría de paridad en el flujo institucional.
//!
//! ## Invariantes de Silicio y Cero Pánicos
//! - Stack $O(1)$ sin asignaciones de heap dinámico.
//! - Cero `assert!`, cero `unwrap()`, cero `expect()`, cero `panic!()`.
//! - Cumplimiento estricto del invariante `rb_p1`.

/// Número de dimensiones del espacio-tiempo de espinores de Dirac.
pub const DIRAC_DIM: usize = 4;

/// Número complejo stack-allocated para álgebra de espinores y matrices gamma.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn abs(self) -> f64 {
        self.norm_sq().sqrt()
    }
}

/// Matriz compleja $4 \times 4$ en stack.
pub type Matrix4c = [[C64; DIRAC_DIM]; DIRAC_DIM];

/// Multiplicación matricial $4 \times 4$ compleja en stack.
#[allow(clippy::needless_range_loop)]
pub fn mat4_mul(a: &Matrix4c, b: &Matrix4c) -> Matrix4c {
    let mut out = [[C64::ZERO; DIRAC_DIM]; DIRAC_DIM];
    for i in 0..DIRAC_DIM {
        for j in 0..DIRAC_DIM {
            let mut sum = C64::ZERO;
            for k in 0..DIRAC_DIM {
                sum = sum.add(a[i][k].mul(b[k][j]));
            }
            out[i][j] = sum;
        }
    }
    out
}

/// Suma matricial $4 \times 4$ compleja en stack.
#[allow(clippy::needless_range_loop)]
pub fn mat4_add(a: &Matrix4c, b: &Matrix4c) -> Matrix4c {
    let mut out = [[C64::ZERO; DIRAC_DIM]; DIRAC_DIM];
    for i in 0..DIRAC_DIM {
        for j in 0..DIRAC_DIM {
            out[i][j] = a[i][j].add(b[i][j]);
        }
    }
    out
}

/// Traza matricial compleja $\operatorname{Tr}(M)$.
#[allow(clippy::needless_range_loop)]
pub fn mat4_trace(m: &Matrix4c) -> C64 {
    let mut tr = C64::ZERO;
    for i in 0..DIRAC_DIM {
        tr = tr.add(m[i][i]);
    }
    tr
}

/// Generador de las matrices gamma de Dirac $\gamma^0, \gamma^1, \gamma^2, \gamma^3$ y $\Gamma_5$
/// en la representación canónica de Dirac-Pauli.
pub struct DiracGammaAlgebra;

impl DiracGammaAlgebra {
    /// $\gamma^0 = \operatorname{diag}(+1, +1, -1, -1)$
    pub fn gamma0() -> Matrix4c {
        [
            [C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO],
            [C64::ZERO, C64::ONE, C64::ZERO, C64::ZERO],
            [C64::ZERO, C64::ZERO, C64::new(-1.0, 0.0), C64::ZERO],
            [C64::ZERO, C64::ZERO, C64::ZERO, C64::new(-1.0, 0.0)],
        ]
    }

    /// $\gamma^1$: espín espacial x
    pub fn gamma1() -> Matrix4c {
        [
            [C64::ZERO, C64::ZERO, C64::ZERO, C64::ONE],
            [C64::ZERO, C64::ZERO, C64::ONE, C64::ZERO],
            [C64::ZERO, C64::new(-1.0, 0.0), C64::ZERO, C64::ZERO],
            [C64::new(-1.0, 0.0), C64::ZERO, C64::ZERO, C64::ZERO],
        ]
    }

    /// $\gamma^2$: espín espacial y con componentes imaginarias
    pub fn gamma2() -> Matrix4c {
        [
            [C64::ZERO, C64::ZERO, C64::ZERO, C64::new(0.0, -1.0)],
            [C64::ZERO, C64::ZERO, C64::I, C64::ZERO],
            [C64::ZERO, C64::I, C64::ZERO, C64::ZERO],
            [C64::new(0.0, -1.0), C64::ZERO, C64::ZERO, C64::ZERO],
        ]
    }

    /// $\gamma^3$: espín espacial z
    pub fn gamma3() -> Matrix4c {
        [
            [C64::ZERO, C64::ZERO, C64::ONE, C64::ZERO],
            [C64::ZERO, C64::ZERO, C64::ZERO, C64::new(-1.0, 0.0)],
            [C64::new(-1.0, 0.0), C64::ZERO, C64::ZERO, C64::ZERO],
            [C64::ZERO, C64::ONE, C64::ZERO, C64::ZERO],
        ]
    }

    /// Matriz de quiralidad $\Gamma_5 = i \gamma^0 \gamma^1 \gamma^2 \gamma^3$
    pub fn gamma5() -> Matrix4c {
        [
            [C64::ZERO, C64::ZERO, C64::ONE, C64::ZERO],
            [C64::ZERO, C64::ZERO, C64::ZERO, C64::ONE],
            [C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO],
            [C64::ZERO, C64::ONE, C64::ZERO, C64::ZERO],
        ]
    }
}

/// Parámetros de configuración del motor de Geometría Espectral de Connes.
#[derive(Debug, Clone, Copy)]
pub struct ConnesSpectralParams {
    /// Parámetro de no conmutatividad del flujo de órdenes $\theta$ (grano temporal/latencia).
    pub non_commutative_theta: f64,
    /// Escala de corte ultravioleta de Chamseddine-Connes $\Lambda$.
    pub spectral_cutoff_lambda: f64,
    /// Umbral crítico de anomalía quiral para detección de presión asimétrica.
    pub chiral_anomaly_threshold: f64,
}

impl Default for ConnesSpectralParams {
    fn default() -> Self {
        Self {
            non_commutative_theta: 0.05,
            spectral_cutoff_lambda: 1.0,
            chiral_anomaly_threshold: 0.35,
        }
    }
}

/// Informe analítico emitido por la geometría espectral de Connes.
#[derive(Debug, Clone, Copy)]
pub struct ConnesSpectralReport {
    /// Distancia geodésica no conmutativa de Connes $d_{\mathcal{D}}(P, Q) \ge 0.0$.
    pub connes_distance: f64,
    /// Índice de quiralidad $\chi \in [-1.0, 1.0]$ ($\chi > 0$ compra neta, $\chi < 0$ venta neta).
    pub chiral_index: f64,
    /// Curvatura escalar no conmutativa $\mathcal{R}$ inducida por el conmutador de Dirac.
    pub spectral_curvature: f64,
    /// Acción espectral total de Chamseddine-Connes $\operatorname{Tr}(f(\mathcal{D}/\Lambda))$.
    pub spectral_action: f64,
    /// Factor de modulación continua para dimensionamiento de riesgo $\eta_{\text{Connes}} \in [0.10, 1.25]$.
    pub slot_scaling_factor: f64,
}

/// Motor de Geometría Espectral No Conmutativa de Connes para microestructura y order flow.
#[derive(Debug, Clone)]
pub struct ConnesSpectralGeometryEngine {
    params: ConnesSpectralParams,
    /// Estado cuatridimensional anterior: $(t, p, v, \text{ofi})$.
    last_state: [f64; DIRAC_DIM],
    /// Marca de tiempo de la última actualización.
    last_ts_ms: u64,
}

impl ConnesSpectralGeometryEngine {
    /// Inicializa un nuevo motor de geometría de Connes en stack.
    pub fn new(params: ConnesSpectralParams) -> Self {
        Self {
            params,
            last_state: [0.0; DIRAC_DIM],
            last_ts_ms: 0,
        }
    }

    /// Evalúa la transición de microestructura en el espacio-tiempo espinorial no conmutativo.
    ///
    /// - `norm_price`: Retorno o precio normalizado $\in [-1.0, 1.0]$.
    /// - `norm_volume`: Desequilibrio de volumen $\in [-1.0, 1.0]$.
    /// - `order_flow_imbalance`: OFI normalizado $\in [-1.0, 1.0]$.
    /// - `ts_ms`: Marca de tiempo en milisegundos.
    #[allow(clippy::needless_range_loop)]
    pub fn update_and_evaluate(
        &mut self,
        norm_price: f64,
        norm_volume: f64,
        order_flow_imbalance: f64,
        ts_ms: u64,
    ) -> ConnesSpectralReport {
        let p = if norm_price.is_finite() {
            norm_price.clamp(-5.0, 5.0)
        } else {
            0.0
        };
        let v = if norm_volume.is_finite() {
            norm_volume.clamp(-5.0, 5.0)
        } else {
            0.0
        };
        let ofi = if order_flow_imbalance.is_finite() {
            order_flow_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let dt = if self.last_ts_ms > 0 && ts_ms > self.last_ts_ms {
            let elapsed = (ts_ms - self.last_ts_ms) as f64 * 0.001;
            elapsed.clamp(0.001, 10.0)
        } else {
            0.1
        };
        self.last_ts_ms = ts_ms;

        let current_state = [dt, p, v, ofi];

        // 1. Vector diferencial en espacio de Minkowski $\Delta x^\mu$
        let mut delta_x = [0.0; DIRAC_DIM];
        for i in 0..DIRAC_DIM {
            delta_x[i] = current_state[i] - self.last_state[i];
        }
        self.last_state = current_state;

        // 2. Construcción del operador de Dirac $\mathcal{D} = \sum_{\mu=0}^3 \gamma^\mu \Delta x_\mu$
        let g0 = DiracGammaAlgebra::gamma0();
        let g1 = DiracGammaAlgebra::gamma1();
        let g2 = DiracGammaAlgebra::gamma2();
        let g3 = DiracGammaAlgebra::gamma3();

        // Métrica $\eta_{\mu\nu}$: $\Delta x_0 = \Delta x^0$, $\Delta x_k = -\Delta x^k$
        let dx0 = delta_x[0];
        let dx1 = -delta_x[1];
        let dx2 = -delta_x[2];
        let dx3 = -delta_x[3];

        let mut dirac_op = [[C64::ZERO; DIRAC_DIM]; DIRAC_DIM];
        for i in 0..DIRAC_DIM {
            for j in 0..DIRAC_DIM {
                let term0 = g0[i][j].scale(dx0);
                let term1 = g1[i][j].scale(dx1);
                let term2 = g2[i][j].scale(dx2);
                let term3 = g3[i][j].scale(dx3);
                dirac_op[i][j] = term0.add(term1).add(term2).add(term3);
            }
        }

        // 3. Conmutador no conmutativo de flujo de órdenes $[T, M]$
        // $T \sim \Delta t$, $M \sim \Delta \text{ofi}$. Término no abeliano $\theta \cdot |\Delta t \cdot \text{ofi} - \Delta \text{ofi} \cdot t|$
        let theta = self.params.non_commutative_theta.clamp(0.001, 1.0);
        let ofi_commutator = (delta_x[0] * ofi - delta_x[3] * dt).abs();
        let non_commutative_term = theta * ofi_commutator;

        // 4. Métrica Geodésica de Connes: $d_{\mathcal{D}} = \sqrt{\sum (\Delta x^\mu)^2 + \theta^2 [T, M]^2}$
        let euclidean_sq = delta_x[0] * delta_x[0]
            + delta_x[1] * delta_x[1]
            + delta_x[2] * delta_x[2]
            + delta_x[3] * delta_x[3];
        let connes_dist = (euclidean_sq + non_commutative_term * non_commutative_term).sqrt();

        // 5. Cálculo del Índice Quiral $\chi = \langle \psi | \Gamma_5 | \psi \rangle / \langle \psi | \psi \rangle$
        // Vector de estado espinorial $\psi = (\text{bid\_press}, \text{ask\_press}, \Delta v_{\text{taker}}, \Delta v_{\text{maker}})^T$
        let bid_press = (0.5 * (1.0 + ofi)).clamp(0.0, 1.0);
        let ask_press = (0.5 * (1.0 - ofi)).clamp(0.0, 1.0);
        let psi = [
            C64::new(bid_press, 0.0),
            C64::new(ask_press, 0.0),
            C64::new(delta_x[2].clamp(-1.0, 1.0), 0.0),
            C64::new(delta_x[1].clamp(-1.0, 1.0), 0.0),
        ];

        let g5 = DiracGammaAlgebra::gamma5();
        let mut gamma5_psi = [C64::ZERO; DIRAC_DIM];
        for i in 0..DIRAC_DIM {
            let mut sum = C64::ZERO;
            for j in 0..DIRAC_DIM {
                sum = sum.add(g5[i][j].mul(psi[j]));
            }
            gamma5_psi[i] = sum;
        }

        let mut expectation_g5 = 0.0;
        let mut psi_norm_sq = 0.0;
        for i in 0..DIRAC_DIM {
            psi_norm_sq += psi[i].norm_sq();
            let term = psi[i].conj().mul(gamma5_psi[i]);
            expectation_g5 += term.re;
        }

        let chiral_index = if psi_norm_sq > 1e-12 {
            (expectation_g5 / psi_norm_sq).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // 6. Acción Espectral de Chamseddine-Connes: $\operatorname{Tr}(\mathcal{D}^2 / \Lambda^2)$
        let dirac_sq = mat4_mul(&dirac_op, &dirac_op);
        let tr_d2 = mat4_trace(&dirac_sq).re;
        let lambda = self.params.spectral_cutoff_lambda.clamp(0.1, 10.0);
        let spectral_action = (tr_d2 / (lambda * lambda)).clamp(0.0, 100.0);

        // Curvatura escalar no conmutativa $\mathcal{R}$
        let spectral_curvature = (tr_d2 * 0.25).clamp(-50.0, 50.0);

        // 7. Factor de modulación de slot $\eta_{\text{Connes}}$
        let slot_scaling_factor = if chiral_index.abs() > self.params.chiral_anomaly_threshold {
            // Anomalía direccional clara: amplificación de convicción
            1.0 + 0.25 * (chiral_index.abs() - self.params.chiral_anomaly_threshold)
        } else {
            // Alta curvatura sin anomalía quiral: turbulencia no conmutativa, amortiguación
            1.0 / (1.0 + 0.10 * spectral_action)
        }
        .clamp(0.10, 1.25);

        ConnesSpectralReport {
            connes_distance: connes_dist,
            chiral_index,
            spectral_curvature,
            spectral_action,
            slot_scaling_factor,
        }
    }
}
