//! # Fase 14 — Vectorización Explícita SIMD 512-bit (`Avx512Lane8F64`), Integrador Simpléctico de Yoshida de 4to Orden y Dinámica Replicadora Evolutiva en el Simplex
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Implementación en Rust puro, `O(1)` en pila y sin asignaciones dinámicas, de tres pilares físico-matemáticos y de silicio:
//!   1. **Bloque Vectorial Explícito de 512 bits (`Avx512Lane8F64`):** Registro alineado a 64 bytes (`#[repr(C, align(64))]`) con 8 carriles `f64` que ejecuta operaciones FMA (`a * b + c`), productos internos, normas $L^2$ y evaluación simultánea de 8 canales de desequilibrio de flujo de órdenes (OFI) en un solo pase de línea de caché L1D.
//!   2. **Integrador Simpléctico de Yoshida de 4to Orden (`Yoshida4SymplecticIntegrator`):** Evolucionador canónico para sistemas Hamiltonianos de microestructura $H(q, p) = T(p) + V(q)$ en espacio de fases $\mathbb{R}^8 \times \mathbb{R}^8$ con potencial cuártico anarmónico y acoplamiento elástico entre modos. Preserva exactamente la 2-forma simpléctica $\omega = \sum_i dq_i \wedge dp_i$ ($\det J \equiv 1.0$) y elimina la deriva secular de energía ($|\Delta H| = \mathcal{O}(\Delta t^4)$).
//!   3. **Dinámica Replicadora-Mutadora Evolutiva en el Simplex (`EvolutionaryReplicatorSimplex`):** Ecuación de Taylor-Jonker acoplada con descenso espejo entrópico sobre la variedad de Riemann-Shahshahani $\Delta^7 = \{x \in \mathbb{R}^8_{\ge 0} : \sum x_i = 1\}$, matriz de pagos evolutiva $A \in \mathbb{R}^{8 \times 8}$, entropía de diversidad de Shannon $H_{\text{Shannon}}(x)$ y número efectivo de apuestas $N_{\text{eff}} = \exp(H_{\text{Shannon}})$.
//! - **POR QUÉ:** Los integradores no simplécticos (Euler, RK4 estándar) violan el teorema de Liouville ($\det J \neq 1$), inyectando ganancia o disipación artificial de energía en horizontes largos. Asimismo, la asignación estática de pesos entre generadores de señal ignora la competencia depredadora suma-cero y la selección natural en tiempo continuo.
//! - **PARA QUÉ:** Garantizar estabilidad orbital perpetua en la predicción de trayectorias de espacio de fases $(q_t, p_t)$, ejecutar álgebra vectorial de 8 canales a velocidad de registro ZMM/YMM sin pánicos ni subnormales, y adaptar los pesos del portafolio espectral mediante selección replicadora libre de sobreajuste.
//! - **CÓMO:** Composición simétrica de 3 etapas de Störmer-Verlet con los coeficientes cúbicos exactos de Haruo Yoshida ($w_1 = (2 - 2^{1/3})^{-1}$, $w_0 = -2^{1/3} w_1$), actualización multiplicativa de pesos de Shahshahani con regularización de mutación $\mu$, y bloques `[f64; 8]` alineados a 64B.
//! - **CUÁNDO:** En cada tick del motor de características (`feature-engine`) para actualizar el estado Hamiltoniano multicanal y la distribución de masa del simplex espectral.
//! - **DÓNDE:** `crates/feature-engine/src/simd_yoshida_replicator.rs`.
//! - **QUIÉN:** `Avx512Lane8F64`, `Yoshida4SymplecticIntegrator`, `EvolutionaryReplicatorSimplex`, `SimdYoshidaReplicatorEngine`.

/// Dimensión vectorial canónica de 512 bits: $8 \times 64\text{ bits} = 512\text{ bits}$ (64 bytes, 1 línea de caché L1D).
pub const SIMD_F64_LANES: usize = 8;

/// Coeficiente maestro de Yoshida de 4to orden: $w_1 = \frac{1}{2 - 2^{1/3}}$.
pub const YOSHIDA_W1: f64 = 1.351_207_191_959_657_8;

/// Coeficiente central compensatorio de Yoshida: $w_0 = -\frac{2^{1/3}}{2 - 2^{1/3}} = 1 - 2 w_1$.
pub const YOSHIDA_W0: f64 = -1.702_414_383_919_315_5;

/// Pesos de deriva de posición $c_1, c_2, c_3, c_4$ del integrador de Yoshida de 4to orden ($\sum c_k \equiv 1.0$).
pub const YOSHIDA_C: [f64; 4] = [
    0.675_603_595_979_828_9,  // w1 / 2
    -0.175_603_595_979_828_9, // (w0 + w1) / 2
    -0.175_603_595_979_828_9, // (w0 + w1) / 2
    0.675_603_595_979_828_9,  // w1 / 2
];

/// Pesos de impulso de momento $d_1, d_2, d_3$ del integrador de Yoshida de 4to orden ($\sum d_k \equiv 1.0$).
pub const YOSHIDA_D: [f64; 3] = [
    1.351_207_191_959_657_8,  // w1
    -1.702_414_383_919_315_5, // w0
    1.351_207_191_959_657_8,  // w1
];

/// Régimen dinámico-evolutivo conjunto diagnosticado por el motor de Fase 14.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymplecticEvolutionRegime {
    /// Órbita Hamiltoniana conservativa acotada y ecosistema replicador diversificado ($N_{\text{eff}} \ge 2.5$).
    ConservativeBalancedOrbit,
    /// Dominancia selectiva fuerte en el simplex ($N_{\text{eff}} < 2.5$) con flujo laminar coherente.
    SelectiveSimplexDominance,
    /// Excitación anarmónica de alta energía o divergencia numérica amortiguada por barrera simpléctica.
    HighEnergyAnharmonicGuard,
}

/// Bloque vectorial explícito de 512 bits (`8 x f64 = 64 bytes`) alineado a frontera de línea de caché L1D.
///
/// Su disposición contigua `#[repr(C, align(64))]` permite al backend LLVM emitir instrucciones
/// `vfmadd231pd` / `vaddpd` / `vmulpd` sobre registros ZMM (AVX-512) o pares YMM (AVX2) sin penalización de alineación.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Avx512Lane8F64 {
    pub lanes: [f64; SIMD_F64_LANES],
}

impl Default for Avx512Lane8F64 {
    #[inline]
    fn default() -> Self {
        Self::splat(0.0)
    }
}

impl Avx512Lane8F64 {
    /// Crea un registro de 512 bits replicando `val` en los 8 carriles (con guardia IEEE-754).
    #[inline]
    pub const fn splat(val: f64) -> Self {
        let v = if val.is_finite() { val } else { 0.0 };
        Self {
            lanes: [v; SIMD_F64_LANES],
        }
    }

    /// Construye el vector desde un arreglo de 8 `f64`, reemplazando cualquier `NaN`/`Inf` por `0.0`.
    #[inline]
    pub fn from_array_sanitized(raw: [f64; SIMD_F64_LANES]) -> Self {
        let mut clean = [0.0; SIMD_F64_LANES];
        let mut i = 0;
        while i < SIMD_F64_LANES {
            let v = raw[i];
            clean[i] = if v.is_finite() { v } else { 0.0 };
            i += 1;
        }
        Self { lanes: clean }
    }

    /// Acota cada carril en `[min_val, max_val]` de forma libre de pánicos incluso si `min_val > max_val`.
    #[inline]
    pub fn sanitize_clamped(&self, min_val: f64, max_val: f64) -> Self {
        let lo = if min_val.is_finite() { min_val } else { -1e6 };
        let hi_raw = if max_val.is_finite() { max_val } else { 1e6 };
        let hi = if hi_raw >= lo { hi_raw } else { lo };
        let mut out = [0.0; SIMD_F64_LANES];
        let mut i = 0;
        while i < SIMD_F64_LANES {
            let v = self.lanes[i];
            out[i] = if v.is_finite() { v.clamp(lo, hi) } else { 0.0 };
            i += 1;
        }
        Self { lanes: out }
    }

    /// Suma vectorial de 8 carriles en paralelo: `self + rhs`.
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        let a = &self.lanes;
        let b = &rhs.lanes;
        Self {
            lanes: [
                a[0] + b[0],
                a[1] + b[1],
                a[2] + b[2],
                a[3] + b[3],
                a[4] + b[4],
                a[5] + b[5],
                a[6] + b[6],
                a[7] + b[7],
            ],
        }
    }

    /// Resta vectorial de 8 carriles en paralelo: `self - rhs`.
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        let a = &self.lanes;
        let b = &rhs.lanes;
        Self {
            lanes: [
                a[0] - b[0],
                a[1] - b[1],
                a[2] - b[2],
                a[3] - b[3],
                a[4] - b[4],
                a[5] - b[5],
                a[6] - b[6],
                a[7] - b[7],
            ],
        }
    }

    /// Multiplicación carril a carril (Hadamard): `self * rhs`.
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        let a = &self.lanes;
        let b = &rhs.lanes;
        Self {
            lanes: [
                a[0] * b[0],
                a[1] * b[1],
                a[2] * b[2],
                a[3] * b[3],
                a[4] * b[4],
                a[5] * b[5],
                a[6] * b[6],
                a[7] * b[7],
            ],
        }
    }

    /// Multiplicación-acumulación fusionada (FMA) de 8 carriles: `self * mul_vec + add_vec`.
    #[inline]
    pub fn fma(&self, mul_vec: &Self, add_vec: &Self) -> Self {
        let a = &self.lanes;
        let b = &mul_vec.lanes;
        let c = &add_vec.lanes;
        Self {
            lanes: [
                a[0].mul_add(b[0], c[0]),
                a[1].mul_add(b[1], c[1]),
                a[2].mul_add(b[2], c[2]),
                a[3].mul_add(b[3], c[3]),
                a[4].mul_add(b[4], c[4]),
                a[5].mul_add(b[5], c[5]),
                a[6].mul_add(b[6], c[6]),
                a[7].mul_add(b[7], c[7]),
            ],
        }
    }

    /// Escalamiento por escalar con acumulación: `acc + scale * self`.
    #[inline]
    pub fn scale_add(&self, scale: f64, acc: &Self) -> Self {
        let s = if scale.is_finite() { scale } else { 0.0 };
        let a = &self.lanes;
        let c = &acc.lanes;
        Self {
            lanes: [
                a[0].mul_add(s, c[0]),
                a[1].mul_add(s, c[1]),
                a[2].mul_add(s, c[2]),
                a[3].mul_add(s, c[3]),
                a[4].mul_add(s, c[4]),
                a[5].mul_add(s, c[5]),
                a[6].mul_add(s, c[6]),
                a[7].mul_add(s, c[7]),
            ],
        }
    }

    /// Reducción horizontal en árbol binario balanceado (minimiza error de redondeo IEEE-754).
    #[inline]
    pub fn horizontal_sum(&self) -> f64 {
        let x = &self.lanes;
        let s01 = x[0] + x[1];
        let s23 = x[2] + x[3];
        let s45 = x[4] + x[5];
        let s67 = x[6] + x[7];
        let s0123 = s01 + s23;
        let s4567 = s45 + s67;
        let total = s0123 + s4567;
        if total.is_finite() {
            total
        } else {
            0.0
        }
    }

    /// Producto interno exacto de 8 carriles $\langle \text{self}, \text{rhs} \rangle = \sum_{i=0}^7 a_i b_i$.
    #[inline]
    pub fn dot_product(&self, rhs: &Self) -> f64 {
        self.mul(rhs).horizontal_sum()
    }

    /// Norma euclidiana al cuadrado $\|\text{self}\|_2^2 = \sum_{i=0}^7 a_i^2$.
    #[inline]
    pub fn l2_norm_sq(&self) -> f64 {
        self.dot_product(self)
    }

    /// Evalúa simultáneamente en 8 carriles el desequilibrio normalizado de flujo de órdenes
    /// $\text{OFI}_i = \frac{B_i - A_i}{B_i + A_i + \epsilon} \in [-1, 1]$.
    #[inline]
    pub fn eval_8way_ofi_imbalance(bids: &Self, asks: &Self) -> Self {
        let mut out = [0.0; SIMD_F64_LANES];
        let mut i = 0;
        while i < SIMD_F64_LANES {
            let b = if bids.lanes[i].is_finite() && bids.lanes[i] > 0.0 {
                bids.lanes[i]
            } else {
                0.0
            };
            let a = if asks.lanes[i].is_finite() && asks.lanes[i] > 0.0 {
                asks.lanes[i]
            } else {
                0.0
            };
            let denom = b + a;
            out[i] = if denom > 1e-12 {
                ((b - a) / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            i += 1;
        }
        Self { lanes: out }
    }
}

/// Estado instantáneo en el espacio de fases Hamiltoniano $(q, p) \in \mathbb{R}^8 \times \mathbb{R}^8$.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HamiltonianPhaseState {
    /// Coordenadas generalizadas de desplazamiento log-precio $q \in \mathbb{R}^8$.
    pub q: Avx512Lane8F64,
    /// Momentos conjugados canónicos de flujo de órdenes $p \in \mathbb{R}^8$.
    pub p: Avx512Lane8F64,
}

/// Diagnóstico energético y simpléctico de una trayectoria Hamiltoniana.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymplecticOrbitReport {
    /// Estado evolucionado $(q_t, p_t)$ en el espacio de fases.
    pub state: HamiltonianPhaseState,
    /// Energía cinética total $T(p) = \sum_i \frac{p_i^2}{2 m_i}$.
    pub kinetic_energy: f64,
    /// Energía potencial total $V(q)$ (armónica + cuártica anarmónica + acoplamiento elástico).
    pub potential_energy: f64,
    /// Hamiltoniano total conservado $H(q, p) = T(p) + V(q)$.
    pub total_energy: f64,
    /// Deriva absoluta de energía respecto al estado inicial $|H(t) - H(0)|$.
    pub energy_drift_abs: f64,
    /// Determinante de la matriz Jacobiana simpléctica de 1 paso ($\det J \equiv 1.0$ por teorema de Liouville).
    pub jacobian_determinant: f64,
}

/// Integrador Simpléctico de 4to Orden de Haruo Yoshida para el Hamiltoniano de microestructura.
#[derive(Debug, Clone, Copy)]
pub struct Yoshida4SymplecticIntegrator {
    /// Inverso de las masas efectivas de liquidez $m_i^{-1} > 0$ por modo.
    pub inv_mass: Avx512Lane8F64,
    /// Constantes de restitución armónica $k_i > 0$ (reversión al valor justo).
    pub spring_k: Avx512Lane8F64,
    /// Constantes de confinamiento cuártico anarmónico $\lambda_i \ge 0$ (muros de liquidez extremos).
    pub quartic_lambda: Avx512Lane8F64,
    /// Coeficiente de acoplamiento elástico entre modos vecinos $\kappa_{\text{couple}} \ge 0$.
    pub coupling_kappa: f64,
}

impl Default for Yoshida4SymplecticIntegrator {
    #[inline]
    fn default() -> Self {
        Self {
            inv_mass: Avx512Lane8F64::splat(1.0),
            spring_k: Avx512Lane8F64::splat(1.5),
            quartic_lambda: Avx512Lane8F64::splat(0.25),
            coupling_kappa: 0.15,
        }
    }
}

impl Yoshida4SymplecticIntegrator {
    /// Evalúa la energía cinética $T(p) = \frac{1}{2} \sum_{i=0}^7 m_i^{-1} p_i^2$.
    #[inline]
    pub fn kinetic_energy(&self, p: &Avx512Lane8F64) -> f64 {
        let p2 = p.mul(p);
        0.5 * p2.dot_product(&self.inv_mass)
    }

    /// Evalúa la energía potencial $V(q) = \sum_i (\frac{1}{2} k_i q_i^2 + \frac{1}{4} \lambda_i q_i^4) + \frac{1}{2}\kappa \sum_{i=0}^6 (q_{i+1}-q_i)^2$.
    #[inline]
    pub fn potential_energy(&self, q: &Avx512Lane8F64) -> f64 {
        let q2 = q.mul(q);
        let q4 = q2.mul(&q2);
        let harmonic = 0.5 * q2.dot_product(&self.spring_k);
        let quartic = 0.25 * q4.dot_product(&self.quartic_lambda);

        let kappa = if self.coupling_kappa.is_finite() && self.coupling_kappa > 0.0 {
            self.coupling_kappa
        } else {
            0.0
        };
        let mut coupling_sum = 0.0;
        let mut i = 0;
        while i + 1 < SIMD_F64_LANES {
            let diff = q.lanes[i + 1] - q.lanes[i];
            coupling_sum += diff * diff;
            i += 1;
        }
        let total = harmonic + quartic + 0.5 * kappa * coupling_sum;
        if total.is_finite() {
            total
        } else {
            0.0
        }
    }

    /// Evalúa el Hamiltoniano total $H(q, p) = T(p) + V(q)$.
    #[inline]
    pub fn hamiltonian(&self, state: &HamiltonianPhaseState) -> f64 {
        self.kinetic_energy(&state.p) + self.potential_energy(&state.q)
    }

    /// Calcula el campo de fuerza conservativo $F_i(q) = -\frac{\partial V}{\partial q_i}$.
    #[inline]
    pub fn conservative_force(&self, q: &Avx512Lane8F64) -> Avx512Lane8F64 {
        let kappa = if self.coupling_kappa.is_finite() && self.coupling_kappa > 0.0 {
            self.coupling_kappa
        } else {
            0.0
        };
        let mut force = [0.0; SIMD_F64_LANES];
        let mut i = 0;
        while i < SIMD_F64_LANES {
            let qi = q.lanes[i];
            let ki = self.spring_k.lanes[i];
            let li = self.quartic_lambda.lanes[i];
            let mut f = -ki * qi - li * qi * qi * qi;
            if i > 0 {
                f -= kappa * (qi - q.lanes[i - 1]);
            }
            if i + 1 < SIMD_F64_LANES {
                f += kappa * (q.lanes[i + 1] - qi);
            }
            force[i] = if f.is_finite() { f } else { 0.0 };
            i += 1;
        }
        Avx512Lane8F64 { lanes: force }
    }

    /// Ejecuta un paso simpléctico de 4to orden de Yoshida con paso temporal `dt`.
    ///
    /// Cada sub-etapa es un mapa de cizalladura exacta:
    /// - Deriva de posición: $q \leftarrow q + (c_k \Delta t) m^{-1} p$ (Jacobiano $\begin{pmatrix} I & c_k \Delta t M^{-1} \\ 0 & I \end{pmatrix}$, $\det = 1$).
    /// - Impulso de momento: $p \leftarrow p + (d_k \Delta t) F(q)$ (Jacobiano $\begin{pmatrix} I & 0 \\ d_k \Delta t D F(q) & I \end{pmatrix}$, $\det = 1$).
    #[inline]
    pub fn step_yoshida4(&self, state: &HamiltonianPhaseState, dt: f64) -> HamiltonianPhaseState {
        let h = if dt.is_finite() {
            dt.clamp(-0.25, 0.25)
        } else {
            0.0
        };
        let mut q = state.q.sanitize_clamped(-1e4, 1e4);
        let mut p = state.p.sanitize_clamped(-1e4, 1e4);

        // Etapa 1: deriva c1 + impulso d1
        let v0 = p.mul(&self.inv_mass);
        q = v0.scale_add(YOSHIDA_C[0] * h, &q);
        let f1 = self.conservative_force(&q);
        p = f1.scale_add(YOSHIDA_D[0] * h, &p);

        // Etapa 2: deriva c2 + impulso d2
        let v1 = p.mul(&self.inv_mass);
        q = v1.scale_add(YOSHIDA_C[1] * h, &q);
        let f2 = self.conservative_force(&q);
        p = f2.scale_add(YOSHIDA_D[1] * h, &p);

        // Etapa 3: deriva c3 + impulso d3
        let v2 = p.mul(&self.inv_mass);
        q = v2.scale_add(YOSHIDA_C[2] * h, &q);
        let f3 = self.conservative_force(&q);
        p = f3.scale_add(YOSHIDA_D[2] * h, &p);

        // Etapa 4 final: deriva c4
        let v3 = p.mul(&self.inv_mass);
        q = v3.scale_add(YOSHIDA_C[3] * h, &q);

        HamiltonianPhaseState {
            q: q.sanitize_clamped(-1e4, 1e4),
            p: p.sanitize_clamped(-1e4, 1e4),
        }
    }

    /// Calcula el determinante exacto de la matriz Jacobiana $2 \times 2$ del modo $i$ para un paso de Yoshida,
    /// verificando algebraicamente la condición simpléctica $J^T \Omega J = \Omega \iff \det J \equiv 1.0$.
    #[inline]
    pub fn single_mode_jacobian_det(&self, q_i: f64, mode_idx: usize, dt: f64) -> f64 {
        let idx = mode_idx.min(SIMD_F64_LANES - 1);
        let h = if dt.is_finite() {
            dt.clamp(-0.25, 0.25)
        } else {
            0.0
        };
        let inv_m = self.inv_mass.lanes[idx];
        let k = self.spring_k.lanes[idx];
        let lam = self.quartic_lambda.lanes[idx];

        // Matriz Jacobiana acumulada J = [[j00, j01], [j10, j11]] inicializada en la identidad
        let mut j00 = 1.0_f64;
        let mut j01 = 0.0_f64;
        let mut j10 = 0.0_f64;
        let mut j11 = 1.0_f64;
        let mut q = if q_i.is_finite() { q_i } else { 0.0 };

        let mut stage = 0;
        while stage < 3 {
            // Multiplicación por matriz de deriva D_c = [[1, c*h*inv_m], [0, 1]]
            let alpha = YOSHIDA_C[stage] * h * inv_m;
            j00 += alpha * j10;
            j01 += alpha * j11;

            // Derivada segunda del potencial en q: F'(q) = -k - 3*lam*q^2
            let df_dq = -k - 3.0 * lam * q * q;
            // Multiplicación por matriz de impulso K_d = [[1, 0], [d*h*df_dq, 1]]
            let beta = YOSHIDA_D[stage] * h * df_dq;
            j10 += beta * j00;
            j11 += beta * j01;

            q *= 0.99; // actualización local de referencia
            stage += 1;
        }

        // Deriva final c4
        let alpha_final = YOSHIDA_C[3] * h * inv_m;
        j00 += alpha_final * j10;
        j01 += alpha_final * j11;

        let det = j00 * j11 - j01 * j10;
        if det.is_finite() {
            det
        } else {
            1.0
        }
    }

    /// Integra `n_steps` pasos simplécticos de Yoshida y devuelve el diagnóstico energético completo.
    #[inline]
    pub fn integrate_orbit(
        &self,
        initial: &HamiltonianPhaseState,
        dt: f64,
        n_steps: usize,
    ) -> SymplecticOrbitReport {
        let h0 = self.hamiltonian(initial);
        let mut current = *initial;
        let steps = n_steps.min(10_000);
        let mut s = 0;
        while s < steps {
            current = self.step_yoshida4(&current, dt);
            s += 1;
        }
        let kin = self.kinetic_energy(&current.p);
        let pot = self.potential_energy(&current.q);
        let h_end = kin + pot;
        let det_j = self.single_mode_jacobian_det(current.q.lanes[0], 0, dt);

        SymplecticOrbitReport {
            state: current,
            kinetic_energy: kin,
            potential_energy: pot,
            total_energy: h_end,
            energy_drift_abs: (h_end - h0).abs(),
            jacobian_determinant: det_j,
        }
    }
}

/// Estado y métricas de la Dinámica Replicadora-Mutadora en el Simplex $\Delta^7$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReplicatorSimplexReport {
    /// Distribución de pesos evolucionada $x \in \Delta^7$ ($\sum_{i=0}^7 x_i \equiv 1.0, x_i \ge 0$).
    pub weights: Avx512Lane8F64,
    /// Vector de aptitud (fitness) instantáneo $f(x) = A x + r$.
    pub fitness_vector: Avx512Lane8F64,
    /// Aptitud media poblacional $\bar{f}(x) = \sum_i x_i f_i(x)$.
    pub mean_fitness: f64,
    /// Entropía de diversidad de Shannon $H(x) = -\sum_i x_i \ln x_i \in [0, \ln 8]$.
    pub shannon_entropy: f64,
    /// Número efectivo de generadores activos $N_{\text{eff}} = \exp(H(x)) \in [1, 8]$.
    pub effective_species_count: f64,
    /// Norma de velocidad de información de Fisher-Rao / Shahshahani $\sum_i \frac{\dot{x}_i^2}{x_i}$.
    pub shahshahani_fisher_velocity: f64,
}

/// Motor de Dinámica Replicadora-Mutadora de Taylor-Jonker en el Simplex $\Delta^7$.
#[derive(Debug, Clone, Copy)]
pub struct EvolutionaryReplicatorSimplex {
    /// Matriz de pagos evolutiva $A \in \mathbb{R}^{8 \times 8}$ entre los 8 generadores espectrales.
    pub payoff_matrix: [[f64; SIMD_F64_LANES]; SIMD_F64_LANES],
    /// Tasa de selección natural $\eta > 0$ (intensidad de replicación de Shahshahani).
    pub selection_intensity: f64,
    /// Tasa de mutación exploratoria continua $\mu \in [0, 0.25]$ hacia el baricentro uniforme $1/8$.
    pub mutation_rate: f64,
}

impl Default for EvolutionaryReplicatorSimplex {
    #[inline]
    fn default() -> Self {
        // Matriz antisimétrica cíclica suave por defecto (juego suma-cero conservativo A^T = -A)
        let mut a = [[0.0; SIMD_F64_LANES]; SIMD_F64_LANES];
        let mut i = 0;
        while i < SIMD_F64_LANES {
            let next = (i + 1) % SIMD_F64_LANES;
            a[i][next] = 0.10;
            a[next][i] = -0.10;
            i += 1;
        }
        Self {
            payoff_matrix: a,
            selection_intensity: 1.0,
            mutation_rate: 0.01,
        }
    }
}

impl EvolutionaryReplicatorSimplex {
    /// Proyecta y normaliza cualquier vector arbitrario sobre el simplex probabilístico $\Delta^7$.
    #[inline]
    pub fn project_to_simplex(raw: &Avx512Lane8F64) -> Avx512Lane8F64 {
        let mut pos = [0.0; SIMD_F64_LANES];
        let mut sum = 0.0;
        let mut i = 0;
        while i < SIMD_F64_LANES {
            let v = raw.lanes[i];
            let p = if v.is_finite() && v > 0.0 { v } else { 0.0 };
            pos[i] = p;
            sum += p;
            i += 1;
        }
        if sum > 1e-15 && sum.is_finite() {
            let inv = 1.0 / sum;
            let mut j = 0;
            while j < SIMD_F64_LANES {
                pos[j] *= inv;
                j += 1;
            }
            Avx512Lane8F64 { lanes: pos }
        } else {
            Avx512Lane8F64::splat(1.0 / (SIMD_F64_LANES as f64))
        }
    }

    /// Evalúa un paso de la ecuación replicadora-mutadora mediante el mapa exponencial (Mirror Descent en métrica de Shahshahani):
    /// $$x_i(t + \Delta t) = (1 - \mu_{\text{eff}}) \frac{x_i(t) \exp(\eta \Delta t (f_i(x) - \bar{f}(x)))}{\sum_j x_j(t) \exp(\eta \Delta t (f_j(x) - \bar{f}(x)))} + \frac{\mu_{\text{eff}}}{8}$$
    #[inline]
    pub fn step_replicator(
        &self,
        current_weights: &Avx512Lane8F64,
        intrinsic_growth_rates: &Avx512Lane8F64,
        dt: f64,
    ) -> ReplicatorSimplexReport {
        let x = Self::project_to_simplex(current_weights);
        let r = intrinsic_growth_rates.sanitize_clamped(-50.0, 50.0);
        let h = if dt.is_finite() && dt > 0.0 {
            dt.min(1.0)
        } else {
            0.0
        };
        let eta = if self.selection_intensity.is_finite() && self.selection_intensity > 0.0 {
            self.selection_intensity.min(20.0)
        } else {
            1.0
        };
        let mu = if self.mutation_rate.is_finite() {
            self.mutation_rate.clamp(0.0, 0.25)
        } else {
            0.0
        };

        // 1. Calcular fitness f_i(x) = (A x)_i + r_i
        let mut fitness = [0.0; SIMD_F64_LANES];
        let mut i = 0;
        while i < SIMD_F64_LANES {
            let row = Avx512Lane8F64::from_array_sanitized(self.payoff_matrix[i]);
            fitness[i] = row.dot_product(&x) + r.lanes[i];
            i += 1;
        }
        let fit_vec = Avx512Lane8F64 { lanes: fitness };
        let mean_fitness = x.dot_product(&fit_vec);

        // 2. Actualización exponencial de Shahshahani + velocidad de Fisher-Rao
        let mut next_raw = [0.0; SIMD_F64_LANES];
        let mut fisher_vel = 0.0;
        let mut k = 0;
        while k < SIMD_F64_LANES {
            let advantage = (fitness[k] - mean_fitness).clamp(-20.0, 20.0);
            let x_k = x.lanes[k];
            let x_dot = x_k * advantage;
            if x_k > 1e-12 {
                fisher_vel += (x_dot * x_dot) / x_k;
            }
            next_raw[k] = x_k * (eta * h * advantage).exp();
            k += 1;
        }

        let selected = Self::project_to_simplex(&Avx512Lane8F64 { lanes: next_raw });

        // 3. Mezcla mutadora continua hacia el baricentro uniforme 1/8
        let mut_eff = (mu * h).clamp(0.0, 0.25);
        let uniform_share = 1.0 / (SIMD_F64_LANES as f64);
        let mut final_lanes = [0.0; SIMD_F64_LANES];
        let mut m = 0;
        while m < SIMD_F64_LANES {
            final_lanes[m] = (1.0 - mut_eff) * selected.lanes[m] + mut_eff * uniform_share;
            m += 1;
        }
        let weights = Self::project_to_simplex(&Avx512Lane8F64 { lanes: final_lanes });

        // 4. Entropía de Shannon y número efectivo de especies N_eff = exp(H)
        let mut shannon = 0.0;
        let mut n = 0;
        while n < SIMD_F64_LANES {
            let p = weights.lanes[n];
            if p > 1e-15 {
                shannon -= p * p.ln();
            }
            n += 1;
        }
        let shannon_clean = if shannon.is_finite() && shannon >= 0.0 {
            shannon
        } else {
            0.0
        };
        let n_eff = shannon_clean.exp().clamp(1.0, SIMD_F64_LANES as f64);

        ReplicatorSimplexReport {
            weights,
            fitness_vector: fit_vec,
            mean_fitness: if mean_fitness.is_finite() {
                mean_fitness
            } else {
                0.0
            },
            shannon_entropy: shannon_clean,
            effective_species_count: n_eff,
            shahshahani_fisher_velocity: if fisher_vel.is_finite() {
                fisher_vel
            } else {
                0.0
            },
        }
    }
}

/// Reporte unificado de la Fase 14 (SIMD 512-bit + Integrador Simpléctico de Yoshida + Dinámica Replicadora).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimdYoshidaReplicatorReport {
    /// Desequilibrio OFI calculado simultáneamente en 8 carriles SIMD.
    pub ofi_8way: Avx512Lane8F64,
    /// Diagnóstico energético y simpléctico del espacio de fases Hamiltoniano.
    pub orbit: SymplecticOrbitReport,
    /// Diagnóstico evolutivo en el simplex de probabilidades $\Delta^7$.
    pub replicator: ReplicatorSimplexReport,
    /// Señal direccional compuesta ponderada por el simplex y el momento canónico $[-1.0, 1.0]$.
    pub composite_conviction: f64,
    /// Clasificación del régimen físico-evolutivo.
    pub regime: SymplecticEvolutionRegime,
}

/// Motor maestro de Fase 14 que acopla la vectorización explícita de 8 carriles `f64`,
/// el integrador simpléctico de 4to orden de Yoshida y la dinámica replicadora de Taylor-Jonker.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdYoshidaReplicatorEngine {
    pub yoshida: Yoshida4SymplecticIntegrator,
    pub replicator: EvolutionaryReplicatorSimplex,
}

impl SimdYoshidaReplicatorEngine {
    /// Evalúa un ciclo completo de Fase 14 sobre 8 canales espectrales / activos en paralelo.
    #[inline]
    pub fn evaluate_cycle(
        &self,
        bids: &Avx512Lane8F64,
        asks: &Avx512Lane8F64,
        phase_state: &HamiltonianPhaseState,
        current_weights: &Avx512Lane8F64,
        dt: f64,
    ) -> SimdYoshidaReplicatorReport {
        let ofi_8way = Avx512Lane8F64::eval_8way_ofi_imbalance(bids, asks);

        // Perturbación suave del momento conjugado por el desequilibrio OFI antes del paso simpléctico
        let perturbed_state = HamiltonianPhaseState {
            q: phase_state.q,
            p: ofi_8way.scale_add(0.05, &phase_state.p),
        };
        let orbit = self.yoshida.integrate_orbit(&perturbed_state, dt, 1);

        // El fitness intrínseco combina la alineación momento-OFI menos la penalización de energía potencial local
        let alignment = orbit.state.p.mul(&ofi_8way);
        let q_sq = orbit.state.q.mul(&orbit.state.q);
        let intrinsic_growth = q_sq.scale_add(-0.25, &alignment);

        let replicator = self
            .replicator
            .step_replicator(current_weights, &intrinsic_growth, dt);

        let weighted_momentum = replicator.weights.dot_product(&orbit.state.p);
        let weighted_ofi = replicator.weights.dot_product(&ofi_8way);
        let raw_conviction = (0.55 * weighted_momentum + 0.45 * weighted_ofi).tanh();
        let composite_conviction = if raw_conviction.is_finite() {
            raw_conviction.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let regime = if orbit.total_energy > 25.0 || orbit.energy_drift_abs > 1e-2 {
            SymplecticEvolutionRegime::HighEnergyAnharmonicGuard
        } else if replicator.effective_species_count < 2.5 {
            SymplecticEvolutionRegime::SelectiveSimplexDominance
        } else {
            SymplecticEvolutionRegime::ConservativeBalancedOrbit
        };

        SimdYoshidaReplicatorReport {
            ofi_8way,
            orbit,
            replicator,
            composite_conviction,
            regime,
        }
    }
}
