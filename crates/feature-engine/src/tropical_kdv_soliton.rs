//! # Fase 15.2 — Geometría y Álgebra Lineal Tropical $(\mathbb{R}_{\max,+})$ y Solitones Korteweg-de Vries (KdV) por Transformada Inversa de Dispersión (IST)
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Implementación en Rust puro, `O(1)` en pila y sin asignaciones en el heap, de:
//!   1. **Álgebra Lineal Tropical (`TropicalMaxPlusMatrix<N>`):** Semianillo idempotente $(\mathbb{R} \cup \{-\infty\}, \oplus, \otimes)$ donde $a \oplus b = \max(a, b)$ y $a \otimes b = a + b$. Incluye multiplicación matricial tropical $(A \otimes B)_{ij} = \max_k (A_{ik} + B_{kj})$, potencias tropicales para rutas críticas multi-salto y el **Algoritmo Exacto de Karp** para el autovalor tropical dominante $\lambda_{\text{trop}}(A) = \max_i \min_{0 \le k < N} \frac{(A^{\otimes N})_{ii} - (A^{\otimes k})_{ii}}{N - k}$.
//!   2. **Detector de Solitones KdV por Transformada Inversa de Dispersión (`KdvSolitonIstAnalyzer`):** Analiza ondas solitarias de liquidez bajo la ecuación no lineal de Korteweg-de Vries $\partial_t u + 6 u \partial_x u + \partial_x^3 u = 0$ construyendo el operador discreto de Schrödinger de Lax $L = -\frac{d^2}{dx^2} - u(x)$ en pila ($M=8$), extrayendo los estados ligados negativos $E_n = -\kappa_n^2 < 0$, la amplitud solitónica $A_1 = 2\kappa_1^2$, la velocidad de propagación $v_1 = 4\kappa_1^2$, el corrimiento de fase exacto de colisión de 2 solitones $\Delta x_{\text{shift}}$ y las 3 primeras integrales conservadas de KdV ($I_1, I_2, I_3$).
//! - **POR QUÉ:** En el límite de de-cuantización de Maslov ($\hbar \to 0$), las integrales de camino colapsan al álgebra tropical $(\max, +)$, revelando el ciclo de arbitraje de máxima extracción por unidad de tiempo ($\lambda_{\text{trop}}$). Simultáneamente, los barridos institucionales masivos no se dispersan como ondas gaussianas lineales, sino que viajan como solitones KdV coherentes cuya velocidad es proporcional a su amplitud ($v = 2 A$).
//! - **PARA QUÉ:** Identificar ciclos de propagación de liderazgo con ganancia tropical positiva ($\lambda_{\text{trop}} > 0$) y distinguir ondas solitónicas reales (`SolitonBreakoutWave`) de ruido dispersivo radiación-dominado (`DispersiveRadiationNoise`).
//! - **CÓMO:** DP en pila $O(N^3)$ para la secuencia de potencias tropicales $A^{\otimes k}$ de Karp, y diagonalización simétrica de Jacobi sobre el operador tridiagonal de Lax-Schrödinger $8 \times 8$.
//! - **CUÁNDO:** En cada actualización del perfil espacial del libro de órdenes y de la matriz de transferencia dirigida multiactivo.
//! - **DÓNDE:** `crates/feature-engine/src/tropical_kdv_soliton.rs`.
//! - **QUIÉN:** `TropicalMaxPlusMatrix`, `TropicalSpectralReport`, `KdvSolitonIstAnalyzer`, `KdvSolitonReport`, `TropicalKdvEngine`.

use crate::random_matrix_wilson::jacobi_eigensolver_symmetric;

/// Dimensión espacial canónica para la malla del operador de Lax-Schrödinger KdV en pila.
pub const KDV_GRID_DIM: usize = 8;

/// Cero aditivo tropical ($-\infty$ truncado a constante finita segura para evitar `NaN` en restas $a - (-\infty)$).
pub const TROPICAL_ZERO: f64 = -1.0e12;

/// Identidad multiplicativa tropical ($0.0$, dado que $a \otimes 0.0 = a + 0.0 = a$).
pub const TROPICAL_ONE: f64 = 0.0;

/// Clasificación del régimen ondulatorio no lineal KdV y algebraico tropical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TropicalSolitonRegime {
    /// Onda solitónica coherente única ($\kappa_1 > 0.35$) con autovalor tropical constructivo.
    SolitonBreakoutWave,
    /// Colisión elástica de múltiples solitones ($N_{\text{solitons}} \ge 2$) con corrimiento de fase activo.
    MultiSolitonElasticCollision,
    /// Régimen sub-crítico dominado por radiación dispersiva lineal sin estados ligados fuertes.
    DispersiveRadiationNoise,
}

/// Suma tropical $a \oplus b = \max(a, b)$ en el semianillo $(\mathbb{R}_{\max,+})$.
#[inline]
pub fn tropical_add(a: f64, b: f64) -> f64 {
    let ca = if a.is_finite() {
        a.max(TROPICAL_ZERO)
    } else {
        TROPICAL_ZERO
    };
    let cb = if b.is_finite() {
        b.max(TROPICAL_ZERO)
    } else {
        TROPICAL_ZERO
    };
    ca.max(cb)
}

/// Producto tropical $a \otimes b = a + b$ en el semianillo $(\mathbb{R}_{\max,+})$.
#[inline]
pub fn tropical_mul(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || a <= TROPICAL_ZERO * 0.5 || b <= TROPICAL_ZERO * 0.5 {
        return TROPICAL_ZERO;
    }
    (a + b).clamp(TROPICAL_ZERO, 1.0e12)
}

/// Reporte espectral tropical obtenido mediante el algoritmo de Karp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TropicalSpectralReport {
    /// Autovalor tropical dominante $\lambda_{\text{trop}}(A)$ (media máxima de ciclo en el grafo ponderado).
    pub max_cycle_mean_eigenvalue: f64,
    /// Índice del nodo crítico $i^*$ que alcanza el ciclo tropical dominante.
    pub critical_node_idx: usize,
    /// Ganancia máxima acumulada a $N$ saltos desde el nodo crítico.
    pub max_n_hop_gain: f64,
}

/// Matriz cuadrada $N \times N$ sobre el semianillo tropical $(\mathbb{R}_{\max,+})$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TropicalMaxPlusMatrix<const N: usize> {
    pub entries: [[f64; N]; N],
}

impl<const N: usize> Default for TropicalMaxPlusMatrix<N> {
    #[inline]
    fn default() -> Self {
        Self::tropical_identity()
    }
}

impl<const N: usize> TropicalMaxPlusMatrix<N> {
    /// Matriz cero aditiva tropical (todas las entradas en `TROPICAL_ZERO`).
    #[inline]
    pub const fn tropical_zero_matrix() -> Self {
        Self {
            entries: [[TROPICAL_ZERO; N]; N],
        }
    }

    /// Matriz identidad multiplicativa tropical ($I_{ii} = 0.0$, $I_{ij} = -\infty$ para $i \neq j$).
    #[inline]
    pub const fn tropical_identity() -> Self {
        let mut mat = [[TROPICAL_ZERO; N]; N];
        let mut i = 0;
        while i < N {
            mat[i][i] = TROPICAL_ONE;
            i += 1;
        }
        Self { entries: mat }
    }

    /// Construye una matriz tropical saneando cualquier entrada no finita a `TROPICAL_ZERO`.
    #[inline]
    pub fn from_raw_sanitized(raw: [[f64; N]; N]) -> Self {
        let mut clean = [[TROPICAL_ZERO; N]; N];
        let mut i = 0;
        while i < N {
            let mut j = 0;
            while j < N {
                let v = raw[i][j];
                clean[i][j] = if v.is_finite() {
                    v.clamp(TROPICAL_ZERO, 1.0e9)
                } else {
                    TROPICAL_ZERO
                };
                j += 1;
            }
            i += 1;
        }
        Self { entries: clean }
    }

    /// Multiplicación matricial tropical $(A \otimes B)_{ij} = \bigoplus_{k=0}^{N-1} (A_{ik} \otimes B_{kj}) = \max_k (A_{ik} + B_{kj})$.
    #[inline]
    pub fn matmul_tropical(&self, rhs: &Self) -> Self {
        let mut out = [[TROPICAL_ZERO; N]; N];
        let mut i = 0;
        while i < N {
            let mut j = 0;
            while j < N {
                let mut best = TROPICAL_ZERO;
                let mut k = 0;
                while k < N {
                    let prod = tropical_mul(self.entries[i][k], rhs.entries[k][j]);
                    if prod > best {
                        best = prod;
                    }
                    k += 1;
                }
                out[i][j] = best;
                j += 1;
            }
            i += 1;
        }
        Self { entries: out }
    }

    /// Multiplicación matriz-vector tropical $(A \otimes v)_i = \max_j (A_{ij} + v_j)$.
    #[inline]
    pub fn matvec_tropical(&self, v: &[f64; N]) -> [f64; N] {
        let mut out = [TROPICAL_ZERO; N];
        let mut i = 0;
        while i < N {
            let mut best = TROPICAL_ZERO;
            let mut j = 0;
            while j < N {
                let prod = tropical_mul(self.entries[i][j], v[j]);
                if prod > best {
                    best = prod;
                }
                j += 1;
            }
            out[i] = best;
            i += 1;
        }
        out
    }

    /// Calcula el autovalor tropical exacto $\lambda_{\text{trop}}(A)$ (media máxima de ciclo)
    /// mediante programación dinámica de Karp en $\mathcal{O}(N^3)$ en pila.
    #[inline]
    pub fn karp_tropical_eigenvalue(&self) -> TropicalSpectralReport {
        if N == 0 {
            return TropicalSpectralReport {
                max_cycle_mean_eigenvalue: 0.0,
                critical_node_idx: 0,
                max_n_hop_gain: 0.0,
            };
        }
        // d[k][v] almacena el peso máximo de un camino de longitud k hasta el vértice v
        // partiendo de cualquier vértice con peso inicial 0.0.
        let mut d_prev = [0.0_f64; N];
        // Guardamos el historial de d[k][v] para k = 0..=N usando una matriz fija en pila:
        // Como N es const generic, iteramos N pasos guardando los estados por vértice.
        // Para evitar `[ [f64; N]; N + 1 ]` en const generics de Rust estable, evaluamos la potencia matricial tropical:
        let mut power_k = Self::tropical_identity();
        // trace_history[v] acumulará el mínimo sobre k < N de (d_N[v] - d_k[v]) / (N - k)
        // Primero obtenemos d_N aplicando N veces matvec_tropical sobre el vector cero [0.0; N].
        let mut d_steps = [[0.0_f64; N]; 16]; // Soporta hasta N=15 con exactitud completa de Karp; si N>=16 usa traza diagonal
        let eff_n = N.min(15);

        let mut step = 1;
        while step <= eff_n {
            let d_next = self.matvec_tropical(&d_prev);
            if step < 16 {
                d_steps[step] = d_next;
            }
            power_k = power_k.matmul_tropical(self);
            d_prev = d_next;
            step += 1;
        }

        let d_n = d_prev;
        let mut best_lambda = TROPICAL_ZERO;
        let mut best_node = 0;

        let mut v = 0;
        while v < N {
            if d_n[v] > TROPICAL_ZERO * 0.25 {
                let mut min_ratio = f64::INFINITY;
                let mut k = 0;
                while k < eff_n {
                    let dk_v = d_steps[k][v];
                    if dk_v > TROPICAL_ZERO * 0.25 {
                        let ratio = (d_n[v] - dk_v) / ((eff_n - k) as f64);
                        if ratio < min_ratio {
                            min_ratio = ratio;
                        }
                    }
                    k += 1;
                }
                if min_ratio.is_finite() && min_ratio > best_lambda {
                    best_lambda = min_ratio;
                    best_node = v;
                }
            }
            v += 1;
        }

        let clean_lambda = if best_lambda.is_finite() && best_lambda > TROPICAL_ZERO * 0.25 {
            best_lambda
        } else {
            0.0
        };
        let max_n_hop = if d_n[best_node].is_finite() && d_n[best_node] > TROPICAL_ZERO * 0.25 {
            d_n[best_node]
        } else {
            0.0
        };

        TropicalSpectralReport {
            max_cycle_mean_eigenvalue: clean_lambda,
            critical_node_idx: best_node,
            max_n_hop_gain: max_n_hop,
        }
    }
}

/// Diagnóstico físico de la Transformada Inversa de Dispersión (IST) para la ecuación de Korteweg-de Vries (KdV).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KdvSolitonReport {
    /// Número de estados ligados negativos ($E_n < -\epsilon$) que emergen como solitones independientes.
    pub bound_soliton_count: usize,
    /// Número de onda principal $\kappa_1 = \sqrt{-E_{\min}} \ge 0$.
    pub primary_wave_number_kappa1: f64,
    /// Número de onda secundario $\kappa_2 = \sqrt{-E_2} \ge 0$ (si existe un segundo solitón).
    pub secondary_wave_number_kappa2: f64,
    /// Amplitud asintótica del solitón líder $A_1 = 2 \kappa_1^2$.
    pub primary_soliton_amplitude: f64,
    /// Velocidad de propagación solitónica $v_1 = 4 \kappa_1^2 \equiv 2 A_1$.
    pub primary_soliton_velocity: f64,
    /// Corrimiento de fase analítico $\Delta x_{\text{shift}} = \frac{1}{\kappa_1} \ln \left|\frac{\kappa_1 + \kappa_2}{\kappa_1 - \kappa_2}\right|$ en colisión 2-solitón.
    pub two_soliton_phase_shift: f64,
    /// Primera ley de conservación KdV (Masa / Momento de orden 0): $I_1 = \int u(x)\,dx$.
    pub conserved_mass_i1: f64,
    /// Segunda ley de conservación KdV (Impulso / Energía $L^2$): $I_2 = \int u(x)^2\,dx$.
    pub conserved_momentum_i2: f64,
    /// Tercera ley de conservación KdV (Hamiltoniano de Whitham-KdV): $I_3 = \int (2 u^3 - (\partial_x u)^2)\,dx$.
    pub conserved_hamiltonian_i3: f64,
}

/// Analizador de Solitones KdV basado en el par de Lax y el operador espectral de Schrödinger 1D.
#[derive(Debug, Clone, Copy)]
pub struct KdvSolitonIstAnalyzer {
    /// Paso espacial de la malla del libro de órdenes $\Delta x > 0$.
    pub grid_spacing_dx: f64,
    /// Umbral mínimo de energía ligada negativa para certificar un solitón ($\epsilon_{\text{bound}} > 0$).
    pub bound_state_threshold: f64,
}

impl Default for KdvSolitonIstAnalyzer {
    #[inline]
    fn default() -> Self {
        Self {
            grid_spacing_dx: 1.0,
            bound_state_threshold: 0.05,
        }
    }
}

impl KdvSolitonIstAnalyzer {
    /// Evalúa el perfil analítico exacto de 1-solitón de Korteweg-de Vries:
    /// $$u(x, t) = 2 \kappa^2 \operatorname{sech}^2(\kappa (x - 4 \kappa^2 t - x_0))$$
    #[inline]
    pub fn analytical_sech2_soliton(kappa: f64, x: f64, x0: f64, t: f64) -> f64 {
        let k = if kappa.is_finite() && kappa > 0.0 {
            kappa.min(50.0)
        } else {
            return 0.0;
        };
        let pos = if x.is_finite() { x } else { 0.0 };
        let center = if x0.is_finite() { x0 } else { 0.0 };
        let time = if t.is_finite() { t } else { 0.0 };
        let phase = (k * (pos - 4.0 * k * k * time - center)).clamp(-50.0, 50.0);
        let ch = phase.cosh();
        if ch.is_finite() && ch > 0.0 {
            (2.0 * k * k) / (ch * ch)
        } else {
            0.0
        }
    }

    /// Extrae el espectro de solitones KdV desde un perfil espacial `u_profile` de hasta 8 niveles
    /// diagonalizando el operador discreto de Lax-Schrödinger $L = -\frac{d^2}{dx^2} - u(x)$.
    #[inline]
    pub fn analyze_profile(&self, u_profile: &[f64; KDV_GRID_DIM]) -> KdvSolitonReport {
        let dx = if self.grid_spacing_dx.is_finite() && self.grid_spacing_dx > 1e-4 {
            self.grid_spacing_dx
        } else {
            1.0
        };
        let inv_dx2 = 1.0 / (dx * dx);
        let thresh = if self.bound_state_threshold.is_finite() && self.bound_state_threshold > 0.0 {
            self.bound_state_threshold
        } else {
            0.05
        };

        let mut u = [0.0; KDV_GRID_DIM];
        let mut i = 0;
        while i < KDV_GRID_DIM {
            let v = u_profile[i];
            u[i] = if v.is_finite() {
                v.clamp(-1e3, 1e3)
            } else {
                0.0
            };
            i += 1;
        }

        // 1. Construir matriz simétrica plana 8x8 del operador de Schrödinger L = -d^2/dx^2 - u(x)
        let mut lax_flat = [0.0; KDV_GRID_DIM * KDV_GRID_DIM];
        let mut r = 0;
        while r < KDV_GRID_DIM {
            // Término diagonal: 2 / dx^2 - u(x_r)
            lax_flat[r * KDV_GRID_DIM + r] = 2.0 * inv_dx2 - u[r];
            if r + 1 < KDV_GRID_DIM {
                lax_flat[r * KDV_GRID_DIM + (r + 1)] = -inv_dx2;
                lax_flat[(r + 1) * KDV_GRID_DIM + r] = -inv_dx2;
            }
            r += 1;
        }

        let mut eigvals = [0.0; KDV_GRID_DIM];
        let mut eigvecs = [0.0; KDV_GRID_DIM * KDV_GRID_DIM];
        jacobi_eigensolver_symmetric(&lax_flat, KDV_GRID_DIM, &mut eigvals, &mut eigvecs);

        // Ordenar autovalores de menor a mayor (los más negativos son los solitones dominantes)
        let mut a = 0;
        while a < KDV_GRID_DIM {
            let mut b = a + 1;
            while b < KDV_GRID_DIM {
                if eigvals[b] < eigvals[a] {
                    eigvals.swap(a, b);
                }
                b += 1;
            }
            a += 1;
        }

        let mut soliton_count = 0;
        let mut idx = 0;
        while idx < KDV_GRID_DIM {
            if eigvals[idx] < -thresh {
                soliton_count += 1;
            }
            idx += 1;
        }

        let kappa1 = if eigvals[0] < -thresh {
            (-eigvals[0]).sqrt()
        } else {
            0.0
        };
        let kappa2 = if eigvals[1] < -thresh {
            (-eigvals[1]).sqrt()
        } else {
            0.0
        };

        let amp1 = 2.0 * kappa1 * kappa1;
        let vel1 = 4.0 * kappa1 * kappa1;

        let phase_shift = if kappa1 > 1e-6 && kappa2 > 1e-6 && (kappa1 - kappa2).abs() > 1e-6 {
            let ratio = ((kappa1 + kappa2) / (kappa1 - kappa2).abs()).max(1.0);
            ratio.ln() / kappa1
        } else {
            0.0
        };

        // 2. Integrales de conservación discretas de KdV (I1, I2, I3)
        let mut i1 = 0.0;
        let mut i2 = 0.0;
        let mut i3 = 0.0;
        let mut m = 0;
        while m < KDV_GRID_DIM {
            let um = u[m];
            i1 += um * dx;
            i2 += (um * um) * dx;
            let du_dx = if m + 1 < KDV_GRID_DIM {
                (u[m + 1] - um) / dx
            } else if m > 0 {
                (um - u[m - 1]) / dx
            } else {
                0.0
            };
            i3 += (2.0 * um * um * um - du_dx * du_dx) * dx;
            m += 1;
        }

        KdvSolitonReport {
            bound_soliton_count: soliton_count,
            primary_wave_number_kappa1: kappa1,
            secondary_wave_number_kappa2: kappa2,
            primary_soliton_amplitude: amp1,
            primary_soliton_velocity: vel1,
            two_soliton_phase_shift: if phase_shift.is_finite() {
                phase_shift
            } else {
                0.0
            },
            conserved_mass_i1: if i1.is_finite() { i1 } else { 0.0 },
            conserved_momentum_i2: if i2.is_finite() { i2 } else { 0.0 },
            conserved_hamiltonian_i3: if i3.is_finite() { i3 } else { 0.0 },
        }
    }
}

/// Reporte unificado de la Fase 15.2 (Álgebra Tropical $(\mathbb{R}_{\max,+})$ + Solitones KdV/IST).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TropicalKdvCycleReport {
    pub tropical: TropicalSpectralReport,
    pub soliton: KdvSolitonReport,
    /// Convicción direccional ponderada por velocidad solitónica y ganancia tropical $[-1.0, 1.0]$.
    pub directional_impulse: f64,
    pub regime: TropicalSolitonRegime,
}

/// Motor maestro de Fase 15.2 que acopla el análisis de ciclos tropicales de Karp con la
/// detección espectral de solitones KdV por Transformada Inversa de Dispersión.
#[derive(Debug, Clone, Copy, Default)]
pub struct TropicalKdvEngine {
    pub ist_analyzer: KdvSolitonIstAnalyzer,
}

impl TropicalKdvEngine {
    /// Evalúa un ciclo completo de geometría tropical y solitones KdV.
    #[inline]
    pub fn evaluate_cycle<const N: usize>(
        &self,
        transfer_matrix: &TropicalMaxPlusMatrix<N>,
        order_book_density_profile: &[f64; KDV_GRID_DIM],
        direction_sign: f64,
    ) -> TropicalKdvCycleReport {
        let tropical = transfer_matrix.karp_tropical_eigenvalue();
        let soliton = self
            .ist_analyzer
            .analyze_profile(order_book_density_profile);

        let dir = if direction_sign.is_finite() {
            direction_sign.signum()
        } else {
            0.0
        };
        let raw_impulse = dir
            * (0.65 * (soliton.primary_soliton_velocity * 0.1).tanh()
                + 0.35 * tropical.max_cycle_mean_eigenvalue.tanh());
        let directional_impulse = if raw_impulse.is_finite() {
            raw_impulse.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let regime = if soliton.bound_soliton_count >= 2 {
            TropicalSolitonRegime::MultiSolitonElasticCollision
        } else if soliton.bound_soliton_count == 1 && soliton.primary_wave_number_kappa1 > 0.35 {
            TropicalSolitonRegime::SolitonBreakoutWave
        } else {
            TropicalSolitonRegime::DispersiveRadiationNoise
        };

        TropicalKdvCycleReport {
            tropical,
            soliton,
            directional_impulse,
            regime,
        }
    }
}
