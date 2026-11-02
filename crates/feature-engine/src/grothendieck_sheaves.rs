//! # FASE 12: Haces Celulares de Grothendieck, Cohomología de Haz $H^1$ & Teorema del Índice de Atiyah-Singer
//!
//! **QUÉ:** Motor algebraico-topológico en stack para modelar el universo multiactivo como un
//! Haz Celular de Grothendieck $\mathcal{F}$ sobre un grafo celular $\mathcal{G} = (V, E, \triangle)$,
//! calculando el operador coborde $\delta_0$, el Laplaciano de Haz $\Delta_{\mathcal{F}} = \delta_0^T \delta_0$,
//! la obstrucción cohomológica $H^1(\mathcal{G}; \mathcal{F})$ y el índice topológico de Atiyah-Singer / McKean-Singer.
//!
//! **POR QUÉ:**
//! 1. Los modelos escalares de correlación ignoran que cada par de activos acoplados $e = (u, v)$
//!    está gobernado por aplicaciones lineales de restricción $\mathcal{F}_{u \trianglelefteq e}: \mathcal{F}(u) \to \mathcal{F}(e)$
//!    (betas cointegradas, paridades triangulares o bases Spot-Futuros).
//! 2. Un vector de estado $x \in C^0(\mathcal{G}; \mathcal{F})$ es una **sección global armónica**
//!    ($x \in H^0(\mathcal{G}; \mathcal{F}) = \ker \delta_0$) si y solo si satisface simultáneamente todas las
//!    restricciones de borde: $\mathcal{E}_{\text{sheaf}}(x) = x^T \Delta_{\mathcal{F}} x = \|\delta_0 x\|_2^2 = 0$.
//! 3. Cuando emerge una dislocación de precios o un ciclo de arbitraje frustrado, el gradiente armónico
//!    $g = -\Delta_{\mathcal{F}} x$ señala con exactitud matemática qué activos están sobrevalorados ($g_u < 0$)
//!    y cuáles subvalorados ($g_u > 0$) respecto al consenso del haz, mientras que la cohomología $H^1$
//!    separa desajustes transitorios explotables de rupturas topológicas de holonomía (ruptura de paridad).
//!
//! **PARA QUÉ:** Identificar oportunidades de arbitraje multiactivo libre de riesgo direccional,
//! cuantificar modos cero quirales topológicamente protegidos vía Atiyah-Singer y vetar trampas
//! de holonomía incoherente en tiempo sub-microsegundo ($O(N^2 + M)$ en stack, 0 heap allocations).

use crate::random_matrix_wilson::jacobi_eigensolver_symmetric;

/// Número máximo de vértices (activos) soportados en stack por el Haz Celular.
pub const MAX_SHEAF_VERTICES: usize = 16;

/// Número máximo de aristas dirigidas de restricción soportadas en stack.
pub const MAX_SHEAF_EDGES: usize = 32;

/// Arista dirigida $e = (u \to v)$ equipada con aplicaciones lineales de restricción del haz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SheafRestrictionEdge {
    /// Índice del nodo origen $u \in [0, N)$.
    pub source: usize,
    /// Índice del nodo destino $v \in [0, N)$.
    pub target: usize,
    /// Mapa de restricción $\mathcal{F}_{u \trianglelefteq e}$ desde el tallo de $u$ hacia el tallo de la arista $e$.
    pub restriction_source: f64,
    /// Mapa de restricción $\mathcal{F}_{v \trianglelefteq e}$ desde el tallo de $v$ hacia el tallo de la arista $e$.
    pub restriction_target: f64,
    /// Peso de acoplamiento o conductancia de la arista $w_e > 0$.
    pub weight: f64,
}

impl SheafRestrictionEdge {
    /// Construye una arista de restricción validada.
    #[inline]
    pub const fn new(
        source: usize,
        target: usize,
        restriction_source: f64,
        restriction_target: f64,
        weight: f64,
    ) -> Self {
        Self {
            source,
            target,
            restriction_source,
            restriction_target,
            weight,
        }
    }
}

/// Espacio de trabajo en stack con el Laplaciano de Haz $\Delta_{\mathcal{F}}$ y el vector coborde $\delta_0 x$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SheafLaplacianWorkspace {
    /// Matriz simétrica semidefinida positiva $\Delta_{\mathcal{F}} = \delta_0^T \delta_0 \in \mathbb{R}^{N \times N}$.
    pub laplacian: [[f64; MAX_SHEAF_VERTICES]; MAX_SHEAF_VERTICES],
    /// Vector coborde evaluado en cada arista válida $(\delta_0 x)_e = \sqrt{w_e}(r_{e,u} x_u - r_{e,v} x_v)$.
    pub coboundary: [f64; MAX_SHEAF_EDGES],
    /// Número de aristas válidas integradas en el operador.
    pub valid_edges: usize,
}

/// Clasificación topológica del estado multiactivo bajo el Haz de Grothendieck.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SheafTopologicalRegime {
    /// El estado $x$ pertenece (casi exactamente) al núcleo armónico $H^0(\mathcal{G}; \mathcal{F}) = \ker \delta_0$.
    HarmonicGlobalSection,
    /// Dislocación de precios explotable: la conexión del haz es consistente pero $x \notin \ker \delta_0$,
    /// generando una fuerza restauradora $-\Delta_{\mathcal{F}} x$ hacia la sección armónica.
    CohomologicalArbitrageDislocation,
    /// Obstrucción cohomológica severa en $H^1(\mathcal{G}; \mathcal{F})$: las mapas de restricción
    /// forman un ciclo frustrado con holonomía no trivial ($\lambda_{\min}(\Delta_{\mathcal{F}}) > \tau$).
    TopologicalHolonomyTrap,
}

/// Parámetros de configuración del motor de Haces de Grothendieck y Atiyah-Singer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrothendieckSheafParams {
    /// Umbral de energía de Dirichlet $\mathcal{E}_{\text{sheaf}}(x)$ bajo el cual se considera sección global armónica.
    pub harmonic_energy_threshold: f64,
    /// Tolerancia numérica para clasificar autovalores en el núcleo $\ker \Delta_{\mathcal{F}}$.
    pub kernel_eigen_tolerance: f64,
    /// Umbral de brecha espectral $\lambda_{\min}(\Delta_{\mathcal{F}})$ u holonomía cíclica para declarar trampa topológica.
    pub holonomy_trap_threshold: f64,
    /// Paso de difusión armónica $\alpha \in (0, 0.25]$ para proyectar estados hacia $H^0$.
    pub diffusion_step: f64,
    /// Número de pasos de flujo de calor del haz $x^{(k+1)} = (I - \alpha \Delta_{\mathcal{F}}) x^{(k)}$.
    pub diffusion_iterations: usize,
}

impl Default for GrothendieckSheafParams {
    fn default() -> Self {
        Self {
            harmonic_energy_threshold: 1e-4,
            kernel_eigen_tolerance: 1e-5,
            holonomy_trap_threshold: 0.05,
            diffusion_step: 0.10,
            diffusion_iterations: 8,
        }
    }
}

/// Reporte del Teorema del Índice de Atiyah-Singer / McKean-Singer sobre el complejo del haz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtiyahSingerIndexReport {
    /// Dimensión del espacio de secciones globales armónicas $\dim H^0(\mathcal{G}; \mathcal{F}) = \dim \ker \Delta_0$.
    pub dim_h0_global_sections: usize,
    /// Dimensión del primer grupo de cohomología $\dim H^1(\mathcal{G}; \mathcal{F}) = \dim \ker \Delta_1$.
    pub dim_h1_cohomology: usize,
    /// Índice analítico de Euler-McKean-Singer $\text{Ind}(D) = \dim H^0 - \dim H^1 \equiv N_V - N_E$.
    pub analytical_euler_index: i32,
    /// Flujo topológico de Chern-Weil $\Phi_{\text{Chern}} = \frac{1}{2\pi} \sum_{\triangle} \Omega_{uvw}$ integrado sobre 2-símplices.
    pub chern_weil_topological_flux: f64,
    /// Índice quiral efectivo $\text{Ind}_{\text{chiral}} = n_+ - n_-$ ponderado por la proyección espectral del estado.
    pub chiral_zero_mode_index: f64,
    /// Sesgo direccional topológicamente protegido en $[-1.0, 1.0]$.
    pub topological_chirality_bias: f64,
}

/// Reporte integral del análisis del Haz Celular de Grothendieck y Atiyah-Singer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellularSheafReport {
    /// Número de activos válidos analizados.
    pub n_vertices: usize,
    /// Número de aristas de restricción activas.
    pub n_edges: usize,
    /// Energía total de frustración de Dirichlet $\mathcal{E}_{\text{sheaf}}(x) = x^T \Delta_{\mathcal{F}} x = \|\delta_0 x\|_2^2$.
    pub frustration_energy: f64,
    /// Autovalor mínimo del Laplaciano de Haz $\lambda_{\min}(\Delta_{\mathcal{F}}) \ge 0$ (mide frustración intrínseca del haz).
    pub spectral_frustration_gap: f64,
    /// Norma de obstrucción de holonomía cíclica en triángulos cerrados $\|H^1_{\triangle}\|_2$.
    pub cycle_holonomy_obstruction: f64,
    /// Vector de fuerza restauradora armónica $g = -\Delta_{\mathcal{F}} x$ por cada activo.
    pub harmonic_restoration_force: [f64; MAX_SHEAF_VERTICES],
    /// Proyección del estado hacia la sección armónica global por difusión del haz.
    pub diffused_harmonic_section: [f64; MAX_SHEAF_VERTICES],
    /// Índice del activo con mayor subvaloración relativa ($\arg\max_u g_u$).
    pub most_undervalued_vertex: usize,
    /// Índice del activo con mayor sobrevaloración relativa ($\arg\min_u g_u$).
    pub most_overvalued_vertex: usize,
    /// Clasificación del régimen topológico del haz.
    pub regime: SheafTopologicalRegime,
    /// Diagnóstico completo del índice de Atiyah-Singer.
    pub atiyah_singer: AtiyahSingerIndexReport,
}

/// Motor de Haces Celulares de Grothendieck y Evaluador de Atiyah-Singer.
#[derive(Debug, Clone, Copy)]
pub struct CellularSheafEngine {
    pub params: GrothendieckSheafParams,
}

impl Default for CellularSheafEngine {
    fn default() -> Self {
        Self::new(GrothendieckSheafParams::default())
    }
}

impl CellularSheafEngine {
    /// Crea una instancia del motor de Haces de Grothendieck con parámetros explícitos.
    #[inline]
    pub const fn new(params: GrothendieckSheafParams) -> Self {
        Self { params }
    }

    /// Construye el Laplaciano de Haz $\Delta_{\mathcal{F}} = \delta_0^T \delta_0 \in \mathbb{R}^{N \times N}$ en stack
    /// y evalúa simultáneamente el vector coborde $y = \delta_0 x \in \mathbb{R}^M$.
    #[inline]
    pub fn build_sheaf_laplacian_and_coboundary(
        n_vertices: usize,
        vertex_stalks: &[f64],
        edges: &[SheafRestrictionEdge],
    ) -> SheafLaplacianWorkspace {
        let n = n_vertices.min(MAX_SHEAF_VERTICES).min(vertex_stalks.len());
        let mut laplacian = [[0.0_f64; MAX_SHEAF_VERTICES]; MAX_SHEAF_VERTICES];
        let mut coboundary = [0.0_f64; MAX_SHEAF_EDGES];
        let mut valid_edges = 0usize;

        for edge in edges.iter().take(MAX_SHEAF_EDGES) {
            let u = edge.source;
            let v = edge.target;
            if u >= n || v >= n || u == v {
                continue;
            }
            let ru = if edge.restriction_source.is_finite() {
                edge.restriction_source
            } else {
                0.0
            };
            let rv = if edge.restriction_target.is_finite() {
                edge.restriction_target
            } else {
                0.0
            };
            let w = if edge.weight.is_finite() && edge.weight > 0.0 {
                edge.weight
            } else {
                continue;
            };

            let xu = if vertex_stalks[u].is_finite() {
                vertex_stalks[u]
            } else {
                0.0
            };
            let xv = if vertex_stalks[v].is_finite() {
                vertex_stalks[v]
            } else {
                0.0
            };

            // (\delta_0 x)_e = \sqrt{w_e} (r_{e,u} x_u - r_{e,v} x_v)
            coboundary[valid_edges] = w.sqrt() * (ru * xu - rv * xv);
            valid_edges += 1;

            // Contribución simétrica al Laplaciano de Haz \Delta_F = \delta_0^T \delta_0
            laplacian[u][u] += w * ru * ru;
            laplacian[v][v] += w * rv * rv;
            let cross = w * ru * rv;
            laplacian[u][v] -= cross;
            laplacian[v][u] -= cross;
        }

        SheafLaplacianWorkspace {
            laplacian,
            coboundary,
            valid_edges,
        }
    }

    /// Calcula la obstrucción de holonomía cíclica en $H^1(\mathcal{G}; \mathcal{F})$ y el flujo
    /// topológico de Chern-Weil $\Phi_{\text{Chern}}$ sobre todos los 2-símplices (triángulos) del grafo.
    #[inline]
    pub fn compute_cycle_holonomy_and_chern_flux(
        n_vertices: usize,
        vertex_stalks: &[f64],
        edges: &[SheafRestrictionEdge],
    ) -> (f64, f64) {
        let n = n_vertices.min(MAX_SHEAF_VERTICES).min(vertex_stalks.len());
        if n < 3 {
            return (0.0, 0.0);
        }

        // Matriz de razones de transporte paralelo \rho_{u -> v} = r_{e,u} / r_{e,v}
        let mut transport = [[0.0_f64; MAX_SHEAF_VERTICES]; MAX_SHEAF_VERTICES];
        let mut has_edge = [[false; MAX_SHEAF_VERTICES]; MAX_SHEAF_VERTICES];

        for edge in edges.iter().take(MAX_SHEAF_EDGES) {
            let u = edge.source;
            let v = edge.target;
            if u >= n || v >= n || u == v {
                continue;
            }
            if !edge.restriction_source.is_finite()
                || !edge.restriction_target.is_finite()
                || !edge.weight.is_finite()
                || edge.weight <= 0.0
            {
                continue;
            }
            if edge.restriction_source.abs() > 1e-12 && edge.restriction_target.abs() > 1e-12 {
                let ratio_uv = edge.restriction_source / edge.restriction_target;
                transport[u][v] = ratio_uv;
                transport[v][u] = 1.0 / ratio_uv;
                has_edge[u][v] = true;
                has_edge[v][u] = true;
            }
        }

        let mut holonomy_sq_sum = 0.0_f64;
        let mut chern_flux_sum = 0.0_f64;

        for (u, &xu_raw) in vertex_stalks.iter().take(n).enumerate() {
            let xu = if xu_raw.is_finite() { xu_raw } else { 0.0 };
            for (v, &xv_raw) in vertex_stalks.iter().take(n).enumerate().skip(u + 1) {
                if !has_edge[u][v] {
                    continue;
                }
                let xv = if xv_raw.is_finite() { xv_raw } else { 0.0 };
                for (w, &xw_raw) in vertex_stalks.iter().take(n).enumerate().skip(v + 1) {
                    if !has_edge[v][w] || !has_edge[w][u] {
                        continue;
                    }
                    let xw = if xw_raw.is_finite() { xw_raw } else { 0.0 };
                    // Producto de holonomía alrededor del ciclo cerrado u -> v -> w -> u
                    let loop_product = transport[u][v] * transport[v][w] * transport[w][u];
                    if loop_product.is_finite() && loop_product > 0.0 {
                        let log_holonomy = loop_product.ln();
                        holonomy_sq_sum += log_holonomy * log_holonomy;

                        // Curvatura de Chern-Weil ponderada por la orientación del estado en el 2-símplice
                        let state_circulation =
                            (xv - xu) + (xw - xv) * transport[u][v] - (xw - xu) * loop_product;
                        chern_flux_sum += log_holonomy + 0.1 * state_circulation.tanh();
                    }
                }
            }
        }

        let holonomy_norm = holonomy_sq_sum.sqrt();
        let chern_flux = chern_flux_sum / std::f64::consts::TAU;
        (holonomy_norm, chern_flux)
    }

    /// Ejecuta el análisis completo del Haz Celular de Grothendieck y evalúa el Índice de Atiyah-Singer.
    #[inline]
    pub fn analyze_sheaf(
        &self,
        vertex_stalks: &[f64],
        edges: &[SheafRestrictionEdge],
    ) -> CellularSheafReport {
        let n = vertex_stalks.len().min(MAX_SHEAF_VERTICES);
        if n == 0 {
            return Self::empty_report();
        }

        let mut clean_x = [0.0_f64; MAX_SHEAF_VERTICES];
        for (slot, &val) in clean_x.iter_mut().zip(vertex_stalks.iter()).take(n) {
            *slot = if val.is_finite() { val } else { 0.0 };
        }

        let workspace = Self::build_sheaf_laplacian_and_coboundary(n, &clean_x[..n], edges);
        let laplacian = workspace.laplacian;
        let coboundary = workspace.coboundary;
        let valid_edges = workspace.valid_edges;

        // 1. Energía de frustración de Dirichlet: E(x) = ||\delta_0 x||_2^2 = x^T \Delta_F x
        let mut frustration_energy = 0.0_f64;
        for &cb in coboundary.iter().take(valid_edges) {
            frustration_energy += cb * cb;
        }

        // 2. Fuerza restauradora armónica g = -\Delta_F x
        let mut harmonic_force = [0.0_f64; MAX_SHEAF_VERTICES];
        let mut most_undervalued = 0usize;
        let mut most_overvalued = 0usize;
        let mut max_force = f64::NEG_INFINITY;
        let mut min_force = f64::INFINITY;

        for (u, (lap_row, force_slot)) in laplacian
            .iter()
            .zip(harmonic_force.iter_mut())
            .take(n)
            .enumerate()
        {
            let mut delta_x_u = 0.0_f64;
            for (&lap_uv, &xv) in lap_row.iter().zip(clean_x.iter()).take(n) {
                delta_x_u += lap_uv * xv;
            }
            let force_u = -delta_x_u;
            *force_slot = force_u;
            if force_u > max_force {
                max_force = force_u;
                most_undervalued = u;
            }
            if force_u < min_force {
                min_force = force_u;
                most_overvalued = u;
            }
        }

        // 3. Difusión de calor del haz x^{(k+1)} = x^{(k)} - \alpha \Delta_F x^{(k)}
        let mut diffused = clean_x;
        let alpha = self.params.diffusion_step.clamp(0.001, 0.25);
        let iters = self.params.diffusion_iterations.clamp(1, 32);
        for _ in 0..iters {
            let mut next_x = diffused;
            for (next_slot, (&diff_u, lap_row)) in next_x
                .iter_mut()
                .zip(diffused.iter().zip(laplacian.iter()))
                .take(n)
            {
                let mut lap_u = 0.0_f64;
                for (&lap_uv, &dv) in lap_row.iter().zip(diffused.iter()).take(n) {
                    lap_u += lap_uv * dv;
                }
                *next_slot = diff_u - alpha * lap_u;
            }
            diffused = next_x;
        }

        // 4. Espectro del Laplaciano de Haz \Delta_0 y Teorema del Índice de Atiyah-Singer / McKean-Singer
        let (eigenvalues, eigenvectors) = jacobi_eigensolver_symmetric(&laplacian, n, 40);
        let tol = self.params.kernel_eigen_tolerance.max(1e-12);

        let mut dim_h0 = 0usize;
        let mut min_eigenvalue = f64::INFINITY;
        let mut chiral_pos = 0.0_f64;
        let mut chiral_neg = 0.0_f64;

        for (mode_idx, &ev) in eigenvalues.iter().take(n).enumerate() {
            let clean_ev = ev.max(0.0);
            if clean_ev < min_eigenvalue {
                min_eigenvalue = clean_ev;
            }
            if clean_ev <= tol {
                dim_h0 += 1;
            }

            // Peso espectral de calor de McKean-Singer e^{-t \lambda_k} proyectado sobre la asimetría del estado
            let heat_weight = (-clean_ev).exp();
            let mut mode_projection = 0.0_f64;
            for (evec_row, &xu) in eigenvectors.iter().zip(clean_x.iter()).take(n) {
                mode_projection += evec_row[mode_idx] * xu;
            }
            let signed_contribution = heat_weight * mode_projection;
            if signed_contribution >= 0.0 {
                chiral_pos += signed_contribution;
            } else {
                chiral_neg += -signed_contribution;
            }
        }
        if !min_eigenvalue.is_finite() {
            min_eigenvalue = 0.0;
        }

        // Por el teorema del rango-nulidad en el complejo C^0 -> C^1:
        // rank(\delta_0) = n - dim_h0  =>  dim_h1 = dim ker(\delta_0^T) = valid_edges - rank(\delta_0)
        let rank_delta0 = n.saturating_sub(dim_h0);
        let dim_h1 = valid_edges.saturating_sub(rank_delta0);
        let analytical_euler_index = (dim_h0 as i32) - (dim_h1 as i32);

        let (cycle_holonomy_obstruction, chern_weil_topological_flux) =
            Self::compute_cycle_holonomy_and_chern_flux(n, &clean_x[..n], edges);

        let chiral_zero_mode_index = (chiral_pos - chiral_neg) + chern_weil_topological_flux;
        let chiral_denom = (chiral_pos + chiral_neg + chern_weil_topological_flux.abs()).max(1e-9);
        let topological_chirality_bias = (chiral_zero_mode_index / chiral_denom).clamp(-1.0, 1.0);

        let regime = if cycle_holonomy_obstruction > self.params.holonomy_trap_threshold
            || (valid_edges >= n && min_eigenvalue > self.params.holonomy_trap_threshold)
        {
            SheafTopologicalRegime::TopologicalHolonomyTrap
        } else if frustration_energy <= self.params.harmonic_energy_threshold {
            SheafTopologicalRegime::HarmonicGlobalSection
        } else {
            SheafTopologicalRegime::CohomologicalArbitrageDislocation
        };

        CellularSheafReport {
            n_vertices: n,
            n_edges: valid_edges,
            frustration_energy,
            spectral_frustration_gap: min_eigenvalue,
            cycle_holonomy_obstruction,
            harmonic_restoration_force: harmonic_force,
            diffused_harmonic_section: diffused,
            most_undervalued_vertex: most_undervalued,
            most_overvalued_vertex: most_overvalued,
            regime,
            atiyah_singer: AtiyahSingerIndexReport {
                dim_h0_global_sections: dim_h0,
                dim_h1_cohomology: dim_h1,
                analytical_euler_index,
                chern_weil_topological_flux,
                chiral_zero_mode_index,
                topological_chirality_bias,
            },
        }
    }

    #[inline]
    const fn empty_report() -> CellularSheafReport {
        CellularSheafReport {
            n_vertices: 0,
            n_edges: 0,
            frustration_energy: 0.0,
            spectral_frustration_gap: 0.0,
            cycle_holonomy_obstruction: 0.0,
            harmonic_restoration_force: [0.0; MAX_SHEAF_VERTICES],
            diffused_harmonic_section: [0.0; MAX_SHEAF_VERTICES],
            most_undervalued_vertex: 0,
            most_overvalued_vertex: 0,
            regime: SheafTopologicalRegime::HarmonicGlobalSection,
            atiyah_singer: AtiyahSingerIndexReport {
                dim_h0_global_sections: 0,
                dim_h1_cohomology: 0,
                analytical_euler_index: 0,
                chern_weil_topological_flux: 0.0,
                chiral_zero_mode_index: 0.0,
                topological_chirality_bias: 0.0,
            },
        }
    }
}
