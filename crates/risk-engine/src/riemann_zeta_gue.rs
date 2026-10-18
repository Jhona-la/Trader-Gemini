//! # Ola Ω140 — Espectro de Ceros de la Función Zeta de Riemann y Teoría Cuántica de Matrices Aleatorias (GUE / Montgomery-Odlyzko)
//!
//! Implementación 100% Rust, stack zero-alloc (`O(1)` memoria, `< 1 µs` latencia) de la
//! correspondencia **Hilbert-Pólya / Montgomery-Odlyzko** entre los ceros no triviales de la
//! función Zeta de Riemann en la línea crítica $\Re(s) = \frac{1}{2}$ ($\rho_n = \frac{1}{2} + i\gamma_n$)
//! y el **Gaussian Unitary Ensemble (GUE, $\beta = 2$)** de la física cuántica caótica.
//!
//! ## Fundamento Matemático y Físico
//!
//! 1. **Despliegue Espectral (Spectral Unfolding) y Espaciamiento Normalizado ($s_i$):**
//!    Dado un espectro ordenado de autovalores o niveles de energía microestructurales
//!    $\lambda_1 \le \lambda_2 \le \dots \le \lambda_N$ ($N \ge 3$), el espaciamiento medio es
//!    $\bar{\Delta}\lambda = (\lambda_N - \lambda_1) / (N - 1)$ y los espaciamientos desplegados son:
//!    $$s_i = \frac{\lambda_{i+1} - \lambda_i}{\bar{\Delta}\lambda}, \quad \langle s \rangle = 1$$
//!
//! 2. **Ley de Correlación de Pares de Montgomery-Odlyzko ($R_2(u)$):**
//!    $$R_2(u) = 1 - \left(\frac{\sin(\pi u)}{\pi u}\right)^2$$
//!    Cuantifica la **repulsión cuadrática de niveles** ($R_2(u) \approx \frac{\pi^2}{3} u^2 \to 0$ cuando $u \to 0$).
//!
//! 3. **Wigner Surmise para GUE ($\beta = 2$, Simetría Unitaria con Ruptura de Inversión Temporal):**
//!    $$P_{\text{GUE}}(s) = \frac{32}{\pi^2} s^2 \exp\left(-\frac{4}{\pi} s^2\right)$$
//!    frente a la distribución integrable de Poisson $P_{\text{Poisson}}(s) = e^{-s}$.
//!
//! 4. **Razón de Brechas Adyacentes de Oganesyan-Huse ($r_i \in [0, 1]$):**
//!    $$r_i = \frac{\min(\Delta\lambda_{i+1}, \Delta\lambda_i)}{\max(\Delta\lambda_{i+1}, \Delta\lambda_i)}$$
//!    Invariante ante la densidad local del espectro, con valores analíticos exactos:
//!    - Colapso degenerado / contagio sistémico: $\langle r \rangle \to 0$
//!    - Integrable de Poisson ($\beta = 0$): $\langle r \rangle_{\text{Poisson}} = 2\ln 2 - 1 \approx 0.3862943611$
//!    - Caos Cuántico Unitario GUE / Línea Crítica de Riemann ($\beta = 2$):
//!      $$\langle r \rangle_{\text{GUE}} = \frac{2\sqrt{3}}{\pi} - \frac{1}{2} \approx 0.60265779$$
//!
//! 5. **Rigidez Espectral de Dyson-Mehta ($\Delta_3$) y Modulación Continua $C^1$ Hermite:**
//!    Evalúa la varianza de largo alcance de la escalera espectral desplegada y produce un
//!    modulador continuo $C^1$ Hermite `gue_conviction_modulation` $\in [0.25, 1.15]$ que protege
//!    el capital micro ($13 USD) ante singularidades de correlación (clustering degenerado) y
//!    potencia la tasa de crecimiento compuesto $g = \frac{\ln 2}{3}\text{ d}^{-1}$ cuando la repulsión
//!    cuántica de niveles se alinea con la línea crítica $\Re(s) = \frac{1}{2}$.

use std::f64::consts::PI;

/// Dimensión máxima de niveles espectrales procesados en stack sin asignaciones en heap.
pub const MAX_RIEMANN_GUE_LEVELS: usize = 16;

/// Esperanza teórica exacta de la razón de brechas adyacentes para Poisson ($\beta = 0$): $2 \ln 2 - 1$.
pub const POISSON_GAP_RATIO_MEAN: f64 = 0.386_294_361_119_890_6;

/// Esperanza teórica exacta de Wigner-GUE ($\beta = 2$ / Ceros de Riemann): $\frac{2\sqrt{3}}{\pi} - \frac{1}{2}$.
pub const GUE_RIEMANN_GAP_RATIO_MEAN: f64 = 0.602_657_790_843_588_5;

/// Resultado inmutable y stack-allocated del diagnóstico espectral Riemann Zeta / GUE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiemannZetaGueState {
    /// Número de niveles espectrales válidos evaluados ($N \in [0, 16]$).
    pub num_levels: usize,
    /// Razón media de brechas adyacentes de Oganesyan-Huse $\langle r \rangle \in [0.0, 1.0]$.
    pub mean_gap_ratio: f64,
    /// Densidad media de repulsión de niveles Wigner GUE $\langle P_{\text{GUE}}(s) \rangle$.
    pub gue_level_repulsion: f64,
    /// Correlación de pares media de Montgomery-Odlyzko $\langle R_2(s) \rangle \in [0.0, 1.0]$.
    pub montgomery_pair_correlation: f64,
    /// Rigidez espectral de Dyson-Mehta $\Delta_3 \ge 0.0$ sobre la escalera desplegada.
    pub dyson_mehta_rigidity: f64,
    /// Estabilidad en la línea crítica de Riemann $\Re(s) = \frac{1}{2}$ en $[0.0, 1.0]$.
    /// $0.0$ indica degeneración/colapso de niveles (contagio sistémico);
    /// $1.0$ indica repulsión cuántica GUE óptima en la línea crítica.
    pub critical_line_stability: f64,
    /// Factor de modulación continua $C^1$ Hermite sobre la convicción/sizing en $[0.25, 1.15]$.
    pub gue_conviction_modulation: f64,
}

impl Default for RiemannZetaGueState {
    #[inline]
    fn default() -> Self {
        Self {
            num_levels: 0,
            mean_gap_ratio: GUE_RIEMANN_GAP_RATIO_MEAN,
            gue_level_repulsion: 0.0,
            montgomery_pair_correlation: 1.0,
            dyson_mehta_rigidity: 0.0,
            critical_line_stability: 0.5,
            gue_conviction_modulation: 1.0,
        }
    }
}

/// Interpolación polinómica cúbica de Hermite $C^1([0, 1])$ con derivadas nulas en los extremos:
/// $H(t) = 3t^2 - 2t^3$.
#[inline]
pub fn hermite_smoothstep_c1(x: f64, edge0: f64, edge1: f64) -> f64 {
    if !x.is_finite() || !edge0.is_finite() || !edge1.is_finite() {
        return 0.0;
    }
    let span = edge1 - edge0;
    if span.abs() <= 1e-15 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Kernel analítico de correlación de pares de Montgomery-Odlyzko en la línea crítica $\Re(s) = \frac{1}{2}$:
/// $$R_2(u) = 1 - \left(\frac{\sin(\pi u)}{\pi u}\right)^2$$
/// Usa expansión de Taylor de sexto orden cerca de $u = 0$ para estabilidad numérica exacta.
#[inline]
pub fn montgomery_odlyzko_pair_correlation(u: f64) -> f64 {
    if !u.is_finite() {
        return 0.0;
    }
    let u_abs = u.abs();
    if u_abs < 1e-4 {
        // sin(x)/x = 1 - x^2/6 + x^4/120 => 1 - (sin(x)/x)^2 = x^2/3 - 2*x^4/45
        let x2 = (PI * u_abs) * (PI * u_abs);
        return (x2 / 3.0 - (2.0 * x2 * x2) / 45.0).clamp(0.0, 1.0);
    }
    let pi_u = PI * u_abs;
    let sinc = pi_u.sin() / pi_u;
    (1.0 - sinc * sinc).clamp(0.0, 1.0)
}

/// Densidad de probabilidad de Wigner Surmise para el Gaussian Unitary Ensemble ($\beta = 2$):
/// $$P_{\text{GUE}}(s) = \frac{32}{\pi^2} s^2 \exp\left(-\frac{4}{\pi} s^2\right)$$
#[inline]
pub fn wigner_gue_density(s: f64) -> f64 {
    if !s.is_finite() || s <= 0.0 {
        return 0.0;
    }
    let s2 = s * s;
    let coeff = 32.0 / (PI * PI);
    let exponent = -(4.0 / PI) * s2;
    (coeff * s2 * exponent.exp()).max(0.0)
}

/// Motor cuántico zero-alloc del espectro de Riemann Zeta / GUE ($\beta = 2$).
#[derive(Debug, Clone, Copy)]
pub struct RiemannZetaGueEngine {
    /// Ventana circular stack-allocated de los últimos niveles espectrales/energéticos observados.
    levels_ring: [f64; MAX_RIEMANN_GUE_LEVELS],
    count: usize,
    head: usize,
    last_state: RiemannZetaGueState,
}

impl Default for RiemannZetaGueEngine {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl RiemannZetaGueEngine {
    #[inline]
    pub const fn new() -> Self {
        Self {
            levels_ring: [0.0; MAX_RIEMANN_GUE_LEVELS],
            count: 0,
            head: 0,
            last_state: RiemannZetaGueState {
                num_levels: 0,
                mean_gap_ratio: GUE_RIEMANN_GAP_RATIO_MEAN,
                gue_level_repulsion: 0.0,
                montgomery_pair_correlation: 1.0,
                dyson_mehta_rigidity: 0.0,
                critical_line_stability: 0.5,
                gue_conviction_modulation: 1.0,
            },
        }
    }

    /// Devuelve el último estado espectral calculado.
    #[inline]
    pub const fn last_state(&self) -> RiemannZetaGueState {
        self.last_state
    }

    /// Registra un nuevo nivel de energía / autovalor efectivo en el anillo stack y actualiza el espectro.
    #[inline]
    pub fn observe_energy_level(&mut self, energy_level: f64) -> RiemannZetaGueState {
        if energy_level.is_finite() {
            self.levels_ring[self.head] = energy_level;
            self.head = (self.head + 1) % MAX_RIEMANN_GUE_LEVELS;
            if self.count < MAX_RIEMANN_GUE_LEVELS {
                self.count += 1;
            }
        }
        if self.count < 4 {
            return self.last_state;
        }
        let state = Self::evaluate_spectrum(&self.levels_ring[..self.count]);
        self.last_state = state;
        state
    }

    /// Evalúa un conjunto de autovalores o niveles de energía multiescala directamente en stack (`zero-alloc`).
    /// Requiere al menos 3 niveles finitos para definir brechas adyacentes desplegadas.
    #[inline]
    pub fn evaluate_spectrum(raw_levels: &[f64]) -> RiemannZetaGueState {
        let mut sorted = [0.0_f64; MAX_RIEMANN_GUE_LEVELS];
        let mut n = 0_usize;
        for &val in raw_levels.iter() {
            if val.is_finite() && n < MAX_RIEMANN_GUE_LEVELS {
                sorted[n] = val;
                n += 1;
            }
        }
        if n < 3 {
            return RiemannZetaGueState::default();
        }

        // Ordenamiento por inserción en stack O(N^2) para N <= 16 (ultrarrápido, sin ramas complejas ni heap)
        let mut i = 1;
        while i < n {
            let key = sorted[i];
            let mut j = i;
            while j > 0 && sorted[j - 1] > key {
                sorted[j] = sorted[j - 1];
                j -= 1;
            }
            sorted[j] = key;
            i += 1;
        }

        let total_span = sorted[n - 1] - sorted[0];
        let num_gaps = n - 1;
        let mean_spacing = total_span / (num_gaps as f64);

        // Si todos los niveles colapsan en un único punto degenerado (total_span ~ 0):
        // Degeneración total -> repulsión nula, estabilidad crítica mínima (0.0), freno protector 0.25.
        if !mean_spacing.is_finite() || mean_spacing <= 1e-12 {
            return RiemannZetaGueState {
                num_levels: n,
                mean_gap_ratio: 0.0,
                gue_level_repulsion: 0.0,
                montgomery_pair_correlation: 0.0,
                dyson_mehta_rigidity: 1.0,
                critical_line_stability: 0.0,
                gue_conviction_modulation: 0.25,
            };
        }

        // 1. Espaciamientos desplegados s_i = (lambda_{i+1} - lambda_i) / mean_spacing
        let mut spacings = [0.0_f64; MAX_RIEMANN_GUE_LEVELS];
        let mut sum_gue_density = 0.0_f64;
        let mut sum_montgomery = 0.0_f64;

        for k in 0..num_gaps {
            let s_k = ((sorted[k + 1] - sorted[k]) / mean_spacing).max(0.0);
            spacings[k] = s_k;
            sum_gue_density += wigner_gue_density(s_k);
            sum_montgomery += montgomery_odlyzko_pair_correlation(s_k);
        }

        let gue_level_repulsion = sum_gue_density / (num_gaps as f64);
        let montgomery_pair_correlation = (sum_montgomery / (num_gaps as f64)).clamp(0.0, 1.0);

        // 2. Razón de brechas adyacentes de Oganesyan-Huse r_k = min(s_{k+1}, s_k) / max(s_{k+1}, s_k)
        // ponderada por el factor continuo de desagrupamiento 1 - exp(-max(s_k, s_{k+1}) / 0.10)
        // para que niveles colapsados en degeneración (s_k -> 0) no fabriquen r_k = 1.0 por ruido numérico.
        let mut sum_gap_ratio = 0.0_f64;
        let num_ratios = num_gaps - 1;
        for k in 0..num_ratios {
            let g1 = spacings[k];
            let g2 = spacings[k + 1];
            let hi = g1.max(g2);
            let lo = g1.min(g2);
            if hi > 1e-12 {
                let unclustered_weight = 1.0 - (-hi / 0.10).exp();
                sum_gap_ratio += ((lo / hi) * unclustered_weight).clamp(0.0, 1.0);
            }
        }
        let mean_gap_ratio = (sum_gap_ratio / (num_ratios as f64)).clamp(0.0, 1.0);

        // 2.5 Varianza empírica de espaciamientos desplegados Var(s) = <(s_k - 1)^2>.
        // En Wigner-Dyson GUE (beta=2), sigma_GUE(s) = sqrt(3*pi/8 - 1) ~ 0.4220, mientras que en un
        // cristal rígido ("picket fence") o rampa monótona uniforme sigma(s) -> 0 y <r> -> 1.
        let mut sum_sq_spacing_dev = 0.0_f64;
        for &s_k in spacings.iter().take(num_gaps) {
            let ds = s_k - 1.0;
            sum_sq_spacing_dev += ds * ds;
        }
        let spacing_std = (sum_sq_spacing_dev / (num_gaps as f64)).sqrt();

        // 3. Rigidez espectral de Dyson-Mehta Delta_3 sobre las posiciones desplegadas u_k in [0, N-1]
        // Compara la posición desplegada u_k = (sorted[k] - sorted[0]) / mean_spacing con la recta ideal k.
        let mut sum_sq_dev = 0.0_f64;
        for (k, &val) in sorted.iter().take(n).enumerate() {
            let u_k = (val - sorted[0]) / mean_spacing;
            let dev = u_k - (k as f64);
            sum_sq_dev += dev * dev;
        }
        let dyson_mehta_rigidity = sum_sq_dev / (n as f64);

        // 4. Síntesis Continua C^1 de Estabilidad en la Línea Crítica de Riemann Re(s) = 1/2 (G8-OMEGA-08):
        // Ventana unimodal C^1 alrededor de <r>_GUE = 0.6027:
        // - Sube desde degeneración/Poisson (<r> in [0.12, 0.6027])
        // - Cae suavemente ante peines regulares / rampas monótonas (<r> in [0.72, 0.90] -> r -> 1)
        let gap_rise = hermite_smoothstep_c1(
            mean_gap_ratio,
            0.12,
            GUE_RIEMANN_GAP_RATIO_MEAN,
        );
        let picket_decay = 1.0 - hermite_smoothstep_c1(mean_gap_ratio, 0.72, 0.90);
        let crystal_rigidity = 1.0 - hermite_smoothstep_c1(spacing_std, 0.06, 0.24);
        let gap_stability = gap_rise * picket_decay;

        // - Componente B: Densidad de repulsión de Wigner GUE P_GUE(s) (modo ~ 0.59 en s = sqrt(pi)/2)
        let gue_repulsion_stability = hermite_smoothstep_c1(gue_level_repulsion, 0.10, 0.48);
        // - Componente C: Repulsión de pares de Montgomery-Odlyzko R_2(s) (óptimo cuando R_2 >= 0.75)
        let montgomery_stability = hermite_smoothstep_c1(montgomery_pair_correlation, 0.15, 0.80);
        // - Componente D: Rigidez espectral de Dyson-Mehta acotada (si Delta_3 >> 1, hay concentración patológica)
        let rigidity_health = 1.0 - hermite_smoothstep_c1(dyson_mehta_rigidity, 0.15, 2.50);

        let raw_stability = (0.30 * gap_stability
            + 0.30 * gue_repulsion_stability
            + 0.25 * montgomery_stability
            + 0.15 * rigidity_health)
            .clamp(0.0, 1.0);

        // El filtro anti-cristal C^1 evita que un peine equiespaciado (r -> 1, sigma(s) -> 0) o una
        // EMA monótona reciba bonificación GUE: sólo el caos cuántico unitario (repulsión con fluctuación
        // de Wigner) preserva `anti_crystal_gate = 1.0`.
        let anti_crystal_gate =
            (0.15 + 0.85 * picket_decay) * (1.0 - 0.65 * crystal_rigidity * (1.0 - picket_decay));
        let critical_line_stability = (raw_stability * anti_crystal_gate).clamp(0.0, 1.0);

        // 5. Modulación continua C^1 de convicción:
        // Mapea critical_line_stability in [0.0, 1.0] -> [0.25, 1.15] usando Hermite C^1.
        let smooth_mod = hermite_smoothstep_c1(critical_line_stability, 0.0, 1.0);
        let gue_conviction_modulation = (0.25 + 0.90 * smooth_mod).clamp(0.25, 1.15);

        RiemannZetaGueState {
            num_levels: n,
            mean_gap_ratio,
            gue_level_repulsion,
            montgomery_pair_correlation,
            dyson_mehta_rigidity,
            critical_line_stability,
            gue_conviction_modulation,
        }
    }
}
