//! # Ola Ω141 — Conjetura de Birch y Swinnerton-Dyer (BSD): Curvas Elípticas de Weierstrass, Traza de Frobenius y Rango Analítico de $L(E, s)$ en Stack Zero-Alloc (< 1 µs)
//!
//! Implementación 100% Rust, stack zero-alloc (`O(1)` memoria, `< 1 µs` latencia) de la
//! **Conjetura de Birch y Swinnerton-Dyer (Problema del Milenio)** aplicada a la geometría
//! aritmética del espacio de fases de precio-flujo en el universo continuo temporal-espectral.
//!
//! ## Fundamento Matemático y Físico
//!
//! 1. **Curva Elíptica en Forma Corta de Weierstrass ($E: y^2 = x^3 + a x + b$):**
//!    En dinámica hamiltoniana no lineal de solitones de libro de órdenes (ecuación de
//!    Korteweg-de Vries / función $\wp$ de Weierstrass), la coordenada potencial $x$
//!    (desplazamiento normalizado de precio) y el momento cinético $y$ (flujo direccional L2/L3)
//!    yacen sobre una curva algebraica cúbica:
//!    $$E: \quad y^2 = x^3 + a x + b$$
//!    con discriminante modular:
//!    $$\Delta = -16 \left( 4 a^3 + 27 b^2 \right)$$
//!    Cuando $\Delta \to 0$, la curva degenera en una singularidad nodal o cuspidal
//!    (separatriz crítica de colapso de liquidez).
//!
//! 2. **Reducción sobre Cuerpos Finitos $\mathbb{F}_p$, Criterio de Euler y Traza de Frobenius ($a_p$):**
//!    Para los primeros $K = 8$ primos impares $p \in \{3, 5, 7, 11, 13, 17, 19, 23\}$,
//!    se cuentan exactamente los puntos proyectivos $N_p = \# E(\mathbb{F}_p)$ mediante el
//!    símbolo de Legendre $\left(\frac{x^3 + A_p x + B_p}{p}\right)$ en aritmética entera
//!    en stack (98 iteraciones totales, 0 asignaciones en heap):
//!    $$a_p = p + 1 - N_p, \quad |a_p| \le 2\sqrt{p} \quad (\text{Cota de Riemann-Hasse})$$
//!
//! 3. **Función $L$ de Hasse-Weil $L(E, s)$ en $s = 1$ y Rango Analítico de Birch-Swinnerton-Dyer:**
//!    $$\lim_{s \to 1} \frac{L(E, s)}{(s - 1)^r} = \frac{\Omega_E \cdot \text{Reg}_E \cdot |\Sha(E)| \cdot \prod c_p}{|E(\mathbb{Q})_{\text{tors}}|^2}$$
//!    Un exceso sistemático de puntos racionales/modulares ($N_p > p + 1$, es decir $a_p < 0$)
//!    indica generadores libres de orden infinito en el grupo de Mordell-Weil $E(\mathbb{Q})$
//!    ($r_{\text{an}} > 0$), lo que en microestructura corresponde a **generadores persistentes
//!    de liquidez multiescala** que sostienen trayectorias sin colapsar en torsión efímera.

/// Número de primos impares usados en la reducción modular de Hasse-Weil en stack.
pub const BSD_NUM_PRIMES: usize = 8;

/// Primeros 8 primos impares para la reducción de la curva elíptica $E(\mathbb{F}_p)$.
pub const BSD_ODD_PRIMES: [i64; BSD_NUM_PRIMES] = [3, 5, 7, 11, 13, 17, 19, 23];

/// Capacidad de la ventana circular en stack para ajustar invariantes de Weierstrass $(a, b)$.
pub const BSD_WINDOW_CAP: usize = 16;

/// Estado inmutable y stack-allocated del diagnóstico aritmético-modular de Birch y Swinnerton-Dyer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BirchSwinnertonDyerMetrics {
    /// Coeficiente lineal de Weierstrass $a$ en $y^2 = x^3 + a x + b$.
    pub weierstrass_a: f64,
    /// Coeficiente constante de Weierstrass $b$ en $y^2 = x^3 + a x + b$.
    pub weierstrass_b: f64,
    /// Discriminante modular $\Delta = -16(4a^3 + 27b^2)$.
    pub discriminant: f64,
    /// Invariante modular $j(E)$ regularizado.
    pub j_invariant: f64,
    /// Producto truncado de Euler de la función $L_K(E, 1) = \prod_{p} \frac{p}{N_p}$.
    pub l_value_at_one: f64,
    /// Traza media normalizada de Hasse-Frobenius $\frac{1}{K}\sum \frac{a_p}{2\sqrt{p}} \in [-1.0, 1.0]$.
    pub frobenius_trace_norm: f64,
    /// Proxy continuo del rango analítico de Birch-Swinnerton-Dyer $r_{\text{BSD}} \in [0.0, 2.5]$.
    pub analytic_rank_proxy: f64,
    /// Salud topológica de no-singularidad $|\Delta|$-suavizada en $[0.0, 1.0]$ ($0.0 =$ cúspide/nodo singular).
    pub non_singularity: f64,
    /// Corriente restauradora elíptica sobre la variedad en $[-1.0, 1.0]$.
    pub bsd_restoring_current: f64,
    /// Modulador continuo $C^1$ Hermite sobre la convicción en $[0.30, 1.15]$.
    pub bsd_conviction_modulation: f64,
}

impl Default for BirchSwinnertonDyerMetrics {
    #[inline]
    fn default() -> Self {
        Self {
            weierstrass_a: -1.0,
            weierstrass_b: 1.0,
            discriminant: -368.0,
            j_invariant: -300.52,
            l_value_at_one: 1.0,
            frobenius_trace_norm: 0.0,
            analytic_rank_proxy: 1.0,
            non_singularity: 1.0,
            bsd_restoring_current: 0.0,
            bsd_conviction_modulation: 1.0,
        }
    }
}

/// Interpolación cúbica de Hermite $C^1([0, 1])$: $H(t) = 3t^2 - 2t^3$.
#[inline]
pub fn bsd_hermite_c1(x: f64, edge0: f64, edge1: f64) -> f64 {
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

/// Exponenciación modular rápida $(base^{exp}) \pmod m$ en stack ($O(\log \text{exp})$).
#[inline]
pub const fn mod_pow_i64(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
    if modulus <= 1 {
        return 0;
    }
    base = base.rem_euclid(modulus);
    let mut result = 1_i64;
    while exp > 0 {
        if (exp & 1) == 1 {
            result = (result * base).rem_euclid(modulus);
        }
        base = (base * base).rem_euclid(modulus);
        exp >>= 1;
    }
    result
}

/// Símbolo de Legendre $\left(\frac{n}{p}\right) \in \{-1, 0, 1\}$ por el criterio de Euler:
/// $n^{(p-1)/2} \pmod p$.
#[inline]
pub const fn legendre_symbol(n: i64, p: i64) -> i64 {
    let rem = n.rem_euclid(p);
    if rem == 0 {
        return 0;
    }
    let euler = mod_pow_i64(rem, (p - 1) / 2, p);
    if euler == 1 {
        1
    } else {
        -1
    }
}

/// Cuenta el número exacto de puntos proyectivos $N_p = \# E(\mathbb{F}_p)$ sobre la curva
/// $y^2 \equiv x^3 + a x + b \pmod p$ (incluyendo el punto en el infinito $\mathcal{O}$) y
/// devuelve la traza de Frobenius $a_p = p + 1 - N_p$.
#[inline]
pub const fn frobenius_trace_mod_p(a_int: i64, b_int: i64, p: i64) -> (i64, i64) {
    let a_mod = a_int.rem_euclid(p);
    let b_mod = b_int.rem_euclid(p);
    // Punto en el infinito O = (0 : 1 : 0)
    let mut n_p = 1_i64;
    let mut x = 0_i64;
    while x < p {
        let x2 = (x * x).rem_euclid(p);
        let x3 = (x2 * x).rem_euclid(p);
        let rhs = (x3 + a_mod * x + b_mod).rem_euclid(p);
        let leg = legendre_symbol(rhs, p);
        n_p += 1 + leg;
        x += 1;
    }
    let a_p = p + 1 - n_p;
    (n_p, a_p)
}

/// Motor aritmético-modular zero-alloc de la Conjetura de Birch y Swinnerton-Dyer.
#[derive(Debug, Clone, Copy)]
pub struct BirchSwinnertonDyerEngine {
    x_ring: [f64; BSD_WINDOW_CAP],
    y_ring: [f64; BSD_WINDOW_CAP],
    count: usize,
    head: usize,
    last_metrics: BirchSwinnertonDyerMetrics,
}

impl Default for BirchSwinnertonDyerEngine {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BirchSwinnertonDyerEngine {
    #[inline]
    pub const fn new() -> Self {
        Self {
            x_ring: [0.0; BSD_WINDOW_CAP],
            y_ring: [0.0; BSD_WINDOW_CAP],
            count: 0,
            head: 0,
            last_metrics: BirchSwinnertonDyerMetrics {
                weierstrass_a: -1.0,
                weierstrass_b: 1.0,
                discriminant: -368.0,
                j_invariant: -300.52,
                l_value_at_one: 1.0,
                frobenius_trace_norm: 0.0,
                analytic_rank_proxy: 1.0,
                non_singularity: 1.0,
                bsd_restoring_current: 0.0,
                bsd_conviction_modulation: 1.0,
            },
        }
    }

    #[inline]
    pub const fn current_metrics(&self) -> BirchSwinnertonDyerMetrics {
        self.last_metrics
    }

    /// Registra un punto del espacio de fases $(x_t, y_t)$ donde $x_t$ es el desplazamiento
    /// potencial de precio y $y_t$ es el impulso de orden-flujo (OFI / velocidad), ajusta
    /// los coeficientes de Weierstrass $(a, b)$ en stack y evalúa los invariantes BSD.
    #[inline]
    pub fn observe_phase_point(&mut self, x_disp: f64, y_flow: f64) -> BirchSwinnertonDyerMetrics {
        if x_disp.is_finite() && y_flow.is_finite() {
            let x_clamped = x_disp.clamp(-4.0, 4.0);
            let y_clamped = y_flow.clamp(-4.0, 4.0);
            self.x_ring[self.head] = x_clamped;
            self.y_ring[self.head] = y_clamped;
            self.head = (self.head + 1) % BSD_WINDOW_CAP;
            if self.count < BSD_WINDOW_CAP {
                self.count += 1;
            }
        }
        if self.count < 4 {
            return self.last_metrics;
        }

        // Ajuste lineal L2 regularizado de w_k = y_k^2 - x_k^3 = a * x_k + b en stack
        let n = self.count as f64;
        let mut sum_x = 0.0_f64;
        let mut sum_w = 0.0_f64;
        for i in 0..self.count {
            let xk = self.x_ring[i];
            let yk = self.y_ring[i];
            let wk = yk * yk - xk * xk * xk;
            sum_x += xk;
            sum_w += wk;
        }
        let mean_x = sum_x / n;
        let mean_w = sum_w / n;

        let mut var_x = 0.0_f64;
        let mut cov_xw = 0.0_f64;
        for i in 0..self.count {
            let xk = self.x_ring[i];
            let yk = self.y_ring[i];
            let wk = yk * yk - xk * xk * xk;
            let dx = xk - mean_x;
            var_x += dx * dx;
            cov_xw += dx * (wk - mean_w);
        }

        let a = (cov_xw / (var_x + 0.05)).clamp(-6.0, 6.0);
        let b = (mean_w - a * mean_x).clamp(-6.0, 6.0);
        let latest_idx = if self.head == 0 {
            self.count - 1
        } else {
            self.head - 1
        };
        let x_now = self.x_ring[latest_idx];
        let y_now = self.y_ring[latest_idx];

        let metrics = Self::evaluate_weierstrass_curve(a, b, x_now, y_now);
        self.last_metrics = metrics;
        metrics
    }

    /// Evalúa directamente una curva elíptica $E: y^2 = x^3 + a x + b$ y el punto $(x, y)$
    /// del espacio de fases en stack (`zero-alloc`).
    #[inline]
    pub fn evaluate_weierstrass_curve(
        a: f64,
        b: f64,
        x_phase: f64,
        y_phase: f64,
    ) -> BirchSwinnertonDyerMetrics {
        if !a.is_finite() || !b.is_finite() {
            return BirchSwinnertonDyerMetrics::default();
        }
        let a_safe = a.clamp(-10.0, 10.0);
        let b_safe = b.clamp(-10.0, 10.0);

        // 1. Discriminante modular Delta = -16 * (4 a^3 + 27 b^2) e invariante j(E)
        let inner = 4.0 * a_safe * a_safe * a_safe + 27.0 * b_safe * b_safe;
        let discriminant = -16.0 * inner;
        let denom = if inner.abs() < 1e-9 {
            if inner >= 0.0 { 1e-9 } else { -1e-9 }
        } else {
            inner
        };
        let j_invariant = (1728.0 * 4.0 * a_safe.powi(3) / denom).clamp(-1e6, 1e6);

        // Salud topológica de no-singularidad: Delta = 0 es una cúspide o nodo singular
        let non_singularity = bsd_hermite_c1(discriminant.abs(), 1e-3, 16.0);

        // 2. Cuantización entera a cuerpo finito F_p y cálculo exacto de trazas de Frobenius a_p
        let a_int = (a_safe * 5.0).round() as i64;
        let b_int = (b_safe * 5.0).round() as i64;

        let mut l_prod = 1.0_f64;
        let mut sum_norm_trace = 0.0_f64;
        let mut sum_birch_mestre = 0.0_f64;
        let mut sum_ln_p = 0.0_f64;

        for &p in BSD_ODD_PRIMES.iter() {
            let (n_p, a_p) = frobenius_trace_mod_p(a_int, b_int, p);
            let p_f = p as f64;
            let n_p_safe = (n_p as f64).max(1.0);
            // Factor local de Euler L_p(E, 1) = p / N_p
            l_prod *= p_f / n_p_safe;

            // Cota de Hasse: |a_p| <= 2 * sqrt(p)
            let hasse_bound = 2.0 * p_f.sqrt();
            let norm_ap = (a_p as f64 / hasse_bound).clamp(-1.0, 1.0);
            sum_norm_trace += norm_ap;

            // Suma de Birch-Goldfeld-Mestre: cuando N_p > p + 1 (a_p < 0), el rango analítico crece
            let ln_p = p_f.ln();
            sum_birch_mestre += (-norm_ap) * ln_p;
            sum_ln_p += ln_p;
        }

        let l_value_at_one = l_prod.clamp(0.05, 20.0);
        let frobenius_trace_norm = (sum_norm_trace / (BSD_NUM_PRIMES as f64)).clamp(-1.0, 1.0);

        // 3. Proxy continuo del rango analítico de Birch y Swinnerton-Dyer r_BSD:
        // Combina la pendiente logarítmica de Birch-Mestre y el anulamiento de L_K(E, 1) -> 0
        let mestre_slope = if sum_ln_p > 0.0 {
            sum_birch_mestre / sum_ln_p
        } else {
            0.0
        };
        let vanishing_order = (1.0 / (l_value_at_one + 0.25)).clamp(0.0, 2.5);
        let analytic_rank_proxy = (non_singularity
            * (0.60 * vanishing_order + 0.80 * (mestre_slope + 0.5).max(0.0)))
        .clamp(0.0, 2.5);

        // 4. Corriente restauradora elíptica sobre la curva E: y^2 - x^3 - a x - b = 0
        let x_s = if x_phase.is_finite() {
            x_phase.clamp(-4.0, 4.0)
        } else {
            0.0
        };
        let y_s = if y_phase.is_finite() {
            y_phase.clamp(-4.0, 4.0)
        } else {
            0.0
        };
        let potential_grad = -(3.0 * x_s * x_s + a_safe) * 0.15 + y_s * 0.45;
        let bsd_restoring_current = (potential_grad.tanh() * non_singularity).clamp(-1.0, 1.0);

        // 5. Modulador continuo C^1 Hermite de convicción en [0.30, 1.15]:
        // Si la curva colapsa a una singularidad (Delta -> 0), atenúa a 0.30;
        // si es no-singular con rango analítico positivo (generadores persistentes), bonifica hasta 1.15.
        let rank_health = bsd_hermite_c1(analytic_rank_proxy, 0.15, 1.25);
        let combined_health = (0.55 * non_singularity + 0.45 * rank_health).clamp(0.0, 1.0);
        let smooth_mod = bsd_hermite_c1(combined_health, 0.0, 1.0);
        let bsd_conviction_modulation = (0.30 + 0.85 * smooth_mod).clamp(0.30, 1.15);

        BirchSwinnertonDyerMetrics {
            weierstrass_a: a_safe,
            weierstrass_b: b_safe,
            discriminant,
            j_invariant,
            l_value_at_one,
            frobenius_trace_norm,
            analytic_rank_proxy,
            non_singularity,
            bsd_restoring_current,
            bsd_conviction_modulation,
        }
    }
}
