//! # Ola Ω143 — Transporte Óptimo de Wasserstein-Kantorovich $W_2$ y Dinámica Hidrodinámica de Benamou-Brenier sobre $L^2(\mathbb{R}^{d \times \tau})$ en Stack Zero-Alloc (< 1 µs)
//!
//! Implementación 100% Rust, stack zero-alloc (`O(1)` memoria, `< 1 µs` latencia) de la
//! métrica geodésica de **Monge-Kantorovich-Wasserstein ($W_2$)**, el **Teorema de Isometría Cuantílica
//! de Brenier-McCann** y la **Formulación Fluida Dinámica de Benamou-Brenier (2000)** sobre el
//! espacio de Hilbert continuo temporal-espectral.
//!
//! ## Fundamento Matemático y Físico
//!
//! 1. **Distancia Cuadrática de Wasserstein-Kantorovich ($W_2$):**
//!    Dadas dos medidas de probabilidad empíricas $\mu_0, \mu_1 \in \mathcal{P}_2(\mathbb{R})$ sobre la
//!    variedad de retornos/flujo espectral, el costo cuadrático mínimo de transporte óptimo es:
//!    $$W_2^2(\mu_0, \mu_1) = \inf_{\pi \in \Pi(\mu_0, \mu_1)} \iint |x - y|^2 \, d\pi(x, y) = \int_0^1 \left| F_{\mu_0}^{-1}(q) - F_{\mu_1}^{-1}(q) \right|^2 dq$$
//!
//! 2. **Fórmula de Acción Cinética de Benamou-Brenier (2000):**
//!    El transporte óptimo $W_2$ equivale a la mínima acción cinética de un fluido compresible
//!    sujeto a la ecuación de continuidad $\partial_t \rho_t + \nabla \cdot (\rho_t \mathbf{v}_t) = 0$:
//!    $$W_2^2(\rho_0, \rho_1) = \inf_{(\rho_t, \mathbf{v}_t)} \int_0^1 \int \rho_t(x) |\mathbf{v}_t(x)|^2 \, dx \, dt$$
//!
//! 3. **Descomposición Geodésica de Bures-Wasserstein y Coherencia de Transporte ($\eta_{\text{OT}}$):**
//!    El cuadrado de la distancia de Wasserstein se descompone ortogonalmente en:
//!    $$W_2^2(\mu_0, \mu_1) = \underbrace{(m_1 - m_0)^2}_{\text{Transporte Advectivo del Baricentro}} + \underbrace{(\sigma_1 - \sigma_0)^2 + \Delta_{\text{forma}}^2}_{\text{Deformación Difusiva / Ruptura de Régimen}}$$
//!    La eficiencia geodésica $\eta_{\text{OT}} = \frac{(m_1 - m_0)^2}{W_2^2 + \epsilon} \in [0, 1]$
//!    distingue un desplazamiento rígido coherente de masa ($\eta_{\text{OT}} \to 1$, tendencia limpia)
//!    de una fractura estructural caótica donde la varianza explota sin transporte neto ($\eta_{\text{OT}} \to 0$).

/// Capacidad de la ventana circular en stack (dividida en mitad de referencia $\mu_0$ y mitad activa $\mu_1$).
pub const WASSERSTEIN_WINDOW_CAP: usize = 16;
/// Tamaño de cada sub-ventana cuantílica en stack ($N/2 = 8$ cuantiles ordenados in-place).
pub const WASSERSTEIN_HALF_CAP: usize = 8;

/// Estado inmutable y stack-allocated del Transporte Óptimo de Wasserstein-Kantorovich $W_2$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WassersteinKantorovichMetrics {
    /// Distancia cuadrática de Wasserstein $W_2(\mu_0, \mu_1) \ge 0.0$.
    pub wasserstein_w2_distance: f64,
    /// Acción cinética mínima de Benamou-Brenier $\mathcal{A}_{\text{BB}} = \frac{1}{2} W_2^2$.
    pub kantorovich_kinetic_action: f64,
    /// Velocidad advectiva del baricentro de Wasserstein $v_{\text{BB}} \in [-1.0, 1.0]$.
    pub barycenter_transport_velocity: f64,
    /// Razón de coherencia geodésica $\eta_{\text{OT}} = \frac{\Delta m^2}{W_2^2 + \epsilon} \in [0.0, 1.0]$.
    pub geodesic_coherence_ratio: f64,
    /// Deformación estructural no advectiva (dispersión de forma y escala) $\ge 0.0$.
    pub shape_deformation_energy: f64,
    /// Modulador continuo $C^1$ Hermite sobre la convicción en $[0.30, 1.15]$.
    pub wasserstein_conviction_modulation: f64,
}

impl Default for WassersteinKantorovichMetrics {
    #[inline]
    fn default() -> Self {
        Self {
            wasserstein_w2_distance: 0.0,
            kantorovich_kinetic_action: 0.0,
            barycenter_transport_velocity: 0.0,
            geodesic_coherence_ratio: 1.0,
            shape_deformation_energy: 0.0,
            wasserstein_conviction_modulation: 1.0,
        }
    }
}

/// Interpolación cúbica de Hermite $C^1([0, 1])$: $H(t) = 3t^2 - 2t^3$.
#[inline]
pub fn wasserstein_hermite_c1(x: f64, edge0: f64, edge1: f64) -> f64 {
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

/// Ordenamiento por inserción in-place en stack (0 asignaciones en heap).
#[inline]
fn sort_slice_in_place(buf: &mut [f64]) {
    let len = buf.len();
    for i in 1..len {
        let key = buf[i];
        let mut j = i;
        while j > 0 && buf[j - 1] > key {
            buf[j] = buf[j - 1];
            j -= 1;
        }
        buf[j] = key;
    }
}

/// Calcula la distancia exacta $W_2(\mu_0, \mu_1)$ entre dos muestras de igual tamaño en stack
/// mediante la isometría cuantílica de Brenier-McCann:
/// $$W_2(\mu_0, \mu_1) = \left( \frac{1}{M} \sum_{k=0}^{M-1} \left| x_{(k)}^{(0)} - x_{(k)}^{(1)} \right|^2 \right)^{1/2}$$
#[inline]
pub fn exact_empirical_w2_distance_8(
    mut sample_a: [f64; WASSERSTEIN_HALF_CAP],
    mut sample_b: [f64; WASSERSTEIN_HALF_CAP],
) -> (f64, f64, f64) {
    for k in 0..WASSERSTEIN_HALF_CAP {
        if !sample_a[k].is_finite() {
            sample_a[k] = 0.0;
        }
        if !sample_b[k].is_finite() {
            sample_b[k] = 0.0;
        }
    }
    sort_slice_in_place(&mut sample_a);
    sort_slice_in_place(&mut sample_b);

    let inv_m = 1.0 / (WASSERSTEIN_HALF_CAP as f64);
    let mut mean_a = 0.0_f64;
    let mut mean_b = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for k in 0..WASSERSTEIN_HALF_CAP {
        mean_a += sample_a[k];
        mean_b += sample_b[k];
        let diff = sample_b[k] - sample_a[k];
        sum_sq += diff * diff;
    }
    mean_a *= inv_m;
    mean_b *= inv_m;

    let w2_sq = sum_sq * inv_m;
    let w2 = w2_sq.sqrt();
    let bary_shift = mean_b - mean_a;
    (w2, w2_sq, bary_shift)
}

/// Motor cuántico-hidrodinámico zero-alloc de Transporte Óptimo de Wasserstein-Kantorovich $W_2$.
#[derive(Debug, Clone, Copy)]
pub struct WassersteinKantorovichEngine {
    ring: [f64; WASSERSTEIN_WINDOW_CAP],
    count: usize,
    head: usize,
    last_metrics: WassersteinKantorovichMetrics,
}

impl Default for WassersteinKantorovichEngine {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl WassersteinKantorovichEngine {
    #[inline]
    pub const fn new() -> Self {
        Self {
            ring: [0.0; WASSERSTEIN_WINDOW_CAP],
            count: 0,
            head: 0,
            last_metrics: WassersteinKantorovichMetrics {
                wasserstein_w2_distance: 0.0,
                kantorovich_kinetic_action: 0.0,
                barycenter_transport_velocity: 0.0,
                geodesic_coherence_ratio: 1.0,
                shape_deformation_energy: 0.0,
                wasserstein_conviction_modulation: 1.0,
            },
        }
    }

    #[inline]
    pub const fn current_metrics(&self) -> WassersteinKantorovichMetrics {
        self.last_metrics
    }

    /// Observa una nueva coordenada espectral combinada (ej. posición relativa + impulso de flujo)
    /// y calcula el transporte óptimo $W_2(\mu_{\text{pasado}}, \mu_{\text{reciente}})$ en tiempo $O(1)$
    /// y 0 asignaciones en heap.
    #[inline]
    pub fn observe_spectral_measure(&mut self, spectral_coord: f64) -> WassersteinKantorovichMetrics {
        if !spectral_coord.is_finite() {
            return self.last_metrics;
        }

        let x_clean = spectral_coord.clamp(-10.0, 10.0);
        self.ring[self.head] = x_clean;
        self.head = (self.head + 1) % WASSERSTEIN_WINDOW_CAP;
        if self.count < WASSERSTEIN_WINDOW_CAP {
            self.count += 1;
        }

        if self.count < WASSERSTEIN_WINDOW_CAP {
            return self.last_metrics;
        }

        // Extraer las dos medidas empíricas cronológicas:
        // mu_0: primeros 8 eventos de la ventana (distribución de referencia)
        // mu_1: últimos 8 eventos de la ventana (distribución actual transportada)
        let mut mu_0 = [0.0_f64; WASSERSTEIN_HALF_CAP];
        let mut mu_1 = [0.0_f64; WASSERSTEIN_HALF_CAP];

        for i in 0..WASSERSTEIN_HALF_CAP {
            let idx_0 = (self.head + i) % WASSERSTEIN_WINDOW_CAP;
            let idx_1 = (self.head + WASSERSTEIN_HALF_CAP + i) % WASSERSTEIN_WINDOW_CAP;
            mu_0[i] = self.ring[idx_0];
            mu_1[i] = self.ring[idx_1];
        }

        let (w2, w2_sq, bary_shift) = exact_empirical_w2_distance_8(mu_0, mu_1);
        let kantorovich_kinetic_action = 0.5 * w2_sq;

        let advective_sq = bary_shift * bary_shift;
        let shape_deformation_energy = (w2_sq - advective_sq).max(0.0);

        // Si el costo total de transporte es casi nulo, la distribución está en equilibrio estacionario
        let geodesic_coherence_ratio = if w2_sq <= 1e-6 {
            1.0
        } else {
            (advective_sq / w2_sq).clamp(0.0, 1.0)
        };

        // Velocidad advectiva de Benamou-Brenier ponderada por coherencia geodésica
        let barycenter_transport_velocity =
            (bary_shift * (0.35 + 0.65 * geodesic_coherence_ratio)).tanh().clamp(-1.0, 1.0);

        // Modulación continua C^1 Hermite de la convicción:
        // - Cuando hay una fractura de régimen caótica (shape_deformation_energy alto y geodesic_coherence_ratio -> 0),
        //   atenúa hacia 0.30.
        // - Cuando hay transporte geodésico coherente (geodesic_coherence_ratio alto y velocidad advectiva clara),
        //   amplifica hasta 1.15.
        let fracture_penalty = wasserstein_hermite_c1(shape_deformation_energy, 0.08, 0.85)
            * (1.0 - geodesic_coherence_ratio);
        let geodesic_bonus = wasserstein_hermite_c1(geodesic_coherence_ratio, 0.45, 0.90)
            * wasserstein_hermite_c1(bary_shift.abs(), 0.05, 0.65);

        let wasserstein_conviction_modulation =
            (1.0 - 0.70 * fracture_penalty + 0.15 * geodesic_bonus).clamp(0.30, 1.15);

        let metrics = WassersteinKantorovichMetrics {
            wasserstein_w2_distance: w2,
            kantorovich_kinetic_action,
            barycenter_transport_velocity,
            geodesic_coherence_ratio,
            shape_deformation_energy,
            wasserstein_conviction_modulation,
        };
        self.last_metrics = metrics;
        metrics
    }
}
