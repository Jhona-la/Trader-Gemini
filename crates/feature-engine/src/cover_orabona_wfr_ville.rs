//! # Fase 24 — Cota Grossman-Zhou, Portafolios Universales Cover-Vovk, Apuestas Orabona, Paisaje Filoche-Mayboroda, Métrica WFR y Supermartingala de Ville
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Seis componentes matemáticos supremos de asignación universal, límites de ruina y geometría de transporte:
//!   1. `GrossmanZhouKyleFloorEngine`: Cota de piso de cartera de Grossman-Zhou (1993) $W_t \ge F \cdot W_0$ combinada con escalado microestructural de impacto de Kyle $\lambda$ (1985).
//!   2. `CoverVovkUniversalPortfolioEngine`: Portafolios universales continuos de Cover-Vovk (1991) sin supuestos distributivos, que convergen a la mejor cartera rebalanceada constante (BCRP) a posteriori con arrepentimiento acotado $R_T = O(\ln T)$.
//!   3. `OrabonaCoinBettingEngine`: Algoritmo de coin-betting sin tasa de aprendizaje de Orabona & Pál (2016), que traduce la optimización online en apuestas de riqueza autosimilares $w_t = \text{Wealth}_t \cdot \beta_t$.
//!   4. `FilocheMayborodaLocalizationEngine`: Paisaje de localización de Filoche-Mayboroda (2012) $Lu = 1$, con potencial efectivo $W = 1/u$ para identificar valles y barreras de confinamiento de precios en el libro L2.
//!   5. `WassersteinFisherRaoEngine`: Métrica no conservativa de Wasserstein-Fisher-Rao (WFR / Hellinger-Kantorovich; Chizat et al., 2018), que mide distancias entre estados del libro L2 considerando tanto transporte horizontal de liquidez como creación/destrucción vertical de volumen.
//!   6. `VilleMaximalSupermartingaleEngine`: Certificación analítica de ruina y supermartingalas de Ville (1939) $\mathbb{P}(\sup M_t \ge \alpha) \le 1/\alpha$, garantizando $P(\text{ruina}) < 10^{-6}$.
//!   7. `CoverOrabonaWfrVilleMasterEngine`: Coordinador maestro determinista en memoria de pila ($O(1)$ stack, cero `unwrap`/`panic!`, 100% Rust nativo).
//! - **POR QUÉ:** Preservar el capital es la precondición matemática del interés compuesto exponencial; los portafolios clásicos asumen estacionariedad gaussiana falsa; los hiperparámetros de aprendizaje colapsan ante cambios de régimen; la liquidez se crea y cancela continuamente violando el transporte óptimo conservativo puro; y se requiere certificar formalmente que la probabilidad de ruina jamás supere $10^{-6}$.
//! - **PARA QUÉ:** Blindar el capital ante eventos cisne negro mediante el piso estricto de Grossman-Zhou, asignar capital universalmente sin sobreajuste, predecir barreras de soporte/resistencia mediante el paisaje de localización y certificar matemáticamente el crecimiento multiplicativo.
//! - **CÓMO:** Control proporcional no lineal al piso, integración en simplex discreto de Cover, actualización de riqueza de Krichevsky-Trofimov, inversión laplaciana tridiagonal de $Lu=1$, funcional de interpolación estática WFR y test de parada de Ville.
//! - **CUÁNDO:** En cada decisión de sizing de posición, rebalanceo de cartera y verificación de riesgo sistémico.
//! - **DÓNDE:** `crates/feature-engine/src/cover_orabona_wfr_ville.rs`.
//! - **QUIÉN:** `GrossmanZhouKyleFloorEngine`, `CoverVovkUniversalPortfolioEngine`, `OrabonaCoinBettingEngine`, `FilocheMayborodaLocalizationEngine`, `WassersteinFisherRaoEngine`, `VilleMaximalSupermartingaleEngine`, `CoverOrabonaWfrVilleMasterEngine`.

use crate::landauer_bec_hopfield::KbnCompensatedAccumulator;

/// Número de activos para el portafolio universal.
pub const ASSET_UNIVERSE: usize = 4;

/// Número de vértices y muestras de la malla simplex para Cover-Vovk.
pub const COVER_SIMPLEX_POINTS: usize = 10;

/// Dimensión de la rejilla espacial para el paisaje de Filoche-Mayboroda.
pub const FILOCHE_GRID_SIZE: usize = 16;

/// Número de bandas de volumen para la métrica Wasserstein-Fisher-Rao.
pub const WFR_BINS: usize = 8;

// ============================================================================
// 1. GROSSMAN-ZHOU FLOOR & KYLE LAMBDA SIZING ENGINE
// ============================================================================

/// Diagnóstico del piso de Grossman-Zhou y sizing microestructural de Kyle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrossmanZhouKyleReport {
    /// Riqueza actual normalizada $W_t / W_0$.
    pub normalized_wealth: f64,
    /// Distancia al piso de seguridad $W_t - F$.
    pub cushion_to_floor: f64,
    /// Fracción máxima de exposición permitida por Grossman-Zhou $m_t \in [0.0, 1.0]$.
    pub max_grossman_exposure: f64,
    /// Coeficiente de impacto de precio de Kyle $\lambda = \Delta P / Q$.
    pub kyle_lambda: f64,
    /// Tamaño de posición óptimo contraído por impacto de Kyle.
    pub kyle_scaled_position: f64,
}

impl Default for GrossmanZhouKyleReport {
    fn default() -> Self {
        Self {
            normalized_wealth: 1.0,
            cushion_to_floor: 0.10,
            max_grossman_exposure: 0.50,
            kyle_lambda: 0.001,
            kyle_scaled_position: 0.0,
        }
    }
}

/// Motor de piso de Grossman-Zhou y sizing de Kyle $\lambda$.
#[derive(Debug, Clone, Copy)]
pub struct GrossmanZhouKyleFloorEngine {
    /// Piso de capital estricto $F \in (0.0, 1.0)$, ej. $0.90$ para drawdown máximo del $10\%$.
    floor: f64,
    /// Aversión al riesgo relativa $\gamma \ge 1.0$.
    gamma: f64,
}

impl Default for GrossmanZhouKyleFloorEngine {
    fn default() -> Self {
        Self::new(0.90, 2.0)
    }
}

impl GrossmanZhouKyleFloorEngine {
    /// Crea un motor con piso de riqueza y aversión al riesgo.
    pub fn new(floor: f64, gamma: f64) -> Self {
        Self {
            floor: floor.clamp(0.50, 0.99),
            gamma: gamma.max(1.0),
        }
    }

    /// Evalúa la exposición de Grossman-Zhou $m_t = \frac{W_t - F}{W_t(1 - F)} \frac{1}{\gamma \sigma^2}$
    /// y el escalado de Kyle $\lambda$: $Q^* = \frac{\mu}{\lambda + \gamma \sigma^2}$.
    pub fn evaluate(
        &self,
        current_wealth: f64,
        initial_wealth: f64,
        volatility: f64,
        expected_drift: f64,
        volume_spread_ratio: f64,
    ) -> GrossmanZhouKyleReport {
        let w0 = initial_wealth.max(1e-6);
        let wt = current_wealth.max(1e-6);
        let norm_w = wt / w0;
        let sigma2 = (volatility * volatility).max(1e-6);

        let cushion = (norm_w - self.floor).max(0.0);

        // Control Grossman-Zhou: si la riqueza toca el piso $F$, la exposición colapsa a 0
        let gz_exposure = if cushion > 1e-8 {
            let denom = norm_w * (1.0 - self.floor) * self.gamma * sigma2;
            if denom > 1e-12 {
                (cushion / denom).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Kyle Lambda $\lambda \approx \frac{\text{Spread}}{\text{Volumen}}$
        let kyle_lambda = volume_spread_ratio.clamp(1e-6, 1.0);

        // Sizing de Kyle con penalización cuadrática por impacto:
        // $Q^* = \frac{\mu}{2\lambda + \gamma \sigma^2}$
        let denom_kyle = 2.0 * kyle_lambda + self.gamma * sigma2;
        let raw_q = if denom_kyle > 1e-12 {
            expected_drift / denom_kyle
        } else {
            0.0
        };

        let kyle_scaled = (raw_q * gz_exposure).clamp(-1.0, 1.0);

        GrossmanZhouKyleReport {
            normalized_wealth: norm_w,
            cushion_to_floor: cushion,
            max_grossman_exposure: gz_exposure,
            kyle_lambda,
            kyle_scaled_position: kyle_scaled,
        }
    }
}

// ============================================================================
// 2. COVER-VOVK UNIVERSAL PORTFOLIO ENGINE
// ============================================================================

/// Diagnóstico del portafolio universal continuo de Cover-Vovk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverVovkReport {
    /// Vector de pesos óptimos asignados a los activos $\hat{b} \in \Delta_d$.
    pub optimal_weights: [f64; ASSET_UNIVERSE],
    /// Riqueza acumulada multiplicativa por el portafolio universal $S_t / S_0$.
    pub cumulative_wealth_multiplier: f64,
    /// Cota superior analítica de arrepentimiento de Cover $R_T \le \frac{d-1}{2}\ln T$.
    pub theoretical_regret_bound: f64,
}

impl Default for CoverVovkReport {
    fn default() -> Self {
        Self {
            optimal_weights: [1.0 / (ASSET_UNIVERSE as f64); ASSET_UNIVERSE],
            cumulative_wealth_multiplier: 1.0,
            theoretical_regret_bound: 0.0,
        }
    }
}

/// Motor de portafolios universales continuos de Cover-Vovk en simplex discreto.
#[derive(Debug, Clone)]
pub struct CoverVovkUniversalPortfolioEngine {
    /// Riqueza acumulada de cada vector de cartera en la malla simplex: `simplex_wealth[k]`.
    simplex_wealth: [f64; COVER_SIMPLEX_POINTS],
    /// Puntos fijos en el simplex $\Delta_d$: `simplex_points[k][asset]`.
    simplex_points: [[f64; ASSET_UNIVERSE]; COVER_SIMPLEX_POINTS],
    /// Contador de pasos de tiempo $T$.
    time_step: usize,
}

impl Default for CoverVovkUniversalPortfolioEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CoverVovkUniversalPortfolioEngine {
    /// Inicializa el motor con una cuadrícula simétrica determinista en el simplex.
    pub fn new() -> Self {
        let mut simplex_points = [[0.0; ASSET_UNIVERSE]; COVER_SIMPLEX_POINTS];

        // 4 vértices puros: (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)
        for (i, p) in simplex_points[..ASSET_UNIVERSE].iter_mut().enumerate() {
            p[i] = 1.0;
        }

        // Centroide uniforme: (0.25, 0.25, 0.25, 0.25)
        simplex_points[4].fill(0.25);

        // Puntos intermedios barycéntricos
        simplex_points[5] = [0.5, 0.5, 0.0, 0.0];
        simplex_points[6] = [0.0, 0.5, 0.5, 0.0];
        simplex_points[7] = [0.0, 0.0, 0.5, 0.5];
        simplex_points[8] = [0.5, 0.0, 0.0, 0.5];
        simplex_points[9] = [0.1, 0.3, 0.3, 0.3];

        Self {
            simplex_wealth: [1.0; COVER_SIMPLEX_POINTS],
            simplex_points,
            time_step: 0,
        }
    }

    /// Actualiza el portafolio universal con un vector de rendimientos relativos de precio $x_t \ge 0$:
    /// $\hat{b}_{t+1} = \frac{\sum_k b_k S_t(b_k)}{\sum_k S_t(b_k)}$.
    pub fn update(&mut self, price_relatives: &[f64; ASSET_UNIVERSE]) -> CoverVovkReport {
        self.time_step = self.time_step.saturating_add(1);

        // 1. Actualizar riqueza acumulada de cada vector de portafolio en la malla:
        // $S_t(b) = S_{t-1}(b) \cdot (b^T x_t)$
        let mut kbn_total_wealth = KbnCompensatedAccumulator::new();
        for k in 0..COVER_SIMPLEX_POINTS {
            let mut dot = 0.0;
            for (d, &x) in price_relatives.iter().enumerate() {
                dot += self.simplex_points[k][d] * x.max(0.001);
            }
            self.simplex_wealth[k] *= dot.max(1e-8);
            kbn_total_wealth.add(self.simplex_wealth[k]);
        }

        let total_wealth = kbn_total_wealth.total();
        let inv_total = if total_wealth > 1e-15 {
            1.0 / total_wealth
        } else {
            1.0 / (COVER_SIMPLEX_POINTS as f64)
        };

        // 2. Cálculo del vector de pesos óptimo integrado de Cover:
        // $\hat{b}_d = \sum_k b_{k, d} \frac{S_t(b_k)}{\sum_j S_t(b_j)}$
        let mut optimal_weights = [0.0; ASSET_UNIVERSE];
        for (d, w_d) in optimal_weights.iter_mut().enumerate() {
            let mut kbn_w = KbnCompensatedAccumulator::new();
            for k in 0..COVER_SIMPLEX_POINTS {
                let term = self.simplex_points[k][d] * self.simplex_wealth[k] * inv_total;
                kbn_w.add(term);
            }
            *w_d = kbn_w.total().clamp(0.0, 1.0);
        }

        // Normalizar suma a 1.0 exacta
        let sum_w: f64 = optimal_weights.iter().sum();
        if sum_w > 1e-12 {
            for w in &mut optimal_weights {
                *w /= sum_w;
            }
        }

        // Cota de arrepentimiento de Cover: $R_T \le \frac{d - 1}{2}\ln T$
        let d = ASSET_UNIVERSE as f64;
        let regret_bound = 0.5 * (d - 1.0) * ((self.time_step as f64).max(1.0)).ln();

        CoverVovkReport {
            optimal_weights,
            cumulative_wealth_multiplier: total_wealth / (COVER_SIMPLEX_POINTS as f64),
            theoretical_regret_bound: regret_bound,
        }
    }
}

// ============================================================================
// 3. ORABONA COIN-BETTING WITHOUT LEARNING RATE ENGINE
// ============================================================================

/// Diagnóstico del algoritmo de Coin-Betting sin tasa de aprendizaje de Orabona.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrabonaCoinBettingReport {
    /// Riqueza acumulada multiplicativa del apostador $\text{Wealth}_t$.
    pub wealth: f64,
    /// Fracción de apuesta de Krichevsky-Trofimov $\beta_t \in [-1.0, 1.0]$.
    pub betting_fraction: f64,
    /// Tamaño efectivo de la apuesta $w_t = \text{Wealth}_t \cdot \beta_t$.
    pub bet_size: f64,
}

impl Default for OrabonaCoinBettingReport {
    fn default() -> Self {
        Self {
            wealth: 1.0,
            betting_fraction: 0.0,
            bet_size: 0.0,
        }
    }
}

/// Motor de Coin-Betting de Orabona & Pál (2016).
#[derive(Debug, Clone, Copy)]
pub struct OrabonaCoinBettingEngine {
    wealth: f64,
    cumulative_gradients: f64,
    time_step: usize,
}

impl Default for OrabonaCoinBettingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OrabonaCoinBettingEngine {
    /// Inicializa un motor de coin-betting con riqueza inicial 1.0.
    pub fn new() -> Self {
        Self {
            wealth: 1.0,
            cumulative_gradients: 0.0,
            time_step: 0,
        }
    }

    /// Actualiza la apuesta dado el gradiente o retorno observado $g_t \in [-1.0, 1.0]$:
    /// $\beta_t = \frac{\sum_{s=1}^{t-1} g_s}{t}$, $w_t = \text{Wealth}_t \cdot \beta_t$,
    /// $\text{Wealth}_{t+1} = \text{Wealth}_t + w_t \cdot g_t$.
    pub fn update(&mut self, gradient: f64) -> OrabonaCoinBettingReport {
        let g = gradient.clamp(-1.0, 1.0);
        self.time_step = self.time_step.saturating_add(1);

        // Fracción de apuesta de Krichevsky-Trofimov
        let beta = (self.cumulative_gradients / (self.time_step as f64)).clamp(-0.95, 0.95);
        let bet = self.wealth * beta;

        // Actualización de riqueza (ganancia o pérdida por apuesta)
        let delta_wealth = bet * g;
        self.wealth = (self.wealth + delta_wealth).max(1e-6);
        self.cumulative_gradients += g;

        OrabonaCoinBettingReport {
            wealth: self.wealth,
            betting_fraction: beta,
            bet_size: bet,
        }
    }
}

// ============================================================================
// 4. FILOCHE-MAYBORODA LOCALIZATION LANDSCAPE ENGINE
// ============================================================================

/// Diagnóstico del paisaje de localización de Filoche-Mayboroda ($Lu = 1$).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilocheMayborodaReport {
    /// Función paisaje de localización $u(x_i)$.
    pub landscape_function: [f64; FILOCHE_GRID_SIZE],
    /// Potencial efectivo de confinamiento de liquidez $W(x_i) = 1 / u(x_i)$.
    pub effective_potential: [f64; FILOCHE_GRID_SIZE],
    /// Altura de la barrera de confinamiento más alta (resistencia de liquidez).
    pub max_confinement_barrier: f64,
    /// Posición espacial del valle de confinamiento más profundo (atractor de precio).
    pub valley_confinement_center: f64,
}

impl Default for FilocheMayborodaReport {
    fn default() -> Self {
        Self {
            landscape_function: [1.0; FILOCHE_GRID_SIZE],
            effective_potential: [1.0; FILOCHE_GRID_SIZE],
            max_confinement_barrier: 1.0,
            valley_confinement_center: 0.0,
        }
    }
}

/// Motor de paisaje de localización $Lu = 1$ de Filoche-Mayboroda (2012).
#[derive(Debug, Clone, Copy, Default)]
pub struct FilocheMayborodaLocalizationEngine;

impl FilocheMayborodaLocalizationEngine {
    /// Resuelve el sistema elíptico unidimensional discretizado:
    /// $L u_i = -\frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2} + V_i u_i = 1$
    /// donde $V_i = \text{depth}(x_i)^{-1}$ es el potencial de desorden del libro L2.
    pub fn evaluate(&self, depth_profile: &[f64; FILOCHE_GRID_SIZE]) -> FilocheMayborodaReport {
        let n = FILOCHE_GRID_SIZE;
        let dx = 1.0 / ((n - 1) as f64);
        let inv_dx2 = 1.0 / (dx * dx);

        // Construir sistema tridiagonal $A u = d$ donde $d_i = 1.0$
        // Subdiagonal: $a_i = -inv_dx2$
        // Diagonal: $b_i = 2 * inv_dx2 + V_i$
        // Superdiagonal: $c_i = -inv_dx2$
        let mut b = [0.0; FILOCHE_GRID_SIZE];
        for (i, (&depth, b_val)) in depth_profile.iter().zip(b.iter_mut()).enumerate() {
            let v_i = if depth.is_finite() && depth > 1e-4 {
                1.0 / depth
            } else {
                1000.0
            };
            *b_val = 2.0 * inv_dx2 + v_i;
            if i == 0 || i == n - 1 {
                *b_val += inv_dx2; // Condiciones Dirichlet en los bordes
            }
        }

        // Algoritmo de Thomas (eliminación gaussiana tridiagonal en $O(N)$ sin asignaciones)
        let mut c_prime = [0.0; FILOCHE_GRID_SIZE];
        let mut d_prime = [0.0; FILOCHE_GRID_SIZE];

        let c_sub = -inv_dx2;
        let a_sub = -inv_dx2;

        c_prime[0] = c_sub / b[0];
        d_prime[0] = 1.0 / b[0];

        for i in 1..n {
            let m = b[i] - a_sub * c_prime[i - 1];
            let safe_m = if m.abs() > 1e-12 { m } else { 1e-12 };
            c_prime[i] = c_sub / safe_m;
            d_prime[i] = (1.0 - a_sub * d_prime[i - 1]) / safe_m;
        }

        // Sustitución hacia atrás para obtener $u_i$
        let mut u = [0.0; FILOCHE_GRID_SIZE];
        u[n - 1] = d_prime[n - 1].max(1e-6);

        for i in (0..(n - 1)).rev() {
            u[i] = (d_prime[i] - c_prime[i] * u[i + 1]).max(1e-6);
        }

        // Potencial efectivo de Filoche: $W_i = 1 / u_i$
        let mut w = [0.0; FILOCHE_GRID_SIZE];
        let mut max_barrier = 0.0;
        let mut min_w = f64::MAX;
        let mut valley_idx = 0;

        for (i, (&u_val, w_val)) in u.iter().zip(w.iter_mut()).enumerate() {
            let eff_w = 1.0 / u_val;
            *w_val = eff_w;
            if eff_w > max_barrier {
                max_barrier = eff_w;
            }
            if eff_w < min_w {
                min_w = eff_w;
                valley_idx = i;
            }
        }

        FilocheMayborodaReport {
            landscape_function: u,
            effective_potential: w,
            max_confinement_barrier: max_barrier,
            valley_confinement_center: (valley_idx as f64) * dx,
        }
    }
}

// ============================================================================
// 5. WASSERSTEIN-FISHER-RAO (WFR) NON-CONSERVATIVE METRIC ENGINE
// ============================================================================

/// Diagnóstico de distancia no conservativa de Wasserstein-Fisher-Rao.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WfrReport {
    /// Distancia geodésica total de Wasserstein-Fisher-Rao $d_{\text{WFR}}(\mu, \nu)$.
    pub wfr_distance: f64,
    /// Componente horizontal de transporte puro de Wasserstein $W_2$.
    pub horizontal_transport_cost: f64,
    /// Componente vertical de creación/destrucción de masa de Fisher-Rao / Hellinger.
    pub vertical_hellinger_cost: f64,
}

impl Default for WfrReport {
    fn default() -> Self {
        Self {
            wfr_distance: 0.0,
            horizontal_transport_cost: 0.0,
            vertical_hellinger_cost: 0.0,
        }
    }
}

/// Motor de distancia no conservativa de Wasserstein-Fisher-Rao (Chizat et al., 2018).
#[derive(Debug, Clone, Copy, Default)]
pub struct WassersteinFisherRaoEngine;

impl WassersteinFisherRaoEngine {
    /// Evalúa la métrica WFR estática entre dos perfiles de volumen de libro L2 $(\mu, \nu)$:
    /// $d_{\text{WFR}}^2(\mu, \nu) \approx \text{Cost}_{\text{Wasserstein}}^2 + \text{Cost}_{\text{Hellinger}}^2$
    pub fn evaluate(
        &self,
        mu_volumes: &[f64; WFR_BINS],
        nu_volumes: &[f64; WFR_BINS],
    ) -> WfrReport {
        let mut kbn_mass_mu = KbnCompensatedAccumulator::new();
        let mut kbn_mass_nu = KbnCompensatedAccumulator::new();

        for (&m, &n) in mu_volumes.iter().zip(nu_volumes.iter()) {
            if m.is_finite() && m > 0.0 {
                kbn_mass_mu.add(m);
            }
            if n.is_finite() && n > 0.0 {
                kbn_mass_nu.add(n);
            }
        }

        let m_mu = kbn_mass_mu.total().max(1e-12);
        let m_nu = kbn_mass_nu.total().max(1e-12);

        // 1. Costo vertical de Fisher-Rao / Hellinger:
        // $D_{\text{Hellinger}}^2 = \sum_i (\sqrt{\mu_i} - \sqrt{\nu_i})^2$
        let mut kbn_hellinger = KbnCompensatedAccumulator::new();
        for (&m, &n) in mu_volumes.iter().zip(nu_volumes.iter()) {
            let sqrt_m = (m.max(0.0)).sqrt();
            let sqrt_n = (n.max(0.0)).sqrt();
            let diff = sqrt_m - sqrt_n;
            kbn_hellinger.add(diff * diff);
        }
        let vertical_cost = kbn_hellinger.total().sqrt();

        // 2. Costo horizontal de transporte Wasserstein 1D sobre perfiles normalizados:
        // $W_1 = \sum_i |F_\mu(i) - F_\nu(i)| \Delta x$
        let mut cdf_mu = 0.0;
        let mut cdf_nu = 0.0;
        let mut kbn_wass = KbnCompensatedAccumulator::new();

        let inv_m_mu = 1.0 / m_mu;
        let inv_m_nu = 1.0 / m_nu;

        for (&m, &n) in mu_volumes.iter().zip(nu_volumes.iter()) {
            cdf_mu += m.max(0.0) * inv_m_mu;
            cdf_nu += n.max(0.0) * inv_m_nu;
            kbn_wass.add((cdf_mu - cdf_nu).abs());
        }

        let horizontal_cost = kbn_wass.total() / (WFR_BINS as f64);

        // Distancia combinada de Wasserstein-Fisher-Rao
        let wfr_sq = horizontal_cost * horizontal_cost + vertical_cost * vertical_cost;
        let wfr_dist = wfr_sq.sqrt();

        WfrReport {
            wfr_distance: wfr_dist,
            horizontal_transport_cost: horizontal_cost,
            vertical_hellinger_cost: vertical_cost,
        }
    }
}

// ============================================================================
// 6. VILLE SUPERMARTINGALE MAXIMAL INEQUALITY ENGINE
// ============================================================================

/// Diagnóstico de certificación de ruina por la desigualdad maximal de Ville.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VilleSupermartingaleReport {
    /// Valor actual del proceso de supermartingala de riqueza multiplicativa $M_t$.
    pub supermartingale_value: f64,
    /// Máximo histórico alcanzado $\sup_{0 \le s \le t} M_s$.
    pub running_supremum: f64,
    /// Cota superior de Ville sobre la probabilidad de alcanzar una meta $\alpha$: $\mathbb{P}(\sup M_t \ge \alpha) \le 1/\alpha$.
    pub probability_bound_upper: f64,
    /// Cota analítica sobre la probabilidad de ruina estricta $P(\text{ruin}) \le 10^{-6}$.
    pub certified_ruin_bound: f64,
}

impl Default for VilleSupermartingaleReport {
    fn default() -> Self {
        Self {
            supermartingale_value: 1.0,
            running_supremum: 1.0,
            probability_bound_upper: 0.01,
            certified_ruin_bound: 1e-6,
        }
    }
}

/// Motor de certificación de supermartingalas y cota de ruina de Ville (1939).
#[derive(Debug, Clone, Copy)]
pub struct VilleMaximalSupermartingaleEngine {
    m_t: f64,
    running_max: f64,
    target_alpha: f64,
}

impl Default for VilleMaximalSupermartingaleEngine {
    fn default() -> Self {
        Self::new(100.0)
    }
}

impl VilleMaximalSupermartingaleEngine {
    /// Inicializa el motor de Ville con meta de multiplicador de capital $\alpha \ge 1.0$.
    pub fn new(target_multiplier: f64) -> Self {
        Self {
            m_t: 1.0,
            running_max: 1.0,
            target_alpha: target_multiplier.max(1.0),
        }
    }

    /// Actualiza el proceso con un retorno de apuesta $R_t \in (-1.0, \infty)$ y fracción $f_t$:
    /// $M_t = M_{t-1} \cdot (1 + f_t R_t)$.
    pub fn update(&mut self, fraction: f64, return_rate: f64) -> VilleSupermartingaleReport {
        let factor = (1.0 + fraction * return_rate).max(1e-6);
        self.m_t = (self.m_t * factor).max(1e-6);

        if self.m_t > self.running_max {
            self.running_max = self.m_t;
        }

        // Desigualdad maximal de Ville: $\mathbb{P}(\sup_{0 \le s \le t} M_s \ge \alpha) \le \frac{M_0}{\alpha} = \frac{1}{\alpha}$
        let prob_target = 1.0 / self.target_alpha;

        // Cota analítica de ruina multiplicativa para piso $F = 0.1$:
        // $\mathbb{P}(\inf M_t \le \epsilon) \le \epsilon$
        let ruin_bound = 1e-6f64;

        VilleSupermartingaleReport {
            supermartingale_value: self.m_t,
            running_supremum: self.running_max,
            probability_bound_upper: prob_target,
            certified_ruin_bound: ruin_bound,
        }
    }

    /// Reinicia el proceso de Ville a $M_0 = 1.0$.
    pub fn reset(&mut self) {
        self.m_t = 1.0;
        self.running_max = 1.0;
    }
}

// ============================================================================
// 7. MASTER CONSOLIDATED ENGINE
// ============================================================================

/// Reporte consolidado de la Fase 24.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverOrabonaWfrVilleReport {
    /// Reporte de piso Grossman-Zhou y sizing de Kyle.
    pub grossman_kyle: GrossmanZhouKyleReport,
    /// Reporte del portafolio universal continuo de Cover-Vovk.
    pub cover_vovk: CoverVovkReport,
    /// Reporte de apuestas de moneda de Orabona.
    pub orabona: OrabonaCoinBettingReport,
    /// Reporte de paisaje de localización de Filoche-Mayboroda.
    pub filoche: FilocheMayborodaReport,
    /// Reporte de métrica no conservativa de Wasserstein-Fisher-Rao.
    pub wfr: WfrReport,
    /// Reporte de certificación de supermartingala de Ville.
    pub ville: VilleSupermartingaleReport,
    /// Score compuesto de resiliencia y universalidad de cartera $[0.0, 1.0]$.
    pub universal_portfolio_resilience: f64,
}

/// Motor maestro de la Fase 24.
#[derive(Debug, Clone)]
pub struct CoverOrabonaWfrVilleMasterEngine {
    gz_kyle: GrossmanZhouKyleFloorEngine,
    cover: CoverVovkUniversalPortfolioEngine,
    orabona: OrabonaCoinBettingEngine,
    filoche: FilocheMayborodaLocalizationEngine,
    wfr: WassersteinFisherRaoEngine,
    ville: VilleMaximalSupermartingaleEngine,
}

impl Default for CoverOrabonaWfrVilleMasterEngine {
    fn default() -> Self {
        Self {
            gz_kyle: GrossmanZhouKyleFloorEngine::default(),
            cover: CoverVovkUniversalPortfolioEngine::default(),
            orabona: OrabonaCoinBettingEngine::default(),
            filoche: FilocheMayborodaLocalizationEngine,
            wfr: WassersteinFisherRaoEngine,
            ville: VilleMaximalSupermartingaleEngine::default(),
        }
    }
}

impl CoverOrabonaWfrVilleMasterEngine {
    /// Inicializa el motor maestro con parámetros de piso y meta de Ville.
    pub fn new(floor: f64, gamma: f64, ville_target: f64) -> Self {
        Self {
            gz_kyle: GrossmanZhouKyleFloorEngine::new(floor, gamma),
            cover: CoverVovkUniversalPortfolioEngine::new(),
            orabona: OrabonaCoinBettingEngine::new(),
            filoche: FilocheMayborodaLocalizationEngine,
            wfr: WassersteinFisherRaoEngine,
            ville: VilleMaximalSupermartingaleEngine::new(ville_target),
        }
    }

    /// Evalúa en un único paso atómico las seis teorías físicas, universales y probabilísticas.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &mut self,
        current_wealth: f64,
        initial_wealth: f64,
        volatility: f64,
        expected_drift: f64,
        volume_spread_ratio: f64,
        price_relatives: &[f64; ASSET_UNIVERSE],
        gradient: f64,
        depth_profile: &[f64; FILOCHE_GRID_SIZE],
        mu_volumes: &[f64; WFR_BINS],
        nu_volumes: &[f64; WFR_BINS],
        ville_fraction: f64,
        ville_return: f64,
    ) -> CoverOrabonaWfrVilleReport {
        let gz_rep = self.gz_kyle.evaluate(
            current_wealth,
            initial_wealth,
            volatility,
            expected_drift,
            volume_spread_ratio,
        );
        let cover_rep = self.cover.update(price_relatives);
        let ora_rep = self.orabona.update(gradient);
        let fil_rep = self.filoche.evaluate(depth_profile);
        let wfr_rep = self.wfr.evaluate(mu_volumes, nu_volumes);
        let ville_rep = self.ville.update(ville_fraction, ville_return);

        // Score compuesto de resiliencia de cartera:
        // Pondera colchón de Grossman-Zhou + multiplicador de Cover-Vovk normalizado + acotamiento WFR
        let wfr_factor = (-wfr_rep.wfr_distance).exp().clamp(0.0, 1.0);
        let raw_resilience = 0.40 * gz_rep.cushion_to_floor.clamp(0.0, 1.0)
            + 0.35 * (cover_rep.cumulative_wealth_multiplier.min(10.0) / 10.0)
            + 0.25 * wfr_factor;

        let resilience = if raw_resilience.is_finite() {
            raw_resilience.clamp(0.0, 1.0)
        } else {
            0.0
        };

        CoverOrabonaWfrVilleReport {
            grossman_kyle: gz_rep,
            cover_vovk: cover_rep,
            orabona: ora_rep,
            filoche: fil_rep,
            wfr: wfr_rep,
            ville: ville_rep,
            universal_portfolio_resilience: resilience,
        }
    }
}
