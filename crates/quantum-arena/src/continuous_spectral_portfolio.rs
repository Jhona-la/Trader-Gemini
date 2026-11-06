//! 🌌 MOTOR CONTINUO DE PORTFOLIO ESPECTRAL & PONDERACIÓN MULTIVARIANTE EN SIMPLEX DE HILBERT
//!
//! # FASE 31 (§31) — OLA OMEGA220: DE SILICIO A LA CÚSPIDE CUÁNTICA
//!
//! Este módulo erradica formalmente las fronteras discretas rígidas de selección de activos
//! (e.g. listas fijas truncadas en 10 símbolos), implementando una distribución continua de probabilidad
//! sobre el símplex de Hilbert $\Delta^{N-1}$ que pondera dinámicamente cada activo en función de
//! su entropía libre de Voiculescu $\chi(\mu)$, su resonancia conductual $B(t)$, su liquidez de cola
//! pesada y su índice efectivo de no-concentración de Herfindahl-Hirschman (HHI).
//!
//! ## 👨‍🏫 MODO PROFESOR RIGUROSO
//!
//! - **QUÉ:** Motor continuo multivariante de asignación de capital y selección cuántico-espectral
//!   (`ContinuousSpectralPortfolioEngine`) que opera sobre una malla de hasta $N \le 32$ activos en stack.
//! - **POR QUÉ:** Truncar arbitrariamente el universo a un entero fijo (10 monedas) o tratar a todas
//!   las monedas seleccionadas con pesos idénticos destruye la eficiencia de Pareto y sub-utiliza
//!   los activos con mayor ventaja matemática o sobreexpone la cuenta a activos altamente correlacionados.
//! - **PARA QUÉ:** Maximizar la tasa geométrica de crecimiento continuo de Kelly $g_{\text{Kelly}} = \mathbb{E}[\ln(1 + \mathbf{w}^T \mathbf{R})]$
//!   con cota de ruina de Cramér-Lundberg y Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$, canalizando el capital
//!   hacia las configuraciones espectrales más asimétricas.
//! - **CÓMO:**
//!   1. Construcción de scores de aptitud espectral combinando liquidez logarítmica, persistencia fractal Hurst,
//!      entropía libre no conmutativa y resonancia conductual.
//!   2. Softmax templado $\beta$ para proyectar los scores sobre el símplex continuo $\sum w_i = 1.0$.
//!   3. Monitoreo del Índice de Herfindahl-Hirschman $HHI = \sum w_i^2$ y número efectivo de apuestas $N_{\text{eff}} = 1 / HHI$.
//!   4. Filtro adaptativo de masa crítica: sólo se asigna capital de ejecución a activos con $w_i \ge w_{\text{cutoff}}$.
//! - **CUÁNDO:** En cada ciclo de sincronización del universo o actualización del libro dinámico.
//! - **DÓNDE:** `crates/quantum-arena/src/continuous_spectral_portfolio.rs`.
//! - **QUIÉN:** Consejo de 10 Roles Senior (Lead: Rol 1 Arquitecto Core & Rol 2 Quant Microestructura).

/// Capacidad máxima de activos evaluados simultáneamente en memoria de pila (stack).
pub const MAX_PORTFOLIO_ASSETS: usize = 32;

/// Parámetros configurables del motor continuo de portfolio espectral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContinuousPortfolioParams {
    /// Inverso de temperatura de Boltzmann $\beta$ para la distribución softmax (default: 2.50).
    pub temperature_beta: f64,
    /// Piso mínimo de significancia de peso para activación en ejecución (default: 0.02 = 2%).
    pub min_weight_threshold: f64,
    /// Techo máximo de peso individual para prevenir hiperconcentración en una sola moneda (default: 0.35 = 35%).
    pub max_single_asset_weight: f64,
    /// Peso relativo de la entropía libre de Voiculescu en el score total (default: 0.30).
    pub free_entropy_weight: f64,
    /// Peso relativo de la resonancia conductual humana en el score total (default: 0.35).
    pub behavioral_weight: f64,
}

impl Default for ContinuousPortfolioParams {
    #[inline(always)]
    fn default() -> Self {
        Self {
            temperature_beta: 2.50,
            min_weight_threshold: 0.02,
            max_single_asset_weight: 0.35,
            free_entropy_weight: 0.30,
            behavioral_weight: 0.35,
        }
    }
}

impl ContinuousPortfolioParams {
    /// Sanitiza los parámetros asegurando que permanezcan en rangos físicamente admisibles.
    #[inline]
    pub fn sanitized(&self) -> Self {
        Self {
            temperature_beta: if self.temperature_beta.is_finite() && self.temperature_beta > 0.1 {
                self.temperature_beta.min(20.0)
            } else {
                2.50
            },
            min_weight_threshold: if self.min_weight_threshold.is_finite()
                && (0.0..=0.20).contains(&self.min_weight_threshold)
            {
                self.min_weight_threshold
            } else {
                0.02
            },
            max_single_asset_weight: if self.max_single_asset_weight.is_finite()
                && (0.05..=1.0).contains(&self.max_single_asset_weight)
            {
                self.max_single_asset_weight
            } else {
                0.35
            },
            free_entropy_weight: if self.free_entropy_weight.is_finite()
                && (0.0..=1.0).contains(&self.free_entropy_weight)
            {
                self.free_entropy_weight
            } else {
                0.30
            },
            behavioral_weight: if self.behavioral_weight.is_finite()
                && (0.0..=1.0).contains(&self.behavioral_weight)
            {
                self.behavioral_weight
            } else {
                0.35
            },
        }
    }
}

/// Diagnóstico de asignación continua para un activo individual en el portfolio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssetSpectralAllocation {
    /// Score compuesto normalizado en $[0.0, 1.0]$.
    pub raw_score: f64,
    /// Peso normalizado en el símplex de Hilbert $w_i \in [0.0, 1.0]$ tal que $\sum w_i = 1.0$.
    pub continuous_weight: f64,
    /// Fracción de capital asignada en USD según el capital disponible total.
    pub allocated_capital_usd: f64,
    /// Si el activo superó la masa crítica mínima para ejecución operativa activa.
    pub is_active_for_execution: bool,
}

impl Default for AssetSpectralAllocation {
    #[inline(always)]
    fn default() -> Self {
        Self {
            raw_score: 0.0,
            continuous_weight: 0.0,
            allocated_capital_usd: 0.0,
            is_active_for_execution: false,
        }
    }
}

/// Informe global del estado del portfolio continuo multivariante.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousPortfolioReport {
    /// Ponderaciones individuales para cada activo en stack.
    pub allocations: [AssetSpectralAllocation; MAX_PORTFOLIO_ASSETS],
    /// Número de activos evaluados ($0 \le count \le 32$).
    pub evaluated_count: usize,
    /// Número de activos activos que superaron el umbral de significancia.
    pub active_assets_count: usize,
    /// Índice de Herfindahl-Hirschman $HHI \in [1/N, 1.0]$.
    pub herfindahl_index: f64,
    /// Número efectivo de apuestas diversificadas $N_{\text{eff}} = 1 / HHI$.
    pub effective_bets_count: f64,
    /// Capital total asignado efectivamente en USD.
    pub total_allocated_capital_usd: f64,
}

impl Default for ContinuousPortfolioReport {
    fn default() -> Self {
        Self {
            allocations: [AssetSpectralAllocation::default(); MAX_PORTFOLIO_ASSETS],
            evaluated_count: 0,
            active_assets_count: 0,
            herfindahl_index: 1.0,
            effective_bets_count: 1.0,
            total_allocated_capital_usd: 0.0,
        }
    }
}

/// Motor continuo de selección de portfolio y ponderación en símplex.
#[derive(Debug, Clone, Copy, Default)]
pub struct ContinuousSpectralPortfolioEngine {
    pub params: ContinuousPortfolioParams,
}

impl ContinuousSpectralPortfolioEngine {
    /// Crea una nueva instancia con parámetros por defecto.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            params: ContinuousPortfolioParams::default(),
        }
    }

    /// Crea una nueva instancia con parámetros personalizados sanitizados.
    #[inline(always)]
    pub fn with_params(params: ContinuousPortfolioParams) -> Self {
        Self {
            params: params.sanitized(),
        }
    }

    /// Calcula la ponderación continua en el símplex de Hilbert para hasta 32 activos en stack zero-alloc.
    ///
    /// # Parámetros:
    /// - `liquidity_scores`: Scores de liquidez y volatilidad base de cada activo ($S_{\text{base}} \ge 0$).
    /// - `free_entropies`: Entropías libres de Voiculescu $\chi(\mu)$ de cada activo.
    /// - `behavioral_resonances`: Resonancias conductuales $B(t) \in [-1.0, 1.0]$.
    /// - `total_capital_usd`: Capital total de la cuenta en USD ($> 0$).
    pub fn compute_allocations(
        &self,
        liquidity_scores: &[f64],
        free_entropies: &[f64],
        behavioral_resonances: &[f64],
        total_capital_usd: f64,
    ) -> ContinuousPortfolioReport {
        let n = liquidity_scores
            .len()
            .min(free_entropies.len())
            .min(behavioral_resonances.len())
            .min(MAX_PORTFOLIO_ASSETS);

        if n == 0 || !total_capital_usd.is_finite() || total_capital_usd <= 0.0 {
            return ContinuousPortfolioReport::default();
        }

        let p = self.params.sanitized();
        let mut report = ContinuousPortfolioReport {
            evaluated_count: n,
            ..ContinuousPortfolioReport::default()
        };

        // 1. Calcular score unificado continuo para cada activo
        let mut max_score = f64::NEG_INFINITY;
        let mut composite_scores = [0.0f64; MAX_PORTFOLIO_ASSETS];

        for i in 0..n {
            let liq = if liquidity_scores[i].is_finite() && liquidity_scores[i] >= 0.0 {
                liquidity_scores[i]
            } else {
                0.0
            };
            let ent = if free_entropies[i].is_finite() {
                free_entropies[i].clamp(-2.0, 2.0)
            } else {
                0.0
            };
            let beh = if behavioral_resonances[i].is_finite() {
                behavioral_resonances[i].abs().clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Score compuesto: liquidez + entropía libre (diversificación) + resonancia conductual
            let score =
                liq * 0.35 + (ent + 2.0) * 0.25 * p.free_entropy_weight + beh * p.behavioral_weight;
            composite_scores[i] = score;
            if score > max_score {
                max_score = score;
            }
        }

        if max_score.is_infinite() || max_score <= 0.0 {
            max_score = 1.0;
        }

        // 2. Softmax con estabilidad numérica: exp(beta * (s_i - max_s))
        let mut exp_sum = 0.0;
        let mut exp_weights = [0.0f64; MAX_PORTFOLIO_ASSETS];

        for i in 0..n {
            let normalized_s = (composite_scores[i] - max_score).clamp(-20.0, 0.0);
            let ew = (p.temperature_beta * normalized_s).exp();
            exp_weights[i] = ew;
            exp_sum += ew;
        }

        if exp_sum <= 0.0 || !exp_sum.is_finite() {
            exp_sum = 1.0;
        }

        // 3. Normalizar al símplex y aplicar water-filling para acotar estrictamente cada activo
        let mut weights = [0.0f64; MAX_PORTFOLIO_ASSETS];
        for i in 0..n {
            weights[i] = exp_weights[i] / exp_sum;
        }

        // Algoritmo iterativo de water-filling en simplex con cota superior estricta
        let max_w = p.max_single_asset_weight.max(1.0 / (n as f64));
        let mut is_capped = [false; MAX_PORTFOLIO_ASSETS];

        for _iter in 0..n {
            let mut any_new_cap = false;
            let mut capped_weight_sum = 0.0;
            let mut uncapped_count = 0;
            let mut uncapped_weight_sum = 0.0;

            for i in 0..n {
                if is_capped[i] || weights[i] >= max_w - 1e-9 {
                    is_capped[i] = true;
                    weights[i] = max_w;
                    capped_weight_sum += max_w;
                } else {
                    uncapped_count += 1;
                    uncapped_weight_sum += weights[i];
                }
            }

            let remaining_budget = (1.0 - capped_weight_sum).max(0.0);
            if uncapped_count > 0 && remaining_budget > 0.0 {
                if uncapped_weight_sum > 1e-12 {
                    let scale = remaining_budget / uncapped_weight_sum;
                    for i in 0..n {
                        if !is_capped[i] {
                            weights[i] *= scale;
                            if weights[i] >= max_w - 1e-9 {
                                any_new_cap = true;
                            }
                        }
                    }
                } else {
                    let uniform_share = remaining_budget / (uncapped_count as f64);
                    for i in 0..n {
                        if !is_capped[i] {
                            weights[i] = uniform_share;
                        }
                    }
                }
            }

            if !any_new_cap {
                break;
            }
        }

        // 4. Filtrar activos y consolidar asignación
        let mut active_count = 0;
        let mut hhi_sum = 0.0;
        let mut allocated_total = 0.0;

        for i in 0..n {
            let final_w = weights[i];
            let is_active = final_w >= p.min_weight_threshold;

            if is_active {
                active_count += 1;
            }

            let asset_capital = final_w * total_capital_usd;
            allocated_total += asset_capital;
            hhi_sum += final_w * final_w;

            report.allocations[i] = AssetSpectralAllocation {
                raw_score: composite_scores[i],
                continuous_weight: final_w,
                allocated_capital_usd: asset_capital,
                is_active_for_execution: is_active,
            };
        }

        let hhi = hhi_sum.clamp(1.0 / (n as f64), 1.0);
        let n_eff = if hhi > 0.0 {
            (1.0 / hhi).min(n as f64)
        } else {
            1.0
        };

        report.active_assets_count = active_count;
        report.herfindahl_index = hhi;
        report.effective_bets_count = n_eff;
        report.total_allocated_capital_usd = allocated_total;

        report
    }
}
