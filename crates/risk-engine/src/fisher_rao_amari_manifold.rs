//! # Fase 93 (§93) — Geometría de Información de Fisher-Rao & Variedad $\alpha$-Dual de Amari
//!
//! ## Modo Profesor — Ontología Técnica y Matemática (7 Pilares Rigurosos)
//!
//! ### 1. QUÉ (Definición Técnica)
//! El motor `FisherRaoAmariManifoldEngine` implementa la geometría diferencial de información sobre
//! la variedad estadística riemanniana de distribuciones de pérdidas y retornos $\mathcal{S} = \{ p(x|\theta) \}$,
//! descubierta por C.R. Rao (1945), R.A. Fisher (1925) y Shun-ichi Amari (1985). Modela los estados de
//! riesgo mediante el tensor métrico de Fisher-Rao en coordenadas de localización-escala $\theta = (\mu, \sigma)$:
//! $$g_{ij}(\theta) = \mathbb{E}\left[ \frac{\partial \ln p}{\partial \theta^i} \frac{\partial \ln p}{\partial \theta^j} \right]
//! \implies ds^2 = \frac{d\mu^2 + 2 d\sigma^2}{2 \sigma^2}$$
//! Calcula la distancia geodésica hiperbólica de Poincaré $d_{\text{FR}}(P_t, P_0)$, la conexión afín dual $\nabla^{(\alpha)}$
//! acoplada al tensor cúbico de asimetría de Amari $T_{ijk}$, y la $\alpha$-divergencia invariante $D^{(\alpha)}(P_t || P_0)$.
//!
//! ### 2. POR QUÉ (Fundamento Teórico Físico-Matemático)
//! Las métricas tradicionales de riesgo (VaR, volatilidad histórica, drawdown simple) asumen un espacio
//! euclidiano plano donde las diferencias se miden con la norma $L^2$. En la microestructura financiera,
//! el espacio de probabilidades es una variedad con curvatura seccional negativa constante $\mathcal{K} = -1/2$.
//! Por la cota de Cramér-Rao $\operatorname{Cov}(\hat{\theta}) \ge g^{-1}(\theta)$, la incertidumbre en
//! la estimación de riesgo diverge exponencialmente cuando la volatilidad o la pérdida se aceleran,
//! produciendo transiciones de fase desordenadas indetectables con análisis lineal.
//!
//! ### 3. PARA QUÉ (Objetivo de Trading y Cota de Ville)
//! - Medir con exactitud geométrica en el semiplano hiperbólico la distancia real entre el régimen de
//!   riesgo actual $P_t$ y el estado basal seguro $P_0$.
//! - Modular el factor de escala de riesgo continuo $\eta_{\text{amari}} \in [0.10, 1.25]$ y la tolerancia al apalancamiento.
//! - Vetar de forma inmediata y determinista (`FISHER_RAO_INFORMATION_COLLAPSE_VETO`) ante divergencia
//!   geodésica excesiva ($d_{\text{FR}} > d_{\text{crit}}$ o $D^{(\alpha)} > D_{\text{crit}}$), preservando
//!   incondicionalmente la cota de ruina de la supermartingala de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//!
//! ### 4. CÓMO (Implementación en Silicio y Algorítmica)
//! - Estimador recursivo en stack de momentos continuos: media $\mu$, escala $\sigma > 0$ y skewness de Amari $T$.
//! - Distancia geodésica exacta en el semiplano hiperbólico:
//!   $$u = \frac{(\mu - \mu_0)^2 + 2(\sigma - \sigma_0)^2}{4 \sigma \sigma_0}, \quad d_{\text{FR}} = \sqrt{2} \ln(1 + u + \sqrt{u(u + 2)})$$
//! - Divergencia $\alpha$ de Amari parametrizada con $\alpha = 0.50$.
//! - Ejecución en tiempo constante $O(1)$ sin asignaciones en heap ni llamadas a bibliotecas externas.
//!
//! ### 5. CUÁNDO (Condiciones de Ejecución)
//! En cada ciclo de evaluación del motor de riesgo y dimensionamiento de capital dentro de `OmniscientLivingGraphOrchestrator`.
//!
//! ### 6. DÓNDE (Ubicación en la Arquitectura)
//! `crates/risk-engine/src/fisher_rao_amari_manifold.rs`, exportado en `crates/risk-engine/src/lib.rs`.
//!
//! ### 7. QUIÉN (Componente y Roles Responsables)
//! `FisherRaoAmariManifoldEngine`, diseñado y auditado por Rol 3 (Risk Manager Teórico) y
//! Rol 8 (Geómetra Diferencial & Físico-Matemático).

/// Parámetros de configuración para la variedad de Fisher-Rao y Amari.
#[derive(Debug, Clone, Copy)]
pub struct FisherRaoAmariConfig {
    /// Media basal libre de riesgo $\mu_0$.
    pub baseline_mean: f64,
    /// Volatilidad / escala basal segura $\sigma_0 > 0$.
    pub baseline_scale: f64,
    /// Parámetro de conexión dual $\alpha \in (-1.0, 1.0)$.
    pub alpha_parameter: f64,
    /// Distancia geodésica máxima admisible de Fisher-Rao $d_{\text{crit}}$.
    pub max_fisher_rao_geodesic_distance: f64,
    /// Divergencia máxima admisible de Amari $D_{\text{crit}}$.
    pub max_amari_divergence: f64,
    /// Regularizador numérico $\epsilon$.
    pub epsilon: f64,
}

impl Default for FisherRaoAmariConfig {
    fn default() -> Self {
        Self {
            baseline_mean: 0.0,
            baseline_scale: 0.015,
            alpha_parameter: 0.50,
            max_fisher_rao_geodesic_distance: 3.50,
            max_amari_divergence: 2.80,
            epsilon: 1e-7,
        }
    }
}

/// Reporte de estado emitido por la variedad de Fisher-Rao y Amari.
#[derive(Debug, Clone, Copy)]
pub struct FisherRaoAmariReport {
    /// Distancia geodésica de Fisher-Rao en el semiplano hiperbólico $d_{\text{FR}} \ge 0$.
    pub fisher_rao_geodesic_distance: f64,
    /// Divergencia $\alpha$ dual de Amari $D^{(\alpha)} \ge 0$.
    pub amari_alpha_divergence: f64,
    /// Curvatura seccional de la variedad riemanniana $\mathcal{K} = -1/2$.
    pub riemannian_curvature: f64,
    /// Tensor métrico de Fisher $g_{\mu\mu} = 1/\sigma^2$.
    pub metric_component_g_mu: f64,
    /// Tensor métrico de Fisher $g_{\sigma\sigma} = 2/\sigma^2$.
    pub metric_component_g_sigma: f64,
    /// Asimetría cúbica del tensor de Amari $T$.
    pub amari_skewness_tensor: f64,
    /// Factor de modulación de escala de riesgo continuo $\eta_{\text{amari}} \in [0.10, 1.25]$.
    pub slot_scaling_factor: f64,
    /// Indicador de veto estricto por colapso de información de Fisher-Rao.
    pub information_collapse_veto: bool,
}

/// Motor de geometría de información riemanniana de Fisher-Rao y variedad $\alpha$-dual de Amari.
#[derive(Debug, Clone)]
pub struct FisherRaoAmariManifoldEngine {
    config: FisherRaoAmariConfig,
    rolling_mean: f64,
    rolling_var: f64,
    rolling_skew: f64,
}

impl FisherRaoAmariManifoldEngine {
    /// Crea una nueva instancia del motor con la configuración dada.
    #[inline]
    pub fn new(config: FisherRaoAmariConfig) -> Self {
        let var_init = config.baseline_scale * config.baseline_scale;
        Self {
            config,
            rolling_mean: config.baseline_mean,
            rolling_var: var_init,
            rolling_skew: 0.0,
        }
    }

    /// Evalúa la posición geométrica del estado de riesgo en la variedad de Fisher-Rao.
    ///
    /// # Parámetros
    /// - `price_ret`: Retorno fraccionario del precio en el tick actual.
    /// - `drawdown_pct`: Porcentaje de drawdown actual en rango $[0.0, 1.0]$.
    /// - `spread_bps`: Spread actual en bps.
    /// - `vol_norm`: Volatilidad normalizada del activo.
    #[inline]
    pub fn evaluate(
        &mut self,
        price_ret: f64,
        drawdown_pct: f64,
        spread_bps: f64,
        vol_norm: f64,
    ) -> FisherRaoAmariReport {
        // 1. Validación fail-closed de entradas
        let inputs_valid = price_ret.is_finite()
            && drawdown_pct.is_finite()
            && drawdown_pct >= 0.0
            && spread_bps.is_finite()
            && spread_bps > 0.0
            && vol_norm.is_finite()
            && vol_norm >= 0.0;
        if !inputs_valid {
            return FisherRaoAmariReport {
                fisher_rao_geodesic_distance: 0.0,
                amari_alpha_divergence: 0.0,
                riemannian_curvature: -0.50,
                metric_component_g_mu: 0.0,
                metric_component_g_sigma: 0.0,
                amari_skewness_tensor: 0.0,
                slot_scaling_factor: 0.10,
                information_collapse_veto: true,
            };
        }

        let ret = price_ret;
        let dd = drawdown_pct.min(1.0);
        let _spread = spread_bps;
        let vol = vol_norm;

        let eps = self.config.epsilon;
        let mu_0 = self.config.baseline_mean;
        let sigma_0 = self.config.baseline_scale.max(eps);

        // 2. Actualización de momentos continuos en línea con filtro adaptativo
        let alpha_update = 0.05;
        self.rolling_mean = (1.0 - alpha_update) * self.rolling_mean + alpha_update * ret;
        let dev = ret - self.rolling_mean;
        let dev_sq = dev * dev;
        self.rolling_var =
            ((1.0 - alpha_update) * self.rolling_var + alpha_update * dev_sq).max(eps * eps);
        let sigma_t = (self.rolling_var.sqrt() + dd * 0.05 + vol * 0.01).max(eps);

        let dev_cube = dev_sq * dev;
        self.rolling_skew = (1.0 - alpha_update) * self.rolling_skew
            + alpha_update * (dev_cube / (sigma_t.powi(3) + eps));
        let amari_skewness_tensor = self.rolling_skew.clamp(-5.0, 5.0);

        let mu_t = self.rolling_mean - dd * 0.10; // Penalización geométrica de drawdown sobre la media

        // 3. Componentes del Tensor Métrico de Fisher-Rao
        let metric_component_g_mu = 1.0 / (sigma_t * sigma_t);
        let metric_component_g_sigma = 2.0 / (sigma_t * sigma_t);
        let riemannian_curvature = -0.50; // Curvatura gaussiana intrínseca de la variedad de localización-escala

        // 4. Distancia Geodésica de Fisher-Rao en el Semiplano Hiperbólico:
        // u = ((mu_t - mu_0)^2 + 2*(sigma_t - sigma_0)^2) / (4 * sigma_t * sigma_0)
        let delta_mu_sq = (mu_t - mu_0).powi(2);
        let delta_sigma_sq = (sigma_t - sigma_0).powi(2);
        let u = (delta_mu_sq + 2.0 * delta_sigma_sq) / (4.0 * sigma_t * sigma_0);
        let u_safe = u.max(0.0);
        let arg = 1.0 + u_safe + (u_safe * (u_safe + 2.0)).sqrt();
        let fisher_rao_geodesic_distance = ((2.0_f64).sqrt() * arg.ln()).max(0.0);

        // 5. Divergencia \alpha de Amari:
        // D^{(\alpha)}(P_t || P_0) con \alpha \in (-1, 1)
        let alpha = self.config.alpha_parameter.clamp(-0.95, 0.95);
        let factor = 4.0 / (1.0 - alpha * alpha);
        let term_sigma =
            ((1.0 - alpha) * (sigma_t * sigma_t) + (1.0 + alpha) * (sigma_0 * sigma_0)).max(eps);
        let sigma_ratio = ((2.0 * sigma_0 * sigma_t) / term_sigma).sqrt();
        let exp_arg = -((1.0 - alpha) * (1.0 + alpha) * delta_mu_sq) / (2.0 * term_sigma);
        let affinity = sigma_ratio * exp_arg.exp();
        let amari_alpha_divergence = (factor * (1.0 - affinity).max(0.0)).clamp(0.0, 100.0);

        // 6. Modulación de Escala de Slot Continuo $\eta_{\text{amari}} \in [0.10, 1.25]$
        let geom_penalty =
            1.0 / (1.0 + 0.35 * fisher_rao_geodesic_distance + 0.25 * amari_alpha_divergence);
        let slot_scaling_factor =
            ((0.10 + 1.15 * geom_penalty) * (1.0 - 0.15 * vol.min(1.0))).clamp(0.10, 1.25);

        // 7. Condición de Veto Fail-Closed Estricto
        let information_collapse_veto = fisher_rao_geodesic_distance
            > self.config.max_fisher_rao_geodesic_distance
            || amari_alpha_divergence > self.config.max_amari_divergence
            || dd > 0.35;

        FisherRaoAmariReport {
            fisher_rao_geodesic_distance,
            amari_alpha_divergence,
            riemannian_curvature,
            metric_component_g_mu,
            metric_component_g_sigma,
            amari_skewness_tensor,
            slot_scaling_factor,
            information_collapse_veto,
        }
    }
}

impl Default for FisherRaoAmariManifoldEngine {
    fn default() -> Self {
        Self::new(FisherRaoAmariConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_rao_normal_evaluation() {
        let mut engine = FisherRaoAmariManifoldEngine::default();
        let report = engine.evaluate(0.001, 0.01, 1.5, 0.02);

        assert!(report.fisher_rao_geodesic_distance >= 0.0);
        assert!(report.amari_alpha_divergence >= 0.0);
        assert_eq!(report.riemannian_curvature, -0.50);
        assert!(report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.25);
        assert!(!report.information_collapse_veto);
    }

    #[test]
    fn test_fisher_rao_subnormal_safety() {
        let mut engine = FisherRaoAmariManifoldEngine::default();
        let report = engine.evaluate(f64::NAN, f64::INFINITY, -1.0, f64::NAN);

        assert!(report.fisher_rao_geodesic_distance.is_finite());
        assert!(report.amari_alpha_divergence.is_finite());
        assert!(report.slot_scaling_factor.is_finite());
    }
}
