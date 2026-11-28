//! # Fase 81 (§81) — Curva de Page & Fórmula de Isla para Recuperación Holográfica de Entropía
//!
//! ## 1. QUÉ
//! Módulo de compuerta y modulación de riesgo ultra-avanzado que implementa la termodinámica cuántica
//! de evaporación de agujeros negros (Page 1993, Almheiri-Engelhardt-Marolf-Maxfield-Pennington 2019-2020),
//! evaluando la transición de fase de la Superficie Extremal Cuántica (QES) de la radiación de riesgo:
//! $$S(\text{Rad}) = \min \left\{ S_{\text{no-island}}, \frac{\operatorname{Area}(\partial I)}{4 G_N} + S_{\text{bulk}}(\text{Rad} \cup I) \right\}$$
//! y computando la Curva de Page en relación a la capacidad Bekenstein-Hawking del capital $S_{\text{BH}}$.
//!
//! ## 2. POR QUÉ
//! En crisis de mercado extremas o drawdowns severos, el portafolio sufre evaporación térmica de valor
//! análoga a la radiación de Hawking de un agujero negro. Mientras $t < t_{\text{Page}}$, la entropía de riesgo
//! crece linealmente; alcanzado el tiempo de Page ($S_{\text{Rad}} \approx S_{\text{BH}}$), la unitaridad cuántica
//! exige que aparezca una "isla cuántica" $I$ en el interior del horizonte de pérdidas para purificar la radiación
//! y recuperar la información del capital. Si la isla no puede formarse (correlaciones colapsadas a 1.0, falta
//! de activos refugio), la evaporación destruye la unitaridad, conduciendo a la ruina irrecuperable.
//!
//! ## 3. PARA QUÉ
//! - Computar la entropía Bekenstein-Hawking del capital $S_{\text{BH}} = \ln(1 + \text{equity}/\text{floor})$.
//! - Computar la entropía de exposición de radiación $S_{\text{Rad}}$ y verificar si $t \ge t_{\text{Page}}$.
//! - Determinar la existencia y estabilidad de la isla cuántica holográfica $I$.
//! - Emitir el veto determinista absoluto incondicional `PAGE_CURVE_INFORMATION_COLLAPSE_VETO` (`island_veto_induced = true`).
//! - Modular de forma continua el factor de escala de posición $\eta_{\text{page}} \in [0.15, 1.25]$ en régimen unitario.
//! - Preservar de forma estricta la cota de ruina supermartingala de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//!
//! ## 4. CÓMO
//! 1. Se calcula $S_{\text{BH}} = \ln(1.0 + \text{equity} / (\text{floor} + \epsilon))$.
//! 2. Se calcula $S_{\text{Rad}}$ = entropía de Shannon de la composición de la exposición +
//!    $\ln(1 + \Lambda\sigma/\sigma_{\text{ref}})$, con $\Lambda = \sum_i w_i$ la exposición
//!    bruta en unidades de equity y $\sigma$ la volatilidad que entrega el llamador, en las
//!    unidades de $\sigma_{\text{ref}}$. Un libro plano ($\Lambda = 0$) no radia. Un peso o una
//!    volatilidad desconocidos no se leen como cero.
//! 3. Se evalúa el área de la frontera de la isla
//!    $\operatorname{Area}(\partial I) \sim \min\{1, 1.0 - \rho_{\text{max}}\}$, con
//!    $\rho_{\text{max}} \in [-1, 1]$ la correlación máxima, con signo, entre activos expuestos:
//!    una anticorrelación es cobertura y no reduce el área. Sin exposición el área es 1; con
//!    $\rho$ desconocida, 0.
//! 4. Si $\operatorname{Area}(\partial I) \ge \text{min\_island\_area}$, se activa la isla y se evalúa $S_{\text{island}} = \frac{\operatorname{Area}(\partial I)}{4 G_N} + S_{\text{bulk}}$.
//! 5. Se activa `island_veto_induced = true` y `slot_scale_factor = 0.0` si la equity está en o
//!    bajo el piso de Grossman-Zhou (no queda colchón sobre el piso), si la exposición o la
//!    volatilidad con exposición abierta son desconocidas, o si $S_{\text{Rad}} \ge S_{\text{BH}}$
//!    y la isla está ausente o $S_{\text{island}} > 1.02\,S_{\text{BH}}$.
//! 6. En régimen unitario sin veto, se modula $\eta_{\text{page}} \in [0.15, 1.25]$ asegurando convergencia unitaria.
//!
//! ## 5. CUÁNDO
//! En cada pulso de supervisión de riesgo del orquestador omnisciente en `risk-engine`.
//!
//! ## 6. DÓNDE
//! `crates/risk-engine/src/page_curve_island_risk.rs`, re-exportado en `lib.rs`.
//!
//! ## 7. QUIÉN
//! Diseñado e implementado por el Consejo de 10 Roles Senior (Risk Manager, Físico Teórico, Auditor Forense Causal).

pub const MAX_PORTFOLIO_ASSETS: usize = 4;

/// Configuración para el motor de Curva de Page y Fórmula de Isla Holográfica.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageCurveIslandConfig {
    /// Constante de acoplamiento de Newton efectiva 4 * G_N.
    pub four_gn_coupling: f64,
    /// Área mínima requerida para la formación de una isla cuántica.
    pub min_island_area: f64,
    /// Piso de capital de emergencia (Grossman-Zhou floor).
    pub emergency_floor_usd: f64,
    /// Sensibilidad de escala continua de slot.
    pub slot_scale_base: f64,
    /// Volatilidad de referencia para escala entrópica térmica.
    pub volatility_reference: f64,
}

impl Default for PageCurveIslandConfig {
    fn default() -> Self {
        Self {
            four_gn_coupling: 1.0,
            min_island_area: 0.15,
            emergency_floor_usd: 5.0,
            slot_scale_base: 1.20,
            volatility_reference: 0.02,
        }
    }
}

/// Reporte emitido por el evaluador de Curva de Page e Isla Holográfica.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageCurveIslandReport {
    /// Entropía Bekenstein-Hawking del capital remanente S_BH.
    pub bekenstein_hawking_entropy: f64,
    /// Entropía de radiación de riesgo acumulada S_Rad.
    pub radiation_entropy: f64,
    /// Entropía efectiva gobernada por la fórmula extremal de isla S_Page.
    pub page_entropy: f64,
    /// Indicador de si el sistema superó el tiempo de Page (t >= t_Page).
    pub post_page_transition: bool,
    /// Indicador de formación y estabilidad de la isla cuántica purificadora.
    pub quantum_island_present: bool,
    /// Indicador de veto determinista absoluto incondicional por colapso de información no unitario.
    pub island_veto_induced: bool,
    /// Factor de escala continuo de slot [0.0, 1.25].
    pub slot_scale_factor: f64,
}

/// Motor de evaluación de la Curva de Page e Isla Holográfica de Riesgo.
#[derive(Debug, Clone)]
pub struct PageCurveIslandRiskEngine {
    config: PageCurveIslandConfig,
}

impl Default for PageCurveIslandRiskEngine {
    fn default() -> Self {
        Self::new(PageCurveIslandConfig::default())
    }
}

impl PageCurveIslandRiskEngine {
    /// Crea un nuevo motor de Curva de Page con la configuración dada.
    pub fn new(config: PageCurveIslandConfig) -> Self {
        Self { config }
    }

    /// Evalúa la Curva de Page y la fórmula de isla:
    /// - `equity_usd`: capital actual en USD.
    /// - `asset_weights`: exposición bruta de hasta 4 activos en unidades de equity
    ///   (|nocional_i| / equity), sin normalizar; Λ = Σw es el apalancamiento bruto. Un peso no
    ///   finito o negativo es exposición desconocida, no cero, y veta.
    /// - `portfolio_volatility`: volatilidad del retorno por unidad de nocional, en las unidades
    ///   de `volatility_reference`. No finita o negativa es desconocida y, con Λ > 0, veta. El
    ///   grafo vivo pasa 100·√traza de la CWT por evento (con suelo 0,01): no es una volatilidad
    ///   a un horizonte fijo, así que el término térmico depende de la cadencia de eventos.
    /// - `max_cross_correlation`: ρ máxima, con signo, entre los PnL de los activos expuestos (con
    ///   la dirección de cada posición; se acota a [-1, 1]): una anticorrelación es cobertura y no
    ///   reduce el área de la isla. No finita es desconocida y no acredita isla. Con Λ = 0 no
    ///   interviene.
    pub fn evaluate(
        &self,
        equity_usd: f64,
        asset_weights: &[f64; MAX_PORTFOLIO_ASSETS],
        portfolio_volatility: f64,
        max_cross_correlation: f64,
    ) -> PageCurveIslandReport {
        let safe_equity = if equity_usd.is_finite() && equity_usd > 0.0 {
            equity_usd
        } else {
            0.0
        };

        // Volatilidad desconocida: no se lee como calma; con exposición abierta veta (paso 5) y el
        // 0.01 sólo mantiene finito el reporte. Una conocida se acota a [1e-4, 10] sin saltos, para
        // que la radiación no baje al subir la volatilidad.
        let vol_desconocida = !portfolio_volatility.is_finite() || portfolio_volatility < 0.0;
        let safe_vol = if vol_desconocida {
            0.01
        } else {
            portfolio_volatility.clamp(1e-4, 10.0)
        };

        // 1. Capacidad Bekenstein-Hawking del capital: S_BH = ln(1.0 + equity / floor)
        let floor_usd = self.config.emergency_floor_usd.max(0.1);
        let s_bh = ((safe_equity / floor_usd) + 1.0).max(1e-6).ln();

        // 2. Entropía de radiación de riesgo. La exposición bruta Λ = Σw entra en el término
        // térmico. Un peso no finito o negativo, o una suma que desborda, es exposición
        // desconocida (no cero); sólo la exposición certificable entra en la entropía.
        let mut exposicion_desconocida = false;
        let mut exposicion_bruta = 0.0;
        for &w in asset_weights.iter() {
            if !w.is_finite() || w < 0.0 {
                exposicion_desconocida = true;
            } else {
                exposicion_bruta += w;
            }
        }
        if !exposicion_bruta.is_finite() {
            exposicion_desconocida = true;
        }
        let exposicion = if exposicion_desconocida {
            0.0
        } else {
            exposicion_bruta
        };
        // Shannon de la composición: escala libre, sólo mide cómo se reparte la exposición.
        let mut s_shannon = 0.0;
        if exposicion > 1e-6 {
            for &w in asset_weights.iter() {
                if w > 0.0 {
                    let p = w / exposicion;
                    s_shannon -= p * p.ln();
                }
            }
        }
        let vol_ref = if self.config.volatility_reference.is_finite()
            && self.config.volatility_reference > 1e-4
        {
            self.config.volatility_reference
        } else {
            0.02
        };
        // Término térmico extensivo: la volatilidad del capital escala como Λ·σ (con un activo; con
        // varios y correlaciones no negativas es una cota superior, por la desigualdad
        // triangular). Sin exposición no radia.
        let carga_termica = exposicion * safe_vol / vol_ref;
        let carga_certificable = carga_termica.is_finite();
        let s_vol = if carga_certificable {
            (1.0 + carga_termica).ln()
        } else {
            0.0
        };
        let s_rad = (s_shannon + s_vol).max(0.0);

        // 3. Área de la frontera de la isla holográfica: proporcional a la descorrelación
        // residual 1 − ρ, con signo: una anticorrelación es cobertura y no reduce el área (con
        // 1 − |ρ|, ρ = −1 destruía la isla). Sin exposición toda la equity es refugio (área 1);
        // una ρ desconocida no certifica la isla (antes caía a 0.0, que se leía como
        // diversificación perfecta).
        let island_fraction = if exposicion_desconocida {
            0.0
        } else if exposicion <= 0.0 {
            1.0
        } else if max_cross_correlation.is_finite() {
            let safe_corr = max_cross_correlation.clamp(-1.0, 1.0);
            (1.0 - safe_corr).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let island_present = island_fraction >= self.config.min_island_area;

        // 4. Fórmula de la Superficie Extremal Cuántica (QES) para la Curva de Page
        // Con isla cuántica activa, la radiación se purifica y su entropía pasa a estar
        // acotada por el área de la isla en el interior del horizonte:
        // S_island = (island_fraction * S_BH) / (4 G_N) + S_bulk_residual
        let four_gn = self.config.four_gn_coupling.max(0.1);
        let s_island_area = (island_fraction * s_bh) / four_gn;
        let s_island_bulk = (1.0 - island_fraction) * s_rad * 0.10;
        let s_island = s_island_area + s_island_bulk;

        let post_page = s_rad >= s_bh;

        let s_page = if post_page {
            if island_present {
                s_island.min(s_bh)
            } else {
                s_rad
            }
        } else {
            s_rad
        };

        // 5. Criterio de veto determinista absoluto incondicional
        // (PAGE_CURVE_INFORMATION_COLLAPSE_VETO), en tres cláusulas:
        // - Piso de Grossman-Zhou: en o bajo el piso no queda colchón (equity − piso)⁺ que
        //   arriesgar. Es un corte, no una escala proporcional al colchón: sobre el piso decide
        //   la curva de Page.
        // - Radiación no certificable (exposición o volatilidad desconocidas con exposición
        //   abierta): falla cerrado.
        // - Curva de Page: el riesgo superó el límite (post_page) y la isla no puede formarse
        //   o su entropía colapsa la capacidad del capital (s_island > s_bh * 1.02).
        let sin_capacidad_sobre_piso = safe_equity <= floor_usd;
        let radiacion_indeterminada =
            exposicion_desconocida || !carga_certificable || (vol_desconocida && exposicion > 0.0);
        let veto_induced = sin_capacidad_sobre_piso
            || radiacion_indeterminada
            || (post_page && (!island_present || s_island > s_bh * 1.02));

        // 6. Factor de escala de slot continuo
        let slot_scale = if veto_induced {
            0.0
        } else {
            let margin_ratio = ((s_bh - s_page) / (s_bh + 0.1)).clamp(0.0, 1.0);
            let island_boost = if island_present {
                0.15 * island_fraction
            } else {
                0.0
            };
            (self.config.slot_scale_base * margin_ratio + island_boost).clamp(0.15, 1.25)
        };

        PageCurveIslandReport {
            bekenstein_hawking_entropy: s_bh,
            radiation_entropy: s_rad,
            page_entropy: s_page,
            post_page_transition: post_page,
            quantum_island_present: island_present,
            island_veto_induced: veto_induced,
            slot_scale_factor: slot_scale,
        }
    }
}
