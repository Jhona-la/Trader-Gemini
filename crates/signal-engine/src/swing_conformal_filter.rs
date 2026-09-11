use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// FILTRO CONFORMAL DE REVERSIÓN CONFLUENTE CON TENDENCIA.
///
/// Emite una opinión direccional cuando (1) el precio se ha desviado de su base
/// de forma estadísticamente significativa, (2) la reversión de esa desviación
/// va a favor de la tendencia macro y (3) el calibrador conformal del motor
/// acepta la predicción.
///
/// # D-618 / D-626 / D-627 (DÉCIMA OLA)
///
/// * **D-618**: la aceptación conformal comparaba `p ≥ 1 − α`, que no es la
///   regla conformal. Ahora la decide el calibrador con predicción selectiva
///   (conjunto = {gana}) y aquí sólo se consume `conformal_accept`.
/// * **D-626**: el umbral del z-score era el literal 1,5 en esta ruta y otra
///   fórmula en `is_swing_confluence_valid`: dos definiciones de la misma
///   regla. Además la puntuación saltaba de 0 a 0,24 al cruzarlo. Ahora el
///   umbral es el valor crítico bilateral al MISMO nivel α del genoma
///   (α = 0,10 ⇒ |z| > 1,645) y la puntuación es continua: vale 0 exactamente
///   en el umbral y crece hacia 1 con la significación.
/// * **D-627**: `is_swing_confluence_valid` usaba `vecm_beta_hedge` —un ratio
///   de cobertura de cointegración— como desplazamiento de un umbral en
///   desviaciones estándar. No tenía llamadores; se elimina.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct SwingConformalFilterEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for SwingConformalFilterEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwingConformalFilterEngine").finish()
    }
}

/// Función de error complementaria, aproximación de Abramowitz y Stegun 7.1.26
/// (error máximo 1,5·10⁻⁷), para `x ≥ 0`.
#[inline]
fn erfc_nonneg(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

/// p-valor bilateral de un z-score bajo la hipótesis nula normal estándar.
#[inline]
pub(crate) fn two_sided_normal_p(z: f64) -> f64 {
    if !z.is_finite() {
        return 1.0;
    }
    erfc_nonneg(z.abs() / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
}

/// Intensidad continua de una desviación significativa al nivel `alpha`:
/// 0 cuando el p-valor alcanza α (en el umbral y por debajo de él), → 1
/// cuando la desviación es extrema.
#[inline]
pub(crate) fn significance_strength(z: f64, alpha: f64) -> f64 {
    let a = if alpha.is_finite() {
        alpha.clamp(1e-4, 0.5)
    } else {
        0.10
    };
    (1.0 - two_sided_normal_p(z) / a).clamp(0.0, 1.0)
}

impl SwingConformalFilterEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Puntuación direccional pura, separada del registro para poder
    /// verificarla.
    #[inline]
    pub(crate) fn score(z: f64, trend: f64, conformal_accept: bool, alpha: f64) -> f64 {
        if !conformal_accept || !z.is_finite() || !trend.is_finite() {
            return 0.0;
        }
        let strength = significance_strength(z, alpha);
        if strength <= 0.0 {
            return 0.0;
        }
        if z < 0.0 && trend >= 0.0 {
            // Precio bajo su base con tendencia alcista: la reversión acompaña.
            strength
        } else if z > 0.0 && trend <= 0.0 {
            // Precio sobre su base con tendencia bajista.
            -strength
        } else {
            0.0
        }
    }
}

impl QuantumStrategy for SwingConformalFilterEngine {
    fn name(&self) -> &str {
        "SwingConformalFilterEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() { None } else { Some(symbol) };
        let cid_opt = if symbol.is_empty() { None } else { Some(coin_id) };
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };
        let get = |key: &str| {
            r.get_scoped_parameter(sym_opt, cid_opt, key, "SwingConformalFilterEngine")
                .map(|p| p.get_value())
        };
        let z = get("vecm_zscore")
            .or_else(|| get("cointegration_zscore"))
            .unwrap_or(0.0);
        let trend = get("ema_trend_swing")
            .or_else(|| get("trend_direction"))
            .unwrap_or(0.0);
        // Fail-open si el motor aún no publica la decisión conformal, coherente
        // con el warmup del calibrador.
        let accept = get("conformal_accept").map(|v| v >= 0.5).unwrap_or(true);
        let alpha = get("conformal_alpha").unwrap_or(0.10);
        Self::score(z, trend, accept, alpha)
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn el_p_valor_bilateral_es_preciso() {
        assert!((two_sided_normal_p(0.0) - 1.0).abs() < 1e-6);
        assert!((two_sided_normal_p(1.959964) - 0.05).abs() < 1e-5);
        assert!((two_sided_normal_p(1.644854) - 0.10).abs() < 1e-5);
    }

    /// D-626: el umbral lo fija el α del genoma, no un literal.
    #[test]
    fn d626_el_umbral_sigue_al_alpha_del_genoma() {
        assert_eq!(SwingConformalFilterEngine::score(-1.5, 1.0, true, 0.10), 0.0);
        assert!(SwingConformalFilterEngine::score(-1.7, 1.0, true, 0.10) > 0.0);
        assert!(SwingConformalFilterEngine::score(-1.5, 1.0, true, 0.20) > 0.0);
    }

    /// D-626: sin salto en el umbral.
    #[test]
    fn d626_la_puntuacion_es_continua_en_el_umbral() {
        let just_above = SwingConformalFilterEngine::score(-1.646, 1.0, true, 0.10);
        assert!(just_above > 0.0 && just_above < 0.01, "salto en el umbral: {just_above}");
    }

    #[test]
    fn direccion_tendencia_y_rechazo_conformal() {
        assert!(SwingConformalFilterEngine::score(-2.5, 1.0, true, 0.10) > 0.5);
        assert!(SwingConformalFilterEngine::score(2.5, -1.0, true, 0.10) < -0.5);
        assert_eq!(SwingConformalFilterEngine::score(-2.5, -1.0, true, 0.10), 0.0);
        assert_eq!(SwingConformalFilterEngine::score(-2.5, 1.0, false, 0.10), 0.0);
    }

    #[test]
    fn evalua_desde_el_registro() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("vecm_zscore", -2.5);
        registry.set("ema_trend_swing", 1.0);
        registry.set("conformal_accept", 1.0);
        registry.set("conformal_alpha", 0.10);
        let mut engine = SwingConformalFilterEngine::new();
        assert!(engine.init(registry.clone()).is_ok());
        assert_eq!(engine.horizon(), strategy_core::TradeHorizon::Continuous);
        assert!(engine.evaluate() > 0.0);
        registry.set("conformal_accept", 0.0);
        assert_eq!(engine.evaluate(), 0.0);
    }
}
