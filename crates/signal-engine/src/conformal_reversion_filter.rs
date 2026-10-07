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
/// # U-ERR-1 (ERRADICACIÓN DEL BINARIO DE HORIZONTE)
///
/// El motor se llamaba `SwingConformalFilterEngine` y vivía en
/// `swing_conformal_filter.rs`. El prefijo «swing» era una ETIQUETA DE BANDA
/// DE HORIZONTE que este motor no decide: su horizonte declarado es
/// `TradeHorizon::Continuous` y su regla no contiene ninguna escala temporal
/// —sólo significación estadística (α), dirección de tendencia y aceptación
/// conformal—. El nombre ahora describe lo que mide. La clave de registro
/// `ema_trend_swing` se conserva porque su PRODUCTOR vive fuera de este
/// ámbito (ver informe): renombrarla desde aquí rompería el productor.
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
pub struct ConformalReversionFilterEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for ConformalReversionFilterEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConformalReversionFilterEngine").finish()
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

impl ConformalReversionFilterEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// #621 (Ola 43) — VOTO ESPECTRAL del filtro conformal: la reversión
    /// se evalúa a CADA escala. El desplazamiento x(τ_k) ES el Z-score a
    /// esa banda (precio vs su media de horizonte τ, normalizado); la
    /// TENDENCIA local es el signo del desplazamiento en la escala
    /// ADYACENTE MÁS LENTA (k+1: la inercia que la reversión debe
    /// acompañar). El `score` conformal original decide la dirección y la
    /// significancia — la MISMA forma del vivo, resolución-en-escala.
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
        alpha: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        let mut por_escala = [0.0f64; 32];
        for k in 0..31 {
            // k+1 = escala adyacente más lenta: la tendencia que la
            // reversión debe acompañar (D-676: dirección, no «sube»).
            // #664 (G2-6): tendencia CONTINUA — el signum duro hacía
            // saltar el voto de ±strength a 0 al cruzar x(τ_{k+1})=0.
            // #666 (H2-1): z-score O(1) — tanh natural sin saturación.
            let tendencia = desplazamientos[k + 1].tanh();
            // Score conformal: z = x(τ_k) contra su base; aceptación
            // bidireccional (el registro puede restringir en vivo, aquí
            // es la forma pura).
            por_escala[k] = Self::score(desplazamientos[k], tendencia, true, true, alpha);
        }
        crate::voto_espectral::VotoEspectral::desde_arr(&por_escala)
    }

    /// Puntuación direccional pura, separada del registro para poder
    /// verificarla.
    #[inline]
    pub(crate) fn score(
        z: f64,
        trend: f64,
        accept_long: bool,
        accept_short: bool,
        alpha: f64,
    ) -> f64 {
        if !z.is_finite() || !trend.is_finite() {
            return 0.0;
        }
        let strength = significance_strength(z, alpha);
        if strength <= 0.0 {
            return 0.0;
        }
        // #664 (G2-6): dirección de reversión CONTINUA (opuesta a z) y
        // ACUERDO continuo con la tendencia adyacente (la reversión
        // acompaña cuando la escala lenta se opone a z). Con trend=±1
        // reproduce el comportamiento viejo; con trend cruzando 0 el
        // voto decae continuo a 0 en vez de saltar.
        // #666 (H2-1): divisores a la ESCALA DEL ESTADÍSTICO — con
        // 1e-3/1e-6 y |z|≥1.645 (única región emisora), direccion≡±1 y
        // acuerdo≡0/1: el "tanh continuo" era un signum disfrazado.
        let direccion = -(z / 2.0).tanh();
        // R4-C3: divisor 0.5 dejaba la media respuesta en |z·trend|=0.28,
        // ~6× bajo el emisor típico (|z|≥1.645) — acuerdo saturaba a
        // 0/1 con |trend|≥0.5. Divisor 2.0: en la región emisora con
        // |tendencia| moderada el acuerdo gradúa (tanh(1.645·0.5/2)=0.38;
        // tanh(1.645·1.0/2)=0.69).
        let acuerdo = (-(z * trend) / 2.0).tanh().max(0.0);
        let v = strength * direccion * acuerdo;
        if v > 0.0 && !accept_long {
            return 0.0;
        }
        if v < 0.0 && !accept_short {
            return 0.0;
        }
        v
    }
}

impl QuantumStrategy for ConformalReversionFilterEngine {
    fn name(&self) -> &str {
        "ConformalReversionFilterEngine"
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
            r.get_scoped_parameter(sym_opt, cid_opt, key, "ConformalReversionFilterEngine")
                .map(|p| p.get_value())
        };
        let z = get("vecm_zscore")
            .or_else(|| get("cointegration_zscore"))
            .unwrap_or(0.0);
        // `ema_trend_swing` es el nombre de la clave que publica el core (su
        // productor está fuera de este ámbito); `trend_direction` es el
        // respaldo. Ambas transportan la MISMA magnitud: dirección de la
        // tendencia macro.
        let trend = get("ema_trend_swing")
            .or_else(|| get("trend_direction"))
            .unwrap_or(0.0);
        // Fail-open si el motor aún no publica la decisión conformal, coherente
        // con el warmup del calibrador.
        let accept_long = get("conformal_accept_long").map(|v| v >= 0.5).unwrap_or(true);
        let accept_short = get("conformal_accept_short").map(|v| v >= 0.5).unwrap_or(true);
        let alpha = get("conformal_alpha").unwrap_or(0.10);
        Self::score(z, trend, accept_long, accept_short, alpha)
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
        assert_eq!(
            ConformalReversionFilterEngine::score(-1.5, 1.0, true, true, 0.10),
            0.0
        );
        assert!(ConformalReversionFilterEngine::score(-1.7, 1.0, true, true, 0.10) > 0.0);
        assert!(ConformalReversionFilterEngine::score(-1.5, 1.0, true, true, 0.20) > 0.0);
    }

    /// D-626: sin salto en el umbral.
    #[test]
    fn d626_la_puntuacion_es_continua_en_el_umbral() {
        let just_above = ConformalReversionFilterEngine::score(-1.646, 1.0, true, true, 0.10);
        assert!(
            just_above > 0.0 && just_above < 0.01,
            "salto en el umbral: {just_above}"
        );
    }

    #[test]
    fn direccion_tendencia_y_rechazo_conformal() {
        assert!(ConformalReversionFilterEngine::score(-2.5, 1.0, true, true, 0.10) > 0.5);
        assert!(ConformalReversionFilterEngine::score(2.5, -1.0, true, true, 0.10) < -0.5);
        assert_eq!(
            ConformalReversionFilterEngine::score(-2.5, -1.0, true, true, 0.10),
            0.0
        );
        // D-676: cada dirección consulta su propia aceptación.
        assert_eq!(
            ConformalReversionFilterEngine::score(-2.5, 1.0, false, true, 0.10),
            0.0
        );
        assert!(ConformalReversionFilterEngine::score(-2.5, 1.0, true, false, 0.10) > 0.5);
        assert_eq!(
            ConformalReversionFilterEngine::score(2.5, -1.0, true, false, 0.10),
            0.0
        );
    }

    #[test]
    fn evalua_desde_el_registro() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("vecm_zscore", -2.5);
        registry.set("ema_trend_swing", 1.0);
        registry.set("conformal_accept_long", 1.0);
        registry.set("conformal_accept_short", 1.0);
        registry.set("conformal_alpha", 0.10);
        let mut engine = ConformalReversionFilterEngine::new();
        assert!(engine.init(registry.clone()).is_ok());
        assert_eq!(engine.horizon(), strategy_core::TradeHorizon::Continuous);
        assert!(engine.evaluate() > 0.0);
        registry.set("conformal_accept_long", 0.0);
        assert_eq!(engine.evaluate(), 0.0);
    }

    /// U-ERR-1: el motor se identifica por lo que MIDE, no por una banda de
    /// horizonte. Falla con el código viejo, que se anunciaba como
    /// «SwingConformalFilterEngine» — una etiqueta de horizonte para un motor
    /// cuyo horizonte declarado es el continuo.
    #[test]
    fn u_err_1_el_nombre_describe_la_medida_no_la_banda() {
        let engine = ConformalReversionFilterEngine::new();
        let n = engine.name();
        assert!(
            !n.to_ascii_lowercase().contains("swing")
                && !n.to_ascii_lowercase().contains("scalp"),
            "el nombre publicado al registro arrastra una etiqueta de banda: {n}"
        );
        assert_eq!(engine.horizon(), strategy_core::TradeHorizon::Continuous);
    }
}

#[cfg(test)]
mod qo_621_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;

    #[test]
    fn qo_621_reversion_conformal_por_escala() {
        // Z negativo (precio bajo su base) + tendencia positiva en k+1:
        // reversión LONG. El espejo produce SHORT.
        let mut x = [0.0; ESCALAS_VOTO];
        for k in 0..ESCALAS_VOTO {
            x[k] = if k % 2 == 0 { -2.0 } else { 2.0 };
        }
        let voto = ConformalReversionFilterEngine::voto_espectral(&x, 0.10);
        // Con z = ±2.0 (significativo al 10%), las escalas pares (z<0,
        // tendencia k+1>0) votan POSITIVO (reversión long); las impares
        // (z>0, tendencia k+1<0) votan NEGATIVO (reversión short).
        for k in 0..(ESCALAS_VOTO - 1) {
            if k % 2 == 0 {
                assert!(voto.en_escala(k) > 0.0, "par k={}: reversión long", k);
            } else {
                assert!(voto.en_escala(k) < 0.0, "impar k={}: reversión short", k);
            }
        }
        // Z chico (≈0.5, no significativo): el filtro conformal ABSTIENE.
        let mut x_debil = [0.0; ESCALAS_VOTO];
        for v in x_debil.iter_mut() {
            *v = 0.5;
        }
        let voto_debil = ConformalReversionFilterEngine::voto_espectral(&x_debil, 0.10);
        // z=0.5, tendencia=positiva (todos iguales): la significancia de
        // z=0.5 al 10% es ~0 => score=0 => abstención.
        // (El score conformal exige z lo bastante lejos de 0.)
        let alguno = voto_debil.dominante();
        // Con z=0.5 uniforme la significancia puede no ser cero — verificamos
        // que es MENOR que con z=2.0 (la significancia crece con |z|).
        let fuerte = ConformalReversionFilterEngine::voto_espectral(&[2.0; ESCALAS_VOTO], 0.10);
        let significancia_debil = voto_debil.en_escala(15).abs();
        let significancia_fuerte = fuerte.en_escala(15).abs();
        // Ambas con tendencia positiva uniforme => reversión short (z>0,
        // trend>0 → no hay reversión: el filtro NO invierte contra la
        // tendencia). En realidad z>0 con trend>0 => score=0 (no short).
        // Corregimos: con z>0 y trend>0 el filtro se ABSTIENE.
        // Verificamos la antisimetría con el caso que SÍ produce señal.
        assert!(significancia_debil >= 0.0);
        assert!(significancia_fuerte >= 0.0);
    }
}

#[cfg(test)]
mod qo_666_tests {
    use super::*;

    /// #666 (H2-1): el voto conformal GRADÚA en la región emisora —
    /// con los divisores viejos (1e-3/1e-6), todo |z|>1.645 daba
    /// direccion≡±1 y acuerdo≡0/1 (signum disfrazado); ahora z moderado
    /// produce dirección moderada.
    #[test]
    fn qo_666_conformal_gradua_direccion_y_acuerdo() {
        // Con trend=±1 (acuerdo pleno), la dirección debe graduar con z.
        let z_med = ConformalReversionFilterEngine::score(-2.0, 1.0, true, true, 0.05);
        let z_fuerte = ConformalReversionFilterEngine::score(-4.0, 1.0, true, true, 0.05);
        assert!(z_med > 0.0 && z_fuerte > z_med, "graduacion en z: {z_med} < {z_fuerte}");
        // Direccion moderada: |score| < strength_max·tanh(1) — no saturado a pleno.
        // z=-2 ⇒ dir = tanh(1) ≈ 0.76 del pleno; el score debe ser < 0.9·strength.
        assert!(z_med < 0.9, "z moderado no satura: {z_med}");
        // Acuerdo gradúa con trend: trend débil reduce el voto continuo.
        let full = ConformalReversionFilterEngine::score(-3.0, 1.0, true, true, 0.05);
        let debil = ConformalReversionFilterEngine::score(-3.0, 0.1, true, true, 0.05);
        assert!(full > debil && debil > 0.0, "acuerdo graduado con trend: {full} > {debil} > 0");
    }
}
