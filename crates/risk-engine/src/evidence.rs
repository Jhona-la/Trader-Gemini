//! EVIDENCIA — NINGÚN ESTADÍSTICO ENTRA AL DIMENSIONADO POR SU VALOR PUNTUAL.
//!
//! # Qué estaba mal
//!
//! El productor del profit factor (`god-engine-core`) escribe, cuando aún no
//! hay pérdidas registradas, el literal **5,0**; y cuando no hay historial en
//! absoluto, **1,50**. Ambos son números inventados que el risk-engine
//! consumía como si fueran medidas:
//!
//! * Una ÚNICA operación ganadora bastaba para que `profit_factor` valiese 5,0.
//!   Con eso, `kelly_cold` abandonaba la rama de exploración (`PF ≤ 1`) y
//!   tomaba el bootstrap genómico, y `leverage_matrix::kelly_from_pf` calculaba
//!   `f* = p·(1 − 1/5) = 0,8·p`: el Kelly saltaba prácticamente a su techo con
//!   una sola observación.
//! * Sin historial, 1,50 afirma un edge del 50 % que nadie ha medido.
//!
//! Un profit factor sin pérdidas observadas **no vale 5: es desconocido**, y
//! lo desconocido, para quien dimensiona, equivale a «sin edge».
//!
//! # La cota
//!
//! `PF = Σ ganancias / Σ pérdidas`. Con `L = 0` el cociente no está definido,
//! así que se aplica la MISMA corrección de continuidad de Jeffreys que el
//! sistema ya adopta para el win rate (`EdgePosterior::jeffreys`, prior
//! Beta(½,½)): media operación de magnitud media ocupa el lugar de la pérdida
//! no observada.
//!
//! ```text
//!   m      = (W + L) / n                 magnitud media por operación
//!   L_corr = L + ½·m                     corrección de continuidad
//!   PF_pto = W / L_corr
//! ```
//!
//! Sobre el punto se aplica la incertidumbre del tamaño de muestra. `W` y `L`
//! son sumas de `n` magnitudes positivas; el error típico de `ln` de una suma
//! de `n` variables positivas es `cv/√n`, y para magnitudes de PnL el
//! coeficiente de variación `cv ≥ 1` (el caso exponencial, el menos
//! desfavorable de los realistas). Con las dos sumas contribuyendo de forma
//! independiente:
//!
//! ```text
//!   SE[ln PF] ≈ √(2/n)      PF_lcb = PF_pto · exp(−z·√(2/n))
//! ```
//!
//! Con `n = 1` y una sola ganancia el factor vale `exp(−2,32) ≈ 0,10`: el PF
//! resultante queda MUY por debajo de 1, es decir «sin edge probado», que es
//! exactamente lo que una sola operación permite afirmar. Con `n` grande el
//! factor tiende a 1 y la cota converge al PF medido. **El tamaño de muestra
//! gobierna**, que era el requisito.

use crate::kelly_envelope::EdgePosterior;

/// Nivel de confianza de TODA cota inferior del sistema: `z = 1,64` ≈ 95 %.
///
/// No es un parámetro de trading sino el nivel de confianza que la envolvente
/// bayesiana (F5.1, `kelly_envelope`) ya fijó para el win rate. Se expone aquí
/// para que el profit factor y la probabilidad de ganar se acoten con el MISMO
/// nivel: una sola lectura del 95 % en todo el motor.
pub const Z_LCB: f64 = 1.64;

/// Valor neutro de un profit factor DESCONOCIDO: 1,0 — exactamente el punto en
/// que no hay edge. Ni 1,50 ni 5,0: lo que no se ha medido no se apuesta.
pub const PF_DESCONOCIDO: f64 = 1.0;

/// Cota inferior del profit factor a partir de sus estadísticos suficientes.
///
/// `gross_wins` / `gross_losses` son las sumas de PnL (positivas) y `trades`
/// el número de operaciones cerradas. Devuelve [`PF_DESCONOCIDO`] mientras no
/// haya ninguna operación: sin evidencia no hay edge que dimensionar.
#[inline]
pub fn profit_factor_lcb(gross_wins: f64, gross_losses: f64, trades: f64) -> f64 {
    if !gross_wins.is_finite() || !gross_losses.is_finite() || !trades.is_finite() {
        return PF_DESCONOCIDO;
    }
    let w = gross_wins.max(0.0);
    let l = gross_losses.max(0.0);
    let n = trades.max(0.0);
    if n < 1.0 || (w + l) <= 0.0 {
        return PF_DESCONOCIDO;
    }
    // Corrección de continuidad de Jeffreys sobre el denominador.
    let magnitud_media = (w + l) / n;
    let l_corr = l + 0.5 * magnitud_media;
    if l_corr <= 0.0 {
        return PF_DESCONOCIDO;
    }
    let pf_punto = w / l_corr;
    let factor = (-Z_LCB * (2.0 / n).sqrt()).exp();
    let lcb = pf_punto * factor;
    if lcb.is_finite() {
        lcb.max(0.0)
    } else {
        PF_DESCONOCIDO
    }
}

/// Cota inferior de la probabilidad de ganar a partir de la frecuencia
/// observada y del tamaño de muestra.
///
/// Reconstruye el posterior Beta de Jeffreys —el MISMO que
/// [`EdgePosterior`]— desde el win rate publicado y el recuento de
/// operaciones: `α = ½ + n·w`, `β = ½ + n·(1 − w)`. Devuelve `None` sin
/// operaciones cerradas: no hay frecuencia observada que acotar.
///
/// La reconstrucción es aproximada porque el arena publica el win rate ya
/// posteriorizado y no los recuentos de ganadas/perdidas; el sesgo es de
/// media operación y desaparece con `n`.
#[inline]
pub fn win_rate_lcb(win_rate: f64, trades: f64) -> Option<f64> {
    if !win_rate.is_finite() || !trades.is_finite() || trades < 1.0 {
        return None;
    }
    let n = trades.min(1e9);
    let w = win_rate.clamp(0.0, 1.0);
    let post = EdgePosterior {
        alpha: 0.5 + n * w,
        beta: 0.5 + n * (1.0 - w),
    };
    let lcb = post.lcb(Z_LCB);
    if lcb.is_finite() {
        Some(lcb)
    } else {
        None
    }
}

/// Cota inferior bayesiana jerárquica (Empirical Bayes Shrinkage).
///
/// Para muestras pequeñas (n < 20), el estimador no colapsa a 0.0 ante una pérdida
/// ni a un valor sub-umbral que bloquee la acumulación de datos (evitando el deadlock
/// absorbente de rej(4)). Contrae la evidencia local hacia el prior del ensamble `p_prior`
/// con masa pseudo-muestral `n_prior`.
#[inline]
pub fn win_rate_hierarchical_lcb(
    win_rate: f64,
    trades: f64,
    p_prior: f64,
    n_prior: f64,
) -> f64 {
    let p0 = if p_prior.is_finite() && p_prior > 0.0 && p_prior < 1.0 {
        p_prior
    } else {
        0.55 // Prior por defecto del ensamble
    };
    let k0 = if n_prior.is_finite() && n_prior > 0.0 {
        n_prior
    } else {
        10.0 // 10 pseudomuestras de regularización
    };
    if !trades.is_finite() || trades < 1.0 || !win_rate.is_finite() {
        return p0;
    }
    let n = trades.min(1e9);
    let w = win_rate.clamp(0.0, 1.0);
    // Posterior Beta jerárquico: prior Beta(k0 * p0, k0 * (1 - p0)) + observaciones (n * w, n * (1 - w))
    let post = EdgePosterior {
        alpha: k0 * p0 + n * w,
        beta: k0 * (1.0 - p0) + n * (1.0 - w),
    };
    let lcb = post.lcb(Z_LCB);
    if lcb.is_finite() {
        lcb.clamp(0.05, 0.95)
    } else {
        p0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EL DEFECTO: una sola operación ganadora producía `PF = 5,0` y con él
    /// `kelly_from_pf` daba `0,8·p` — el Kelly casi en su techo con UNA
    /// observación. La cota inferior dice lo único que un dato permite decir:
    /// no hay edge probado.
    #[test]
    fn una_sola_ganancia_no_prueba_edge() {
        let pf = profit_factor_lcb(10.0, 0.0, 1.0);
        assert!(
            pf < 1.0,
            "una operación ganadora no puede declarar edge: {pf}"
        );
        // Y por tanto el Kelly derivado de ella es nulo.
        assert_eq!(crate::leverage_matrix::kelly_from_pf(0.7, pf), 0.0);
    }

    /// Sin historial el PF es DESCONOCIDO, no 1,50.
    #[test]
    fn sin_historial_el_pf_es_neutro() {
        assert_eq!(profit_factor_lcb(0.0, 0.0, 0.0), PF_DESCONOCIDO);
        assert_eq!(profit_factor_lcb(f64::NAN, 0.0, 10.0), PF_DESCONOCIDO);
    }

    /// El tamaño de muestra gobierna: el MISMO cociente observado produce una
    /// cota más alta cuanta más evidencia lo sostiene, y converge a él.
    #[test]
    fn el_tamano_de_muestra_gobierna_la_cota() {
        let poca = profit_factor_lcb(150.0, 100.0, 10.0);
        let media = profit_factor_lcb(1_500.0, 1_000.0, 100.0);
        let mucha = profit_factor_lcb(15_000.0, 10_000.0, 1_000.0);
        assert!(poca < media && media < mucha, "{poca} {media} {mucha}");
        // El punto medido (con la corrección de continuidad) es ~1,5: la cota
        // nunca lo supera y se le acerca con evidencia.
        assert!(mucha < 1.5, "la cota jamás excede el punto medido: {mucha}");
        assert!(mucha > 1.3, "con 1000 operaciones debe converger: {mucha}");
    }

    /// Sólo pérdidas: PF por los suelos, sin división por cero.
    #[test]
    fn solo_perdidas_no_explota() {
        let pf = profit_factor_lcb(0.0, 500.0, 50.0);
        assert!(pf.is_finite() && pf < 1.0, "{pf}");
    }

    #[test]
    fn la_cota_del_win_rate_es_conservadora_y_exige_muestra() {
        assert!(win_rate_lcb(0.6, 0.0).is_none());
        let p10 = win_rate_lcb(0.6, 10.0).unwrap();
        let p1000 = win_rate_lcb(0.6, 1_000.0).unwrap();
        assert!(p10 < 0.6 && p1000 < 0.6, "la cota va por debajo de la media");
        assert!(p1000 > p10, "más evidencia, cota más alta: {p10} {p1000}");
    }

    #[test]
    fn cota_jerarquica_no_colapsa_a_cero_en_n_1_evitando_deadlock() {
        // En n=0 devuelve el prior intacto
        assert_eq!(win_rate_hierarchical_lcb(0.5, 0.0, 0.55, 10.0), 0.55);

        // En n=1 con pérdida, NO colapsa a 0.0 (evita deadlock de rej(4))
        let p_loss_1 = win_rate_hierarchical_lcb(0.0, 1.0, 0.55, 10.0);
        assert!(p_loss_1 > 0.20, "n=1 con perdida no debe colapsar a 0: {p_loss_1}");

        // En n=1 con ganancia, refleja convicción inicial
        let p_win_1 = win_rate_hierarchical_lcb(1.0, 1.0, 0.55, 10.0);
        assert!(p_win_1 > p_loss_1 && p_win_1 > 0.30, "n=1 con ganancia: {p_win_1}");

        // Con muestra grande (n=1000), converge al valor medido y lava el prior
        let p_grande = win_rate_hierarchical_lcb(0.70, 1000.0, 0.55, 10.0);
        assert!((p_grande - 0.70).abs() < 0.05, "n grande converge a la media: {p_grande}");
    }
}
