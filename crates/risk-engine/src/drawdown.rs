//! D-744 — LA CAÍDA MÁXIMA NO ES UNA OPINIÓN: ES UNA PRUEBA DE HIPÓTESIS.
//!
//! # Qué estaba mal
//!
//! El gen `global_max_drawdown` nacía como `1 − taker·100` —con la comisión
//! por defecto, 0,95— y su banda evolutiva era [0,50; 0,99]. Ningún genoma
//! podía cortar por debajo de perder la mitad de la cuenta, y el genoma base
//! esperaba a perder el 95 % (con $13, a $0,65) antes de que el sistema inmune
//! aplanara. Peor: el mismo número se leía con TRES semánticas incompatibles
//! —cortacircuitos del host, veto de entradas del risk-engine (mezclado con un
//! 0,85 literal en régimen micro) y colchón de margen del orquestador—, de modo
//! que evolucionar el gen movía tres cosas distintas en direcciones distintas.
//!
//! # Qué es realmente un cortacircuitos de drawdown
//!
//! Arriesgando una fracción `r` del capital por operación, una racha de `k`
//! pérdidas consecutivas deja el capital en `(1 − r)^k`: una caída de
//! `1 − (1 − r)^k`. La pregunta correcta no es «¿cuánto estoy dispuesto a
//! perder?» sino «¿es esta caída compatible con el edge que mi dimensionado
//! asume?». Con probabilidad de pérdida `q`, el número esperado de rachas de
//! longitud `k` en `H` operaciones es ≈ `H·q^k`; la racha más larga que el
//! azar produce con probabilidad `α` cumple `H·q^k = α`, es decir
//!
//! ```text
//!   k(α) = ln(H/α) / ln(1/q)          DD_max = 1 − (1 − r)^{k(α)}
//! ```
//!
//! Superar `DD_max` significa que la realidad NO se comporta como el edge
//! declarado: eso es exactamente lo que un cortacircuitos debe detectar, y por
//! eso el umbral respira con el riesgo que el motor toma de verdad (`r`, medido
//! al dimensionar cada orden) y con su tasa de pérdida observada (`q`).
//!
//! El gen sobrevive con un significado claro y una sola lectura: su valor es la
//! CONFIANZA de la prueba, `α = 1 − gen`. Con el valor base 0,95 el motor corta
//! cuando la caída sólo ocurriría por azar el 5 % de las veces; su banda
//! [0,50; 0,99] es la banda de confianza [50 %, 99 %], y evolucionarla ya no
//! mueve el colchón de margen ni el veto por otro camino.

/// Horizonte de operaciones sobre el que se exige supervivencia. Es el MISMO
/// que usa el tope de ruina del dimensionado (`ruin::TRADE_HORIZON`): si el
/// sizing promete sobrevivir a la peor racha de H operaciones, el
/// cortacircuitos debe medir contra esa misma promesa.
pub use crate::ruin::TRADE_HORIZON;

/// Confianza de la prueba a partir del gen: `α = 1 − gen`, acotada al rango
/// donde la prueba tiene sentido (de 50 % a 99,9 % de confianza).
#[inline]
pub fn alfa_desde_gen(gen: f64) -> f64 {
    let g = if gen.is_finite() { gen.clamp(0.0, 1.0) } else { 0.95 };
    (1.0 - g).clamp(0.001, 0.50)
}

/// Racha de pérdidas consecutivas que el azar produce con probabilidad `alfa`
/// en `TRADE_HORIZON` operaciones, con probabilidad de pérdida `q`.
#[inline]
pub fn racha_por_azar(q: f64, alfa: f64) -> f64 {
    let q_c = q.clamp(0.01, 0.99);
    let a = alfa.clamp(1e-6, 0.99);
    let k = (TRADE_HORIZON / a).ln() / (1.0 / q_c).ln();
    if k.is_finite() {
        k.clamp(1.0, TRADE_HORIZON)
    } else {
        TRADE_HORIZON
    }
}

/// Caída máxima compatible con el edge declarado.
///
/// `riesgo_por_operacion`: fracción del capital que se pierde si el stop se
/// toca (medida al dimensionar). `q_perdida`: probabilidad de pérdida
/// observada. `gen_confianza`: el gen `global_max_drawdown` leído como
/// confianza de la prueba.
///
/// Sin riesgo medido todavía (r = 0) devuelve `None`: no hay evidencia sobre
/// la que decidir, y un cortacircuitos que dispara sin evidencia es ruido.
#[inline]
pub fn drawdown_compatible(
    riesgo_por_operacion: f64,
    q_perdida: f64,
    gen_confianza: f64,
) -> Option<f64> {
    if !riesgo_por_operacion.is_finite() || riesgo_por_operacion <= 0.0 {
        return None;
    }
    let r = riesgo_por_operacion.clamp(1e-6, 0.99);
    let k = racha_por_azar(q_perdida, alfa_desde_gen(gen_confianza));
    let dd = 1.0 - (1.0 - r).powf(k);
    if dd.is_finite() {
        // Una caída del 100 % no es un umbral: por encima del 95 % la cuenta
        // ya no opera con nada. Y por debajo del 2 % el ruido de una sola
        // operación dispararía el freno.
        Some(dd.clamp(0.02, 0.95))
    } else {
        None
    }
}

/// Media móvil exponencial del riesgo por operación: el peso de la última
/// orden es 1/n hasta n = `memoria`, y 1/memoria a partir de ahí — así la
/// primera orden no queda diluida ni la última manda sola.
#[inline]
pub fn actualizar_riesgo_ewma(previo: f64, nuevo: f64, memoria: f64) -> f64 {
    if !nuevo.is_finite() || nuevo <= 0.0 {
        return previo;
    }
    let m = memoria.max(2.0);
    if !previo.is_finite() || previo <= 0.0 {
        return nuevo;
    }
    previo + (nuevo - previo) / m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn el_umbral_crece_con_el_riesgo_tomado() {
        let suave = drawdown_compatible(0.01, 0.5, 0.95).unwrap();
        let agresivo = drawdown_compatible(0.10, 0.5, 0.95).unwrap();
        assert!(
            agresivo > suave,
            "arriesgar más por operación tolera más caída: {suave} vs {agresivo}"
        );
    }

    #[test]
    fn el_umbral_crece_cuando_se_pierde_mas_a_menudo() {
        let buen_edge = drawdown_compatible(0.05, 0.40, 0.95).unwrap();
        let mal_edge = drawdown_compatible(0.05, 0.65, 0.95).unwrap();
        assert!(
            mal_edge > buen_edge,
            "con más pérdidas esperadas la racha es más larga: {buen_edge} vs {mal_edge}"
        );
    }

    #[test]
    fn mas_confianza_exigida_tolera_mas_caida_antes_de_declarar_rota_la_hipotesis() {
        let c50 = drawdown_compatible(0.05, 0.5, 0.50).unwrap();
        let c99 = drawdown_compatible(0.05, 0.5, 0.99).unwrap();
        assert!(c99 > c50, "{c50} vs {c99}");
    }

    #[test]
    fn sin_riesgo_medido_no_hay_umbral() {
        assert!(drawdown_compatible(0.0, 0.5, 0.95).is_none());
        assert!(drawdown_compatible(f64::NAN, 0.5, 0.95).is_none());
    }

    #[test]
    fn el_genoma_base_ya_no_espera_a_perderlo_casi_todo() {
        // Riesgo típico del régimen micro (5 % del capital por operación) con
        // una tasa de pérdida de moneda al aire: el freno salta MUY por debajo
        // del 95 % que el gen imponía literalmente.
        let dd = drawdown_compatible(0.05, 0.5, 0.95).unwrap();
        assert!(dd < 0.60, "umbral demasiado permisivo: {dd}");
        assert!(dd > 0.10, "umbral demasiado nervioso: {dd}");
    }

    #[test]
    fn la_ewma_del_riesgo_arranca_en_la_primera_orden_y_no_salta_con_la_ultima() {
        let r1 = actualizar_riesgo_ewma(0.0, 0.05, 20.0);
        assert!((r1 - 0.05).abs() < 1e-12);
        let r2 = actualizar_riesgo_ewma(r1, 0.50, 20.0);
        assert!(r2 > 0.05 && r2 < 0.10, "{r2}");
    }
}
