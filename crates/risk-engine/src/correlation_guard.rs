//! D-748 — LA CORRELACIÓN SE MIDE O NO SE NOMBRA.
//!
//! # Qué estaba mal
//!
//! El «guard de correlación» no medía ninguna correlación. Contaba posiciones
//! abiertas en la misma dirección y las comparaba con un límite obtenido de
//! multiplicar un COEFICIENTE DE CORRELACIÓN por cinco:
//!
//! ```text
//!   max_allowed_cluster = (global_correlation_threshold · 5).round()
//! ```
//!
//! Un coeficiente de correlación vive en `[-1, 1]`; un número de posiciones es
//! un entero sin unidades. El `× 5` es un cambio de unidades inventado: con el
//! gen en 0,50 el límite salía 2 o 3 sin que nadie hubiese mirado si las dos
//! monedas se mueven juntas. Dos posiciones en ALTCOINs gemelas contaban lo
//! mismo que una en BTC y otra en un activo descorrelacionado.
//!
//! # Qué se hace ahora
//!
//! 1. **Se mide la correlación de verdad.** Los retornos están disponibles por
//!    moneda en el anillo de ticks del arena (`tick_ring`). Se llevan ambas
//!    series a una REJILLA TEMPORAL COMÚN —paso derivado del intervalo medio
//!    entre ticks del feed más lento, porque interpolar por debajo de él
//!    fabrica retornos nulos— y se calcula el coeficiente de Pearson de los
//!    retornos logarítmicos del mid.
//! 2. **El tamaño de muestra decide si la medida existe.** El error típico de
//!    un coeficiente de Pearson con `K` puntos es `≈ 1/√(K−3)`. Si ese error no
//!    permite distinguir la correlación del umbral que se quiere comprobar, NO
//!    hay medida: la posición se trata como correlacionada (el caso adverso),
//!    nunca como independiente.
//! 3. **El gen recupera su significado literal.** `global_correlation_threshold`
//!    es el umbral de correlación a partir del cual dos posiciones dejan de ser
//!    apuestas distintas y pasan a ser la misma apuesta repetida.
//! 4. **El límite de exposición sale del riesgo, no de un múltiplo.** Un grupo
//!    de `n` posiciones correlacionadas pierde a la vez: es UN evento de riesgo
//!    `n · r`, donde `r` es el riesgo por operación que el motor MIDE al
//!    dimensionar (`arena.riesgo_por_operacion`). Ese evento se somete al
//!    mismo control de ruina que gobierna cualquier otra fracción de riesgo del
//!    sistema (`ruin::clamp_ruin`: streak-bound + axioma del 25 %). Se veta
//!    cuando `(n + 1) · r` excede ese tope.
//!
//! Sin `r` medido todavía no hay con qué acotar la exposición; en ese caso se
//! rechaza duplicar una apuesta que no se sabe dimensionar, que es la postura
//! conservadora y no un número inventado.

use quantum_arena::state::CompactTick;

/// Ticks que se traen del anillo para estimar la correlación. El anillo tiene
/// 32 768 posiciones; se toma la cola reciente, que es la que describe el
/// régimen en el que se va a abrir la posición.
pub const MAX_TICKS_MUESTRA: usize = 512;

/// Puntos máximos de la rejilla común. Con el error típico de Pearson
/// `1/√(K−3)`, 256 puntos resuelven correlaciones de hasta ±0,063.
pub const MAX_PUNTOS_REJILLA: usize = 256;

#[inline]
fn mid(t: &CompactTick) -> f64 {
    let b = t.bid_price;
    let a = t.ask_price;
    if b > 0.0 && a > 0.0 {
        (b + a) * 0.5
    } else if b > 0.0 {
        b
    } else {
        a
    }
}

/// Intervalo medio entre ticks de la serie, en milisegundos. Es la resolución
/// REAL del feed: por debajo de ella una rejilla sólo repite el último precio
/// y fabrica retornos nulos que sesgan la correlación hacia cero.
#[inline]
fn intervalo_medio_ms(ticks: &[CompactTick]) -> Option<u64> {
    if ticks.len() < 2 {
        return None;
    }
    let t0 = ticks[0].timestamp;
    let t1 = ticks[ticks.len() - 1].timestamp;
    if t1 <= t0 {
        return None;
    }
    Some(((t1 - t0) / (ticks.len() as u64 - 1)).max(1))
}

/// Serie de log-precios sobre la rejilla `t0 + k·paso`, con el último precio
/// observado (LOCF). Devuelve cuántos puntos se pudieron construir.
fn log_precios_en_rejilla(
    ticks: &[CompactTick],
    t0: u64,
    paso_ms: u64,
    out: &mut [f64],
) -> usize {
    if ticks.is_empty() || paso_ms == 0 {
        return 0;
    }
    let mut idx = 0usize;
    let mut n = 0usize;
    for (k, slot) in out.iter_mut().enumerate() {
        let t = t0.saturating_add((k as u64).saturating_mul(paso_ms));
        while idx + 1 < ticks.len() && ticks[idx + 1].timestamp <= t {
            idx += 1;
        }
        if ticks[idx].timestamp > t {
            return n;
        }
        let m = mid(&ticks[idx]);
        if !(m > 0.0) || !m.is_finite() {
            return n;
        }
        *slot = m.ln();
        n = k + 1;
    }
    n
}

/// Coeficiente de correlación de Pearson. `None` si alguna serie es constante
/// (desviación nula: la correlación no está definida) o hay menos de dos
/// puntos.
#[inline]
pub fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len().min(b.len());
    if n < 2 {
        return None;
    }
    let inv = 1.0 / n as f64;
    let ma = a[..n].iter().sum::<f64>() * inv;
    let mb = b[..n].iter().sum::<f64>() * inv;
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return None;
    }
    let r = sab / (saa * sbb).sqrt();
    if r.is_finite() {
        Some(r.clamp(-1.0, 1.0))
    } else {
        None
    }
}

/// Muestra mínima para que un coeficiente de Pearson RESUELVA una correlación
/// del tamaño `resolucion`: el error típico `1/√(K−3)` debe caber en ella.
#[inline]
pub fn puntos_minimos(resolucion: f64) -> usize {
    let res = if resolucion.is_finite() {
        resolucion.clamp(0.01, 1.0)
    } else {
        1.0
    };
    // K ≥ 3 + 1/res²  ⇒  SE = 1/√(K−3) ≤ res
    let k = 3.0 + 1.0 / (res * res);
    (k.ceil() as usize).max(4)
}

/// Correlación MEDIDA entre los retornos de dos series de ticks.
///
/// `resolucion` es el tamaño de correlación que hay que poder distinguir
/// (el umbral genómico). Devuelve `None` cuando las series no se solapan en el
/// tiempo o la muestra no alcanza para resolver esa magnitud: quien llama debe
/// tratar ese caso como el adverso, no como independencia.
pub fn correlacion_de_retornos(
    a: &[CompactTick],
    b: &[CompactTick],
    resolucion: f64,
) -> Option<f64> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    // Ventana común: sin solape no hay nada que comparar.
    let t0 = a[0].timestamp.max(b[0].timestamp);
    let t_fin = a[a.len() - 1].timestamp.min(b[b.len() - 1].timestamp);
    if t_fin <= t0 {
        return None;
    }
    // El paso lo fija el feed MÁS LENTO: muestrear por debajo de su resolución
    // sólo replica precios y sesga la correlación.
    let paso = intervalo_medio_ms(a)?.max(intervalo_medio_ms(b)?);
    let puntos = (((t_fin - t0) / paso) as usize + 1).min(MAX_PUNTOS_REJILLA);
    let minimos = puntos_minimos(resolucion);
    if puntos < minimos + 1 {
        return None; // +1 porque los retornos son diferencias
    }
    let mut la = [0.0f64; MAX_PUNTOS_REJILLA];
    let mut lb = [0.0f64; MAX_PUNTOS_REJILLA];
    let na = log_precios_en_rejilla(a, t0, paso, &mut la[..puntos]);
    let nb = log_precios_en_rejilla(b, t0, paso, &mut lb[..puntos]);
    let n = na.min(nb);
    if n < minimos + 1 {
        return None;
    }
    let mut ra = [0.0f64; MAX_PUNTOS_REJILLA];
    let mut rb = [0.0f64; MAX_PUNTOS_REJILLA];
    for i in 1..n {
        ra[i - 1] = la[i] - la[i - 1];
        rb[i - 1] = lb[i] - lb[i - 1];
    }
    pearson(&ra[..n - 1], &rb[..n - 1])
}

pub struct CorrelationGuardEngine;

impl CorrelationGuardEngine {
    /// ¿La correlación medida convierte a las dos posiciones en la MISMA
    /// apuesta? Sin medida utilizable la respuesta es `true`: el caso adverso.
    ///
    /// El llamador ya filtró por DIRECCIÓN COMÚN (dos largos o dos cortos).
    /// Con la misma dirección, sólo la correlación POSITIVA las hace perder a
    /// la vez; sobre activos anticorrelacionados, cuando una pierde la otra
    /// gana: es una cobertura, no la misma apuesta.
    ///
    /// Auditoría PR #5 (D-750b): antes se comparaba `|r|`, de modo que una
    /// cobertura con r = −0,7 contaba como exposición duplicada y el guard
    /// vetaba justo la operación que diversifica.
    #[inline]
    pub fn es_la_misma_apuesta(correlacion_medida: Option<f64>, umbral_gen: f64) -> bool {
        let umbral = if umbral_gen.is_finite() {
            umbral_gen.clamp(0.01, 1.0)
        } else {
            0.01
        };
        match correlacion_medida {
            Some(r) if r.is_finite() => r >= umbral,
            _ => true,
        }
    }

    /// Veto por exposición direccional.
    ///
    /// `posiciones_misma_apuesta` es el número de posiciones YA abiertas que la
    /// medida declara la misma apuesta que la candidata.
    /// `riesgo_por_operacion` es la fracción de capital que el motor pierde si
    /// el stop se toca, MEDIDA al dimensionar cada orden.
    /// `q_perdida` es la probabilidad de pérdida observada.
    ///
    /// El grupo pierde a la vez: es un evento de riesgo `(n + 1) · r`, y se le
    /// aplica el mismo tope de ruina que a cualquier otra fracción del sistema.
    #[inline]
    pub fn veto_por_exposicion_direccional(
        posiciones_misma_apuesta: usize,
        riesgo_por_operacion: f64,
        q_perdida: f64,
    ) -> bool {
        if posiciones_misma_apuesta == 0 {
            return false;
        }
        let tope = crate::ruin::clamp_ruin(1.0, q_perdida);
        if !riesgo_por_operacion.is_finite() || riesgo_por_operacion <= 0.0 {
            // Sin riesgo medido no hay con qué acotar: no se duplica una
            // apuesta que no se sabe dimensionar.
            return true;
        }
        let riesgo_del_grupo = (posiciones_misma_apuesta as f64 + 1.0) * riesgo_por_operacion;
        riesgo_del_grupo > tope
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serie(n: usize, paso_ms: u64, precios: impl Fn(usize) -> f64) -> Vec<CompactTick> {
        (0..n)
            .map(|i| {
                let p = precios(i);
                CompactTick {
                    timestamp: 1_000_000 + i as u64 * paso_ms,
                    bid_price: p * 0.9999,
                    ask_price: p * 1.0001,
                    bid_qty: 1.0,
                    ask_qty: 1.0,
                }
            })
            .collect()
    }

    /// EL DEFECTO: el guard no medía correlación. Dos series IDÉNTICAS y dos
    /// series independientes producían exactamente la misma decisión porque
    /// sólo se contaban posiciones. Ahora la medida las distingue.
    #[test]
    fn la_correlacion_se_mide_de_verdad() {
        let a = serie(300, 100, |i| 100.0 + (i as f64 * 0.37).sin());
        let gemela = serie(300, 100, |i| 250.0 + 2.5 * (i as f64 * 0.37).sin());
        let r_gemela = correlacion_de_retornos(&a, &gemela, 0.5).expect("muestra suficiente");
        assert!(
            r_gemela > 0.95,
            "series proporcionales deben salir casi perfectamente correlacionadas: {r_gemela}"
        );

        // Serie independiente (secuencia determinista sin relación de fase).
        let otra = serie(300, 100, |i| {
            let x = i as f64;
            100.0 + (x * 1.913).sin() + (x * 0.577).cos()
        });
        let r_otra = correlacion_de_retornos(&a, &otra, 0.5).expect("muestra suficiente");
        assert!(
            r_otra.abs() < r_gemela,
            "la medida debe separar lo gemelo de lo independiente: {r_otra} vs {r_gemela}"
        );
    }

    /// D-750b — misma dirección sobre activos anticorrelacionados es una
    /// cobertura: no cuenta como la misma apuesta. Correlación positiva sí.
    #[test]
    fn d750b_la_anticorrelacion_con_la_misma_direccion_es_cobertura() {
        assert!(!CorrelationGuardEngine::es_la_misma_apuesta(Some(-0.7), 0.5));
        assert!(CorrelationGuardEngine::es_la_misma_apuesta(Some(0.7), 0.5));
        assert!(!CorrelationGuardEngine::es_la_misma_apuesta(Some(0.2), 0.5));
    }

    /// Sin medida utilizable se asume el caso adverso, jamás independencia.
    #[test]
    fn sin_medida_se_asume_la_misma_apuesta() {
        assert!(CorrelationGuardEngine::es_la_misma_apuesta(None, 0.5));
        // Muestra demasiado corta para resolver el umbral ⇒ no hay medida.
        let corta_a = serie(5, 100, |i| 100.0 + i as f64);
        let corta_b = serie(5, 100, |i| 50.0 - i as f64);
        assert!(correlacion_de_retornos(&corta_a, &corta_b, 0.2).is_none());
    }

    /// Series sin solape temporal no son comparables.
    #[test]
    fn sin_solape_no_hay_correlacion() {
        let a = serie(200, 100, |i| 100.0 + i as f64 * 0.01);
        let mut b = serie(200, 100, |i| 100.0 + i as f64 * 0.01);
        for t in b.iter_mut() {
            t.timestamp += 10_000_000;
        }
        assert!(correlacion_de_retornos(&a, &b, 0.5).is_none());
    }

    /// EL DEFECTO: el límite de posiciones salía de multiplicar un coeficiente
    /// por 5. Ahora sale del riesgo medido contra el tope de ruina del sistema.
    #[test]
    fn el_limite_sale_del_riesgo_medido() {
        // Con q = 0,50 el tope de un evento de riesgo es el axioma del 25 %.
        let tope = crate::ruin::clamp_ruin(1.0, 0.5);
        // Riesgo pequeño: caben varias apuestas correlacionadas.
        let r_pequeno = tope / 10.0;
        assert!(!CorrelationGuardEngine::veto_por_exposicion_direccional(
            2, r_pequeno, 0.5
        ));
        // Riesgo grande: la segunda ya excede el tope del grupo.
        let r_grande = tope * 0.8;
        assert!(CorrelationGuardEngine::veto_por_exposicion_direccional(
            1, r_grande, 0.5
        ));
        // La primera posición jamás se veta por exposición direccional.
        assert!(!CorrelationGuardEngine::veto_por_exposicion_direccional(
            0, r_grande, 0.5
        ));
    }

    /// Sin riesgo medido no se duplica una apuesta que no se sabe dimensionar.
    #[test]
    fn sin_riesgo_medido_no_se_duplica_la_apuesta() {
        assert!(CorrelationGuardEngine::veto_por_exposicion_direccional(
            1, 0.0, 0.5
        ));
        assert!(CorrelationGuardEngine::veto_por_exposicion_direccional(
            1,
            f64::NAN,
            0.5
        ));
    }

    /// La muestra mínima es la que resuelve el umbral, no un número redondo.
    #[test]
    fn la_muestra_minima_sale_del_error_tipico() {
        // Resolver ±0,5 exige K ≥ 3 + 4 = 7; resolver ±0,1 exige K ≥ 103.
        assert_eq!(puntos_minimos(0.5), 7);
        assert_eq!(puntos_minimos(0.1), 103);
        assert!(puntos_minimos(f64::NAN) >= 4);
    }
}
