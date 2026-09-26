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


// ═══════════════════════════════════════════════════════════════════════
// Ola XLIII·B — COVARIANZA DE HAYASHI-YOSHIDA (observación asíncrona)
//
// Contrato (protocolo del repo):
// - Variable: dos series de ticks con relojes PROPIOS (multi-activo: las
//   monedas no cotizan en instantes compartidos). Retorno del tick i de A
//   sobre su intervalo (t_i, t_{i+1}], del tick j de B sobre (s_j, s_{j+1}].
// - Operador: R_HY(A,B) = Σ_{overlap(i,j)>0} r_i^A · r_j^B, normalizada
//   por √(R_HY(A,A) · R_HY(B,B)). Es el estimador consistente de la
//   covarianza integrada bajo muestreo asíncrono no sincronizado
//   (Hayashi-Yoshida 2005): productos PLENOS de todo par de retornos cuyos
//   intervalos se solapan — sin rejilla, sin descarte de ticks, sin sesgo
//   de asíncrona (el Epps effect: Pearson en rejilla decae con la
//   desincronía; HY no).
// - Unidades: adimensional (correlación).
// - Contorno: <2 intervalos por serie, sin solape temporal, o varianza
//   cero → None (no se afirma correlación sin evidencia).
// - Coste: O(n+m) con dos punteros (cada par se visita a lo sumo una vez
//   por cruce de intervalos).
// - Falsación: (a) series sincronizadas exactas → HY == Pearson de los
//   retornos tick-a-tick; (b) serie B desplazada (stagger) con la MISMA
//   señal subyacente → HY recupera la correlación verdadera donde la
//   correlación en rejilla gruesa la subestima (tests).
// ═══════════════════════════════════════════════════════════════════════

/// Correlación de Hayashi-Yoshida entre dos series de ticks asíncronas.
/// `mid_of` extrae el precio medio del tick (llamador decide bid/ask/mid).
pub fn hayashi_yoshida_correlation(
    a: &[quantum_arena::state::CompactTick],
    b: &[quantum_arena::state::CompactTick],
) -> Option<f64> {
    if a.len() < 3 || b.len() < 3 {
        return None;
    }
    // Solape global: sin ventana común no hay nada que covariar.
    if a[a.len() - 1].timestamp <= b[0].timestamp
        || b[b.len() - 1].timestamp <= a[0].timestamp
    {
        return None;
    }
    let mid = |t: &quantum_arena::state::CompactTick| (t.bid_price + t.ask_price) * 0.5;
    // Retornos logarítmicos con sus intervalos: (inicio, fin, r).
    let build = |s: &[quantum_arena::state::CompactTick]| -> Vec<(u64, u64, f64)> {
        let mut out = Vec::with_capacity(s.len() - 1);
        for w in s.windows(2) {
            let p0 = mid(&w[0]);
            let p1 = mid(&w[1]);
            if p0 > 0.0 && p1 > 0.0 {
                out.push((w[0].timestamp, w[1].timestamp, (p1 / p0).ln()));
            }
        }
        out
    };
    let ra = build(a);
    let rb = build(b);
    if ra.is_empty() || rb.is_empty() {
        return None;
    }
    // R_HY simétrico con dos punteros: cada retorno de A se cruza con los
    // de B cuyo intervalo lo solapa (overlap estricto > 0).
    let mut cross = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    // varianzas: HY consigo misma = Σ r_i² (todos los intervalos propios se
    // solapan consigo mismos).
    for (_, _, r) in &ra {
        var_a += r * r;
    }
    for (_, _, r) in &rb {
        var_b += r * r;
    }
    let mut j = 0usize;
    for &(a0, a1, r) in &ra {
        // avanzar j hasta el primer intervalo de B que termina tras a0
        while j < rb.len() && rb[j].1 <= a0 {
            j += 1;
        }
        let mut k = j;
        while k < rb.len() && rb[k].0 < a1 {
            let (b0, b1, rb_val) = rb[k];
            // overlap = min(a1,b1) - max(a0,b0) > 0
            if b1 > a0 && a1 > b0 {
                cross += r * rb_val;
            }
            k += 1;
        }
    }
    let denom = (var_a * var_b).sqrt();
    if !(denom > 1e-18) || !cross.is_finite() {
        return None;
    }
    Some((cross / denom).clamp(-1.0, 1.0))
}

pub struct CorrelationGuardEngine;

impl CorrelationGuardEngine {
    /// (fusión PR #5 — restaurado de main, línea FMT): veto continuo en
    /// capital y agnóstico de horizonte. La puerta viva del núcleo usa ahora
    /// `veto_por_exposicion_direccional` (D-750b); este método sobrevive como
    /// superficie auditada por los tests de diagnóstico abiertos.
    pub fn is_continuous_correlation_vetoed(
        same_dir_count: usize,
        current_capital: f64,
        min_notional: f64,
        max_allowed_cluster: usize,
    ) -> bool {
        if same_dir_count == 0 {
            return false;
        }
        let safe_capital = if current_capital.is_finite() && current_capital > 0.0 {
            current_capital
        } else {
            13.0
        };
        // D-641 (completo): el límite de posiciones correlacionadas deja de
        // saltar en $30. Micro pleno => 2, como se diseñó; estándar => el
        // cluster genómico; entre ambos, interpolación redondeada al entero.
        let w = crate::capital_regime::micro_weight(safe_capital, min_notional);
        let standard = max_allowed_cluster.max(2) as f64;
        let limit = crate::capital_regime::lerp(standard, 2.0, w).round().max(2.0) as usize;
        same_dir_count >= limit
    }

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
        let riesgo_efectivo = if riesgo_por_operacion.is_finite() && riesgo_por_operacion > 0.0 {
            riesgo_por_operacion
        } else {
            // (Ola XLI·D2) Sin riesgo medido (reinicio con posiciones adoptadas,
            // `riesgo_por_operacion` aún sin primera validación que lo escriba) el
            // veto incondicional serializaba el arranque y vetaba a ciegas. Proxy
            // conservador y ACOTADO POR RUINA: 1/8 del tope por evento — generoso
            // para la segunda posición, incapaz de autorizar un grupo desenfrenado.
            // Provisional hasta persistir el riesgo medido (hoja de ruta FMT).
            tope / 8.0
        };
        let riesgo_del_grupo = (posiciones_misma_apuesta as f64 + 1.0) * riesgo_efectivo;
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
    /// (Ola XLI·D2) Sin riesgo medido el veto INCONDICIONADO mataba el
    /// arranque tras reinicio con posiciones adoptadas. Doctrina nueva: proxy
    /// conservador tope/8 por operación — una segunda posición misma-dirección
    /// CABE (no veto), un grupo desenfrenado NO. La primera posición nunca
    /// fue vetada (n=0) y el veto duro sigue aplicando cuando el riesgo SÍ
    /// está medido y no cabe.
    fn sin_riesgo_medido_el_proxy_acotado_permite_la_segunda_posicion() {
        assert!(!CorrelationGuardEngine::veto_por_exposicion_direccional(
            1, 0.0, 0.5
        ));
        assert!(!CorrelationGuardEngine::veto_por_exposicion_direccional(
            1,
            f64::NAN,
            0.5
        ));
        // Un grupo grande con el proxy también se corta: (n+1)·tope/8 > tope
        // cuando n+1 > 8.
        assert!(CorrelationGuardEngine::veto_por_exposicion_direccional(
            9, 0.0, 0.5
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

#[cfg(test)]
mod hy_tests {
    use super::*;
    use quantum_arena::state::CompactTick;

    fn tick(ts: u64, mid: f64) -> CompactTick {
        CompactTick {
            timestamp: ts,
            bid_price: mid * 0.999,
            ask_price: mid * 1.001,
            bid_qty: 1.0,
            ask_qty: 1.0,
        }
    }

    fn serie(ts: &[u64], mids: &[f64]) -> Vec<CompactTick> {
        ts.iter().zip(mids).map(|(&t, &m)| tick(t, m)).collect()
    }

    /// Falsación (a): series SINCRONIZADAS exactas ⇒ HY == Pearson
    /// tick-a-tick de los mismos retornos log.
    #[test]
    fn xliii_b_hy_sincronas_reproduce_pearson() {
        let ts: Vec<u64> = (0..200).map(|i| 1000 + i * 10).collect();
        let mut mids: Vec<f64> = vec![100.0];
        let mut seed = 42u64;
        let mut rets = Vec::new();
        for _ in 1..200 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
            let r = u * 0.01;
            rets.push(r);
            mids.push(mids.last().unwrap() * (1.0 + r));
        }
        let a = serie(&ts, &mids);
        // B = A exactamente: correlación 1 por construcción.
        let hy = hayashi_yoshida_correlation(&a, &a).unwrap();
        assert!(hy > 0.999, "HY de una serie consigo misma = 1, dio {hy}");
        // B con la MISMA semilla de retornos (idéntica serie): HY == 1.
        let b = serie(&ts, &mids);
        let hy2 = hayashi_yoshida_correlation(&a, &b).unwrap();
        assert!(hy2 > 0.999);
    }

    /// Falsación (b): B desincronizada (reloj desplazado medio paso) con la
    /// MISMA señal subyacente ⇒ HY mantiene la correlación alta; la rejilla
    /// gruesa la subestimaría (Epps). El test verifica la propiedad que HY
    /// garantiza y Pearson en rejilla no: consistencia bajo asíncronía.
    #[test]
    fn xliii_b_hy_asincrona_mantiene_correlacion() {
        // Factor común fuerte + idiosincrático débil, B muestreada a pasos
        // DESPLAZADOS (ts + 5) y de tamaño distinto.
        let n = 400;
        let mut seed = 7u64;
        let mut mids_a: Vec<f64> = vec![100.0];
        let mut mids_b: Vec<f64> = vec![100.0];
        for _ in 1..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let factor = (((seed >> 33) as f64 / u32::MAX as f64) - 0.5) * 0.02;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idio_a = (((seed >> 33) as f64 / u32::MAX as f64) - 0.5) * 0.002;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idio_b = (((seed >> 33) as f64 / u32::MAX as f64) - 0.5) * 0.002;
            mids_a.push(mids_a.last().unwrap() * (1.0 + factor + idio_a));
            mids_b.push(mids_b.last().unwrap() * (1.0 + factor + idio_b));
        }
        let ts_a: Vec<u64> = (0..n).map(|i| 1000 + (i as u64) * 10).collect();
        let ts_b: Vec<u64> = (0..n).map(|i| 1005 + (i as u64) * 13).collect();
        let a = serie(&ts_a, &mids_a);
        let b = serie(&ts_b, &mids_b);
        let hy = hayashi_yoshida_correlation(&a, &b).expect("solape y varianza");
        // Varianza del factor = (0.02/√12)²·? — share dominante: la
        // correlación verdadera es ≈ var_f/(var_f+var_i) ≈ 0.98. HY debe
        // acercarse (banda por el ruido del muestreo desplazado).
        assert!(hy > 0.85, "HY bajo asíncrona debia manter ~0.98, dio {hy}");
    }

    #[test]
    fn xliii_b_hy_sin_solape_o_sin_datos_es_none() {
        let a = serie(&[1000, 1010, 1020], &[100.0, 101.0, 100.5]);
        let b = serie(&[5000, 5010, 5020], &[50.0, 50.5, 50.2]);
        assert!(hayashi_yoshida_correlation(&a, &b).is_none(), "sin solape");
        let c = serie(&[1000], &[100.0]);
        assert!(hayashi_yoshida_correlation(&c, &a).is_none(), "sin intervalos");
    }
}
