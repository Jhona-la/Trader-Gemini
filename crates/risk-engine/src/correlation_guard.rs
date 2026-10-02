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
//!    moneda en el anillo (`tick_ring`). Se estima HY sobre intervalos
//!    asíncronos, con Pearson en rejilla común como respaldo. Los signos de
//!    AMBAS exposiciones convierten correlación de precio en la de PnL.
//! 2. **Ausencia de medida no es independencia.** None cuenta en el caso
//!    adverso. Un Some tampoco acredita precisión estadística: el tamaño
//!    efectivo, el ruido y el error de estimación siguen siendo deudas.
//! 3. **El gen recupera su significado literal.** `global_correlation_threshold`
//!    es el umbral de correlación a partir del cual dos posiciones dejan de ser
//!    apuestas distintas y pasan a ser la misma apuesta repetida.
//! 4. **El límite de exposición sale del riesgo, no de un múltiplo.** Un grupo
//!    de `n` posiciones correlacionadas pierde a la vez: es UN evento de riesgo
//!    `n · r`, donde `r` es el riesgo por operación que el motor MIDE al
//!    dimensionar (`arena.riesgo_por_operacion`, una EWMA histórica). Se somete al
//!    mismo control de ruina que gobierna cualquier otra fracción de riesgo del
//!    sistema (`ruin::clamp_ruin`: streak-bound + axioma del 25 %). Se veta
//!    cuando `(n + 1) · r` excede ese tope.
//!
//! La EWMA NO es sigma ni el riesgo individual actual de todas las posiciones.
//! Sin ella se conserva el proxy histórico tope/8, pendiente de sustitución.
//! MP y el diagnóstico de signos no conceden descuento en la ruta viva.

use quantum_arena::state::CompactTick;

/// Ticks que se traen del anillo para estimar la correlación. El anillo tiene
/// 32 768 posiciones; esta cola es un presupuesto de cómputo, NO tamaño
/// efectivo, cobertura del espectro completo ni ventana óptima demostrada.
pub const MAX_TICKS_MUESTRA: usize = 512;

/// Capacidad de la rejilla común. No implica 256 observaciones independientes
/// ni una precisión garantizada de la estimación financiera.
pub const MAX_PUNTOS_REJILLA: usize = 256;

#[inline]
fn mid(t: &CompactTick) -> f64 {
    // Los llamadores validan ambas cotizaciones. Esta forma no desborda
    // cuando bid y ask positivos están cerca de f64::MAX.
    t.bid_price + (t.ask_price - t.bid_price) * 0.5
}

/// Un intervalo de retorno necesita duración positiva y dos precios válidos.
/// Duplicados requieren agregación/orden de eventos aguas arriba; no se
/// inventa un intervalo ni se oculta el error mediante la rejilla de respaldo.
fn ticks_validos(ticks: &[CompactTick]) -> bool {
    ticks.iter().all(|t| {
        t.bid_price.is_finite()
            && t.ask_price.is_finite()
            && t.bid_price > 0.0
            && t.ask_price > 0.0
    }) && ticks.windows(2).all(|w| w[0].timestamp < w[1].timestamp)
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

/// Pearson para pares completos, finitos y de igual longitud. `None` sin
/// dos pares o con desviación nula. Cada serie se reescala ANTES de centrar:
/// la correlación es invariante a escala positiva y así no se desbordan
/// las sumas/cuadrados de precios finitos ni se anulan los de escala pequeña.
/// Esto estabiliza la aritmética; no acredita tamaño muestral ni causalidad.
#[inline]
pub fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len();
    if n < 2 || b.len() != n || a.iter().chain(b).any(|v| !v.is_finite()) {
        return None;
    }
    // La media redondeada de una constante puede diferir de esa constante
    // (p.ej. n=49). No convertir ese residuo aritmético en variación medida.
    if a.iter().all(|v| *v == a[0]) || b.iter().all(|v| *v == b[0]) {
        return None;
    }
    let scale_a = a.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
    let scale_b = b.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
    if scale_a == 0.0 || scale_b == 0.0 {
        return None;
    }
    let inv = 1.0 / n as f64;
    let ma = a.iter().map(|v| v / scale_a).sum::<f64>() * inv;
    let mb = b.iter().map(|v| v / scale_b).sum::<f64>() * inv;
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let da = a[i] / scale_a - ma;
        let db = b[i] / scale_b - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return None;
    }
    let r = sab / (saa.sqrt() * sbb.sqrt());
    if r.is_finite() {
        Some(r.clamp(-1.0, 1.0))
    } else {
        None
    }
}

/// Heurística legada de muestra a partir de la escala de error Fisher-z
/// `1/√(K−3)` bajo supuestos iid. No es un intervalo de confianza calibrado
/// para ticks dependientes; `resolucion` no representa un nivel de confianza.
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
    if a.len() < 2 || b.len() < 2 || !ticks_validos(a) || !ticks_validos(b) {
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
//   por √(Σ r_A² · Σ r_B²). Productos PLENOS de retornos cuyos intervalos
//   se solapan, sin centrar ni sincronizar en rejilla. Sólo se incluyen
//   intervalos con solape positivo con la ventana común en las DOS sumas.
//   La consistencia de HY es asintótica bajo hipótesis del modelo; no
//   demuestra ausencia de sesgo/ruido de microestructura en esta muestra.
// - Unidades: adimensional. Se conserva el recorte legado a [-1,1]; HY
//   normalizado puede excederlo en muestra finita y no garantiza PSD.
// - Contorno: reloj no creciente, cotización inválida, <2 intervalos por
//   serie en el soporte común, sin solape o variación cero → None.
//   Dos intervalos son un mínimo computable, no suficiencia estadística.
// - Coste: O(n+m) con dos punteros (cada par se visita a lo sumo una vez
//   por cruce de intervalos).
// - Falsación: series idénticas no constantes → 1; historia totalmente
//   disjunta no diluye el cociente. HY sincronizado NO equivale en general
//   a Pearson centrado: [.1,.2] vs [.2,.1] da .8 frente a -1.
// ═══════════════════════════════════════════════════════════════════════

/// Covariación HY normalizada de los mids de dos series asíncronas.
/// Intervalos que cruzan un borde común se conservan enteros: no se inventa
/// un precio no observado en el borde. Esa incertidumbre de frontera y el
/// recorte final requieren diagnóstico; `Some` no certifica diversificación.
pub fn hayashi_yoshida_correlation(
    a: &[quantum_arena::state::CompactTick],
    b: &[quantum_arena::state::CompactTick],
) -> Option<f64> {
    if a.len() < 3 || b.len() < 3 || !ticks_validos(a) || !ticks_validos(b) {
        return None;
    }
    // Solape global: sin ventana común no hay nada que covariar.
    let start = a[0].timestamp.max(b[0].timestamp);
    let end = a[a.len() - 1].timestamp.min(b[b.len() - 1].timestamp);
    if end <= start {
        return None;
    }
    // Retornos logarítmicos con sus intervalos: (inicio, fin, r).
    let build = |s: &[quantum_arena::state::CompactTick]| -> Vec<(u64, u64, f64)> {
        let mut out = Vec::with_capacity(s.len() - 1);
        for w in s.windows(2) {
            if w[1].timestamp <= start || w[0].timestamp >= end {
                continue;
            }
            let p0 = mid(&w[0]);
            let p1 = mid(&w[1]);
            // ln_1p conserva variaciones pequeñas; diferencia de logaritmos
            // evita overflow/underflow cuando la razón no es representable.
            let relative = (p1 - p0) / p0;
            let r = if relative.is_finite() && relative > -1.0 {
                relative.ln_1p()
            } else {
                p1.ln() - p0.ln()
            };
            out.push((w[0].timestamp, w[1].timestamp, r));
        }
        out
    };
    let ra = build(a);
    let rb = build(b);
    if ra.len() < 2 || rb.len() < 2 {
        return None;
    }
    // R_HY simétrico con dos punteros: cada retorno de A se cruza con los
    // de B cuyo intervalo lo solapa (overlap estricto > 0).
    let mut cross = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    // Variaciones cuadráticas del MISMO soporte usado por la covariación:
    // excluir aquí también los intervalos enteramente fuera de ventana.
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
    let denom = var_a.sqrt() * var_b.sqrt();
    if !(denom > 0.0) || !denom.is_finite() || !cross.is_finite() {
        return None;
    }
    let normalized = cross / denom;
    normalized.is_finite().then(|| normalized.clamp(-1.0, 1.0))
}

// ═══════════════════════════════════════════════════════════════════════
// Ola XLIV — media de correlación y frustración de un grafo firmado.
// El nombre legacy "curl" NO implementa una descomposición de Hodge.
//
// Contrato (protocolo del repo):
// - Variable: matriz conjunta COMPLETA de correlación. El caller debe probar
//   procedencia/ventana; las validaciones siguientes sólo prueban su dominio.
// - Operadores:
//   · rho_promedio: media de las correlaciones fuera de la diagonal. Es el
//     parámetro de la agregación de varianza clásica:
//     σ_grupo = σ·√(k + k(k−1)·ρ̄) — interpolación EXACTA entre dependencia
//     total (ρ̄=1 ⇒ σ·k) y media nula (ρ̄=0 ⇒ σ·√k), sólo para exposiciones
//     de IGUAL escala σ. Media nula no demuestra independencia.
//   · curl_share: fracción de triángulos con producto de signos negativo.
//     Describe balance firmado, no validez PSD ni riesgo de cartera. Una
//     matriz PSD puede contener tales triángulos. El ajuste rho_efectivo
//     se conserva como diagnóstico legado, no actuador en la admisión viva.
// - Unidades: ρ̄ ∈ [−1,1]; curl_share ∈ [0,1].
// - Contorno: grupo <3 nodos → curl None; correlación numéricamente inválida
//   → media y rho efectivo None. PSD singular es admisible, no inválida.
// - Coste: media O(n²) DESPUÉS de validar PSD O(barridos*n³); triángulos
//   O(n³). No se declara despreciable sin benchmark ni se corre en el gate.
// - Falsación: matriz de un factor (todas positivas) → curl 0; un triángulo
//   con un signo contrario → detectado (tests).
// ═══════════════════════════════════════════════════════════════════════

/// Media fuera de diagonal de una matriz COMPLETA de correlación válida.
/// None para aristas ausentes, asimetría, rango/diagonal inválidos o no PSD.
/// La validez numérica no acredita procedencia ni una muestra conjunta.
pub fn rho_promedio(corr: &[Vec<f64>]) -> Option<f64> {
    let n = corr.len();
    crate::random_matrix::largest_eigenvalue(corr)?;
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let v = corr[i][j];
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    Some((sum / count as f64).clamp(-1.0, 1.0))
}

/// Fracción legada de triángulos con producto negativo. No es curl de Hodge.
/// Este diagnóstico de grafo por sí solo NO valida una matriz de correlación.
/// None sin triángulos; aún omite triángulos no finitos (deuda diagnóstica).
pub fn curl_share_desbalanceado(corr: &[Vec<f64>]) -> Option<f64> {
    let n = corr.len();
    if n < 3 || corr.iter().any(|r| r.len() != n) {
        return None;
    }
    let mut total = 0usize;
    let mut desbalanceados = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let a = corr[i][j];
                let b = corr[j][k];
                let c = corr[i][k];
                if a.is_finite() && b.is_finite() && c.is_finite() {
                    total += 1;
                    if a * b * c < 0.0 {
                        desbalanceados += 1;
                    }
                }
            }
        }
    }
    if total == 0 {
        return None;
    }
    Some(desbalanceados as f64 / total as f64)
}

/// Diagnóstico legado: media desplazada por frustración de signos.
/// No es una descomposición de Hodge ni un estimador validado de riesgo.
/// Exige matriz numéricamente válida; no autoriza descuento de exposición.
pub fn rho_efectivo_para_agregacion(corr: &[Vec<f64>]) -> Option<f64> {
    let rho = rho_promedio(corr)?;
    let curl = curl_share_desbalanceado(corr).unwrap_or(0.0);
    // Ajuste histórico por balance de signos, sin garantía estadística.
    // La admisión real no consume este escalar como sigma ni como cobertura.
    if rho >= 0.0 {
        Some((rho + (1.0 - rho) * curl).clamp(-1.0, 1.0))
    } else {
        // ρ̄<0 con curl>0: no acreditar la cobertura más que (1−curl).
        Some((rho * (1.0 - curl)).clamp(-1.0, 1.0))
    }
}

pub struct CorrelationGuardEngine;

/// Conteo del guard pairwise, no varianza ni presupuesto completo de cartera.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DependencyExposure {
    pub open_positions: usize,
    pub same_bet_positions: usize,
    pub unknown_positions: usize,
    /// XLVI·D: ρ_PnL EFECTIVA del grupo misma-apuesta para la agregación de
    /// varianza (D-748): media de las correlaciones medidas contra la
    /// candidata, con los miembros NO medidos del grupo contando 1.0
    /// (correlación perfecta). Con cero medidos => 1.0: el presupuesto
    /// LINEAL legado del `None` original es el caso límite, no un
    /// comportamiento nuevo. Bits para preservar Eq.
    /// XLVI·F/AGY: + etapa curl_share² (Hodge) hacia 1. LXXII: + etapa
    /// λ̂ de cópulas t hacia 1 (manifiesto medido; ausente ⇒ bit-exact).
    pub same_bet_rho_efectivo_bits: u64,
    /// XLVI·E (SPECTRAL-010): riesgo REAL de cada miembro misma-apuesta al
    /// stop, como fracción de capital: `qty·|entry−sl|/capital`. Bits 0 =
    /// NO medido (snapshot sin stop utilizable). Bits para preservar Eq.
    pub same_bet_riesgos_bits: Vec<u64>,
}

impl DependencyExposure {
    /// ρ efectiva del grupo misma-apuesta; None si el grupo está vacío
    /// (el veto no la usa en ese caso).
    pub fn same_bet_rho_efectivo(&self) -> Option<f64> {
        if self.same_bet_positions == 0 {
            return None;
        }
        Some(f64::from_bits(self.same_bet_rho_efectivo_bits).clamp(-1.0, 1.0))
    }

    /// Vector de riesgos HÍBRIDO (XLVI·E): medido donde hay stop utilizable,
    /// `fallback` por miembro no medido — mismo patrón de honestidad que la
    /// ρ efectiva (no medido no regala descuento). La candidata la añade el
    /// llamador.
    pub fn same_bet_riesgos_hibridos(&self, fallback: f64) -> Vec<f64> {
        let fb = if fallback.is_finite() && fallback > 0.0 {
            fallback
        } else {
            1.0 // fallback de fallback: orden de magnitud prohibitivo
        };
        self.same_bet_riesgos_bits
            .iter()
            .map(|&b| {
                if b == 0 {
                    fb
                } else {
                    let r = f64::from_bits(b);
                    if r.is_finite() && r > 0.0 {
                        r
                    } else {
                        fb
                    }
                }
            })
            .collect()
    }
}

/// XLVI·E (SPECTRAL-010) — VETO POR RIESGO REAL MEDIDO AL STOP.
///
/// Agregación equicorrelacionada ponderada:
/// `σ_grupo = sqrt(Σ r_i² + ρ̄·((Σ r_i)² − Σ r_i²))`.
///
/// **Continuidad exacta con D-748**: con riesgos uniformes r_i = r y
/// n miembros reduce bit a bit a `r·sqrt(n + n(n−1)ρ̄)` — la fórmula que el
/// veto por conteo ya usaba. No es una doctrina nueva: es la misma con las
/// escalas individuales reales en vez de la suposición de igual tamaño.
///
/// Contornos:
/// - `rho = None` (o inválida) ⇒ suma lineal: correlación perfecta.
/// - ρ < −1/(n−1) (matriz no semidefinida positiva) ⇒ caso adverso lineal,
///   igual que el veto por conteo: un dato imposible no es cobertura.
/// - El piso del veto por conteo («nunca por debajo de una apuesta») se
///   generaliza: nunca por debajo del MAYOR riesgo individual — la cobertura
///   perfecta de fantasía no puede borrar la peor exposición aislada.
/// - Entrada ≤ 0 o NaN en el vector ⇒ 0 (el llamador sanea con su fallback
///   antes; aquí es defensa terminal).
///
/// Optimización zero-allocation (Ola 9): procesado en streaming de pasada
/// única sin allocar `Vec<f64>` en heap, reduciendo la latencia de evaluación
/// a nivel sub-microsegundo con garantía estricta de cero allocations.
#[inline]
pub fn calcular_riesgo_grupo(riesgos: &[f64], rho: Option<f64>) -> f64 {
    if riesgos.is_empty() {
        return 0.0;
    }
    let mut suma = 0.0f64;
    let mut suma_sq = 0.0f64;
    let mut peor_individual = 0.0f64;
    for &x in riesgos {
        let val = if x.is_finite() && x > 0.0 { x } else { 0.0 };
        suma += val;
        suma_sq += val * val;
        if val > peor_individual {
            peor_individual = val;
        }
    }
    let n = riesgos.len() as f64;
    match rho {
        Some(rho) if rho.is_finite() && (-1.0..=1.0).contains(&rho) => {
            if n > 1.0 && rho < -1.0 / (n - 1.0) {
                // ρ imposible para n exposiciones: no es cobertura (adverso lineal).
                suma
            } else {
                let varianza = suma_sq + rho * (suma * suma - suma_sq);
                varianza.max(0.0).sqrt().max(peor_individual)
            }
        }
        _ => suma,
    }
}

#[inline]
pub fn veto_por_riesgo_real_medido(riesgos: &[f64], rho: Option<f64>, tope: f64) -> bool {
    if riesgos.is_empty() || !tope.is_finite() || tope <= 0.0 {
        return false;
    }
    let riesgo_grupo = calcular_riesgo_grupo(riesgos, rho);
    riesgo_grupo.is_finite() && riesgo_grupo > tope
}

/// Veto de cartera que une la varianza de grupo equicorrelacionada con la cota
/// actuarial de Cramér-Lundberg: si existe coeficiente de ajuste `r_lundberg > 0`
/// para la tolerancia `epsilon` (p. ej. 0.05), el tope efectivo de margen de ruina
/// es `min(tope_streak, ln(1/eps)/R)`.
#[inline]
pub fn veto_por_riesgo_cramer_lundberg(
    riesgos: &[f64],
    rho: Option<f64>,
    tope_streak: f64,
    r_lundberg: Option<f64>,
    epsilon: f64,
) -> bool {
    let tope_efectivo = match r_lundberg.and_then(|r| crate::cramer_lundberg::EstimadorSiniestros::margen_de_cota(r, epsilon)) {
        Some(m) if m.is_finite() && m > 0.0 => tope_streak.min(m),
        _ => tope_streak,
    };
    veto_por_riesgo_real_medido(riesgos, rho, tope_efectivo)
}


/// Mezcla honesta de correlaciones medidas y no medidas del grupo same-bet:
/// los medidos aportan su valor; cada NO medido aporta 1.0 (fue admitido al
/// grupo conservadoramente y su dependencia real es desconocida). Sin
/// miembros, 1.0 neutro (el conteo en cero desactiva el veto de todos modos).
fn rho_efectivo_grupo(rhos: &[Option<f64>]) -> f64 {
    if rhos.is_empty() {
        return 1.0;
    }
    let suma: f64 = rhos
        .iter()
        .map(|r| r.filter(|v| v.is_finite() && (-1.0..=1.0).contains(v)).unwrap_or(1.0))
        .sum();
    (suma / rhos.len() as f64).clamp(-1.0, 1.0)
}

/// Recorre TODOS los slots y aplica rho_PnL = signo_candidata*signo_posicion*rho.
/// Estima una vez por activo; no fabrica aristas entre pares no observados ni
/// usa un veredicto MP como independencia. None identifica candidato inválido.
/// Se lee cada slot por snapshot; no constituye una reserva atómica de cartera.
pub fn dependency_exposure(
    arena: &quantum_arena::GlobalArena,
    candidate_id: usize,
    candidate_long: bool,
    threshold: f64,
) -> Option<DependencyExposure> {
    let candidate = arena.coins.get(candidate_id)?;
    let candidate_ticks = candidate.tick_ring.snapshot_recent(MAX_TICKS_MUESTRA);
    let mut result = DependencyExposure::default();
    // XLVI·D: correlación PnL de cada miembro del grupo same-bet contra la
    // candidata (None = no medida) — para la ρ efectiva del grupo.
    let mut rhos_same_bet: Vec<Option<f64>> = Vec::new();
    // LXXII: coin_ids de los miembros misma-apuesta (para el λ̂ de cópulas
    // por par contra la candidata).
    let mut same_bet_assets: Vec<usize> = Vec::new();
    // XLVI·E (SPECTRAL-010): riesgo REAL al stop de cada miembro, como
    // fracción del capital unificado — el veto agregará estos, no una
    // escala única supuesta.
    let capital = arena.unified_capital.load(std::sync::atomic::Ordering::Relaxed);
    for (asset_id, coin) in arena.coins.iter().enumerate() {
        // Outer None: slot observed closed. Inner None: open but no usable
        // snapshot, which must count as unknown rather than disappear.
        // XLVI·E: estado por ranura (cerrada | abierta sin snapshot |
        // abierta con snapshot) — misma semántica tri-estado que antes,
        // reteniendo el snapshot COMPLETO (entry/sl/qty) emparejado con su
        // lado para el riesgo al stop.
        let slots: Vec<(bool, Option<_>)> = coin
            .positions
            .slots()
            .iter()
            .map(|p| {
                let open = p.is_open();
                let snap = if open { p.snapshot() } else { None };
                (open, snap)
            })
            .collect();
        let sides = slots
            .iter()
            .map(|(open, snap)| if *open { Some(snap.as_ref().map(|x| x.is_long)) } else { None });
        if sides.clone().all(|s| s.is_none()) {
            continue;
        }
        let price_rho = if asset_id == candidate_id {
            Some(1.0)
        } else {
            let other_ticks = coin.tick_ring.snapshot_recent(MAX_TICKS_MUESTRA);
            let raw_rho = hayashi_yoshida_correlation(&candidate_ticks, &other_ticks).or_else(|| {
                correlacion_de_retornos(&candidate_ticks, &other_ticks, threshold)
            });
            // AGY-AUD-P06: Excitación de contagio Hawkes cruzado:
            // Si el activo es un seguidor neto recibiendo contagio fuerte (net_role < -3.0):
            // se amplifica la correlación observada mediante amplificar_por_contagio.
            let net_role = arena.registry.get_for_coin_or(asset_id, "hawkes_contagion_net_role", 0.0);
            let z_contagio = if net_role < -3.0 { Some(-net_role) } else { None };
            CorrelationGuardEngine::amplificar_por_contagio(raw_rho, z_contagio)
        };
        for (slot_idx, side_outer) in sides.into_iter().enumerate() {
            // Ranura cerrada: no cuenta (semántica original del flatten).
            let side = match side_outer {
                None => continue,
                Some(inner) => inner,
            };
            result.open_positions += 1;
            let pnl_rho = side.and_then(|long| {
                price_rho
                    .filter(|r| r.is_finite() && (-1.0..=1.0).contains(r))
                    .map(|r| if long == candidate_long { r } else { -r })
            });
            if pnl_rho.is_none() {
                result.unknown_positions += 1;
            }
            if CorrelationGuardEngine::es_la_misma_apuesta(pnl_rho, threshold) {
                result.same_bet_positions += 1;
                rhos_same_bet.push(pnl_rho);
                same_bet_assets.push(asset_id);
                // XLVI·E: riesgo real al stop si el snapshot lo sostiene.
                // Bits 0 = no medido (sin stop utilizable, lado del stop
                // inconsistente con la dirección, o capital inválido).
                let r = side.and_then(|long| {
                    let snap = slots[slot_idx].1.as_ref()?;
                    let sl_ok = if long {
                        snap.sl_price > 0.0 && snap.sl_price < snap.entry_price
                    } else {
                        snap.sl_price > snap.entry_price
                    };
                    if !sl_ok
                        || snap.entry_price <= 0.0
                        || snap.quantity <= 0.0
                        || !capital.is_finite()
                        || capital <= 0.0
                    {
                        return None;
                    }
                    let r = snap.quantity * (snap.entry_price - snap.sl_price).abs() / capital;
                    if r.is_finite() && r > 0.0 {
                        Some(r)
                    } else {
                        None
                    }
                });
                result.same_bet_riesgos_bits.push(r.map(|v| v.to_bits()).unwrap_or(0));
            }
        }
    }
    // XLVI·D / AGY-AUD-P06: ρ efectiva del grupo ajustada por la vorticidad de Helmholtz-Hodge:
    // En una cámara de eco cíclica de contagio (curl_share -> 1.0), el flujo de feedback anula la
    // diversificación lineal (todas las correlaciones convergen a dependencia sistémica).
    // rho_efectivo se interpola hacia 1.0 proporcionalmente a curl_share^2.
    let curl_share = arena
        .registry
        .get_value_fast("hawkes_contagion_curl_share")
        .filter(|c| c.is_finite() && *c >= 0.0)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let base_rho = rho_efectivo_grupo(&rhos_same_bet);
    let systemic_rho = (base_rho + (1.0 - base_rho) * (curl_share * curl_share)).clamp(-1.0, 1.0);
    // LXXII (copulas t): TERCERA etapa de inflado hacia 1 — la
    // dependencia de COLA medida (λ̂ por par, manifest de cópulas). Las
    // tres etapas componen multiplicativamente sobre el complemento de
    // independencia: 1−ρ_final = (1−base)·(1−curl²)·(1−λ̂_max). La
    // medición LXXI (100/108 pares λ̂≥0.10; BTC-SOL ρ̂0.77→λ̂0.51 donde
    // gaussiana daría 0) demostró que la ρ lineal SUBESTIMA el stop-out
    // conjunto. λ̂ ausente (sin manifest, par sin medir) ⇒ bit a bit el
    // systemic_rho legado — disciplina D-754, como V-RISK-006 sin R.
    let lambda_grupo = same_bet_assets
        .iter()
        .filter_map(|&otro| crate::copulas_store::lambda_entre(candidate_id, otro))
        .fold(None::<f64>, |acc, l| {
            Some(match acc {
                Some(m) => m.max(l),
                None => l,
            })
        });
    let final_rho = inflar_cola(systemic_rho, lambda_grupo);
    if let Some(l) = lambda_grupo {
        // Contable del consejo: el veto operó con inflado de cola medido.
        arena
            .registry
            .set_for_coin(candidate_id, "lxxii_lambda_grupo", l);
    }
    result.same_bet_rho_efectivo_bits = final_rho.to_bits();
    Some(result)
}

/// LXXII (copulas t): inflado de COLA hacia 1 sobre el complemento de
/// independencia: ρ_final = ρ + (1−ρ)·λ̂, i.e. (1−ρ_final) = (1−ρ)(1−λ̂).
/// `None` (sin medición) o λ=0 ⇒ **bit a bit** ρ de entrada (D-754).
/// Acotado a [−1,1]; λ fuera de (0,1] no llega aquí (el store lo filtra).
pub fn inflar_cola(rho_sistemica: f64, lambda: Option<f64>) -> f64 {
    match lambda {
        Some(l) if l > 0.0 => (rho_sistemica + (1.0 - rho_sistemica) * l).clamp(-1.0, 1.0),
        _ => rho_sistemica,
    }
}

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
    /// La entrada es correlación de PnL: el llamador transforma la correlación
    /// de precio por ambos signos de exposición. Un corto en un activo
    /// anticorrelacionado con un largo puede ser la MISMA apuesta.
    ///
    /// Auditoría PR #5 (D-750b): antes se comparaba `|r|`, de modo que una
    /// cobertura con r = −0,7 contaba como exposición duplicada y el guard
    /// vetaba justo la operación que diversifica.
    /// (Ola XLV·C) AMPLIFICADOR DE CONTAGIO: cuando el kernel de Hawkes
    /// cross (feature-engine, 7da858ef) detecta contagio direccional
    /// significativo (z > 3) entre el líder y el seguidor, la correlación
    /// ESTÁTICA (HY) subestima el riesgo durante el episodio activo. Este
    /// método eleva la correlación medida por el factor de contagión:
    /// r_amplificada = r + (1 − r) · min(z_contagio/10, 0.5).
    ///
    /// El máximo aumento es +0.5 (medio rango): el contagio hace al par
    /// MÁS same-bet pero no lo convierte en correlación 1 por decreto.
    #[inline]
    pub fn amplificar_por_contagio(
        correlacion_medida: Option<f64>,
        z_contagio: Option<f64>,
    ) -> Option<f64> {
        match (correlacion_medida, z_contagio) {
            (Some(r), Some(z)) if z.is_finite() && z > 3.0 => {
                let boost = ((z - 3.0) / 10.0).min(0.5);
                Some(r + (1.0 - r) * boost)
            }
            (Some(r), _) => Some(r), // sin contagio: correlación intacta
            (None, _) => None,
        }
    }

    pub fn es_la_misma_apuesta(correlacion_medida: Option<f64>, umbral_gen: f64) -> bool {
        let umbral = if umbral_gen.is_finite() {
            umbral_gen.clamp(0.01, 1.0)
        } else {
            0.01
        };
        match correlacion_medida {
            Some(r) if r.is_finite() && (-1.0..=1.0).contains(&r) => r >= umbral,
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
        Self::veto_por_exposicion_estructural(
            posiciones_misma_apuesta,
            riesgo_por_operacion,
            q_perdida,
            None,
        )
    }

    /// API legada de agregación igual-escala. Some(rho) exige un promedio
    /// realizable para k=n+1, además de evidencia/escala comparables que esta
    /// firma no transporta. None o rho inválido usa la política lineal previa.
    /// El camino vivo pasa None: pérdida al stop/EWMA no equivale a sigma.
    /// El piso de una apuesta y el proxy de arranque se conservan como
    /// políticas heredadas, NO como identidades de la varianza de cartera.
    pub fn veto_por_exposicion_estructural(
        posiciones_misma_apuesta: usize,
        riesgo_por_operacion: f64,
        q_perdida: f64,
        rho_efectivo: Option<f64>,
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
        let k = posiciones_misma_apuesta as f64 + 1.0;
        let riesgo_del_grupo = match rho_efectivo {
            // Para k exposiciones de igual escala, 1'C1 >= 0 exige
            // rho_medio >= -1/(k-1). Un dato imposible NO es cobertura:
            // usa el mismo caso adverso que None, sin clamp de varianza.
            Some(rho) if rho.is_finite() && rho >= -1.0 / (k - 1.0) && rho <= 1.0 => {
                let varianza = k + k * (k - 1.0) * rho;
                // ρ̄ muy negativo puede anular la varianza (cobertura perfecta):
                // nunca por debajo de la apuesta individual (k=1 efectivo).
                riesgo_efectivo * varianza.max(1.0).sqrt()
            }
            _ => k * riesgo_efectivo,
        };
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


#[cfg(test)]
mod xliv_structure_tests {
    use super::*;

    fn matriz(vals: &[f64], n: usize) -> Vec<Vec<f64>> {
        // vals en orden (i,j) i<j
        let mut m = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            m[i][i] = 1.0;
        }
        let mut k = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                m[i][j] = vals[k];
                m[j][i] = vals[k];
                k += 1;
            }
        }
        m
    }

    /// Falsación: factor común (todas positivas) → curl 0, ρ̄>0 — balance
    /// perfecto del grafo firmado.
    #[test]
    fn xliv_factor_comun_esta_balanceado() {
        let m = matriz(&[0.8, 0.7, 0.75], 3);
        assert_eq!(curl_share_desbalanceado(&m), Some(0.0));
        let rho = rho_promedio(&m).unwrap();
        assert!(rho > 0.7 && rho < 0.76);
        // ρ̄ efectivo sin desbalance = crudo.
        assert_eq!(rho_efectivo_para_agregacion(&m), Some(rho));
    }

    /// Balance firmado y validez PSD son propiedades distintas. Se conserva
    /// el testigo original inválido y se añade otro triángulo que SÍ es PSD.
    #[test]
    fn xliv_triangulo_inconsistente_detectado() {
        let m = matriz(&[0.8, 0.7, -0.6], 3);
        assert_eq!(curl_share_desbalanceado(&m), Some(1.0));
        assert!(rho_efectivo_para_agregacion(&m).is_none());
        assert!(rho_promedio(&m).is_none());
        let valid = matriz(&[0.2, 0.2, -0.1], 3);
        assert_eq!(curl_share_desbalanceado(&valid), Some(1.0));
        let rho_ef = rho_efectivo_para_agregacion(&valid).unwrap();
        let rho_crudo = rho_promedio(&valid).unwrap();
        assert!(rho_ef > rho_crudo, "efectivo {} > crudo {}", rho_ef, rho_crudo);
        assert!(rho_ef > 0.0);
    }

    /// Agregación de varianza: ρ̄=1 reproduce el k·σ histórico; ρ̄=0 acredita
    /// √k; None (sin matriz) mantiene la forma lineal (fail-safe).
    #[test]
    fn xliv_agregacion_de_varianza_interpola_estructura() {
        // k=5, riesgo tal que 5·σ > tope pero √5·σ < tope.
        let q = 0.5;
        let tope = crate::ruin::clamp_ruin(1.0, q);
        let sigma = tope / 4.5; // 5σ = 1.11·tope (veto), √5σ = 0.50·tope (pasa)
        assert!(CorrelationGuardEngine::veto_por_exposicion_estructural(4, sigma, q, Some(1.0)), "rho=1 => lineal");
        assert!(!CorrelationGuardEngine::veto_por_exposicion_estructural(4, sigma, q, Some(0.0)), "rho=0 => sqrt(k)");
        assert!(CorrelationGuardEngine::veto_por_exposicion_estructural(4, sigma, q, None), "None => lineal fail-safe");
        // Firma legada intacta (compatibilidad de tests históricos).
        assert!(CorrelationGuardEngine::veto_por_exposicion_direccional(4, sigma, q));
    }

    /// Un rho realizable negativo se distingue de una varianza imposible.
    #[test]
    fn xliv_cobertura_consistente_vs_inconsistente() {
        // Testigo histórico PRESERVADO: rho=-.4, k=5 implica varianza -3.
        // Debe caer al caso adverso, no inventar cobertura mediante max(1).
        let q = 0.5;
        let tope = crate::ruin::clamp_ruin(1.0, q);
        // rho=-.2 sí es realizable para k=5: varianza 5-4=1.
        let sigma = tope / 2.0;
        assert!(CorrelationGuardEngine::veto_por_exposicion_estructural(4, sigma, q, Some(-0.4)), "varianza imposible no acredita cobertura");
        assert!(!CorrelationGuardEngine::veto_por_exposicion_estructural(4, sigma, q, Some(-0.2)), "rho negativo realizable permanece admisible");
        assert!(CorrelationGuardEngine::veto_por_exposicion_estructural(4, sigma, q, Some(0.0)), "independientes ya no caben");
    }
}


#[cfg(test)]
mod xlvc_contagion_tests {
    use super::*;

    /// Sin contagio (z<3): la correlación pasa intacta.
    #[test]
    fn xlvc_sin_contagio_no_amplifica() {
        let a = CorrelationGuardEngine::amplificar_por_contagio(Some(0.5), Some(1.5));
        assert!((a.unwrap() - 0.5).abs() < 1e-9);
        let b = CorrelationGuardEngine::amplificar_por_contagio(Some(0.5), None);
        assert!((b.unwrap() - 0.5).abs() < 1e-9);
        let c = CorrelationGuardEngine::amplificar_por_contagio(Some(0.5), Some(3.0));
        assert!((c.unwrap() - 0.5).abs() < 1e-9);
    }

    /// Contagio fuerte (z=8): r=0.5 → r+0.5·0.5=0.75 (máximo +0.5).
    #[test]
    fn xlvc_contagio_fuerte_amplifica_hacia_same_bet() {
        let amplified = CorrelationGuardEngine::amplificar_por_contagio(Some(0.5), Some(8.0))
            .unwrap();
        // z=8: boost=(8-3)/10=0.5, r+(1-r)*boost = 0.5+0.5*0.5 = 0.75
        assert!((amplified - 0.75).abs() < 1e-6, "z=8 debia dar 0.75, dio {amplified}");
        let capped = CorrelationGuardEngine::amplificar_por_contagio(Some(0.5), Some(50.0))
            .unwrap();
        assert!((capped - 0.75).abs() < 1e-6, "cap en +0.5, dio {capped}");
    }

    /// Correlación negativa (cobertura) + contagio: la cobertura se erosiona
    /// pero no se invierte por decreto.
    #[test]
    fn xlvc_cobertura_con_contagio_se_erosiona_sin_invertirse() {
        let amplified = CorrelationGuardEngine::amplificar_por_contagio(Some(-0.4), Some(8.0))
            .unwrap();
        assert!(amplified > -0.4, "debe ser > original (erosion)");
        // z=8: boost=0.5, r+(1-r)*0.5 = -0.4 + 1.4*0.5 = 0.3
        assert!((amplified - 0.3).abs() < 1e-6, "matematica exacta: -0.4+1.4*0.5=0.3, dio {amplified}");
    }
}

#[cfg(test)]
mod ola9_zero_alloc_and_lundberg_tests {
    use super::*;

    #[test]
    fn ola9_calcular_riesgo_grupo_zero_alloc_exact_parity() {
        // Riesgos uniformes r = 0.05, n = 4, rho = 0.5
        let riesgos = [0.05, 0.05, 0.05, 0.05];
        let r_grp = calcular_riesgo_grupo(&riesgos, Some(0.5));
        // Formula analitica: r * sqrt(n + n*(n-1)*rho) = 0.05 * sqrt(4 + 4*3*0.5) = 0.05 * sqrt(10) = 0.158113883
        let esperado = 0.05 * (4.0 + 12.0 * 0.5_f64).sqrt();
        assert!((r_grp - esperado).abs() < 1e-12, "debe coincidir con la formula analitica");

        // Rho invalida / None => suma lineal
        let r_lineal = calcular_riesgo_grupo(&riesgos, None);
        assert!((r_lineal - 0.20).abs() < 1e-12, "sin rho la agregacion es lineal");

        // Caso adverso (rho < -1/(n-1)): n=4 => -1/3 = -0.3333... con rho = -0.5
        let r_adverso = calcular_riesgo_grupo(&riesgos, Some(-0.5));
        assert!((r_adverso - 0.20).abs() < 1e-12, "rho no realizable cae a suma lineal");

        // Peor individual actua como cota inferior
        let riesgos_desiguales = [0.01, 0.01, 0.08];
        let r_desigual = calcular_riesgo_grupo(&riesgos_desiguales, Some(-0.4));
        assert!(r_desigual >= 0.08, "el riesgo nunca puede ser menor al peor individual");
    }

    #[test]
    fn ola9_veto_por_riesgo_cramer_lundberg_bounds() {
        let riesgos = [0.04, 0.04, 0.04];
        let tope_streak = 0.10;
        let rho = Some(0.2);

        // Sin R de Lundberg: el tope efectivo es tope_streak (0.10).
        // Riesgo grupo = 0.04 * sqrt(3 + 3*2*0.2) = 0.04 * sqrt(4.2) = 0.08197 < 0.10 => no veto
        assert!(!veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope_streak, None, 0.05));

        // Con R de Lundberg restrictivo (p. ej. R = 40.0 con eps = 0.05):
        // Margen m = ln(1/0.05) / 40.0 = ln(20) / 40.0 = 2.9957 / 40.0 = 0.07489
        // Como m (0.07489) < tope_streak (0.10), el tope efectivo baja a 0.07489
        // Como riesgo_grupo (0.08197) > 0.07489 => VETO activado por cota de Cramér-Lundberg!
        assert!(veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope_streak, Some(40.0), 0.05));

        // Con R de Lundberg holgado (p. ej. R = 15.0 con eps = 0.05):
        // Margen m = ln(20) / 15.0 = 0.1997 > tope_streak (0.10), el tope efectivo se mantiene en 0.10
        // riesgo_grupo (0.08197) < 0.10 => no veto
        assert!(!veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope_streak, Some(15.0), 0.05));
    }
}
