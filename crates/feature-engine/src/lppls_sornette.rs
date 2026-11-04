//! LPPLS de Sornette: detector de criticidad log-periódica (mandato §11.7,
//! GLM-189). «Procesos auto-excitados de Hawkes multivariantes acoplados
//! con ecuaciones log-periódicas de ley de potencia (LPPLS de Sornette)
//! para anticipar puntos críticos de ruptura» — la cadena Hawkes YA vive
//! aquí (`hawkes.rs`/`hawkes_cross.rs`, GLM quinta ola); este módulo es la
//! pieza que faltaba de esa frase: el ajustador LPPLS.
//!
//! Verificación previa (norma GLM-186): LPPLS/Sornette = 0 apariciones en
//! main. Este núcleo es la primera implementación.
//!
//! **Modelo** (forma linealizada estándar de Zhou–Sornette): con
//! u = (t_c − t)/T_w el tiempo normalizado al futuro crítico,
//!
//!   p(t) = A + B·u^m + C·u^m·cos(ω·ln u + φ)
//!        = A + B·u^m + C1·u^m·cos(ω·ln u) + C2·u^m·sin(ω·ln u)
//!
//! (A, B, C1, C2) son LINEALES dado (t_c, m, ω): se resuelven por
//! ecuaciones normales 4×4 con pivoteo parcial (sin dependencias). La
//! grilla (t_c, m, ω) es la calibración inicial documentada del rango
//! empírico de Sornette: m ∈ {0.2, 0.4, 0.6}, ω ∈ {5..13} (la frecuencia
//! log-periódica universal ω ≈ 6.4 ± 1.5, ampliada) y t_c = t_end + δ·T_w
//! con δ ∈ {0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0}.
//!
//! **Evidencia**: fracción de varianza que SOLO la log-periodicidad
//! explica más allá de cualquier tendencia suave: ev = (SSE_lisa −
//! SSE_lppls)/SST con la alternativa lisa [1, u^m, u^{2m}]. La selección
//! del punto de grilla es por MÍNIMO SSE (máxima verosimilitud), NO por
//! máxima evidencia — la evidencia-max elige puntos degenerados donde la
//! alternativa lisa imita la envolvente (medido a mano antes de escribir
//! el núcleo, dos descartes de diseño).
//!
//! **Calibración medida sobre el nulo** (16 caminantes aleatorios
//! deterministas, LCG propio del contrato, σ=0.5, n=256): la evidencia
//! sola alcanza hasta 0.34 y cruza 0.10 en ~la mitad de las
//! realizaciones — la evidencia SOLA no es un detector. `estable`
//! (unanimidad de tres ventanas ancladas al final: completa, 75 % y
//! 50 % — ω ±2 bins, m ±0.4, t_c absoluo ±35 % del horizonte menor) NO
//! disparó en ninguna de las 16 realizaciones del nulo. La señal
//! compuesta documentada es `evidencia ≥ 0.10 && estable`; su consumo
//! natural es ACOPLADO al radio de ramificación κ de Hawkes (como pide
//! el §11.7): LPPLS anticipa el CUÁNDO, κ mide el contagio que lo
//! alimenta — ofrecido al consejo para el wiring.
//!
//! **Causalidad**: sólo la ventana dada; t_c es SIEMPRE futuro
//! (t_c > t_último + δ_mín·T_w). Coste: 3 ventanas × 243 puntos de
//! grilla × O(n) ≈ 1.5 Mflop por llamada — trivial a cadencia %64.
//!
//! **Honestidad numérica**: entradas no finitas/≤0 ⇒ Err (fall-closed,
//! jamás se inyecta cero); SST=0 (serie constante) ⇒ Err; ajuste singular
//! en un punto de grilla ⇒ ese punto se salta. El ajuste de ventana
//! completa recupera (t_c, m, ω) EXACTOS en sintética de mercado con
//! ruido (certificado en el contrato); en señal débil la evidencia cae
//! bajo el umbral SIN fabricar confianza — el estadístico responde a la
//! física, no al deseo.

/// Exponentes de la ley de potencias (rango empírico Sornette m ∈ (0,1)).
pub const GRID_M: [f64; 3] = [0.2, 0.4, 0.6];
/// Frecuencias log-periódicas (rango empírico ω ≈ 6.4 ± 1.5, ampliado).
pub const GRID_OMEGA: [f64; 9] = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0];
/// Horizontes críticos como fracción de la ventana (t_c = t_end + δ·T_w).
pub const GRID_DELTA_TC: [f64; 9] = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0];
/// Umbral de evidencia de la ventana completa (calibración medida).
pub const UMBRAL_EVIDENCIA: f64 = 0.10;
/// Umbral de evidencia de las ventanas de cola para la unanimidad.
pub const UMBRAL_EVIDENCIA_COLA: f64 = 0.05;
/// Longitud mínima de serie (las ventanas de cola necesitan masa).
pub const MIN_MUESTRAS: usize = 64;

/// Informe del ajuste LPPLS de una ventana de precios.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformeLppls {
    /// t_c estimado ADELANTE de la última muestra, en milisegundos
    /// (δ̂·T_w·paso_ms del ajuste de ventana completa).
    pub t_c_ms_adelante: f64,
    /// Exponente de la ley de potencias m̂ (bin de la grilla).
    pub m: f64,
    /// Frecuencia log-periódica ω̂ (bin de la grilla).
    pub omega: f64,
    /// Evidencia de la ventana completa: fracción de varianza que SÓLO
    /// la log-periodicidad explica más allá de la tendencia suave. NO
    /// usar sola: cruza el umbral en ~la mitad de los caminantes
    /// medidos (calibración en la cabecera del módulo).
    pub evidencia: f64,
    /// Evidencia de la ventana de cola 75 % (anclada al final).
    pub evidencia_cola75: f64,
    /// Evidencia de la ventana de cola 50 % (anclada al final).
    pub evidencia_cola50: f64,
    /// Unanimidad de las tres ventanas ancladas al final con evidencia
    /// sobre los umbrales. 0/16 en el nulo caminante medido. La señal
    /// compuesta documentada es `evidencia >= UMBRAL_EVIDENCIA &&
    /// estable`.
    pub estable: bool,
    /// Muestras de la ventana completa.
    pub n: usize,
}

/// Ajuste de UNA ventana: mejor (t_c, m, ω) por mínimo SSE.
#[derive(Debug, Clone, Copy)]
struct AjusteVentana {
    /// t_c absoluto en unidades de muestra GLOBAL (t0 + local).
    t_c_abs: f64,
    m: f64,
    omega: f64,
    evidencia: f64,
}

/// Resuelve un sistema lineal denso k×k (k ≤ 4, almacenado en arreglos
/// fijos) por Gauss–Jordan con pivoteo parcial. `None` si singular
/// (pivote < 1e-12): el punto de grilla se salta, sin pánico (RB-P1).
/// `k` parametrizado: la base lisa tiene rango 3 y su 4ª columna es CERO
/// — un solver 4×4 rígido moriría en el pivote nulo (cazado por el
/// oráculo local ANTES del push).
fn resolver(mat: [[f64; 4]; 4], rhs: [f64; 4], k: usize) -> Option<[f64; 4]> {
    let mut a = mat;
    let mut b = rhs;
    for c in 0..k {
        let mut piv = c;
        for r in (c + 1)..k {
            if a[r][c].abs() > a[piv][c].abs() {
                piv = r;
            }
        }
        if a[piv][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, piv);
        b.swap(c, piv);
        for r in 0..k {
            if r != c {
                let f = a[r][c] / a[c][c];
                for col in c..k {
                    a[r][col] -= f * a[c][col];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut x = [0.0f64; 4];
    for i in 0..k {
        x[i] = b[i] / a[i][i];
        if !x[i].is_finite() {
            return None;
        }
    }
    Some(x)
}

/// Fila de la matriz de diseño en u con parámetro m (y ω si `periodica`).
/// Liza: [1, u^m, u^{2m}, 0]; periódica: [1, u^m, u^m·cos, u^m·sin].
fn fila_design(u_i: f64, m: f64, omega: f64, periodica: bool) -> [f64; 4] {
    let um = u_i.powf(m);
    if periodica {
        let fase = omega * u_i.max(1e-300).ln();
        [1.0, um, um * fase.cos(), um * fase.sin()]
    } else {
        [1.0, um, um * um, 0.0]
    }
}

/// SSE del ajuste lineal por mínimos cuadrados con la base dada (3
/// columnas lizas o 4 periódicas). `None` si el sistema es singular o el
/// SSE no es finito.
fn sse_base(y: &[f64], u: &[f64], m: f64, omega: f64, periodica: bool) -> Option<f64> {
    let n = y.len();
    let k = if periodica { 4 } else { 3 };
    let mut xt_x = [[0.0f64; 4]; 4];
    let mut xt_y = [0.0f64; 4];
    for i in 0..n {
        let x = fila_design(u[i], m, omega, periodica);
        for r in 0..k {
            xt_y[r] += x[r] * y[i];
            for c in 0..k {
                xt_x[r][c] += x[r] * x[c];
            }
        }
    }
    let coef = resolver(xt_x, xt_y, k)?;
    let mut sse = 0.0f64;
    for i in 0..n {
        let x = fila_design(u[i], m, omega, periodica);
        let mut pred = 0.0f64;
        for r in 0..k {
            pred += coef[r] * x[r];
        }
        let e = y[i] - pred;
        sse += e * e;
    }
    if sse.is_finite() { Some(sse) } else { None }
}

/// Construye el vector u = (t_c − t)/T_w de la ventana (tiempo en
/// unidades de muestra global; t_c SIEMPRE futuro).
fn vector_u(n: usize, t0: usize, t_c: f64) -> Vec<f64> {
    let tw = (n - 1) as f64;
    (0..n).map(|i| (t_c - (t0 + i) as f64) / tw).collect()
}

/// Ajuste LPPLS de una ventana (y, índice global inicial t0): mejor
/// (t_c, m, ω) por mínimo SSE y su evidencia contra la alternativa lisa
/// en el MISMO punto de grilla. `None` si nada fitea (constante/singular).
fn ajustar_ventana(y: &[f64], t0: usize) -> Option<AjusteVentana> {
    let n = y.len();
    let t_end = (t0 + n - 1) as f64;
    let tw = (n - 1) as f64;
    let media = y.iter().sum::<f64>() / n as f64;
    let sst: f64 = y.iter().map(|v| (v - media) * (v - media)).sum();
    if !(sst > 0.0) {
        return None;
    }
    // (sse_lp, delta, m, omega) del mejor ajuste periódico por mínima SSE.
    let mut mejor: Option<(f64, f64, f64, f64)> = None;
    for &d in GRID_DELTA_TC.iter() {
        let t_c = t_end + d * tw;
        let u = vector_u(n, t0, t_c);
        for &m in GRID_M.iter() {
            // Alternativa lisa de la MISMA familia: si ya lo explica todo
            // (o es singular), este punto no puede aportar evidencia
            // log-periódica — se salta. El valor del SSE liso del GANADOR
            // se recalcula al final para la evidencia en su punto exacto.
            let lisa_util = match sse_base(y, &u, m, 0.0, false) {
                Some(v) => v >= 1e-12 * sst,
                None => false,
            };
            if !lisa_util {
                continue;
            }
            for &omega in GRID_OMEGA.iter() {
                let sse_lp = match sse_base(y, &u, m, omega, true) {
                    Some(v) => v,
                    None => continue,
                };
                let es_mejor = match mejor {
                    Some((mejor_sse, _, _, _)) => sse_lp < mejor_sse,
                    None => true,
                };
                if es_mejor {
                    mejor = Some((sse_lp, d, m, omega));
                }
            }
        }
    }
    let (sse_lp, d, m, omega) = mejor?;
    // Evidencia en el punto ganador contra la lisa del MISMO (d, m).
    let t_c = t_end + d * tw;
    let u = vector_u(n, t0, t_c);
    let sse_lisa = sse_base(y, &u, m, 0.0, false)?;
    let evidencia = (sse_lisa - sse_lp) / sst;
    if !evidencia.is_finite() {
        return None;
    }
    Some(AjusteVentana {
        t_c_abs: t_c,
        m,
        omega,
        evidencia,
    })
}

/// Ajuste LPPLS de una ventana de precios (n >= 64, todos > 0 y finitos,
/// no constante, `paso_ms` > 0). El informe trae el ajuste de la ventana
/// completa, las evidencias de las dos colas ancladas al final y la
/// unanimidad `estable`. Err = fall-closed (jamás ceros fabricados).
pub fn lppls_ventana(precios: &[f64], paso_ms: u64) -> Result<InformeLppls, &'static str> {
    if paso_ms == 0 {
        return Err("paso_ms debe ser > 0");
    }
    if precios.len() < MIN_MUESTRAS {
        return Err("serie demasiado corta (n >= 64)");
    }
    if precios.iter().any(|p| !p.is_finite() || *p <= 0.0) {
        return Err("precios deben ser finitos y > 0");
    }
    let n = precios.len();
    let t_end = (n - 1) as f64;
    let full = ajustar_ventana(precios, 0).ok_or("ventana degenerada (constante o singular)")?;
    let cola75 = ajustar_ventana(&precios[n / 4..], n / 4);
    let cola50 = ajustar_ventana(&precios[n / 2..], n / 2);
    let (ev75, ev50) = match (cola75, cola50) {
        (Some(a), Some(b)) => (a.evidencia, b.evidencia),
        _ => (f64::NAN, f64::NAN),
    };
    let estable = match (cola75, cola50) {
        (Some(a), Some(b)) => {
            let omegas = [full.omega, a.omega, b.omega];
            let emes = [full.m, a.m, b.m];
            let tcs = [full.t_c_abs, a.t_c_abs, b.t_c_abs];
            let dispersion_tc = tcs.iter().cloned().fold(f64::MIN, f64::max)
                - tcs.iter().cloned().fold(f64::MAX, f64::min);
            let horizonte_menor = tcs.iter().cloned().fold(f64::MAX, f64::min) - t_end;
            full.evidencia > UMBRAL_EVIDENCIA
                && a.evidencia > UMBRAL_EVIDENCIA_COLA
                && b.evidencia > UMBRAL_EVIDENCIA_COLA
                && (omegas.iter().cloned().fold(f64::MIN, f64::max)
                    - omegas.iter().cloned().fold(f64::MAX, f64::min))
                    <= 2.0
                && (emes.iter().cloned().fold(f64::MIN, f64::max)
                    - emes.iter().cloned().fold(f64::MAX, f64::min))
                    <= 0.4
                && dispersion_tc <= 0.35 * horizonte_menor
        }
        _ => false,
    };
    Ok(InformeLppls {
        t_c_ms_adelante: (full.t_c_abs - t_end) * paso_ms as f64,
        m: full.m,
        omega: full.omega,
        evidencia: full.evidencia,
        evidencia_cola75: ev75,
        evidencia_cola50: ev50,
        estable,
        n,
    })
}
