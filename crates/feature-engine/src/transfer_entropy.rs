//! TRANSFER ENTROPY SOBRE STREAMS DE EVENTOS (Ola XLVIII·D).
//!
//! Candidato del triage teórico (docs/TRIAGE_TEORICO_2026-09-29.md):
//! Schreiber (2000) — T_{X→Y} = I(Y⁺ ; X⁻ | Y⁻): la información que el
//! pasado de X aporta sobre el futuro de Y MÁS ALLÁ del propio pasado de Y.
//! La ASIMETRÍA es la propiedad que la correlación no tiene: mide
//! DIRECCIÓN de flujo de información, no co-movimiento.
//!
//! Complemento del Hawkes cruzado (α_cross mide contagio dentro de una
//! VENTANA de lag elegida; TE mide dirección sin elegir lag — cualquier
//! memoria de 1 paso que escape del propio Y). Contigo (operador) y con
//! el contagio XLV forma la tríada de liderazgo: quién emite (roles),
//! cuánto contagia dentro del lag (α), y hacia dónde fluye la información
//! sin supuesto de lag (TE).
//!
//! Contrato de transferencia:
//! - **Variable**: indicador binario de actividad por ventana común: para
//!   cada serie de timestamps, x_w = 1 si ≥1 evento cayó en la ventana w.
//! - **Operador**: TE de orden 1 con suavizado Krichevsky–Trofimov
//!   (add-½ en los conteos — evita log 0 e infinitos por celdas vacías;
//!   estimador documentado, no ad-hoc):
//!   T = Σ p(y⁺,y,x) · log₂ [ p(y⁺|y,x) / p(y⁺|y) ].
//! - **Unidades**: bits por paso de ventana.
//! - **Contorno**: <64 ventanas totales ⇒ None (la tabla 2×2×2 con KT
//!   necesita muestra; con menos, la medida no es estimable); streams sin
//!   solape temporal ⇒ None.
//! - **Identificabilidad**: con orden 1 mide el flujo de memoria de UN
//!   paso; memoria más larga queda en el residuo — documentado, no
//!   afirmado como TE total.
//! - **Coste**: O(n_ventanas) con tablas de conteo; sin hot path.
//! - **Falsación**: (a) series independientes ⇒ TE ≈ 0 en bits y en ambas
//!   direcciones (tolerancia medible, no cero — KT introduce sesgo
//!   positivo pequeño); (b) Y copia a X con retardo 1 ⇒ T_{X→Y} > umbral
//!   y T_{Y→X} ≈ 0 — LA ASIMETRÍA ES EL CONTRATO; (c) bidireccional
//!   simétrica ⇒ ambas direcciones comparables.

/// Tamaño de ventana por defecto para simbolizar streams de eventos
/// (200 ms: la escala del kernel de contagio más corto).
pub const VENTANA_MS_DEFAULT: u64 = 200;
/// Muestra mínima de ventanas para estimar la tabla 2×2×2 con KT.
pub const MIN_VENTANAS: usize = 64;
/// Suavizado Krichevsky–Trofimov.
const KT: f64 = 0.5;

/// Simboliza un stream de timestamps en la rejilla de ventanas común:
/// ind[w] = 1 si ≥1 evento cae en [t0 + w·Δ, t0 + (w+1)·Δ).
/// Devuelve (indicadores, t0, n_ventanas) con la rejilla que cubre la
/// unión de ambos streams (el llamador pasa ya t0/n para ambas series).
fn simbolizar(ts: &[u64], t0: u64, ventana_ms: u64, n_ventanas: usize) -> Vec<u8> {
    let mut ind = vec![0u8; n_ventanas];
    for &t in ts {
        let w = ((t.saturating_sub(t0)) / ventana_ms) as usize;
        if w < n_ventanas {
            ind[w] = 1;
        }
    }
    ind
}

/// TE de orden 1 entre dos series SÍMBOLO (0/1) ya alineadas, en bits.
/// Suavizado KT add-½ en todas las tablas. None si la muestra es mínima.
pub fn te_binaria_kt(x: &[u8], y: &[u8]) -> Option<f64> {
    let n = x.len().min(y.len());
    if n < MIN_VENTANAS {
        return None;
    }
    // Tablas: n[y⁺][y][x] y márgenes.
    let mut joint = [[[0.0_f64; 2]; 2]; 2]; // [y_fut][y][x]
    for t in 0..n.saturating_sub(1) {
        let (yf, y, xx) = (y[t + 1] as usize, y[t] as usize, x[t] as usize);
        joint[yf][y][xx] += 1.0;
    }
    let pasos = (n - 1) as f64;
    let mut te = 0.0_f64;
    for yf in 0..2 {
        for y in 0..2 {
            // Margen p(y⁺|y) una vez por (yf, y).
            let n_y = joint[0][y][0] + joint[0][y][1] + joint[1][y][0] + joint[1][y][1];
            let n_yf_dado_y = joint[yf][y][0] + joint[yf][y][1];
            let p_yf_dado_y = (n_yf_dado_y + KT) / (n_y + 2.0 * KT);
            for xx in 0..2 {
                let n_j = joint[yf][y][xx];
                // p(y⁺, y, x) y p(y⁺|y, x) con KT.
                let p_joint = (n_j + 2.0 * KT) / (pasos + 4.0 * 2.0 * KT);
                let n_yx = joint[0][y][xx] + joint[1][y][xx];
                let p_yf_dado_yx = (n_j + KT) / (n_yx + 2.0 * KT);
                te += p_joint * (p_yf_dado_yx / p_yf_dado_y).log2();
            }
        }
    }
    if te.is_finite() {
        Some(te.max(0.0)) // TE poblacional ≥ 0; el estimador puede dar ~0⁻
    } else {
        None
    }
}

/// Transfer entropy DIRECCIONAL entre dos streams de eventos (timestamps
/// ordenados). Simboliza ambos sobre la MISMA rejilla (t0 = mínimo global,
/// n_ventanas = span/ventana) y devuelve (T_{x→y}, T_{y→x}) en bits por
/// paso. None en cualquiera de las direcciones si la muestra es mínima o
/// los streams no comparten solape.
pub fn transfer_entropy_eventos(
    ts_x: &[u64],
    ts_y: &[u64],
    ventana_ms: u64,
) -> (Option<f64>, Option<f64>) {
    if ventana_ms == 0 || ts_x.len() < 2 || ts_y.len() < 2 {
        return (None, None);
    }
    let t0 = ts_x[0].min(ts_y[0]);
    let t_end = ts_x[ts_x.len() - 1].max(ts_y[ts_y.len() - 1]);
    let span = t_end.saturating_sub(t0);
    let n_ventanas = (span / ventana_ms + 1) as usize;
    if n_ventanas < MIN_VENTANAS {
        return (None, None);
    }
    let x = simbolizar(ts_x, t0, ventana_ms, n_ventanas);
    let y = simbolizar(ts_y, t0, ventana_ms, n_ventanas);
    (te_binaria_kt(&x, &y), te_binaria_kt(&y, &x))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Génesis determinista (mismo xorshift del resto del dominio).
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33)
        }
    }

    /// FALSACIÓN (a): streams INDEPENDIENTES ⇒ TE ≈ 0 en ambas direcciones.
    /// KT introduce un sesgo positivo pequeño que DECRECE con n — se mide,
    /// no se clama cero exacto: tolerancia empírica documentada (<0.01 bits
    /// con 20k ventanas).
    #[test]
    fn independientes_te_casi_cero() {
        let mut r = rng(0xBEEF);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut t = 1_000u64;
        for _ in 0..20_000 {
            t += VENTANA_MS_DEFAULT;
            if r() % 4 == 0 {
                xs.push(t);
            }
            if r() % 4 == 0 {
                ys.push(t + r() % 50);
            }
        }
        let (txy, tyx) = transfer_entropy_eventos(&xs, &ys, VENTANA_MS_DEFAULT);
        let (a, b) = (txy.unwrap(), tyx.unwrap());
        assert!(a < 0.01, "T(x→y)={a} en independientes — demasiado");
        assert!(b < 0.01, "T(y→x)={b} en independientes — demasiado");
    }

    /// FALSACIÓN (b) — LA ASIMETRÍA ES EL CONTRATO: y emite DESPUÉS de x
    /// (retardo 1 ventana) ⇒ T(x→y) sustancial y T(y→x) ≈ 0. La correlación
    /// no distingue esta dirección; TE sí.
    #[test]
    fn y_sigue_a_x_una_ventana_despues_te_direccional() {
        let mut r = rng(0x5EED);
        let mut xs = Vec::new();
        let mut ruido_y = Vec::new();
        let mut t = 1_000u64;
        for _ in 0..8_000 {
            t += VENTANA_MS_DEFAULT;
            if r() % 3 == 0 {
                xs.push(t);
            }
            // ruido propio de y sin relación con x:
            if r() % 8 == 0 {
                ruido_y.push(t + 150);
            }
        }
        // Acople limpio: y responde exactamente 1 ventana después de x.
        let mut ys: Vec<u64> = xs.iter().map(|&t| t + VENTANA_MS_DEFAULT).collect();
        ys.extend(ruido_y);
        ys.sort();
        let (txy, tyx) = transfer_entropy_eventos(&xs, &ys, VENTANA_MS_DEFAULT);
        let (a, b) = (txy.expect("muestra suficiente"), tyx.expect("muestra suficiente"));
        assert!(a > 0.15, "T(x→y)={a} — el líder debe fluir al seguidor");
        assert!(b < 0.05, "T(y→x)={b} — el seguidor no predice al líder");
        assert!(a > 3.0 * b.max(1e-6), "asimetría x→y dominante: {a} vs {b}");
    }

    /// FALSACIÓN (c): acople BIDIRECCIONAL simétrico (y repite x y x repite
    /// y con igual probabilidad) ⇒ ambas direcciones comparables.
    #[test]
    fn bidireccional_simetrica_te_comparable() {
        let mut r = rng(0xCAFE);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut t = 1_000u64;
        for _ in 0..6_000 {
            t += VENTANA_MS_DEFAULT;
            match r() % 3 {
                0 => {
                    xs.push(t);
                    ys.push(t + VENTANA_MS_DEFAULT);
                }
                1 => {
                    ys.push(t);
                    xs.push(t + VENTANA_MS_DEFAULT);
                }
                _ => {}
            }
        }
        let (txy, tyx) = transfer_entropy_eventos(&xs, &ys, VENTANA_MS_DEFAULT);
        let (a, b) = (txy.unwrap(), tyx.unwrap());
        let ratio = if b > 1e-9 { a / b } else { f64::INFINITY };
        assert!(
            (0.5..2.0).contains(&ratio),
            "bidireccional debe ser comparable: T(x→y)={a} T(y→x)={b} ratio={ratio}"
        );
    }

    /// Contornos: muestra mínima ⇒ None; ventana 0 ⇒ None; streams cortos
    /// ⇒ None. La honestidad de "sin muestra no hay medida" aplica.
    #[test]
    fn contornos_sin_muestra_son_none() {
        let corto: Vec<u64> = (0..5).map(|i| 1_000 + i * 100).collect();
        let (a, b) = transfer_entropy_eventos(&corto, &corto.clone(), VENTANA_MS_DEFAULT);
        assert!(a.is_none() && b.is_none(), "muestra mínima");
        let largo: Vec<u64> = (0..500).map(|i| 1_000 + i * VENTANA_MS_DEFAULT).collect();
        let (a, _) = transfer_entropy_eventos(&largo, &largo.clone(), 0);
        assert!(a.is_none(), "ventana 0");
    }
}
