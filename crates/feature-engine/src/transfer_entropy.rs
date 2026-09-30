//! Transfer entropy binaria de orden 1: I(Y[t+1]; X[t] | Y[t]).
//!
//! Se estima actividad (>=1 evento por bin), NO retornos, intensidad ni edge.
//! La anchura del bin fija el paso predictivo y la memoria retenida. Esto NO
//! elimina el lag ni representa todo el espectro temporal. Schreiber (2000):
//! https://arxiv.org/abs/nlin/0001042. Un estimador en tiempo continuo es otro
//! contrato, no el resultado de reducir indefinidamente esta ventana.
//!
//! # Ley conjunta y unidades
//! Con M transiciones, q(a,b,c) = (N[a,b,c] + 1/2)/(M + 4), sobre ocho celdas.
//! Se marginaliza ESTA MISMA ley para todos los condicionales:
//! T = sum q(a,b,c) log2(q(a,b,c) q(b) / (q(a,b) q(b,c))).
//! Es CMI plug-in de una conjunta suavizada Dirichlet(1/2); cada condicional
//! binario dado (b,c) tiene la forma KT, pero el condicional marginal dado b
//! NO recibe otro prior independiente add-1/2. No es E[CMI | datos], ni un
//! estimador insesgado. El prior puede producir información positiva incluso
//! con una fuente constante: no confundir su regularización con evidencia.
//! El resultado está en bits por transición de ventana, no bits/s.
//!
//! # Cobertura y límites
//! La API explícita recibe un intervalo común observado [inicio, fin).
//! Solo usa bins completos. El llamador debe garantizar reloj/unidades
//! comunes y cobertura sin outages; timestamps de eventos no lo acreditan.
//! El wrapper histórico usa la intersección de los extremos como aproximación
//! declarada, nunca la unión. No es apropiado para certificar cobertura real.
//! 64 bins es una política heredada de soporte, no una cota de estimabilidad;
//! la API explícita permite elegir el mínimo (>=2) y devuelve los conteos.
//!
//! Conteo disperso exacto en u64, O(E log(E+1)) tiempo y O(1) memoria auxiliar,
//! E = eventos recibidos, sin recorrer/reservar cada bin vacío. El cálculo
//! final usa f64: conteos >2^53 no mantienen precisión de una unidad.
//! No hay test de significancia, corrección de múltiples pares/escalas,
//! condicionamiento por factores comunes ni conexión operativa a un veto.

/// Escala histórica de ejemplo. NO es una escala universal ni calibrada.
pub const VENTANA_MS_DEFAULT: u64 = 200;
/// Política heredada de soporte; no equivale a significancia o independencia.
pub const MIN_VENTANAS: usize = 64;
const KT: f64 = 0.5;

type Counts = [[[u64; 2]; 2]; 2];

/// Cobertura común DECLARADA por el llamador, en un único reloj de milisegundos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservationWindow {
    pub start_ms: u64,
    /// Extremo exclusivo; el último bin parcial no se utiliza.
    pub end_ms: u64,
    pub bin_width_ms: u64,
    /// Política de soporte explícita. Debe ser >=2; no mide n efectivo.
    pub min_windows: u64,
}

/// Errores de dominio/soporte, no rechazos de una orden de trading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferEntropyError {
    ZeroBinWidth,
    InvalidInterval,
    InvalidMinimumWindows,
    UnsortedTimestamps,
    InsufficientWindows,
    NumericalFailure,
}

/// Resultado auditable de la rejilla declarada; no contiene un p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferEntropyEstimate {
    pub x_to_y_bits: f64,
    pub y_to_x_bits: f64,
    pub complete_windows: u64,
    pub transitions: u64,
    pub discarded_tail_ms: u64,
    /// Índices [destino futuro][destino pasado][fuente pasada].
    pub counts_x_to_y: [[[u64; 2]; 2]; 2],
    pub counts_y_to_x: [[[u64; 2]; 2]; 2],
}

/// CMI de UNA distribución suavizada; todos sus márgenes se derivan de ella.
fn cmi(counts: &Counts, transitions: u64) -> Option<f64> {
    let mut q = [[[0.0; 2]; 2]; 2];
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                q[a][b][c] = counts[a][b][c] as f64 + KT;
            }
        }
    }
    let mass = transitions as f64 + 8.0 * KT;
    let mut result = 0.0;
    for a in 0..2 {
        for b in 0..2 {
            let ab = q[a][b][0] + q[a][b][1];
            let past = q[0][b][0] + q[0][b][1] + q[1][b][0] + q[1][b][1];
            for c in 0..2 {
                let bc = q[0][b][c] + q[1][b][c];
                result += q[a][b][c] / mass * ((q[a][b][c] / bc) / (ab / past)).log2();
            }
        }
    }
    // 8 positive cells: only a small floating-point residual may be clamped.
    // A substantial negative value is a defect, not "negative information".
    let tolerance = 64.0 * f64::EPSILON;
    (result.is_finite() && (-tolerance..=1.0 + tolerance).contains(&result))
        .then(|| result.clamp(0.0, 1.0))
}

/// Series binarias YA alineadas: igual longitud y símbolos exclusivamente 0/1.
/// None con dominio inválido o soporte inferior a la política histórica.
/// El nombre se conserva por compatibilidad; véase el contrato del prior arriba.
pub fn te_binaria_kt(x: &[u8], y: &[u8]) -> Option<f64> {
    if x.len() != y.len() || x.len() < MIN_VENTANAS || x.iter().chain(y).any(|&v| v > 1) {
        return None;
    }
    let mut counts = [[[0_u64; 2]; 2]; 2];
    for t in 0..x.len() - 1 {
        counts[y[t + 1] as usize][y[t] as usize][x[t] as usize] += 1;
    }
    cmi(&counts, (x.len() - 1) as u64)
}

fn sorted(ts: &[u64]) -> bool {
    ts.windows(2).all(|pair| pair[0] <= pair[1])
}

// A bin b affects transition b (past) and b-1 (future). Each iterator is
// ordered because the original timestamps were validated, duplicates allowed.
fn affected_transitions(
    ts: &[u64],
    observation: ObservationWindow,
    windows: u64,
    future: bool,
) -> impl Iterator<Item = u64> + '_ {
    ts.iter().filter_map(move |&t| {
        let bin = t.checked_sub(observation.start_ms)? / observation.bin_width_ms;
        if bin >= windows {
            return None;
        }
        let step = if future { bin.checked_sub(1)? } else { bin };
        (step < windows - 1).then_some(step)
    })
}

fn active(ts: &[u64], observation: ObservationWindow, bin: u64) -> usize {
    // bin < complete_windows implies start + bin * width <= end: no overflow.
    let left = observation.start_ms + bin * observation.bin_width_ms;
    let pos = ts.partition_point(|&t| t < left);
    usize::from(
        ts.get(pos)
            .is_some_and(|&t| (t - left) / observation.bin_width_ms == 0),
    )
}

/// Estima ambas direcciones sobre cobertura común explícita. Streams vacíos
/// son admisibles: significan silencio OBSERVADO según el contrato del caller.
/// Eventos fuera del intervalo/bins completos se ignoran; no se ordenan datos.
/// Los duplicados no aumentan la actividad binaria.
pub fn transfer_entropy_observada(
    ts_x: &[u64],
    ts_y: &[u64],
    observation: ObservationWindow,
) -> Result<TransferEntropyEstimate, TransferEntropyError> {
    use TransferEntropyError::*;
    if observation.bin_width_ms == 0 {
        return Err(ZeroBinWidth);
    }
    let span = observation
        .end_ms
        .checked_sub(observation.start_ms)
        .filter(|&span| span > 0)
        .ok_or(InvalidInterval)?;
    if observation.min_windows < 2 {
        return Err(InvalidMinimumWindows);
    }
    if !sorted(ts_x) || !sorted(ts_y) {
        return Err(UnsortedTimestamps);
    }
    let windows = span / observation.bin_width_ms;
    if windows < observation.min_windows {
        return Err(InsufficientWindows);
    }
    let transitions = windows - 1;
    let mut xy = [[[0_u64; 2]; 2]; 2];
    let mut yx = xy;
    // All transitions initially silent. Visit only those touching active bins.
    xy[0][0][0] = transitions;
    yx[0][0][0] = transitions;
    let mut candidates = [
        affected_transitions(ts_x, observation, windows, false).peekable(),
        affected_transitions(ts_x, observation, windows, true).peekable(),
        affected_transitions(ts_y, observation, windows, false).peekable(),
        affected_transitions(ts_y, observation, windows, true).peekable(),
    ];
    while let Some(t) = candidates
        .iter_mut()
        .filter_map(|it| it.peek().copied())
        .min()
    {
        let x = active(ts_x, observation, t);
        let y = active(ts_y, observation, t);
        let xf = active(ts_x, observation, t + 1);
        let yf = active(ts_y, observation, t + 1);
        xy[0][0][0] -= 1;
        yx[0][0][0] -= 1;
        xy[yf][y][x] += 1;
        yx[xf][x][y] += 1;
        for it in &mut candidates {
            while it.peek().is_some_and(|&next| next <= t) {
                it.next();
            }
        }
    }
    Ok(TransferEntropyEstimate {
        x_to_y_bits: cmi(&xy, transitions).ok_or(NumericalFailure)?,
        y_to_x_bits: cmi(&yx, transitions).ok_or(NumericalFailure)?,
        complete_windows: windows,
        transitions,
        discarded_tail_ms: span % observation.bin_width_ms,
        counts_x_to_y: xy,
        counts_y_to_x: yx,
    })
}

/// Wrapper histórico: aproxima cobertura por [max(primeros), min(últimos)).
/// Extremos de eventos NO prueban cobertura; para datos reales usar la API
/// explícita con lineage del feed. Sin solape, orden o soporte devuelve None.
/// Los timestamps están en ms, no ns; la resolución del reloj es parte del dato.
pub fn transfer_entropy_eventos(
    ts_x: &[u64],
    ts_y: &[u64],
    ventana_ms: u64,
) -> (Option<f64>, Option<f64>) {
    if ts_x.len() < 2 || ts_y.len() < 2 {
        return (None, None);
    }
    let observation = ObservationWindow {
        start_ms: ts_x[0].max(ts_y[0]),
        end_ms: ts_x[ts_x.len() - 1].min(ts_y[ts_y.len() - 1]),
        bin_width_ms: ventana_ms,
        min_windows: MIN_VENTANAS as u64,
    };
    match transfer_entropy_observada(ts_x, ts_y, observation) {
        Ok(value) => (Some(value.x_to_y_bits), Some(value.y_to_x_bits)),
        Err(_) => (None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generador congruencial lineal determinista para estas simulaciones.
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s >> 33
        }
    }

    /// FALSACIÓN (a): streams INDEPENDIENTES ⇒ TE ≈ 0 en ambas direcciones.
    /// Tolerancia empírica para ESTA simulación (<0.01 bits con 20k ventanas),
    /// no una garantía general de sesgo, independencia o significancia.
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
        let (a, b) = (
            txy.expect("muestra suficiente"),
            tyx.expect("muestra suficiente"),
        );
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
