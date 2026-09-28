//! EXCITACIÓN CRUZADA DE HAWKES (Ola XLV·B) — kernel de contagio.
//!
//! Primer paso hacia el Hawkes multivariado (T03): mide cómo la intensidad
//! de un activo LÍDER (BTC) excita la de un SEGUIDOR (altcoin), con la
//! asimetría temporal correcta (el líder precede, no al revés).
//!
//! Contrato (protocolo del repo):
//! - Variable: dos series de eventos (ts, magnitud opcional) de activos
//!   distintos, y un lag máximo de contagio.
//! - Operador: α_cross(δ) = (# de eventos del seguidor en (t_líder,
//!   t_líder+δ]) / (# de eventos del líder) — la fracción de eventos del
//!   líder que se sigue de un evento del seguidor dentro del lag δ. Es el
//!   estimador no paramétrico del kernel de excitación cruzada (discreto).
//! - Unidades: adimensional [0,1] por lag; la suma sobre lags da la
//!   "masa de contagio" total.
//! - Contorno: <20 eventos del líder, o <5 coincidencias en el mejor lag
//!   → None (no se afirma contagio sin soporte).
//! - Identificabilidad: el contagio DIRECCIONAL (líder→seguidor) se
//!   distingue del bidireccional comparando α_cross(A→B) vs α_cross(B→A).
//! - Coste: O(n_líder · n_seguidor_local) con ventana deslizante.
//! - Falsación: series independientes → α_cross ≈ tasa_base (test);
//!   seguidor copiando al líder con delay fijo → pico en ese lag (test).

/// Estimación del kernel de excitación cruzada entre dos series de eventos.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossExcitation {
    /// Lag (ms) con mayor tasa de excitación cruzada.
    pub peak_lag_ms: u64,
    /// Tasa en el lag pico: fracción de eventos del líder seguidos por un
    /// evento del seguidor dentro de (t, t+lag_pico].
    pub peak_rate: f64,
    /// Tasa base (lo esperado por azar): densidad de eventos del seguidor
    /// × ventana de lag. peak_rate >> base = contagio real.
    pub base_rate: f64,
    /// Significancia aproximada: (peak_rate − base) / √(base·(1−base)/n_líder).
    pub z_score: f64,
}

/// Calcula la excitación cruzada líder→seguidor para la rejilla de lags.
/// `leader_ts` y `follower_ts` deben estar ordenados ascendentemente.
pub fn cross_excitation(
    leader_ts: &[u64],
    follower_ts: &[u64],
    follower_span_ms: u64,
    lag_grid_ms: &[u64],
) -> Option<CrossExcitation> {
    if leader_ts.len() < 20 || follower_ts.len() < 5 || lag_grid_ms.is_empty() {
        return None;
    }
    // Densidad base del seguidor: eventos por ms.
    let span = follower_span_ms.max(1);
    let density = follower_ts.len() as f64 / span as f64;

    let n = leader_ts.len() as f64;
    let mut best = CrossExcitation {
        peak_lag_ms: 0,
        peak_rate: 0.0,
        base_rate: 0.0,
        z_score: 0.0,
    };

    for &lag in lag_grid_ms {
        if lag == 0 {
            continue;
        }
        // Para cada evento del líder, ¿hay un evento del seguidor en (t, t+lag]?
        let mut hits = 0usize;
        let mut fi = 0usize;
        for &lt in leader_ts {
            let window_end = lt.saturating_add(lag);
            // avanzar fi hasta el primer follower >= lt
            while fi < follower_ts.len() && follower_ts[fi] < lt {
                fi += 1;
            }
            // mirar si hay un follower en (lt, window_end]
            let mut found = false;
            let mut fj = fi;
            while fj < follower_ts.len() && follower_ts[fj] <= window_end {
                if follower_ts[fj] > lt {
                    found = true;
                    break;
                }
                fj += 1;
            }
            if found {
                hits += 1;
            }
        }
        let rate = hits as f64 / n;
        let base = density * lag as f64; // probabilidad por azar en ventana de `lag` ms
        let base_clamped = base.clamp(1e-9, 1.0 - 1e-9);
        let se = (base_clamped * (1.0 - base_clamped) / n).sqrt();
        let z = (rate - base_clamped) / se;
        if z > best.z_score {
            best = CrossExcitation {
                peak_lag_ms: lag,
                peak_rate: rate,
                base_rate: base_clamped,
                z_score: z,
            };
        }
    }

    // Umbral de significancia: z > 3.0 (99.7% una cola)
    if best.z_score < 3.0 {
        return None;
    }
    Some(best)
}

// ═══════════════════════════════════════════════════════════════════════
// Ola XLV·D — MATRIZ DE EXCITACIÓN MULTIVARIADA (T03 completo)
//
// Contrato:
// - Variable: N series de eventos (una por activo del universo) + rejilla
//   de lags comunes.
// - Operador: matriz α[i][j] = z-score del kernel de contagio i→j en el
//   mejor lag. La diagonal es 0 (auto-excitación ya medida por el Hawkes
//   univariado). El elemento α[i][j] > 3 significa que los eventos de i
//   PREDICEN los de j con significancia — la flecha del contagio.
// - Unidades: z-score (adimensional).
// - Contorno: cualquier par sin evidencia suficiente → 0 en esa celda
//   (ausencia de afirmación, no afirmación de ausencia).
// - Coste: O(N²·n_local) — N par evaluaciones, cada una con ventana
//   deslizante sobre n_seguidor eventos.
// - Identificabilidad: la matriz es ASIMÉTRICA por construcción —
//   α[i][j] ≠ α[j][i] identifica líder vs seguidor.
// - Falsación: universo con un líder que copian todos → columna del líder
//   con z altos, resto bajo (test); universo independiente → todo bajo.
// ═══════════════════════════════════════════════════════════════════════

/// Matriz de excitación cruzada N×N: α[i][j] = z-score del contagio i→j.
/// None si N < 2 (una matriz 1×1 no dice nada de estructura).
pub fn contagion_matrix(
    event_series: &[Vec<u64>],
    spans_ms: &[u64],
    lag_grid_ms: &[u64],
) -> Option<Vec<Vec<f64>>> {
    let n = event_series.len();
    if n < 2 || spans_ms.len() != n || lag_grid_ms.is_empty() {
        return None;
    }
    let mut matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue; // auto-excitación medida por Hawkes univariado
            }
            if let Some(exc) =
                cross_excitation(&event_series[i], &event_series[j], spans_ms[j], lag_grid_ms)
            {
                matrix[i][j] = exc.z_score;
            }
            // sin evidencia → 0 (no se afirma contagio)
        }
    }
    Some(matrix)
}

/// Resume la matriz de contagio en un escalar por activo: la SUMA de
/// z-scores de contagio EMITIDO (fila) y RECIBIDO (columna).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContagionRole {
    /// Σ_j α[i][j] — cuánto contagia este activo a los demás.
    pub emitted: f64,
    /// Σ_i α[i][j] — cuánto es contagiado por los demás.
    pub received: f64,
    /// emitted − received: >0 = líder neto, <0 = seguidor neto.
    pub net_role: f64,
}

pub fn contagion_roles(matrix: &[Vec<f64>]) -> Option<Vec<ContagionRole>> {
    let n = matrix.len();
    if n < 2 || matrix.iter().any(|r| r.len() != n) {
        return None;
    }
    let mut roles = Vec::with_capacity(n);
    for j in 0..n {
        let emitted: f64 = matrix[j].iter().sum();
        let received: f64 = (0..n).map(|i| matrix[i][j]).sum();
        roles.push(ContagionRole {
            emitted,
            received,
            net_role: emitted - received,
        });
    }
    Some(roles)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Falsación: seguidor que COPIA al líder con delay 200ms → pico en 200ms.
    #[test]
    fn xlv_seguidor_copiando_da_pico_en_lag_correcto() {
        let leader: Vec<u64> = (0..200).map(|i| 1_000 + (i as u64) * 1_000).collect();
        let follower: Vec<u64> =
            leader.iter().map(|&t| t + 200).collect();
        let span = *follower.last().unwrap() - follower[0];
        let lags: Vec<u64> = vec![100, 200, 500, 1_000, 2_000];
        let exc = cross_excitation(&leader, &follower, span, &lags)
            .expect("contagio claro con 200 eventos copiando");
        assert!(exc.peak_lag_ms == 200 || exc.peak_lag_ms == 500,
            "pico debia estar en 200 o 500, dio {} (rate={:.3}, z={:.1})",
            exc.peak_lag_ms, exc.peak_rate, exc.z_score);
        assert!(exc.z_score > 5.0, "contagio perfecto debia dar z alto, dio {:.1}", exc.z_score);
    }

    /// Falsación: series independientes → None o z bajo.
    #[test]
    fn xlv_series_independientes_no_afirman_contagio() {
        let mut seed = 0xDEADBEEFu64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u64 % 10_000
        };
        let leader: Vec<u64> = (0..100).map(|_| next() * 100 + 1_000).collect();
        let follower: Vec<u64> = (0..100).map(|_| next() * 100 + 1_000).collect();
        let mut l = leader.clone(); l.sort();
        let mut f = follower.clone(); f.sort();
        let span = *f.last().unwrap() - f[0];
        let lags: Vec<u64> = vec![200, 500, 1_000];
        if let Some(exc) = cross_excitation(&l, &f, span, &lags) {
            assert!(exc.z_score < 6.0, "independientes no debian dar z={:.1}", exc.z_score);
        }
    }

    /// Falsación matriz: un líder copiado por 2 seguidores → asimétrica.
    #[test]
    fn xlv_lider_con_seguidores_da_matriz_asimetrica() {
        let leader: Vec<u64> = (0..100).map(|i| 1_000 + (i as u64) * 2_000).collect();
        let s1: Vec<u64> = leader.iter().map(|&t| t + 150).collect();
        let s2: Vec<u64> = leader.iter().map(|&t| t + 300).collect();
        let series = vec![leader.clone(), s1, s2];
        let spans: Vec<u64> = series.iter().map(|s| s.last().unwrap() - s[0]).collect();
        let lags = vec![200u64, 500, 1_000];
        let matrix = contagion_matrix(&series, &spans, &lags).expect("matriz 3x3");
        assert!(matrix[0][1] > 3.0, "lider→s1 z={}", matrix[0][1]);
        assert!(matrix[0][2] > 3.0, "lider→s2 z={}", matrix[0][2]);
        assert!(matrix[1][0] < 3.0, "s1→líder z={} (asimetria)", matrix[1][0]);
        let roles = contagion_roles(&matrix).expect("roles");
        // El líder debe tener el net_role MÁS ALTO del universo (contagia
        // más de lo que es contagiado). s1 puede ser positivo también
        // (relay: leader→s1→s2), pero siempre MENOS que el líder.
        assert!(roles[0].net_role > 0.0, "lider net={}", roles[0].net_role);
        assert!(
            roles[0].net_role >= roles[1].net_role,
            "lider ({}) debia dominar a s1 ({})",
            roles[0].net_role, roles[1].net_role
        );
    }
}
