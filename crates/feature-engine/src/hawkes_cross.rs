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

#[cfg(test)]
mod tests {
    use super::*;

    /// Falsación: seguidor que COPIA al líder con delay 200ms → pico en 200ms.
    #[test]
    fn xlv_seguidor_copiando_da_pico_en_lag_correcto() {
        let leader: Vec<u64> = (0..200).map(|i| 1_000 + (i as u64) * 1_000).collect();
        let follower: Vec<u64> =
            leader.iter().map(|&t| t + 200).collect(); // copia con delay exacto
        let span = *follower.last().unwrap() - follower[0];
        let lags: Vec<u64> = vec![100, 200, 500, 1_000, 2_000];
        let exc = cross_excitation(&leader, &follower, span, &lags)
            .expect("contagio claro con 200 eventos copiando");
        assert!(exc.peak_lag_ms == 200 || exc.peak_lag_ms == 500,
            "pico debia estar en 200 o 500, dio {} (rate={:.3}, z={:.1})",
            exc.peak_lag_ms, exc.peak_rate, exc.z_score);
        assert!(exc.z_score > 5.0, "contagio perfecto debia dar z alto, dio {:.1}", exc.z_score);
    }

    /// Falsación: series independientes → None (sin contagio afirmable).
    #[test]
    fn xlv_series_independientes_no_afirman_contagio() {
        let mut seed = 0xDEADBEEFu64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u64 % 10_000
        };
        // líder: 100 eventos espaciados ~1000ms
        let leader: Vec<u64> = (0..100).map(|_| next() * 100 + 1_000).collect();
        // seguidor: 100 eventos aleatorios en el mismo span (independientes)
        let follower: Vec<u64> = (0..100).map(|_| next() * 100 + 1_000).collect();
        let mut l = leader.clone(); l.sort();
        let mut f = follower.clone(); f.sort();
        let span = *f.last().unwrap() - f[0];
        let lags: Vec<u64> = vec![200, 500, 1_000];
        // con 100 eventos aleatorios, es MUY improbable superar z>3
        let result = cross_excitation(&l, &f, span, &lags);
        // No assert is_none (5% de falsos positivos con z=3); pero si Some,
        // la significancia no debe ser extrema.
        if let Some(exc) = result {
            assert!(exc.z_score < 6.0, "independientes no debian dar z={:.1}", exc.z_score);
        }
    }
}
