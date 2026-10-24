//! Barrido de Monte Carlo de la prueba de reversión de la base (CL-97 y
//! C-F5b), compartido por `cf3_prueba_reversion.rs`,
//! `cf5b_prueba_escala_decision.rs` y la medición del crítico.
#![allow(dead_code)]

use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;
use strategy_core::prueba_reversion::{PruebaReversion, CRITICO, CRITICO_LARGO};

pub struct Lcg(pub u64);
impl Lcg {
    pub fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Resultado de un barrido, por edad (inmaduro cuenta como no rechazo):
/// - `rechazo`: el veredicto (las dos pruebas);
/// - `rechazo_corta` y `rechazo_larga`: cada prueba por separado;
/// - `q95` y `q95_largo`: cuantil 95 del estadístico de cada prueba en el
///   nodo k (C-F5f: no el del nodo que contrasta el veredicto, que antes de
///   madurar la prueba larga del k es otro más pequeño, CL-150);
/// - `reversion`: fracción de la desviación votada que revierte en τ_k,
///   −Σ d·Δx/Σ d², con d = base observada − media del nodo en el voto y Δx
///   el cambio de la base verdadera (sin ruido de medida) en τ_k. NaN con
///   menos de 5 votos o si el barrido no la sigue; `reversion_total`, la
///   misma con los votos de todas las edades.
pub struct Barrido {
    pub edades: Vec<f64>,
    pub rechazo: Vec<f64>,
    pub rechazo_corta: Vec<f64>,
    pub rechazo_larga: Vec<f64>,
    pub q95: Vec<f64>,
    pub q95_largo: Vec<f64>,
    pub reversion: Vec<f64>,
    pub reversion_total: f64,
}

/// Componente de la base: OU con `theta` por ms y desviación típica
/// estacionaria `sd`, o caminata (`theta` = 0) con innovación `sd` por
/// √`referencia_ms`.
#[derive(Clone, Copy)]
pub struct Componente {
    pub theta: f64,
    pub sd: f64,
    pub referencia_ms: f64,
}

/// `n` trayectorias de la base en el nodo `k`: OU con `theta` (por ms; 0 =
/// caminata con innovación s por √paso), ruido de medida `r`·s, pares cada
/// `paso` ms de media. Mira el veredicto a τ_k en cada edad (en memorias).
pub fn barrido(k: usize, n: u64, theta: f64, r: f64, paso: f64, edades: &[f64], semilla: u64) -> Barrido {
    let c = Componente {
        theta,
        sd: 1e-4,
        referencia_ms: paso,
    };
    barrido_con(k, n, &[c], r, paso, edades, semilla, false, None)
}

/// Como [`barrido`], con una base suma de componentes y la reversión de lo
/// votado. El ruido de medida es `r` veces la desviación típica de las
/// componentes OU (o la innovación de la primera, si no hay ninguna).
pub fn barrido_mezcla(
    k: usize,
    n: u64,
    comps: &[Componente],
    r: f64,
    paso: f64,
    edades: &[f64],
    semilla: u64,
) -> Barrido {
    barrido_con(k, n, comps, r, paso, edades, semilla, true, None)
}

/// C-F5e: como [`barrido_mezcla`], con volatilidad por rachas: las
/// innovaciones y el ruido de medida se multiplican por v = e^{g − s²}, con
/// g un OU de desviación típica `s` y vida media `semivida_celdas` celdas
/// (E[v²] = 1).
#[allow(clippy::too_many_arguments)]
pub fn barrido_rachas(
    k: usize,
    n: u64,
    comps: &[Componente],
    r: f64,
    paso: f64,
    edades: &[f64],
    semilla: u64,
    (s, semivida_celdas): (f64, f64),
) -> Barrido {
    barrido_con(k, n, comps, r, paso, edades, semilla, true, Some((s, semivida_celdas)))
}

#[allow(clippy::too_many_arguments)]
fn barrido_con(
    k: usize,
    n: u64,
    comps: &[Componente],
    r: f64,
    paso: f64,
    edades: &[f64],
    semilla: u64,
    seguir_reversion: bool,
    rachas: Option<(f64, f64)>,
) -> Barrido {
    let tau = SPECTRUM_SCALES_MS[k];
    let theta_g = rachas.map_or(0.0, |(_, h)| std::f64::consts::LN_2 / (h * tau / 8.0));
    let memoria = 16.0 * tau;
    let s0 = {
        let v: f64 = comps.iter().filter(|c| c.theta > 0.0).map(|c| c.sd * c.sd).sum();
        if v > 0.0 {
            v.sqrt()
        } else {
            comps.first().map_or(1e-4, |c| c.sd)
        }
    };
    let ne = edades.len();
    let mut rech = vec![0u64; ne];
    let mut rech_c = vec![0u64; ne];
    let mut rech_l = vec![0u64; ne];
    let mut ts_t: Vec<Vec<f64>> = vec![Vec::new(); ne];
    let mut ts_l: Vec<Vec<f64>> = vec![Vec::new(); ne];
    let mut num = vec![0.0f64; ne];
    let mut den = vec![0.0f64; ne];
    let mut votos = vec![0u64; ne];
    let mut g = Lcg(0x5DEE_CE66_D1A4_F87B ^ semilla);
    for _ in 0..n {
        let mut p = PruebaReversion::new();
        let mut xs: Vec<f64> = comps
            .iter()
            .map(|c| if c.theta > 0.0 { c.sd * g.n() } else { 0.0 })
            .collect();
        let mut t = 0.0f64;
        let mut e = 0usize;
        let mut lv = rachas.map_or(0.0, |(sd, _)| sd * g.n());
        // Votos pendientes de medir: (edad, d, x en el voto, marca objetivo).
        let mut pendientes: Vec<(usize, f64, f64, f64)> = Vec::new();
        while e < ne || !pendientes.is_empty() {
            let h = paso * (0.5 + g.u());
            t += h;
            let v = match rachas {
                Some((sd, _)) => {
                    let d = (-theta_g * h).exp();
                    lv = lv * d + sd * (1.0 - d * d).sqrt() * g.n();
                    (lv - sd * sd).exp()
                }
                None => 1.0,
            };
            for (x, c) in xs.iter_mut().zip(comps) {
                *x = if c.theta > 0.0 {
                    let d = (-c.theta * h).exp();
                    *x * d + v * c.sd * (1.0 - d * d).sqrt() * g.n()
                } else {
                    *x + v * c.sd * (h / c.referencia_ms).sqrt() * g.n()
                };
            }
            let x: f64 = xs.iter().sum();
            p.observar(t as u64, x + v * r * s0 * g.n());
            pendientes.retain(|&(i, d, x0, objetivo)| {
                if t >= objetivo {
                    num[i] += d * (x - x0);
                    den[i] += d * d;
                    false
                } else {
                    true
                }
            });
            while e < ne && t >= edades[e] * memoria {
                let v = p.veredicto(tau).expect("hay pares");
                if v.demostrada {
                    rech[e] += 1;
                    if seguir_reversion {
                        votos[e] += 1;
                        pendientes.push((e, v.base - v.media, x, t + tau));
                    }
                }
                let corta = p.estado_nodo(k).is_some_and(|s| s.madura && s.t > CRITICO);
                let larga = p
                    .estado_largo(k)
                    .is_some_and(|l| l.madura && l.t > CRITICO_LARGO);
                rech_c[e] += u64::from(corta);
                rech_l[e] += u64::from(larga);
                ts_t[e].push(p.estado_nodo(k).map_or(f64::NEG_INFINITY, |s| s.t));
                ts_l[e].push(p.estado_largo(k).map_or(f64::NEG_INFINITY, |l| l.t));
                e += 1;
            }
        }
    }
    let q = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[(0.95 * v.len() as f64) as usize]
    };
    let frac = |c: &[u64]| c.iter().map(|&c| c as f64 / n as f64).collect();
    Barrido {
        edades: edades.to_vec(),
        rechazo: frac(&rech),
        rechazo_corta: frac(&rech_c),
        rechazo_larga: frac(&rech_l),
        q95: ts_t.iter_mut().map(q).collect(),
        q95_largo: ts_l.iter_mut().map(q).collect(),
        reversion: (0..ne)
            .map(|i| if votos[i] >= 5 && den[i] > 0.0 { -num[i] / den[i] } else { f64::NAN })
            .collect(),
        reversion_total: {
            let (nt, dt): (f64, f64) = (num.iter().sum(), den.iter().sum());
            if votos.iter().sum::<u64>() >= 5 && dt > 0.0 {
                -nt / dt
            } else {
                f64::NAN
            }
        },
    }
}

pub fn informe(nombre: &str, b: &Barrido) {
    for i in 0..b.edades.len() {
        println!(
            "observado: {nombre}, edad {:.1} memorias: rechazo {:.3} (corta {:.3}, larga {:.3}), \
             q95 de t {:.2} y de t_largo {:.2}, reversión de lo votado {:.2}",
            b.edades[i],
            b.rechazo[i],
            b.rechazo_corta[i],
            b.rechazo_larga[i],
            b.q95[i],
            b.q95_largo[i],
            b.reversion[i]
        );
    }
}

/// θ0 por ms en el nodo k: ln2/(2τ_k).
pub fn theta0(k: usize) -> f64 {
    std::f64::consts::LN_2 / (2.0 * SPECTRUM_SCALES_MS[k])
}

/// θ por ms de un OU con vida media de `celdas` celdas del nodo k
/// (Δ = τ_k/8).
pub fn theta_celdas(k: usize, celdas: f64) -> f64 {
    std::f64::consts::LN_2 / (celdas * SPECTRUM_SCALES_MS[k] / 8.0)
}
