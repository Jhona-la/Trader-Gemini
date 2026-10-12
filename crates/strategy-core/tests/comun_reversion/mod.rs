//! Barrido de Monte Carlo de la prueba de reversión de la base (CL-97),
//! compartido por `cf3_prueba_reversion.rs` y la medición del crítico.
#![allow(dead_code)]

use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;
use strategy_core::prueba_reversion::PruebaReversion;

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

/// Resultado de un barrido: por edad, la tasa de rechazo y el cuantil 95 del
/// estadístico (sobre los nodos maduros; inmaduro cuenta como no rechazo).
pub struct Barrido {
    pub edades: Vec<f64>,
    pub rechazo: Vec<f64>,
    pub q95: Vec<f64>,
}

/// `n` trayectorias de la base en el nodo `k`: OU con `theta` (por ms; 0 =
/// caminata con innovación s por √paso), ruido de medida `r`·s, pares cada
/// `paso` ms de media. Mira el veredicto a τ_k en cada edad (en memorias).
pub fn barrido(k: usize, n: u64, theta: f64, r: f64, paso: f64, edades: &[f64], semilla: u64) -> Barrido {
    let tau = SPECTRUM_SCALES_MS[k];
    let memoria = 16.0 * tau;
    let s = 1e-4;
    let mut rech = vec![0u64; edades.len()];
    let mut ts_t: Vec<Vec<f64>> = vec![Vec::new(); edades.len()];
    let mut g = Lcg(0x5DEE_CE66_D1A4_F87B ^ semilla);
    for _ in 0..n {
        let mut p = PruebaReversion::new();
        let mut x = if theta > 0.0 { s * g.n() } else { 0.0 };
        let mut t = 0.0f64;
        let mut e = 0usize;
        while e < edades.len() {
            let h = paso * (0.5 + g.u());
            t += h;
            x = if theta > 0.0 {
                let d = (-theta * h).exp();
                x * d + s * (1.0 - d * d).sqrt() * g.n()
            } else {
                x + s * (h / paso).sqrt() * g.n()
            };
            p.observar(t as u64, x + r * s * g.n());
            while e < edades.len() && t >= edades[e] * memoria {
                let v = p.veredicto(tau).expect("hay pares");
                if v.demostrada {
                    rech[e] += 1;
                }
                ts_t[e].push(v.t);
                e += 1;
            }
        }
    }
    let q95 = ts_t
        .iter_mut()
        .map(|v| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[(0.95 * v.len() as f64) as usize]
        })
        .collect();
    Barrido {
        edades: edades.to_vec(),
        rechazo: rech.iter().map(|&c| c as f64 / n as f64).collect(),
        q95,
    }
}

pub fn informe(nombre: &str, b: &Barrido) {
    for i in 0..b.edades.len() {
        println!(
            "observado: {nombre}, edad {:.1} memorias: rechazo {:.3}, q95 de t {:.2}",
            b.edades[i], b.rechazo[i], b.q95[i]
        );
    }
}

/// θ0 por ms en el nodo k: ln2/(2τ_k).
pub fn theta0(k: usize) -> f64 {
    std::f64::consts::LN_2 / (2.0 * SPECTRUM_SCALES_MS[k])
}

