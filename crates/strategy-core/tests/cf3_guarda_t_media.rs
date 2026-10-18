//! C-F3 / F3.7-02 (Línea C): la guarda t½ ≤ 2τ* debe rechazar caminatas
//! aleatorias independientes.
//!
//! Teoría que se prueba: dos log-precios que son caminatas independientes
//! NO están cointegrados; cualquier combinación lineal ln a − β·ln b es no
//! estacionaria (raíz unitaria) y su vida media de reversión es infinita.
//! Una guarda de cointegración debe rechazarlas casi siempre: en un
//! contraste al 10 % pasan ≤ 10 % de las parejas.
//!
//! Antes de CL-97 el código no contrastaba ningún nulo de raíz unitaria: la
//! β por RLS (λ = 0,998 por evento) seguía el cociente ln a/ln b con ~500
//! eventos de memoria, el residuo era paso-alto y revertía por construcción,
//! y el WLS de θ (memoria de 300 s) tenía el sesgo de Dickey-Fuller. Pasaban
//! 40/40 parejas, con t½ entre 19 s y 137 s frente a 2τ* = 1 200 s.
//!
//! CL-97 (F3.7-02 v2): la guarda es una prueba de reversión sobre pares
//! síncronos (`strategy_core::prueba_reversion`): H0 t½ ≥ 2τ_k en el nodo
//! τ_k ≤ τ* (aquí τ_19 = 274,9 s), β fija, instrumento b(t−2Δ), nivel 5 %.
//! Una pareja «pasa la guarda» si el veredicto es reversión demostrada (lo
//! que el núcleo publica como t½ ≤ 2τ* con z ≠ 0). En 20 min el nodo 19 no
//! madura (la prueba corta necesita 8·τ_19 = 37 min y el veredicto, con la
//! prueba a la escala de decisión, 33·τ_19 = 2,5 h); desde C-F5e se
//! contrasta el nodo maduro más grande bajo τ*, aquí el 17 (9,5 min). Las
//! variantes largas (4 h, sin ruido y con ruido de medida) prueban la guarda
//! con el nodo 19 maduro.
//!
//! Generador: LCG de 64 bits con Box-Muller; 40 semillas; dos caminatas
//! log-normales independientes con σ = 2e-5 por paso, 6 000 pasos de 200 ms
//! (20 min); τ* = 600 000 ms. Cada paso es un par síncrono.
//!
//! Convención: una prueba que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use strategy_core::stat_arb::StatArbEngine;

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Normal estándar por Box-Muller.
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Una pareja pasa la guarda si la prueba demuestra la reversión a τ* (cota
/// publicada ≤ 2τ*). La configuración es la viva del núcleo; la β
/// adaptativa se deja activa para probar que la prueba no la usa.
fn pasa_la_guarda(eng: &StatArbEngine, tau_ms: f64) -> (bool, f64) {
    // Resolución de la base de 1e-6 (un tick de 0,1 USD a 100 000 USD): sin
    // piso StatArb no vota (CL-111) y la guarda no probaría nada.
    match eng.veredicto(tau_ms, 1e-6) {
        Some(v) => (v.demostrada && v.t_media_cota_ms <= 2.0 * tau_ms, v.prueba.t),
        None => (false, f64::NEG_INFINITY),
    }
}

/// Caminatas independientes: `semillas` parejas de `pasos` pasos de `dt_ms`
/// con σ = `sigma` por paso y ruido de medida iid de desviación `ruido` en
/// cada log-precio. Devuelve las que pasan la guarda al final y los t.
fn caminatas(semillas: u64, pasos: u64, dt_ms: u64, sigma: f64, ruido: f64, tau_ms: f64) -> (u64, Vec<f64>) {
    let mut pasan = 0u64;
    let mut ts_t = Vec::new();
    for s in 0..semillas {
        let mut g = Lcg(0x9E37_79B9_7F4A_7C15 ^ (s + 1));
        let mut eng = StatArbEngine::new(30, 1.5)
            .with_adaptive_beta(true)
            .with_prueba_reversion();
        let (mut la, mut lb) = (50_000f64.ln(), 50_000f64.ln());
        let mut ts = 1_000u64;
        for _ in 0..pasos {
            la += sigma * g.n();
            lb += sigma * g.n();
            let (ma, mb) = (la + ruido * g.n(), lb + ruido * g.n());
            eng.observar_par(ma.exp(), mb.exp(), ts);
            ts += dt_ms;
        }
        let (pasa, t) = pasa_la_guarda(&eng, tau_ms);
        ts_t.push(t);
        if pasa {
            pasan += 1;
        }
    }
    (pasan, ts_t)
}

fn mediana(v: &[f64]) -> f64 {
    let mut o = v.to_vec();
    o.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    o[o.len() / 2]
}

/// Esperado: ≤ 4 de 40 pasan (≤ 10 %). Una guarda con un contraste al 5 %
/// supera este umbral con P(Binomial(40; 0,05) > 4) ≈ 0,05; con una tasa
/// verdadera del 10 % fallaría en ≈ 37 % de las semillas, así que el umbral
/// exige un contraste de nivel ≈ 5 %. Medido antes de CL-97: pasaban 40/40,
/// con t½ entre 19 s y 137 s (mediana 47 s) frente a 2τ* = 1 200 s.
#[test]
fn cf3_guarda_t_media_no_deja_pasar_caminatas_independientes() {
    let tau_ms = 600_000.0;
    let semillas = 40u64;
    let (pasan, t) = caminatas(semillas, 6_000, 200, 2e-5, 0.0, tau_ms);
    println!("observado: pasan {pasan}/{semillas}; t mediano {:.2} (crítico 1,75)", mediana(&t));
    assert!(
        pasan * 10 <= semillas,
        "caminatas independientes que pasan la guarda: {pasan}/{semillas}"
    );
}

/// CL-97. La misma verdad con la guarda madura: 4 h de pares cada segundo
/// (3,3 memorias del nodo 19), σ = 2e-5·√5 por paso (la misma volatilidad).
#[test]
fn cl97_guarda_madura_no_deja_pasar_caminatas_independientes() {
    let tau_ms = 600_000.0;
    let semillas = 40u64;
    let (pasan, t) = caminatas(semillas, 14_400, 1_000, 2e-5 * 5f64.sqrt(), 0.0, tau_ms);
    println!("observado: pasan {pasan}/{semillas}; t mediano {:.2} (crítico 1,75)", mediana(&t));
    assert!(pasan * 10 <= semillas, "pasan {pasan}/{semillas}");
}

/// CL-97. Con ruido de medida (tick, horquilla) del doble de la innovación
/// por paso en cada pata: el ruido hace la base de media móvil y con mínimos
/// cuadrados fabricaría reversión; el instrumento b(t−2Δ) no lo ve.
#[test]
fn cl97_el_ruido_de_medida_no_fabrica_reversion() {
    let tau_ms = 600_000.0;
    let semillas = 40u64;
    let sigma = 2e-5 * 5f64.sqrt();
    let (pasan, t) = caminatas(semillas, 14_400, 1_000, sigma, 2.0 * sigma, tau_ms);
    println!("observado: pasan {pasan}/{semillas}; t mediano {:.2} (crítico 1,75)", mediana(&t));
    assert!(pasan * 10 <= semillas, "pasan {pasan}/{semillas}");
}
