//! C-F3 / F3.7-02 (Línea C): la guarda t½ ≤ 2τ* debe rechazar caminatas
//! aleatorias independientes.
//!
//! Teoría que se prueba: dos log-precios que son caminatas independientes
//! NO están cointegrados; cualquier combinación lineal ln a − β·ln b es no
//! estacionaria (raíz unitaria) y su vida media de reversión es infinita.
//! Una guarda de cointegración debe rechazarlas casi siempre: en un
//! contraste al 10 % pasan ≤ 10 % de las parejas.
//!
//! El código no contrasta ningún nulo de raíz unitaria (Johansen,
//! Engle-Granger, ADF, KPSS): la β por RLS con λ = 0,998 por evento sigue el
//! cociente ln a/ln b con ~500 eventos de memoria (100 s a 200 ms), el
//! residuo resultante es paso-alto y revierte por construcción, y el WLS de θ
//! (memoria de 300 s) tiene el sesgo de Dickey-Fuller (θ̂ > 0 con θ = 0). Se
//! espera t½ ≪ 1 200 s en casi todas las semillas.
//!
//! Configuración viva (la que el núcleo usa por moneda): StatArbEngine
//! (30, 1,5) con SDE OU en reloj físico y β adaptativa. Una pareja «pasa la
//! guarda» si el núcleo publicaría z (`last_ou_zscore` es Some: SDE madura)
//! y el lector la dejaría votar (t½ finita y ≤ 2τ*, la condición de
//! `StatArbEngine::evaluate_for_coin`).
//!
//! Generador: LCG de 64 bits con Box-Muller; 40 semillas; dos caminatas
//! log-normales independientes con σ = 2e-5 por paso, 6 000 pasos de 200 ms
//! (20 min); τ* = 600 000 ms.
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

/// Esperado: ≤ 4 de 40 pasan (≤ 10 %). Una guarda con un contraste al 5 %
/// supera este umbral con P(Binomial(40; 0,05) > 4) ≈ 0,05; con una tasa
/// verdadera del 10 % fallaría en ≈ 37 % de las semillas, así que el umbral
/// exige un contraste de nivel ≈ 5 %. Medido hoy: pasan 40/40, con t½
/// entre 19 s y 137 s (mediana 47 s) frente a 2τ* = 1 200 s.
#[test]
#[ignore = "OPEN F3.7-02 (Línea C, strategy-core): no hay contraste de raíz unitaria; la β RLS por evento deja un residuo paso-alto que revierte por construcción y el WLS de θ tiene sesgo de Dickey-Fuller, así que la guarda t½ ≤ 2τ* deja pasar caminatas independientes"]
fn cf3_guarda_t_media_no_deja_pasar_caminatas_independientes() {
    let tau_ms = 600_000.0;
    let semillas = 40u64;
    let mut pasan = 0u64;
    let mut t_medias = Vec::new();
    for s in 0..semillas {
        let mut g = Lcg(0x9E37_79B9_7F4A_7C15 ^ (s + 1));
        let mut eng = StatArbEngine::new(30, 1.5)
            .with_continuous_ou_sde()
            .with_adaptive_beta(true);
        let (mut la, mut lb) = (50_000f64.ln(), 50_000f64.ln());
        let mut ts = 1_000u64;
        for _ in 0..6_000 {
            la += 2e-5 * g.n();
            lb += 2e-5 * g.n();
            let _ = eng.update_with_clock(la.exp(), lb.exp(), ts, tau_ms);
            ts += 200;
        }
        let t_half_ms = eng.half_life_seconds() * 1000.0;
        t_medias.push(t_half_ms);
        if eng.last_ou_zscore().is_some() && t_half_ms.is_finite() && t_half_ms <= 2.0 * tau_ms {
            pasan += 1;
        }
    }
    let mut orden = t_medias.clone();
    orden.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "observado: pasan {pasan}/{semillas}; t½ (ms) mín {:.1}, mediana {:.1}, máx {:.1}; umbral 2τ* = {:.0}",
        orden[0],
        orden[orden.len() / 2],
        orden[orden.len() - 1],
        2.0 * tau_ms
    );
    assert!(
        pasan * 10 <= semillas,
        "caminatas independientes que pasan la guarda: {pasan}/{semillas}"
    );
}
