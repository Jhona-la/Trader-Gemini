//! F-A6 (Fable, 2026-10-18; hallazgo F1-017) — el Hawkes por símbolo tiene
//! estado O(1) y es EXACTAMENTE la suma de kernels.
//!
//! Teoría que se prueba. Para el kernel exponencial, S(t) = Σᵢ α·e^{−β(t−tᵢ)}
//! satisface la recursión S(t_k) = S(t_{k−1})·e^{−β(t_k−t_{k−1})} + α y, entre
//! eventos, S(t) = S(t_last)·e^{−β(t−t_last)}. La deque anterior recorría toda
//! la historia con un `exp()` por evento BAJO un `Mutex`, cuatro veces por
//! evento del núcleo, y truncaba la cola a 5/β. El contrato: (a) la
//! intensidad coincide con la suma explícita completa en una traza aleatoria
//! (en los eventos y entre eventos), con error relativo ≤ 1e-9; (b) el coste
//! de `intensity` no crece con la historia (10 000 eventos en ventana se
//! consultan tan rápido como 10); (c) un evento retrógrado no excita ni mueve
//! el reloj; (d) el módulo ya no contiene `Mutex` ni `VecDeque` (contrato de
//! fuente, como LG-23 / CL-14); (e) el estado estacionario de #535 se
//! conserva (λ/μ̂ → 1 + α/β en un stream regular).
use signal_engine::hawkes_bessel::{
    HawkesBesselEngine, HawkesBesselParams, DEFAULT_ALPHA, DEFAULT_BETA, STEADY_STATE_RATIO,
};
use std::time::Instant;

/// xorshift64* determinista (sin dependencias).
struct Rng(u64);
impl Rng {
    fn u01(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// Traza con huecos de escalas muy distintas (de 1 ms a 30 s) para que
/// convivan contribuciones casi plenas y casi extinguidas.
fn traza(seed: u64, n: usize) -> Vec<f64> {
    let mut r = Rng(seed);
    let mut t = 0.0;
    (0..n)
        .map(|_| {
            let u = r.u01();
            let dt = if u < 0.70 { 0.001 + 0.2 * r.u01() } else if u < 0.95 { 0.2 + 3.0 * r.u01() } else { 3.0 + 27.0 * r.u01() };
            t += dt;
            t
        })
        .collect()
}

/// Suma explícita completa (sin purga), la definición de λ.
fn suma_explicita(eventos: &[f64], t: f64, mu: f64, alpha: f64, beta: f64) -> f64 {
    mu + eventos
        .iter()
        .filter(|&&ti| ti <= t)
        .map(|&ti| alpha * (-beta * (t - ti)).exp())
        .sum::<f64>()
}

#[test]
fn la_recursion_es_la_suma_explicita_en_eventos_y_entre_eventos() {
    let ev = traza(0x9E37_79B9_7F4A_7C15, 2_000);
    let mut h = HawkesBesselEngine::new();
    let mut err_max = 0.0_f64;
    for (i, &t) in ev.iter().enumerate() {
        h.record_event(t);
        // En el evento (dt = 0, el propio evento aporta α entero)…
        let exacto = suma_explicita(&ev[..=i], t, h.base_rate(), DEFAULT_ALPHA, DEFAULT_BETA);
        let rel = (h.intensity(t) - exacto).abs() / exacto;
        err_max = err_max.max(rel);
        // …y entre eventos, a un cuarto del hueco siguiente.
        if let Some(&sig) = ev.get(i + 1) {
            let tq = t + 0.25 * (sig - t);
            let exacto_q = suma_explicita(&ev[..=i], tq, h.base_rate(), DEFAULT_ALPHA, DEFAULT_BETA);
            let rel_q = (h.intensity(tq) - exacto_q).abs() / exacto_q;
            err_max = err_max.max(rel_q);
        }
    }
    assert!(err_max <= 1e-9, "error relativo máximo frente a la suma explícita: {err_max:e}");
    // La excitación acumulada también es la suma (sin μ̂).
    let t_fin = *ev.last().expect("traza");
    let s_exacta = suma_explicita(&ev, t_fin, 0.0, DEFAULT_ALPHA, DEFAULT_BETA);
    assert!((h.excitacion_acumulada() - s_exacta).abs() <= 1e-9 * s_exacta);
}

#[test]
fn el_coste_de_consultar_no_crece_con_la_historia() {
    // 10 000 eventos en 1 s frente a 10: con la deque, la primera consulta
    // costaba 1 000× más (un exp por evento); con la recursión, lo mismo.
    let consultar = |h: &HawkesBesselEngine, t0: f64| {
        let ini = Instant::now();
        let mut acc = 0.0;
        for k in 0..20_000 {
            acc += h.intensity(t0 + k as f64 * 1e-6);
        }
        (ini.elapsed(), acc)
    };
    let mut poco = HawkesBesselEngine::new();
    for k in 0..10 {
        poco.record_event(k as f64 * 0.1);
    }
    let mut mucho = HawkesBesselEngine::new();
    for k in 0..10_000 {
        mucho.record_event(k as f64 * 1e-4);
    }
    let (t_poco, a1) = consultar(&poco, 1.0);
    let (t_mucho, a2) = consultar(&mucho, 1.0);
    assert!(a1.is_finite() && a2.is_finite());
    // Cota holgada para el ruido de un entorno de pruebas compartido: la
    // deque daba ×1 000, la recursión da ×1 (con jitter).
    assert!(
        t_mucho.as_secs_f64() <= 5.0 * t_poco.as_secs_f64() + 0.005,
        "20 000 consultas: 10 eventos {t_poco:?} frente a 10 000 eventos {t_mucho:?}"
    );
}

#[test]
fn un_evento_retrogrado_no_excita_ni_mueve_el_reloj() {
    let mut h = HawkesBesselEngine::new();
    h.record_event(10.0);
    h.record_event(10.5);
    let antes = (h.excitacion_acumulada(), h.intensity(10.5), h.ultimo_evento_s(), h.base_rate());
    h.record_event(9.0); // retrógrado
    h.record_event(f64::NAN); // no finito
    let despues = (h.excitacion_acumulada(), h.intensity(10.5), h.ultimo_evento_s(), h.base_rate());
    assert_eq!(antes, despues, "un evento retrógrado o no finito no cambia el estado");
    assert_eq!(h.ultimo_evento_s(), Some(10.5));
    // Un evento simultáneo (dt = 0) sí excita, sin tocar μ̂ (necesita dt > 0).
    let mu = h.base_rate();
    h.record_event(10.5);
    assert!((h.excitacion_acumulada() - (antes.0 + DEFAULT_ALPHA)).abs() < 1e-12);
    assert_eq!(h.base_rate(), mu);
}

#[test]
fn el_modulo_no_tiene_mutex_ni_deque() {
    // Contrato de fuente: la historia O(1) no puede volver a ser una cola
    // bajo cerrojo sin que esta prueba lo diga.
    let fuente = include_str!("../src/hawkes_bessel.rs");
    let codigo: String = fuente
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!codigo.contains("Mutex"), "hawkes_bessel.rs vuelve a usar Mutex");
    assert!(!codigo.contains("VecDeque"), "hawkes_bessel.rs vuelve a usar VecDeque");
    assert!(!codigo.contains("max_events"), "max_events era un parámetro muerto");
}

#[test]
fn el_estado_estacionario_de_535_se_conserva_con_parametros_propios() {
    // #535: en un stream regular λ/μ̂ → 1 + α/β en fase uniforme; la
    // recursión no lo mueve, con los parámetros por defecto o con otros.
    for (alpha, beta, dt_ev) in [(DEFAULT_ALPHA, DEFAULT_BETA, 0.1), (0.4, 0.8, 0.05), (0.1, 2.0, 1.0)] {
        let p = HawkesBesselParams { alpha, beta, ..HawkesBesselParams::default() };
        let mut h = HawkesBesselEngine::with_params(p);
        let (mut suma, mut n) = (0.0, 0.0);
        let mut t = 0.0;
        for ciclo in 0..2_000 {
            h.record_event(t);
            if ciclo >= 200 {
                // Fase uniforme muestreada en los puntos medios de cuatro
                // subintervalos (regla del punto medio: sin sesgo de borde).
                for k in 0..4 {
                    suma += h.intensity_ratio(t + dt_ev * (k as f64 + 0.5) / 4.0);
                    n += 1.0;
                }
            }
            t += dt_ev;
        }
        let medio = suma / n;
        let ss = p.steady_state_ratio();
        assert!((medio - ss).abs() < 0.05 * ss, "α={alpha} β={beta}: ⟨λ/μ̂⟩={medio} vs SS={ss}");
    }
    assert!((HawkesBesselParams::default().steady_state_ratio() - STEADY_STATE_RATIO).abs() < 1e-12);
}
