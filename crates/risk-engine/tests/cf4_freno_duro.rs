//! C-F4 · LC-F4-10 (Línea C, CL-106; decisión del dueño «Ambos», 2026-10-12):
//! el freno duro de la cuenta es una prueba de nivel α con el tamaño real de
//! las órdenes, el mismo umbral para el host y el riesgo.
//!
//! Teoría. Una operación gana `w = (TP − f)/SL` o pierde `l = (SL + f)/SL`
//! stops. Con H0 «ventaja de +0,05 R» (R = SL + f), la caída máxima de la
//! cuenta en 200 operaciones tiene un cuantil 95 % `K(w, l)` en stops; el
//! umbral es `K·r`, con `r` el riesgo medido al stop en fracción del capital,
//! acotado por d* (ADR-0016 D3).
//!
//! Contrato: los números del Monte Carlo que el dueño vio para decidir (13
//! USD, orden mínima de 5,10 USD, RR 2,25, fricción de 14,5 pb), con
//! tolerancia estadística:
//! - stop del 0,5 %: umbral 30,7 R (7,8 % de la cuenta); falsa alarma 5 % en
//!   200 operaciones con +0,05 R y 0 con +0,30 R; detección (mediana) en 553
//!   operaciones con E[R] = 0 y en 211 con −0,10 R;
//! - stop del 1 %: 33,3 R (15,0 %); detección en 572 y 227.
//! El freno de antes (racha, gen 0,5, q = 0,5) daba 2,2 % y 3,8 %.
//!
//! El código mide la caída en R de forma aditiva, que es lo que decidió el
//! dueño («32 R»). El Monte Carlo que vio medía la caída relativa al pico del
//! capital, que crece con la ventaja: su K salía entre un 2 % (stop del
//! 0,5 %) y un 5 % (stop del 1 %) menor. La tolerancia del umbral lo cubre;
//! la falsa alarma y la detección se contrastan con el criterio del código.

use quantum_arena::GlobalArena;
use risk_engine::drawdown::{
    caida_alfa_en_stops, caida_alfa_freno_duro, registrar_pico, umbral_freno_duro, ALFA_FALSACION,
    VENTAJA_H0_FRENO_DURO_R,
};
use risk_engine::{RiskEngine, REJECT_NAMES};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc};

const RR: f64 = 2.25;
const FRICCION: f64 = 0.00145;
const NOCIONAL: f64 = 5.10;
const CAPITAL: f64 = 13.0;

/// (w, l, r) de una orden mínima con 13 USD y el stop dado.
fn geometria(sl: f64) -> (f64, f64, f64) {
    ((RR * sl - FRICCION) / sl, (sl + FRICCION) / sl, NOCIONAL * sl / CAPITAL)
}

/// Generador independiente del de la función (otra semilla, otro algoritmo).
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Primera operación (1-based) en la que la caída en stops alcanza `k`, o
/// `None` si no la alcanza en `n`. Ventaja `mu_r` en R.
fn primer_disparo(azar: &mut Lcg, w: f64, l: f64, mu_r: f64, k: f64, n: usize) -> Option<usize> {
    let p = (mu_r * l + l) / (w + l);
    let (mut acum, mut pico) = (0.0f64, 0.0f64);
    for t in 1..=n {
        acum += if azar.u() < p { w } else { -l };
        pico = pico.max(acum);
        if pico - acum >= k {
            return Some(t);
        }
    }
    None
}

fn mediana(mut v: Vec<usize>) -> usize {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Esperado: el umbral en R es el del Monte Carlo del dueño, +0 % a +6 %
/// (caída aditiva en R frente a caída relativa al pico; ver la cabecera).
#[test]
fn cl106_el_umbral_es_el_cuantil_de_la_caida_con_el_tamano_real() {
    for (sl, k_r, pct) in [(0.005, 30.7, 0.078), (0.010, 33.3, 0.150)] {
        let (w, l, r) = geometria(sl);
        let k = caida_alfa_en_stops(w, l, VENTAJA_H0_FRENO_DURO_R, ALFA_FALSACION).unwrap();
        let u = umbral_freno_duro(r, w, l);
        println!(
            "observado: stop {} %: K = {k:.2} stops = {:.2} R, umbral {:.2} % (MC del dueño {k_r} R, {} %)",
            sl * 100.0,
            k / l,
            u * 100.0,
            pct * 100.0
        );
        let rel = k / l / k_r - 1.0;
        assert!((0.0..=0.06).contains(&rel), "K en R {} frente a {k_r}", k / l);
        let rel_u = u / pct - 1.0;
        assert!((-0.01..=0.07).contains(&rel_u), "umbral {u} frente a {pct}");
    }
}

/// Esperado: falsa alarma ≈ 5 % en 200 operaciones con +0,05 R y ≈ 0 con
/// +0,30 R; detección (mediana) como en el Monte Carlo del dueño.
#[test]
fn cl106_falsa_alarma_y_retraso_de_deteccion() {
    let mut azar = Lcg(20261012);
    for (sl, det0, det_neg) in [(0.005, 553usize, 211usize), (0.010, 572, 227)] {
        let (w, l, _) = geometria(sl);
        let k = caida_alfa_freno_duro(w, l).unwrap();
        let caminos = 4_000;
        let fa = |azar: &mut Lcg, mu: f64| {
            (0..caminos).filter(|_| primer_disparo(azar, w, l, mu, k, 200).is_some()).count() as f64
                / caminos as f64
        };
        let fa05 = fa(&mut azar, 0.05);
        let fa30 = fa(&mut azar, 0.30);
        let retraso = |azar: &mut Lcg, mu: f64| {
            mediana(
                (0..2_000)
                    .map(|_| primer_disparo(azar, w, l, mu, k, 5_000).unwrap_or(5_001))
                    .collect(),
            )
        };
        let d0 = retraso(&mut azar, 0.0);
        let dneg = retraso(&mut azar, -0.10);
        println!(
            "observado: stop {} %: falsa alarma {:.2} % (+0,05 R) y {:.2} % (+0,30 R); detección {d0} (E[R] = 0) y {dneg} (−0,10 R)",
            sl * 100.0,
            fa05 * 100.0,
            fa30 * 100.0
        );
        // 4 000 caminos: σ de una proporción del 5 % ≈ 0,34 %; tolerancia ~4σ.
        assert!((fa05 - 0.05).abs() <= 0.015, "falsa alarma con +0,05 R: {fa05}");
        assert!(fa30 <= 0.005, "falsa alarma con +0,30 R: {fa30}");
        assert!((d0 as f64 - det0 as f64).abs() <= 0.15 * det0 as f64, "detección con E[R]=0: {d0}");
        assert!((dneg as f64 - det_neg as f64).abs() <= 0.15 * det_neg as f64, "detección con −0,10 R: {dneg}");
    }
}

fn larga() -> SignalIntent {
    SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    }
}

/// ¿Veta por drawdown un motor de riesgo con la caída dada, a 1 000 USD
/// (régimen estándar) y la geometría del stop del 0,5 %?
fn veta(caida: f64) -> bool {
    let pico = 1_000.0;
    let a: Arc<GlobalArena> = GlobalArena::build_in_own_stack(pico * (1.0 - caida));
    a.coins[0].current_price.store(100.0, Relaxed);
    a.coins[0].current_atr.store(1.0, Relaxed);
    a.config.global_max_drawdown.store(0.5, Relaxed);
    let (w, l, _) = geometria(0.005);
    a.riesgo_por_operacion.store(0.002, Relaxed);
    a.ganancia_en_stops.store(w, Relaxed);
    a.perdida_en_stops.store(l, Relaxed);
    registrar_pico(&a.pico_capital, pico);
    let orden = RiskEngine::new(pico).evaluate_quantum_order(0, &larga(), &a);
    orden.signal == SignalType::Flat
        && risk_engine::ultimo_rechazo().map(|i| REJECT_NAMES[i]) == Some("drawdown")
}

/// Esperado: con r = 0,2 % y la geometría del stop del 0,5 % el umbral es ≈ 7,9 %
/// (≈ 40 stops). Una caída del 3 % ya no veta; antes la prueba de racha
/// (gen 0,5, q = 0,5) la vetaba con un umbral del 1,7 %. Una del 10 % sí veta.
#[test]
fn cl106_una_caida_del_3_por_ciento_ya_no_bloquea_la_cuenta() {
    let (v3, v10) = (veta(0.03), veta(0.10));
    println!("observado: caída del 3 % ⇒ veta {v3}; del 10 % ⇒ veta {v10}");
    assert!(!v3, "una caída del 3 % (≈ 15 stops) no rechaza H0");
    assert!(v10, "una caída del 10 % (≈ 50 stops) sí");
}

/// Esperado: sin riesgo medido rige d*; la función está acotada por d*.
#[test]
fn cl106_sin_riesgo_medido_rige_d_estrella() {
    let d = risk_engine::drawdown::drawdown_de_falsacion(0.5, 0.05);
    let (w, l, _) = geometria(0.005);
    assert_eq!(umbral_freno_duro(0.0, w, l), d);
    assert_eq!(umbral_freno_duro(f64::NAN, w, l), d);
    assert_eq!(umbral_freno_duro(0.5, w, l), d, "con riesgo enorme manda d*");
    assert_eq!(umbral_freno_duro(0.002, f64::NAN, l), d, "sin geometría rige d*");
}

/// El host y el riesgo usan la MISMA función.
#[test]
fn cl106_host_y_riesgo_usan_el_mismo_umbral() {
    let host: String = include_str!("../../../src/bin/god_engine.rs").split_whitespace().collect();
    let riesgo: String = include_str!("../src/lib.rs").split_whitespace().collect();
    assert!(host.contains("letmax_dd=risk_engine::drawdown::umbral_freno_duro_de(&arena_imm);"));
    assert!(riesgo.contains("ifdd>=crate::drawdown::umbral_freno_duro_de(arena){returnrej(REJ_DRAWDOWN);}"));
}
