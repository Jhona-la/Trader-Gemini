//! C-F4 · LC-F4-10 (Línea C, CL-106; decisión del dueño «Ambos», 2026-10-12):
//! el freno duro de la cuenta es una prueba de nivel α con el tamaño real de
//! las órdenes, el mismo umbral para el host y el riesgo.
//!
//! Teoría. Una operación gana `w = (TP − f)/SL` o pierde `l = (SL + f)/SL`
//! stops. Con H0 «ventaja de +0,05 R» (R = SL + f), la caída máxima de la
//! cuenta en 200 operaciones tiene un cuantil 95 % `K(w, l)` en stops; el
//! freno dispara cuando la caída de los cierres en stops llega a `K`, o la
//! caída del capital a d* (ADR-0016 D3). Con la orden mínima de nocional
//! fijo, `K` stops son `K·r` de la cuenta.
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
//! El umbral es una caída en stops medida desde el pico, aditiva, que es lo
//! que decidió el dueño («32 R»). Desde CL-130 (R1 de la revisión del PR
//! #62) la caída se cuenta en stops de CADA POSICIÓN, sumando los resultados
//! de los cierres: con tamaño variable (LH-04), dividir la caída en USD por
//! la EWMA del riesgo de las últimas órdenes contaba de más. El Monte Carlo
//! que vio el dueño medía la caída relativa al pico del capital, que crece
//! con la ventaja: su K salía entre un 2 % (stop del 0,5 %) y un 5 % (stop
//! del 1 %) menor. La tolerancia del umbral lo cubre; la falsa alarma y la
//! detección se contrastan con el criterio del código.

use quantum_arena::GlobalArena;
use risk_engine::drawdown::{
    caida_alfa_en_stops, caida_alfa_freno_duro, drawdown_de_falsacion, freno_duro, registrar_pico,
    registrar_resultado_en_stops, ALFA_FALSACION, VENTAJA_H0_FRENO_DURO_R,
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
        // K stops de la orden mínima, en fracción de la cuenta.
        let u = (k * r).min(drawdown_de_falsacion(0.5, 0.05));
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

/// ¿Veta por drawdown un motor de riesgo con la cuenta a 1 000 USD, una
/// caída del capital `caida` desde el pico, los cierres dados (en stops de
/// sus posiciones) y la geometría del stop del 0,5 %?
fn veta(caida: f64, cierres: &[f64]) -> bool {
    let pico = 1_000.0;
    let a: Arc<GlobalArena> = GlobalArena::build_in_own_stack(pico * (1.0 - caida));
    a.coins[0].current_price.store(100.0, Relaxed);
    a.coins[0].current_atr.store(1.0, Relaxed);
    a.config.global_max_drawdown.store(0.5, Relaxed);
    let (w, l, _) = geometria(0.005);
    a.ganancia_en_stops.store(w, Relaxed);
    a.perdida_en_stops.store(l, Relaxed);
    registrar_pico(&a.pico_capital, pico);
    for &r in cierres {
        registrar_resultado_en_stops(&a, r);
    }
    let orden = RiskEngine::new(pico).evaluate_quantum_order(0, &larga(), &a);
    orden.signal == SignalType::Flat
        && risk_engine::ultimo_rechazo().map(|i| REJECT_NAMES[i]) == Some("drawdown")
}

/// Esperado: con la geometría del stop del 0,5 % (K ≈ 40 stops), 12 SL
/// (≈ 15 stops) no vetan; antes la prueba de racha (gen 0,5, q = 0,5) vetaba
/// con un umbral de 1,7 % de la cuenta. 39 SL (≈ 50 stops) sí vetan.
/// B1 de la revisión del PR #62: una caída del capital del 10 % sin cierres
/// que la expliquen en stops (por ejemplo, restaurada de otra sesión con el
/// estado borrado) no veta por debajo de d*; antes vetaba en cuanto la
/// primera orden fijaba la EWMA del riesgo.
#[test]
fn cl130_quince_stops_no_bloquean_la_cuenta_y_cincuenta_si() {
    let (_, l, _) = geometria(0.005);
    let sl = |n: usize| vec![-l; n];
    let (v12, v39, v_usd) = (veta(0.0, &sl(12)), veta(0.0, &sl(39)), veta(0.10, &[]));
    println!("observado: 12 SL ⇒ veta {v12}; 39 SL ⇒ veta {v39}; 10 % sin cierres ⇒ veta {v_usd}");
    assert!(!v12, "12 SL (≈ 15 stops) no rechazan H0");
    assert!(v39, "39 SL (≈ 50 stops) sí");
    assert!(!v_usd, "una caída sin cierres en stops sólo veta en d*");
    assert!(veta(0.64, &[]), "d* veta sin cierres");
}

/// Esperado: la caída en stops es la de los cierres desde su máximo; las
/// ganancias la reducen. Sin geometría medida no hay umbral en stops y rige
/// d*.
#[test]
fn cl130_la_caida_en_stops_y_d_estrella() {
    let a: Arc<GlobalArena> = GlobalArena::build_in_own_stack(100.0);
    let (w, l, _) = geometria(0.005);
    for r in [w, w, -l, -l, -l] {
        registrar_resultado_en_stops(&a, r);
    }
    let f = freno_duro(&a, 100.0, 100.0);
    assert!((f.caida_en_stops - 3.0 * l).abs() < 1e-12, "{f}");
    assert_eq!(f.umbral_en_stops, None, "sin geometría medida no hay K");
    registrar_resultado_en_stops(&a, f64::NAN);
    assert!((freno_duro(&a, 100.0, 100.0).caida_en_stops - 3.0 * l).abs() < 1e-12, "un NaN no cuenta");
    for _ in 0..100 {
        registrar_resultado_en_stops(&a, -l);
    }
    let d = drawdown_de_falsacion(0.5, 0.05);
    assert!(!freno_duro(&a, 100.0, 100.0).echado(), "sin geometría medida, 100 SL no bastan");
    assert!(freno_duro(&a, 100.0, 100.0 * (1.0 - d)).echado(), "d* sí");
    a.ganancia_en_stops.store(w, Relaxed);
    a.perdida_en_stops.store(l, Relaxed);
    assert!(freno_duro(&a, 100.0, 100.0).por_stops(), "con geometría, K");
}

/// El host y el riesgo usan la MISMA función.
#[test]
fn cl106_host_y_riesgo_usan_el_mismo_umbral() {
    let host: String = include_str!("../../../src/bin/god_engine.rs").split_whitespace().collect();
    let riesgo: String = include_str!("../src/lib.rs").split_whitespace().collect();
    assert!(host.contains("letfreno=risk_engine::drawdown::freno_duro(&arena_imm,peak_capital,cap);"));
    assert!(host.contains("letdd_breach=freno.echado();"));
    assert!(riesgo.contains(
        "ifcrate::drawdown::freno_duro(arena,self.peak_capital,current_capital).echado(){returnrej(REJ_DRAWDOWN);}"
    ));
}

/// CL-106b → CL-130 (T1-02 de la Línea G; R1 de la revisión del PR #62): el
/// freno dispara tras K stops de las posiciones, se tome el tamaño que se
/// tome. Con nocional fijo, medido en fracción del capital actual, el umbral
/// se alejaba al caer (CL-106b); con tamaño que baja en la racha, medido con
/// la EWMA del riesgo de las últimas órdenes, se acercaba (R1). Contando en
/// stops de cada posición, no depende de ninguno de los dos.
///
/// Escenario: pico de 13 USD y SL sucesivos con stops del 0,5 %, 1 % y 2 %;
/// el tamaño de cada posición ya está en su resultado en stops (−l), y la
/// prueba con posiciones reales de tamaño decreciente es
/// god-engine-core/tests/cf5c_freno_duro_en_stops.rs.
/// Esperado: veta en el primer cierre con la caída en stops ≥ K.
#[test]
fn cl106b_con_cualquier_tamano_el_freno_dispara_tras_k_stops() {
    for sl in [0.005, 0.010, 0.020] {
        let (w, l, _) = geometria(sl);
        let k = caida_alfa_freno_duro(w, l).unwrap();
        let a: Arc<GlobalArena> = GlobalArena::build_in_own_stack(CAPITAL);
        a.coins[0].current_price.store(100.0, Relaxed);
        a.coins[0].current_atr.store(1.0, Relaxed);
        a.ganancia_en_stops.store(w, Relaxed);
        a.perdida_en_stops.store(l, Relaxed);
        let mut disparo = None;
        for n in 1..=200usize {
            registrar_resultado_en_stops(&a, -l);
            let orden = RiskEngine::new(CAPITAL).evaluate_quantum_order(0, &larga(), &a);
            if orden.signal == SignalType::Flat
                && risk_engine::ultimo_rechazo().map(|i| REJECT_NAMES[i]) == Some("drawdown")
            {
                disparo = Some(n);
                break;
            }
        }
        let n = disparo.expect("el freno tiene que disparar");
        println!("observado: stop {} %: K = {k:.1} stops; dispara en el SL {n} ({:.1} stops)", sl * 100.0, n as f64 * l);
        assert!(n as f64 * l >= k && (n - 1) as f64 * l < k, "stop {} %", sl * 100.0);
    }
}
