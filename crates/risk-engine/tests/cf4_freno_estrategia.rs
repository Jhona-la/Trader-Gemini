//! C-F4 · LC-F4-10 (Línea C, CL-107; decisión del dueño «Ambos», 2026-10-12):
//! el freno de la estrategia es un proceso e de Ville sobre la R de cada
//! cierre, con H0 = +0,30 R; rechaza con M ≥ 20 y desde entonces el riesgo no
//! admite entradas nuevas, sin aplanar ni disparar el kill-switch.
//!
//! Fixture. La geometría de una orden mínima con 13 USD, RR 2,25 y 14,5 pb de
//! fricción, con stop del 0,5 % (w = 1,96 y l = 1,29 stops) y del 1 %
//! (w = 2,105, l = 1,145). Cada cierre gana w o pierde l stops con la
//! probabilidad que da la ventaja (en R = SL + f).

use quantum_arena::GlobalArena;
use risk_engine::freno_estrategia::{FrenoEstrategia, VENTAJA_H0_ESTRATEGIA_R};
use risk_engine::{RiskEngine, REJECT_NAMES};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

const RR: f64 = 2.25;
const FRICCION: f64 = 0.00145;

fn geometria(sl: f64) -> (f64, f64) {
    ((RR * sl - FRICCION) / sl, (sl + FRICCION) / sl)
}

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Cierre (en stops) con ventaja `mu_r` en R.
fn cierre(azar: &mut Lcg, w: f64, l: f64, mu_r: f64) -> f64 {
    let p = (mu_r * l + l) / (w + l);
    if azar.u() < p { w } else { -l }
}

/// Operación (1-based) en la que el freno rechaza, o `None` en `n`.
fn rechazo(azar: &mut Lcg, w: f64, l: f64, mu_r: f64, n: usize) -> Option<usize> {
    let mut f = FrenoEstrategia::new();
    for t in 1..=n {
        f.registrar(cierre(azar, w, l, mu_r), w, l);
        if f.rechazada() {
            return Some(t);
        }
    }
    None
}

fn mediana(mut v: Vec<usize>) -> usize {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Contrato: falsa alarma ≤ α (con tolerancia de muestreo) en 2 000
/// operaciones bajo H0, y detección (mediana) como en el Monte Carlo que vio
/// el dueño (mezcla de apuestas fijas), ±20 %:
/// - stop del 0,5 %: 168 operaciones con +0,05 R, 117 con E[R] = 0, 61 con −0,10 R;
/// - stop del 1 %: 209, 141 y 75.
#[test]
fn cl107_falsa_alarma_y_deteccion_del_freno_de_la_estrategia() {
    let mut azar = Lcg(20261012);
    for (sl, e005, e0, eneg) in [(0.005, 168.0, 117.0, 61.0), (0.010, 209.0, 141.0, 75.0)] {
        let (w, l) = geometria(sl);
        let caminos = 1_000;
        let fa = (0..caminos)
            .filter(|_| rechazo(&mut azar, w, l, VENTAJA_H0_ESTRATEGIA_R, 2_000).is_some())
            .count() as f64
            / caminos as f64;
        let det = |azar: &mut Lcg, mu: f64| {
            mediana(
                (0..1_000)
                    .map(|_| rechazo(azar, w, l, mu, 5_000).unwrap_or(5_001))
                    .collect(),
            )
        };
        let d005 = det(&mut azar, 0.05);
        let d0 = det(&mut azar, 0.0);
        let dneg = det(&mut azar, -0.10);
        println!(
            "observado: stop {} %: falsa alarma {:.1} % en 2 000 operaciones con +0,30 R; detección (mediana) {d005} con +0,05 R, {d0} con E[R] = 0 y {dneg} con −0,10 R",
            sl * 100.0,
            fa * 100.0
        );
        // Ville: ≤ 5 % para todo t; 1 000 caminos, σ ≈ 0,7 %.
        assert!(fa <= 0.07, "falsa alarma {fa}");
        for (obs, esperado) in [(d005, e005), (d0, e0), (dneg, eneg)] {
            assert!(
                (obs as f64 / esperado - 1.0).abs() <= 0.20,
                "detección {obs} frente a {esperado} (stop {sl})"
            );
        }
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

/// Motivo del rechazo de una entrada, o «admitida».
fn motivo(risk: &mut RiskEngine, a: &GlobalArena) -> &'static str {
    let orden = risk.evaluate_quantum_order(0, &larga(), a);
    if orden.signal == SignalType::Flat {
        risk_engine::ultimo_rechazo().map_or("sin motivo", |i| REJECT_NAMES[i])
    } else {
        "admitida"
    }
}

/// Esperado: tras una racha de cierres sin la ventaja de H0 el freno rechaza
/// y el riesgo deja de admitir entradas con `estrategia_falsada`. Antes no
/// había freno de la estrategia: la misma racha no cambiaba nada.
#[test]
fn cl107_con_la_estrategia_falsada_no_entra_nada_nuevo() {
    let a = GlobalArena::build_in_own_stack(1_000.0);
    a.coins[0].current_price.store(100.0, Relaxed);
    a.coins[0].current_atr.store(1.0, Relaxed);
    let (w, l) = geometria(0.005);
    let mut risk = RiskEngine::new(1_000.0);
    let antes = motivo(&mut risk, &a);
    // E[R] = 0 con resultados aleatorios (semilla fija). Una secuencia
    // periódica de la misma tasa es antipersistente y engaña a la apuesta
    // adaptativa (tarda ~550 cierres en vez de ~117).
    let mut azar = Lcg(7);
    let mut n = 0;
    while !risk.freno_estrategia.rechazada() && n < 2_000 {
        risk.freno_estrategia.registrar(cierre(&mut azar, w, l, 0.0), w, l);
        n += 1;
    }
    let despues = motivo(&mut risk, &a);
    println!("observado: antes {antes}; tras {n} cierres sin ventaja, {despues}");
    assert_ne!(antes, "estrategia_falsada");
    assert!(risk.freno_estrategia.rechazada(), "el freno rechaza H0 en {n} cierres");
    assert_eq!(despues, "estrategia_falsada");
}

/// Falso positivo: con la ventaja de H0 en una secuencia fija favorable el
/// freno no rechaza.
#[test]
fn cl107_con_la_ventaja_de_h0_no_frena() {
    let (w, l) = geometria(0.005);
    let mut f = FrenoEstrategia::new();
    for n in 0..2_000 {
        let p = (VENTAJA_H0_ESTRATEGIA_R * l + l) / (w + l);
        let gana = (n as f64 * p).fract() < p;
        f.registrar(if gana { w } else { -l }, w, l);
    }
    println!("observado: con +0,30 R exacto, e = {:.4}", f.ville.e_value);
    assert!(!f.rechazada());
}

/// El freno de la estrategia no aplana ni dispara el kill-switch: el host no
/// decide con él ni lo alimenta (sólo lo nombra al registrar en el log el
/// estado restaurado, CL-108).
#[test]
fn cl107_el_host_no_aplana_por_el_freno_de_la_estrategia() {
    let host = include_str!("../../../src/bin/god_engine.rs");
    assert!(!host.contains("freno_estrategia.rechazada()"));
    assert!(!host.contains("freno_estrategia.registrar"));
    assert!(!host.contains("REJ_ESTRATEGIA_FALSADA"));
}
