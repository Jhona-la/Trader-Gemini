//! C-F3 / F3.3-06 (Línea C): el peso «hebbiano» del perceptrón debe seguir
//! la esperanza (EV) de las operaciones cerradas, no su tasa de acierto.
//!
//! Teoría que se prueba (RULE[growth_over_wr]): el crecimiento compuesto lo
//! fija E[R], no el WR. Un peso que multiplica el compuesto (×[0,5; 2] en el
//! lector del núcleo) debe subir con EV > 0 y bajar con EV < 0, sea cual sea
//! el WR. Antes de CL-76 `update_weight` sumaba o restaba un paso
//! lr = 0,05/(1 + 100·σ) según el SIGNO del PnL, sin su magnitud, así que su
//! deriva por operación era (2·WR − 1)·lr: con WR 40 % bajaba aunque el EV
//! fuera +0,30 R (w ≈ 0,29), y con WR 60 % subía aunque el EV fuera −0,10 R
//! (w = 2,0). Desde CL-76 recibe el resultado en R y ln w deriva lr·E[R].
//!
//! Solución conocida: secuencias deterministas con WR exacto (acumulador de
//! fase, sin azar). Geometría RR 2,25 (la activa): WR 40 % a +2,25/−1
//! (EV = 0,4·2,25 − 0,6 = +0,30 R) y WR 60 % a +0,5/−1 (EV = −0,10 R), con
//! σ = 0,002 (lr ≈ 0,0417). Tras 1 000 cierres la deriva de ln w
//! (lr·E[R]·1 000 ≈ +12,5 y −4,2) satura el peso en sus límites [0,25; 2]
//! con holgura, así que la verdad
//! «EV > 0 ⇒ w ≥ 1 y EV < 0 ⇒ w ≤ 1» no depende de la longitud exacta.
//!
//! Mismo defecto que F3.6-06 (tests/cf3_6_hebb.rs), visto desde la
//! confianza del compuesto.
//!
//! Fue OPEN F3.3-06 hasta CL-76, que lo activa sin tocar la aserción.

use signal_engine::perceptron_gate::PerceptronGateEngine;

/// Peso final tras 1 000 cierres con frecuencia de acierto exacta `p`;
/// `ganancia` y `perdida` en unidades de riesgo (R).
fn peso_final(p: f64, ganancia: f64, perdida: f64) -> f64 {
    let mut w = 1.0;
    let mut acc = 0.0;
    for _ in 0..1000 {
        acc += p;
        let gana = acc >= 1.0 - 1e-12;
        if gana {
            acc -= 1.0;
        }
        let pnl = if gana { ganancia } else { -perdida };
        PerceptronGateEngine::update_weight(&mut w, pnl, 0.002);
    }
    w
}

#[test]
fn cf3_hebbiano_sigue_el_edge_y_no_la_tasa_de_acierto() {
    let ev_positivo = peso_final(0.40, 2.25, 1.0); // EV = +0,30 R
    let ev_negativo = peso_final(0.60, 0.50, 1.0); // EV = -0,10 R
    println!("observado: EV +0,30 R (WR 40 %) -> w = {ev_positivo:.4}; EV −0,10 R (WR 60 %) -> w = {ev_negativo:.4}");
    assert!(
        ev_positivo >= 1.0,
        "EV > 0 con WR 40 % acaba en {ev_positivo}"
    );
    assert!(
        ev_negativo <= 1.0,
        "EV < 0 con WR 60 % acaba en {ev_negativo}"
    );
}
