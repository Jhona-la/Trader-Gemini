//! C-F3 / F3.3-06 (Línea C): el peso «hebbiano» del perceptrón debe seguir
//! la esperanza (EV) de las operaciones cerradas, no su tasa de acierto.
//!
//! Teoría que se prueba (RULE[growth_over_wr]): el crecimiento compuesto lo
//! fija E[R], no el WR. Un peso que multiplica el compuesto (×[0,5; 2] en el
//! lector del núcleo) debe subir con EV > 0 y bajar con EV < 0, sea cual sea
//! el WR. `update_weight` suma o resta un paso lr = 0,05/(1 + 100·σ) según el
//! SIGNO del PnL, sin su magnitud, así que su deriva por operación es
//! (2·WR − 1)·lr: con WR 40 % baja aunque el EV sea +0,30 R, y con WR 60 %
//! sube aunque el EV sea −0,10 R.
//!
//! Solución conocida: secuencias deterministas con WR exacto (acumulador de
//! fase, sin azar). Geometría RR 2,25 (la activa): WR 40 % a +2,25/−1
//! (EV = 0,4·2,25 − 0,6 = +0,30 R) y WR 60 % a +0,5/−1 (EV = −0,10 R), con
//! σ = 0,002 (lr ≈ 0,0417). Tras 1 000 cierres la deriva (±0,2·lr·1 000 ≈ ±8)
//! satura el peso en sus límites [0,25; 2] con holgura, así que la verdad
//! «EV > 0 ⇒ w ≥ 1 y EV < 0 ⇒ w ≤ 1» no depende de la longitud exacta.
//!
//! Mismo defecto que F3.6-06 (tests/cf3_6_hebb.rs), visto desde la
//! confianza del compuesto.
//!
//! Convención: un test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use signal_engine::perceptron_gate::PerceptronGateEngine;

/// Peso final tras 1 000 cierres con frecuencia de acierto exacta `p`.
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
#[ignore = "OPEN F3.3-06 (Línea C, perceptron_gate): update_weight suma ±lr según el signo del PnL, no su magnitud; aprende el WR y no el EV (EV +0,30 R con WR 40 % baja a ≈0,29; EV −0,10 R con WR 60 % sube a 2,0)"]
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
