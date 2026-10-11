//! C-F3 / F3.6-06 (Línea C): el peso del perceptrón debe seguir la
//! esperanza, no la tasa de acierto.
//!
//! Teoría que se prueba (RULE[growth_over_wr]): con pago 1:2, un WR del 60 %
//! tiene EV = (3·1 − 2·2)/5 = −0,2 por operación y hace decrecer el capital
//! compuesto. Un peso que multiplica el compuesto (hebbian_mult ∈ [0,5; 2] en
//! el núcleo) no puede subir con EV < 0. `update_weight` suma +lr por cada
//! ganancia y −lr por cada pérdida, sin la magnitud del PnL, así que cada
//! ciclo [+1, +1, +1, −2, −2] deja +lr neto.
//!
//! Solución conocida: 40 ciclos deterministas (200 cierres) con el σ que pasa
//! el núcleo (atr_pct = 0,002, lr = 0,05/1,2 ≈ 0,0417). Esperado w ≤ 1;
//! tolerancia 1e-12 (sólo redondeo). Hoy w satura en 2 − 2·lr ≈ 1,9167.
//!
//! Mismo defecto que F3.3-06 (tests/cf3_hebbiano_contract.rs), visto desde el
//! motor.
//!
//! Convención: un test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use signal_engine::perceptron_gate::PerceptronGateEngine as P;

#[test]
#[ignore = "OPEN F3.6-06 (Línea C, perceptron_gate::update_weight; lector hebbian_mult de ramas 1-10 en Línea E): la regla suma ±lr según el signo del PnL; con WR 60 % y pago 1:2 (EV −0,2) el peso sube a 1,917"]
fn cf3_6_peso_hebb_no_premia_acierto_con_esperanza_negativa() {
    // WR 60 % con pago 1:2 => EV = (3*1 - 2*2)/5 = -0,2 por operacion.
    let mut w = 1.0;
    for _ in 0..40 {
        for pnl in [1.0, 1.0, 1.0, -2.0, -2.0] {
            P::update_weight(&mut w, pnl, 0.002); // el núcleo pasa atr_pct
        }
    }
    println!("observado: w = {w:.4} tras 200 cierres con EV = −0,2");
    assert!(
        w <= 1.0 + 1e-12,
        "con EV < 0 el peso no debe crecer: w = {}",
        w
    );
}
