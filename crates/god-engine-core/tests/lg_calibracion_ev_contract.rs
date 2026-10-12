//! LG-5 (Línea G, 2026-10-12; candidato G022, verificado CONFIRMED HIGH) —
//! pedido con prueba en rojo para Codex (calibración) y Línea E (D1).
//!
//! QUÉ ESTÁ MAL: el núcleo entrena Platt (por moneda y global) y el conformal
//! con `is_win = net_trade_pnl > 0` (`lib.rs`, cierre de posición), con
//! cualquier motivo de salida: breakeven, trailing, cosecha, decaimiento. La
//! puerta EV del risk-engine consume esa probabilidad como si fuera
//! P(TP antes que SL): `EV = p·TP − (1 − p)·SL` (`risk-engine/src/lib.rs`,
//! `expected_value_pct`), y el Kelly la usa con b = TP/SL. Con muchas
//! victorias pequeñas (BE a +0,02 R) la p sube sin que la esperanza suba: la
//! puerta aprueba —y el Kelly dimensiona— órdenes de esperanza negativa.
//!
//! QUÉ DEBE GARANTIZAR EL ARREGLO: la cantidad que la puerta multiplica por
//! TP y SL tiene que ser coherente con esa fórmula —P(TP primero) con
//! etiqueta de barrera (`etiqueta_barrera`) y las salidas intermedias tratadas
//! aparte— o la puerta debe consumir directamente la R neta esperada
//! calibrada sobre la puntuación. Esta prueba no fija cuál de las dos: fija
//! que la EV que ve la puerta no sobreestime la R neta media realizada.
use god_engine_core::calibration::PlattCalibrator;

/// Reparto de salidas con geometría RR 2,25: 15 % TP (+2,25 R), 40 %
/// breakeven o trailing con neto ≈ +0,02 R y 45 % SL (−1 R). R media
/// realizada: 0,15·2,25 + 0,40·0,02 − 0,45 ≈ −0,105 R.
#[test]
#[ignore = "OPEN LG-5 (G022): Platt aprende P(neto > 0) y la puerta EV la usa como P(TP primero); dueño Codex + Línea E"]
fn lg5_la_ev_de_la_puerta_no_sobreestima_la_r_neta_realizada() {
    let rr = 2.25;
    let mut platt = PlattCalibrator::new();
    let score = 0.60;
    let mut r_total = 0.0;
    let mut n = 0.0;
    for i in 0..600u32 {
        let k = i % 20; // 3 TP, 8 BE, 9 SL de cada 20
        let (gano_neto, r) = if k < 3 {
            (true, rr)
        } else if k < 11 {
            (true, 0.02)
        } else {
            (false, -1.0)
        };
        platt.update(score, gano_neto);
        r_total += r;
        n += 1.0;
    }
    let r_media = r_total / n;
    let p = platt.calibrate(score);
    let ev_puerta = p * rr - (1.0 - p);
    assert!(r_media < 0.0, "precondición: la esperanza realizada es negativa ({r_media})");
    assert!(
        ev_puerta <= r_media + 0.05,
        "la puerta ve EV = {ev_puerta:+.3} R con p = {p:.3}, pero la R neta media realizada es {r_media:+.3} R"
    );
}
