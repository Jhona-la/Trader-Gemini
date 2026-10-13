//! LG-5 (Línea G, 2026-10-12; candidato G022, verificado CONFIRMED HIGH) —
//! pedido con prueba en rojo para Codex (calibración) y Línea E (D1).
//!
//! DEFECTO HISTÓRICO: el núcleo entrena Platt (por moneda y global) y el conformal
//! con `is_win = net_trade_pnl > 0` (`lib.rs`, cierre de posición), con
//! cualquier motivo de salida: breakeven, trailing, cosecha, decaimiento. La
//! puerta EV del risk-engine consume esa probabilidad como si fuera
//! P(TP antes que SL): `EV = p·TP − (1 − p)·SL` (`risk-engine/src/lib.rs`,
//! `expected_value_pct`), y el Kelly la usa con b = TP/SL. Con muchas
//! victorias pequeñas (BE a +0,02 R) la p sube sin que la esperanza suba: la
//! puerta aprueba —y el Kelly dimensiona— órdenes de esperanza negativa.
//!
//! LG5 conserva este contraejemplo y consume una media neta por nocional,
//! uniforme sobre las mismas 512 filas retenidas. Es una media descriptiva
//! moneda/lado/procedencia, no calibrada por score/geometría/tau ni una cota.
//! El oráculo NO compara arbitrariamente una ventana de 512 con los 600
//! cierres completos: ambas cantidades deben usar la misma población.
use god_engine_core::calibration::{PlattCalibrator, WINDOW};
use god_engine_core::net_payoff::{EntryPayoffEvidence, NetPayoffModel};
use strategy_core::{NetPayoffSource, SignalType};

/// Reparto de salidas con geometría RR 2,25: 15 % TP (+2,25 R), 40 %
/// breakeven o trailing con neto ≈ +0,02 R y 45 % SL (−1 R). R media
/// realizada: 0,15·2,25 + 0,40·0,02 − 0,45 ≈ −0,105 R.
#[test]
fn lg5_la_ev_de_la_puerta_no_sobreestima_la_r_neta_realizada() {
    let rr = 2.25;
    let mut platt = PlattCalibrator::new();
    let score = 0.60;
    let mut r_total = 0.0;
    let mut n = 0.0;
    let mut payoff = NetPayoffModel::new(0);
    let mut retained_sum = 0.0;
    let stop_fraction = 0.01; // fixture conversion only: 1 R = 1% entry notional
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
        let entry = EntryPayoffEvidence {
            slot: 0,
            generation: i as u64 + 1,
            is_long: true,
            entry_time_ms: i as u64,
            score,
            tau_ms: 60_000,
            entry_price: 100.0,
            initial_tp_price: 102.25,
            initial_sl_price: 99.0,
            source: NetPayoffSource::LocalSimulation,
        };
        assert!(payoff.observe(
            entry,
            r * stop_fraction * 100.0,
            100.0,
            1.0,
            i as u64 + 1,
            i as u64 + 1
        ));
        if i as usize >= 600 - WINDOW {
            retained_sum += r * stop_fraction;
        }
        r_total += r;
        n += 1.0;
    }
    let r_media = r_total / n;
    let p = platt.calibrate(score);
    let ev_incorrecta = p * rr - (1.0 - p);
    let measured = payoff
        .estimate(0, SignalType::Long, NetPayoffSource::LocalSimulation, 600)
        .unwrap();
    let retained_mean = retained_sum / WINDOW as f64;
    assert!(
        r_media < 0.0,
        "precondición: la esperanza realizada es negativa ({r_media})"
    );
    assert!(
        ev_incorrecta > 0.0,
        "contraejemplo original: p neta aplicada a TP/SL da {ev_incorrecta:+.3} R con p={p:.3}"
    );
    assert_eq!(measured.sample_count, WINDOW);
    assert!(measured.mean_net_return_pct < 0.0);
    assert!((measured.mean_net_return_pct - retained_mean).abs() < 1e-14);
}
