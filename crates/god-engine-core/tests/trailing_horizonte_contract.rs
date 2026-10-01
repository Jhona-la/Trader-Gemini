//! CL-34 — la escalera del trailing mide en la dispersión del horizonte de la
//! posición, la misma escala en la que el gate fijó su stop.
use god_engine_core::escala_del_trailing;
use god_engine_core::trailing::evaluate_quantum_trailing_with_fee;
use risk_engine::tp_sl::dispersion_al_horizonte;

const ENTRADA: f64 = 100.0;
const HURST: f64 = 0.5;

/// Sigue un largo al que el precio lleva a `precio` durante `llamadas`
/// evaluaciones, realimentando fase, MFE, pico y stop como hace el núcleo.
/// Devuelve el stop final.
fn stop_del_trailing(escala: f64, tp: f64, precio: f64, llamadas: usize) -> f64 {
    let (mut fase, mut mfe, mut pico, mut stop) = (0, 0.0, 0.0, 0.0);
    for _ in 0..llamadas {
        let r = evaluate_quantum_trailing_with_fee(
            1, ENTRADA, precio, escala, fase, mfe, pico, stop, 4.43, 1.0, 1.0, 2.25, 1.0, 0.0005,
            tp, 0.0,
        );
        fase = r.new_phase;
        mfe = r.mfe_atr;
        pico = r.max_pnl_pct;
        stop = r.stop_price;
    }
    stop
}

#[test]
fn cl34_a_un_minuto_la_escala_es_el_atr_de_siempre() {
    let atr = 0.0015;
    assert_eq!(escala_del_trailing(atr, ENTRADA, 60_000.0, HURST), atr * ENTRADA);
}

#[test]
fn cl34_la_misma_geometria_da_el_mismo_trailing_a_cualquier_reloj() {
    // Dos posiciones con la MISMA dispersión al horizonte —el mismo stop
    // (k = 1) y el mismo TP (RR 2,25)—: una a 4 h con el ATR de 1 minuto de
    // 15,36 pb y otra a 1 minuto con el ATR que da esa misma dispersión.
    let atr_1m = 0.001536;
    let tau_largo = 4.0 * 3_600_000.0;
    let dispersion = dispersion_al_horizonte(atr_1m, tau_largo, HURST);
    let tp = 2.25 * dispersion;
    let precio = ENTRADA * (1.0 + 0.70 * tp); // trailing recién armado (CL-19)

    let largo = escala_del_trailing(atr_1m, ENTRADA, tau_largo, HURST);
    let corto = escala_del_trailing(dispersion, ENTRADA, 60_000.0, HURST);
    assert!((largo - corto).abs() < 1e-9 * corto, "escalas {largo} vs {corto}");

    let stop_largo = stop_del_trailing(largo, tp, precio, 6);
    let stop_corto = stop_del_trailing(corto, tp, precio, 6);
    assert!(
        (stop_largo - stop_corto).abs() < 1e-9 * precio,
        "stop a 4 h {stop_largo} vs a 1 min {stop_corto}"
    );
    // Y el retroceso que tolera se mide en la dispersión del horizonte, no
    // en el ruido de 1 minuto (antes: ≈ 7 pb con una dispersión de 238 pb).
    let retroceso = (precio - stop_largo) / precio;
    assert!(
        retroceso >= 0.40 * dispersion,
        "retroceso {retroceso} frente a dispersión {dispersion}"
    );
}
