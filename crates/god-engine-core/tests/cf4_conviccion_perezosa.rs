//! CL-91 (tau_conviccion-1, revisión de los PR #43–#55): el p* de la
//! convicción de rama se calcula sólo cuando puede cambiar el resultado.
//!
//! `conviccion_de_rama_con(registro, piso, p*)` toma el p* de forma perezosa.
//! Sin historial (n = 0) el intervalo de Wilson es [0, 1] y la convicción es
//! el piso para cualquier p*, así que no se calcula (la sonda de banda y la τ
//! de la orden que hay detrás tienen coste). En cualquier otro caso devuelve
//! bit a bit lo mismo que `conviccion_de_rama_en`.

use god_engine_core::{conviccion_de_rama_con, conviccion_de_rama_en, TasaAcierto};
use std::cell::Cell;

/// Verdad: para todo historial (n ≤ 60, todos los aciertos), piso y p*
/// (también fuera de (0, 1) y NaN), la versión perezosa devuelve los mismos
/// bits que la directa.
#[test]
fn cl91_la_conviccion_perezosa_es_la_misma() {
    let pisos = [0.0, 0.3, 0.6, 0.95];
    let ps = [0.2, 0.31, 0.5, 0.8, 0.0, 1.0, f64::NAN];
    for n in 0..=60u32 {
        for aciertos in 0..=n {
            let reg = TasaAcierto { n, aciertos };
            for &piso in &pisos {
                for &p in &ps {
                    let directa = conviccion_de_rama_en(&reg, piso, p);
                    let perezosa = conviccion_de_rama_con(&reg, piso, || p);
                    assert_eq!(
                        directa.to_bits(),
                        perezosa.to_bits(),
                        "n = {n}, aciertos = {aciertos}, piso = {piso}, p* = {p}"
                    );
                }
            }
        }
    }
}

/// Verdad: sin historial no se calcula el p*; con historial se calcula una
/// sola vez.
#[test]
fn cl91_sin_historial_no_se_calcula_el_p_estrella() {
    let llamadas = Cell::new(0u32);
    let p = || {
        llamadas.set(llamadas.get() + 1);
        0.31
    };
    let c = conviccion_de_rama_con(&TasaAcierto::default(), 0.6, p);
    assert_eq!(c, 0.6);
    assert_eq!(llamadas.get(), 0, "con n = 0 el p* no debe calcularse");

    let reg = TasaAcierto { n: 40, aciertos: 30 };
    let _ = conviccion_de_rama_con(&reg, 0.6, || {
        llamadas.set(llamadas.get() + 1);
        0.31
    });
    assert_eq!(llamadas.get(), 1, "con historial el p* se calcula una vez");
}
