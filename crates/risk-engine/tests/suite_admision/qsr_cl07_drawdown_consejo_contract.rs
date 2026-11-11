//! QSR-CL-07 — El límite de drawdown del consejo es el d* de QS-D3.
//!
//! `quantum_arena::config::CANONICAL_COUNCIL_MAX_DRAWDOWN` es un literal (un
//! `const` no puede llamar a `powf`). Su documentación decía c = 0,25 de
//! Kelly, que da 1 − 0,05^(1/7) ≈ 0,348 y no el valor del literal. El literal
//! es el d* de ½ Kelly (`FRACCION_KELLY_MAXIMA`) a α = 0,05
//! (`ALFA_FALSACION`): 1 − 0,05^(1/3) ≈ 0,6316. Este contrato ata el literal a
//! `drawdown_de_falsacion`, para que ninguno de los dos cambie solo.

use quantum_arena::config::CANONICAL_COUNCIL_MAX_DRAWDOWN;
use risk_engine::drawdown::{drawdown_de_falsacion, ALFA_FALSACION, FRACCION_KELLY_MAXIMA};

#[test]
fn el_limite_del_consejo_es_el_d_estrella_de_medio_kelly() {
    let d_estrella = drawdown_de_falsacion(FRACCION_KELLY_MAXIMA, ALFA_FALSACION);
    // El literal queda 1 ulp por encima del redondeo correcto de
    // 0,63159685013596134 y `powf` 1 ulp por debajo: 4 ulps de margen.
    let tolerancia = 4.0 * f64::EPSILON * d_estrella;
    assert!(
        (CANONICAL_COUNCIL_MAX_DRAWDOWN - d_estrella).abs() <= tolerancia,
        "límite del consejo {CANONICAL_COUNCIL_MAX_DRAWDOWN} frente a d*(½ Kelly, α) = {d_estrella}"
    );
}

#[test]
fn no_es_el_d_estrella_de_un_cuarto_de_kelly() {
    let d_cuarto = drawdown_de_falsacion(0.25, ALFA_FALSACION);
    let esperado = 1.0 - ALFA_FALSACION.powf(1.0 / 7.0);
    assert!(
        (d_cuarto - esperado).abs() <= 4.0 * f64::EPSILON,
        "d*(¼ Kelly) = {d_cuarto}"
    );
    assert!(
        CANONICAL_COUNCIL_MAX_DRAWDOWN - d_cuarto > 0.25,
        "el límite del consejo ({CANONICAL_COUNCIL_MAX_DRAWDOWN}) no es d*(¼ Kelly) = {d_cuarto}"
    );
}
