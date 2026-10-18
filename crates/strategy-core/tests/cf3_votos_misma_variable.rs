//! C-F3 / F3.7-05 (Línea C): StatArb (en su respaldo) y VECM leen el mismo
//! escalar, `vecm_zscore`, y el consenso los suma como votos independientes.
//!
//! Teoría que se prueba: un consenso que suma votos como evidencia
//! independiente no debe contener dos votos que sean la misma función
//! monótona de una sola variable; si coinciden en signo en el 100 % de los
//! casos activos, la confianza del compuesto cuenta dos veces un único dato.
//!
//! Código: con la clave propia `statarb_ou_zscore` ausente (ni del símbolo,
//! ni de la moneda, ni global), `StatArbEngine::evaluate_for_coin` cae a
//! `vecm_zscore` y vota −sign(z)·tanh((|z| − 1,5)/1,5); VECM vota
//! −sign(z)·min((|z| − 1,5)/1,5, 1). Los dos valen 0 con |z| ≤ 1,5 y fuera de
//! ahí tienen el mismo signo por construcción.
//!
//! Alcance (escéptico de completitud, hallazgo disputado): en vivo el
//! respaldo de StatArb sólo se recorre mientras NINGUNA moneda ha madurado su
//! SDE (el núcleo escribe `statarb_ou_zscore` también en global), es decir,
//! en el arranque frío. La duplicación permanente está en otro sitio: sin
//! par spot, `vecm_zscore` y `quantum_position_deviation` son la misma
//! expresión, y VECM, conformal y el oscilador cuántico votan sobre ella
//! (faltante 3 de F3.7, prueba en signal-engine, otro grupo). Esta prueba
//! documenta el camino de StatArb, determinista.
//!
//! Generador: barrido determinista z ∈ [−3, 3] con paso 0,01 en un registro
//! que sólo tiene `vecm_zscore`.
//!
//! Convención: una prueba que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::{stat_arb::StatArbEngine, QuantumStrategy};

// CL-113 (W-2 de la revisión del PR #60): la prueba que estaba aquí
// (`cf3_dos_votos_del_consenso_no_son_la_misma_variable`) era vacua: StatArb
// no vota sobre `vecm_zscore` desde CL-97, así que nunca votaba en su
// escenario, y `0.0.signum()` es +1. La sustituye
// `god-engine-core/tests/cf3_reversion_base.rs::cl113_statarb_y_vecm_con_spot_no_son_la_misma_variable`,
// con spot y reversión demostrada (OPEN F3.7-05).

/// CL-97. Verdad: sin claves propias StatArb no vota con ningún valor de
/// `vecm_zscore`.
#[test]
fn cl97_statarb_no_vota_sobre_vecm_zscore() {
    let reg = Arc::new(OmniscientRegistry::new());
    let mut sa = StatArbEngine::new(30, 1.5);
    assert!(sa.init(reg.clone()).is_ok());
    for i in -300..=300 {
        reg.set("vecm_zscore", i as f64 / 100.0);
        assert_eq!(sa.evaluate(), 0.0, "vecm_zscore = {}", i as f64 / 100.0);
    }
}
