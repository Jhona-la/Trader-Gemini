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
use strategy_core::{stat_arb::StatArbEngine, vecm_arbitrage::JohansenVecmEngine, QuantumStrategy};

/// Esperado: mismos < activos (no coinciden en signo en todos los casos en
/// que alguno vota). Medido hoy: coinciden en 300/300; falla de forma
/// determinista.
#[test]
#[ignore = "OPEN F3.7-05 (Línea C, disputado: en vivo sólo en el arranque frío de StatArb): sin statarb_ou_zscore, StatArb cae a vecm_zscore, el mismo escalar que vota VECM, y los dos votan −sign(z) para |z| > 1,5: una variable contada dos veces en el consenso"]
fn cf3_dos_votos_del_consenso_no_son_la_misma_variable() {
    let reg = Arc::new(OmniscientRegistry::new());
    let mut sa = StatArbEngine::new(30, 1.5);
    let mut ve = JohansenVecmEngine::default();
    assert!(sa.init(reg.clone()).is_ok());
    assert!(ve.init(reg.clone()).is_ok());
    let (mut activos, mut mismos) = (0u32, 0u32);
    for i in -300..=300 {
        let z = i as f64 / 100.0;
        reg.set("vecm_zscore", z);
        let (a, b) = (sa.evaluate(), ve.evaluate());
        if a != 0.0 || b != 0.0 {
            activos += 1;
            if a.signum() == b.signum() {
                mismos += 1;
            }
        }
    }
    println!("observado: StatArb y VECM coinciden en signo en {mismos}/{activos} casos activos");
    assert!(activos > 0, "precondición: hay votos activos");
    assert!(
        mismos < activos,
        "StatArb y VECM coinciden en signo en {mismos}/{activos}: una variable contada dos veces"
    );
}
