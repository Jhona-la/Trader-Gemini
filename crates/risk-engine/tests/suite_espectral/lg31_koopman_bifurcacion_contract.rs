//! LG-31 (Línea G, ola 10; hallazgo LG10-01 / LG10-T-01, confirmado por dos
//! escépticos): el atenuador de Koopman recibe `koopman_bifurcacion`, que el
//! productor publica como `max_i |λ_i|` (crates/feature-engine/src/koopman_dmd.rs:
//! `bifurcacion_max: modulo0`; su doc dice «≥ 1,0 es régimen de bifurcación»),
//! y lo juzgaba contra 0,10, como si fuera una distancia que vale ≈ 0 en régimen
//! estable. En tape real |λ| ≈ 0,77 (GLM 150: p50 0,7687, 0 % ≥ 1), así que
//! el factor quedaba en 1/(1 + 5·0,67) ≈ 0,23 en todo régimen sano: recortaba
//! la Kelly del cierre y, en el host, multiplicaba la apuesta mínima de
//! exploración por 0,23 y la dejaba bajo el mínimo ejecutable (veto absorbente:
//! sin operaciones no hay evidencia que lo corrija).
//!
//! Las entradas de estas pruebas respetan el dominio del productor:
//! `bifurcacion = max|λ| ≥ |λ_dominante|` (hoy son el mismo número).

use risk_engine::kelly_envelope::RiskEnvelope;

/// |λ| medido en tape real de BTC (GLM 150) y los cuantiles p95 / p99.
const LAMBDA_TAPE: [f64; 3] = [0.7687, 0.814, 0.95];

#[test]
fn lg31_regimen_contractivo_del_tape_no_atenua() {
    for &l in &LAMBDA_TAPE {
        assert_eq!(
            RiskEnvelope::koopman_stability_dampener(l, l),
            1.0,
            "|λ| = {l} (contractivo) no debe atenuar la Kelly"
        );
    }
    assert_eq!(RiskEnvelope::koopman_stability_dampener(1.0, 1.0), 1.0, "frontera unitaria");
}

#[test]
fn lg31_fuera_del_circulo_unidad_si_atenua() {
    let d110 = RiskEnvelope::koopman_stability_dampener(1.10, 1.10);
    let d130 = RiskEnvelope::koopman_stability_dampener(1.30, 1.30);
    assert!((d110 - 1.0 / (1.0 + 3.0 * 0.10 + 5.0 * 0.10)).abs() < 1e-12, "{d110}");
    assert!((d130 - 1.0 / (1.0 + 3.0 * 0.30 + 5.0 * 0.30)).abs() < 1e-12, "{d130}");
    assert!(d130 < d110 && d110 < 1.0);
}

/// La sonda de exploración de una cuenta micro sin edge medido (n < 30) no
/// puede depender de Koopman en régimen contractivo: con |λ| del tape debe dar
/// exactamente lo mismo que sin lectura de Koopman (NaN ⇒ neutral).
#[test]
fn lg31_la_sonda_micro_no_queda_vetada_en_regimen_contractivo() {
    let fresh = RiskEnvelope::new();
    let (capital, stop, min_notional) = (13.0, 0.002, 5.0);
    let (lev_neutro, op_neutro) =
        fresh.max_leverage_spectral(capital, stop, min_notional, 1.64, 10.0, f64::NAN, f64::NAN);
    assert!(op_neutro && lev_neutro > 0.0, "precondición: la sonda neutral es operable");
    for &l in &LAMBDA_TAPE {
        let (lev, op) = fresh.max_leverage_spectral(capital, stop, min_notional, 1.64, 10.0, l, l);
        assert!(op, "|λ| = {l}: la sonda micro no debe vetarse en régimen contractivo");
        assert_eq!(lev.to_bits(), lev_neutro.to_bits(), "|λ| = {l}: mismo apalancamiento que sin Koopman");
    }
}
