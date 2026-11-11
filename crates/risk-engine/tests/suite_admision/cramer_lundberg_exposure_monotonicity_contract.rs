//! Contrato de Monotonía y Física Actuarial de Cramér-Lundberg (Ola Ω130)
//!
//! F2.1-01: Verifica que la cota actuarial de Cramér-Lundberg sobre exposición
//! f <= tope_streak / m donde m = ln(1/ε)/R sea estrictamente monótona no-decreciente
//! en R: más edge medido nunca veta lo que menos edge admite.

use risk_engine::correlation_guard::{veto_por_riesgo_cramer_lundberg, veto_por_riesgo_real_medido};

#[test]
fn lundberg_monotonicity_contract_high_edge_never_vetoes_low_edge_admission() {
    let riesgos = [0.035, 0.040, 0.045];
    let rho = Some(0.25);
    let tope_streak = 0.10;
    let eps = 0.05;

    // Rejilla de coeficientes de ajuste R desde 0.1 hasta 2000.0
    let rs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0, 100.0, 500.0, 2000.0];

    for w in rs.windows(2) {
        let r_low = w[0];
        let r_high = w[1];

        let veto_low = veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope_streak, Some(r_low), eps);
        let veto_high = veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope_streak, Some(r_high), eps);

        // Si con alto edge se veta, con bajo edge OBLIGATORIAMENTE debe vetarse también.
        // Es decir: veto_high => veto_low, o equivalentemente: !veto_high || veto_low.
        assert!(
            !veto_high || veto_low,
            "Violación de monotonicidad F2.1-01: R_high={r_high} vetó pero R_low={r_low} admitió!"
        );
    }
}

#[test]
fn lundberg_none_fallback_and_boundary_conditions() {
    let riesgos = [0.03, 0.03, 0.03];
    let rho = Some(0.15);
    let tope_streak = 0.08;
    let eps = 0.05;

    // Sin evidencia de R (None): debe coincidir exactamente con el veto real medido al tope_streak
    let veto_none = veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope_streak, None, eps);
    let veto_base = veto_por_riesgo_real_medido(&riesgos, rho, tope_streak);
    assert_eq!(veto_none, veto_base, "Sin R debe equivaler bit a bit a veto_por_riesgo_real_medido(tope_streak)");

    // R no positivo (0.0 o negativo o NaN): debe comportarse como None
    let veto_zero = veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope_streak, Some(0.0), eps);
    assert_eq!(veto_zero, veto_base, "R <= 0 debe caer al tope_streak sin inventar cota");

    let veto_nan = veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope_streak, Some(f64::NAN), eps);
    assert_eq!(veto_nan, veto_base, "R NaN debe caer al tope_streak de manera segura");
}

#[test]
fn lundberg_effective_cap_preserves_streak_ceiling() {
    let riesgos = [0.05, 0.05, 0.05];
    let rho = Some(0.5);
    let tope_streak = 0.05; // tope muy estricto
    let eps = 0.05;

    // Incluso con edge infinito (R=10000), el tope efectivo NUNCA excede tope_streak
    let r_huge = Some(10000.0);
    let veto_huge = veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope_streak, r_huge, eps);
    let veto_base = veto_por_riesgo_real_medido(&riesgos, rho, tope_streak);
    assert_eq!(veto_huge, veto_base, "Edge infinito no puede relajar más allá del tope_streak");
}
