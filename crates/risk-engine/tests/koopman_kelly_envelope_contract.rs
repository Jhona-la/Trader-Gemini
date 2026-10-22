//! CONTRATO FORMAL DE ACOPLAMIENTO DE ESTABILIDAD KOOPMAN/DMD EN KELLY ENVELOPE (Ola Ω215)
//!
//! QUÉ: Verificación de la atenuación espectral del riesgo ante bifurcaciones no lineales.
//! POR QUÉ: Los observables Koopman streaming (|λ|, Δ_bif) de GLM 149 estaban desacoplados de riesgo.
//! PARA QUÉ: Evitar sobre-exposición durante rupturas de régimen en cuentas micro ($13 USD).
//! CÓMO: 4 contratos matemáticos validando régimen estable, régimen divergente, inmunidad NaN y pisos.

use risk_engine::kelly_envelope::RiskEnvelope;

#[test]
fn c1_koopman_stable_regime_preserves_full_kelly_fraction() {
    // Para modos asintóticamente estables (|λ| <= 1.0) y baja bifurcación (<= 0.10)
    let dampener = RiskEnvelope::koopman_stability_dampener(0.98, 0.05);
    assert_eq!(
        dampener, 1.0,
        "Modos contractivos (|λ| <= 1.0) no deben atenuar la fracción de Kelly"
    );

    let dampener_boundary = RiskEnvelope::koopman_stability_dampener(1.00, 0.10);
    assert_eq!(
        dampener_boundary, 1.0,
        "Frontera unitaria (|λ| = 1.0, bif = 0.10) preserva fracción 1.0"
    );
}

#[test]
fn c2_koopman_divergent_bifurcation_dampens_hyperbolically() {
    // Modo divergente (|λ| = 1.20, bif = 0.10):
    // exceso_modulo = 0.20, exceso_bif = 0.0
    // dampener = 1.0 / (1.0 + 3.0 * 0.20) = 1.0 / 1.60 = 0.625
    let dampener_divergent = RiskEnvelope::koopman_stability_dampener(1.20, 0.10);
    let expected = 1.0 / 1.60;
    assert!(
        (dampener_divergent - expected).abs() < 1e-9,
        "Atenuación por divergencia |λ|=1.20 debe ser 0.625: got {}",
        dampener_divergent
    );

    // Bifurcación severa (|λ| = 1.20, bif = 0.30):
    // exceso_modulo = 0.20, exceso_bif = 0.20
    // dampener = 1.0 / (1.0 + 3.0 * 0.20 + 5.0 * 0.20) = 1.0 / (1.0 + 0.60 + 1.00) = 1.0 / 2.60 ≈ 0.384615
    let dampener_bifurcated = RiskEnvelope::koopman_stability_dampener(1.20, 0.30);
    let expected_bif = 1.0 / 2.60;
    assert!(
        (dampener_bifurcated - expected_bif).abs() < 1e-9,
        "Atenuación con bifurcación combinada: {} vs {}",
        dampener_bifurcated,
        expected_bif
    );

    assert!(
        dampener_bifurcated < dampener_divergent,
        "Mayor bifurcación debe reducir estrictamente la fracción autorizada"
    );
}

#[test]
fn c3_koopman_nan_and_infinite_immunity() {
    // Si los observables Koopman no están maduros o traen NaN/Inf,
    // fallback neutral sin interrumpir la operativa
    assert_eq!(
        RiskEnvelope::koopman_stability_dampener(f64::NAN, 0.05),
        1.0
    );
    assert_eq!(
        RiskEnvelope::koopman_stability_dampener(0.95, f64::NAN),
        1.0
    );
    assert_eq!(
        RiskEnvelope::koopman_stability_dampener(f64::NAN, f64::NAN),
        1.0
    );
    assert_eq!(
        RiskEnvelope::koopman_stability_dampener(f64::INFINITY, 0.05),
        1.0
    );
}

#[test]
fn c4_koopman_spectral_exposure_budget_continuity() {
    let mut env = RiskEnvelope::new();
    // Registrar 100 operaciones con edge comprobado (60% WR, payoff 2.0)
    for i in 0..100 {
        env.record_trade(i % 10 < 6, 20.0, -10.0);
    }

    let capital = 13.0; // Cuenta micro del operador
    let stop_pct = 0.015; // 1.5% stop
    let cost_pct = 0.001; // 0.1% roundtrip taker fee

    // 1. Presupuesto en régimen estable
    let budget_stable = env
        .exposure_budget_spectral(capital, stop_pct, cost_pct, 1.64, 10.0, 0.95, 0.05)
        .expect("Presupuesto estable debe existir");

    // 2. Presupuesto en régimen caótico / bifurcación
    let budget_chaotic = env
        .exposure_budget_spectral(capital, stop_pct, cost_pct, 1.64, 10.0, 1.30, 0.35)
        .expect("Presupuesto caótico debe existir");

    assert!(
        budget_chaotic.risk_budget_usd() < budget_stable.risk_budget_usd(),
        "El presupuesto de riesgo en dólares debe reducirse en régimen caótico: {} vs {}",
        budget_chaotic.risk_budget_usd(),
        budget_stable.risk_budget_usd()
    );

    assert!(
        budget_chaotic.max_notional() < budget_stable.max_notional(),
        "El nocional máximo autorizado debe reducirse en régimen caótico: {} vs {}",
        budget_chaotic.max_notional(),
        budget_stable.max_notional()
    );

    // Ambas deben mantenerse estrictamente positivas y por debajo del axioma de ruina
    assert!(budget_chaotic.risk_budget_usd() > 0.0);
    assert!(budget_chaotic.risk_budget_usd() <= capital * 0.25);
}
