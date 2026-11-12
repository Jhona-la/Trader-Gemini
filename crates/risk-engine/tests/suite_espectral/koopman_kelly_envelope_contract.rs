//! CONTRATO FORMAL DE ACOPLAMIENTO DE ESTABILIDAD KOOPMAN/DMD EN KELLY ENVELOPE (Ola Ω215)
//!
//! QUÉ: Verificación de la atenuación espectral del riesgo ante bifurcaciones no lineales.
//! POR QUÉ: Los observables Koopman streaming (|λ|, Δ_bif) de GLM 149 estaban desacoplados de riesgo.
//! PARA QUÉ: Evitar sobre-exposición durante rupturas de régimen en cuentas micro ($13 USD).
//! CÓMO: 4 contratos matemáticos validando régimen estable, régimen divergente, inmunidad NaN y pisos.
//!
//! LG-31: el productor publica `bifurcacion_max` = max_i |λ_i| (umbral 1,0, igual
//! que el módulo), así que las entradas posibles tienen bif = |λ| dominante. Los
//! valores anteriores (bif 0,05–0,35 con |λ| ≥ 0,98) no podían darse.

use risk_engine::kelly_envelope::{nocional_por_riesgo, RiskEnvelope};

#[test]
fn c1_koopman_stable_regime_preserves_full_kelly_fraction() {
    // Para modos asintóticamente estables (|λ| <= 1.0)
    let dampener = RiskEnvelope::koopman_stability_dampener(0.98, 0.98);
    assert_eq!(
        dampener, 1.0,
        "Modos contractivos (|λ| <= 1.0) no deben atenuar la fracción de Kelly"
    );

    let dampener_boundary = RiskEnvelope::koopman_stability_dampener(1.00, 1.00);
    assert_eq!(
        dampener_boundary, 1.0,
        "Frontera unitaria (|λ| = 1.0) preserva fracción 1.0"
    );
}

#[test]
fn c2_koopman_divergent_bifurcation_dampens_hyperbolically() {
    // Modo divergente (|λ| = 1.20 = bif):
    // exceso_modulo = 0.20, exceso_bif = 0.20
    // dampener = 1.0 / (1.0 + 3.0 * 0.20 + 5.0 * 0.20) = 1.0 / 2.60 ≈ 0.384615
    let dampener_divergent = RiskEnvelope::koopman_stability_dampener(1.20, 1.20);
    let expected = 1.0 / 2.60;
    assert!(
        (dampener_divergent - expected).abs() < 1e-9,
        "Atenuación por divergencia |λ|=1.20 debe ser 1/2,60: got {}",
        dampener_divergent
    );

    // Divergencia mayor (|λ| = 1.40 = bif):
    // dampener = 1.0 / (1.0 + 3.0 * 0.40 + 5.0 * 0.40) = 1.0 / 4.20 ≈ 0.238095
    let dampener_bifurcated = RiskEnvelope::koopman_stability_dampener(1.40, 1.40);
    let expected_bif = 1.0 / 4.20;
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
    // RET-48: `exposure_budget_spectral` se retiró (sin consumidor operativo).
    // Su contrato se migra a la cadena viva: f = risk_fraction_spectral,
    // B = C·f y N_max = B/(d + c) con `nocional_por_riesgo` (CL-159, el
    // mismo paso que dimensiona LH-04 en el gate).
    let mut env = RiskEnvelope::new();
    // Registrar 100 operaciones con edge comprobado (60% WR, payoff 2.0)
    for i in 0..100 {
        env.record_trade(i % 10 < 6, 20.0, -10.0);
    }

    let capital = 13.0; // Cuenta micro del operador
    let stop_pct = 0.015; // 1.5% stop
    let cost_pct = 0.001; // 0.1% roundtrip taker fee

    // 1. Régimen estable y 2. régimen caótico / bifurcación.
    // LG-31 (rescate GLM-166): bif ≡ |λ| dominante del productor (umbral 1,0),
    // así que bif acompaña al módulo (0.95/0.95 y 1.30/1.30) — los valores
    // legacy (bif 0.05–0.35 con |λ| ≥ 0.98) no podían darse.
    let f_stable = env.risk_fraction_spectral(1.64, 10.0, 0.95, 0.95);
    let f_chaotic = env.risk_fraction_spectral(1.64, 10.0, 1.30, 1.30);
    let budget_stable = capital * f_stable;
    let budget_chaotic = capital * f_chaotic;
    let notional_stable =
        nocional_por_riesgo(f_stable, capital, stop_pct, cost_pct).unwrap_or(f64::NAN);
    let notional_chaotic =
        nocional_por_riesgo(f_chaotic, capital, stop_pct, cost_pct).unwrap_or(f64::NAN);

    assert!(
        budget_chaotic < budget_stable,
        "El presupuesto de riesgo en dólares debe reducirse en régimen caótico: {} vs {}",
        budget_chaotic,
        budget_stable
    );
    assert!(
        notional_chaotic < notional_stable,
        "El nocional máximo autorizado debe reducirse en régimen caótico: {} vs {}",
        notional_chaotic,
        notional_stable
    );
    // La pérdida al stop con el coste es el presupuesto: N·(d + c) = B.
    assert!(
        (notional_chaotic * (stop_pct + cost_pct) - budget_chaotic).abs()
            <= 4.0 * f64::EPSILON * budget_chaotic
    );

    // Ambas deben mantenerse estrictamente positivas y por debajo del axioma de ruina
    assert!(budget_chaotic > 0.0);
    assert!(budget_chaotic <= capital * 0.25);

    // El techo vivo del host (max_leverage_spectral, N_max/C sin coste) se
    // atenúa con la misma fracción (mismos bif ≡ |λ| del productor, LG-31).
    let (lev_stable, ok_stable) =
        env.max_leverage_spectral(capital, stop_pct, 5.0, 1.64, 10.0, 0.95, 0.95);
    let (lev_chaotic, ok_chaotic) =
        env.max_leverage_spectral(capital, stop_pct, 5.0, 1.64, 10.0, 1.30, 1.30);
    assert!(ok_stable && ok_chaotic);
    assert_eq!(lev_stable, f_stable / stop_pct);
    assert_eq!(lev_chaotic, f_chaotic / stop_pct);
}
