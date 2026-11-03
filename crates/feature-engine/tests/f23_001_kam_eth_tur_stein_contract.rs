//! # Contrato Formal F23-001 — KAM, Chirikov, ETH/MBL, Lattice-Boltzmann D1Q3, TUR y James-Stein
//!
//! Invariantes matemáticas y físicas comprobadas:
//! - **C1:** El criterio de Chirikov distingue toros KAM intactos ($S < 0.75$) de difusión caótica ($S \ge 1.0$).
//! - **C2:** La estadística de espaciamiento de niveles de Dyson-Mehta clasifica termalización ETH ($\langle r \rangle \approx 0.536$) vs localización cuántica MBL ($\langle r \rangle \le 0.42$).
//! - **C3:** La hidrodinámica LBM D1Q3 preserva densidad no negativa y viscosidad cinemática $\nu = (\tau - 0.5)/3$.
//! - **C4:** La relación de incertidumbre termodinámica (TUR) de Barato-Seifert $\text{Var}(J)/\langle J \rangle^2 \ge 2/\sigma$ se satisface estrictamente.
//! - **C5:** La contracción de James-Stein para $d=4$ contrae la norma euclídea $\|\hat{\mu}_{\text{JS}}\| \le \|y\|$ con factor $c \in [0, 1]$.
//! - **C6:** El motor maestro `KamEthTurSteinMasterEngine` es resiliente a patologías numéricas sin ningún pánico.

use feature_engine::kam_eth_tur_stein::*;

#[test]
fn c1_kam_chirikov_resonance_overlap_regimes() {
    let engine = KamChirikovResonanceEngine;

    // Caso 1: Frecuencias bien separadas con resonancias angostas (S < 0.75)
    // freq_1 = 1.0, freq_2 = 1.8 (delta_omega = 0.8)
    // width_1 = 0.1, width_2 = 0.1 => S = (0.1 + 0.1) / (2 * 0.8) = 0.2 / 1.6 = 0.125
    let rep_stable = engine.evaluate(1.0, 1.8, 0.1, 0.1);
    assert_eq!(rep_stable.regime, KamChirikovRegime::IntactKamToriStable);
    assert!(rep_stable.chirikov_parameter < 0.75);
    assert!(
        rep_stable.regular_tori_fraction > 0.80,
        "La fracción de toros regulares debe ser alta, fue {}",
        rep_stable.regular_tori_fraction
    );

    // Caso 2: Frecuencias muy cercanas con resonancias anchas (solapamiento caótico S >= 1.0)
    // freq_1 = 1.0, freq_2 = 1.05 (delta_omega = 0.05)
    // width_1 = 0.1, width_2 = 0.1 => S = 0.2 / (2 * 0.05) = 2.0
    let rep_chaos = engine.evaluate(1.0, 1.05, 0.1, 0.1);
    assert_eq!(
        rep_chaos.regime,
        KamChirikovRegime::ChirikovChaoticDiffusion
    );
    assert!(rep_chaos.chirikov_parameter >= 1.0);
    assert!(
        rep_chaos.regular_tori_fraction < 0.20,
        "La fracción de toros debe ser muy baja en régimen caótico, fue {}",
        rep_chaos.regular_tori_fraction
    );
}

#[test]
fn c2_eth_vs_mbl_orderbook_level_spacing_ratio() {
    let engine = EthMblOrderbookEngine;

    // Caso 1: Espectro Wigner-Dyson (repulsión de niveles uniforme característica de ETH)
    // Espaciamientos alternados suavemente: ~1.0, 1.1, 0.9, 1.05, 0.95...
    let eth_levels = [
        0.0, 1.0, 2.1, 3.0, 4.05, 5.0, 6.1, 7.05, 8.1, 9.15, 10.1, 11.15,
    ];
    let rep_eth = engine.evaluate(&eth_levels);

    assert!(
        rep_eth.mean_level_spacing_ratio >= 0.48,
        "La razón media de Dyson-Mehta en ETH debe ser >= 0.48, fue {}",
        rep_eth.mean_level_spacing_ratio
    );
    assert_eq!(rep_eth.regime, EthMblRegime::ThermalizedEth);
    assert!(rep_eth.eth_affinity > rep_eth.mbl_affinity);

    // Caso 2: Espectro tipo Poisson con acumulación / niveles muy próximos (MBL)
    let mbl_levels = [0.0, 0.02, 1.5, 1.51, 3.2, 3.21, 5.0, 5.01, 8.0, 8.01];
    let rep_mbl = engine.evaluate(&mbl_levels);

    assert!(
        rep_mbl.mean_level_spacing_ratio <= 0.42,
        "La razón media en MBL debe ser <= 0.42, fue {}",
        rep_mbl.mean_level_spacing_ratio
    );
    assert_eq!(rep_mbl.regime, EthMblRegime::ManyBodyLocalizedMbl);
}

#[test]
fn c3_lattice_boltzmann_d1q3_hydrodynamics_and_viscosity() {
    let tau = 0.8;
    let mut lbm = LatticeBoltzmannD1Q3Engine::new(tau);

    let injected = [1.0; LATTICE_D1Q3_NODES];
    let rep = lbm.step(&injected);

    assert!(
        rep.mean_density > 0.0,
        "La densidad media debe ser positiva"
    );
    assert!(
        rep.mean_velocity.abs() <= 0.5,
        "La velocidad debe permanecer en régimen subsónico estable"
    );

    // Comprobar la fórmula analítica de viscosidad cinemática de Chapman-Enskog para LBM:
    // \nu = (tau - 0.5) / 3
    let expected_nu = (tau - 0.5) / 3.0;
    assert!(
        (rep.kinematic_viscosity - expected_nu).abs() < 1e-12,
        "Viscosidad LBM {} difiere de la analítica {}",
        rep.kinematic_viscosity,
        expected_nu
    );
}

#[test]
fn c4_barato_seifert_tur_inequality_bound() {
    let engine = BaratoSeifertTurEngine;

    // Corriente de flujo con media 2.0, varianza 1.0, disipación de entropía 4.0
    // Fano ratio: Var(J) / <J>^2 = 1.0 / 4.0 = 0.25
    // Cota TUR: 2 / sigma = 2 / 4 = 0.50
    // En este caso 0.25 < 0.50 violaría TUR física si fuera un sistema en estado estacionario continuo;
    // probamos con valores físicos donde Var(J) = 3.0, <J> = 2.0 => Fano = 3/4 = 0.75 >= 0.50
    let rep = engine.evaluate(2.0, 3.0, 4.0);

    assert!(
        rep.satisfies_tur_inequality,
        "La desigualdad TUR debe cumplirse"
    );
    assert!(
        rep.fano_uncertainty_ratio >= rep.theoretical_tur_bound,
        "Fano {} debe ser >= cota TUR {}",
        rep.fano_uncertainty_ratio,
        rep.theoretical_tur_bound
    );
    assert!(rep.tur_efficiency_factor >= 1.0);
}

#[test]
fn c5_james_stein_bayesian_shrinkage_norm_reduction() {
    let engine = JamesSteinTweedieShrinkageEngine;

    let raw_drifts = [2.0, -1.5, 0.8, 1.2];
    let noise_var = 0.5;

    let rep = engine.evaluate(&raw_drifts, noise_var);

    assert!(
        rep.shrinkage_factor > 0.0 && rep.shrinkage_factor < 1.0,
        "El factor de contracción JS debe estar estrictamente en (0, 1), fue {}",
        rep.shrinkage_factor
    );

    // Comprobar que la norma del vector contraído es estrictamente menor
    let norm_raw_sq: f64 = rep.raw_drift.iter().map(|&x| x * x).sum();
    let norm_shrunken_sq: f64 = rep.shrunken_drift.iter().map(|&x| x * x).sum();

    assert!(
        norm_shrunken_sq < norm_raw_sq,
        "La norma contraída ({}) debe ser menor que la original ({})",
        norm_shrunken_sq,
        norm_raw_sq
    );

    // Corrección de Tweedie para la componente extrema (+2.0) debe ser negativa (reversión a la media)
    assert!(
        rep.tweedie_winner_correction < 0.0,
        "La corrección de Tweedie para drift positivo debe ser negativa, fue {}",
        rep.tweedie_winner_correction
    );
}

#[test]
fn c6_kam_eth_tur_stein_master_engine_zero_panic() {
    let mut master = KamEthTurSteinMasterEngine::new(0.9);

    let injected = [0.5; LATTICE_D1Q3_NODES];
    let drifts = [0.1; JS_DIM];

    // Prueba 1: Ejecución normal
    let rep_normal = master.evaluate(
        1.0,
        1.5,
        0.1,
        0.1,
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        &injected,
        1.0,
        1.0,
        2.0,
        &drifts,
        0.1,
    );

    assert!(
        rep_normal.ergodic_predictability_score >= 0.0
            && rep_normal.ergodic_predictability_score <= 1.0
    );

    // Prueba 2: Resiliencia ante NaNs, ceros e infinitos
    let bad_injected = [f64::NAN; LATTICE_D1Q3_NODES];
    let bad_drifts = [f64::INFINITY; JS_DIM];
    let rep_bad = master.evaluate(
        f64::NAN,
        0.0,
        f64::INFINITY,
        -1.0,
        &[],
        &bad_injected,
        f64::NAN,
        f64::INFINITY,
        0.0,
        &bad_drifts,
        f64::NAN,
    );

    assert!(rep_bad.ergodic_predictability_score.is_finite());
    assert!(rep_bad.lattice_boltzmann.mean_density.is_finite());
    assert!(rep_bad.shrinkage.shrinkage_factor.is_finite());
}
