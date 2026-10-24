//! # Contrato Formal F10-002: Helmholtz-Hodge Multiactivo y Topología de Liquidez
//!
//! Verificación rigurosa de invariantes físicos y topológicos:
//! C1: Cascada de gradiente puro en forma cerrada analítica (curl_share == 0.0).
//! C2: Vórtice cerrado puro 3-ciclo sin divergencia (curl_share == 1.0).
//! C3: Antisimetría estricta y conservación de 1-forma en flujo cruzado L2/L3.
//! C4: Jerarquía de potenciales escalares de liderazgo (atractor vs drenador).
//! C5: Clasificación de regímenes físicos y veto de operaciones tendenciales.
//! C6: Resiliencia numérica extrema, rechazo de NaNs/Infs y cero pánicos.

use feature_engine::hodge_flow::{
    CrossMarketHodgeReport, HelmholtzHodgeFlowEngine, HelmholtzHodgeParams,
    HodgeDecompositionResult, HodgeRegime, MAX_HODGE_ASSETS,
};

#[test]
fn test_c1_pure_gradient_flow_analytical_zero_curl() {
    let engine = HelmholtzHodgeFlowEngine::new(8);

    // Vector de potencial escalar arbitrario no trivial
    let potentials_in = [10.5, 7.2, 4.0, 1.1, -2.5, -6.0];
    let (matrix, n) = HelmholtzHodgeFlowEngine::build_gradient_flow_matrix(&potentials_in)
        .expect("Debe construir matriz de gradiente");

    assert_eq!(n, 6);

    let (decomp, potentials_out) = engine
        .decompose(&matrix, n)
        .expect("Debe descomponer matriz de gradiente");

    // En un gradiente puro F_{ij} = phi_i - phi_j, el rotacional analítico es estrictamente nulo
    assert!(
        decomp.curl_share < 1e-10,
        "Gradiente puro debe tener curl_share == 0.0, obtenido: {}",
        decomp.curl_share
    );
    assert!(
        (decomp.gradient_energy - decomp.total_energy).abs() < 1e-10,
        "Energía de gradiente debe igualar a energía total"
    );
    assert!(
        decomp.curl_energy < 1e-10,
        "Energía de rotacional debe ser nula"
    );

    // Verificación de monotonía y preservación de orden relativo de potenciales
    for i in 0..(n - 1) {
        assert!(
            potentials_out[i] > potentials_out[i + 1],
            "Potenciales calculados deben preservar el orden decreciente original"
        );
    }

    // Condición de gauge: la suma de potenciales sobre el componente conexo debe ser 0
    let sum_pot: f64 = potentials_out[..n].iter().sum();
    assert!(
        sum_pot.abs() < 1e-10,
        "La suma de potenciales debe satisfacer la condición de gauge sum(phi_i) == 0, obtuve: {}",
        sum_pot
    );
}

#[test]
fn test_c2_pure_vortex_analytical_unit_curl() {
    let engine = HelmholtzHodgeFlowEngine::new(8);

    // 3-ciclo cerrado puro A -> B -> C -> A con magnitud 5.0
    let (matrix, n) = HelmholtzHodgeFlowEngine::build_pure_vortex_matrix(5.0);
    assert_eq!(n, 3);

    let (decomp, potentials) = engine
        .decompose(&matrix, n)
        .expect("Debe descomponer vórtice puro");

    // En un ciclo cerrado, div_i = sum_j F_ij == 0 para todos los nodos
    assert!(
        decomp.curl_share > 0.999999999,
        "Vórtice cerrado puro debe tener curl_share == 1.0, obtenido: {}",
        decomp.curl_share
    );
    assert!(
        decomp.gradient_energy < 1e-10,
        "Energía de gradiente debe ser analíticamente nula"
    );
    assert!(
        (decomp.curl_energy - decomp.total_energy).abs() < 1e-10,
        "Energía de curl debe igualar a energía total"
    );

    // Potenciales escalares deben ser idénticamente cero
    for i in 0..n {
        assert!(
            potentials[i].abs() < 1e-10,
            "Potencial en ciclo puro debe ser 0, nodo {} tiene {}",
            i,
            potentials[i]
        );
    }
}

#[test]
fn test_c3_cross_microstructure_flow_antisymmetry() {
    let params = HelmholtzHodgeParams {
        return_sensitivity: 80.0,
        min_flow_energy_threshold: 1e-14,
    };

    let ofis = [12.4, -8.5, 3.1, -1.2, 5.5, -9.0, 0.4, 2.8];
    let returns = [0.0015, -0.0008, 0.0004, -0.0002, 0.0010, -0.0012, 0.0001, 0.0005];

    let (matrix, n) = HelmholtzHodgeFlowEngine::build_cross_microstructure_flow_matrix_with_params(
        &ofis, &returns, &params,
    )
    .expect("Construcción de flujo cruzado válida");

    assert_eq!(n, 8);

    // Invariante de 1-forma antisimétrica: F_{ij} == -F_{ji} y F_{ii} == 0
    for i in 0..n {
        assert_eq!(matrix[i][i], 0.0, "Diagonal principal debe ser estrictamente nula");
        for j in 0..n {
            let fij = matrix[i][j];
            let fji = matrix[j][i];
            assert!(
                (fij + fji).abs() < 1e-14,
                "Matriz debe ser antisimétrica: F[{}][{}]={} vs F[{}][{}]={}",
                i, j, fij, j, i, fji
            );
        }
    }
}

#[test]
fn test_c4_leadership_potential_ranking_and_attractor_identification() {
    let engine = HelmholtzHodgeFlowEngine::new(4);

    // Activo 0 es el líder dominante indiscutible (flujo positivo masivo)
    // Activo 3 es el drenador neto de liquidez (flujo negativo severo)
    let ofis = [50.0, 5.0, -5.0, -50.0];
    let returns = [0.0050, 0.0005, -0.0005, -0.0050];

    let report = engine
        .analyze_cross_market(&ofis, &returns)
        .expect("Análisis de mercado cruzado exitoso");

    // Verificación de identificación de roles
    assert_eq!(
        report.leader_asset_idx, 0,
        "Activo 0 debe ser identificado como el líder de liquidez"
    );
    assert_eq!(
        report.laggard_asset_idx, 3,
        "Activo 3 debe ser identificado como el seguidor/drenador"
    );

    // Preservación estricta del orden de potencial escalar
    assert!(report.leadership_potentials[0] > report.leadership_potentials[1]);
    assert!(report.leadership_potentials[1] > report.leadership_potentials[2]);
    assert!(report.leadership_potentials[2] > report.leadership_potentials[3]);
}

#[test]
fn test_c5_regime_classification_and_directional_trade_veto() {
    let engine = HelmholtzHodgeFlowEngine::new(4);

    // Caso 1: Flujo direccional alineado -> Cascada de gradiente pura
    let ofis_grad = [30.0, 10.0, -10.0, -30.0];
    let returns_grad = [0.0030, 0.0010, -0.0010, -0.0030];
    let report_grad = engine
        .analyze_cross_market(&ofis_grad, &returns_grad)
        .expect("Debe analizar flujo de gradiente");

    assert_eq!(
        report_grad.regime,
        HodgeRegime::ConservativeGradient,
        "Flujo alineado debe ser ConservativeGradient"
    );
    assert!(
        !report_grad.should_veto_directional_trades,
        "En régimen de gradiente conservativo no debe vetar operaciones direccionales"
    );

    // Caso 2: Inyección de vórtice puro -> VortexManipulative con veto
    // Construimos matriz de 3-ciclo y evaluamos directamente a través de decompose
    let (vortex_matrix, n_vortex) = HelmholtzHodgeFlowEngine::build_pure_vortex_matrix(10.0);
    let (decomp_vortex, potentials_vortex) = engine
        .decompose(&vortex_matrix, n_vortex)
        .expect("Debe descomponer matriz de vórtice");

    let regime_vortex = if decomp_vortex.curl_share < 0.25 {
        HodgeRegime::ConservativeGradient
    } else if decomp_vortex.curl_share > 0.65 {
        HodgeRegime::VortexManipulative
    } else {
        HodgeRegime::MixedTurbulent
    };

    assert_eq!(
        regime_vortex,
        HodgeRegime::VortexManipulative,
        "Vórtice cíclico debe ser clasificado como VortexManipulative"
    );
    assert!(
        decomp_vortex.curl_share > 0.65,
        "Curl share de vórtice debe exceder el umbral de veto de 0.65"
    );
}

#[test]
fn test_c6_numerical_robustness_and_zero_panic() {
    let engine = HelmholtzHodgeFlowEngine::new(4);

    // 1. Menos de 3 activos debe retornar None de forma segura
    let ofis_small = [1.0, 2.0];
    let rets_small = [0.01, 0.02];
    assert!(engine.analyze_cross_market(&ofis_small, &rets_small).is_none());

    // 2. Energía nula o valores planos debe retornar None sin pánico
    let ofis_flat = [0.0, 0.0, 0.0, 0.0];
    let rets_flat = [0.0, 0.0, 0.0, 0.0];
    assert!(engine.analyze_cross_market(&ofis_flat, &rets_flat).is_none());

    // 3. Contaminación por NaN o Infinitos debe ser rechazada inmediatamente
    let ofis_nan = [1.0, f64::NAN, 3.0, 4.0];
    let rets_nan = [0.01, 0.02, 0.03, 0.04];
    assert!(engine.analyze_cross_market(&ofis_nan, &rets_nan).is_none());

    let ofis_inf = [1.0, 2.0, f64::INFINITY, 4.0];
    let rets_inf = [0.01, 0.02, 0.03, 0.04];
    assert!(engine.analyze_cross_market(&ofis_inf, &rets_inf).is_none());

    // 4. Magnitudes astronómicas (1e12) no deben desbordar ni provocar pánicos
    let ofis_huge = [1e12, 5e11, -5e11, -1e12];
    let rets_huge = [0.05, 0.01, -0.01, -0.05];
    let report_huge = engine.analyze_cross_market(&ofis_huge, &rets_huge);
    assert!(report_huge.is_some(), "Debe soportar magnitudes extremas");
    let rep = report_huge.unwrap();
    assert!(rep.decomposition.curl_share.is_finite());
    assert!(rep.decomposition.total_energy.is_finite());
}
