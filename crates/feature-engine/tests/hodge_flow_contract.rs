//! CONTRATO FORMAL DE DESCOMPOSICIÓN DE HELMHOLTZ-HODGE SOBRE FLUJOS DE LIQUIDEZ (OLA Ω37)
//!
//! Verifica analítica y numéricamente los teoremas fundamentales de la descomposición
//! ortogonal L2 sobre grafos completos en feature-engine.

use feature_engine::hodge_flow::{HelmholtzHodgeFlowEngine, MAX_HODGE_ASSETS};

#[test]
fn hodge_contrato_flujo_gradiente_puro_tiene_curl_cero() {
    let engine = HelmholtzHodgeFlowEngine::new(4);
    // Un flujo generado por potenciales escalares locales (ej. OFI = [12.0, 8.0, 4.0, 0.0])
    // F_{ij} = phi_i - phi_j es un gradiente exacto puro.
    let scalars = [12.0, 8.0, 4.0, 0.0];
    let (matrix, n) = HelmholtzHodgeFlowEngine::build_gradient_flow_matrix(&scalars)
        .expect("matriz válida");

    let (res, potentials) = engine.decompose(&matrix, n).expect("descomposición exitosa");

    // En un gradiente puro, el rotacional (curl) debe ser idénticamente 0.0
    assert!(
        res.curl_share < 1e-12,
        "El curl de un gradiente puro debe ser 0.0: got {}",
        res.curl_share
    );
    assert!(
        (res.gradient_energy - res.total_energy).abs() < 1e-12,
        "La energía de gradiente debe ser igual a la energía total: grad={}, total={}",
        res.gradient_energy,
        res.total_energy
    );

    // Los potenciales recuperados deben coincidir con los escalares originales (salvo la constante gauge sum=0)
    let mean_scalar: f64 = scalars.iter().sum::<f64>() / (n as f64);
    for i in 0..n {
        let expected_phi = scalars[i] - mean_scalar;
        assert!(
            (potentials[i] - expected_phi).abs() < 1e-10,
            "Potencial recuperado en nodo {} erróneo: got {}, expected {}",
            i,
            potentials[i],
            expected_phi
        );
    }
}

#[test]
fn hodge_contrato_vortice_puro_tiene_curl_uno() {
    let engine = HelmholtzHodgeFlowEngine::new(3);
    // 3-ciclo cerrado puro A -> B -> C -> A
    let (vortex_matrix, n) = HelmholtzHodgeFlowEngine::build_pure_vortex_matrix(3.5);

    let (res, potentials) = engine.decompose(&vortex_matrix, n).expect("descomposición exitosa");

    // En un vórtice puro cerrado, el curl debe ser idénticamente 1.0 (100% circulación)
    assert!(
        (res.curl_share - 1.0).abs() < 1e-12,
        "El curl de un vórtice puro debe ser 1.0: got {}",
        res.curl_share
    );
    assert!(
        res.gradient_energy < 1e-12,
        "La energía de gradiente de un vórtice cerrado debe ser 0.0: got {}",
        res.gradient_energy
    );

    // Los potenciales escalares deben ser idénticamente cero (divergencia nula)
    for i in 0..n {
        assert!(
            potentials[i].abs() < 1e-12,
            "Potencial en nodo {} debe ser 0: got {}",
            i,
            potentials[i]
        );
    }
}

#[test]
fn hodge_contrato_ortogonalidad_l2_de_componentes() {
    let engine = HelmholtzHodgeFlowEngine::new(4);
    // Flujo mixto con componentes tanto de gradiente como de vórtice
    let mut mixed_matrix = [[0.0_f64; MAX_HODGE_ASSETS]; MAX_HODGE_ASSETS];
    mixed_matrix[0][1] = 5.0;
    mixed_matrix[1][0] = -5.0;
    mixed_matrix[1][2] = 3.0;
    mixed_matrix[2][1] = -3.0;
    mixed_matrix[2][3] = 4.0;
    mixed_matrix[3][2] = -4.0;
    mixed_matrix[3][0] = 2.0;
    mixed_matrix[0][3] = -2.0;

    let (res, _) = engine.decompose(&mixed_matrix, 4).expect("descomposición exitosa");

    // Teorema de Pitágoras / Ortogonalidad L2: ||F||^2 = ||grad||^2 + ||curl||^2
    let sum_energies = res.gradient_energy + res.curl_energy;
    assert!(
        (sum_energies - res.total_energy).abs() < 1e-12,
        "Violación de la ortogonalidad L2 de Helmholtz-Hodge: grad + curl = {}, total = {}",
        sum_energies,
        res.total_energy
    );

    // El curl_share debe estar estrictamente acotado en [0, 1]
    assert!(res.curl_share >= 0.0 && res.curl_share <= 1.0);
}

#[test]
fn hodge_contrato_inmunidad_a_nan_y_grafos_pequenos() {
    let engine = HelmholtzHodgeFlowEngine::new(4);

    // Grafo con N < 3 debe retornar None (sin ciclos no hay rotacional)
    let matrix_2 = [[0.0_f64; MAX_HODGE_ASSETS]; MAX_HODGE_ASSETS];
    assert!(engine.decompose(&matrix_2, 2).is_none());

    // Matriz con NaN debe retornar None de forma segura
    let mut matrix_nan = [[0.0_f64; MAX_HODGE_ASSETS]; MAX_HODGE_ASSETS];
    matrix_nan[0][1] = f64::NAN;
    matrix_nan[1][0] = -1.0;
    assert!(engine.decompose(&matrix_nan, 3).is_none());

    // Matriz con flujo nulo debe retornar None (energía 0)
    let matrix_zero = [[0.0_f64; MAX_HODGE_ASSETS]; MAX_HODGE_ASSETS];
    assert!(engine.decompose(&matrix_zero, 3).is_none());
}

#[test]
fn hodge_contrato_flujo_cruzado_microestructura_produce_curl_dinamico() {
    let engine = HelmholtzHodgeFlowEngine::new(3);

    let ofis_asym = [15.0, -5.0, 2.0];
    let rets_asym = [0.005, 0.012, -0.008];
    let (mat_asym, n_asym) = HelmholtzHodgeFlowEngine::build_cross_microstructure_flow_matrix(&ofis_asym, &rets_asym)
        .expect("matriz asimétrica válida");
    let (res, _) = engine.decompose(&mat_asym, n_asym).expect("descomposición válida");

    assert!(res.total_energy > 0.0);
    assert!(res.curl_share >= 0.0 && res.curl_share <= 1.0);
    assert!(res.curl_share > 0.0, "El flujo cruzado asimétrico debe producir rotacional no nulo: got {}", res.curl_share);
}

#[test]
fn test_r7_r3_a1_curl_invariante_a_padding_de_nodos_mudos() {
    // R7-R3-A-1: 3 nodos activos con gradiente puro embebidos en una matriz de 30 activos (27 nodos mudos)
    // Demuestra que el potencial y la energía se normalizan por el número real de nodos no aislados r=3,
    // erradicando el suelo artificial espurio curl_share >= 1 - r/30 (que antes daba curl = 0.90).
    let engine = HelmholtzHodgeFlowEngine::new(30);
    let mut matrix = [[0.0_f64; MAX_HODGE_ASSETS]; MAX_HODGE_ASSETS];

    // Potenciales escalares en nodos 0, 1, 2: [12.0, 8.0, 4.0]
    // Flujo gradiente puro: F_{ij} = phi_i - phi_j
    matrix[0][1] = 4.0;
    matrix[1][0] = -4.0;
    matrix[0][2] = 8.0;
    matrix[2][0] = -8.0;
    matrix[1][2] = 4.0;
    matrix[2][1] = -4.0;

    let (res, potentials) = engine.decompose(&matrix, 30)
        .expect("debe descomponer con 3 nodos activos de 30");

    assert_eq!(res.num_nodes, 3, "El conteo de nodos debe ser 3 nodos activos, no 30");
    assert!(
        res.curl_share < 1e-12,
        "El curl de un gradiente puro con padding de nodos mudos debe ser exactamente 0.0, got {}",
        res.curl_share
    );
    assert!(
        (res.gradient_energy - res.total_energy).abs() < 1e-12,
        "La energía del gradiente debe ser exactamente igual a la total: grad={}, total={}",
        res.gradient_energy,
        res.total_energy
    );

    // Los nodos mudos (3..30) deben tener potencial exactamente 0.0
    for i in 3..30 {
        assert_eq!(potentials[i], 0.0, "Nodo mudo {} debe tener potencial 0.0", i);
    }
}
