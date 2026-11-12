use risk_engine::hodge::hodge_curl_share;

#[test]
fn hodge_pure_vortex_has_maximal_curl_share() {
    // 3-ciclo puro: 0 -> 1 -> 2 -> 0 con intensidad 5.0
    let f = vec![
        vec![0.0, 5.0, -5.0],
        vec![-5.0, 0.0, 5.0],
        vec![5.0, -5.0, 0.0],
    ];
    let curl = hodge_curl_share(&f).expect("flujo válido de 3 nodos");
    assert!(
        (curl - 1.0).abs() < 1e-12,
        "3-ciclo puro debe tener curl_share = 1.0 exacto, obtenido {curl}"
    );
}

#[test]
fn hodge_pure_transitive_cascade_has_zero_curl_share() {
    // Cascada transitiva pura: potencial phi = [3.0, 1.0, -2.0]
    let phi = [3.0, 1.0, -2.0];
    let n = phi.len();
    let mut f = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            f[i][j] = phi[i] - phi[j];
        }
    }
    let curl = hodge_curl_share(&f).expect("cascada válida");
    assert!(
        curl < 1e-12,
        "Cascada transitiva pura debe tener curl_share ≈ 0.0, obtenido {curl}"
    );
}

#[test]
fn hodge_scale_invariance_under_positive_scalar() {
    // El flujo escalado por lambda > 0 debe tener idéntico curl_share
    let mut f = vec![vec![0.0; 4]; 4];
    f[0][1] = 2.0; f[1][0] = -2.0;
    f[1][2] = 3.0; f[2][1] = -3.0;
    f[2][0] = 1.0; f[0][2] = -1.0;
    f[2][3] = 4.0; f[3][2] = -4.0;

    let base_curl = hodge_curl_share(&f).expect("flujo base");

    for &lambda in &[0.01, 0.5, 2.0, 100.0, 10_000.0] {
        let mut f_scaled = vec![vec![0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                f_scaled[i][j] = f[i][j] * lambda;
            }
        }
        let scaled_curl = hodge_curl_share(&f_scaled).expect("flujo escalado");
        assert!(
            (scaled_curl - base_curl).abs() < 1e-12,
            "Hodge curl_share debe ser estrictamente invariante a la escala lambda: base={base_curl}, scaled={scaled_curl}"
        );
    }
}

#[test]
fn hodge_permutation_invariance() {
    // Si permutamos el orden de los activos, el curl_share global debe ser idéntico
    let f = vec![
        vec![0.0, 2.0, -1.0, 0.5],
        vec![-2.0, 0.0, 3.0, -0.5],
        vec![1.0, -3.0, 0.0, 1.5],
        vec![-0.5, 0.5, -1.5, 0.0],
    ];
    let base_curl = hodge_curl_share(&f).expect("flujo original");

    // Permutación sigma: [3, 0, 2, 1]
    let perm = [3, 0, 2, 1];
    let mut f_perm = vec![vec![0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            f_perm[i][j] = f[perm[i]][perm[j]];
        }
    }
    let perm_curl = hodge_curl_share(&f_perm).expect("flujo permutado");
    assert!(
        (perm_curl - base_curl).abs() < 1e-12,
        "Hodge curl_share debe ser invariante bajo permutación de etiquetas: base={base_curl}, perm={perm_curl}"
    );
}

#[test]
fn hodge_stack_buffer_capacity_up_to_64_nodes() {
    // Verificar que matrices de hasta 64 nodos operan sin desbordamiento ni pánico
    for n in [3, 8, 16, 32, 64] {
        let mut f = vec![vec![0.0; n]; n];
        for i in 0..n {
            let next = (i + 1) % n;
            f[i][next] = 1.0;
            f[next][i] = -1.0;
        }
        let curl = hodge_curl_share(&f);
        assert!(curl.is_some(), "Debe soportar n={n} en buffer de stack");
        let val = curl.unwrap();
        assert!(val.is_finite() && (0.0..=1.0).contains(&val));
    }
}

#[test]
fn hodge_fail_closed_on_nan_and_inf() {
    let mut f = vec![
        vec![0.0, 1.0, -1.0],
        vec![-1.0, 0.0, 1.0],
        vec![1.0, -1.0, 0.0],
    ];
    f[0][1] = f64::NAN;
    assert_eq!(hodge_curl_share(&f), None);

    f[0][1] = f64::INFINITY;
    assert_eq!(hodge_curl_share(&f), None);

    f[0][1] = f64::NEG_INFINITY;
    assert_eq!(hodge_curl_share(&f), None);
}
