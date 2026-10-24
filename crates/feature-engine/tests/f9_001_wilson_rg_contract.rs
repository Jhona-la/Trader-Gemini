//! # Contrato Formal F9-001: Matrices Aleatorias & Wilson Renormalization Group
//!
//! Verificación rigurosa de invariantes físicos y matemáticos:
//! C1: Eigensolver Jacobi exacto, descomposición espectral y ortonormalidad estricta.
//! C2: Bordes analíticos Marchenko-Pastur, filtrado Wishart y conservación de traza Tr(C) = N.
//! C3: Separación espectral de modo de mercado dominante vs bulk de ruido y SNR.
//! C4: Grupo de Renormalización de Wilson: decimation momentum-shell y clasificación de operadores.
//! C5: Exponente continuo de Lyapunov: estabilidad periódica vs caos determinista.
//! C6: Motor unificado multi-activo (N=16), robustez numérica y cero pánicos.

use feature_engine::random_matrix_wilson::{
    jacobi_eigensolver_symmetric, ContinuousLyapunovEstimator, MarchenkoPasturFilter,
    RandomMatrixWilsonEngine, WilsonOperatorClass, WilsonRGEngine, MAX_ASSETS,
};

#[test]
fn test_c1_jacobi_eigensolver_orthonormality_and_accuracy() {
    // Matriz simétrica 4x4 con valores analíticos conocidos
    // A = [[4, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]]
    let mut matrix = [[0.0; MAX_ASSETS]; MAX_ASSETS];
    let n = 4;
    matrix[0][0] = 4.0; matrix[0][1] = 1.0;
    matrix[1][0] = 1.0; matrix[1][1] = 4.0; matrix[1][2] = 1.0;
    matrix[2][1] = 1.0; matrix[2][2] = 4.0; matrix[2][3] = 1.0;
    matrix[3][2] = 1.0; matrix[3][3] = 4.0;

    let mut eigenvalues = [0.0; MAX_ASSETS];
    let mut eigenvectors = [[0.0; MAX_ASSETS]; MAX_ASSETS];

    let success = jacobi_eigensolver_symmetric(&matrix, n, &mut eigenvalues, &mut eigenvectors);
    assert!(success, "Jacobi eigensolver debe converger exitosamente");

    // Autovalores analíticos exactos de matriz tridiagonal Toeplitz:
    // lambda_k = 4 + 2 * cos(k * pi / 5) para k = 1, 2, 3, 4
    // lambda_1 = 4 + 2 * cos(pi/5) = 4 + 2 * 0.809016994 = 5.6180339887
    // lambda_2 = 4 + 2 * cos(2*pi/5) = 4 + 2 * 0.309016994 = 4.6180339887
    // lambda_3 = 4 + 2 * cos(3*pi/5) = 4 - 2 * 0.309016994 = 3.3819660113
    // lambda_4 = 4 + 2 * cos(4*pi/5) = 4 - 2 * 0.809016994 = 2.3819660113
    assert!((eigenvalues[0] - 5.6180339887).abs() < 1e-6, "Lambda 0 error: {}", eigenvalues[0]);
    assert!((eigenvalues[1] - 4.6180339887).abs() < 1e-6, "Lambda 1 error: {}", eigenvalues[1]);
    assert!((eigenvalues[2] - 3.3819660113).abs() < 1e-6, "Lambda 2 error: {}", eigenvalues[2]);
    assert!((eigenvalues[3] - 2.3819660113).abs() < 1e-6, "Lambda 3 error: {}", eigenvalues[3]);

    // Verificar ortonormalidad estricta: V^T * V = I
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for r in 0..n {
                dot += eigenvectors[r][i] * eigenvectors[r][j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-8, "Fallo de ortonormalidad en ({}, {}): {}", i, j, dot);
        }
    }

    // Verificar A * v_0 = lambda_0 * v_0
    for r in 0..n {
        let mut av = 0.0;
        for c in 0..n {
            av += matrix[r][c] * eigenvectors[c][0];
        }
        let lv = eigenvalues[0] * eigenvectors[r][0];
        assert!((av - lv).abs() < 1e-8, "A*v != lambda*v en fila {}: {} vs {}", r, av, lv);
    }
}

#[test]
fn test_c2_marchenko_pastur_bounds_and_trace_preservation() {
    let n = 8;
    let t = 128;
    let bounds = MarchenkoPasturFilter::compute_bounds(n, t, 1.0);

    // Con N=8, T=128: q = 8 / 128 = 0.0625, sqrt(q) = 0.25
    // lambda_- = 1.0 * (1 - 0.25)^2 = 0.75^2 = 0.5625
    // lambda_+ = 1.0 * (1 + 0.25)^2 = 1.25^2 = 1.5625
    assert!((bounds.lambda_minus - 0.5625).abs() < 1e-6, "Borde inferior MP incorrecto: {}", bounds.lambda_minus);
    assert!((bounds.lambda_plus - 1.5625).abs() < 1e-6, "Borde superior MP incorrecto: {}", bounds.lambda_plus);
    assert!(bounds.lambda_minus < bounds.lambda_plus, "lambda_- debe ser menor que lambda_+");

    // Construir una matriz de correlación simétrica con elementos fuera de diagonal
    let mut corr = [[0.0; MAX_ASSETS]; MAX_ASSETS];
    for i in 0..n {
        corr[i][i] = 1.0;
        for j in (i + 1)..n {
            let r = 0.15 * ((i + j) as f64).sin();
            corr[i][j] = r;
            corr[j][i] = r;
        }
    }

    let result = MarchenkoPasturFilter::denoise_correlation(&corr, n, t);

    // 1. Simetría exacta y diagonal idéntica a 1.0
    for i in 0..n {
        assert!((result.cleaned_matrix[i][i] - 1.0).abs() < 1e-10, "Diagonal limpia debe ser 1.0");
        for j in 0..n {
            assert!(
                (result.cleaned_matrix[i][j] - result.cleaned_matrix[j][i]).abs() < 1e-10,
                "Matriz debe ser simétrica"
            );
            assert!(
                result.cleaned_matrix[i][j] >= -1.0 && result.cleaned_matrix[i][j] <= 1.0,
                "Correlaciones en [-1, 1]"
            );
        }
    }

    // 2. Conservación de traza
    let mut raw_trace = 0.0;
    let mut clean_trace = 0.0;
    for i in 0..n {
        raw_trace += result.raw_eigenvalues[i];
        clean_trace += result.cleaned_eigenvalues[i];
    }
    assert!(
        (raw_trace - (n as f64)).abs() < 1e-6,
        "Traza original debe ser N: {}",
        raw_trace
    );
    assert!(
        (clean_trace - (n as f64)).abs() < 1e-6,
        "Traza filtrada debe preservar N: {}",
        clean_trace
    );

    // 3. Positividad estricta de autovalores limpios
    for i in 0..n {
        assert!(result.cleaned_eigenvalues[i] > 0.0, "Autovalor limpio debe ser positivo");
    }
}

#[test]
fn test_c3_market_mode_isolation_and_snr() {
    let n = 6;
    let t = 100;

    // Matriz con un factor común dominante de mercado: corr_ij = 0.75 para todo i != j
    let mut corr = [[0.0; MAX_ASSETS]; MAX_ASSETS];
    for i in 0..n {
        for j in 0..n {
            corr[i][j] = if i == j { 1.0 } else { 0.75 };
        }
    }

    let result = MarchenkoPasturFilter::denoise_correlation(&corr, n, t);

    // Para una matriz con 1s en la diagonal y rho fuera:
    // lambda_max = 1 + (N - 1) * rho = 1 + 5 * 0.75 = 4.75
    // Los otros 5 autovalores son 1 - rho = 0.25
    assert!(result.signal_modes_count >= 1, "Debe detectar al menos el modo de mercado");
    assert!(result.raw_eigenvalues[0] > 4.0, "Modo dominante debe superar 4.0: {}", result.raw_eigenvalues[0]);
    assert!(result.signal_to_noise_ratio > 0.60, "SNR debe superar el 60%: {}", result.signal_to_noise_ratio);

    // Los autovalores del bulk de ruido deben haber sido contraídos
    for i in 1..n {
        assert!(result.cleaned_eigenvalues[i] > 0.0, "Autovalores de ruido deben ser positivos");
    }
}

#[test]
fn test_c4_wilson_rg_coarse_graining_and_operator_classification() {
    // 1. Serie con persistencia de largo plazo (random walk con deriva)
    let mut persistent_series = [0.0; 256];
    let mut current = 100.0;
    for i in 0..256 {
        current += 0.05 + 0.01 * ((i as f64) * 0.1).sin();
        persistent_series[i] = current;
    }

    let report_persistent = WilsonRGEngine::compute_scale_flow(&persistent_series);
    // Para una tendencia persistente, el operador de acoplamiento es Relevante
    assert_eq!(
        report_persistent.operator_class,
        WilsonOperatorClass::Relevant,
        "Serie persistente debe clasificarse como Relevante"
    );
    assert!(report_persistent.g1_couplings[0] > 0.80, "Autocorrelación base debe ser masiva para tendencia: {}", report_persistent.g1_couplings[0]);
    assert!(report_persistent.effective_beta1.is_finite(), "Beta efectiva debe ser finita");

    // 2. Serie con oscilación rápida de alta frecuencia (ruido blanco alternante)
    let mut noisy_series = [0.0; 256];
    for i in 0..256 {
        noisy_series[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }

    let report_noisy = WilsonRGEngine::compute_scale_flow(&noisy_series);
    // Para oscilaciones de alta frecuencia, el acoplamiento colapsa a escalas macro (Irrelevante)
    assert_eq!(
        report_noisy.operator_class,
        WilsonOperatorClass::Irrelevant,
        "Serie de rebote rápido debe clasificarse como Irrelevante"
    );
    assert!(report_noisy.g1_couplings[0] < -0.50, "Autocorrelación base debe ser fuertemente negativa para rebote: {}", report_noisy.g1_couplings[0]);
    assert!(report_noisy.g1_couplings[1].abs() < 1e-6, "A escala 1 el rebote se aniquila por completo: {}", report_noisy.g1_couplings[1]);

    // 3. Finitud y límites en todas las 8 escalas
    for scale in 0..feature_engine::random_matrix_wilson::MAX_DYADIC_SCALES {
        assert!(report_persistent.g1_couplings[scale].is_finite());
        assert!(report_persistent.g2_couplings[scale].is_finite());
    }
}

#[test]
fn test_c5_continuous_lyapunov_stability_vs_chaos() {
    let mut estimator = ContinuousLyapunovEstimator::new();

    // 1. Serie periódica suave: onda senoidal pura (dinámica no caótica / disipativa)
    let mut last_diag = feature_engine::random_matrix_wilson::LyapunovDiagnostic::default();
    for i in 0..120 {
        let t = (i as f64) * 0.15;
        let v0 = t.sin();
        let v1 = (t - 0.15).sin();
        let v2 = (t - 0.30).sin();
        let v3 = (t - 0.45).sin();
        last_diag = estimator.update([v0, v1, v2, v3]);
    }

    // Para onda senoidal regular, el sistema no diverge exponencialmente
    assert!(!last_diag.is_chaotic, "Onda senoidal no debe ser caótica (lambda: {})", last_diag.lambda_max);
    assert!(last_diag.horizon_compression > 0.80, "Compresión de horizonte debe ser alta para ciclo regular");

    // 2. Serie caótica determinista: Mapa Logístico con r = 4.0 (caos total, lambda_max teórico = ln(2) ≈ 0.693)
    let mut chaotic_estimator = ContinuousLyapunovEstimator::new();
    let mut x = 0.314159265;
    let mut hist = [0.0; 4];
    let mut chaotic_diag = feature_engine::random_matrix_wilson::LyapunovDiagnostic::default();

    for _i in 0..150 {
        x = 4.0 * x * (1.0 - x);
        hist[3] = hist[2];
        hist[2] = hist[1];
        hist[1] = hist[0];
        hist[0] = x;
        chaotic_diag = chaotic_estimator.update(hist);
    }

    assert!(chaotic_diag.lambda_max > 0.0, "Mapa caótico debe tener exponente de Lyapunov positivo");
    assert!(chaotic_diag.horizon_compression < 0.95, "Régimen caótico debe comprimir el horizonte de retención");
}

#[test]
fn test_c6_unified_engine_multi_asset_robustness_and_zero_panic() {
    let n_assets = 16;
    let mut engine = RandomMatrixWilsonEngine::new(n_assets);

    // Alimentar 128 ticks para cada uno de los 16 activos con señales realistas y perturbaciones
    for tick in 0..128 {
        for asset in 0..n_assets {
            let phase = (tick as f64) * 0.05 + (asset as f64) * 0.30;
            let ret = (phase.sin() * 0.002) + 0.0005 * ((tick + asset) % 3) as f64;
            engine.push_return(asset, ret);
        }
    }

    let denoised_res = engine.update_correlation_denoising();
    assert!(denoised_res.is_some(), "Denoising debe generar un resultado tras 128 ticks");

    let res = denoised_res.unwrap();
    assert_eq!(res.bounds.spectral_ratio_q, (16.0 / 128.0));

    // Consultas a prueba de pánico para activos válidos e inválidos
    for i in 0..n_assets {
        for j in 0..n_assets {
            let r = engine.get_denoised_correlation(i, j);
            assert!(r.is_finite(), "Correlación debe ser finita");
            assert!(r >= -1.0 && r <= 1.0, "Correlación debe estar acotada");
            if i == j {
                assert!((r - 1.0).abs() < 1e-6, "Autocorrelación debe ser 1.0");
            }
        }

        let beta1 = engine.get_wilson_beta1(i);
        assert!(beta1.is_finite(), "Beta 1 de Wilson debe ser finita");

        let lyap = engine.get_lyapunov_diagnostic(i);
        assert!(lyap.lambda_max.is_finite(), "Lyapunov debe ser finito");
        assert!(lyap.horizon_compression > 0.0 && lyap.horizon_compression <= 1.0);
    }

    // Consulta fuera de rango no causa pánico y devuelve valores seguros por defecto
    let out_corr = engine.get_denoised_correlation(99, 99);
    assert_eq!(out_corr, 1.0);
    let out_beta = engine.get_wilson_beta1(100);
    assert_eq!(out_beta, 0.0);
    let out_lyap = engine.get_lyapunov_diagnostic(101);
    assert_eq!(out_lyap.lambda_max, 0.0);

    // Alimentar valores atípicos y subnormales para verificar inmunidad numérica
    engine.push_return(0, 1e-300);
    engine.push_return(1, -0.0);
    engine.push_return(2, f64::MIN_POSITIVE);
    let robust_res = engine.update_correlation_denoising();
    assert!(robust_res.is_some(), "Debe soportar entradas subnormales sin pánico");
}
