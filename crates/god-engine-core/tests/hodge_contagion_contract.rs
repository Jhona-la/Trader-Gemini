//! INTEGRACIÓN Hodge × cadena Hawkes (Ola XLVI·C, T05).
//!
//! Verifica el contrato de DATOS: la matriz de contagio que produce el
//! kernel Hawkes real (feature-engine) alimenta la descomposición de Hodge
//! (risk-engine) y el veredicto estructural es el correcto —
//!   cascada transitiva  ⇒ curl BAJO (existe jerarquía de liderazgo),
//!   ciclo A→B→C→A     ⇒ curl ALTO (cámara de eco, sin potencial).
//!
//! Sin arena ni motor: sólo la tubería matemática que el publicador XLV·H
//! cablea (publish_contagion_roles → hawkes_contagion_curl_share).

use feature_engine::hawkes_cross::contagion_matrix;
use feature_engine::hodge_flow::{HelmholtzHodgeFlowEngine, MAX_HODGE_ASSETS};
use risk_engine::hodge::hodge_curl_share;

/// Cascada: un líder excita a dos seguidores con retardos distintos.
/// El flujo es mayormente gradiente (potencial de liderazgo existe):
/// curl_share debe quedar por debajo de 0.5.
#[test]
fn xlvic_cascada_hawkes_real_da_curl_bajo() {
    let leader: Vec<u64> = (0..200).map(|i| 1_000 + (i as u64) * 2_000).collect();
    let s1: Vec<u64> = leader.iter().map(|&t| t + 150).collect();
    let s2: Vec<u64> = leader.iter().map(|&t| t + 300).collect();
    let series = vec![leader, s1, s2];
    let spans: Vec<u64> = series.iter().map(|s| s.last().unwrap() - s[0]).collect();
    let lags = vec![200u64, 500, 1_000];

    let matrix = contagion_matrix(&series, &spans, &lags).expect("matriz de cascada");
    let curl = hodge_curl_share(&matrix).expect("hodge sobre cascada");
    assert!(
        curl < 0.5,
        "cascada transitiva: curl={curl} — la jerarquía debería dominar"
    );
}

/// Ciclo: A excita B (+300ms), B excita C (+300ms), C excita a la SIGUIENTE
/// A (+600ms) — la ronda dura 1200ms y TODOS los brazos caen dentro de la
/// rejilla de lags [300,700,1200]. El armón C→A es el que rompe cualquier
/// potencial (φ_A > φ_B > φ_C > φ_A es imposible): curl > 0.5. Con la
/// rejilla anterior el brazo C→A era invisible (1700ms) y el campo medido
/// era una cadena transitiva — lección: el ciclo sólo existe para Hodge si
/// el kernel lo VE cerrarse.
#[test]
fn xlvic_ciclo_hawkes_real_da_curl_alto() {
    let mut a: Vec<u64> = Vec::new();
    let mut b: Vec<u64> = Vec::new();
    let mut c: Vec<u64> = Vec::new();
    let mut t = 1_000u64;
    for _ in 0..150 {
        a.push(t);
        b.push(t + 300);
        c.push(t + 600);
        t += 1_200; // la siguiente A llega 600ms después de C
    }
    let series = vec![a, b, c];
    let spans: Vec<u64> = series.iter().map(|s| s.last().unwrap() - s[0]).collect();
    let lags = vec![300u64, 700, 1_200];

    let matrix = contagion_matrix(&series, &spans, &lags).expect("matriz de ciclo");
    let curl = hodge_curl_share(&matrix).expect("hodge sobre ciclo");
    assert!(
        curl > 0.5,
        "ciclo cerrado: curl={curl} — la cámara de eco debería dominar"
    );
}

/// El mismo campo visto dos veces da el mismo veredicto (determinismo de la
/// tubería completa: kernel determinista + proyección determinista).
#[test]
fn xlvic_tuberia_hawkes_hodge_determinista() {
    let leader: Vec<u64> = (0..150).map(|i| 1_000 + (i as u64) * 2_000).collect();
    let s1: Vec<u64> = leader.iter().map(|&t| t + 150).collect();
    let s2: Vec<u64> = leader.iter().map(|&t| t + 250).collect();
    let series = vec![leader, s1, s2];
    let spans: Vec<u64> = series.iter().map(|s| s.last().unwrap() - s[0]).collect();

    let m1 = contagion_matrix(&series, &spans, &[200, 500, 1_000]).unwrap();
    let m2 = contagion_matrix(&series, &spans, &[200, 500, 1_000]).unwrap();
    let h1 = hodge_curl_share(&m1).unwrap();
    let h2 = hodge_curl_share(&m2).unwrap();
    assert_eq!(h1.to_bits(), h2.to_bits(), "misma entrada, otro bit");
}

/// R6-A11 (Ola Ω51): Paridad formal unificada entre risk_engine::hodge y feature_engine::hodge_flow.
/// Verifica que sobre cualquier matriz de flujo (cascada transitiva, 3-ciclo puro, y matriz Hawkes real),
/// ambas implementaciones devuelven idéntico curl_share con precisión de máquina (< 1e-12).
#[test]
fn r6_a11_hodge_paridad_unificada_risk_vs_feature() {
    let engine = HelmholtzHodgeFlowEngine::new(4);

    // 1. Cascada transitiva
    let phi = [3.0_f64, 2.0, 1.0, 0.0];
    let n = 4;
    let mut matrix_vec = vec![vec![0.0; n]; n];
    let mut matrix_arr = [[0.0_f64; MAX_HODGE_ASSETS]; MAX_HODGE_ASSETS];
    for i in 0..n {
        for j in 0..n {
            let v = phi[i] - phi[j];
            matrix_vec[i][j] = v;
            matrix_arr[i][j] = v;
        }
    }
    let curl_risk = hodge_curl_share(&matrix_vec).expect("risk curl cascada");
    let (res_feature, _) = engine.decompose(&matrix_arr, n).expect("feature curl cascada");
    assert!(
        (curl_risk - res_feature.curl_share).abs() < 1e-12,
        "Paridad falló en cascada: risk={}, feature={}",
        curl_risk,
        res_feature.curl_share
    );

    // 2. 3-ciclo puro con rotacional
    let (vortex_mat, n_vortex) = HelmholtzHodgeFlowEngine::build_pure_vortex_matrix(2.5);
    let mut vortex_vec = vec![vec![0.0; n_vortex]; n_vortex];
    for i in 0..n_vortex {
        for j in 0..n_vortex {
            vortex_vec[i][j] = vortex_mat[i][j];
        }
    }
    let curl_vortex_risk = hodge_curl_share(&vortex_vec).expect("risk curl vórtice");
    let (res_vortex_feat, _) = engine.decompose(&vortex_mat, n_vortex).expect("feature curl vórtice");
    assert!(
        (curl_vortex_risk - res_vortex_feat.curl_share).abs() < 1e-12,
        "Paridad falló en vórtice: risk={}, feature={}",
        curl_vortex_risk,
        res_vortex_feat.curl_share
    );

    // 3. Matriz Hawkes real de 3 series
    let mut a: Vec<u64> = Vec::new();
    let mut b: Vec<u64> = Vec::new();
    let mut c: Vec<u64> = Vec::new();
    let mut t = 1_000u64;
    for _ in 0..100 {
        a.push(t);
        b.push(t + 200);
        c.push(t + 400);
        t += 1_000;
    }
    let series = vec![a, b, c];
    let spans: Vec<u64> = series.iter().map(|s| s.last().unwrap() - s[0]).collect();
    let lags = vec![200u64, 500, 1_000];
    let hawkes_matrix = contagion_matrix(&series, &spans, &lags).expect("hawkes matrix");
    let n_h = hawkes_matrix.len();
    let mut hawkes_arr = [[0.0_f64; MAX_HODGE_ASSETS]; MAX_HODGE_ASSETS];
    for i in 0..n_h {
        for j in 0..n_h {
            hawkes_arr[i][j] = hawkes_matrix[i][j];
        }
    }
    let curl_hawkes_risk = hodge_curl_share(&hawkes_matrix).expect("risk hawkes");
    let (res_hawkes_feat, _) = engine.decompose(&hawkes_arr, n_h).expect("feature hawkes");
    assert!(
        (curl_hawkes_risk - res_hawkes_feat.curl_share).abs() < 1e-12,
        "Paridad falló en Hawkes real: risk={}, feature={}",
        curl_hawkes_risk,
        res_hawkes_feat.curl_share
    );
}
