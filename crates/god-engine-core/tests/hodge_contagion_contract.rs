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
