//! Contratos del núcleo `metrica_fisher_empirica` (GLM-221, Fase C1;
//! diseño cerrado en el ciclo 195). Determinista: muestras por rechazo con
//! LCG propio — nada aleatorio del sistema, nada flaky.

use feature_engine::metrica_fisher_empirica::{
    M_MIN_MUESTRAS, beta_q, metrica_fisher_empirica, ricci_q_gaussiana,
    score_q_gaussiana,
};

/// LCG determinista (64 bits, multiplicador de Knuth) → f64 en [0, 1).
fn lcg(estado: &mut u64) -> f64 {
    *estado = estado
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*estado >> 11) as f64) / (1u64 << 53) as f64
}

/// Muestras u de la q-gaussiana estandarizada por RECHAZO determinista.
/// Para q > 1 el soporte es INFINITO (ley de potencias: base = 1+(q−1)βu²):
/// se propone uniforme en [−L, L] con L tal que dens(L) = 1e−5·dens(0) —
/// colas razonablemente representadas con N grande (el espejo python de
/// GLM-221 destapó que el soporte compacto asumido en el diseño 195 sólo
/// existe para q < 1, fuera del rango de la familia).
fn muestras_q_gaussiana(q: f64, mu: f64, sigma: f64, n: usize, semilla: u64) -> Vec<f64> {
    let beta = beta_q(q).expect("q valido en el generador del contrato");
    let expo = 1.0 / (q - 1.0); // dens(u) = (1+(q-1)βu²)^{-1/(q-1)}
    let limite = ((10.0f64.powf(5.0 / expo) - 1.0) / ((q - 1.0) * beta)).sqrt();
    let mut estado = semilla;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u = (2.0 * lcg(&mut estado) - 1.0) * limite;
        let dens = (1.0 + (q - 1.0) * beta * u * u).powf(-expo);
        if lcg(&mut estado) < dens {
            out.push(mu + sigma * u);
        }
    }
    out
}

#[test]
fn beta_q_rango_y_normalizacion() {
    // Fuera del rango abierto (1, 5/3): fall-closed.
    assert!(beta_q(1.0).is_none(), "q=1 es el limite gaussiano, no de la familia");
    assert!(beta_q(5.0 / 3.0).is_none(), "q=5/3 diverge la varianza");
    assert!(beta_q(1.8).is_none());
    assert!(beta_q(0.9).is_none());
    assert!(beta_q(f64::NAN).is_none());
    // Dentro: valores exactos del diseño.
    assert!((beta_q(1.5).unwrap() - 1.0 / 0.5).abs() < 1e-12);
    assert!((beta_q(1.2).unwrap() - 1.0 / 1.4).abs() < 1e-12);
}

#[test]
fn scores_verificados_contra_diferencias_finitas() {
    // Identidad de diseño: s = ∇_θ log p con p ∝ [1−(1−q)βu²]^{1/(1−q)}.
    // Verificación contra diferencias centrales de log p (6+ decimales).
    fn log_p(q: f64, mu: f64, sigma: f64, z: f64) -> f64 {
        let beta = beta_q(q).unwrap();
        let u = (z - mu) / sigma;
        (1.0 - (1.0 - q) * beta * u * u).ln() / (1.0 - q)
    }
    for (q, mu, sigma, z) in [
        (1.2, 0.3, 1.7, 0.9),
        (1.5, -0.4, 2.3, 1.1),
        (1.55, 0.0, 0.8, -0.6),
    ] {
        let [s0, s1] = score_q_gaussiana(q, mu, sigma, z).unwrap();
        let h = 1e-6;
        let fd_mu = (log_p(q, mu + h, sigma, z) - log_p(q, mu - h, sigma, z)) / (2.0 * h);
        let fd_sigma = (log_p(q, mu, sigma + h, z) - log_p(q, mu, sigma - h, z)) / (2.0 * h);
        assert!(
            (s0 - fd_mu).abs() < 1e-6,
            "s_mu q={q}: analitico {s0} vs fd {fd_mu}"
        );
        assert!(
            (s1 - fd_sigma).abs() < 1e-6,
            "s_sigma q={q}: analitico {s1} vs fd {fd_sigma}"
        );
    }
}

#[test]
fn fall_closed_de_metrica_y_scores() {
    // σ ≤ 0, no finitos, q fuera de rango.
    assert!(score_q_gaussiana(1.3, 0.0, 0.0, 0.5).is_none());
    assert!(score_q_gaussiana(1.3, 0.0, -1.0, 0.5).is_none());
    assert!(score_q_gaussiana(1.7, 0.0, 1.0, 0.5).is_none());
    assert!(score_q_gaussiana(f64::NAN, 0.0, 1.0, 0.5).is_none());
    // Para q > 1 el soporte es ℝ (ley de potencias): un z gigante tiene score
    // FINITO y pequeño (~1/u) — observable legítimo, no falla. La guarda
    // base ≤ 0 queda como defensa para q < 1 (fuera del rango admitido).
    let gigante = score_q_gaussiana(1.5, 0.0, 1.0, 1e9).expect("soporte infinito en q>1");
    assert!(gigante[0].abs() < 1e-6 && gigante[1].abs() < 1e-6);
    // Métrica: menos muestras que el mínimo → None (nada fabricado).
    let pocas = muestras_q_gaussiana(1.5, 0.0, 1.0, M_MIN_MUESTRAS - 1, 0xC0FFE);
    assert!(metrica_fisher_empirica(1.5, 0.0, 1.0, &pocas).is_none());
    // Ricci: q fuera de rango → Err.
    let muestras = muestras_q_gaussiana(1.5, 0.0, 1.0, M_MIN_MUESTRAS, 0xBADA55);
    assert!(ricci_q_gaussiana(1.8, 0.0, 1.0, &muestras).is_err());
    assert!(ricci_q_gaussiana(1.0, 0.0, 1.0, &muestras).is_err());
}

#[test]
fn ricci_continuidad_gaussiana_y_colas_planas() {
    // Los dos números del diseño (194/195, convención del estimador 193 —
    // el espejo python de GLM-221 midió las magnitudes con este muestreo:
    // |R|≈0.87 en q=1.001 y ≈0.48 en q=1.5, sesgo del truncado de colas):
    //   q→1⁺  ⇒ R ≈ −1 (continuidad con Fisher-Rao del semiplano, 191/193)
    //   q=1.5 ⇒ R ≈ −0.40 (las colas pesadas APLANAN la geometría)
    // Tolerancias honestas para métrica EMPÍRICA: ruido 1/√N (N=8192)
    // amplificado por las derivadas del estimador + sesgo del dominio L.
    let n = 8192;
    let muestras_casi = muestras_q_gaussiana(1.001, 0.0, 1.0, n, 0x5EED_0001);
    let r_casi = ricci_q_gaussiana(1.001, 0.0, 1.0, &muestras_casi)
        .expect("Ricci continuo en q->1 con beta_q");
    assert!(
        (r_casi - (-1.0)).abs() < 0.25,
        "q=1.001: esperado R ~ -1, obtenido {r_casi}"
    );

    let muestras_15 = muestras_q_gaussiana(1.5, 0.0, 1.0, n, 0x5EED_0002);
    let r_15 =
        ricci_q_gaussiana(1.5, 0.0, 1.0, &muestras_15).expect("Ricci en q=1.5 con beta_q");
    assert!(
        (r_15 - (-0.45)).abs() < 0.25,
        "q=1.5: esperado R ~ -0.40/-0.48, obtenido {r_15}"
    );

    // La tendencia es el hallazgo 194: q mayor (colas más pesadas) APLANA
    // la geometría — |R| decrece.
    assert!(
        r_15.abs() < r_casi.abs(),
        "colas mas pesadas aplanan: |R(q=1.5)|={} debe ser < |R(q=1.001)|={}",
        r_15,
        r_casi
    );
}
