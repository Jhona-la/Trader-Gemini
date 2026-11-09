//! Contratos del núcleo `metrica_fisher_empirica` (GLM-221, Fase C1;
//! diseño cerrado en el ciclo 195). Determinista: muestras por rechazo con
//! LCG propio — nada aleatorio del sistema, nada flaky.

use feature_engine::metrica_fisher_empirica::{
    beta_q, metrica_fisher_empirica, ricci_q_gaussiana, score_q_gaussiana, M_MIN_MUESTRAS,
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
    assert!(
        beta_q(1.0).is_none(),
        "q=1 es el limite gaussiano, no de la familia"
    );
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
        (1.0 - (1.0 - q) * beta * u * u).ln() / (1.0 - q) - sigma.ln()
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
fn score_en_limite_gaussiano_normalizado() {
    let q = 1.000001;
    let mu = 0.3;
    let sigma = 1.7;
    let z = 0.9;
    let u = (z - mu) / sigma;
    let [s_mu, s_sigma] = score_q_gaussiana(q, mu, sigma, z).unwrap();

    assert!((s_mu - u / sigma).abs() < 0.005, "s_mu={s_mu}");
    assert!(
        (s_sigma - (u * u - 1.0) / sigma).abs() < 0.005,
        "score sigma sin normalizar: {s_sigma}"
    );
}

#[test]
fn metrica_en_limite_gaussiano_normalizado() {
    let sigma = 1.0;
    let muestras = muestras_q_gaussiana(1.001, 0.0, sigma, 8192, 0x5EED_0011);
    let g = metrica_fisher_empirica(1.001, 0.0, sigma, &muestras).unwrap();

    assert!((g[0][0] - 1.0 / (sigma * sigma)).abs() < 0.20);
    assert!(
        (g[1][1] - 2.0 / (sigma * sigma)).abs() < 0.30,
        "g_sigma_sigma debe tender a 2/sigma^2, no 3/sigma^2: {}",
        g[1][1]
    );
    assert!(g[0][1].abs() < 0.20);
    assert!(g[1][0].abs() < 0.20);
}

#[test]
fn fall_closed_de_metrica_y_scores() {
    // σ ≤ 0, no finitos, q fuera de rango.
    assert!(score_q_gaussiana(1.3, 0.0, 0.0, 0.5).is_none());
    assert!(score_q_gaussiana(1.3, 0.0, -1.0, 0.5).is_none());
    assert!(score_q_gaussiana(1.7, 0.0, 1.0, 0.5).is_none());
    assert!(score_q_gaussiana(f64::NAN, 0.0, 1.0, 0.5).is_none());
    // Para q > 1 el soporte es ℝ. Con q=1.5 y |u|→∞, s_mu→0 pero
    // s_sigma→2/(q−1)−1=3: no se puede esperar un score nulo en sigma.
    let gigante = score_q_gaussiana(1.5, 0.0, 1.0, 1e9).expect("soporte infinito en q>1");
    assert!(gigante[0].abs() < 1e-6);
    assert!((gigante[1] - 3.0).abs() < 1e-6);
    for invalido in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(score_q_gaussiana(1.5, 0.0, 1.0, invalido).is_none());
        assert!(score_q_gaussiana(1.5, invalido, 1.0, 0.0).is_none());
        assert!(score_q_gaussiana(1.5, 0.0, invalido, 0.0).is_none());
    }
    // Métrica: menos muestras que el mínimo → None (nada fabricado).
    let pocas = muestras_q_gaussiana(1.5, 0.0, 1.0, M_MIN_MUESTRAS - 1, 0xC0FFE);
    assert!(metrica_fisher_empirica(1.5, 0.0, 1.0, &pocas).is_none());
    // Ricci: q fuera de rango → Err.
    let muestras = muestras_q_gaussiana(1.5, 0.0, 1.0, M_MIN_MUESTRAS, 0xBADA55);
    assert!(ricci_q_gaussiana(1.8, 0.0, 1.0, &muestras).is_err());
    assert!(ricci_q_gaussiana(1.0, 0.0, 1.0, &muestras).is_err());
}

#[test]
fn ricci_continuidad_gaussiana_y_colas_normalizadas() {
    let n = 8192;
    let muestras_casi = muestras_q_gaussiana(1.001, 0.0, 1.0, n, 0x5EED_0001);
    let r_casi = ricci_q_gaussiana(1.001, 0.0, 1.0, &muestras_casi)
        .expect("Ricci continuo en q->1 con beta_q");
    assert!(
        (r_casi - (-1.0)).abs() < 0.25,
        "q=1.001: esperado R ~ -1, obtenido {r_casi}"
    );

    // Con q=1.5 y beta=2, s_sigma=(3u²−1)/(σ(1+u²)). La densidad
    // proporcional a (1+u²)^-2 da E[(3u²−1)²/(1+u²)²]=1 exactamente:
    // el numerador es 9J2−6J1+J0=π/2 y el normalizador π/2. Por tanto
    // R=-2/B=-2; -0.45 correspondía a la densidad sin normalizar.
    let muestras_15 = muestras_q_gaussiana(1.5, 0.0, 1.0, n, 0x5EED_0002);
    let r_15 = ricci_q_gaussiana(1.5, 0.0, 1.0, &muestras_15).expect("Ricci en q=1.5 con beta_q");
    assert!(
        (r_15 - (-2.0)).abs() < 0.25,
        "q=1.5: esperado R=-2 por la métrica normalizada, obtenido {r_15}"
    );
}
