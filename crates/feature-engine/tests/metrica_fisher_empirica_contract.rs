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
    // Identidad de diseño: s = ∇_θ log p con p = f(u)/(C_q·σ),
    // f(u) = [1−(1−q)βu²]^{1/(1−q)}. El −ln σ de la normalización entra en
    // log p; C_q no depende de (μ, σ) y se omite. Verificación contra
    // diferencias centrales de log p (6+ decimales).
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
    // Para q > 1 el soporte es ℝ (ley de potencias): un z gigante tiene score
    // FINITO — observable legítimo, no falla. s_μ ~ 1/u → 0 y s_σ tiende a
    // (2/(q−1) − 1)/σ, que en q = 1.5 vale 3. La guarda base ≤ 0 queda como
    // defensa para q < 1 (fuera del rango admitido).
    let gigante = score_q_gaussiana(1.5, 0.0, 1.0, 1e9).expect("soporte infinito en q>1");
    assert!(gigante[0].abs() < 1e-6, "s_mu en z gigante: {}", gigante[0]);
    assert!(
        (gigante[1] - 3.0).abs() < 1e-6,
        "s_sigma en z gigante: {} (limite 3)",
        gigante[1]
    );
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

/// ν = (3−q)/(q−1): la q-gaussiana con β_q es una t de Student con ν grados de
/// libertad reescalada a varianza 1.
fn nu(q: f64) -> f64 {
    (3.0 - q) / (q - 1.0)
}

#[test]
fn scores_de_media_cero_bajo_la_propia_familia() {
    // Identidad de regularidad: E_p[s] = 0. Sin el −1/σ de la normalización,
    // E[s_σ] vale 1/σ (E[2βu²/base] = 1), y el producto externo deja de ser
    // la métrica de Fisher. Error estándar de cada media con N = 8192:
    // √(a/N) y √(b/N), ≤ 0.016; la tolerancia 0.08 son ≈ 5 errores.
    let n = 8192;
    for (q, semilla) in [(1.2, 0xA11CE_u64), (1.5, 0xB0B_u64)] {
        let muestras = muestras_q_gaussiana(q, 0.0, 1.0, n, semilla);
        let mut suma = [0.0f64; 2];
        for &z in &muestras {
            let s = score_q_gaussiana(q, 0.0, 1.0, z).unwrap();
            suma[0] += s[0];
            suma[1] += s[1];
        }
        let media = [suma[0] / n as f64, suma[1] / n as f64];
        assert!(media[0].abs() < 0.08, "q={q}: E[s_mu]={}", media[0]);
        assert!(media[1].abs() < 0.08, "q={q}: E[s_sigma]={}", media[1]);
    }
}

#[test]
fn metrica_empirica_contra_la_forma_cerrada_de_student() {
    // G = diag(a, b) con a = ν(ν+1)/((ν+3)(ν−2)) y b = 2ν/(ν+3) (t de Student
    // a varianza 1; el límite gaussiano es diag(1, 2)). g = G/σ²: con σ = 2 la
    // métrica es un cuarto de la de σ = 1. Errores estándar con N = 8192:
    // a ≤ 0.016, c ≤ 0.021, b ≤ 0.038 (q = 1.2); las tolerancias son
    // ≈ 5 errores de cada entrada.
    let n = 8192;
    let sigma = 2.0;
    for (q, semilla) in [(1.2, 0xC1_u64), (1.5, 0xC2_u64)] {
        let v = nu(q);
        let a = v * (v + 1.0) / ((v + 3.0) * (v - 2.0));
        let b = 2.0 * v / (v + 3.0);
        let muestras = muestras_q_gaussiana(q, 0.5, sigma, n, semilla);
        let g = metrica_fisher_empirica(q, 0.5, sigma, &muestras).unwrap();
        let s2 = sigma * sigma;
        assert!(
            (g[0][0] * s2 - a).abs() < 0.08,
            "q={q}: a={} vs {a}",
            g[0][0] * s2
        );
        assert!(
            (g[1][1] * s2 - b).abs() < 0.20,
            "q={q}: b={} vs {b}",
            g[1][1] * s2
        );
        assert!((g[0][1] * s2).abs() < 0.10, "q={q}: c={}", g[0][1] * s2);
    }
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

    assert!(
        r_15.abs() > r_casi.abs(),
        "colas mas pesadas aumentan curvatura hiperbolica: |R(q=1.5)|={} debe ser > |R(q=1.001)|={}",
        r_15,
        r_casi
    );
}

#[test]
fn ricci_continuidad_gaussiana_y_colas_curvan() {
    // Forma cerrada: R = −(ν+3)/ν. q→1⁺ ⇒ R → −1 (continuidad con Fisher-Rao
    // del semiplano, 191/193); q = 1.2 ⇒ R = −4/3; q = 1.5 ⇒ R = −2. El
    // estimador da R̂ = −2â/(âb̂ − ĉ²) con la G empírica; su error estándar
    // con N = 8192 es ≈ 0.03–0.04 y la tolerancia 0.20 son ≈ 5 errores.
    let n = 8192;
    let mut r = [0.0f64; 3];
    for (k, (q, semilla)) in [
        (1.001, 0x5EED_0001_u64),
        (1.2, 0x5EED_0004_u64),
        (1.5, 0x5EED_0002_u64),
    ]
    .into_iter()
    .enumerate()
    {
        let esperado = -(nu(q) + 3.0) / nu(q);
        let muestras = muestras_q_gaussiana(q, 0.0, 1.0, n, semilla);
        r[k] = ricci_q_gaussiana(q, 0.0, 1.0, &muestras).expect("Ricci con beta_q");
        assert!(
            (r[k] - esperado).abs() < 0.20,
            "q={q}: esperado R = {esperado}, obtenido {}",
            r[k]
        );
    }

    // Las colas pesadas CURVAN la geometría: R(1.5) < R(1.2) < R(1.001). Los
    // saltos (≈ 0.33 y 0.67) son de 9 y 19 errores estándar.
    assert!(
        r[2] < r[1] && r[1] < r[0],
        "colas mas pesadas curvan: R(1.001)={}, R(1.2)={}, R(1.5)={}",
        r[0],
        r[1],
        r[2]
    );

    // Verifica la estandarización y el reescalado por σ² del estimador: en
    // otros (μ, σ), incluidas σ muy pequeña y muy grande, el mismo q da el
    // mismo R. (El paso h del estimador es absoluto: derivar en el punto
    // muestreado daría números sin sentido con σ = 0.005.)
    for (mu, sigma, semilla) in [
        (-3.0, 0.25, 0x5EED_0003_u64),
        (0.0, 0.005, 0x5EED_0005_u64),
        (7.0, 1e5, 0x5EED_0006_u64),
    ] {
        let muestras = muestras_q_gaussiana(1.5, mu, sigma, n, semilla);
        let r_otro = ricci_q_gaussiana(1.5, mu, sigma, &muestras).expect("Ricci en otro punto");
        assert!(
            (r_otro - (-2.0)).abs() < 0.20,
            "q=1.5 en ({mu}, {sigma}): {r_otro}"
        );
    }
}
