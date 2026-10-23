//! Contratos DMRG 2-sitios (GLM 152→154) con SOLUCIÓN CONOCIDA — los
//! límites analíticos del Ising transverso. El optimizador debe recuperarlos
//! o declarar ausencia; jamás inventar.
//!
//! GLM 154: los 6 contratos ACTIVOS. Cadena de fixes de la línea:
//! 153 = MPO campo multiplicativo (∏(I−hσˣ)), seed del entorno derecho en
//! wd=0, último sitio con SVD roto; 154 = MPO estrictamente 2-cuerpos (el
//! canal pendiente transportaba σᶻ y generaba cadenas σᶻ₀⋯σᶻ_k de muchos
//! cuerpos — invisible a los límites con J=1 por coincidencia espectral
//! E_min = −J²(L−1) = −J(L−1)) y optimizador de PARES con re-emisión
//! U/σVᵀ: el 1-sitio tenía D' ≤ min(dl·d, dr) ≤ dr y JAMÁS crecía
//! enlaces desde el estado producto, dejando los contratos D>1
//! matemáticamente inalcanzables.
use feature_engine::mps::{dmrg_ising, entrelazamiento_corte, mps_desde_estado};

/// Energía exacta del Ising transverso L sitios (obc) por diagonalización
/// EXHAUSTIVA del Hamiltoniano 2^L×2^L (L ≤ 10: 1024² — la referencia JW
/// sería frágil de reimplementar; la diagonalización directa con un Jacobi
/// cíclico local es autocontenida y exacta para certificar el principio
/// variacional E_DMRG ≥ E₀).
fn energia_exacta_ising(l: usize, j: f64, h: f64) -> f64 {
    let dim = 1usize << l;
    let spin_z = |s: usize, i: usize| -> f64 {
        if (s >> i) & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    };
    let flip = |s: usize, i: usize| -> usize { s ^ (1 << i) };
    // H[s][s'] construido explícito y mínimo autovalor por Jacobi cíclico
    // denso LOCAL (exacto y determinista para 2^L ≤ 1024; el del módulo es
    // privado). El método de potencias del 152 nunca se ejercitó y estaba
    // roto (E₀ = +5.82 para L=8 crítico).
    let mut hmat = vec![0.0f64; dim * dim];
    for s in 0..dim {
        let mut diag = 0.0;
        for i in 0..l - 1 {
            diag += -j * spin_z(s, i) * spin_z(s, i + 1);
        }
        hmat[s * dim + s] = diag;
        for i in 0..l {
            let t = flip(s, i);
            hmat[t * dim + s] += -h; // σˣ = suma sobre columnas (simétrico)
        }
    }
    // Mínimo autovalor por iteración de Rayleigh sobre vectores aleatorios
    // deterministas con ortogonalización (suficiente para separar E₀ de la
    // banda con 2^L ≤ 1024 y gap finito).
    min_autovalor(&hmat, dim)
}

fn min_autovalor(a: &[f64], n: usize) -> f64 {
    // GLM 154: el método de potencias con deflación original del 152 nunca
    // se ejercitó (los 6 contratos estaban OPEN) y devolvía E₀ = +5.82 para
    // L=8 crítico — basura. Jacobi cíclico denso local: exacto y
    // determinista para 2^L ≤ 1024 (el del módulo es privado del crate).
    let mut m = a.to_vec();
    const MAX_SWEEPS: usize = 96;
    for _ in 0..=MAX_SWEEPS {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[i * n + j] * m[i * n + j];
            }
        }
        if off <= 1e-18 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Semejanza completa (filas + columnas) — la lección del 151.
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
            }
        }
    }
    (0..n).map(|i| m[i * n + i]).fold(f64::INFINITY, f64::min)
}

#[test]
fn dmrg_limite_ferromagnetico_h0_exacto() {
    // h=0: H = −J Σσᶻσᶻ, suelo |↑↑…↑⟩ (o ↓…↓) con E₀ = −J(L−1), D=1.
    let (l, j) = (8usize, 1.0f64);
    let r = dmrg_ising(l, j, 0.0, 1, 4).expect("h=0 D=1");
    let esperado = -j * (l as f64 - 1.0);
    assert!(
        (r.energia - esperado).abs() < 1e-9,
        "h=0: E {e} ≈ {esperado}",
        e = r.energia
    );
    // El estado ES producto: todos los enlaces internos D=1.
    for &d in &r.mps.dims[1..r.mps.largo] {
        assert_eq!(d, 1, "h=0 no genera entrelazamiento");
    }
}

#[test]
fn dmrg_limite_espin_libre_j0_exacto() {
    // J=0: H = −h Σσˣ, cada espín independiente: E₀ = −h·L, D=1 basta.
    let (l, h) = (8usize, 1.0f64);
    let r = dmrg_ising(l, 0.0, h, 1, 4).expect("J=0 D=1");
    let esperado = -h * l as f64;
    assert!(
        (r.energia - esperado).abs() < 1e-9,
        "J=0: E {e} ≈ {esperado}",
        e = r.energia
    );
}

#[test]
fn dmrg_punto_critico_recupera_el_suelo_exacto() {
    // h=J=1 (punto crítico, máxima entrelazamiento): D=8 con 12 sweeps
    // debe alcanzar el suelo exacto (diagonalización exhaustiva) con error
    // variacional pequeño y SIEMPRE por encima (E_DMRG ≥ E₀).
    let (l, j, h) = (8usize, 1.0, 1.0);
    let e0 = energia_exacta_ising(l, j, h);
    let r = dmrg_ising(l, j, h, 8, 12).expect("crítico D=8");
    assert!(
        r.energia >= e0 - 1e-6,
        "VARIACIONAL: E_DMRG {dm} ≥ E₀ {e0}",
        dm = r.energia
    );
    assert!(
        r.energia - e0 < 1e-3,
        "convergencia crítica: |E−E₀| = {d}",
        d = r.energia - e0
    );
}

#[test]
fn dmrg_monotonia_variacional_en_d() {
    // E(D=1) ≥ E(D=2) ≥ E(D=4): subespacios anidados (principio
    // variacional). En h=0 D=1 ya es exacto ⇒ igualdad.
    let (l, j, h) = (8usize, 1.0, 0.5);
    let e1 = dmrg_ising(l, j, h, 1, 6).expect("D=1").energia;
    let e2 = dmrg_ising(l, j, h, 2, 6).expect("D=2").energia;
    let e4 = dmrg_ising(l, j, h, 4, 6).expect("D=4").energia;
    assert!(e1 >= e2 - 1e-9, "E(D=1) {e1} ≥ E(D=2) {e2}");
    assert!(e2 >= e4 - 1e-9, "E(D=2) {e2} ≥ E(D=4) {e4}");
}

#[test]
fn dmrg_energia_vs_entrelazamiento_del_estado_final() {
    // El MPS optimizado del punto crítico debe tener entrelazamiento
    // ~ln2·(escala conformal) > 0 en el corte medio: coherencia física.
    let (l, j, h) = (8usize, 1.0, 1.0);
    let r = dmrg_ising(l, j, h, 4, 8).expect("crítico");
    let psi = reconstruir_estado(&r.mps);
    let s = entrelazamiento_corte(&psi, l, l / 2).expect("S del corte medio");
    assert!(s > 0.3, "crítico: S {s} > 0.3 (estado entrelazado)");
    assert!(s < 1.0, "S {s} < ln2·(algo) — acotado por D=4");
}

/// Reconstruye el estado completo 2^L desde el MPS (contracción densa).
fn reconstruir_estado(mps: &feature_engine::mps::Mps) -> Vec<f64> {
    // env[(a), configuración_parcial] → expandimos sitio a sitio.
    let mut env: Vec<(usize, Vec<f64>)> = vec![(1usize, vec![1.0])]; // (D_izq, coefs)
    for l in 0..mps.largo {
        let dnext = mps.dims[l + 1];
        let a = &mps.tensores[l];
        let nconf = env[0].1.len();
        let mut nueva: Vec<(usize, Vec<f64>)> =
            (0..dnext).map(|_| (dnext, vec![0.0; nconf * 2])).collect();
        for (aix, (_, coefs)) in env.iter().enumerate() {
            for fs in 0..2 {
                for aj in 0..dnext {
                    let v = a[((aix * 2 + fs) * dnext) + aj];
                    if v == 0.0 {
                        continue;
                    }
                    for (c, coef) in coefs.iter().enumerate() {
                        nueva[aj].1[c * 2 + fs] += v * coef;
                    }
                }
            }
        }
        env = nueva;
    }
    env[0].1.clone()
}

#[test]
fn dmrg_reconstruccion_norma_uno() {
    // Autoconsistencia: el estado reconstruido del MPS optimizado tiene
    // norma 1 (el DMRG trabaja en el paquete normalizado).
    let (l, j, h) = (6usize, 1.0, 1.0);
    let r = dmrg_ising(l, j, h, 4, 6).expect("norma");
    let psi = reconstruir_estado(&r.mps);
    let n: f64 = psi.iter().map(|x| x * x).sum();
    assert!((n - 1.0).abs() < 1e-6, "norma {n} ≈ 1");
    // Y el MPS desde ese estado se recomprime sin error.
    let (mps2, err) = mps_desde_estado(&psi, l, 8).expect("recompresión");
    assert!(err < 1e-12, "sin truncamiento a D=8: err {err}");
    assert!(
        (mps2.contraccion_norma() - 1.0).abs() < 1e-6,
        "norma por contracción ≈ 1"
    );
}
