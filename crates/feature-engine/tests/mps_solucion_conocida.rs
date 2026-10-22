//! Contratos del núcleo MPS (GLM 151) con SOLUCIÓN CONOCIDA — los casos de
//! libro de Schollwöck (2011, «DMRG in the age of matrix product states»):
//! producto, GHZ y W definen el comportamiento exacto de la compresión y
//! del entrelazamiento. El motor debe recuperarlos dentro de tolerancia o
//! declarar ausencia (CL-15), jamás inventar.
use feature_engine::mps::{entrelazamiento_corte, mps_desde_estado, MpsError, MPS_MAX_D};

fn normalizar(v: &mut [f64]) {
    let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 1e-300 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// GHZ_L = (|0...0⟩ + |1...1⟩)/√2.
fn ghz(l: usize) -> Vec<f64> {
    let mut v = vec![0.0; 1 << l];
    v[0] = 1.0;
    v[(1 << l) - 1] = 1.0;
    normalizar(&mut v);
    v
}

/// W_L = (un solo |1⟩ en cada posición)/√L.
fn w_state(l: usize) -> Vec<f64> {
    let mut v = vec![0.0; 1 << l];
    for i in 0..l {
        v[1 << i] = 1.0;
    }
    normalizar(&mut v);
    v
}

/// Producto: |0101...⟩.
fn producto(l: usize) -> Vec<f64> {
    let mut v = vec![0.0; 1 << l];
    v[0b0101_0101 & ((1 << l) - 1)] = 1.0;
    v
}

#[test]
fn mps_producto_sin_entrelazamiento_d1_suficiente() {
    let l = 6;
    let psi = producto(l);
    let s = entrelazamiento_corte(&psi, l, l / 2).expect("producto estimable");
    assert!(s.abs() < 1e-9, "producto ⇒ S=0, llegó {s}");
    let (mps, err) = mps_desde_estado(&psi, l, MPS_MAX_D).expect("compresión");
    assert!(err < 1e-18, "sin truncamiento: err {err}");
    // D=1 en TODOS los enlaces internos.
    for &d in &mps.dims[1..mps.largo] {
        assert_eq!(d, 1, "producto no necesita entrelazamiento");
    }
    let norma = mps.contraccion_norma();
    assert!((norma - 1.0).abs() < 1e-9, "norma {norma} ≈ 1");
}

#[test]
fn mps_ghz_entrelazamiento_ln2_y_bond_2() {
    let l = 6;
    let psi = ghz(l);
    // Solución de libro: S = ln 2 en CADA corte (máximo para D=2).
    let ln2 = std::f64::consts::LN_2;
    for corte in 1..l {
        let s = entrelazamiento_corte(&psi, l, corte).expect("GHZ estimable");
        assert!(
            (s - ln2).abs() < 1e-9,
            "GHZ corte {corte}: S {s} ≈ ln2 {ln2}"
        );
    }
    let (mps, err) = mps_desde_estado(&psi, l, MPS_MAX_D).expect("compresión");
    assert!(err < 1e-18, "GHZ con D=2 exacto: err {err}");
    let norma = mps.contraccion_norma();
    assert!((norma - 1.0).abs() < 1e-9, "norma {norma} ≈ 1");
}

#[test]
fn mps_estado_w_entrelazamiento_forma_cerrada() {
    let l = 6;
    let psi = w_state(l);
    let ln2 = std::f64::consts::LN_2;
    // Corte MEDIO: bloques simétricos ⇒ S = ln 2 exacto (ρ_L = (1/L)J
    // con autovalor 1/2 — el caso de libro que el corte simétrico maximiza).
    let s = entrelazamiento_corte(&psi, l, l / 2).expect("W estimable");
    assert!((s - ln2).abs() < 1e-9, "W corte medio: S {s} ≈ ln2");
    // Corte 1|L-1 (asimétrico): S = −p·ln p − (1−p)·ln(1−p), p = 1/L < ln2.
    let k = 1usize;
    let p = k as f64 / l as f64;
    let esperado = -p * p.ln() - (1.0 - p) * (1.0 - p).ln();
    let s1 = entrelazamiento_corte(&psi, l, k).expect("W corte 1 estimable");
    assert!(
        (s1 - esperado).abs() < 1e-9,
        "W corte 1: S {s1} ≈ forma cerrada {esperado} (< ln2)"
    );
    let (mps, err) = mps_desde_estado(&psi, l, MPS_MAX_D).expect("compresión");
    assert!(err < 1e-18, "W con D=2 exacto: err {err}");
    let norma = mps.contraccion_norma();
    assert!((norma - 1.0).abs() < 1e-9, "norma {norma} ≈ 1");
}

#[test]
fn mps_autoconsistencia_norma_y_error() {
    // GHZ-escalera (⊗ GHZ_2): D=2 exacto ⇒ norma 1.0 y error ~0 en L=10.
    let pares = 5;
    let l = 2 * pares;
    let mut v = vec![0.0f64; 1 << l];
    // |ψ⟩ = ⊗_{i} (|00⟩+|11⟩)/√2: índice con pares de bits iguales.
    fn construir(idx: usize, prof: usize, l: usize, v: &mut Vec<f64>, amp: f64) {
        if prof * 2 == l {
            v[idx] = amp;
            return;
        }
        let salto = 1usize << (l - (prof + 1) * 2);
        construir(idx, prof + 1, l, v, amp * std::f64::consts::FRAC_1_SQRT_2);
        construir(idx + 3 * salto, prof + 1, l, v, amp * std::f64::consts::FRAC_1_SQRT_2);
    }
    construir(0, 0, l, &mut v, 1.0);
    let (mps, err) = mps_desde_estado(&v, l, 2).expect("GHZ-escalera comprime");
    let norma = mps.contraccion_norma();
    assert!((norma - 1.0).abs() < 1e-9, "escalera D=2 exacta: norma {norma}");
    assert!(err < 1e-18, "escalera sin truncamiento: err {err}");

    // Aleatorio L=10 con D=4: capturada = 1 − err (AUTOconsistencia interna
    // de la compresión; el nivel absoluto depende del entrelazamiento del
    // estado — un aleatorio tiene S alto y D=4 retiene una fracción).
    let mut semilla = 0x9E3779B97F4A7C15u64;
    let mut r = vec![0.0f64; 1 << l];
    for x in r.iter_mut() {
        semilla = semilla
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x = ((semilla >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
    }
    let nrm: f64 = r.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in r.iter_mut() {
        *x /= nrm;
    }
    let (m4, err4) = mps_desde_estado(&r, l, 4).expect("aleatorio comprime");
    let capturada = m4.contraccion_norma();
    assert!(capturada > 0.0 && capturada <= 1.0 + 1e-9, "fracción física");
    assert!(
        (capturada - (1.0 - err4)).abs() < 1e-6,
        "autoconsistencia norma+error: capturada {capturada} ≈ 1−err {err4}"
    );
    // Y con D=8 captura MÁS que con D=4 (monotonía del truncamiento).
    let (m8, _e8) = mps_desde_estado(&r, l, 8).expect("aleatorio D=8");
    assert!(
        m8.contraccion_norma() >= capturada - 1e-12,
        "D=8 ≥ D=4 (monotonía)"
    );
}

#[test]
fn mps_ortogonalidad_soportes_disjuntos() {
    let l = 5;
    let a = producto(l); // |01010⟩
    let mut b = vec![0.0; 1 << l];
    b[0] = 1.0; // |00000⟩ — soporte disjunto
    let (ma, _) = mps_desde_estado(&a, l, 2).expect("a");
    let (mb, _) = mps_desde_estado(&b, l, 2).expect("b");
    let solape = ma
        .contraccion_solape(&mb)
        .expect("solape de cadenas iguales");
    assert!(solape.abs() < 1e-9, "soportes disjuntos ⇒ ⟨a|b⟩=0, llegó {solape}");
    // Auto-solape = norma² = 1.
    let auto = ma.contraccion_solape(&ma).expect("auto");
    assert!((auto - 1.0).abs() < 1e-9, "⟨a|a⟩ {auto} ≈ 1");
}

#[test]
fn mps_truncamiento_agota_bond_y_reporta_error() {
    // GHZ con bond_objetivo=1 NO puede representarse: el error de
    // truncamiento debe reflejar la mitad descartada (= 0.5 de norma²).
    let l = 4;
    let psi = ghz(l);
    let (mps, err) = mps_desde_estado(&psi, l, 1).expect("comprime con D=1");
    let capturada = mps.contraccion_norma();
    assert!(
        (capturada - 0.5).abs() < 1e-9,
        "GHZ con D=1 captura exactamente una mitad: {capturada}"
    );
    assert!(err > 0.4, "el error reporta lo descartado (espectro completo): {err}");
}

#[test]
fn mps_entradas_invalidas_son_ausencia_honesta() {
    let corto: Vec<f64> = vec![1.0];
    match mps_desde_estado(&corto, 0, 2) {
        Err(MpsError::LongitudInvalida { .. }) => {}
        otro => panic!("L=0 ⇒ Err, llegó {otro:?}"),
    }
    let con_nan = vec![1.0, f64::NAN, 0.0, 0.0];
    match mps_desde_estado(&con_nan, 2, 2) {
        Err(MpsError::EntradaNoFinita) => {}
        otro => panic!("NaN ⇒ Err, llegó {otro:?}"),
    }
    let mal_tam = vec![1.0, 0.0, 0.0];
    match mps_desde_estado(&mal_tam, 2, 2) {
        Err(MpsError::EstadoLongitud { .. }) => {}
        otro => panic!("tamaño ⇒ Err, llegó {otro:?}"),
    }
    match mps_desde_estado(&vec![1.0, 0.0, 0.0, 0.0], 2, 0) {
        Err(MpsError::BondInvalido { .. }) => {}
        otro => panic!("bond 0 ⇒ Err, llegó {otro:?}"),
    }
    match entrelazamiento_corte(&vec![1.0, 0.0, 0.0, 0.0], 2, 0) {
        Err(MpsError::LongitudInvalida { .. }) => {}
        otro => panic!("corte 0 ⇒ Err, llegó {otro:?}"),
    }
}
