//! Contratos del detector LPPLS de Sornette (GLM-189, mandato §11.7).
//! Toda la numerología de este contrato fue VERIFICADA A MANO (python,
//! mismo LCG y mismo orden de acumulación) antes de escribir el núcleo:
//! dos diseños de estadístico fueron descartados en ese paso (mejora_rel
//! sola no discrimina — el caminante dispara a 0.66; el F clásico asume
//! ruido blanco y el null real de un precio es un caminante autocorrelado
//! — maxF 500). Lo certificado aquí es lo que el núcleo garantiza.
use feature_engine::lppls_sornette::{InformeLppls, MIN_MUESTRAS, UMBRAL_EVIDENCIA, lppls_ventana};

/// LCG determinista del contrato (idéntico al verificado a mano: Knuth
/// 64-bit, uniforme en [0,1) por los 53 bits altos, gauss por Box–Muller
/// con par (u1, u2) en orden fijo). Sin dependencia de `rand`:
/// bit-reproducible en CI y en local.
struct Lcg(u64);

impl Lcg {
    fn unif(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.unif();
        let u2 = self.unif().max(1e-300);
        (2.0 * std::f64::consts::PI * u1).cos() * (-2.0 * u2.ln()).sqrt()
    }
}

/// Sintética LPPLS de mercado: t_c = t_end + 0.5·T_w, m=0.4, ω=8 (los
/// valores del rango empírico), con ruido gaussiano de desviación `sigma`.
fn sintetica_lppls(amplitud_c: f64, sigma: f64, semilla: u64, n: usize) -> Vec<f64> {
    let mut rng = Lcg(semilla);
    let tw = (n - 1) as f64;
    let t_c = tw + 0.5 * tw; // t_end = n−1 = tw muestras
    (0..n)
        .map(|i| {
            let u = (t_c - i as f64) / tw;
            let um = u.powf(0.4);
            let base = 100.0 - 20.0 * um + amplitud_c * um * (8.0 * u.ln()).cos();
            if sigma > 0.0 {
                base + rng.gauss() * sigma
            } else {
                base
            }
        })
        .collect()
}

/// Caminante aleatorio determinista (el NULL REAL de un precio — no ruido
/// blanco: la lección que descartó el estadístico F).
fn caminante(semilla: u64, n: usize, sigma: f64) -> Vec<f64> {
    let mut rng = Lcg(semilla);
    let mut y = vec![100.0f64];
    for _ in 1..n {
        let prev = y[y.len() - 1];
        y.push(prev + rng.gauss() * sigma);
    }
    y
}

#[test]
fn sintetica_de_mercado_recupera_tc_m_omega_exactos() {
    // C=6, σ=0.5, semilla 7: verificado a mano — el ajuste de ventana
    // completa por mínimo SSE selecciona EXACTAMENTE el bin verdadero.
    let y = sintetica_lppls(6.0, 0.5, 7, 256);
    let r = lppls_ventana(&y, 1).expect("sintética válida");
    // t_c verdadero: 0.5·T_w·paso = 0.5·255 = 127.5 ms adelante.
    assert!(
        (r.t_c_ms_adelante - 127.5).abs() < 1e-6,
        "t_c recuperado: {} ms (verdadero 127.5)",
        r.t_c_ms_adelante
    );
    assert_eq!(r.m, 0.4, "exponente m recuperado exacto");
    assert_eq!(r.omega, 8.0, "frecuencia log-periódica ω recuperada exacta");
    assert!(
        r.evidencia > 0.30 && r.evidencia < 0.45,
        "evidencia de la sintética de mercado: {} (medida a mano 0.383)",
        r.evidencia
    );
    // La robustez al ruido del ajustador: mismo bin verdadero con σ=1.0.
    let y_ruidosa = sintetica_lppls(6.0, 1.0, 7, 256);
    let r2 = lppls_ventana(&y_ruidosa, 1).expect("sintética válida");
    assert_eq!(
        (r2.t_c_ms_adelante, r2.m, r2.omega),
        (127.5, 0.4, 8.0),
        "el ajuste exacto sobrevive a σ=1.0 (medido a mano)"
    );
}

#[test]
fn la_evidencia_es_monotona_en_la_amplitud_y_sin_confianza_fabricada() {
    // Misma semilla de ruido, amplitud log-periódica decreciente: la
    // evidencia CAE (C=6: 0.383 > C=3: 0.217 > C=1.5: 0.085, medido a
    // mano) y en señal débil queda BAJO el umbral sin fabricar confianza.
    let evs: Vec<f64> = [6.0f64, 3.0, 1.5]
        .iter()
        .map(|&c| {
            let y = sintetica_lppls(c, 0.5, 7, 256);
            lppls_ventana(&y, 1).expect("sintética válida").evidencia
        })
        .collect();
    assert!(
        evs[0] > evs[1] && evs[1] > evs[2],
        "monotonía en C: {:?} (medido a mano 0.383 > 0.217 > 0.085)",
        evs
    );
    assert!(
        evs[2] < UMBRAL_EVIDENCIA,
        "señal débil bajo el umbral: {} < {} — sin confianza fabricada",
        evs[2],
        UMBRAL_EVIDENCIA
    );
}

#[test]
fn el_nulo_caminante_nunca_encadena() {
    // 16 caminantes deterministas (el null real de un precio): la
    // EVIDENCIA sola cruza el umbral en ~la mitad (calibración de la
    // cabecera del módulo — por eso NO se usa sola), pero la señal
    // compuesta `evidencia >= umbral && estable` NO dispara en NINGUNO
    // (estable = unanimidad de 3 ventanas ancladas al final; 0/16 medido).
    let mut max_evidencia = 0.0f64;
    for semilla in 1u64..=16 {
        let y = caminante(semilla, 256, 0.5);
        let r = lppls_ventana(&y, 1).expect("caminante no degenerado");
        assert!(
            !r.estable,
            "semilla {semilla}: el caminante NO encadena (0/16 medido a mano)"
        );
        let dispara = r.evidencia >= UMBRAL_EVIDENCIA && r.estable;
        assert!(
            !dispara,
            "semilla {semilla}: la señal compuesta no dispara en ruido"
        );
        max_evidencia = max_evidencia.max(r.evidencia);
    }
    // La calibración documentada queda certificada EN el contrato: el
    // máximo de evidencia del nulo medido es 0.340 (semilla 14).
    assert!(
        max_evidencia > 0.20 && max_evidencia < 0.40,
        "calibración del nulo: max evidencia {} (medida a mano 0.340) — \
         evidencia sola NO es detector, está documentado",
        max_evidencia
    );
}

#[test]
fn falla_cerrado_sin_ceros_fabricados() {
    // Serie corta.
    let corta = vec![100.0f64; MIN_MUESTRAS - 1];
    assert!(lppls_ventana(&corta, 1).is_err());
    // Precio no finito o <= 0: Err, nunca se inyecta cero.
    let mut con_nan = sintetica_lppls(6.0, 0.5, 7, 256);
    con_nan[128] = f64::NAN;
    assert!(lppls_ventana(&con_nan, 1).is_err());
    let mut con_cero = sintetica_lppls(6.0, 0.5, 7, 256);
    con_cero[128] = 0.0;
    assert!(lppls_ventana(&con_cero, 1).is_err());
    // paso_ms cero.
    let y = sintetica_lppls(6.0, 0.5, 7, 256);
    assert!(lppls_ventana(&y, 0).is_err());
    // Serie constante (SST=0): degenerada, no evidencia inventada.
    let constante = vec![100.0f64; 256];
    assert!(lppls_ventana(&constante, 1).is_err());
    // El informe de una serie válida reproduce n y mantiene t_c futuro.
    let r: InformeLppls = lppls_ventana(&y, 1000).expect("válida");
    assert_eq!(r.n, 256);
    assert!(
        r.t_c_ms_adelante > 0.0,
        "t_c SIEMPRE futuro: {} ms",
        r.t_c_ms_adelante
    );
}
