//! Contratos del filtrado de Marchenko-Pastur (GLM-182, mandato §12.1)
//! con SOLUCIÓN CONOCIDA: la banda λ± = (1 ± √(N/T))² separa ruido de
//! señal en matrices de correlación.
//!
//! Formas cerradas usadas:
//! - Identidad: espectro {1}×N — todo dentro de la banda (λ+ ≥ 1 siempre).
//! - Equicorrelación b: espectro {1+(N−1)·b, 1−b ×(N−1)} — el dominante
//!   es SEÑAL (≫ λ+ con b alto) y el resto SUB-CRÍTICO (1−b < λ−).
//! - Wishart puro (T×N iid): todo el espectro converge a [λ−, λ+] con
//!   fluctuaciones de Tracy-Widom en el borde — determinista bajo seed
//!   fija, con margen documentado.
use feature_engine::matriz_rmt::{MatrizCorrelacionEwma, filtrar_mp};

/// LCG determinista (Knuth MMIX) — el mismo del patrón cf2 del crate.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn matriz_identidad(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

fn matriz_equicorrelada(n: usize, b: f64) -> Vec<f64> {
    let mut m = matriz_identidad(n);
    for i in 0..n {
        for j in (i + 1)..n {
            m[i * n + j] = b;
            m[j * n + i] = b;
        }
    }
    m
}

#[test]
fn identidad_es_todo_ruido_y_la_banda_es_exacta() {
    let n = 8;
    let t = 200.0;
    let informe = filtrar_mp(&matriz_identidad(n), n, t).expect("identidad filtrable");
    // λ± = (1 ± √(8/200))² = (1 ± 0.2)² → 0.64 / 1.44 exactos.
    assert!((informe.lambda_menos - 0.64).abs() < 1e-12);
    assert!((informe.lambda_mas - 1.44).abs() < 1e-12);
    assert_eq!(informe.n_ruido, n, "identidad: todo ruido");
    assert_eq!(informe.n_senal, 0);
    assert_eq!(informe.n_subcritico, 0);
    assert!(informe.autovalores.iter().all(|&l| (l - 1.0).abs() < 1e-9));
}

#[test]
fn equicorrelacion_un_factor_da_una_senal_exacta() {
    let (n, b, t) = (8usize, 0.7_f64, 200.0);
    let informe = filtrar_mp(&matriz_equicorrelada(n, b), n, t).expect("filtrable");
    // Forma cerrada: dominante 1+(N−1)b = 5.9; resto 1−b = 0.3 (×7).
    let dominante = 1.0 + (n as f64 - 1.0) * b;
    assert!(
        (informe.autovalores[0] - dominante).abs() < 1e-9,
        "Jacobi debe dar el dominante exacto {dominante}, dio {}",
        informe.autovalores[0]
    );
    for k in 1..n {
        assert!(
            (informe.autovalores[k] - (1.0 - b)).abs() < 1e-9,
            "modo interior {k}: {} ≈ 1−b",
            informe.autovalores[k]
        );
    }
    // Banda [0.64, 1.44]: 5.9 es señal; 0.3 × 7 son sub-críticos; ruido 0.
    assert_eq!(informe.n_senal, 1);
    assert_eq!(informe.n_subcritico, n - 1);
    assert_eq!(informe.n_ruido, 0);
}

#[test]
fn ruido_wishart_puro_no_produce_senal() {
    // T×N iid estándar: el espectro muestral vive en la banda MP; el
    // máximo fluctúa sobre λ+ por Tracy-Widom (~T^{-2/3}). Con seed fija
    // el resultado es determinista; T=6000, N=8 deja el máximo holgado
    // bajo λ+·1.10 y NINGUNA señal bajo el borde exacto de esta seed.
    let (t_total, n) = (6000usize, 8usize);
    let mut rng = Lcg(42);
    let mut suma = vec![0.0; n];
    let mut suma2 = vec![0.0; n * n];
    let mut serie = Vec::with_capacity(t_total * n);
    for _ in 0..t_total {
        for _ in 0..n {
            serie.push(rng.next_f64());
        }
    }
    for fila in 0..t_total {
        for i in 0..n {
            suma[i] += serie[fila * n + i];
        }
    }
    for i in 0..n {
        suma[i] /= t_total as f64;
    }
    for fila in 0..t_total {
        for i in 0..n {
            for j in 0..n {
                let di = serie[fila * n + i] - suma[i];
                let dj = serie[fila * n + j] - suma[j];
                suma2[i * n + j] += di * dj;
            }
        }
    }
    // Normalizar a correlación (el test construye la muestra completa:
    // T efectivo = t_total, sin EWMA).
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let r =
                (suma2[i * n + j] / (suma2[i * n + i] * suma2[j * n + j]).sqrt()).clamp(-1.0, 1.0);
            m[i * n + j] = r;
            m[j * n + i] = r;
        }
    }
    let informe = filtrar_mp(&m, n, t_total as f64).expect("wishart filtrable");
    let maximo = informe.autovalores[0];
    assert!(
        maximo <= informe.lambda_mas * 1.10,
        "máximo {maximo} dentro del borde TW (λ+ {})",
        informe.lambda_mas
    );
    assert!(
        informe.n_senal <= 1,
        "a lo sumo una falsa señal de borde TW, dio {}",
        informe.n_senal
    );
    assert_eq!(informe.n_subcritico, 0, "ruido iid: nada bajo λ−");
}

#[test]
fn la_banda_se_estrecha_con_la_evidencia() {
    let n = 8;
    let ancha = filtrar_mp(&matriz_identidad(n), n, 50.0).expect("filtrable");
    let angosta = filtrar_mp(&matriz_identidad(n), n, 5000.0).expect("filtrable");
    assert!(ancha.lambda_mas > angosta.lambda_mas);
    assert!(ancha.lambda_menos < angosta.lambda_menos);
    // Errores honestos: T chico o no finito.
    assert!(filtrar_mp(&matriz_identidad(n), n, 1.5).is_err());
    assert!(filtrar_mp(&matriz_identidad(n), n, f64::NAN).is_err());
    assert!(filtrar_mp(&matriz_identidad(1), 1, 100.0).is_err());
}

#[test]
fn streaming_captura_el_factor_comun() {
    // 3 activos: a = f + e1, b = f + e2 (factor común fuerte), c = ruido.
    // Con period 200 (T efectivo 200) la banda es [0.88, 1.13]; la pareja
    // correlacionada produce un dominante ≈ 1+ρ ≈ 1.8 ⇒ señal.
    let mut matriz = MatrizCorrelacionEwma::new(3, 200.0);
    let mut rng = Lcg(7);
    let mut precio = [100.0, 100.0, 100.0];
    let mut informe = None;
    for corte in 0..1200u32 {
        let f = 0.01 * rng.next_f64();
        let ret = [
            f + 0.003 * rng.next_f64(),
            f + 0.003 * rng.next_f64(),
            0.01 * rng.next_f64(),
        ];
        for (p, r) in precio.iter_mut().zip(ret.iter()) {
            *p *= 1.0 + r;
        }
        matriz.try_update(&precio).expect("corte válido");
        if corte % 64 == 63 && corte > 400 {
            if let Ok(info) = matriz.analizar() {
                informe = Some(info);
            }
        }
    }
    let info = informe.expect("análisis posible tras warmup");
    assert!(
        info.n_senal >= 1,
        "el factor común debe ser super-crítico; espectro {:?}",
        info.autovalores
    );
    assert!(
        info.autovalores[0] > info.lambda_mas,
        "dominante {} > λ+ {}",
        info.autovalores[0],
        info.lambda_mas
    );
}
