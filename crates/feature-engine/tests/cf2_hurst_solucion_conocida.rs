//! C-F2.3b (Línea C): el DFA1 de `hurst_dfa.rs` frente a soluciones conocidas.
//!
//! Teoría que se prueba:
//! - Ruido gaussiano fraccionario (fGn) con H = 0,7, generado exacto por
//!   Durbin-Levinson (Hosking 1984): la pendiente media del DFA1 sobre
//!   1024 retornos debe quedar a menos de 0,05 de 0,7, y la lectura literal
//!   del plan («error < 0,05») se aplica también a cada realización.
//! - Paseo aleatorio (H = 0,5 exacto): la media de la pendiente debería
//!   quedar dentro de 4 errores estándar de 0,5. El DFA1 tiene un sesgo de
//!   muestra finita con forma cerrada, E[F²(s)] = (s² − 4)/(15 s) para el
//!   perfil de ruido blanco, cuya pendiente MCO sobre la rejilla del motor
//!   (s = 4…256) vale ≈ 0,526; la segunda prueba de ruido blanco demuestra
//!   que la desviación respecto de 0,5 es ese sesgo y no un error de código.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use feature_engine::hurst_dfa::HurstDfa;

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
    /// Normal estándar por Box-Muller.
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Durbin-Levinson (Hosking 1984): coeficientes φ_k y desviación de la
/// innovación de un fGn de varianza 1 con exponente `h`.
fn hosking(h: f64, n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    // Autocovarianza del fGn: γ(k) = ½(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H}).
    let g = |k: usize| {
        let k = k as f64;
        0.5 * ((k + 1.0).powf(2.0 * h) - 2.0 * k.powf(2.0 * h) + (k - 1.0).abs().powf(2.0 * h))
    };
    let (mut phis, mut sd, mut v) = (vec![Vec::new()], vec![1.0], 1.0);
    let mut prev: Vec<f64> = Vec::new();
    for k in 1..n {
        let mut num = g(k);
        for j in 1..k {
            num -= prev[j - 1] * g(k - j);
        }
        let pkk = num / v;
        let mut cur = vec![0.0; k];
        for j in 1..k {
            cur[j - 1] = prev[j - 1] - pkk * prev[k - j - 1];
        }
        cur[k - 1] = pkk;
        v *= 1.0 - pkk * pkk;
        sd.push(v.sqrt());
        phis.push(cur.clone());
        prev = cur;
    }
    (phis, sd)
}

/// Una realización exacta del fGn: x_k = Σ φ_{k,j} x_{k−j} + σ_k ε_k.
fn fgn(phis: &[Vec<f64>], sd: &[f64], rng: &mut Lcg) -> Vec<f64> {
    let mut x = vec![0.0; sd.len()];
    for k in 0..sd.len() {
        let mut m = 0.0;
        for j in 1..=k {
            m += phis[k][j - 1] * x[k - j];
        }
        x[k] = m + sd[k] * rng.n();
    }
    x
}

/// Pendiente cruda del DFA tras alimentar 1024 retornos (escalados a 1e-3)
/// como log-precio acumulado. Con 1024 retornos el motor recalcula justo en
/// el último ((1024 − 512)/32 = 16 recomputaciones tras el calentamiento).
fn pendiente(incrementos: &[f64]) -> f64 {
    let mut dfa = HurstDfa::new();
    let mut lnp = 100f64.ln();
    dfa.update(lnp.exp());
    for &x in incrementos {
        lnp += 1e-3 * x;
        dfa.update(lnp.exp());
    }
    assert_eq!(dfa.samples(), 1024);
    dfa.raw_exponent.expect("ajuste con 1024 retornos")
}

fn media_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt();
    (m, s)
}

/// `m` paseos aleatorios de 1024 retornos gaussianos iid y su pendiente DFA.
fn ruido_blanco(m: usize, semilla: u64) -> Vec<f64> {
    let mut rng = Lcg(semilla);
    (0..m)
        .map(|_| {
            let x: Vec<f64> = (0..1024).map(|_| rng.n()).collect();
            pendiente(&x)
        })
        .collect()
}

/// Verdad: H = 0,7. Tolerancia 0,05: el sesgo de muestra finita del DFA1 a
/// N = 1024 es de orden +0,02 (curvatura a escalas pequeñas; la media global
/// restada es una tendencia lineal del perfil que cada ventana elimina) y la
/// SE de la media de 100 realizaciones es ≈ 0,004, así que el margen supera
/// 7 SE por encima del sesgo esperado.
#[test]
fn cf2_dfa_fgn_07_media_dentro_de_005() {
    let (phis, sd) = hosking(0.7, 1024);
    let mut rng = Lcg(0x5EED_F23B);
    let hs: Vec<f64> = (0..100).map(|_| pendiente(&fgn(&phis, &sd, &mut rng))).collect();
    let (m, s) = media_sd(&hs);
    eprintln!("[cf2 dfa fGn 0,7] media {m:.4}, sd {s:.4}, se {:.4}", s / 10.0);
    assert!((m - 0.7).abs() < 0.05, "media {m:.4}, sd {s:.4}, se {:.4}", s / 10.0);
}

/// Verdad: H = 0,7 en cada realización. Lectura literal del plan: al menos
/// 95 de 100 realizaciones con |pendiente − 0,7| < 0,05. Medido: sd por
/// realización ≈ 0,037 y sesgo ≈ +0,01, de donde la fracción esperada dentro
/// de ±0,05 es Φ(0,04/0,037) − Φ(−0,06/0,037) ≈ 0,81 (se observan 81/100;
/// el 95 queda a ≈ 3,6 errores binomiales). No es un error aritmético: es la
/// precisión del DFA1 con 1024 retornos, la misma dispersión que hace que una
/// sola realización caiga a ambos lados de las bandas fijas (F2.3b-01). Sólo
/// pasará con más muestras o un estimador más eficiente, no con descontar
/// el sesgo. Revisión cruzada (CL-65): por eso el arreglo del sesgo de
/// F2.3b-01 NO la pone en verde. Es un requisito de precisión del plan, que
/// se cumple con N ≈ 4096 (sd ≈ 0,0185, fracción esperada > 0,99) o con un
/// estimador más eficiente; su dueño decide cuál.
#[test]
#[ignore = "OPEN F2.3b-01 (Línea C), requisito de precisión, no se cierra con el arreglo del sesgo: la sd por realización del DFA1 a N = 1024 (≈ 0,037) deja ≈ 81/100 realizaciones con error < 0,05, no 95"]
fn cf2_dfa_fgn_07_error_por_realizacion() {
    let (phis, sd) = hosking(0.7, 1024);
    let mut rng = Lcg(0x5EED_F23C);
    let hs: Vec<f64> = (0..100).map(|_| pendiente(&fgn(&phis, &sd, &mut rng))).collect();
    let (m, s) = media_sd(&hs);
    let dentro = hs.iter().filter(|&&h| (h - 0.7).abs() < 0.05).count();
    assert!(
        dentro >= 95,
        "sólo {dentro}/100 realizaciones con error < 0,05 (media {m:.4}, sd {s:.4})"
    );
}

/// Verdad: H = 0,5 (paseo aleatorio). Tolerancia 4 SE con SE = sd/√200
/// (≈ 0,0025), el intervalo de Monte Carlo que la API no ofrece. Por la forma
/// cerrada (s² − 4)/(15 s) la media esperada es ≈ 0,52, unas 8 SE por encima
/// (medido: 0,5214 con SE 0,0022, ≈ 9,7 SE). Revisión cruzada (CL-65): el
/// arreglo natural (corregir E[F²] analíticamente) deja el término de Jensen
/// (−0,005 a −0,007, derivado en el test siguiente), ≈ 3 SE; la cota tiene un
/// suelo de 0,01 para que ese residuo conocido no la deje roja por semilla.
#[test]
#[ignore = "OPEN F2.3b-01 (Línea C): el DFA1 a N = 1024 tiene un sesgo de muestra finita de ≈ +0,02 sobre el paseo aleatorio y el estimador no lo descuenta"]
fn cf2_dfa_paseo_aleatorio_centrado_en_medio() {
    let hs = ruido_blanco(200, 0xB1A5_0500);
    let (m, s) = media_sd(&hs);
    let se = s / (hs.len() as f64).sqrt();
    assert!(
        (m - 0.5).abs() < (4.0 * se).max(0.01),
        "media {m:.4} con se {se:.4} (sd {s:.4}): el nulo no está centrado en 0,5"
    );
}

/// Verdad: la pendiente MCO de ½·ln((s² − 4)/(15 s)) sobre ln s en la
/// rejilla del motor, ≈ 0,5259. Tolerancia 0,02 (corrección del escéptico):
/// el término de Jensen por escala, −Var(F²)/(4E[F²]²), con 3,7-5 grados de
/// libertad efectivos por ventana y 4…256 ventanas, baja la pendiente entre
/// 0,005 y 0,007; más 4 SE (4 · 0,0025 = 0,01) da 0,017 < 0,02 (medido:
/// media 0,5194, diferencia −0,0065, la que predice Jensen).
///
/// CARACTERIZACIÓN (revisión cruzada, CL-65): fija la pendiente SIN corregir
/// del estimador actual, y contradice a `cf2_dfa_paseo_aleatorio_centrado_en_medio`
/// una vez arreglado F2.3b-01. Quien arregle el sesgo debe, en el mismo
/// commit, comparar esta prueba contra la pendiente sin corregir (si el
/// estimador la expone) o retirarla.
#[test]
fn cf2_dfa_ruido_blanco_sesgo_de_forma_cerrada() {
    // E[F²(s)] = (s² − 4)/(15 s) para DFA1 sobre ruido blanco integrado.
    let escalas = [4.0f64, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
    let x: Vec<f64> = escalas.iter().map(|s| s.ln()).collect();
    let y: Vec<f64> = escalas
        .iter()
        .map(|s| 0.5 * ((s * s - 4.0) / (15.0 * s)).ln())
        .collect();
    let (mx, my) = (x.iter().sum::<f64>() / 7.0, y.iter().sum::<f64>() / 7.0);
    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let cerrada = sxy / sxx;
    assert!((cerrada - 0.5259).abs() < 1e-3, "pendiente cerrada {cerrada:.5}");
    let hs = ruido_blanco(200, 0xB1A5_0501);
    let (m, s) = media_sd(&hs);
    eprintln!("[cf2 dfa ruido blanco] media {m:.4}, sd {s:.4}, cerrada {cerrada:.4}");
    assert!(
        (m - cerrada).abs() < 0.02,
        "media {m:.4} (sd {s:.4}) frente a la forma cerrada {cerrada:.4}"
    );
}
