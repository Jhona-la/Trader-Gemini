//! C-F2.1 (Línea C): matemática del riesgo y de la cartera contra soluciones
//! conocidas.
//!
//! Pruebas de SOLUCIÓN CONOCIDA de la fase C-F2 (hallazgos en
//! `docs/audit/HALLAZGOS_C-F2_2026-10-11.md`, dominio F2.1). Cada prueba
//! fija un valor o una propiedad que da la teoría y mide el código contra él:
//!
//! - Cramér-Lundberg (`cramer_lundberg.rs`): en un paseo reticular ±a la ruina
//!   es exacta, ψ(m) = (q/p)^{m/a} = e^{−R·m}; con incrementos gaussianos
//!   R = 2μ/σ²; con nocional fijo R_capital = R_nocional·C/N; con bloques de
//!   cierres dependientes el R del paseo es el de los incrementos por bloque.
//! - El tope de ruina (`correlation_guard::veto_por_riesgo_cramer_lundberg`)
//!   debe ser monótono en R, porque ψ(m) = e^{−R·m} decrece con R.
//! - Hodge (`hodge.rs`): en un nulo isótropo denso sobre K_n la fracción de
//!   gradiente es una Beta(d/2, (D−d)/2), así que E[curl] = 1 − 2/n; un árbol
//!   no tiene parte rotacional.
//! - Marchenko-Pastur (`random_matrix.rs`): nivel del veredicto del borde
//!   asintótico en ruido iid, sin correlación poblacional.
//!
//! Generadores propios (LCG de Knuth y Box-Muller), semillas fijas y sin
//! dependencias nuevas.
//!
//! Convención: una prueba que hoy falla por la causa de un hallazgo abierto
//! lleva `#[ignore = "OPEN F2.x-nn (dueño): causa"]` con el assert intacto.
//! Documenta el defecto y se activa (se quita el ignore) con el arreglo. Las
//! pruebas activas son regresiones de la teoría y deben pasar siempre.

use risk_engine::correlation_guard::veto_por_riesgo_cramer_lundberg;
use risk_engine::cramer_lundberg::EstimadorSiniestros;
use risk_engine::hodge::hodge_curl_share;
use risk_engine::random_matrix::{systematic_mode, MppVerdict};

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniforme en (0, 1) abierto.
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Normal estándar por Box-Muller.
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Salto por nocional del paseo reticular (1 %).
const A: f64 = 0.01;
/// Probabilidad de ganar del paseo reticular.
const P: f64 = 0.55;

/// Paseo ±a: p·e^{−Ra} + q·e^{Ra} = 1 ⇒ R = ln(p/q)/a.
fn r_reticular(p: f64, a: f64) -> f64 {
    (p / (1.0 - p)).ln() / a
}

/// Cobertura mínima exigida a una LCB nominal del 95 %: 0,95 − 0,015 − 3 SE
/// binomiales. Los 0,015 cubren la discreción de p̂ con n ≈ 256 (paseo ±a) y
/// el sesgo de muestra finita de la LCB de Wald (caso gaussiano).
fn cobertura_minima(reps: usize) -> f64 {
    0.95 - 0.015 - 3.0 * (0.95 * 0.05 / reps as f64).sqrt()
}

/// Muestra exacta de 110 ganancias (+1 %) y 90 pérdidas (−1 %): p̂ = 0,55.
fn muestra_exacta() -> EstimadorSiniestros {
    let mut est = EstimadorSiniestros::new();
    for i in 0..200 {
        est.observar(if i % 20 < 11 { A } else { -A });
    }
    est
}

/// Una réplica de LCB cubre si LCB ≤ R verdadero, o si no hay LCB (None
/// retira la cota: sin R no se promete nada).
fn cubre(lcb: Option<f64>, r_verdadero: f64) -> bool {
    match lcb {
        None => true,
        Some(l) => l <= r_verdadero,
    }
}

/// Correlación de Pearson con diagonal exactamente 1 y simetría exacta.
fn correlacion(rets: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = rets.len();
    let t = rets[0].len() as f64;
    let z: Vec<Vec<f64>> = rets
        .iter()
        .map(|r| {
            let m = r.iter().sum::<f64>() / t;
            let sd = (r.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / t).sqrt();
            r.iter().map(|x| (x - m) / sd).collect()
        })
        .collect();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        c[i][i] = 1.0;
        for j in (i + 1)..n {
            let v = (z[i].iter().zip(&z[j]).map(|(a, b)| a * b).sum::<f64>() / t).clamp(-1.0, 1.0);
            c[i][j] = v;
            c[j][i] = v;
        }
    }
    c
}

fn gauss(rng: &mut Lcg, n: usize, t: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| (0..t).map(|_| rng.normal()).collect())
        .collect()
}

/// Valor verdadero: R̂ = ln(0,55/0,45)/0,01 = 20,067 (bisección a 1e-12) y,
/// en la retícula ±1 sin rebase, ψ(s) = (q/p)^s = 0,818; 0,548; 0,367; 0,134
/// para s = 1, 3, 5, 10. Tolerancia: 4·√(ψ(1−ψ)/20 000) ≈ 0,010–0,014 más
/// 1e-5. El sesgo por absorber en +50 es r^{s+50}(1−r^s)/(1−r^{s+50}) ≤ 1,1e-5,
/// muy por debajo de 4·SE.
#[test]
fn cf2_lundberg_paseo_reticular_cota_exacta() {
    let r_hat = muestra_exacta().lundberg().expect("deriva positiva");
    assert!(
        (r_hat - r_reticular(P, A)).abs() < 1e-6,
        "R̂ = {r_hat}, cerrado = {}",
        r_reticular(P, A)
    );
    let margenes = [1i64, 3, 5, 10];
    let caminatas = 20_000usize;
    let mut rng = Lcg::new(0xC0F2_0001);
    let mut cruces = [0usize; 4];
    for _ in 0..caminatas {
        let (mut pos, mut min) = (0i64, 0i64);
        while pos < 50 && min > -10 {
            pos += if rng.uniform() < P { 1 } else { -1 };
            min = min.min(pos);
        }
        for (k, &s) in margenes.iter().enumerate() {
            if min <= -s {
                cruces[k] += 1;
            }
        }
    }
    for (k, &s) in margenes.iter().enumerate() {
        let psi = (-r_hat * A * s as f64).exp(); // (q/p)^s exacto: sin rebase
        let emp = cruces[k] as f64 / caminatas as f64;
        let se = (psi * (1.0 - psi) / caminatas as f64).sqrt();
        println!("s = {s}: frecuencia {emp}, ψ = {psi}");
        assert!(
            (emp - psi).abs() <= 4.0 * se + 1e-5,
            "s={s}: frecuencia {emp} frente a ψ = {psi} (4 SE = {})",
            4.0 * se
        );
    }
}

/// Valor verdadero: E[e^{−R·y}] = e^{−Rμ + R²σ²/2} = 1 ⇒ R = 2μ/σ² = 80 con
/// μ = 0,001 y σ = 0,005. La sd sándwich de R̂ con n = 256 es ≈ 26, así que la
/// SE de la media con 400 réplicas es ≈ 1,3 (se usa la empírica). El sesgo de
/// segundo orden del estimador M es ≈ +1,0; se deja 2,5·2R/(n−3) ≈ 1,6.
/// Tolerancia: 4·SE + 1,6 ≈ 6,8. Al menos el 99 % de réplicas da Some, porque
/// P(media ≤ 0) = Φ(−3,2) ≈ 7e-4.
#[test]
fn cf2_lundberg_gaussiano_recupera_2mu_sobre_sigma2() {
    let (mu, sigma, n, reps) = (0.001_f64, 0.005_f64, 256usize, 400usize);
    let r_true = 2.0 * mu / (sigma * sigma);
    let mut rng = Lcg::new(0xC0F2_0002);
    let (mut s1, mut s2, mut validos) = (0.0f64, 0.0f64, 0usize);
    for _ in 0..reps {
        let mut est = EstimadorSiniestros::new();
        for _ in 0..n {
            est.observar(mu + sigma * rng.normal());
        }
        if let Some(r) = est.lundberg() {
            s1 += r;
            s2 += r * r;
            validos += 1;
        }
    }
    assert!(
        validos as f64 >= 0.99 * reps as f64,
        "sólo {validos}/{reps} réplicas con R"
    );
    let m = s1 / validos as f64;
    let se = ((s2 / validos as f64 - m * m).max(0.0) / validos as f64).sqrt();
    let sesgo = 2.5 * 2.0 * r_true / (n as f64 - 3.0);
    println!("media de R̂ = {m}, SE = {se}, réplicas con R = {validos}");
    assert!(
        (m - r_true).abs() <= 4.0 * se + sesgo,
        "media de R̂ = {m}, R = {r_true}, SE = {se}"
    );
}

/// Valor verdadero: R = ln(p/q)/a = 20,067 y cobertura nominal 0,95. Con
/// n = 256 la LCB cubre si y sólo si hay ≤ 153 victorias, así que la
/// cobertura esperada es P(Bin(256; 0,55) ≤ 153) ≈ 0,945. Umbral 0,920
/// (cobertura_minima), unos 4,8 SE por debajo.
///
/// Guarda de vacuidad: `cubre` cuenta None como cubierta, así que una LCB
/// que devolviera siempre None pasaría la cobertura. Hoy la cota es Some si y
/// sólo si hay ≥ 142 victorias: P(Bin(256; 0,55) ≥ 142) = 0,4658 exacto. Se
/// exige ≥ 0,42 (4 SE con 2 000 réplicas: 4·0,0112 = 0,045).
#[test]
fn cf2_lundberg_lcb_cubre_en_iid() {
    let (n, reps) = (256usize, 2_000usize);
    let r_true = r_reticular(P, A);
    let mut rng = Lcg::new(0xC0F2_0003);
    let (mut cubiertas, mut informativas) = (0usize, 0usize);
    for _ in 0..reps {
        let mut est = EstimadorSiniestros::new();
        for _ in 0..n {
            est.observar(if rng.uniform() < P { A } else { -A });
        }
        let lcb = est.lundberg_lcb(1.645);
        informativas += lcb.is_some() as usize;
        if cubre(lcb, r_true) {
            cubiertas += 1;
        }
    }
    let c = cubiertas as f64 / reps as f64;
    let f_some = informativas as f64 / reps as f64;
    println!("cobertura iid = {c}, fracción con cota = {f_some}");
    assert!(c >= cobertura_minima(reps), "cobertura = {c}");
    assert!(
        f_some >= 0.42,
        "sólo {f_some} de las réplicas dan cota (esperado 0,4658): la cobertura sería vacía"
    );
}

/// Caso extremo de F2.1-02: cada bloque repite el MISMO cierre b = 3 veces
/// (MAX_SPECTRAL_SLOTS = 3), 85 bloques y 255 cierres. El paseo en los finales
/// de bloque tiene pasos iid ±3a, así que R verdadero = ln(p/q)/(3a) = 6,69.
/// Se exige la misma cobertura que en iid (≥ 0,920). Hoy la LCB trata los 255
/// cierres como iid y sólo cubre si hay ≤ 48 bloques ganadores de 85:
/// cobertura ≈ 0,65. La cobertura cuenta None como cubierta, así que un
/// arreglo que retirara siempre la cota la pasaría; lo impiden las guardas de
/// vacuidad de los controles iid activos (cf2_lundberg_lcb_cubre_en_iid y
/// cf2_lundberg_bloques_gaussianos_independientes_cubren).
#[test]
#[ignore = "OPEN F2.1-02 (Línea C): R̂ y su LCB suponen cierres iid; con cierres dependientes R sale inflado por la longitud del bloque"]
fn cf2_lundberg_cierres_dependientes_inflan_r() {
    let (b, bloques, reps) = (3usize, 85usize, 2_000usize);
    let r_true = r_reticular(P, A) / b as f64;
    let mut rng = Lcg::new(0xC0F2_0004);
    let mut cubiertas = 0usize;
    for _ in 0..reps {
        let mut est = EstimadorSiniestros::new();
        for _ in 0..bloques {
            let y = if rng.uniform() < P { A } else { -A };
            for _ in 0..b {
                est.observar(y);
            }
        }
        if cubre(est.lundberg_lcb(1.645), r_true) {
            cubiertas += 1;
        }
    }
    let c = cubiertas as f64 / reps as f64;
    assert!(
        c >= cobertura_minima(reps),
        "cobertura = {c} (mínimo {}) con R verdadero {r_true}: la LCB iid sobrestima R",
        cobertura_minima(reps)
    );
}

/// Cobertura de la LCB con bloques gaussianos equicorrelacionados:
/// y_k = μ + σ(√ρ·Z_bloque + √(1−ρ)·Z_k), b = 3, 85 bloques (255 cierres).
/// El incremento por bloque es N(bμ, bσ²(1+(b−1)ρ)), así que la tasa de
/// decaimiento exacta es R = 2μ/(σ²(1+(b−1)ρ)). Devuelve (cobertura,
/// R verdadero, fracción de réplicas con cota Some).
fn cobertura_bloques_gaussianos(rho: f64, semilla: u64, reps: usize) -> (f64, f64, f64) {
    let (mu, sigma, b, bloques) = (0.001_f64, 0.005_f64, 3usize, 85usize);
    let r_true = 2.0 * mu / (sigma * sigma * (1.0 + (b as f64 - 1.0) * rho));
    let mut rng = Lcg::new(semilla);
    let (mut cubiertas, mut informativas) = (0usize, 0usize);
    for _ in 0..reps {
        let mut est = EstimadorSiniestros::new();
        for _ in 0..bloques {
            let comun = rng.normal();
            for _ in 0..b {
                let propio = rng.normal();
                est.observar(mu + sigma * (rho.sqrt() * comun + (1.0 - rho).sqrt() * propio));
            }
        }
        let lcb = est.lundberg_lcb(1.645);
        informativas += lcb.is_some() as usize;
        if cubre(lcb, r_true) {
            cubiertas += 1;
        }
    }
    let r = reps as f64;
    (cubiertas as f64 / r, r_true, informativas as f64 / r)
}

/// Control de la variante gaussiana (corrección del escéptico de F2.1): con
/// ρ = 0 los cierres son iid, R = 2μ/σ² = 80, y la LCB debe cubrir con el
/// mismo umbral que en iid (≥ 0,920 con 2 000 réplicas, nominal 0,95; la
/// cobertura esperada, medida con 20 000 réplicas fuera del repo, es 0,955).
/// Guarda de vacuidad (None cuenta como cubierta): la cota es Some cuando la
/// media es significativa, ≈ P(Z > 1,645 − 3,19) ≈ 0,94; medido fuera del
/// repo 0,937 ± 0,002. Se exige ≥ 0,91 (4 SE con 2 000 réplicas: 0,022).
#[test]
fn cf2_lundberg_bloques_gaussianos_independientes_cubren() {
    let reps = 2_000usize;
    let (c, r_true, f_some) = cobertura_bloques_gaussianos(0.0, 0xC0F2_0014, reps);
    println!("ρ = 0: cobertura = {c}, fracción con cota = {f_some}");
    assert!(
        c >= cobertura_minima(reps),
        "ρ = 0: cobertura = {c} con R verdadero {r_true}"
    );
    assert!(
        f_some >= 0.91,
        "ρ = 0: sólo {f_some} de las réplicas dan cota (esperado 0,937): la cobertura sería vacía"
    );
}

/// Dependencia parcial (corrección del escéptico de F2.1): con ρ = 0,5 el R
/// verdadero es 40 y con ρ = 1 es 26,7. La LCB debe cubrir ≥ 0,920 en los dos
/// casos. Hoy trata los 255 cierres como iid y estima R ≈ 80 con la SE iid:
/// cobertura ≈ 0,5 con ρ = 0,5 y menor con ρ = 1.
#[test]
#[ignore = "OPEN F2.1-02 (Línea C): R̂ y su LCB suponen cierres iid; con bloques equicorrelacionados la LCB sobrestima R"]
fn cf2_lundberg_bloques_gaussianos_equicorrelacionados_inflan_r() {
    let reps = 2_000usize;
    let (c05, r05, _) = cobertura_bloques_gaussianos(0.5, 0xC0F2_0015, reps);
    let (c10, r10, _) = cobertura_bloques_gaussianos(1.0, 0xC0F2_0016, reps);
    let minimo = cobertura_minima(reps);
    assert!(
        c05 >= minimo && c10 >= minimo,
        "cobertura con ρ = 0,5: {c05} (R = {r05}); con ρ = 1: {c10} (R = {r10}); mínimo {minimo}"
    );
}

/// Conversión de unidades con NOCIONAL FIJO N = f·C0 (f = 0,35, C0 = 13 USD):
/// la caída de capital D = f·a·s equivale a s pasos del paseo por nocional,
/// así que ψ_cap(D) = e^{−(R/f)·D} exacto en la retícula: 0,818; 0,548 y
/// 0,367 para s = 1, 3, 5. Tolerancia 4·SE + 1e-5 (SE binomial con 20 000
/// paseos; el sesgo de absorber en +50 es ≤ 1,1e-5).
///
/// Alcance: como D = f·a·s, f se cancela y ψ_cap(D) = e^{−R·a·s} es la misma
/// retícula de cf2_lundberg_paseo_reticular_cota_exacta con barrera en −5. El
/// único valor que aporta el código es R̂, ya fijado allí; esta prueba
/// documenta la teoría de la conversión y la holgura de ÷L_max, pero no ejerce
/// código nuevo. La integración con el veto vivo (corrección (a) del
/// escéptico 1 de F2.1) queda pendiente para C-F4.
///
/// La conversión viva ÷L_max (L = 20) da ψ_L = ψ^{f/L}: con f ≤ L es una cota
/// VÁLIDA pero LAXA (exponente f/L). Se comprueba que no queda por debajo de
/// lo observado (validez, 4 SE) y que queda a más de 10 SE por encima
/// (holgura). El sistema compone (nocional proporcional al capital); este
/// test supone nocional fijo y no ejerce el consumidor vivo (lib.rs:670-674).
/// Con esta semilla la caída s = 5 queda a +3,4 SE por azar (0,378 frente a
/// 0,367); el mismo generador sobre 300 semillas da z medio 0,02 y sd 0,96, y
/// el paseo sólo usa aritmética entera e IEEE sin libm, así que el resultado
/// es idéntico en gnu y msvc.
#[test]
fn cf2_lundberg_exposicion_real_convierte_r() {
    let r_n = muestra_exacta().lundberg().expect("R por nocional");
    let (c0, f, l_max) = (13.0_f64, 0.35_f64, 20.0_f64);
    let nocional = f * c0;
    let caidas = [1usize, 3, 5];
    let caminatas = 20_000usize;
    let mut rng = Lcg::new(0xC0F2_0005);
    let mut cruces = [0usize; 3];
    for _ in 0..caminatas {
        let (mut cap, mut min_cap, mut neto) = (c0, c0, 0i64);
        while neto < 50 && neto > -5 {
            let y = if rng.uniform() < P { A } else { -A };
            cap += nocional * y;
            neto += if y > 0.0 { 1 } else { -1 };
            min_cap = min_cap.min(cap);
        }
        for (k, &s) in caidas.iter().enumerate() {
            if (c0 - min_cap) / c0 >= f * A * s as f64 - 1e-12 {
                cruces[k] += 1;
            }
        }
    }
    for (k, &s) in caidas.iter().enumerate() {
        let d = f * A * s as f64;
        let emp = cruces[k] as f64 / caminatas as f64;
        let psi_nc = (-(r_n / f) * d).exp(); // R_capital = R_nocional·C/N
        let psi_lmax = (-(r_n / l_max) * d).exp(); // conversión viva ÷L_max
        let se = (psi_nc * (1.0 - psi_nc) / caminatas as f64).sqrt();
        println!("D = {d}: frecuencia {emp}, ψ_f = {psi_nc}, ψ_L = {psi_lmax}");
        assert!(
            (emp - psi_nc).abs() <= 4.0 * se + 1e-5,
            "D={d}: frecuencia {emp} frente a ψ = e^(-(R/f)·D) = {psi_nc}"
        );
        assert!(
            psi_lmax + 4.0 * se >= emp,
            "D={d}: ÷L_max promete {psi_lmax} y la ruina observada es {emp}: cota inválida"
        );
        assert!(
            psi_lmax - emp > 10.0 * se,
            "D={d}: ÷L_max es válida pero laxa (exponente f/L): promete {psi_lmax}, observado {emp}"
        );
    }
}

/// Valor verdadero: ψ(m) = e^{−R·m} decrece con R, así que un tope de ruina
/// debe admitir un conjunto no decreciente en R: para R1 < R2,
/// veto(R2) ⇒ veto(R1). Comparación booleana exacta, sin tolerancia. Riesgos
/// [0,04; 0,04; 0,04], ρ = 0,2 (riesgo de grupo 0,04·√4,2 = 0,0820), tope 0,10
/// y ε = 0,05. Hoy: R = 15 admite (m = 0,1997) y R = 40 veta (m = 0,0749).
#[test]
#[ignore = "OPEN F2.1-01 (Codex): el veto Lundberg compara el riesgo al stop con un margen ln(1/ε)/R; un R mayor veta lo que uno menor admite"]
fn cf2_lundberg_veto_monotono_en_r() {
    let riesgos = [0.04_f64, 0.04, 0.04];
    let (rho, tope, eps) = (Some(0.2), 0.10, 0.05);
    let rs = [1.0_f64, 5.0, 15.0, 40.0, 100.0, 1_000.0];
    for w in rs.windows(2) {
        let veto_bajo = veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope, Some(w[0]), eps);
        let veto_alto = veto_por_riesgo_cramer_lundberg(&riesgos, rho, tope, Some(w[1]), eps);
        assert!(
            !veto_alto || veto_bajo,
            "R = {} veta y R = {} admite: más edge medido no puede ser más ruina",
            w[1],
            w[0]
        );
    }
}

/// Valor verdadero: un flujo antisimétrico con aristas iid N(0, 1) sobre K_8
/// es isótropo en R^D con D = 28; su fracción de gradiente (subespacio de
/// dimensión d = 7) es Beta(d/2, (D−d)/2). E[curl] = 1 − 2/n = 0,75 y
/// Var = 0,25·0,75/15 = 0,0125. SE de la media con 4 000 réplicas = 0,00177;
/// tolerancia 4·SE = 0,0071. Es el nulo isótropo DENSO, no el del campo vivo
/// censurado (z ≥ 3, ventanas distintas por moneda): documenta que el valor
/// nulo del curl con 8 monedas es 0,75, no 0.
#[test]
fn cf2_hodge_nulo_isotropo_da_uno_menos_dos_sobre_n() {
    let (n, reps) = (8usize, 4_000usize);
    let mut rng = Lcg::new(0xC0F2_0007);
    let mut suma = 0.0;
    for _ in 0..reps {
        let mut f = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let g = rng.normal();
                f[i][j] = g;
                f[j][i] = -g;
            }
        }
        suma += hodge_curl_share(&f).expect("flujo denso");
    }
    let media = suma / reps as f64;
    let (d, dim) = ((n - 1) as f64, (n * (n - 1) / 2) as f64);
    let esperado = 1.0 - d / dim;
    let var = (d / dim) * (1.0 - d / dim) / (dim / 2.0 + 1.0);
    let se = (var / reps as f64).sqrt();
    println!("media del curl = {media}, esperado {esperado}");
    assert!(
        (media - esperado).abs() <= 4.0 * se,
        "media del curl = {media}, esperado {esperado}, 4 SE = {}",
        4.0 * se
    );
}

/// Codificación de la ausencia (corrección del escéptico de F2.1): el par
/// crudo (z_ij, z_ji) = (0, 0) significa «no observado», que es lo que emite
/// hawkes_cross cuando z < 3 (sus z son ≥ 3 o exactamente 0). Aquí A→B y
/// B→C tienen z = 4 y el par A–C es (0, 0). En el grafo observado (el camino
/// A–B–C, un árbol) el espacio de ciclos es vacío: curl = 0, tolerancia 1e-9
/// (aritmética exacta). Hoy se proyecta sobre K_3 con f_AC = 0 medido:
/// div = (2, 0, −2), energía del gradiente 8/3 de 8, curl = 2/3.
#[test]
#[ignore = "OPEN F2.1-03 (Línea C): hodge_curl_share proyecta sobre K_r y cuenta el par censurado (0, 0) como flujo cero medido"]
fn cf2_hodge_cadena_censurada_no_es_ciclo() {
    let f = vec![
        vec![0.0, 4.0, 0.0],
        vec![0.0, 0.0, 4.0],
        vec![0.0, 0.0, 0.0],
    ];
    let curl = hodge_curl_share(&f).expect("tres nodos con flujo");
    assert!(
        curl < 1e-9,
        "curl = {curl}: la arista no observada A–C cuenta como flujo cero medido"
    );
}

/// Contraprueba de la anterior (corrección del escéptico de F2.1): si el par
/// A–C SÍ se midió y está equilibrado (z = 5 en ambos sentidos), el flujo
/// neto 0 es una medición y el grafo observado es K_3. La proyección sobre
/// K_3 da exactamente curl = 2/3 (div = (2, 0, −2), energía 8, gradiente 8/3).
/// Tolerancia 1e-12 (aritmética exacta). Un arreglo de F2.1-03 que trate todo
/// 0 neto como arista ausente rompería esta prueba.
#[test]
fn cf2_hodge_par_medido_equilibrado_sigue_en_k3() {
    let f = vec![
        vec![0.0, 4.0, 5.0],
        vec![0.0, 0.0, 4.0],
        vec![5.0, 0.0, 0.0],
    ];
    let curl = hodge_curl_share(&f).expect("tres nodos con flujo");
    assert!(
        (curl - 2.0 / 3.0).abs() < 1e-12,
        "curl = {curl}: con A–C medido y equilibrado el grafo es K_3 (2/3)"
    );
}

/// Nivel del veredicto en ruido iid (N = 8, T = 512, el régimen de los tests
/// xli_c1; sin correlación poblacional, así que todo SystematicMode es falso).
/// Propiedad que necesitaría cualquier consumidor que usara el veredicto como
/// prueba: tamaño ≤ 0,05 + 4·SE nominal (SE = √(0,05·0,95/2 000) = 0,0049;
/// umbral 0,0695).
///
/// La predicción de F2.1-06 (≈ 0,17 por Tracy-Widom con el centrado de
/// Johnstone) NO se cumple: la entrada es una matriz de CORRELACIÓN, cuya
/// traza vale N exactamente, y eso quita la fluctuación de escala que el
/// límite TW de la covarianza sí incluye. Medido aquí: ≈ 0,019. Contraste
/// independiente (numpy, fuera del repo): correlación 0,021 (8 000 réplicas);
/// covarianza con escala conocida 0,12. Con N creciente el tamaño sube
/// despacio hacia el límite TW (0,032–0,037 con N = 16, T = 512; 0,045 con
/// N = 40, T = 160; 0,062 con N = 100, T = 400), así que el borde no es una
/// prueba calibrada del 5 %, pero en el régimen del roster (N ≤ 16) es
/// conservador. La prueba fija ese régimen; si alguien la cambia a N grande
/// debe volver a medir.
///
/// Guarda de vacuidad: un veredicto que diera siempre AllNoise pasaría la cota
/// superior. Referencia independiente (numpy, fuera del repo, 40 000 réplicas
/// con la misma correlación de Pearson): 0,0202 ± 0,0007. Se exige
/// tamaño ≥ 0,0075 (4 SE con 2 000 réplicas: 4·0,0031 = 0,0125).
#[test]
fn cf2_mp_tamano_del_veredicto_en_ruido() {
    let (n, t, reps) = (8usize, 512usize, 2_000usize);
    let mut rng = Lcg::new(0xC0F2_000B);
    let mut sistematicos = 0usize;
    for _ in 0..reps {
        let c = correlacion(&gauss(&mut rng, n, t));
        match systematic_mode(&c, t) {
            Some(MppVerdict::SystematicMode { .. }) => sistematicos += 1,
            Some(MppVerdict::AllNoise) => {}
            None => panic!("correlación válida rechazada"),
        }
    }
    let tamano = sistematicos as f64 / reps as f64;
    let se = (0.05_f64 * 0.95 / reps as f64).sqrt();
    println!("tamaño en ruido iid con N = {n}, T = {t}: {tamano}");
    assert!(
        tamano <= 0.05 + 4.0 * se,
        "tamaño en ruido iid = {tamano}: el veredicto rechaza ruido por encima del 5 %"
    );
    assert!(
        tamano >= 0.0075,
        "tamaño en ruido iid = {tamano}, referencia 0,0202: el veredicto no dispara nunca"
    );
}
