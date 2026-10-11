//! C-F2.3b (Línea C): el «Hurst» por cociente de varianzas de
//! `multifractal.rs` frente a un fGn conocido, y el contrato de
//! `cache_generation`.
//!
//! Teoría que se prueba:
//! - Para un fGn con H = 0,7, Var(r^{(k)}) = k^{2H}·Var(r^{(1)}), así que un
//!   estimador de Hurst sobre la ventana macro de 50 retornos debería dar, en
//!   media de muchas realizaciones, 0,7 ± 0,05. El motor encoge hacia 0,5 con
//!   peso w = n/(n + 12) = 50/62 (techo 0,661 con datos perfectos) y la resta
//!   de la media muestral baja los cocientes de varianzas, así que se
//!   predice ≈ 0,62-0,64.
//! - Según su doc, `cache_generation` cambia exactamente cuando el cache del
//!   espectro f(α) se refresca, una vez cada 16 consultas: en 32 consultas
//!   seguidas debe haber como mucho 2 generaciones distintas.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use feature_engine::multifractal::MultiScaleHurstConfluence;

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

/// Verdad: H = 0,7. Tolerancia 0,05 sobre la media de 400 realizaciones de
/// 50 retornos (justo la ventana macro). La sd por realización medida es
/// ≈ 0,073, así que la SE de la media es ≈ 0,0037 y la banda son 13 SE.
/// Medido hoy: 0,620, a ≈ 21 SE de 0,7 (el techo del encogimiento es 0,661).
#[test]
#[ignore = "OPEN F2.3b-09 (Línea C): el cociente de varianzas macro está encogido hacia 0,5 (w = 50/62) y sesgado por la media muestral: con H = 0,7 da ≈ 0,62-0,64"]
fn cf2_vr_macro_fgn_07_dentro_de_005() {
    let (phis, sd) = hosking(0.7, 50);
    let mut rng = Lcg(0x0F6A_0070);
    let m = 400;
    let (mut s_micro, mut s_macro, mut s2_macro) = (0.0, 0.0, 0.0);
    for _ in 0..m {
        let x = fgn(&phis, &sd, &mut rng);
        let mut c = MultiScaleHurstConfluence::new();
        let mut lnp = 100f64.ln();
        c.update(lnp.exp());
        let mut ult = (0.5, 0.5, 0.5, 0.0, false, false);
        for &xi in &x {
            lnp += 1e-3 * xi;
            ult = c.update(lnp.exp());
        }
        s_micro += ult.0;
        s_macro += ult.2;
        s2_macro += ult.2 * ult.2;
    }
    let (micro, macro_) = (s_micro / m as f64, s_macro / m as f64);
    let sd_macro = ((s2_macro / m as f64 - macro_ * macro_) * m as f64 / (m - 1) as f64).sqrt();
    let se = sd_macro / (m as f64).sqrt();
    assert!(
        (macro_ - 0.7).abs() < 0.05,
        "macro {macro_:.4} (sd {sd_macro:.4}, se {se:.4}), micro {micro:.4} con H = 0,7"
    );
}

/// Verdad (doc de `cache_generation`): la generación cambia sólo al
/// refrescar el cache, una vez cada 16 consultas. Determinista: con 60
/// precios el motor macro está maduro (≥ 32 retornos) y 32 consultas deben
/// dar como mucho 2 generaciones distintas. Hoy salen 32 (de 1 a 32).
#[test]
#[ignore = "OPEN F2.3b-10 (Línea C): cache_generation devuelve el contador de consultas, que cambia en cada consulta, y la deduplicación de D0 nunca actúa"]
fn cf2_generacion_cache_cambia_una_vez_cada_16() {
    let mut c = MultiScaleHurstConfluence::new();
    let mut rng = Lcg(7);
    let mut lnp = 100f64.ln();
    for _ in 0..60 {
        lnp += 1e-3 * rng.n();
        c.update(lnp.exp());
    }
    let mut gens = Vec::new();
    let mut espectros = 0usize;
    for _ in 0..32 {
        if c.espectro_cacheada().is_some() {
            espectros += 1;
        }
        gens.push(c.cache_generation());
    }
    assert_eq!(espectros, 32, "el motor macro debe estar maduro");
    let primera = gens[0];
    let ultima = gens[31];
    gens.dedup();
    assert!(
        gens.len() <= 2,
        "{} generaciones distintas en 32 consultas (de {primera} a {ultima})",
        gens.len()
    );
}
