//! C-F3 / F3.6-05 (Línea C): la «resonancia estocástica» del motor debe
//! tener un máximo interior frente al ruido.
//!
//! Teoría que se prueba: la resonancia estocástica (Benzi, Gammaitoni et
//! al.) es el fenómeno por el que una señal débil, por debajo del umbral de
//! un sistema no lineal con umbral o barrera (pozo bi-estable), se transmite
//! MEJOR con un nivel intermedio de ruido: la correlación entrada/salida (o
//! la SNR a la frecuencia de la señal) frente a σ tiene un máximo INTERIOR.
//! El motor dice modelar ese pozo bi-estable, pero su ganancia es
//! g(s) = s·(1 + sech(ln(|s|/σ))): impar, monótona (g'(s) = 1 +
//! 4σ³|s|/(s² + σ²)² > 0), sin memoria y sin umbral. Para un sistema así el
//! ruido sólo degrada: el máximo está en el borde de menor σ.
//!
//! Solución conocida: senoide débil A = 0,01, periodo 50, n = 20 000
//! muestras y ruido gaussiano σ ∈ {0,0025; 0,005; 0,01; 0,02; 0,04; 0,08}
//! (de A/4 a 8A), con el motor alimentado con la varianza verdadera σ². Se
//! exige 0 < imax < 5. El error de la correlación de Pearson con n = 20 000
//! es ≈ (1 − ρ²)/√n ≤ 0,007, muy por debajo de las diferencias entre σ
//! vecinas (≥ 0,08 en la simulación del borrador), así que el argmax es
//! estable frente a la semilla.
//!
//! Convención: un test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use signal_engine::stochastic_resonance::StochasticResonanceEngine as Sr;
use std::f64::consts::PI;

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Normal estándar por Box-Muller.
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
}

#[test]
#[ignore = "OPEN F3.6-05 (Línea C, stochastic_resonance): la ganancia 1 + sech(ln(|s|/σ)) es impar, monótona, sin umbral ni memoria; el ruido sólo degrada y el máximo de corr(salida, señal) está en el σ menor (imax = 0)"]
fn cf3_6_resonancia_estocastica_tiene_maximo_interior() {
    let (a, periodo, n) = (0.01, 50.0, 20_000usize);
    let sigmas = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08];
    let mut corr = Vec::new();
    for &sigma in &sigmas {
        let mut rng = Lcg(0x5EED_F360);
        let (mut s, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for t in 0..n {
            let limpia = a * (2.0 * PI * t as f64 / periodo).sin();
            let x = limpia + sigma * rng.normal();
            s.push(limpia);
            y.push(Sr::amplify_signal_with_noise(x, sigma * sigma));
        }
        corr.push(pearson(&s, &y));
    }
    let imax = (0..corr.len())
        .max_by(|&i, &j| corr[i].total_cmp(&corr[j]))
        .unwrap();
    println!("observado: corr(σ) = {corr:?}, imax = {imax}");
    assert!(
        imax > 0 && imax + 1 < sigmas.len(),
        "sin máximo interior: corr(sigma) = {:?}",
        corr
    );
}
