//! CL-70 (C-F4, LC-F6-01): pruebas de solución conocida del estimador σ de
//! 60 s (`god_engine_core::sigma_minuto::SigmaMinuto`).
//!
//! Verdades:
//! - Deriva pura de δ por segundo: r60 = 60·δ en todos los segundos, así que
//!   σ̂60 = 60·δ exacto (el estimador no centra: mide E[r60²]).
//! - Browniano con σ1 por segundo y eventos de Poisson a cualquier ritmo:
//!   E[r60²] = 60·σ1² (por estacionariedad, el intervalo entre los últimos
//!   eventos de los segundos g y g − 60 mide 60 s en esperanza).
//! - AR(1) de retornos por segundo con coeficiente φ e innovación σe:
//!   Var(Σ₆₀) = σe²/(1 − φ²)·[60 + 2·Σ_{k=1}^{59} (60 − k)·φ^k].
//! - Huecos de 600 s: la ponderación por duración deja la media sin sesgo.
//! - Calentamiento por klines de 1 min con el mismo browniano: misma
//!   cantidad.
//!
//! La comparación es sobre la media temporal de la varianza publicada tras
//! cada vela, con su error estándar por medias de bloques (la EWMA
//! correlaciona velas vecinas). Tolerancia: 4 EE + 0,5 % de margen numérico.

use god_engine_core::sigma_minuto::SigmaMinuto;

/// LCG de 64 bits (Knuth/MMIX), uniforme en (0, 1) y normal por Box-Muller.
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Media y EE (medias de 20 bloques) de una serie.
fn media_ee(xs: &[f64]) -> (f64, f64) {
    let n = xs.len();
    let m = xs.iter().sum::<f64>() / n as f64;
    let b = 20usize;
    let tam = n / b;
    let medias: Vec<f64> = (0..b)
        .map(|k| xs[k * tam..(k + 1) * tam].iter().sum::<f64>() / tam as f64)
        .collect();
    let mb = medias.iter().sum::<f64>() / b as f64;
    let var = medias.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (b - 1) as f64;
    (m, (var / b as f64).sqrt())
}

/// Simula `minutos` minutos de un log-precio con incrementos por segundo
/// `incremento(rng)`, observado en eventos de Poisson de `tasa` por segundo
/// (o uno por minuto exacto si `tasa` es 0), cierra una vela cada 60 s de
/// marca y devuelve las varianzas publicadas tras cada cierre (desde el
/// minuto 30).
fn simular(
    semilla: u64,
    minutos: u64,
    tasa: f64,
    mut incremento: impl FnMut(&mut Lcg) -> f64,
) -> Vec<f64> {
    let mut rng = Lcg(semilla);
    let mut est = SigmaMinuto::new();
    let mut x = 100f64.ln();
    let mut salida = Vec::new();
    // Incrementos por milisegundo agregados en pasos de 1 s: el log-precio
    // en t es x(⌊t⌋) + fracción del incremento del segundo en curso. Para
    // un browniano basta con muestrear el valor en el instante del evento:
    // se simula el incremento de cada segundo en 10 sub-pasos.
    let sub = 10u64;
    let siguiente = |rng: &mut Lcg, t: f64| -> f64 {
        if tasa > 0.0 {
            t - (rng.u().ln()) / tasa
        } else {
            t + 60.0
        }
    };
    let mut t_evento = siguiente(&mut rng, 0.0);
    let mut t = 0.0f64;
    for minuto in 0..minutos {
        for _s in 0..60 {
            let inc = incremento(&mut rng);
            for _ in 0..sub {
                let t_fin = t + 1.0 / sub as f64;
                while t_evento < t_fin {
                    est.observar((t_evento * 1000.0) as u64, x.exp());
                    t_evento = siguiente(&mut rng, t_evento);
                }
                x += inc / sub as f64;
                t = t_fin;
            }
        }
        est.cerrar_vela();
        if minuto >= 30 {
            if let Some(v) = est.varianza_60s() {
                salida.push(v);
            }
        }
    }
    salida
}

#[test]
fn cl70_deriva_pura_da_sesenta_delta_exacto() {
    let delta = 1e-5f64;
    let mut est = SigmaMinuto::new();
    let mut x = 100f64.ln();
    for s in 0..600u64 {
        est.observar(s * 1000 + 500, x.exp());
        x += delta;
        if s % 60 == 59 {
            est.cerrar_vela();
        }
    }
    let sigma = est.sigma_60s().expect("con 10 velas hay σ̂");
    assert!(
        (sigma / (60.0 * delta) - 1.0).abs() < 1e-9,
        "σ̂60 = {sigma:e} frente a 60·δ = {:e}",
        60.0 * delta
    );
}

/// Verdad: 60·σ1² con σ1 = 2e-4 por segundo, a 0,5, 1 y 10 eventos/s y a un
/// evento por minuto. 30 000 minutos por caso (EE ≈ 0,6 %).
#[test]
fn cl70_browniano_misma_varianza_con_cualquier_densidad() {
    let sigma1 = 2e-4f64;
    let verdad = 60.0 * sigma1 * sigma1;
    for (i, tasa) in [0.5f64, 1.0, 10.0, 0.0].into_iter().enumerate() {
        let vs = simular(0xC170_0000 + i as u64, 30_000, tasa, |r| sigma1 * r.n());
        let (m, ee) = media_ee(&vs);
        println!(
            "tasa {tasa}: media {:.5} (EE {:.5}) de la verdad",
            m / verdad,
            ee / verdad
        );
        assert!(
            (m / verdad - 1.0).abs() < 4.0 * ee / verdad + 0.005,
            "tasa {tasa}: varianza media {m:e} frente a 60·σ1² = {verdad:e} (EE {ee:e})"
        );
    }
}

/// Verdad: Var(Σ₆₀) de un AR(1) por segundo con φ = ±0,3. Un ATR que
/// reescala σ por segundo con √60 se equivocaría en (1 + φ)/(1 − φ).
#[test]
fn cl70_ar1_mide_la_varianza_de_sesenta_segundos() {
    let se = 2e-4f64;
    for (i, phi) in [0.3f64, -0.3].into_iter().enumerate() {
        let mut suma = 60.0f64;
        for k in 1..60 {
            suma += 2.0 * (60 - k) as f64 * phi.powi(k as i32);
        }
        let verdad = se * se / (1.0 - phi * phi) * suma;
        let mut previo = 0.0f64;
        let vs = simular(0xC170_1000 + i as u64, 30_000, 1.0, |r| {
            previo = phi * previo + se * r.n();
            previo
        });
        let (m, ee) = media_ee(&vs);
        println!(
            "φ = {phi}: media {:.5} (EE {:.5}) de la verdad",
            m / verdad,
            ee / verdad
        );
        assert!(
            (m / verdad - 1.0).abs() < 4.0 * ee / verdad + 0.005,
            "φ = {phi}: varianza media {m:e} frente a {verdad:e} (EE {ee:e})"
        );
    }
}

/// Verdad: 60·σ1² con huecos de 600 s cada diez velas. El tiempo de mercado
/// sigue corriendo en el hueco (el browniano no se detiene), pero no llega
/// ningún evento; la vela del hueco cierra con el primer evento posterior,
/// como en el núcleo. Sin la ponderación por duración el sesgo sería ≈ +70 %.
///
/// La media es TEMPORAL: cada varianza publicada pesa el tiempo que rige
/// (la duración de la vela siguiente), que es como la leen los consumidores.
/// Promediar una vez por vela sobrepondera el valor alto que deja el salto
/// tras el hueco (rige 9 velas) frente al bajo de la vela del hueco, que
/// representa 600 s (medido así: +27 % con una EWMA de medias por vela).
///
/// CL-86 (atr-1): desde que un hueco de más de 61 s es dato ausente (sin
/// r60 nulos ni decaimiento), la vela del hueco ya no baja la varianza y la
/// media sale sin sesgo también justo después del hueco
/// (`cf4_sigma_hueco.rs`). Esta prueba conserva la verdad de la media
/// temporal.
#[test]
fn cl70_huecos_de_seiscientos_segundos_sin_sesgo() {
    let sigma1 = 2e-4f64;
    let verdad = 60.0 * sigma1 * sigma1;
    let mut rng = Lcg(0xC170_2000);
    let mut est = SigmaMinuto::new();
    let mut x = 100f64.ln();
    let mut t_s = 0u64;
    // (varianza publicada tras la vela, duración de la vela siguiente)
    let mut publicadas: Vec<(f64, f64)> = Vec::new();
    let duracion_de = |vela: u64| if vela % 10 == 9 { 600u64 } else { 60 };
    for vela in 0..6_000u64 {
        let duracion = duracion_de(vela);
        let hueco = duracion == 600;
        for s in 0..duracion {
            x += sigma1 * rng.n();
            if !hueco || s == duracion - 1 {
                est.observar((t_s + s) * 1000 + 999, x.exp());
            }
        }
        t_s += duracion;
        est.cerrar_vela();
        if vela >= 300 {
            if let Some(v) = est.varianza_60s() {
                publicadas.push((v, duracion_de(vela + 1) as f64));
            }
        }
    }
    // Media temporal y su EE por medias de 20 bloques (ponderadas).
    let peso_total: f64 = publicadas.iter().map(|p| p.1).sum();
    let m = publicadas.iter().map(|p| p.0 * p.1).sum::<f64>() / peso_total;
    let tam = publicadas.len() / 20;
    let medias: Vec<f64> = (0..20)
        .map(|k| {
            let b = &publicadas[k * tam..(k + 1) * tam];
            b.iter().map(|p| p.0 * p.1).sum::<f64>() / b.iter().map(|p| p.1).sum::<f64>()
        })
        .collect();
    let mb = medias.iter().sum::<f64>() / 20.0;
    let ee = (medias.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / 19.0 / 20.0).sqrt();
    println!(
        "huecos: media temporal {:.5} (EE {:.5}) de la verdad",
        m / verdad,
        ee / verdad
    );
    assert!(
        (m / verdad - 1.0).abs() < 4.0 * ee / verdad + 0.005,
        "con huecos: varianza media {m:e} frente a {verdad:e} (EE {ee:e})"
    );
}

/// Verdad: el calentamiento por klines de 1 min (r60 entre cierres) y el
/// camino vivo estiman la misma varianza del mismo browniano.
#[test]
fn cl70_calentamiento_y_vivo_miden_lo_mismo() {
    let sigma1 = 2e-4f64;
    let verdad = 60.0 * sigma1 * sigma1;
    let mut rng = Lcg(0xC170_3000);
    let mut caliente = SigmaMinuto::new();
    let mut x = 100f64.ln();
    let mut vs = Vec::new();
    for minuto in 0..30_000u64 {
        for _ in 0..60 {
            x += sigma1 * rng.n();
        }
        caliente.observar_cierre_kline(x.exp());
        if minuto >= 30 {
            if let Some(v) = caliente.varianza_60s() {
                vs.push(v);
            }
        }
    }
    let (m, ee) = media_ee(&vs);
    println!(
        "calentamiento: media {:.5} (EE {:.5}) de la verdad",
        m / verdad,
        ee / verdad
    );
    assert!(
        (m / verdad - 1.0).abs() < 4.0 * ee / verdad + 0.005,
        "calentamiento: varianza media {m:e} frente a {verdad:e} (EE {ee:e})"
    );
}

/// Verdad: el estimador es invariante a la escala del precio (mide
/// log-retornos): ×1000 en todos los precios cambia σ̂ menos de 1e-9
/// relativo (redondeo de ln).
#[test]
fn cl70_invariante_a_la_escala_del_precio() {
    let mut rng = Lcg(0xC170_4000);
    let (mut a, mut b) = (SigmaMinuto::new(), SigmaMinuto::new());
    let mut x = 100f64.ln();
    for s in 0..1_200u64 {
        x += 2e-4 * rng.n();
        a.observar(s * 1000 + 250, x.exp());
        b.observar(s * 1000 + 250, 1000.0 * x.exp());
        if s % 60 == 59 {
            a.cerrar_vela();
            b.cerrar_vela();
        }
    }
    let (sa, sb) = (a.sigma_60s().unwrap(), b.sigma_60s().unwrap());
    assert!((sa / sb - 1.0).abs() < 1e-9, "σ̂ {sa:e} frente a {sb:e}");
}

/// Robustez: sin eventos no hay σ̂; precios no válidos y marcas que
/// retroceden se ignoran; una vela sin r60 no mueve la varianza.
#[test]
fn cl70_entradas_no_validas_y_sin_datos() {
    let mut e = SigmaMinuto::new();
    e.cerrar_vela();
    assert!(e.sigma_60s().is_none(), "sin datos no hay σ̂");
    for s in 0..130u64 {
        e.observar(s * 1000, 100.0 * (1.0 + 1e-4 * (s % 3) as f64));
        e.observar(s * 1000 + 1, f64::NAN);
        e.observar(s * 1000 + 2, -5.0);
    }
    e.observar(10, 1e9); // retrocede: se ignora
    e.cerrar_vela();
    let v = e.varianza_60s().expect("con 130 s hay r60");
    assert!(v.is_finite() && v >= 0.0 && v < 1e-6, "varianza {v:e}");
    e.cerrar_vela(); // sin r60 nuevos
    assert_eq!(e.varianza_60s(), Some(v));
}
