//! C-F2.3b (Línea C): bandas de Hurst, modulación de la duración y reloj de
//! 1 minuto del Hurst vivo, frente al nulo browniano.
//!
//! Teoría que se prueba:
//! - Un paseo aleatorio (y cualquier martingala, p. ej. GARCH(1,1)) tiene
//!   H = 0,5 exacto: un clasificador de «persistente» (H > HURST_PERSISTENT)
//!   o «antipersistente» (H < HURST_ANTI_PERSISTENT) con tamaño nominal del
//!   5 % no debe superar el 10 % de falsos positivos en ninguno de los dos
//!   lados.
//! - Sin memoria, `hurst_duration_modulation` no debe cambiar la duración
//!   esperada (factor 1) ni saturar a ×2 o ×½.
//! - El Hurst vivo se muestrea al cierre de la vela interna de 1 minuto; una
//!   rejilla física fija de 1 minuto da un retorno por minuto cerrado
//!   (contrato del reloj, no solución estadística).
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use god_engine_core::math_kernels::RecursiveHurst;
use god_engine_core::stateful_engine::StatefulEngine;
use god_engine_core::{HURST_ANTI_PERSISTENT, HURST_PERSISTENT, hurst_duration_modulation};

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

/// Número de paseos por prueba. Con 400, el error estándar de una fracción
/// de tamaño nominal 0,05 es √(0,05·0,95/400) ≈ 0,0109, así que el umbral
/// 0,10 queda a 4,6 SE del nominal.
const PASEOS: usize = 400;

/// H vivo (`RecursiveHurst::current`, DFA con R² ≥ 0,85) de `m` paseos de
/// 1024 retornos: la misma cantidad de datos que el DFA vivo tras calentar
/// (1024 velas de 1 minuto). `garch` = None da retornos gaussianos de 1e-3;
/// Some((α, β)) da GARCH(1,1) con varianza incondicional (1e-3)².
fn hurst_paseos(m: usize, semilla: u64, garch: Option<(f64, f64)>) -> Vec<f64> {
    let mut rng = Lcg(semilla);
    let var0 = 1e-6;
    (0..m)
        .map(|_| {
            let mut h = RecursiveHurst::new();
            let mut lnp = 100f64.ln();
            h.update(lnp.exp());
            let mut var = var0;
            let mut r_prev = 0.0f64;
            for _ in 0..1024 {
                let r = match garch {
                    None => 1e-3 * rng.n(),
                    Some((a, b)) => {
                        var = var0 * (1.0 - a - b) + a * r_prev * r_prev + b * var;
                        var.sqrt() * rng.n()
                    }
                };
                r_prev = r;
                lnp += r;
                h.update(lnp.exp());
            }
            assert_eq!(h.samples(), 1024);
            h.current()
        })
        .collect()
}

fn fracciones(hs: &[f64]) -> (f64, f64, f64, f64) {
    let m = hs.len() as f64;
    let p = hs.iter().filter(|&&h| h > HURST_PERSISTENT).count() as f64 / m;
    let a = hs.iter().filter(|&&h| h < HURST_ANTI_PERSISTENT).count() as f64 / m;
    let media = hs.iter().sum::<f64>() / m;
    let sd = (hs.iter().map(|h| (h - media) * (h - media)).sum::<f64>() / (m - 1.0)).sqrt();
    (p, a, media, sd)
}

/// Verdad: H = 0,5. Fracción con H > 0,52 ≤ 0,10 y, en espejo (corrección
/// del escéptico de física), fracción con H < 0,45 ≤ 0,10, para que un
/// arreglo no pase moviendo el error al otro lado. Umbral a 4,6 SE del
/// tamaño nominal del 5 % con 400 paseos. Hoy la media nula del DFA es
/// ≈ 0,52 con SD ≈ 0,035, así que la fracción persistente ronda el 50 %.
#[test]
#[ignore = "OPEN F2.3b-01 (Línea C): el DFA vivo da H ≈ 0,52 en un paseo aleatorio y la banda 0,52 lo clasifica persistente en torno a la mitad de las veces"]
fn cf2_paseo_aleatorio_no_se_clasifica_persistente() {
    let hs = hurst_paseos(PASEOS, 0xB0A5_2026, None);
    let (p, a, media, sd) = fracciones(&hs);
    println!(
        "observado: persistente {p:.3}, antipersistente {a:.3}, H medio {media:.4}, SD {sd:.4}"
    );
    assert!(
        p <= 0.10 && a <= 0.10,
        "paseos aleatorios: persistente {p:.3}, antipersistente {a:.3} (H medio {media:.4}, SD {sd:.4})"
    );
}

/// Variante del escéptico de física: retornos GARCH(1,1) con α = 0,10 y
/// β = 0,85 (colas pesadas y agrupación de volatilidad, pero sin memoria
/// lineal: martingala, H = 0,5 exacto). Mismo contrato y misma banda.
#[test]
#[ignore = "OPEN F2.3b-01 (Línea C): con retornos GARCH(1,1) sin memoria lineal el DFA vivo también clasifica persistente en torno a la mitad de las veces"]
fn cf2_paseo_garch_no_se_clasifica_persistente() {
    let hs = hurst_paseos(PASEOS, 0xB0A5_2028, Some((0.10, 0.85)));
    let (p, a, media, sd) = fracciones(&hs);
    println!(
        "observado (GARCH): persistente {p:.3}, antipersistente {a:.3}, H medio {media:.4}, SD {sd:.4}"
    );
    assert!(
        p <= 0.10 && a <= 0.10,
        "GARCH(1,1) sin memoria: persistente {p:.3}, antipersistente {a:.3} (H medio {media:.4}, SD {sd:.4})"
    );
}

/// Verdad: sin memoria la duración no cambia (factor 1). El factor mediano
/// (duración/600 000) debe estar en [0,9; 1,1] y, como mucho, el 10 % de los
/// casos puede saturar (factor ≥ 1,99 o ≤ 0,51; corrección del escéptico de
/// física: un arreglo que sólo quite el sesgo dejaría saturado el ≈ 57 % con
/// SD 0,035). Hoy, con H mediano ≈ 0,52, el factor es 2^((H − 0,5)/0,02) ≈
/// 1,9-2 (recortado a 2).
#[test]
// F1-055 (Ola Ω168): Resuelto — modulación continua adaptativa por dynamic_hurst_thresholds e incertidumbre DFA
fn cf2_duracion_no_se_modula_bajo_paseo_aleatorio() {
    let hs = hurst_paseos(PASEOS, 0xB0A5_2027, None);
    let mut f: Vec<f64> = hs
        .iter()
        .map(|&h| hurst_duration_modulation(600_000, h) as f64 / 600_000.0)
        .collect();
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mitad = f.len() / 2;
    let mediana = 0.5 * (f[mitad - 1] + f[mitad]);
    let saturados = f.iter().filter(|&&x| x >= 1.99 || x <= 0.51).count();
    println!(
        "observado: factor mediano {mediana:.3}, {saturados}/{} saturados",
        f.len()
    );
    assert!(
        (0.9..=1.1).contains(&mediana) && saturados * 10 <= f.len(),
        "factor mediano {mediana:.3}, {saturados}/{} saturados (≤ 10 % exigido)",
        f.len()
    );
}

/// Contrato del reloj (no es una solución estadística): eventos de precio
/// constante en 0, 61 000, 120 500 y 181 000 ms. Una rejilla física fija de
/// 1 minuto cierra los minutos 1, 2 y 3, es decir 3 precios y 2 retornos.
/// Hoy la vela interna se re-ancla en el evento de cierre (61 000): en
/// 120 500 sólo han pasado 59 500 ms y no cierra, así que hay 1 retorno.
#[test]
#[ignore = "OPEN F2.3b-08 (Línea C): la vela interna del Hurst se re-ancla en el evento de cierre y comprime los huecos en un solo retorno"]
fn cf2_reloj_hurst_rejilla_fija_de_un_minuto() {
    let mut fe = StatefulEngine::new();
    for &t in &[0u64, 61_000, 120_500, 181_000] {
        fe.process_tick(100.0, 0.0, t);
    }
    assert_eq!(
        fe.hurst.samples(),
        2,
        "la vela interna se re-ancla en el evento de cierre: {} retornos con cierres de rejilla en 61 000, 120 500 y 181 000 ms",
        fe.hurst.samples()
    );
}
