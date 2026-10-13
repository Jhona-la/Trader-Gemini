//! CL-71 (C-F4, LC-F6-01, FA-9 de F2.2-07): el ATR de las decisiones es el
//! rango equivalente de la σ de 60 s, y los z de tendencia que lo usan
//! tienen el tamaño nominal en un paseo aleatorio.
//!
//! El escudo de momento adverso y las puertas de tendencia estandarizan la
//! distancia entre EMAs de vela con la σ que sale del ATR
//! (`diffusion::sigma_from_atr`, ATR/√(8/π)). Con el ATR viejo (suelo de
//! 10 pb y trinquete) esa σ estaba inflada: con σ = 4 pb/min el ATR daba
//! σ_ATR ≥ 6,27 pb, los z se encogían ×0,64 y |z| > 1,96 ocurría en ≈ 0,2 %
//! de los minutos en lugar del 5 %.
//!
//! Verdad (FA-9): en un paseo aleatorio gaussiano con un evento por segundo,
//! la fracción de minutos con |z| > 1,96 del diferencial EMA 9/21 es 5 % si
//! la σ es correcta. Banda [4 %; 8 %]: el ruido de σ̂60 sube la tasa
//! (convexidad de 2Φ(−1,96·s)) y un sesgo de σ̂ la mueve en cualquier
//! sentido. Medido con CL-71: 4,4 % (476/10 800); en estos caminos σ̂
//! sobrestima σ un ~4 % (σ_ATR/σ = 1,039 con σ = 4 pb en cf2), lo que encoge
//! los z y deja la tasa algo por debajo del 5 %. Antes de CL-71 era ≈ 0,2 %.

use god_engine_core::diffusion::{ema_spread_z, EMA_FAST_BARS, EMA_SLOW_BARS, Z95};
use god_engine_core::stateful_engine::StatefulEngine;

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

/// Un camino: `minutos` minutos a un evento por segundo con σ por minuto
/// `sigma_min`. Tras `quemado` minutos, cuenta los minutos (muestreados al
/// cierre de la vela interna) con |z| > 1,96. Devuelve (excesos, minutos).
fn excesos_en_camino(semilla: u64, sigma_min: f64, minutos: u64, quemado: u64) -> (u64, u64) {
    let sigma_s = sigma_min / 60f64.sqrt();
    let mut rng = Lcg(semilla);
    let mut e = StatefulEngine::new();
    let mut ln_p = 100f64.ln();
    let (mut excesos, mut medidos) = (0u64, 0u64);
    for i in 0..minutos * 60 {
        ln_p += sigma_s * rng.n();
        e.process_tick(ln_p.exp(), 0.0, 1_000 * i);
        // La vela interna cierra con el primer evento pasados 60 s.
        if i % 60 == 0 && i / 60 >= quemado && e.kline_ema_slow > 0.0 {
            let rel = (e.kline_ema_fast - e.kline_ema_slow) / e.kline_ema_slow;
            let z = ema_spread_z(rel, e.get_atr_pct(), EMA_FAST_BARS, EMA_SLOW_BARS);
            medidos += 1;
            if z.abs() > Z95 {
                excesos += 1;
            }
        }
    }
    (excesos, medidos)
}

#[test]
fn cl71_z_de_tendencia_con_tamano_nominal_en_paseo_aleatorio() {
    let (mut excesos, mut medidos) = (0u64, 0u64);
    for k in 0..30u64 {
        let (x, n) = excesos_en_camino(0xC171_0000 + k, 0.0004, 8 * 60, 120);
        excesos += x;
        medidos += n;
    }
    let tasa = excesos as f64 / medidos as f64;
    println!("|z| > 1,96 en {excesos}/{medidos} minutos: {tasa:.4}");
    assert!(
        (0.04..=0.08).contains(&tasa),
        "tasa de |z| > 1,96 = {tasa:.4} en un paseo aleatorio (nominal 0,05; banda [0,04; 0,08])"
    );
}

/// Verdad: σ por minuto de 4 pb ⇒ ATR = √(8/π)·4 pb del precio al cabo de
/// unas horas (media temporal tras 2 h, ±10 %).
#[test]
fn cl71_atr_es_el_rango_equivalente_de_sigma_de_un_minuto() {
    let sigma_min = 0.0004;
    let sigma_s = sigma_min / 60f64.sqrt();
    let mut rng = Lcg(0xC171_1000);
    let mut e = StatefulEngine::new();
    let mut ln_p = 100f64.ln();
    let (mut suma, mut n) = (0.0f64, 0u64);
    for i in 0..(12 * 3600u64) {
        ln_p += sigma_s * rng.n();
        e.process_tick(ln_p.exp(), 0.0, 1_000 * i);
        if i >= 2 * 3600 {
            suma += e.get_atr_pct();
            n += 1;
        }
    }
    let media = suma / n as f64;
    let verdad = god_engine_core::diffusion::PARKINSON_RANGE_FACTOR * sigma_min;
    assert!(
        (media / verdad - 1.0).abs() < 0.10,
        "ATR medio {media:e} frente a √(8/π)·σ = {verdad:e}"
    );
}

/// Un reinicio deja el motor como uno nuevo: la misma cinta después del
/// `reset` da el mismo ATR bit a bit que en un motor recién creado.
#[test]
fn cl71_reset_equivale_a_motor_nuevo() {
    let cinta: Vec<(f64, u64)> = {
        let mut rng = Lcg(0xC171_2000);
        let mut ln_p = 100f64.ln();
        (0..3 * 3600u64)
            .map(|i| {
                ln_p += 0.0003 * rng.n();
                (ln_p.exp(), 1_000 * i)
            })
            .collect()
    };
    let mut usado = StatefulEngine::new();
    for &(p, t) in cinta.iter().take(5_000) {
        usado.process_tick(p * 1.3, 0.0, t);
    }
    usado.reset();
    let mut nuevo = StatefulEngine::new();
    for &(p, t) in &cinta {
        usado.process_tick(p, 0.0, t);
        nuevo.process_tick(p, 0.0, t);
        assert_eq!(usado.v_t.to_bits(), nuevo.v_t.to_bits(), "t = {t}");
    }
}

/// Choque (escenario del escéptico de CL-71): tras tres horas a σ = 4 pb/min,
/// el precio cae 100 pb en 30 s a un evento por segundo. El ATR viejo subía
/// al rango de la vela en el mismo evento (trinquete); el nuevo no puede
/// quedarse en la escala de calma hasta el cierre de la vela. Verdad: en
/// cada evento del choque, ATR ≥ √(8/π)·R_abierta/q99 (la envolvente), y a
/// los 30 s el ATR es ≥ 5 veces el de antes del choque (√(8/π)·100 pb/3,04
/// = 52 pb frente a ≈ 6,4 pb).
#[test]
fn cl71_choque_de_volatilidad_sube_el_atr_en_el_mismo_minuto() {
    use god_engine_core::stateful_engine::CUANTIL_99_RANGO;
    let sigma_s = 0.0004 / 60f64.sqrt();
    let mut rng = Lcg(0xC171_3000);
    let mut e = StatefulEngine::new();
    let mut ln_p = 100f64.ln();
    let mut t = 0u64;
    for _ in 0..3 * 3600u64 {
        ln_p += sigma_s * rng.n();
        e.process_tick(ln_p.exp(), 0.0, t);
        t += 1_000;
    }
    let antes = e.get_atr_pct();
    let p0 = ln_p.exp();
    for k in 1..=30u64 {
        let p = p0 * (1.0 - 0.01 * k as f64 / 30.0);
        e.process_tick(p, 0.0, t);
        t += 1_000;
        let envolvente = god_engine_core::diffusion::PARKINSON_RANGE_FACTOR
            * (e.kline_high - e.kline_low)
            / CUANTIL_99_RANGO;
        assert!(
            e.v_t >= envolvente * (1.0 - 1e-12),
            "evento {k}: v_t {} por debajo de la envolvente {envolvente}",
            e.v_t
        );
    }
    let despues = e.get_atr_pct();
    assert!(
        despues >= 5.0 * antes,
        "ATR tras −100 pb en 30 s: {despues:e} frente a {antes:e} antes del choque"
    );
}

/// Verdad: el cuantil del 99 % del rango de un browniano estándar en [0, 1]
/// es ≈ 3,04 (`CUANTIL_99_RANGO`). Monte Carlo independiente: 40 000 caminos
/// de 1 024 pasos con la corrección de continuidad 2·0,5826/√n; la media del
/// rango debe dar √(8/π) (±0,01) y el cuantil 3,04 (±0,08, ≈ 3 EE del
/// cuantil con esta muestra).
#[test]
fn cl71_cuantil_99_del_rango_browniano() {
    use god_engine_core::stateful_engine::CUANTIL_99_RANGO;
    let (caminos, pasos) = (40_000usize, 1_024usize);
    let mut rng = Lcg(0xC171_4000);
    let dt = (1.0 / pasos as f64).sqrt();
    let correccion = 2.0 * 0.5826 / (pasos as f64).sqrt();
    let mut rangos: Vec<f64> = (0..caminos)
        .map(|_| {
            let (mut w, mut alto, mut bajo) = (0.0f64, 0.0f64, 0.0f64);
            for _ in 0..pasos {
                w += dt * rng.n();
                alto = alto.max(w);
                bajo = bajo.min(w);
            }
            alto - bajo + correccion
        })
        .collect();
    let media = rangos.iter().sum::<f64>() / caminos as f64;
    rangos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q99 = rangos[(0.99 * caminos as f64) as usize];
    println!("rango: media {media:.4}, q99 {q99:.4}");
    assert!(
        (media - god_engine_core::diffusion::PARKINSON_RANGE_FACTOR).abs() < 0.01,
        "media del rango {media}"
    );
    assert!(
        (q99 - CUANTIL_99_RANGO).abs() < 0.08,
        "q99 del rango {q99} frente a {CUANTIL_99_RANGO}"
    );
}
