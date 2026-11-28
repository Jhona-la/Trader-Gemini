//! Contratos del observable Ricci-q en StatefulEngine (GLM 223, wiring del
//! núcleo del ciclo 221). Determinista (LCG propio), sin cambio de conducta
//! del engine: el observable se publica en frontera %64, NaN en warmup —
//! patrón σz/λ_KS/β_RG/D^α/LPPLS/FR.

use god_engine_core::stateful_engine::{RICCI_Q_FIJO, StatefulEngine};

/// LCG determinista (mismo generador de los contratos del 221).
fn lcg(estado: &mut u64) -> f64 {
    *estado = estado
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*estado >> 11) as f64) / (1u64 << 53) as f64
}

/// Aproximación determinista a una normal estándar (suma de 12 uniforms − 6).
fn normal(estado: &mut u64) -> f64 {
    (0..12).map(|_| lcg(estado)).sum::<f64>() - 6.0
}

fn feed_gaussiano(engine: &mut StatefulEngine, n: usize, semilla: u64) {
    feed_gaussiano_desde(engine, n, semilla, 100.0);
}

/// Sigue el paseo desde `p0` y devuelve el último precio. Llamar dos veces a
/// `feed_gaussiano` reinicia el precio en 100: ese salto (≈ −4,7 % con estas
/// semillas, unas 46σ) entra en el anillo como un solo outlier y lleva R a
/// ≈ −2,6. Así, la población deja de ser casi gaussiana.
fn feed_gaussiano_desde(engine: &mut StatefulEngine, n: usize, semilla: u64, p0: f64) -> f64 {
    let mut e = semilla;
    let mut p = p0;
    for _ in 0..n {
        p *= 1.0 + 0.001 * normal(&mut e);
        engine.process_tick(p, 1.0, 0);
    }
    p
}

fn feed_colas(engine: &mut StatefulEngine, n: usize, semilla: u64) {
    // Mezcla: normal suave + saltos gruesos (5% de los ticks, ±8σ):
    // población con colas EFECTIVAS mucho más pesadas que la gaussiana.
    let mut e = semilla;
    let mut p = 100.0_f64;
    for _ in 0..n {
        let z = normal(&mut e);
        let salto = if lcg(&mut e) < 0.05 { 8.0 * z } else { z };
        p *= 1.0 + 0.001 * salto;
        engine.process_tick(p, 1.0, 0);
    }
}

#[test]
fn glm223_warmup_nan_y_publicacion_en_frontera() {
    // Warmup: sin 600 retornos, el observable es NaN (ausencia honesta).
    let mut e1 = StatefulEngine::new();
    let p = feed_gaussiano_desde(&mut e1, 599, 0x5EED11, 100.0);
    assert!(e1.ricci_q.is_nan(), "warmup: NaN, no cero");

    // Con el anillo lleno, la primera frontera %64 publica un valor finito
    // NEGATIVO. Con q = 1.2 (ν = 9) la propia familia da R = −(ν+3)/ν = −4/3;
    // la métrica es model-relative y esta población (suma de 12 uniformes,
    // casi gaussiana y con colas algo más finas que el modelo) la curva un
    // poco más: R → −1.476 con N → ∞ (−1.455 para una gaussiana exacta) y
    // dispersión ≈ 0.057 con 600 muestras. La banda (−1.75, −1.20) es ±5 SE
    // alrededor de ese valor.
    let p = feed_gaussiano_desde(&mut e1, 64, 0x5EED12, p); // cruza fronteras con anillo lleno
    assert!(
        e1.ricci_q.is_finite(),
        "con datos suficientes el observable se publica, no queda colgado en NaN"
    );
    assert!(
        e1.ricci_q < -1.20 && e1.ricci_q > -1.75,
        "R de población casi gaussiana con q = 1.2: ≈ −1.48, obtenido {}",
        e1.ricci_q
    );
    // El mínimo de época sólo existe tras una época COMPLETA con datos.
    feed_gaussiano_desde(&mut e1, 128, 0x5EED13, p);
    if e1.ricci_q_min_epoca.is_finite() {
        assert!(e1.ricci_q_min_epoca <= e1.ricci_q || e1.ricci_q.is_nan());
    }
}

#[test]
fn glm223_conducta_invariante_bit_a_bit() {
    // Mismo feed ⇒ mismo observable, bit a bit (determinismo del anillo y
    // del estimador; el consumo del consejo exige reproducibilidad).
    let mut a = StatefulEngine::new();
    let mut b = StatefulEngine::new();
    feed_gaussiano(&mut a, 900, 0xC0FFEE);
    feed_gaussiano(&mut b, 900, 0xC0FFEE);
    assert_eq!(a.ricci_q.to_bits(), b.ricci_q.to_bits());
    assert_eq!(a.ricci_q_min_epoca.to_bits(), b.ricci_q_min_epoca.to_bits());
}

#[test]
fn glm223_colas_efectivas_aplanan_la_geometria() {
    // LA SEÑAL DE RÉGIMEN (hipótesis §3.4×221): una población con colas
    // efectivas pesadas APLANA la geometría — |R(colas)| < |R(gaussiano)|.
    // Si este contrato falla, la hipótesis del observable queda REFUTADA y
    // el wiring se replantea (honestidad antes que señal decorativa).
    let mut g = StatefulEngine::new();
    feed_gaussiano(&mut g, 1200, 0x5EED21);
    let mut c = StatefulEngine::new();
    feed_colas(&mut c, 1200, 0x5EED22);
    assert!(
        g.ricci_q.is_finite() && c.ricci_q.is_finite(),
        "ambos regímenes publican (g={}, c={})",
        g.ricci_q,
        c.ricci_q
    );
    assert!(
        c.ricci_q.abs() < g.ricci_q.abs(),
        "colas efectivas deben aplanar: |R_colas|={} < |R_gauss|={}",
        c.ricci_q,
        g.ricci_q
    );
}

#[test]
fn glm223_q_fijo_dentro_del_rango_certificado() {
    // El núcleo 221 certifica (1, 5/3) abierto; el wiring debe respetarlo.
    assert!(RICCI_Q_FIJO > 1.0 && RICCI_Q_FIJO < 5.0 / 3.0);
}
