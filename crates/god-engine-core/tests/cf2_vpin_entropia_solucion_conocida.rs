//! C-F2.3b (Línea C): VPIN y entropía de Shannon del núcleo frente a
//! soluciones conocidas, y el lado del agresor en el camino vivo.
//!
//! Teoría que se prueba:
//! - `ContinuousVPIN` es una ventana exponencial sobre el desequilibrio
//!   D_t = (D_{t−1} ± v)·K/(K+1), con K = bucket/v trades por bucket. Con
//!   agresor equilibrado e independiente, D/(K·v) es aproximadamente
//!   N(0, 1/(2K+1)) y E[VPIN] = √(2/(π(2K+1))); con agresor comprador de
//!   probabilidad p, E[VPIN] = |2p − 1|.
//! - El VPIN vivo de `StatefulEngine` debe medir el agresor real; hoy decide
//!   el lado por la regla del tick sobre el mid e ignora `is_buyer_maker`.
//! - `ShannonEntropy` reparte el z de Welford en 10 bins sobre ±z95: con
//!   retornos gaussianos su valor normalizado es la entropía de esos bins
//!   (≈ 0,9695 menos el sesgo plug-in). La misma difusión con eventos de
//!   retorno 0 intercalados debería dar el mismo desorden.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use god_engine_core::math_kernels::{ContinuousVPIN, ShannonEntropy};
use god_engine_core::stateful_engine::StatefulEngine;

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

// ─────────────────────────────── VPIN ────────────────────────────────

/// Media de VPIN en 200 000 trades de `v_usd` tras 5 000 de quemado, con
/// agresor comprador de probabilidad `p_compra` independiente.
fn vpin_medio(p_compra: f64, v_usd: f64, piso_usd: f64, semilla: u64) -> f64 {
    let mut rng = Lcg(semilla);
    let mut v = ContinuousVPIN::new(piso_usd);
    let (quemado, n) = (5_000usize, 200_000usize);
    let mut s = 0.0;
    for t in 0..(quemado + n) {
        let is_buyer_maker = rng.u() >= p_compra; // agresor comprador con probabilidad p
        let x = v.update(v_usd, is_buyer_maker);
        if t >= quemado {
            s += x;
        }
    }
    s / n as f64
}

/// E[VPIN] nulo con K trades por bucket: √(2/(π(2K+1))).
fn nulo_cerrado(k: f64) -> f64 {
    (2.0 / (std::f64::consts::PI * (2.0 * k + 1.0))).sqrt()
}

/// Verdad: con piso 1 USD y trades de 50 USD el bucket es 100·50, K = 100 y
/// E[VPIN] = √(2/(π·201)) = 0,05628. SD de una lectura √((1 − 2/π)/201) =
/// 0,0425; autocorrelación K/(K+1) ⇒ N_eff ≈ 200 000/201 ≈ 995 y SE ≈
/// 0,00135. Tolerancia ±0,006 = 4,4 SE.
#[test]
fn cf2_vpin_nulo_equilibrado_forma_cerrada() {
    let m = vpin_medio(0.5, 50.0, 1.0, 0x7F1A);
    let e = nulo_cerrado(100.0);
    println!("observado: VPIN nulo medio {m:.5} frente a {e:.5}");
    assert!(
        (m - e).abs() < 0.006,
        "VPIN nulo medio {m:.5} frente a la forma cerrada {e:.5} (K = 100)"
    );
}

/// Verdad: con agresor comprador al 80 %, E[D] = K·E[X] = 0,6·K·v, así que
/// VPIN = |2p − 1| = 0,6 (D no cruza 0). SD = 0,8/√201 = 0,0564, SE con
/// N_eff ≈ 995 = 0,0018; tolerancia ±0,008 = 4,4 SE. Nota: un flujo al 80 %
/// comprador (0,6) no alcanza el veto causal de 0,75, que exige más del
/// 87,5 % de un solo lado sostenido.
#[test]
fn cf2_vpin_flujo_sesgado_p08() {
    let m = vpin_medio(0.8, 50.0, 1.0, 0x7F1B);
    println!("observado: VPIN medio con p = 0,8: {m:.5}");
    assert!(
        (m - 0.6).abs() < 0.008,
        "VPIN medio {m:.5} frente a |2p − 1| = 0,6"
    );
}

/// Sustituye a `cf2_vpin_nulo_no_depende_del_tamano_del_trade` (corrección
/// de los dos escépticos): en el VPIN de reloj de volumen de Easley, López
/// de Prado y O'Hara el nulo depende por construcción de cuántos trades
/// caben en el bucket, así que exigir un nulo independiente del tamaño
/// contradice la teoría. Lo que se fija es la forma cerrada CON el piso vivo
/// de 10 000 USD (stateful_engine.rs): K = max(100, 10 000/v).
/// Trades de 10 USD: K = 1000, E = 0,01784, SD = √((1 − 2/π)/2001) =
/// 0,01348, N_eff ≈ 200 000/2001 ≈ 100, SE ≈ 0,00135.
/// Trades de 1000 USD: K = 100, E = 0,05628, SE ≈ 0,00135 (como arriba).
/// Tolerancia ±0,006 = 4,4 SE en los dos. El requisito normativo pasa a los
/// umbrales (cada umbral debe ser un cuantil del nulo de su propio K).
#[test]
fn cf2_vpin_nulo_sigue_forma_cerrada_con_piso() {
    let pequeno = vpin_medio(0.5, 10.0, 10_000.0, 0x7F1C);
    let grande = vpin_medio(0.5, 1_000.0, 10_000.0, 0x7F1D);
    let (e_pequeno, e_grande) = (nulo_cerrado(1_000.0), nulo_cerrado(100.0));
    println!(
        "observado: nulo con 10 USD {pequeno:.5} (cerrado {e_pequeno:.5}); con 1 000 USD {grande:.5} (cerrado {e_grande:.5})"
    );
    assert!(
        (pequeno - e_pequeno).abs() < 0.006,
        "trades de 10 USD (K = 1000): {pequeno:.5} frente a {e_pequeno:.5}"
    );
    assert!(
        (grande - e_grande).abs() < 0.006,
        "trades de 1 000 USD (K = 100): {grande:.5} frente a {e_grande:.5}"
    );
}

/// Camino vivo: un depth a 100,00, otro a 100,01 (sube un tick) y 4000
/// trades de 0,5 unidades al mismo mid de 100,01 cada 50 ms, como llegan en
/// vivo con prints dentro del libro. El agresor real es equilibrado y lo
/// mide en paralelo un `ContinuousVPIN` de control (K ≈ 200: nulo ≈ 0,040,
/// SD ≈ 0,030, P(> 0,2) ≈ 6e-5). Verdad: el VPIN vivo debe quedar por debajo
/// del veto causal de 0,75. Hoy todos los trades son ticks planos que
/// heredan «compra» de la subida del depth: VPIN vivo = 1,0 exacto.
///
/// Revisión cruzada (CL-65): la causa es que `process_tick` no recibe el lado,
/// así que el arreglo cambia su firma. Quien lo arregle pasa `is_buyer_maker`
/// (ya calculado en el bucle) a la nueva llamada, o conduce la prueba por
/// `GodEngineCore::process_event`, que ya lo transporta (lib.rs:1703).
#[test]
#[ignore = "OPEN F2.3b-02 (Línea C, núcleo/StatefulEngine), CONTRATO DE API: process_tick no recibe el agresor; el arreglo le pasa is_buyer_maker (ya está en el bucle) o la conduce por process_event. El VPIN vivo decide el lado por la regla del tick sobre el mid e ignora is_buyer_maker (1,0 con flujo real equilibrado)"]
fn cf2_vpin_vivo_ignora_el_agresor() {
    let mut fe = StatefulEngine::new();
    fe.process_tick(100.00, 0.0, 1_000); // depth
    fe.process_tick(100.01, 0.0, 1_100); // depth: el mid sube un tick
    let mut rng = Lcg(0xACE);
    let mut verdad = ContinuousVPIN::new(10_000.0);
    let mut t = 1_100u64;
    for _ in 0..4_000 {
        t += 50;
        let is_buyer_maker = rng.u() < 0.5; // flujo real equilibrado
        fe.process_tick(100.01, 0.5, t); // trade con el mid del libro, sin cambio
        verdad.update(0.5 * 100.01, is_buyer_maker);
    }
    let (vivo, real) = (fe.cvpin.current_vpin(), verdad.current_vpin());
    assert!(real < 0.2, "control con el agresor real: {real:.4}");
    assert!(
        vivo < 0.75,
        "VPIN vivo {vivo:.4} (veto causal 0,75) con flujo real equilibrado (control {real:.4})"
    );
}

// ────────────────────────────── Entropía ─────────────────────────────

/// Φ(z) por Simpson sobre [−10, z] con 20 000 subintervalos.
fn phi(z: f64) -> f64 {
    let (a, n) = (-10.0f64, 20_000usize);
    let h = (z - a) / n as f64;
    let f = |x: f64| (-0.5 * x * x).exp() / std::f64::consts::TAU.sqrt();
    let mut s = f(a) + f(z);
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        s += w * f(a + i as f64 * h);
    }
    s * h / 3.0
}

/// Entropía normalizada esperada de 10 bins sobre ±z95 con las colas en los
/// extremos, menos el sesgo plug-in (K − 1)/(2·N_eff) con N_eff =
/// (1 + 0,999)/(1 − 0,999) = 1999 (decaimiento 0,999 por evento).
fn entropia_gaussiana_esperada() -> f64 {
    const Z95: f64 = 1.959963984540054; // diffusion.rs, Z95
    let ancho = 2.0 * Z95 / 10.0;
    let mut cdf = vec![0.0];
    for j in 1..10 {
        cdf.push(phi(-Z95 + j as f64 * ancho));
    }
    cdf.push(1.0);
    let h: f64 = cdf
        .windows(2)
        .map(|w| {
            let p = w[1] - w[0];
            -p * p.ln()
        })
        .sum();
    let sesgo = 9.0 / (2.0 * 1999.0);
    (h - sesgo) / 10f64.ln()
}

/// Última lectura de `ShannonEntropy` tras 20 000 retornos N(0, 1e-8),
/// intercalados con eventos de retorno 0 en una fracción `frac_ceros` de los
/// eventos.
fn entropia(frac_ceros: f64, semilla: u64) -> f64 {
    let mut rng = Lcg(semilla);
    let mut e = ShannonEntropy::new();
    let (mut h, mut movimientos) = (0.0, 0usize);
    while movimientos < 20_000 {
        if rng.u() < frac_ceros {
            h = e.update(0.0);
        } else {
            h = e.update(1e-4 * rng.n());
            movimientos += 1;
        }
    }
    h
}

/// Verdad: H = 2,2324 nats sobre los 10 bins, 0,9695 normalizada; con el
/// sesgo plug-in −9/(2·1999)/ln 10 queda ≈ 0,9685. SD de una lectura
/// √((Σ p ln² p − H²)/N_eff)/ln 10 ≈ 0,0035; tolerancia 0,015 = 4,3 SD.
#[test]
fn cf2_entropia_gaussiana_forma_cerrada() {
    let esperado = entropia_gaussiana_esperada();
    let h = entropia(0.0, 0xE27);
    println!("observado: entropía {h:.4} frente a {esperado:.4}");
    assert!(
        (h - esperado).abs() < 0.015,
        "entropía {h:.4} frente a la forma cerrada {esperado:.4}"
    );
}

/// Especificación del reloj de la entropía (no solución conocida del
/// estimador, como apunta el escéptico de física): la misma difusión de
/// 20 000 movimientos, con 0 % y con 80 % de eventos sin cambio de mid
/// (trades al mid del libro, depth sin cambio de BBO), debería dar la misma
/// medida de desorden: |a − b| < 0,05. Hoy la dispersión de Welford baja a
/// √0,2·σ, los movimientos caen en los bins extremos y los ceros en los
/// centrales: la entropía predicha es 0,37-0,59 frente a ≈ 0,9685.
#[test]
#[ignore = "OPEN F2.3b-03 (Línea C, núcleo): la entropía viva mide la proporción de eventos sin cambio de mid, no la forma de los retornos"]
fn cf2_entropia_invariante_a_eventos_sin_cambio() {
    let a = entropia(0.0, 0xE28);
    let b = entropia(0.8, 0xE29);
    assert!(
        (a - b).abs() < 0.05,
        "misma difusión: {a:.4} sin ceros frente a {b:.4} con 80 % de eventos sin cambio de mid"
    );
}
