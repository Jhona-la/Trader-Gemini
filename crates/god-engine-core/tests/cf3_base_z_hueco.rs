//! CL-87 (spot_z-1, revisión de los PR #43–#55): `BaseZ` no acredita la
//! base retenida durante un hueco.
//!
//! Teoría: el z de la base usa promedios temporales con muestreo y
//! retención. La base se sostiene entre observaciones, y eso es una buena
//! aproximación mientras el dato sea fresco: entre eventos (ms a s), o como
//! mucho la edad máxima del spot (30 s, CL-79). Si no llega ninguna
//! observación durante más tiempo (spot caducado, entradas bloqueadas por el
//! kill-switch, pánico de latencia, deriva, arranque o reconexión), no hay
//! datos. Sostener la base todo el hueco la acredita como si se hubiera
//! observado constante: la dispersión del nodo cae al piso y el z satura en
//! ±3 al volver, sin que la base se haya movido.
//!
//! Base de las pruebas: OU con t½ = 30 s, desviación 3 pb, alrededor de
//! 10 pb, observada cada 1 s, z a la escala τ_19 = 274,9 s (la τ dominante
//! típica). Para una base estacionaria y un z calibrado, P(|z| > 1,5) ≈
//! 0,13.

use god_engine_core::base_z::BaseZ;
use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Base OU exacta en pasos de 1 s.
struct Base {
    b: f64,
    e: f64,
    sd: f64,
}
impl Base {
    fn new() -> Self {
        let e = (-std::f64::consts::LN_2 / 30.0).exp();
        Self { b: 0.0010, e, sd: 3e-4 * (1.0 - e * e).sqrt() }
    }
    fn paso(&mut self, rng: &mut Lcg) -> f64 {
        self.b = 0.0010 + (self.b - 0.0010) * self.e + self.sd * rng.n();
        self.b
    }
}

const PISO: f64 = 1e-6;

/// Lecturas de z (con el recorte ±3 del núcleo) en los `ventana` s
/// siguientes a un hueco de `hueco` s tras `previo` s de observaciones.
/// `previo` = 1 reproduce el arranque: una observación y el hueco.
fn tras_hueco(semilla: u64, previo: u64, hueco: u64, ventana: u64) -> Vec<f64> {
    let tau = SPECTRUM_SCALES_MS[19];
    let mut rng = Lcg(0xC187_0000 ^ semilla);
    let mut base = Base::new();
    let mut z = BaseZ::new();
    for s in 0..previo {
        z.observar(s * 1000, base.paso(&mut rng), tau, PISO);
    }
    for _ in 0..hueco {
        base.paso(&mut rng);
    }
    let inicio = previo + hueco;
    (0..ventana)
        .filter_map(|k| z.observar((inicio + k) * 1000, base.paso(&mut rng), tau, PISO))
        .map(|v| v.clamp(-3.0, 3.0))
        .collect()
}

/// (fracción de lecturas con |z| > 1,5, fracción saturada en ±3, lecturas).
fn colas(lecturas: &[f64]) -> (f64, f64, usize) {
    let n = lecturas.len();
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    let altas = lecturas.iter().filter(|v| v.abs() > 1.5).count();
    let sat = lecturas.iter().filter(|v| v.abs() >= 3.0).count();
    (altas as f64 / n as f64, sat as f64 / n as f64, n)
}

/// Verdad: tras un hueco (spot caducado o entradas bloqueadas) de 120 s,
/// 600 s o 1 h, las lecturas de los 2 min siguientes (donde el defecto se
/// concentra) y de los 10 min siguientes (ya recalentado) están calibradas
/// como las de una base estacionaria. Esperado, sobre 100 caminos: P(|z| > 1,5)
/// ≤ 0,25 (gaussiana 0,13) y P(|z| = 3) ≤ 0,03. Abstenerse mientras falta
/// historia es válido (las lecturas `None` no cuentan). El hueco 0 es el
/// control: la misma base observada sin interrupción.
#[test]
fn cl87_tras_un_hueco_el_z_de_la_base_no_satura() {
    for hueco in [0u64, 120, 600, 3_600] {
        for ventana in [120u64, 600] {
            let mut todas = Vec::new();
            for s in 0..100 {
                todas.extend(tras_hueco(s, 3 * 3_600, hueco, ventana));
            }
            let (altas, sat, n) = colas(&todas);
            println!(
                "hueco {hueco} s, {ventana} s después: {n} lecturas, P(|z| > 1,5) = {altas:.3}, P(|z| = 3) = {sat:.3}"
            );
            assert!(
                altas <= 0.25 && sat <= 0.03,
                "hueco {hueco} s, {ventana} s después: P(|z| > 1,5) = {altas:.3}, P(|z| = 3) = {sat:.3} ({n} lecturas)"
            );
        }
    }
}

/// Verdad: al arrancar, si la segunda observación llega mucho después de la
/// primera, la primera no se acredita como observada durante ese tiempo.
/// Mismo criterio que la prueba anterior, en los 10 min tras la segunda
/// observación, con 1 h entre ambas.
#[test]
fn cl87_al_arrancar_la_primera_observacion_no_llena_el_hueco() {
    let mut todas = Vec::new();
    for s in 0..100 {
        todas.extend(tras_hueco(500 + s, 1, 3_600, 600));
    }
    let (altas, sat, n) = colas(&todas);
    println!("arranque: {n} lecturas, P(|z| > 1,5) = {altas:.3}, P(|z| = 3) = {sat:.3}");
    assert!(
        altas <= 0.25 && sat <= 0.03,
        "arranque: P(|z| > 1,5) = {altas:.3}, P(|z| = 3) = {sat:.3} ({n} lecturas)"
    );
}

/// Verdad: la reconexión del feed (`reset_engines`) corta la serie de la
/// base, igual que reinicia los motores de rasgos: después no hay historia y
/// el z se abstiene hasta recalentar.
#[test]
fn cl87_reset_engines_reinicia_el_z_de_la_base() {
    quantum_arena::symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    let arena = quantum_arena::GlobalArena::build_in_own_stack(1_000.0);
    let mut core = god_engine_core::GodEngineCore::new(arena);
    let tau = SPECTRUM_SCALES_MS[18];
    for s in 0..2_000u64 {
        core.base_z[0].observar(s * 1000, 0.0010 + 1e-5 * (s % 2) as f64, tau, PISO);
    }
    assert!(
        core.base_z[0].observar(2_000_000, 0.0010, tau, PISO).is_some(),
        "precondición: con 2 000 s de historia hay z"
    );
    core.reset_engines();
    assert!(
        core.base_z[0].observar(2_001_000, 0.0010, tau, PISO).is_none(),
        "tras reset_engines el z de la base no debe tener historia"
    );
}
