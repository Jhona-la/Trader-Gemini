//! C-F2.3b (AGY, pedido con test): el número de Reynolds de
//! `navier_stokes.rs` frente a un mid browniano de σ conocida.
//!
//! Teoría que se prueba: con spread fijo de 0,01 USD, profundidad 1/1, sin
//! trades (agresión 1) y ATR de 0,001 sobre mid ≈ 100, la escala es
//! L = max(spread, mid·ATR·0,05) = max(0,01; 0,005) = 0,01 y τ = 0,2 s, de
//! modo que Re = τ·|Δmid/dt|/L = 20·|Δmid|/dt. Para un mid browniano de
//! σ = 0,02 USD/√s, |Δmid| = σ√dt·|Z| y E[Re] = 20σ·√(2/π)/√dt.
//! - Forma cerrada (regresión): 3,1915 a 10 ms y 1,00925 a 100 ms.
//! - Invariancia a la cadencia (especificación de un número de régimen): la
//!   MISMA difusión observada cada 10 ms o cada 100 ms no debería cambiar de
//!   régimen; hoy el cociente es √10 ≈ 3,16 porque Re ∝ 1/√dt mide cadencia.
//!
//! En vivo el dt relevante es el intervalo entre trades o klines: un evento
//! de depth puro llega con `allow_entries = false` y el núcleo sale en
//! `entries_blocked` antes de actualizar Navier-Stokes (god-engine-core,
//! lib.rs, verificado en la rama ciclo13).
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use feature_engine::navier_stokes::NavierStokesReynoldsEngine;

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

const SIGMA: f64 = 0.02; // USD por raíz de segundo

/// Media del Reynolds crudo (no la EWMA) sobre `n` pasos de dt constante.
fn re_medio(dt_ms: u64, n: usize, semilla: u64) -> f64 {
    let dt = dt_ms as f64 / 1000.0;
    let mut rng = Lcg(semilla);
    let mut ns = NavierStokesReynoldsEngine::new();
    let (mut mid, mut t) = (100.0f64, 1_000u64);
    ns.update(mid - 0.005, mid + 0.005, 1.0, 1.0, 0.0, 0.001, t);
    let mut suma = 0.0;
    for _ in 0..n {
        mid += SIGMA * dt.sqrt() * rng.n();
        t += dt_ms;
        ns.update(mid - 0.005, mid + 0.005, 1.0, 1.0, 0.0, 0.001, t);
        suma += ns.reynolds_number;
    }
    suma / n as f64
}

fn forma_cerrada(dt_s: f64) -> f64 {
    20.0 * SIGMA * (2.0 / std::f64::consts::PI).sqrt() / dt_s.sqrt()
}

/// Verdades: 3,1915 a 10 ms y 1,00925 a 100 ms. Muestras independientes;
/// sd(Re) = 20σ·0,6028/√dt = 2,411 y 0,762, SE con N = 20 000 de 0,017 y
/// 0,0054; tolerancias 0,08 y 0,025 (≈ 4,7 y 4,6 SE). El tope en 100 está
/// a más de 25 sd.
///
/// CARACTERIZACIÓN (revisión cruzada, CL-65): fija la ley defectuosa
/// Re ∝ dt^{−1/2}, la misma afirmación que `cf2_re_invariante_a_la_cadencia`
/// niega. Quien arregle F2.3b-05 la retira o la reescribe con la forma
/// cerrada nueva en el mismo commit en que activa la OPEN.
#[test]
fn cf2_re_forma_cerrada_10ms_y_100ms() {
    let a = re_medio(10, 20_000, 11);
    let b = re_medio(100, 20_000, 12);
    eprintln!(
        "[cf2 navier] Re(10 ms) {a:.4} frente a {:.4}; Re(100 ms) {b:.4} frente a {:.5}",
        forma_cerrada(0.01),
        forma_cerrada(0.1)
    );
    assert!((a - forma_cerrada(0.01)).abs() < 0.08, "Re(10 ms) {a:.4}");
    assert!((b - forma_cerrada(0.1)).abs() < 0.025, "Re(100 ms) {b:.4}");
}

/// Verdad (especificación de un número de régimen): cociente 1 entre la
/// misma difusión observada a 10 ms y a 100 ms. Tolerancia 0,10 frente a una
/// SE del cociente de ≈ 0,01. Por la forma cerrada hoy vale √10 ≈ 3,16.
#[test]
#[ignore = "OPEN F2.3b-05 (AGY): el Reynolds es proporcional a 1/√dt y mide la cadencia de eventos, no el régimen de la difusión (cociente √10 entre 10 ms y 100 ms)"]
fn cf2_re_invariante_a_la_cadencia() {
    let a = re_medio(10, 20_000, 13);
    let b = re_medio(100, 20_000, 14);
    assert!(
        (a / b - 1.0).abs() < 0.10,
        "misma difusión: Re(10 ms) {a:.3} frente a Re(100 ms) {b:.3}, cociente {:.3}",
        a / b
    );
}
