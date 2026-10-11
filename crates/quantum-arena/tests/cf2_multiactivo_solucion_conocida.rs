//! C-F2.3a (Línea C): IC cruzado por escala (`EspectralMultiactivo`) frente a
//! una correlación conocida.
//!
//! Prueba de solución conocida de la fase C-F2 (ver
//! `docs/audit/HALLAZGOS_C-F2_2026-10-11.md`). Dos caminatas aleatorias con
//! incrementos de correlación instantánea ρ = 0,6: la covarianza de dos
//! retornos de bloque de duración τ que se solapan τ − gap es ρσ²(τ − gap),
//! así que corr = ρ·(τ − gap)/τ, y con bloques alineados es ρ. El IC cruzado
//! «resuelto en escala» que publica el módulo debe estimar ρ a la escala, no
//! la fracción de solape que deja la fase de arranque de cada moneda.
//!
//! Cada moneda se alimenta como en el núcleo: tras `update` de su espectro,
//! `observar_maduracion` con el último bloque maduro de la escala 18 (69 s a
//! un evento por segundo). Generadores propios (LCG + Box-Muller) con semilla
//! fija.
//!
//! Convención: un test que documenta un defecto confirmado lleva
//! `#[ignore = "OPEN F2.x-nn (dueño): causa"]` con el assert intacto. Falla
//! hoy por la causa del hallazgo y se activa (se quita el `ignore`) con el
//! arreglo. Los tests sin `ignore` son regresiones de la teoría y deben pasar.

use quantum_arena::espectral_multiactivo::EspectralMultiactivo;
use quantum_arena::temporal_spectrum::{TemporalSpectrum, SPECTRUM_SCALES_MS};

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

const T0: u64 = 1_700_000_000_000;
const DT: u64 = 1_000;
const P0: f64 = 100.0;
const RHO: f64 = 0.6;
const ESCALA: usize = 18;

/// Moneda 0 (A) y monedas 1..=k (B_j), cada B_j con correlación instantánea ρ
/// frente a A y su propio arranque `desfases_s[j]` (en eventos de 1 s).
/// Devuelve la media temporal de coherencia_par(0, j, 18) desde el evento
/// `desde`, por cada B_j.
fn ic_cruzado(seed: u64, desfases_s: &[u64], eventos: u64, desde: u64) -> Vec<f64> {
    let sd = 1e-4;
    let k = desfases_s.len();
    let mut g = Lcg(seed);
    let mut xa = 0.0f64;
    let mut xb = vec![0.0f64; k];
    let mut sa = TemporalSpectrum::new();
    let mut sb: Vec<TemporalSpectrum> = (0..k).map(|_| TemporalSpectrum::new()).collect();
    let mut ma = EspectralMultiactivo::new(k + 1);
    let tau = SPECTRUM_SCALES_MS[ESCALA];
    let mut suma = vec![0.0f64; k];
    let mut n = vec![0usize; k];
    for i in 0..eventos {
        let z1 = g.n();
        xa += sd * z1;
        for x in xb.iter_mut() {
            *x += sd * (RHO * z1 + (1.0 - RHO * RHO).sqrt() * g.n());
        }
        let ts = T0 + i * DT;
        sa.update(P0 * xa.exp(), ts);
        if let Some((t, r)) = sa.ultimo_bloque_maduro(ESCALA) {
            ma.observar_maduracion(0, tau, ESCALA, t, r);
        }
        for j in 0..k {
            if i >= desfases_s[j] {
                sb[j].update(P0 * xb[j].exp(), ts);
                if let Some((t, r)) = sb[j].ultimo_bloque_maduro(ESCALA) {
                    ma.observar_maduracion(j + 1, tau, ESCALA, t, r);
                }
            }
        }
        if i >= desde {
            for j in 0..k {
                if let Some(ic) = ma.coherencia_par(0, j + 1, ESCALA) {
                    suma[j] += ic;
                    n[j] += 1;
                }
            }
        }
    }
    for j in 0..k {
        assert!(
            n[j] as u64 > (eventos - desde) / 2,
            "el par (0, {}) no maduró ({})",
            j + 1,
            n[j]
        );
    }
    (0..k).map(|j| suma[j] / n[j] as f64).collect()
}

// Valor verdadero: con bloques alineados (mismo arranque) corr = ρ = 0,6.
// ≈ 830 pares de bloques tras el evento 14 400: SE ≈ (1 − ρ²)/√830 = 0,022;
// tolerancia 0,10 ≈ 4,5 SE.
#[test]
fn cf2_ic_cruzado_estima_rho_a_la_escala_alineado() {
    let alineado = ic_cruzado(0xC0F2_0401, &[0], 72_000, 14_400)[0];
    println!("IC cruzado alineado a 69 s: {alineado:.4}");
    assert!(
        (alineado - RHO).abs() < 0.10,
        "control alineado: IC cruzado {alineado:.3}, verdad 0,6"
    );
}

// Valor verdadero: ρ = 0,6 a la escala. Contrato: |IC − 0,6| < 0,10 (≈ 4,5 SE)
// también si la moneda B arranca 28 s después que A.
// Hoy: el cierre de B se empareja con el de A con gap 28 s < 0,5τ = 34,4 s,
// sin corrección por solape, y el IC queda en 0,6·41/69 ≈ 0,357.
#[test]
#[ignore = "OPEN F2.3-07 (Línea C, C-F2.3): el IC cruzado mide ρ·solape/τ; con 28 s de desfase da ≈ 0,357"]
fn cf2_ic_cruzado_estima_rho_a_la_escala() {
    let desfasado = ic_cruzado(0xC0F2_0402, &[28], 72_000, 14_400)[0];
    assert!(
        (desfasado - RHO).abs() < 0.10,
        "bloques desfasados 28 s: IC cruzado {desfasado:.3}, verdad 0,6 (solape 41/69 da ≈ 0,357)"
    );
}

// Corrección del escéptico COMPLETITUD: lo que ocurre en vivo es una fase de
// arranque arbitraria por moneda. Con desfase δ uniforme en [0, τ) el solape
// emparejado es max(δ, τ − δ), y hoy se espera E[IC] = 0,75·ρ = 0,45.
// Seis monedas B con δ pseudoaleatorio (semilla fija) frente a la misma A,
// 36 000 eventos (≈ 310 pares por moneda, SE ≈ 0,036 por moneda y ≈ 0,015 en
// la media de seis). Contrato: la media de los seis IC cumple
// |IC − 0,6| < 0,10 (≈ 6 SE).
#[test]
#[ignore = "OPEN F2.3-07 (Línea C, C-F2.3): con fases de arranque arbitrarias el IC cruzado medio es ≈ 0,75·ρ"]
fn cf2_ic_cruzado_estima_rho_con_fase_aleatoria() {
    let mut g = Lcg(0xC0F2_0403);
    let tau_s = (SPECTRUM_SCALES_MS[ESCALA] / DT as f64).ceil() as u64;
    let desfases: Vec<u64> = (0..6).map(|_| (g.u() * tau_s as f64) as u64).collect();
    let ics = ic_cruzado(0xC0F2_0404, &desfases, 36_000, 14_400);
    let media = ics.iter().sum::<f64>() / ics.len() as f64;
    let esperado_hoy: f64 = desfases
        .iter()
        .map(|&d| RHO * (d.max(tau_s - d) as f64) / tau_s as f64)
        .sum::<f64>()
        / desfases.len() as f64;
    assert!(
        (media - RHO).abs() < 0.10,
        "IC cruzado medio con fases aleatorias {media:.3} (por moneda {ics:?}, desfases {desfases:?} s); verdad 0,6; solape medio emparejado da {esperado_hoy:.3}"
    );
}
