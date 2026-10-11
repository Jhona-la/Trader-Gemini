//! C-F2.3b (AGY, pedido con test): Yang-Mills con retornos asíncronos y con
//! una dislocación triangular plantada.
//!
//! Teoría que se prueba. El motor define A_ij = r_i − β_ij·r_j sobre
//! log-retornos y F_ijk = A_ij + A_jk + A_ki (yang_mills_gauge.rs:184-190).
//! Reordenando, F_ijk = r_i(1 − β_ki) + r_j(1 − β_ij) + r_k(1 − β_jk): con
//! β ≡ 1 (el valor de arranque, :65-74, y ningún código de producción llama
//! a with_beta) la conexión es gauge puro y F ≡ 0 para CUALQUIER vector de
//! retornos. Sólo la deriva del LMS de β (≈ γ·r² ≈ 5e-9 por evento) deja
//! curvatura, de orden r·(1 − β). Por tanto:
//! - con retornos asíncronos la acción y la corriente son prácticamente
//!   nulas (respuesta a la pregunta del plan, LC-F2-06);
//! - una dislocación triangular plantada (el residuo de cointegración de tres
//!   precios en USDT) es invisible: la acción media queda ~5 órdenes por
//!   debajo de la varianza del residuo (F2.3b-07).
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use strategy_core::yang_mills_gauge::YangMillsGaugeEngine;

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1)
/// y normal estándar por Box-Muller.
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

/// Mayor |1 − β| de los tres pares del triángulo (0, 1, 2).
fn desviacion_beta(ym: &YangMillsGaugeEngine) -> f64 {
    let b = &ym.beta_matrix;
    [b[0][1], b[0][2], b[1][2], b[1][0], b[2][0], b[2][1]]
        .iter()
        .map(|x| (1.0 - x).abs())
        .fold(0.0, f64::max)
}

/// Nulo asíncrono. Fase 1, el caso límite: en cada llamada se mueve una sola
/// moneda (turno rotatorio) con un log-retorno de ±1e-3. β_ij sólo decae,
/// como β(1 − γr²), cuando se mueve la moneda de índice mayor del par: tras
/// 10 000 movimientos 1 − β ≈ 10 000·0,005·1e-6 = 5e-5, |F| = |r(1 − β)| ≈ 5e-8,
/// S = ½F² ≈ 1,25e-15 y |J| = tanh|F| ≈ 5e-8.
///
/// Corrección del escéptico: en vivo no cambia sólo la moneda del evento.
/// latest_prices se actualiza en todo evento (depth incluido) y YM sólo se
/// evalúa en trades y klines, así que entre dos llamadas pueden moverse
/// varias monedas. Fase 2: cada moneda se mueve con probabilidad ½ en cada
/// llamada. La deriva de β es del mismo orden (1 − β ≲ 1e-4) y
/// |F| ≤ Σ|r|·|1 − β| ≲ 3e-7. Cotas: S < 1e-12 (margen > 40×) y |J| < 1e-6
/// (margen > 3×); la prueba es determinista, sin error estadístico.
#[test]
fn cf2_ym_retornos_asincronos_son_inertes() {
    let mut rng = Lcg(0x9A6E);
    for fase in 0..2 {
        let mut ym = YangMillsGaugeEngine::new(3);
        let mut p = [100.0f64, 50.0, 2.0];
        ym.update_and_calculate_curvature(&p);
        let (mut max_s, mut max_j) = (0.0f64, 0.0f64);
        for t in 0..30_000usize {
            for c in 0..3 {
                let mueve = if fase == 0 { t % 3 == c } else { rng.u() < 0.5 };
                if mueve {
                    let signo: f64 = if rng.u() < 0.5 { 1.0 } else { -1.0 };
                    p[c] *= (1e-3 * signo).exp();
                }
            }
            let (s, j) = ym.update_and_calculate_curvature(&p);
            max_s = max_s.max(s);
            max_j = max_j.max(j[0].abs()).max(j[1].abs()).max(j[2].abs());
        }
        println!(
            "fase {fase}: acción máxima {max_s:e}, corriente máxima {max_j:e}, |1-beta| {:e}",
            desviacion_beta(&ym)
        );
        assert!(
            max_s < 1e-12 && max_j < 1e-6,
            "fase {fase}: acción máxima {max_s:e}, corriente máxima {max_j:e}, |1-beta| {:e}",
            desviacion_beta(&ym)
        );
    }
}

/// Dislocación triangular plantada. Dos monedas con retornos síncronos
/// independientes (σ = 1e-3) y una tercera alimentada como PRECIO EN USDT
/// p2 = p0/p1·e^u (no es un par cruzado negociado), con u un AR(1) de φ = 0,9
/// y σ_u = 1e-3: un residuo de cointegración que revierte, con
/// Var(Δu) = 2σ_u²(1 − φ) = 2e-7 por paso. Con numerario común no existe una
/// holonomía observable; ésta es la única estructura triangular disponible,
/// y un instrumento que la viera daría una acción del orden de 2e-7.
///
/// El umbral 1e-9 no es estadístico: es la promesa del doc del módulo
/// («energía de dislocación o tensión de arbitraje», yang_mills_gauge.rs:26-28),
/// dos órdenes por debajo de Var(Δu). Hoy F = Σ r(1 − β) con el LMS a un ritmo
/// γ·Var(r) ≈ 1e-8 por paso: tras 20 000 pasos 1 − β ≈ 1e-4 y la acción
/// media queda en 1e-14 a 1e-13, de 4 a 5 órdenes por debajo del umbral.
#[test]
#[ignore = "OPEN F2.3b-07 (AGY): con beta = 1 la curvatura es gauge puro, F = suma de r(1-beta), y la dislocación plantada no entra"]
fn cf2_ym_dislocacion_triangular_plantada_se_detecta() {
    let mut ym = YangMillsGaugeEngine::new(3);
    let mut rng = Lcg(0xD15C);
    let (mut l0, mut l1, mut u) = (100f64.ln(), 50f64.ln(), 0.0f64);
    let precios = |l0: f64, l1: f64, u: f64| [l0.exp(), l1.exp(), (l0 - l1 + u).exp()];
    ym.update_and_calculate_curvature(&precios(l0, l1, u));
    let n = 20_000usize;
    let (mut s, mut var_du) = (0.0, 0.0);
    for _ in 0..n {
        l0 += 1e-3 * rng.n();
        l1 += 1e-3 * rng.n();
        let u_prev = u;
        u = 0.9 * u + (1.0f64 - 0.81).sqrt() * 1e-3 * rng.n();
        var_du += (u - u_prev) * (u - u_prev);
        s += ym.update_and_calculate_curvature(&precios(l0, l1, u)).0;
    }
    let media = s / n as f64;
    assert!(
        media > 1e-9,
        "acción media {media:e} con una dislocación de Var(du) {:e} por paso; |1-beta| {:e}",
        var_du / n as f64,
        desviacion_beta(&ym)
    );
}
