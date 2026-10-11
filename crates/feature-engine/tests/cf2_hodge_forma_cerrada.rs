//! C-F2.3b (AGY por historia, Ω37; pedido con test): el `curl_share` de la
//! descomposición de Helmholtz-Hodge sobre el flujo cruzado OFI × retorno de
//! `hodge_flow.rs` es una forma cerrada de momentos transversales.
//!
//! Teoría que se prueba. Con a = OFI y b = tanh(100·r), el constructor da
//! F_ij = ½(a_i b_j − a_j b_i), la parte antisimétrica de un producto
//! exterior de rango 2. Sobre el grafo completo K_n:
//! - energía total (identidad de Lagrange): T = ¼(|a|²|b|² − (a·b)²);
//! - divergencia div_i = (n/2)(a_i b̄ − b_i ā), luego la energía del
//!   gradiente es G = (n/4)Σ(b̄a′_i − āb′_i)², con a′ = a − ā y b′ = b − b̄;
//! - el resto, C = ¼(|a′|²|b′|² − (a′·b′)²), es exactamente a′∧b′, de
//!   divergencia nula: curl_share = C/(C + G).
//!
//! Ley nula (corrección del escéptico): con a y b gaussianos independientes,
//! (ā, b̄) es independiente de (a′, b′), la matriz de Gram de (a′, b′) es una
//! Wishart₂(n − 1, I) y G/C = z′W⁻¹z = 2F/(n − 2) con F ~ F(2, n − 2)
//! (Hotelling). De ahí curl_share ~ Beta((n − 2)/2, 1): media (n − 2)/n y
//! P(curl ≤ x) = x^{(n−2)/2}. Ruido independiente, sin ninguna circulación ni
//! dislocación económica, da curl ≈ 0,93 con n = 30. La descomposición es
//! correcta (a′∧b′ ES rotacional de Hodge); lo que falsifica es la lectura
//! «curl → 1 = vórtice cerrado / reversión / arbitraje» (doc de
//! `hodge_flow.rs` y del constructor) y su consumidor en el consejo
//! (SeniorMicroestructura, god-engine-core), cuya prueba queda fuera de este
//! crate.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use feature_engine::hodge_flow::HelmholtzHodgeFlowEngine;

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

/// (T, G, curl) por forma cerrada con a = OFI y b = tanh(100 r).
fn cerrada(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut saa, mut sbb, mut sab, mut g) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        saa += da * da;
        sbb += db * db;
        sab += da * db;
        g += (mb * da - ma * db).powi(2);
    }
    let c = 0.25 * (saa * sbb - sab * sab);
    let g = 0.25 * n * g;
    (c + g, g, c / (c + g))
}

/// Verdad: las tres identidades algebraicas, en 200 casos con n de 3 a 30 y
/// OFI con media 0 o 0,7. Tolerancias: T relativa 1e-10 (corrección del
/// escéptico: la forma por momentos resta AB − C² y pierde dígitos cuando
/// |ρ| es alto), G 1e-10·T y curl 1e-9 absoluta; sólo cubren redondeo.
#[test]
fn cf2_hodge_curl_share_es_funcion_de_momentos_transversales() {
    let mut rng = Lcg(0x40D6E);
    let eng = HelmholtzHodgeFlowEngine::new(32);
    let mut peor = 0.0f64;
    for caso in 0..200usize {
        let n = 3 + caso % 28;
        let sesgo = if caso % 2 == 0 { 0.7 } else { 0.0 };
        let ofi: Vec<f64> = (0..n).map(|_| rng.n() + sesgo).collect();
        let ret: Vec<f64> = (0..n).map(|_| 1e-4 * rng.n()).collect();
        let b: Vec<f64> = ret.iter().map(|r| (100.0 * r).tanh()).collect();
        let (m, k) =
            HelmholtzHodgeFlowEngine::build_cross_microstructure_flow_matrix(&ofi, &ret).unwrap();
        let (d, _) = eng.decompose(&m, k).expect("flujo no nulo");
        assert_eq!(d.num_nodes, n, "caso {caso}: todos los nodos tienen flujo");
        let (t, g, curl) = cerrada(&ofi, &b);
        peor = peor.max((d.curl_share - curl).abs());
        assert!(
            (d.total_energy - t).abs() <= 1e-10 * t,
            "caso {caso}: T {} frente a {t}",
            d.total_energy
        );
        assert!(
            (d.gradient_energy - g).abs() <= 1e-10 * t,
            "caso {caso}: G {} frente a {g}",
            d.gradient_energy
        );
        assert!(
            (d.curl_share - curl).abs() < 1e-9,
            "caso {caso}: {} frente a {curl}",
            d.curl_share
        );
    }
    eprintln!("[cf2 hodge] peor |curl − cerrada| = {peor:e}");
}

/// Verdad (ley nula exacta): con n = 30 y OFI, retornos gaussianos
/// independientes, curl_share ~ Beta(14, 1): media 14/15 = 0,9333,
/// sd = √(14/(15²·16)) = 0,0624 y P(curl ≤ 0,9) = 0,9¹⁴ = 0,2288.
/// Tolerancias con m = 2000 universos: media ± 4 SE = 4·0,0624/√2000 =
/// 0,0056; fracción ± 4 SE binomiales = 4·√(0,2288·0,7712/2000) = 0,0376.
/// tanh(100 r) con 100 r ~ N(0, 1e-4) es gaussiano salvo un término cúbico
/// relativo de ≈ 3e-5, despreciable. Sustituye a la aserción original
/// «media < 0,5» (umbral no derivado; ver el //! del archivo).
#[test]
fn cf2_hodge_ruido_independiente_da_curl_cercano_a_uno() {
    let mut rng = Lcg(0xC011);
    let eng = HelmholtzHodgeFlowEngine::new(32);
    let (n, m) = (30usize, 2_000usize);
    let alfa = (n as f64 - 2.0) / 2.0;
    let (mut suma, mut bajo_09) = (0.0, 0usize);
    for _ in 0..m {
        let ofi: Vec<f64> = (0..n).map(|_| rng.n()).collect();
        let ret: Vec<f64> = (0..n).map(|_| 1e-4 * rng.n()).collect();
        let (mat, k) =
            HelmholtzHodgeFlowEngine::build_cross_microstructure_flow_matrix(&ofi, &ret).unwrap();
        let curl = eng.decompose(&mat, k).unwrap().0.curl_share;
        suma += curl;
        if curl <= 0.9 {
            bajo_09 += 1;
        }
    }
    let media = suma / m as f64;
    let esperada = alfa / (alfa + 1.0);
    let sd = (alfa / ((alfa + 1.0).powi(2) * (alfa + 2.0))).sqrt();
    let se = sd / (m as f64).sqrt();
    let frac = bajo_09 as f64 / m as f64;
    let p09 = 0.9f64.powf(alfa);
    let se_p = (p09 * (1.0 - p09) / m as f64).sqrt();
    eprintln!(
        "[cf2 hodge] curl medio {media:.4} frente a (n−2)/n = {esperada:.4} (se {se:.4}); \
         P(curl ≤ 0,9) {frac:.4} frente a {p09:.4}"
    );
    assert!(
        (media - esperada).abs() < 4.0 * se,
        "curl_share medio {media:.4} sin circulación frente a (n−2)/n = {esperada:.4} ± {:.4}",
        4.0 * se
    );
    assert!(
        (frac - p09).abs() < 4.0 * se_p,
        "P(curl ≤ 0,9) {frac:.4} frente a 0,9^{alfa} = {p09:.4} ± {:.4}",
        4.0 * se_p
    );
}
