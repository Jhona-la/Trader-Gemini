//! C-F2.3a (Línea C): sesgo del pronóstico de σ (`SpectralForecastBank`)
//! frente a una σ conocida.
//!
//! Prueba de solución conocida de la fase C-F2 (ver
//! `docs/audit/HALLAZGOS_C-F2_2026-10-11.md`). Con n incrementos gaussianos
//! de σ conocida, la varianza realizada cumple RV/σ² ~ χ²_n/n y
//! E[ln σ̂ − ln σ] = ½·(ψ(n/2) + ln 2 − ln n): −0,1352 con n = 4. El banco
//! pronostica ln RV (lo correcto para la media de ln RV) y `sigma_at` devuelve
//! √exp(pronóstico), que es la MEDIANA de σ̂, no σ: sin corrección de Jensen el
//! sesgo en ln σ es −0,135 con cuatro trades por ventana del ancla de 17 s.
//!
//! Dos generadores, ambos con semilla fija (LCG + Box-Muller propios):
//! - σ constante: aísla el sesgo de Jensen en forma cerrada. En ese mundo el
//!   banco no bate a la climatología y el núcleo no publica el pronóstico.
//! - σ por regímenes con calendario conocido (1× y 3× cada 30 min): el banco
//!   bate al mejor nulo, el núcleo SÍ publicaría el pronóstico, y se mide el
//!   sesgo y la cobertura del intervalo ±1,645·σ̂ frente a la σ condicional
//!   verdadera.
//!
//! Convención: un test que documenta un defecto confirmado lleva
//! `#[ignore = "OPEN F2.x-nn (dueño): causa"]` con el assert intacto. Falla
//! hoy por la causa del hallazgo y se activa (se quita el `ignore`) con el
//! arreglo. Los tests sin `ignore` son regresiones de la teoría y deben pasar.

use quantum_arena::spectral_tape::{SpectralForecastBank, ANCLAS_PRONOSTICO};

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
const P0: f64 = 100.0;
/// Un trade cada 4 294 ms: la ventana (t, t + 17 179 ms] del ancla 17 contiene
/// exactamente cuatro incrementos (4·4 294 = 17 176).
const PASO_MS: u64 = 4_294;
/// σ por √ms: 0,2 % en la escala de 68,7 s.
fn sigma_ms() -> f64 {
    2.0e-3 / 68_719.0f64.sqrt()
}

// Valor verdadero: σ del ancla 17 = σ_ms·√(4·4 294). Contrato: |sesgo log| <
// 0,05. Hoy: −0,1352 (Jensen con n = 4). Error típico: la media es de
// PRONÓSTICOS y su error lo fija la memoria del ajuste (≈ 256 muestras de
// h/4 ≈ 18 min); con 24 000 trades (28,6 h) la segunda mitad abarca ≈ 47
// memorias y el SE del sesgo medio es ≈ 0,01, así que 0,05 son ≈ 5 SE.
// En este mundo habilidad_volatilidad() ≤ 0 y el núcleo publica ceros: la
// prueba aísla el sesgo; la siguiente lo mide donde el núcleo sí publica.
#[test]
#[ignore = "OPEN F2.3-06 (Línea C, C-F2.3 fila 3): sigma_at devuelve la mediana √exp(E ln RV), sin corrección de Jensen"]
fn cf2_sigma_pronosticada_es_insesgada_con_cuatro_trades_por_ventana() {
    let sd_paso = sigma_ms() * (PASO_MS as f64).sqrt();
    let verdad17 = sigma_ms() * (4.0 * PASO_MS as f64).sqrt();
    let mut g = Lcg(0xC0F2_0301);
    let mut bank = SpectralForecastBank::new();
    let mut x = 0.0f64;
    let total = 24_000usize;
    let (mut suma, mut n) = (0.0f64, 0usize);
    for i in 0..total {
        x += sd_paso * g.n();
        let ts = T0 + i as u64 * PASO_MS;
        bank.on_trade(ts, P0 * x.exp(), 1.0, i % 2 == 0);
        if i >= total / 2 {
            if let Some(s) = bank.sigma_at(ANCLAS_PRONOSTICO[0], ts) {
                if s > 0.0 {
                    suma += (s / verdad17).ln();
                    n += 1;
                }
            }
        }
    }
    assert!(n > 10_000, "el ancla 17 no maduró ({n})");
    let b17 = suma / n as f64;
    assert!(
        b17.abs() < 0.05,
        "sesgo log de σ en el ancla 17 = {b17:.4}; Jensen con n = 4 da −0,1352; habilidad del banco {:?}",
        bank.habilidad_volatilidad()
    );
}

// Corrección del escéptico COMPLETITUD: con σ constante el banco no bate al
// mejor nulo y el núcleo publica ceros, así que el sesgo anterior nunca llega
// a vivo. Aquí σ alterna entre 1× y 3× cada 30 min con calendario conocido.
// Precondición: habilidad_volatilidad() > 0 (el núcleo publicaría σ).
// Se miden las ventanas (t, t + h] enteras dentro de un régimen y a ≥ 5 min de
// su arranque (el pronóstico ya vio el régimen a la escala base de 17 s); la σ
// condicional verdadera es σ_régimen·√(4·4 294).
// Valor verdadero de la cobertura: P(|r_h| ≤ 1,645·σ_h) = 0,90. Con ≈ 4 900
// ventanas no solapadas el SE binomial es 0,0041; tolerancia ±0,02 (≈ 5 SE).
// Un σ̂ insesgado con ruido de pronóstico de sd ≈ 0,1 en ln σ sólo resta
// ≈ 0,005 a la cobertura (concavidad de 2Φ(1,645·e^u) − 1).
// Hoy, con σ̂ = σ·e^{−0,135}, se espera 2Φ(1,645·0,8735) − 1 = 0,849
// (≈ 12 SE por debajo). Se exige además |sesgo log| < 0,05 (≈ 0,01 de SE).
//
// Revisión cruzada (CL-65): el ruido del pronóstico observado aquí (sd ≈ 0,2
// en ln σ, no 0,1) resta ≈ 0,29·Var(u) incluso con σ̂ insesgado, y dejaba la
// cobertura esperada tras el arreglo en ≈ 0,888, a 2 SE del borde. La
// referencia es ahora el desarrollo de f(u) = 2Φ(1,645·e^u) − 1 en u = 0 con
// los momentos medidos de u = ln(σ̂/σ) en las mismas ventanas:
// f ≈ 0,90 + 0,339·E[u] − 0,289·E[u²] (f′(0) = 2φ(1,645)·1,645,
// f″(0)/2 = φ(1,645)·1,645·(1 − 1,645²)). Tolerancia 4·√(0,09/ventanas).
// Con σ̂ = mediana, E[u] ≈ −0,135 y la cobertura cae igual que la
// referencia, así que el defecto queda en la aserción del sesgo.
#[test]
#[ignore = "OPEN F2.3-06 (Línea C, C-F2.3 fila 3): la σ publicada es la mediana; el intervalo de 1,645σ̂ cubre ≈ 85 %, no 90 %"]
fn cf2_sigma_pronosticada_cubre_el_90_con_volatilidad_por_regimenes() {
    let regimen_ms: u64 = 1_800_000;
    let multiplicadores = [1.0f64, 3.0];
    let mult_en = |ts: u64| multiplicadores[(((ts - T0) / regimen_ms) % 2) as usize];
    let h17 = ANCLAS_PRONOSTICO[0];
    let mut g = Lcg(0xC0F2_0302);
    let mut bank = SpectralForecastBank::new();
    let total = 36_000usize; // 43 h
    let mut xs = Vec::with_capacity(total);
    let mut x = 0.0f64;
    for i in 0..total {
        let ts = T0 + i as u64 * PASO_MS;
        // El incremento que llega en ts pertenece al régimen vigente en ts.
        x += mult_en(ts) * sigma_ms() * (PASO_MS as f64).sqrt() * g.n();
        xs.push(x);
    }
    let mut pronosticos: Vec<Option<f64>> = Vec::with_capacity(total);
    for (i, &xi) in xs.iter().enumerate() {
        let ts = T0 + i as u64 * PASO_MS;
        bank.on_trade(ts, P0 * xi.exp(), 1.0, i % 2 == 0);
        pronosticos.push(bank.sigma_at(h17, ts));
    }
    let habilidad = bank.habilidad_volatilidad();
    assert!(
        habilidad.map_or(false, |h| h > 0.0),
        "precondición: el banco debe batir al mejor nulo, habilidad = {habilidad:?}"
    );
    let (mut suma_log, mut n_log) = (0.0f64, 0usize);
    let (mut cubre, mut ventanas) = (0usize, 0usize);
    let (mut suma_u, mut suma_u2) = (0.0f64, 0.0f64);
    let desde = total / 3;
    for i in desde..total - 4 {
        let ts = T0 + i as u64 * PASO_MS;
        let fase = (ts - T0) % regimen_ms;
        let ts_fin = T0 + (i + 4) as u64 * PASO_MS;
        let mismo_regimen = (ts - T0) / regimen_ms == (ts_fin - T0) / regimen_ms;
        if !mismo_regimen || fase < 300_000 {
            continue;
        }
        let Some(s) = pronosticos[i] else { continue };
        let verdad = mult_en(ts) * sigma_ms() * (4.0 * PASO_MS as f64).sqrt();
        suma_log += (s / verdad).ln();
        n_log += 1;
        if i % 4 == 0 {
            let r = xs[i + 4] - xs[i];
            ventanas += 1;
            let u = (s / verdad).ln();
            suma_u += u;
            suma_u2 += u * u;
            if r.abs() <= 1.645 * s {
                cubre += 1;
            }
        }
    }
    let sesgo = suma_log / n_log as f64;
    let cobertura = cubre as f64 / ventanas as f64;
    let (eu, eu2) = (suma_u / ventanas as f64, suma_u2 / ventanas as f64);
    let referencia = 0.90 + 0.339 * eu - 0.289 * eu2;
    let tol = 4.0 * (0.09 / ventanas as f64).sqrt();
    println!(
        "habilidad {habilidad:?}; sesgo log {sesgo:.4} (n = {n_log}); cobertura {cobertura:.4} ({cubre}/{ventanas});          E[u] {eu:.4}, E[u²] {eu2:.4}, referencia {referencia:.4} ± {tol:.4}"
    );
    assert!(
        (cobertura - referencia).abs() <= tol && sesgo.abs() < 0.05,
        "cobertura de ±1,645σ̂ = {cobertura:.4} ({cubre}/{ventanas}) frente a la referencia {referencia:.4} ± {tol:.4}          (E[u] {eu:.4}, E[u²] {eu2:.4}); sesgo log = {sesgo:.4} (Jensen −0,1352); habilidad del banco {habilidad:?}"
    );
}
