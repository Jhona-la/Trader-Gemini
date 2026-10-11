//! C-F2.2 (Línea C): e-procesos de Ville y cuantiles P² frente a soluciones
//! conocidas.
//!
//! Prueba de solución conocida de la fase C-F2 (ver
//! `docs/audit/HALLAZGOS_C-F2_2026-10-11.md`). Cada test alimenta a la API
//! pública un proceso cuyo valor verdadero se conoce en forma cerrada:
//!
//! - `EProceso` (evalues.rs): bajo H0 exacta (signo equiprobable) el capital
//!   es martingala y Ville acota P(sup e ≥ 1/α) ≤ α; con un acierto de signo
//!   conocido (p = 0,55, o el p = ½ + asin(ρ)/π de dos gaussianas con
//!   correlación ρ) la deriva por bloque del log-capital es conocida y con
//!   ella la potencia frente al umbral de familia M/α.
//! - `AdaptiveQuantileEngine` (adaptive_quantiles.rs): el p80 de |N(0,σ²)|
//!   vale 1,2816·σ, de modo que la tasa de disparo del umbral leído ANTES de
//!   actualizar (como en el núcleo) es 0,20 por definición; con el
//!   multiplicador de intermitencia del núcleo, la cola gaussiana da la tasa
//!   en forma cerrada.
//!
//! Generadores propios (LCG + Box-Muller) con semilla fija: los tests son
//! deterministas y no usan dependencias nuevas.
//!
//! Convención: un test que documenta un defecto confirmado lleva
//! `#[ignore = "OPEN F2.x-nn (dueño): causa"]` con el assert intacto. Falla
//! hoy por la causa del hallazgo y se activa (se quita el `ignore`) con el
//! arreglo. Los tests sin `ignore` son regresiones de la teoría y deben pasar.

use quantum_arena::evalues::EProceso;
use quantum_arena::AdaptiveQuantileEngine;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Tamaño de la familia que consulta el consenso de motores (13 motores ×
/// 32 escalas, skill_motores.rs): umbral de capital M/α = 8 320.
const FAMILIA_MOTORES: usize = 13 * 32;

// Valor verdadero: con factor 1 ± 0,1 y signo equiprobable el capital es una
// martingala; Ville acota P(cruce de 20) ≤ α = 0,05. La enumeración exacta de
// los 5 000 pasos da 0,0479 (el sobretiro deja el valor justo bajo α).
// Tolerancia: α + 4 errores típicos binomiales con 2 000 caminos,
// 0,05 + 4·√(0,05·0,95/2000) = 0,0695.
#[test]
fn cf2_eproceso_nivel_bajo_el_nulo() {
    let mut g = Lcg::new(661);
    let (reps, t_max) = (2_000usize, 5_000usize);
    let mut cruza = 0usize;
    for _ in 0..reps {
        let mut e = EProceso::new();
        for _ in 0..t_max {
            e.observar(1.0, if g.u() < 0.5 { 1.0 } else { -1.0 });
            if e.significativo() {
                cruza += 1;
                break;
            }
        }
    }
    let nivel = cruza as f64 / reps as f64;
    let cota = 0.05 + 4.0 * (0.05f64 * 0.95 / reps as f64).sqrt();
    println!("nivel bajo H0: {nivel:.4} ({cruza}/{reps})");
    assert!(
        nivel <= cota,
        "nivel bajo H0 = {nivel:.4} ({cruza}/{reps}); Ville exige ≤ 0,05 (cota con 4 SE {cota:.4}); exacto 0,0479"
    );
}

// Valor verdadero: con (z1, ρ·z1 + √(1−ρ²)·z2) y ρ = 0,05 el IC es 0,05 y el
// acuerdo de signo es p = ½ + asin(ρ)/π = 0,5159. Contrato (corrección del
// escéptico REFUTAR): con T = 60 000 bloques, un e-proceso con potencia
// razonable (mezcla o λ GRO, ≈ 26 nats de crecimiento frente a
// ln 8 320 = 9,03) certifica con potencia ≈ 0,99; se exige ≥ 0,80, que deja
// > 5 errores típicos (√(0,8·0,2/200) = 0,028) de margen.
// Con el λ fijo de 0,10 la deriva es p·ln 1,1 + (1−p)·ln 0,9 = −0,0018 nats
// por bloque (−110 nats en 60 000): hoy la potencia es ≈ 0,03.
// Nota: con IC 0,05 ningún e-proceso certifica dentro de la familia de 416 en
// menos de ~2·10⁴ bloques. En tiempo físico, 60 000 bloques son 21 días a la
// escala de 30 s, 48 días a la de 69 s y 7 meses a la de 4,6 min; el
// siguiente paso tiene que incluir e-BH o reducir M.
#[test]
#[ignore = "OPEN F2.2-04 (Línea C, C-F2.2): λ fijo 0,10 da deriva negativa con acierto < 0,5251; IC 0,05 no despega"]
fn cf2_eproceso_detecta_ic_005_en_la_familia_de_416() {
    let rho = 0.05f64;
    let mut g = Lcg::new(663);
    let (reps, t_max) = (200usize, 60_000usize);
    let mut cruza = 0usize;
    let mut ln_capital_medio = 0.0f64;
    for _ in 0..reps {
        let mut e = EProceso::new();
        for _ in 0..t_max {
            let (z1, z2) = (g.normal(), g.normal());
            e.observar(z1, rho * z1 + (1.0 - rho * rho).sqrt() * z2);
            if e.significativo_familia(FAMILIA_MOTORES) {
                cruza += 1;
                break;
            }
        }
        ln_capital_medio += e.capital().ln() / reps as f64;
    }
    let potencia = cruza as f64 / reps as f64;
    assert!(
        potencia >= 0.80,
        "potencia {potencia:.3} ({cruza}/{reps}) con IC 0,05 en 60 000 bloques; ln capital medio al final {ln_capital_medio:.1} nats (umbral ln 8320 = 9,03; deriva λ fijo −0,0018/bloque)"
    );
}

// Valor verdadero: acierto de signo p = 0,55 ⇒ deriva del log-capital
// 0,55·ln 1,1 − 0,45·|ln 0,9| = +0,00501 nats por bloque con sd ≈ 0,0998.
// A 3 000 bloques ln W tiene media 15,0 y sd 5,5 frente a ln 8 320 = 9,03; la
// enumeración exacta del cruce da potencia 0,9096. Tolerancia: ≥ 0,85, a 4,2
// errores típicos con 400 caminos (√(0,91·0,09/400) = 0,0143).
#[test]
fn cf2_eproceso_detecta_p_055_en_la_familia_de_416() {
    let mut g = Lcg::new(55);
    let (reps, t_max) = (400usize, 3_000usize);
    let mut cruza = 0usize;
    for _ in 0..reps {
        let mut e = EProceso::new();
        for _ in 0..t_max {
            e.observar(1.0, if g.u() < 0.55 { 1.0 } else { -1.0 });
            if e.significativo_familia(FAMILIA_MOTORES) {
                cruza += 1;
                break;
            }
        }
    }
    let potencia = cruza as f64 / reps as f64;
    println!("potencia con p = 0,55: {potencia:.4} ({cruza}/{reps})");
    assert!(
        potencia >= 0.85,
        "potencia {potencia:.3} ({cruza}/{reps}) con p = 0,55 en 3 000 bloques; teoría ≈ 0,92"
    );
}

/// Tasa de disparo del piso de OBI: umbral `dynamic_obi_threshold()·mult`
/// leído ANTES de actualizar el estimador, como el núcleo
/// (god-engine-core lib.rs, `piso_obi_medido` y la actualización D-756).
/// El multiplicador de intermitencia 1 + 0,5·χ se replica aquí porque
/// `intermittency_mult_de` es privado del núcleo. 200 000 muestras |N(0,1)|
/// de calentamiento; después 20 000 de |N(0, σ²)| y se cuenta la tasa en las
/// 10 000 últimas.
fn tasa_de_disparo(mult: f64, sigma_despues: f64) -> f64 {
    let mut g = Lcg::new(42);
    let mut q = AdaptiveQuantileEngine::new();
    for _ in 0..200_000 {
        q.update(0.0, g.normal().abs(), 0.0, 0.0);
    }
    let mut disparos = 0usize;
    for t in 0..20_000 {
        let x = (sigma_despues * g.normal()).abs();
        if t >= 10_000 && x >= q.dynamic_obi_threshold() * mult {
            disparos += 1;
        }
        q.update(0.0, x, 0.0, 0.0);
    }
    disparos as f64 / 10_000.0
}

// Valor verdadero: el p80 de |N(0,1)| es 1,2816 y la tasa de excedencia es
// 0,20 por definición. Tolerancia ±0,02 = 5 errores típicos binomiales con
// 10 000 pruebas (√(0,2·0,8/10 000) = 0,004).
#[test]
fn cf2_p2_tasa_de_disparo_iid_en_el_nominal() {
    let t = tasa_de_disparo(1.0, 1.0);
    println!("tasa de disparo iid: {t:.4}");
    assert!(
        (t - 0.20).abs() <= 0.02,
        "tasa {t:.4} con datos iid; nominal 0,20"
    );
}

// Corrección de los dos escépticos: el 0,20 era la etiqueta («percentil 80
// medido»), no la intención del código; el multiplicador 1 + 0,5·χ endurece
// el piso a propósito (lib.rs, Ola XLI·C2). Mientras el dueño no declare la
// tasa objetivo r(χ), la prueba fija la tasa analítica del diseño vivo:
// 2·(1 − Φ((1 + 0,5χ)·z₀,₈)) con χ = 0,4 vale 0,1241, que NO es un p80.
// Tolerancia ±0,02 = 6 errores típicos binomiales (√(0,124·0,876/10 000)
// = 0,0033). Mide el piso; el umbral vivo es max(gen, piso) y su tasa depende
// además de la curva del genoma.
#[test]
fn cf2_umbral_obi_con_intermitencia_sigue_la_cola_analitica() {
    let chi = 0.4f64;
    let t = tasa_de_disparo(1.0 + 0.5 * chi, 1.0);
    let analitica = 0.124_082_4;
    println!("tasa de disparo con χ = 0,4: {t:.4} (analítica {analitica:.4})");
    assert!(
        (t - analitica).abs() <= 0.02,
        "tasa {t:.4} con el piso p80 multiplicado por 1,2; cola analítica {analitica:.4} (no es un p80)"
    );
}

// Valor verdadero: tras duplicar σ, el p80 de |N(0,2²)| vale 2,563 y la tasa
// nominal vuelve a 0,20. Contrato (presupone elegir el estimando RECIENTE,
// fila 42 del plan; choca con quantile_numeric_contract.rs, que fija hoy el
// estimando acumulativo): tras 10 000 observaciones del régimen nuevo la tasa
// es 0,20 ± 0,02 (5 SE). El estimador no lleva reloj; al ritmo del núcleo (una
// actualización por evento de libro, del orden de 10/s por símbolo) 10 000
// observaciones son ≈ 17 min.
// Hoy: P² acumulativo ⇒ el p80 de la mezcla sigue en ≈ 1,38 y
// P(|N(0,2²)| > 1,38) ≈ 0,49.
#[test]
#[ignore = "OPEN F2.2-08 (Línea C, C-F2.2): P² acumulativo no olvida; tras σ×2 el «p80 medido» dispara ≈ 49 %"]
fn cf2_p2_tras_cambio_de_regimen_vuelve_al_nominal() {
    let t = tasa_de_disparo(1.0, 2.0);
    assert!(
        (t - 0.20).abs() <= 0.02,
        "tasa {t:.4} tras duplicar sigma; nominal 0,20, mezcla acumulativa ≈ 0,49"
    );
}
