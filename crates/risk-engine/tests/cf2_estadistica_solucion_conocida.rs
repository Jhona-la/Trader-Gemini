//! C-F2.2 (Línea C): estadística de risk-engine contra soluciones conocidas.
//!
//! Pruebas de SOLUCIÓN CONOCIDA de la fase C-F2 (hallazgos en
//! `docs/audit/HALLAZGOS_C-F2_2026-10-11.md`, dominio F2.2). Cada prueba usa
//! un generador cuyo parámetro verdadero está fijado por construcción y mide
//! el código contra la propiedad que ese parámetro implica:
//!
//! - Cotas inferiores (`evidence.rs`): con PF real 1 o win rate p conocido,
//!   una cota al 95 % debe quedar por debajo del valor verdadero en ≥ 95 % de
//!   las muestras (binomial exacta o Monte Carlo).
//! - E-proceso de Ville (`ville_e_process.rs`) y el predicado vivo de rollback
//!   (`online_daemon.rs:1205-1241`): sobre retornos de media ≥ 0 todo rollback
//!   es falso (nivel ≤ α); con E[X] > 0 un e-proceso de crecimiento debe
//!   certificar.
//! - DSR (`selection_stats.rs`): nivel y potencia frente al Šidák exacto del
//!   máximo de N gaussianas independientes, y frente a su propia fórmula.
//!
//! Generadores propios (LCG de Knuth, Box-Muller y exponencial por inversión),
//! semillas fijas y sin dependencias nuevas.
//!
//! Convención: una prueba que hoy falla por la causa de un hallazgo abierto
//! lleva `#[ignore = "OPEN F2.x-nn (dueño): causa"]` con el assert intacto.
//! Documenta el defecto y se activa (se quita el ignore) con el arreglo. Las
//! pruebas activas son regresiones de la teoría y deben pasar siempre.

use quantum_arena::genome::SuperGenotype;
use risk_engine::evidence::{profit_factor_lcb, win_rate_hierarchical_lcb, win_rate_lcb};
use risk_engine::selection_stats::{
    edge_survives_multiplicity, expected_max_sharpe, inverse_normal_cdf, normal_cdf,
};
use risk_engine::VilleEProcess;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// Uniforme en (0, 1) abierto.
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Normal estándar por Box-Muller.
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Exponencial de media 1 por inversión.
    fn exp1(&mut self) -> f64 {
        -self.u().ln()
    }
}

/// Probabilidades binomiales Bin(n, p) por recurrencia.
fn binom_pmf(n: usize, p: f64) -> Vec<f64> {
    let mut v = vec![0.0; n + 1];
    v[0] = (1.0 - p).powi(n as i32);
    for k in 0..n {
        v[k + 1] = v[k] * (n - k) as f64 / (k + 1) as f64 * p / (1.0 - p);
    }
    v
}

/// Réplica literal del predicado de Ville del vigilante vivo
/// (online_daemon.rs:1205-1209) y de su armado (:1388, :1948).
fn vigilante_degradado(v: &VilleEProcess) -> bool {
    v.is_exhausted() || v.is_evidence_decayed(0.50)
}

fn nuevo_vigilante() -> VilleEProcess {
    VilleEProcess::with_bounds(0.05, 0.05, 0.50).unwrap()
}

/// Réplica de la regla t del mismo predicado (online_daemon.rs:1211-1240 con
/// summarize_returns de evolution-engine, que no es dependencia de este
/// crate): con n ≥ 20, t = media/sd·√n (sd con n−1) ≤ −2; con n ≥ 5, lote de
/// pérdida constante. El t es invariante a escala, así que da igual medir en R
/// o en retorno sobre capital.
struct ReglaT {
    n: usize,
    suma: f64,
    suma2: f64,
    primero: f64,
    constante: bool,
}

impl ReglaT {
    fn new() -> Self {
        ReglaT {
            n: 0,
            suma: 0.0,
            suma2: 0.0,
            primero: 0.0,
            constante: true,
        }
    }

    fn observar(&mut self, x: f64) {
        if self.n == 0 {
            self.primero = x;
        } else if x != self.primero {
            self.constante = false;
        }
        self.n += 1;
        self.suma += x;
        self.suma2 += x * x;
    }

    fn degradado(&self) -> bool {
        if self.n < 5 {
            return false;
        }
        if self.constante {
            return self.primero < 0.0;
        }
        if self.n < 20 {
            return false;
        }
        let n = self.n as f64;
        let media = self.suma / n;
        let var = ((self.suma2 - n * media * media) / (n - 1.0)).max(0.0);
        var > 0.0 && media / var.sqrt() * n.sqrt() <= -2.0
    }
}

/// Valor verdadero: PF real = 1 por construcción, (a) ±1 R con p = ½ y
/// (b) +2 R/−1 R con p = ⅓. Enumeración binomial exacta con n = 100 (sin error
/// Monte Carlo). Se exige P(LCB ≤ 1) ≥ 0,94: nominal 0,95 con un punto de
/// margen por la discreción. Hoy: 0,9033 (1:1) y 0,8630 (2:1), porque
/// SE[ln PF] = √(2/n) omite la covarianza (+2/n) y los términos de p.
#[test]
#[ignore = "OPEN F2.2-03 (Línea C): la cota del profit factor usa SE[ln PF] = √(2/n) y cubre ~90 % y ~86 %, no el 95 %"]
fn cf2_pf_lcb_cubre_al_95_con_pf_real_1() {
    let n = 100usize;
    let pmf = binom_pmf(n, 0.5);
    let c11: f64 = (0..=n)
        .filter(|&k| profit_factor_lcb(k as f64, (n - k) as f64, n as f64) <= 1.0)
        .map(|k| pmf[k])
        .sum();
    let pmf3 = binom_pmf(n, 1.0 / 3.0);
    let c21: f64 = (0..=n)
        .filter(|&k| profit_factor_lcb(2.0 * k as f64, (n - k) as f64, n as f64) <= 1.0)
        .map(|k| pmf3[k])
        .sum();
    assert!(
        c11 >= 0.94 && c21 >= 0.94,
        "cobertura exacta 1:1 = {c11:.4}, 2:1 = {c21:.4}; nominal 0,95"
    );
}

/// Valor verdadero: lado ganador o perdedor con p = ½ y la MISMA magnitud
/// Exp(1) en los dos lados, así que PF real = 1. 4 000 réplicas de 100
/// operaciones. Se exige cobertura ≥ 0,93 (nominal 0,95 menos 4 SE: la SE con
/// cobertura 0,95 es 0,0034). Hoy Var(ln PF) ≈ 8/n frente a 2/n del código:
/// z efectiva 1,64·√(2/8) = 0,82 y cobertura ≈ 0,79 (SE 0,0063).
#[test]
#[ignore = "OPEN F2.2-03 (Línea C): con magnitudes exponenciales Var(ln PF) ≈ 8/n y la cota con √(2/n) cubre ~80 %"]
fn cf2_pf_lcb_cubre_con_magnitudes_exponenciales() {
    let mut g = Lcg::new(20_261_011);
    let (reps, n) = (4_000usize, 100usize);
    let mut cubre = 0usize;
    for _ in 0..reps {
        let (mut w, mut l) = (0.0, 0.0);
        for _ in 0..n {
            let m = g.exp1();
            if g.u() < 0.5 {
                w += m
            } else {
                l += m
            }
        }
        if profit_factor_lcb(w, l, n as f64) <= 1.0 {
            cubre += 1;
        }
    }
    let c = cubre as f64 / reps as f64;
    assert!(
        c >= 0.93,
        "cobertura {c:.4} con PF real 1 y magnitudes Exp(1)"
    );
}

/// Versión secuencial (corrección del escéptico de F2.2, FA-1): el consumidor
/// vivo mira la cota en CADA cierre, así que la propiedad que usa es
/// P(∀ n ≤ 100: LCB_n ≤ 1) con PF real 1. Programación dinámica exacta sobre
/// el número de ganadas (sin error Monte Carlo), para 1:1 con p = ½ y 2:1 con
/// p = ⅓. Una sucesión de confianza válida en todo instante cubriría ≥ 0,95;
/// se exige ≥ 0,94. Hoy ≈ 0,54 y ≈ 0,57: la cota a n fijo no es anytime (ni
/// con el SE corregido lo sería), así que esta prueba sólo se activa si el
/// arreglo de F2.2-03 da una cota válida bajo parada opcional.
#[test]
#[ignore = "OPEN F2.2-03 y FA-1 (Línea C): la cota del profit factor mirada en cada cierre no es anytime-valid; con PF real 1 supera 1 en ~45 % de los caminos"]
fn cf2_pf_lcb_cobertura_secuencial_en_cada_cierre() {
    let n_max = 100usize;
    let mut coberturas = Vec::new();
    for &(ganancia, p) in &[(1.0_f64, 0.5_f64), (2.0, 1.0 / 3.0)] {
        // masa[k] = P(k ganadas tras t cierres y LCB_s ≤ 1 para todo s ≤ t).
        let mut masa = vec![1.0_f64];
        for t in 1..=n_max {
            let mut nueva = vec![0.0_f64; t + 1];
            for (k, &m) in masa.iter().enumerate() {
                if m == 0.0 {
                    continue;
                }
                nueva[k + 1] += m * p;
                nueva[k] += m * (1.0 - p);
            }
            for (k, m) in nueva.iter_mut().enumerate() {
                if profit_factor_lcb(ganancia * k as f64, (t - k) as f64, t as f64) > 1.0 {
                    *m = 0.0;
                }
            }
            masa = nueva;
        }
        coberturas.push(masa.iter().sum::<f64>());
    }
    println!(
        "cobertura secuencial 1:1 = {:.4}, 2:1 = {:.4}",
        coberturas[0], coberturas[1]
    );
    assert!(
        coberturas.iter().all(|&c| c >= 0.94),
        "cobertura secuencial hasta 100 cierres: 1:1 = {:.4}, 2:1 = {:.4}; nominal 0,95",
        coberturas[0],
        coberturas[1]
    );
}

/// Valor verdadero: p conocido; enumeración binomial exacta con
/// n ∈ {20, 50, 100} y p ∈ {0,3; 0,5; 0,6; 0,8}. Se exige P(LCB ≤ p) ≥ 0,925
/// en cada celda (nominal 0,95 con 2,5 puntos por la discreción de la
/// binomial). Mínimo esperado 0,9308 (n = 20, p = 0,8).
///
/// Guarda de vacuidad: una cota que valiera siempre 0 (o None) cubriría el
/// 100 %. Hoy el máximo por celda es 0,9557 (n = 100, p = 0,5); se exige
/// ≤ 0,99 en cada celda, holgura que admite un reemplazo exacto más
/// conservador (Clopper-Pearson) y rechaza la cota vacía.
#[test]
fn cf2_win_rate_lcb_cubre_al_95() {
    let (mut minimo, mut maximo) = (1.0_f64, 0.0_f64);
    for &n in &[20usize, 50, 100] {
        for &p in &[0.3, 0.5, 0.6, 0.8] {
            let pmf = binom_pmf(n, p);
            let c: f64 = (0..=n)
                .filter(|&k| win_rate_lcb(k as f64 / n as f64, n as f64).map_or(true, |b| b <= p))
                .map(|k| pmf[k])
                .sum();
            minimo = minimo.min(c);
            maximo = maximo.max(c);
            assert!(c >= 0.925, "n={n} p={p}: cobertura {c:.4}");
            assert!(c <= 0.99, "n={n} p={p}: cobertura {c:.4}: la cota es vacía");
        }
    }
    println!("cobertura con la frecuencia cruda: mínima {minimo:.4}, máxima {maximo:.4}");
}

/// Entrada viva (corrección de los dos escépticos de F2.2): el arena publica
/// el win rate ya posteriorizado por `SuperGenotype::posterior_win_rate`
/// (prior WORST_TOLERATED_WR = 0,40 con κ = 1,96² pseudo-operaciones,
/// state.rs:596 y god-engine-core lib.rs:4085-4091) y ruin.rs:92 lo pasa a
/// `win_rate_lcb`. Se encadena la función viva desde el prior vivo y se
/// enumera la binomial exacta en la misma rejilla. Mismo umbral ≥ 0,925.
/// Con prior 0,40 el doble prior desplaza la cota hacia abajo para p > 0,40
/// (más cobertura) y apenas la sube para p = 0,3; mínimo esperado ≈ 0,947.
#[test]
fn cf2_win_rate_lcb_cubre_con_la_tasa_posteriorizada() {
    let mut minimo = 1.0_f64;
    for &n in &[20usize, 50, 100] {
        for &p in &[0.3, 0.5, 0.6, 0.8] {
            let pmf = binom_pmf(n, p);
            let mut c = 0.0;
            for (k, &pk) in pmf.iter().enumerate() {
                // La media posterior no depende del orden de los cierres.
                let mut w = SuperGenotype::WORST_TOLERATED_WR;
                for i in 0..n {
                    w = SuperGenotype::posterior_win_rate(w, i as f64, i < k);
                }
                if win_rate_lcb(w, n as f64).map_or(true, |b| b <= p) {
                    c += pk;
                }
            }
            minimo = minimo.min(c);
            assert!(
                c >= 0.925,
                "n={n} p={p}: cobertura {c:.4} con la tasa posteriorizada"
            );
        }
    }
    println!("cobertura mínima con la tasa posteriorizada: {minimo:.4}");
}

/// Valor verdadero: con prior Beta(k0·p0, k0·(1−p0)) = Beta(5,5; 4,5) y la
/// misma aproximación normal de `EdgePosterior::lcb`, la cota sin datos es
/// 0,55 − 1,64·0,15 = 0,304, y una ganada (Beta(6,5; 4,5)) no puede bajarla.
/// Tolerancia 0,01 (forma cerrada). Hoy con n = 0 devuelve 0,55 (la media del
/// prior) y con una ganada 0,358 < 0,55. Si se arregla, hay que reescribir
/// evidence.rs:227 (assert_eq con 0,55).
#[test]
#[ignore = "OPEN F2.2-09 (Línea C): win_rate_hierarchical_lcb devuelve la media del prior con n = 0 y una ganada baja la cota"]
fn cf2_cota_jerarquica_es_coherente_y_monotona() {
    let sin_datos = win_rate_hierarchical_lcb(0.5, 0.0, 0.55, 10.0);
    let una_ganada = win_rate_hierarchical_lcb(1.0, 1.0, 0.55, 10.0);
    let cota_prior = 0.55 - 1.64 * (0.55 * 0.45 / 11.0f64).sqrt();
    assert!(
        una_ganada >= sin_datos,
        "una ganada baja la cota: {sin_datos} -> {una_ganada}"
    );
    assert!(
        (sin_datos - cota_prior).abs() < 0.01,
        "n = 0 devuelve {sin_datos}; la cota del prior es {cota_prior:.3}"
    );
}

/// Valor verdadero: la secuencia [+1, +1, +1, −1, −1] R suma +1 R; un
/// e-proceso de deriva negativa, Π(1 − λx), vale < 1 sobre ella, así que no
/// hay evidencia de degradación y el predicado vivo debe ser false. Con 5
/// observaciones la regla t no actúa (exige ≥ 20) y no es pérdida constante.
/// Hoy: paso 4 con λ = 0,5 (varianza en el suelo EPS) da e = 0,579; paso 5
/// da e = 0,289 y caída de evidencia 0,75 ≥ 0,5.
#[test]
#[ignore = "OPEN F2.2-01 (Línea C): el vigilante apuesta a favor del edge y dispara por caída de evidencia; revierte un genoma con +1 R"]
fn cf2_vigilante_no_revierte_un_genoma_rentable_testigo() {
    let mut v = nuevo_vigilante();
    for r in [1.0, 1.0, 1.0, -1.0, -1.0] {
        v.update(r);
    }
    assert!(
        !vigilante_degradado(&v),
        "rollback con +1 R: e = {} pico = {} caída = {}",
        v.e_value,
        v.peak_e_value,
        v.evidence_drawdown()
    );
}

/// Valor verdadero: 500 caminos de 300 operaciones con pérdida −1 R y
/// (p = 0,55, +1 R), (p = 0,45, +2 R) y (p = 0,50, +1 R): medias +0,10 R,
/// +0,35 R y 0 R, así que todo rollback es falso por construcción. Se exige
/// tasa ≤ α + 4·SE = 0,05 + 4·√(0,05·0,95/500) = 0,089 (la media 0 es la
/// frontera del nulo). Se mide la parte de Ville (F2.2-01) y el predicado
/// compuesto vivo (Ville, o t ≤ −2 con ≥ 20 operaciones, o pérdida constante
/// con ≥ 5), evaluado tras cada operación, que es la cadencia más frecuente
/// posible del ciclo. La regla t es una réplica local (evolution-engine no es
/// dependencia de risk-engine). Hoy la parte de Ville dispara en el 100 % de
/// los caminos (paso medio 15 a 23). La regla t y la de pérdida constante
/// solas dan ≈ 0,05 con medias +0,10 y +0,35 R (de eso, la racha inicial de 5
/// pérdidas, q⁵, aporta 0,018 y 0,050), pero ≈ 0,20 con media 0 (mirar un t de
/// n fijo en cada cierre): el arreglo de F2.2-01 tiene que cubrir también esa
/// regla para que el compuesto pase.
#[test]
#[ignore = "OPEN F2.2-01 (Línea C): el vigilante de rollback dispara por caída de evidencia sobre retornos de media >= 0; con media 0 la regla t mirada en cada cierre también da ~0,20"]
fn cf2_vigilante_nivel_sin_degradacion() {
    let mut g = Lcg::new(36);
    let reps = 500usize;
    let tope = 0.05 + 4.0 * (0.05_f64 * 0.95 / reps as f64).sqrt();
    let mut informe = Vec::new();
    for &(p, gan) in &[(0.55, 1.0), (0.45, 2.0), (0.50, 1.0)] {
        let (mut por_ville, mut por_t, mut compuesto) = (0usize, 0usize, 0usize);
        let mut suma_paso_ville = 0usize;
        for _ in 0..reps {
            let mut v = nuevo_vigilante();
            let mut t = ReglaT::new();
            let (mut alarma_ville, mut alarma_t) = (false, false);
            for paso in 1..=300usize {
                let x = if g.u() < p { gan } else { -1.0 };
                v.update(x);
                t.observar(x);
                if !alarma_ville && vigilante_degradado(&v) {
                    alarma_ville = true;
                    suma_paso_ville += paso;
                }
                if !alarma_t && t.degradado() {
                    alarma_t = true;
                }
            }
            por_ville += alarma_ville as usize;
            por_t += alarma_t as usize;
            compuesto += (alarma_ville || alarma_t) as usize;
        }
        let f = |c: usize| c as f64 / reps as f64;
        let paso_medio = if por_ville > 0 {
            suma_paso_ville as f64 / por_ville as f64
        } else {
            0.0
        };
        informe.push((p, gan, f(por_ville), f(por_t), f(compuesto), paso_medio));
    }
    for &(p, gan, ville, t, comp, paso) in &informe {
        println!(
            "p = {p}, +{gan} R: Ville {ville:.3} (paso medio {paso:.1}), t {t:.3}, compuesto {comp:.3}"
        );
    }
    for &(p, gan, ville, t, comp, paso) in &informe {
        assert!(
            ville <= tope,
            "p = {p}, +{gan} R: rollback falso por Ville {ville:.3} (paso medio {paso:.1}); tope {tope:.3}"
        );
        assert!(
            comp <= tope,
            "p = {p}, +{gan} R: rollback falso del predicado compuesto {comp:.3} (regla t {t:.3}); tope {tope:.3}"
        );
    }
}

/// Valor verdadero: ±1 R con p = 0,40, media −0,2 R fijada por el generador.
/// Criterio de la fila 36: detección ≥ 0,80 en 300 operaciones (500 caminos;
/// SE ≤ 0,018). Hoy pasa porque el predicado dispara con todo; sólo informa
/// junto con la prueba de nivel. Con λ = λ_min la deriva log es −0,0113 por
/// paso y la caída del 50 % llega en unas 60 operaciones.
#[test]
fn cf2_vigilante_potencia_ante_menos_02_r() {
    let mut g = Lcg::new(3636);
    let reps = 500usize;
    let mut det = 0usize;
    for _ in 0..reps {
        let mut v = nuevo_vigilante();
        for _ in 0..300 {
            v.update(if g.u() < 0.40 { 1.0 } else { -1.0 });
            if vigilante_degradado(&v) {
                det += 1;
                break;
            }
        }
    }
    let potencia = det as f64 / reps as f64;
    println!("detección con −0,2 R: {potencia}");
    assert!(
        potencia >= 0.80,
        "detección {potencia} ante −0,2 R en 300 operaciones"
    );
}

/// Valor verdadero: +2 R con p = 0,45 y −1 R en otro caso, E[X] = +0,35 R
/// fijada por el generador; como x ≥ −1 y λ < 1, 1 + λx > 0 sin recortar por
/// arriba. Se exige que el pico alcance 1/α = 20 en ≥ 80 % de 200 caminos de
/// 2 000 operaciones (SE ≤ 0,028). Hoy update recorta a ±1: la media vista es
/// 0,45 − 0,55 = −0,10 y certifica ≈ 1 %. El demonio también recorta antes,
/// en online_daemon.rs:1143; el arreglo tiene que cubrir los dos sitios.
#[test]
#[ignore = "OPEN F2.2-02 (Línea C): Ville recorta a ±1 R y mide la tasa de acierto contra ½ en lugar de la esperanza en R"]
fn cf2_ville_certifica_edge_con_rr_2() {
    let mut g = Lcg::new(4);
    let reps = 200usize;
    let mut cert = 0usize;
    for _ in 0..reps {
        let mut v = nuevo_vigilante();
        for _ in 0..2_000 {
            v.update(if g.u() < 0.45 { 2.0 } else { -1.0 });
        }
        if v.peak_e_value >= v.threshold {
            cert += 1;
        }
    }
    assert!(
        cert as f64 / reps as f64 >= 0.80,
        "certifica {cert}/{reps} con E[X] = +0,35 R"
    );
}

/// Serie de la candidata ganadora condicionada a su media muestral. Para
/// observaciones gaussianas iid la media es independiente del vector de
/// residuos (x_i − x̄), y la selección sólo mira las medias, así que basta
/// sortear las N medias y construir sólo la serie del máximo: misma ley
/// exacta que generar las N series enteras, con coste N + n por réplica.
fn serie_con_media(g: &mut Lcg, n: usize, media: f64) -> Vec<f64> {
    let mut x: Vec<f64> = (0..n).map(|_| g.normal()).collect();
    let m = x.iter().sum::<f64>() / n as f64;
    for v in x.iter_mut() {
        *v += media - m;
    }
    x
}

/// Una réplica de selección: N pruebas de n observaciones N(μ_k, 1), la
/// primera con μ = `plantado` y el resto con 0. Devuelve (DSR pasa, Šidák con
/// σ conocida pasa). El Šidák exacto rechaza si max_k x̄_k·√n ≥ z_S con
/// z_S = Φ⁻¹((1−α)^{1/N}): nivel exactamente α bajo el nulo global.
fn replica_seleccion(
    g: &mut Lcg,
    n_pruebas: usize,
    n_obs: usize,
    plantado: f64,
    z_sidak: f64,
) -> (bool, bool) {
    let raiz_n = (n_obs as f64).sqrt();
    let mut mejor = f64::NEG_INFINITY;
    for k in 0..n_pruebas {
        let mu = if k == 0 { plantado } else { 0.0 };
        let media = mu + g.normal() / raiz_n;
        if media > mejor {
            mejor = media;
        }
    }
    let serie = serie_con_media(g, n_obs, mejor);
    let dsr = edge_survives_multiplicity(&serie, n_pruebas).passes;
    (dsr, mejor * raiz_n >= z_sidak)
}

fn z_sidak(n_pruebas: usize, alfa: f64) -> f64 {
    // 1 − (1 − α)^{1/N} sin cancelación.
    let cola = -((1.0 - alfa).ln() / n_pruebas as f64).exp_m1();
    inverse_normal_cdf(1.0 - cola)
}

/// Valor verdadero (corrección de los dos escépticos de F2.2): Sharpe
/// plantado 0,25 por observación, n = 250. El Šidák exacto con σ conocida es
/// la prueba de nivel α del máximo de N gaussianas independientes; con
/// N = 100 da potencia ≈ 0,74 y con N = 20 010 (acumulación del demonio)
/// ≈ 0,27. Se exige al DSR, para N = 100 (1 000 réplicas) y N = 20 010 (400):
/// nivel ≤ 0,05 + 4·SE nominal, y potencia ≥ la del Šidák − 4·SE de la
/// diferencia; además, con N = 100, potencia ≥ 0,60 (fila 34; SE ≈ 0,016).
/// Hoy el DSR exige SR̂ ≥ (E[max] + 1,645)·σ_SR: nivel ≈ 0 y potencia ≈ 0,40
/// (N = 100) y ≈ 0,03 (N = 20 010). La caída por rondas acumuladas
/// (accumulated_trials del demonio) es del cableado de C-F7 y no se prueba
/// aquí.
#[test]
#[ignore = "OPEN F2.2-10 (Línea C): el DSR deflacta dos veces (E[max] + 1,645 σ): nivel ≈ 0 y potencia muy por debajo del Šidák exacto"]
fn cf2_dsr_nivel_y_potencia_con_100_pruebas() {
    let n_obs = 250usize;
    let mut g = Lcg::new(34);
    let mut informe = Vec::new();
    for &(n_pruebas, reps) in &[(100usize, 1_000usize), (20_010, 400)] {
        let zs = z_sidak(n_pruebas, 0.05);
        let mut tasas = [[0usize; 2]; 2]; // [nulo, plantado][dsr, šidák]
        for (i, &plantado) in [0.0_f64, 0.25].iter().enumerate() {
            for _ in 0..reps {
                let (dsr, sidak) = replica_seleccion(&mut g, n_pruebas, n_obs, plantado, zs);
                tasas[i][0] += dsr as usize;
                tasas[i][1] += sidak as usize;
            }
        }
        let f = |c: usize| c as f64 / reps as f64;
        informe.push((
            n_pruebas,
            reps,
            f(tasas[0][0]),
            f(tasas[0][1]),
            f(tasas[1][0]),
            f(tasas[1][1]),
        ));
    }
    for &(n_pruebas, _, nivel_d, nivel_s, pot_d, pot_s) in &informe {
        println!(
            "N = {n_pruebas}: DSR nivel {nivel_d:.3} potencia {pot_d:.3}; Šidák nivel {nivel_s:.3} potencia {pot_s:.3}"
        );
    }
    for &(n_pruebas, reps, nivel_d, nivel_s, pot_d, pot_s) in &informe {
        let r = reps as f64;
        let tope_nivel = 0.05 + 4.0 * (0.05 * 0.95 / r).sqrt();
        // Control del generador: el Šidák exacto tiene nivel α.
        assert!(
            nivel_s <= tope_nivel,
            "N = {n_pruebas}: nivel del Šidák {nivel_s}"
        );
        assert!(
            nivel_d <= tope_nivel,
            "N = {n_pruebas}: nivel del DSR {nivel_d}"
        );
        let se_dif = ((pot_d * (1.0 - pot_d) + pot_s * (1.0 - pot_s)) / r).sqrt();
        assert!(
            pot_d >= pot_s - 4.0 * se_dif,
            "N = {n_pruebas}: potencia del DSR {pot_d:.3} frente a {pot_s:.3} del Šidák exacto (nivel DSR {nivel_d:.3})"
        );
    }
    let pot_100 = informe[0].4;
    assert!(
        pot_100 >= 0.60,
        "potencia del DSR con SR 0,25, N = 100, n = 250: {pot_100:.3}"
    );
}

/// Regresión de la fórmula (corrección del escéptico de completitud de
/// F2.2): con σ_SR(s) = √((1 + s²/2)/(n−1)) (gaussiana: γ3 = 0, γ4 = 3), el
/// DSR pasa si y sólo si SR̂ ≥ c con c = K/√(n−1−K²/2), K = E[max]_N + Φ⁻¹(0,95).
/// Como SR̂ ≈ N(μ, (1 + μ²/2)/n), la potencia analítica es
/// 1 − Φ((c − μ)/√((1 + μ²/2)/n)). Con SR 0,2, n = 250 y N = 100: c ≈ 0,269 y
/// potencia ≈ 0,14. Se exige |empírica − analítica| ≤ 4·SE con 1 000
/// réplicas (SE ≈ 0,011). Fija que la falta de potencia de F2.2-10 es la de la
/// fórmula y no ruido de los momentos; si el arreglo cambia la fórmula, esta
/// prueba se actualiza con él.
#[test]
fn cf2_dsr_potencia_coincide_con_su_formula() {
    let (n_pruebas, n_obs, reps, mu) = (100usize, 250usize, 1_000usize, 0.2_f64);
    let mut g = Lcg::new(3434);
    let zs = z_sidak(n_pruebas, 0.05);
    let mut pasa = 0usize;
    for _ in 0..reps {
        pasa += replica_seleccion(&mut g, n_pruebas, n_obs, mu, zs).0 as usize;
    }
    let empirica = pasa as f64 / reps as f64;
    let k = expected_max_sharpe(n_pruebas, 1.0) + inverse_normal_cdf(0.95);
    let n = n_obs as f64;
    let c = k / (n - 1.0 - k * k / 2.0).sqrt();
    let analitica = 1.0 - normal_cdf((c - mu) / ((1.0 + mu * mu / 2.0) / n).sqrt());
    let se = (analitica * (1.0 - analitica) / reps as f64).sqrt();
    println!("potencia del DSR: empírica {empirica}, analítica {analitica} (c = {c})");
    assert!(
        (empirica - analitica).abs() <= 4.0 * se,
        "potencia empírica {empirica} frente a la de la fórmula {analitica} (c = {c})"
    );
}

/// Criterio de la fila 34 con muestra suficiente (corrección del escéptico de
/// completitud de F2.2): SR 0,2, N = 100 y n = 1 000. σ_SR ≈ 0,0316, el listón
/// c ≈ 0,133 y la potencia analítica ≈ 0,98. Se exige nivel ≤ 0,05 + 4·SE
/// (500 réplicas: 0,0695) y potencia > 0,50. Con datos suficientes el DSR sí
/// detecta; el defecto de F2.2-10 es de potencia con n corto y N grande.
#[test]
fn cf2_dsr_detecta_sr_02_con_1000_observaciones() {
    let (n_pruebas, n_obs, reps) = (100usize, 1_000usize, 500usize);
    let mut g = Lcg::new(343_434);
    let zs = z_sidak(n_pruebas, 0.05);
    let mut tasa = [0.0f64; 2];
    for (i, &plantado) in [0.0_f64, 0.2].iter().enumerate() {
        let mut pasa = 0usize;
        for _ in 0..reps {
            pasa += replica_seleccion(&mut g, n_pruebas, n_obs, plantado, zs).0 as usize;
        }
        tasa[i] = pasa as f64 / reps as f64;
    }
    println!("n = 1 000: nivel {}, potencia {}", tasa[0], tasa[1]);
    let tope_nivel = 0.05 + 4.0 * (0.05_f64 * 0.95 / reps as f64).sqrt();
    assert!(tasa[0] <= tope_nivel, "nivel {}", tasa[0]);
    assert!(
        tasa[1] > 0.50,
        "potencia {} con SR 0,2, N = 100, n = 1 000",
        tasa[1]
    );
}
