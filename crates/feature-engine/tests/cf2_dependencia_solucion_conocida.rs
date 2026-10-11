//! C-F2.4 (Línea C): contagio, lead-lag, transfer entropy y cópula t frente
//! a soluciones conocidas.
//!
//! Teoría que se prueba:
//! - Contagio (`hawkes_cross`): la excitación cruzada líder → seguidor debe
//!   ser ≈ la tasa base cuando no hay acople causal y detectar un acople
//!   plantado. El nulo correcto conserva la estructura propia de cada serie
//!   (desplazamiento circular) y rompe la relación cruzada.
//! - Lead-lag (`LeadLagAlphaEngine`): con un desfase plantado y contenido en
//!   la rejilla, el lag de ρ máxima debe recuperarlo; con dos series
//!   independientes, la tasa de «líder» declarado debe quedarse en el tamaño
//!   nominal del 5 % (umbral t ≥ 2 de la propia documentación).
//! - Transfer entropy binaria: bajo independencia, G = 2·M·ln2·TE_bits es
//!   asintóticamente χ² con 2 grados de libertad (Wilks); con un canal
//!   binario plantado Y_t | X_(t−1), TE = I(Y_t; X_(t−1)) en forma cerrada.
//! - Cópula t: ρ = sin(πτ/2) y λ = 2·T_(ν+1)(−√((ν+1)(1−ρ)/(1+ρ))); la
//!   cópula gaussiana tiene dependencia de cola exactamente 0.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.
//! Sin dependencias nuevas: LCG propio y Box-Muller.

use feature_engine::copulas::{lambda_cola_t, nu_mle, rho_desde_tau, tau_kendall};
use feature_engine::hawkes_cross::{contagion_matrix, contagion_roles, cross_excitation};
use feature_engine::lead_lag::LeadLagAlphaEngine;
use feature_engine::transfer_entropy::{transfer_entropy_observada, ObservationWindow};

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn u01(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
    /// Normal estándar por Box-Muller.
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.u01(), self.u01());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    /// Exponencial de tasa `tasa`.
    fn exp(&mut self, tasa: f64) -> f64 {
        -self.u01().ln() / tasa
    }
    /// Poisson de media `m` (método de Knuth; m pequeño).
    fn poisson(&mut self, m: f64) -> usize {
        let (limite, mut k, mut p) = ((-m).exp(), 0usize, 1.0f64);
        loop {
            p *= self.u01();
            if p <= limite {
                return k;
            }
            k += 1;
        }
    }
}

/// Rejilla de lags del publicador vivo (god-engine-core/src/contagion_publisher.rs:52).
const LAGS_VIVOS: [u64; 6] = [200, 500, 1_000, 5_000, 15_000, 30_000];

/// Poisson en ms, estrictamente creciente (contagion_matrix lo exige).
fn poisson_ms(rng: &mut Lcg, tasa_por_s: f64, hasta_ms: u64) -> Vec<u64> {
    let (mut t, mut v) = (0.0f64, Vec::<u64>::new());
    loop {
        t += rng.exp(tasa_por_s) * 1000.0;
        if t >= hasta_ms as f64 {
            return v;
        }
        let tm = t as u64;
        if v.last().map_or(true, |&u| tm > u) {
            v.push(tm);
        }
    }
}

/// Hawkes exponencial por su representación de racimos: inmigrantes
/// Poisson(μ) y cada evento engendra Poisson(n) hijos a retardos Exp(β).
/// Tasa media μ/(1 − n). Devuelve ms estrictamente crecientes.
fn hawkes_ms(rng: &mut Lcg, mu_por_s: f64, n: f64, beta_por_s: f64, hasta_ms: u64) -> Vec<u64> {
    let fin = hasta_ms as f64 / 1000.0;
    let mut pendientes: Vec<f64> = Vec::new();
    let mut t = 0.0f64;
    loop {
        t += rng.exp(mu_por_s);
        if t >= fin {
            break;
        }
        pendientes.push(t);
    }
    let mut todos = Vec::new();
    while let Some(s) = pendientes.pop() {
        todos.push(s);
        for _ in 0..rng.poisson(n) {
            let h = s + rng.exp(beta_por_s);
            if h < fin {
                pendientes.push(h);
            }
        }
    }
    todos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut v: Vec<u64> = Vec::with_capacity(todos.len());
    for s in todos {
        let tm = (s * 1000.0) as u64;
        if v.last().map_or(true, |&u| tm > u) {
            v.push(tm);
        }
    }
    v
}

/// Ventana activa i => un evento en el centro del bin.
fn activos_a_eventos(act: &[bool], ancho: u64) -> Vec<u64> {
    act.iter()
        .enumerate()
        .filter(|&(_, &a)| a)
        .map(|(i, _)| i as u64 * ancho + ancho / 2)
        .collect()
}

/// Pseudo-observaciones rango/(n+1), como src/bin/copulas_manifest.rs:76-87.
fn rangos(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut r = vec![0.0; n];
    for (k, &i) in idx.iter().enumerate() {
        r[i] = (k + 1) as f64 / (n + 1) as f64;
    }
    r
}

/// Normal bivariada con correlación rho; con Some(k) se divide por
/// √(χ²_k/k) común a las dos coordenadas ⇒ t bivariada con k grados.
fn muestra_t(rng: &mut Lcg, n: usize, rho: f64, nu: Option<usize>) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let z1 = rng.normal();
        let z2 = rho * z1 + (1.0 - rho * rho).sqrt() * rng.normal();
        let s = match nu {
            Some(k) => {
                let mut w = 0.0;
                for _ in 0..k {
                    let g = rng.normal();
                    w += g * g;
                }
                (w / k as f64).sqrt()
            }
            None => 1.0,
        };
        x.push(z1 / s);
        y.push(z2 / s);
    }
    (x, y)
}

// ─────────────────────────────────────────────────────────────────
// Contagio (hawkes_cross)
// ─────────────────────────────────────────────────────────────────

/// Verdad: dos monedas sin acople causal que sólo comparten la cadencia del
/// exchange (reloj de @depth5 a 250 ms) con fases distintas: la matriz no
/// debe declarar contagio (m < 3, el propio umbral z de cross_excitation) ni
/// roles (|net| < 1). Hoy: en el lag de 200 ms todo evento de a tiene un b en
/// (t, t+200] (rate = 1 frente a base 0,80) ⇒ z ≈ 7,9 en un sentido y 0 en
/// el otro; los lags ≥ 500 ms tienen la base recortada a 1.
#[test]
#[ignore = "OPEN F2.4-02 (Qoder/GLM): el nulo binomial de cross_excitation supone llegadas sin estructura; dos relojes periódicos con fase estable dan z ≈ 7,9 en 200 ms"]
fn cf2_contagio_relojes_periodicos_independientes_no_son_contagio() {
    let mut rng = Lcg(0xB1B1_0001);
    let a: Vec<u64> = (0..256u64)
        .map(|k| 1_000 + k * 250 + (rng.u01() * 10.0) as u64)
        .collect();
    let b: Vec<u64> = (0..256u64)
        .map(|k| 1_030 + k * 250 + (rng.u01() * 10.0) as u64)
        .collect();
    // span = último − primero, como contagion_publisher.rs:38.
    let spans = [a[255] - a[0], b[255] - b[0]];
    let m = contagion_matrix(&[a, b], &spans, &LAGS_VIVOS).expect("series válidas");
    let roles = contagion_roles(&m).expect("matriz finita");
    assert!(
        m[0][1] < 3.0 && m[1][0] < 3.0,
        "relojes sin acople declarados contagio: matriz {m:?}, roles {roles:?}"
    );
    assert!(
        roles.iter().all(|r| r.net_role.abs() < 1.0),
        "roles {roles:?} (hoy ±7,9)"
    );
}

/// Verdad: líder Poisson a 2/s; seguidor Poisson a 1/s más una copia de cada
/// evento del líder a +200 ms con p = 0,5 (acople plantado). Con acople,
/// rate ≈ 0,665 frente a base 0,4 y n ≈ 1200 ⇒ z ≈ 18. En el nulo (seguidor
/// desplazado circularmente) P(acierto en 200 ms) = 1 − e^(−0,4) = 0,33 < 0,4
/// ⇒ z < 0. Tolerancias: falsos ≤ 5 % + 4σ binomial con n = 200 (≤ 22) y
/// potencia ≥ 190/200. Sólo cubre el régimen denso (densidad·lag = 0,4), donde
/// la base lineal es conservadora: el pase demuestra conservadurismo, no
/// calibración.
#[test]
fn cf2_contagio_pares_desfasados_circularmente_tasa_de_lideres() {
    let mut rng = Lcg(0xB2B2_0002);
    let span = 600_000u64;
    let (mut falsos, mut detectados) = (0usize, 0usize);
    for _ in 0..200 {
        let lider = poisson_ms(&mut rng, 2.0, span);
        let mut seg = poisson_ms(&mut rng, 1.0, span);
        for &t in &lider {
            if rng.u01() < 0.5 && t + 200 < span {
                seg.push(t + 200);
            }
        }
        seg.sort_unstable();
        seg.dedup();
        if cross_excitation(&lider, &seg, span, &LAGS_VIVOS).is_some() {
            detectados += 1;
        }
        // Nulo: el desplazamiento circular rompe el acople y conserva la estructura propia.
        let d = 60_000 + (rng.u01() * 480_000.0) as u64;
        let mut nulo: Vec<u64> = seg.iter().map(|&t| (t + d) % span).collect();
        nulo.sort_unstable();
        nulo.dedup();
        if cross_excitation(&lider, &nulo, span, &LAGS_VIVOS).is_some() {
            falsos += 1;
        }
    }
    assert!(falsos <= 22, "falsos {falsos}/200 (nominal 10, cota 5 % + 4σ)");
    assert!(detectados >= 190, "potencia {detectados}/200");
}

/// Verdad: el mismo acople plantado, pero con la copia a +1000 ms (un lag de
/// la rejilla viva). La potencia debe ser la misma (≥ 190/200). Hoy: a 2/s
/// la base en 500 ms y en 1000 ms vale ≥ 1 y se recorta a 1 − 1e-9 (z ≤ 0),
/// y en 200 ms la copia no cae en la ventana: no se detecta nunca.
#[test]
#[ignore = "OPEN F2.4-02 (Qoder/GLM): la base lineal densidad·lag se recorta a 1 cuando densidad·lag ≥ 1 y deja ciegos los lags de 500 ms en adelante"]
fn cf2_contagio_potencia_con_acople_a_1000ms() {
    let mut rng = Lcg(0xB2B2_0003);
    let span = 600_000u64;
    let mut detectados = 0usize;
    let mut picos = std::collections::BTreeMap::<u64, usize>::new();
    for _ in 0..200 {
        let lider = poisson_ms(&mut rng, 2.0, span);
        let mut seg = poisson_ms(&mut rng, 1.0, span);
        for &t in &lider {
            if rng.u01() < 0.5 && t + 1_000 < span {
                seg.push(t + 1_000);
            }
        }
        seg.sort_unstable();
        seg.dedup();
        if let Some(e) = cross_excitation(&lider, &seg, span, &LAGS_VIVOS) {
            detectados += 1;
            *picos.entry(e.peak_lag_ms).or_default() += 1;
        }
    }
    assert!(
        detectados >= 190,
        "potencia con la copia a +1000 ms: {detectados}/200 (picos {picos:?})"
    );
}

/// Verdad: líder Hawkes (n = 0,6, β = 1/s, tasa media 0,5/s) y seguidor
/// Poisson a 0,2/s independiente: no hay contagio. Régimen disperso
/// (densidad·lag = 0,04 en 200 ms), donde la base lineal ≈ 1 − e^(−x) deja
/// de ser holgada y los racimos del líder solapan sus ventanas (el error
/// binomial infravalora la varianza). Tolerancia: falsos ≤ 5 % + 4σ binomial
/// con n = 200 (≤ 22). Hoy 1/200: también aquí el estimador es conservador
/// (el seguidor independiente acierta con 1 − e^(−x) < x), así que la
/// inflación por racimos de F2-C5 no aparece en este régimen.
#[test]
fn cf2_contagio_nulo_disperso_con_lider_hawkes() {
    let mut rng = Lcg(0xB2B2_0004);
    let span = 600_000u64;
    let mut falsos = 0usize;
    let mut n_lider = 0usize;
    for _ in 0..200 {
        let lider = hawkes_ms(&mut rng, 0.2, 0.6, 1.0, span);
        let seg = poisson_ms(&mut rng, 0.2, span);
        n_lider += lider.len();
        if cross_excitation(&lider, &seg, span, &LAGS_VIVOS).is_some() {
            falsos += 1;
        }
    }
    let tasa_lider = n_lider as f64 / 200.0 / 600.0;
    assert!(
        (0.40..=0.60).contains(&tasa_lider),
        "generador Hawkes fuera de su tasa media 0,5/s: {tasa_lider}"
    );
    assert!(falsos <= 22, "falsos {falsos}/200 con líder agrupado (nominal 10)");
}

/// Verdad: contrato de dominio (pedido (2) de LC-F1-35): un empate de
/// timestamps en una serie no invalida las otras N − 1 series sanas, así que
/// la matriz debe existir. Hoy: hawkes_cross.rs:152-155 devuelve None si
/// alguna serie tiene w[1] <= w[0].
#[test]
#[ignore = "OPEN F2.4-09 (Qoder/GLM): un solo empate de timestamps en una serie devuelve None para toda la matriz de contagio (LC-F1-35, pedido 2)"]
fn cf2_contagio_un_empate_no_anula_la_matriz() {
    let mut rng = Lcg(0xB3B3_0003);
    let span = 120_000u64;
    let mut series: Vec<Vec<u64>> = (0..3).map(|_| poisson_ms(&mut rng, 2.0, span)).collect();
    let t = series[2][10];
    series[2].insert(11, t); // dos eventos en el mismo ms
    let m = contagion_matrix(&series, &[span; 3], &LAGS_VIVOS);
    assert!(
        m.is_some(),
        "un solo empate anula la matriz entera (hawkes_cross.rs:152-155): {m:?}"
    );
}

// ─────────────────────────────────────────────────────────────────
// Lead-lag
// ─────────────────────────────────────────────────────────────────

/// Verdad: líder AR(1) con φ = 0,9 cada 250 ms; alt(k) = líder(k − 8) +
/// 0,3·σ·N(0,1): desfase plantado de 2000 ms, contenido en la rejilla. En el
/// lag plantado ρ ≈ 0,96, frente a ≈ 0,63 en 1000 ms. Las 400 lecturas están
/// autocorrelacionadas, así que el error se midió por semillas: en 40
/// semillas la recuperación tiene media 0,951 y sd 0,015 (mín. 0,91). Se
/// exige ≥ 0,89 (≥ 4 sd bajo la media; el auditor pedía 0,90). Sólo certifica
/// la cadencia de depth sin trades (4 Hz), no la viva.
#[test]
fn cf2_lead_lag_recupera_desfase_plantado_2000ms() {
    let mut rng = Lcg(0xB4B4_0004);
    let sd = 0.1 / (1.0f64 - 0.81).sqrt(); // desviación estacionaria del OFI AR(1)
    let mut ll = LeadLagAlphaEngine::new(50);
    let (mut hist, mut x) = (Vec::<f64>::new(), 0.0f64);
    let (mut aciertos, mut lecturas) = (0usize, 0usize);
    for k in 0..600usize {
        x = 0.9 * x + 0.1 * rng.normal();
        hist.push(x);
        let ts = k as f64 * 250.0;
        ll.update_leader(true, x, ts);
        let base = if k >= 8 { hist[k - 8] } else { 0.0 };
        let alt = base + 0.3 * sd * rng.normal();
        let _ = ll.predict_altcoin_impulse_con_reloj(7, alt, ts);
        if k >= 200 {
            lecturas += 1;
            if ll.ultimo_lag_btc_ms == 2000.0 {
                aciertos += 1;
            }
        }
    }
    let tasa = aciertos as f64 / lecturas as f64;
    assert!(tasa >= 0.89, "recuperación del lag plantado {tasa}");
}

/// Verdad: el mismo desfase plantado de 2000 ms, pero a la cadencia viva del
/// líder: update_leader corre en cada evento de BTC (trade, depth y kline,
/// god-engine-core/src/lib.rs:3104-3108) con una ventana de 50 MUESTRAS
/// (lib.rs:1182). Aquí el líder publica cada 25 ms (40 eventos/s) su OFI, que
/// cambia cada 250 ms; el alt llega a 4 Hz. La recuperación debe seguir
/// siendo ≥ 0,89, la misma cota derivada que el test gemelo de cadencia lenta
/// (revisión cruzada, CL-65: exigía 0,90 con media 0,951 y sd 0,015, a 3,4 sd,
/// y podía fallar tras el arreglo). Hoy: la ventana cubre 1,25 s, todos los objetivos t + 2000
/// caen en el futuro, el alt se extrapola congelado (lead_lag.rs:316) y ρ = 0
/// en 2000 ms: recuperación 0.
#[test]
#[ignore = "OPEN F2.4-07 (Línea C): la ventana del líder es de 50 muestras; a ~40 eventos/s cubre 1,25 s y el lag de 2000 ms sólo ve el alt congelado"]
fn cf2_lead_lag_desfase_plantado_a_cadencia_viva() {
    let mut rng = Lcg(0xB4B4_0005);
    let sd = 0.1 / (1.0f64 - 0.81).sqrt();
    let mut ll = LeadLagAlphaEngine::new(50);
    let (mut hist, mut x) = (Vec::<f64>::new(), 0.0f64);
    let (mut aciertos, mut lecturas) = (0usize, 0usize);
    let mut lags = std::collections::BTreeMap::<u64, usize>::new();
    for paso in 0..6_000usize {
        let ts = paso as f64 * 25.0;
        if paso % 10 == 0 {
            x = 0.9 * x + 0.1 * rng.normal();
            hist.push(x);
        }
        ll.update_leader(true, x, ts);
        if paso % 10 == 0 {
            let k = paso / 10;
            let base = if k >= 8 { hist[k - 8] } else { 0.0 };
            let alt = base + 0.3 * sd * rng.normal();
            let _ = ll.predict_altcoin_impulse_con_reloj(7, alt, ts);
            if k >= 200 {
                lecturas += 1;
                *lags.entry(ll.ultimo_lag_btc_ms as u64).or_default() += 1;
                if ll.ultimo_lag_btc_ms == 2000.0 {
                    aciertos += 1;
                }
            }
        }
    }
    let tasa = aciertos as f64 / lecturas as f64;
    assert!(
        tasa >= 0.89,
        "recuperación a cadencia viva {tasa} (lags leídos {lags:?})"
    );
}

/// Verdad: líder y alt AR(1) independientes (φ = 0,9) cada 250 ms; todo lag
/// declarado es un falso líder y el tamaño nominal del umbral t ≥ 2 es el
/// 5 %. Tolerancia: 5 % + 3σ binomial con n = 400 (≤ 0,083), la del auditor.
/// Hoy ≈ 0,83 medido (el espejo con extrapolación congelada, lead_lag.rs:316,
/// predecía ≈ 0,88):
/// Pearson sobre niveles autocorrelados (Granger–Newbold) contra un umbral
/// que supone muestras independientes, y máximo sobre 5 lags sin corregir.
/// El contrato falla también con ruido blanco (ver el test siguiente): el
/// arreglo debe corregir a la vez la autocorrelación y la multiplicidad.
#[test]
#[ignore = "OPEN F2.4-07 (Línea C): Pearson sobre OFI autocorrelado con umbral para muestras iid y máximo sobre 5 lags: falsos líderes muy por encima del 5 %"]
fn cf2_lead_lag_nulo_tasa_de_falsos_lideres() {
    let mut rng = Lcg(0xB5B5_0005);
    let mut falsos = 0usize;
    for _ in 0..400 {
        let mut ll = LeadLagAlphaEngine::new(50);
        let (mut x, mut y) = (0.0f64, 0.0f64);
        for k in 0..120usize {
            x = 0.9 * x + 0.1 * rng.normal();
            y = 0.9 * y + 0.1 * rng.normal(); // independiente del líder
            let ts = k as f64 * 250.0;
            ll.update_leader(true, x, ts);
            let _ = ll.predict_altcoin_impulse_con_reloj(7, y, ts + 125.0);
        }
        if ll.ultimo_lag_btc_ms > 0.0 {
            falsos += 1;
        }
    }
    let tasa = falsos as f64 / 400.0;
    assert!(tasa <= 0.083, "falsos líderes {tasa} (nominal 0,05; medido hoy ≈ 0,83)");
}

/// Verdad: el mismo nulo con líder y alt de ruido blanco (sin
/// autocorrelación): tamaño nominal 5 %, tolerancia 5 % + 3σ (≤ 0,083).
/// Aísla la multiplicidad: aun con muestras independientes, se toma el
/// máximo de |ρ| sobre 5 lags sin corregir (espejo del escéptico ≈ 0,19).
#[test]
#[ignore = "OPEN F2.4-07 (Línea C): el máximo de |ρ| sobre 5 lags sin corrección por multiplicidad infla los falsos líderes aun con ruido blanco"]
fn cf2_lead_lag_nulo_ruido_blanco_tasa_de_falsos_lideres() {
    let mut rng = Lcg(0xB5B5_0006);
    let mut falsos = 0usize;
    for _ in 0..400 {
        let mut ll = LeadLagAlphaEngine::new(50);
        for k in 0..120usize {
            let x = 0.1 * rng.normal();
            let y = 0.1 * rng.normal();
            let ts = k as f64 * 250.0;
            ll.update_leader(true, x, ts);
            let _ = ll.predict_altcoin_impulse_con_reloj(7, y, ts + 125.0);
        }
        if ll.ultimo_lag_btc_ms > 0.0 {
            falsos += 1;
        }
    }
    let tasa = falsos as f64 / 400.0;
    assert!(tasa <= 0.083, "falsos líderes con ruido blanco {tasa} (nominal 0,05; espejo ≈ 0,19)");
}

// ─────────────────────────────────────────────────────────────────
// Transfer entropy
// ─────────────────────────────────────────────────────────────────

/// Verdad: actividad Bernoulli(0,25) independiente en X y en Y; bajo
/// independencia condicional G = 2·M·ln2·TE_bits ~ χ²₂ (Wilks), y el umbral
/// 5,991 da un 5 % de rechazos. El prior KT desplaza O(1/M) con M = 19 999.
/// Tolerancia: rechazos en [1, 22] de 200 (nominal 10; P(0) = 3,5e-5 y
/// 22 = 10 + 4σ binomial).
#[test]
fn cf2_te_nulo_independiente_calibrado_chi2() {
    let mut rng = Lcg(0xB6B6_0006);
    let (ancho, ventanas) = (200u64, 20_000u64);
    let obs = ObservationWindow {
        start_ms: 0,
        end_ms: ancho * ventanas,
        bin_width_ms: ancho,
        min_windows: 64,
    };
    let mut rechazos = 0usize;
    for _ in 0..200 {
        let ax: Vec<bool> = (0..ventanas).map(|_| rng.u01() < 0.25).collect();
        let ay: Vec<bool> = (0..ventanas).map(|_| rng.u01() < 0.25).collect();
        let est = transfer_entropy_observada(
            &activos_a_eventos(&ax, ancho),
            &activos_a_eventos(&ay, ancho),
            obs,
        )
        .expect("ventana válida");
        let umbral = 5.991464547107979 / (2.0 * est.transitions as f64 * std::f64::consts::LN_2);
        if est.x_to_y_bits > umbral {
            rechazos += 1;
        }
    }
    assert!((1..=22).contains(&rechazos), "rechazos {rechazos}/200 (nominal 10)");
}

/// Verdad: x ~ Bernoulli(0,3); y_t ~ Bernoulli(0,6) si x_(t−1), Bernoulli(0,15)
/// si no. Y_(t−1) es independiente de (Y_t, X_(t−1)), así que
/// TE = I(Y_t; X_(t−1)) = h(0,285) − [0,3·h(0,6) + 0,7·h(0,15)] = 0,1440 bits.
/// Error del plug-in con M = 4999 ≈ 0,008 bits: tolerancia 0,03 ≈ 4σ. Los 19
/// surrogados circulares (pasos de n/20 ≥ 250 bins) dan ≈ 3·10⁻⁴: ninguno
/// debe igualar la TE observada.
#[test]
fn cf2_te_acople_plantado_valor_y_surrogados() {
    let mut rng = Lcg(0xB7B7_0007);
    let (ancho, n) = (200u64, 5000usize);
    let obs = ObservationWindow {
        start_ms: 0,
        end_ms: ancho * n as u64,
        bin_width_ms: ancho,
        min_windows: 64,
    };
    let ax: Vec<bool> = (0..n).map(|_| rng.u01() < 0.3).collect();
    let mut ay = vec![false; n];
    for t in 1..n {
        let p = if ax[t - 1] { 0.6 } else { 0.15 };
        ay[t] = rng.u01() < p;
    }
    let ey = activos_a_eventos(&ay, ancho);
    let te = transfer_entropy_observada(&activos_a_eventos(&ax, ancho), &ey, obs)
        .unwrap()
        .x_to_y_bits;
    assert!(
        (te - 0.1440).abs() <= 0.03,
        "TE {te} frente a I(Y_t;X_(t-1)) = 0,1440 bits"
    );
    let mut superan = 0;
    let mut maximo = 0.0f64;
    for s in 1..=19usize {
        let d = s * n / 20;
        let sx: Vec<bool> = (0..n).map(|i| ax[(i + d) % n]).collect();
        let te_s = transfer_entropy_observada(&activos_a_eventos(&sx, ancho), &ey, obs)
            .unwrap()
            .x_to_y_bits;
        maximo = maximo.max(te_s);
        if te_s >= te {
            superan += 1;
        }
    }
    assert_eq!(
        superan, 0,
        "algún surrogado circular iguala o supera la TE observada {te} (máximo {maximo})"
    );
}

// ─────────────────────────────────────────────────────────────────
// Cópula t
// ─────────────────────────────────────────────────────────────────

/// Verdad: t bivariada con ρ = 0,7 y ν = 3 ⇒ λ = 2·T₄(−√(4·0,3/1,7)) = 0,448.
/// n = 1500 (MIN_N del manifiesto, src/bin/copulas_manifest.rs:100) en lugar
/// de 2000 para que el test quede por debajo de ~5 s (nu_mle hace 13 × n × 2
/// bisecciones de 200 pasos). Cota de Daniels Var(τ̂) ≤ 2(1 − τ²)/n con
/// τ = 0,494: σ(τ̂) ≤ 0,032 y dρ/dτ = 1,12 ⇒ σ(ρ̂) ≤ 0,036, tolerancia 0,145
/// (4σ). ν̂ ∈ [2, 5] equivale a λ(0,7; ν̂) ∈ [0,343; 0,519], dentro de
/// 0,448 ± 0,11.
///
/// Revisión cruzada (CL-65): esa aserción usa la ρ verdadera y sólo
/// re-expresa ν̂ ∈ [2, 5]. Se exige además la λ que se PUBLICA, λ(ρ̂, ν̂):
/// tolerancia 0,11 + |dλ/dρ|·0,145 ≈ 0,11 + 0,83·0,145 ≈ 0,23 (propagación
/// lineal del peor caso de ρ̂).
#[test]
fn cf2_copula_t_rho07_nu3_recupera_lambda_de_cola() {
    let mut rng = Lcg(0xB8B8_0008);
    let (x, y) = muestra_t(&mut rng, 1500, 0.7, Some(3));
    let rho = rho_desde_tau(tau_kendall(&x, &y).unwrap());
    assert!((rho - 0.7).abs() <= 0.145, "rho {rho}");
    let (nu, _) = nu_mle(&rangos(&x), &rangos(&y), rho).unwrap();
    assert!((2.0..=5.0).contains(&nu), "nu {nu}");
    let lam = lambda_cola_t(0.7, nu);
    assert!(
        (lam - 0.448).abs() <= 0.11,
        "lambda {lam} (exacto 2*T_4(-sqrt(4*0,3/1,7)) = 0,448)"
    );
    let publicada = lambda_cola_t(rho, nu);
    assert!(
        (publicada - 0.448).abs() <= 0.23,
        "lambda publicada {publicada} con rho {rho} y nu {nu} (exacto 0,448)"
    );
}

/// Verdad: cópula gaussiana (λ = 0 exacto). La decisión viva es publicar un
/// par en el manifiesto si λ̂ ≥ LAMBDA_MIN = 0,05 (src/bin/copulas_manifest.rs:103)
/// con n ≥ MIN_N = 1500 (:100). Se mide la tasa de falsos manifiestos sobre
/// R = 20 réplicas de n = 1500 con ρ = 0,6 y con ρ = 0,77 (el par BTC-SOL del
/// manifiesto, donde λ(0,77; 30) ≈ 0,054 ya supera el umbral). Tolerancia:
/// ≤ 5 % + 3σ binomial con R = 20 (≤ 0,20, es decir ≤ 4 de 20) en cada ρ.
/// Hoy pasa (0/20 con ρ = 0,6 y 2/20 con ρ = 0,77), pero cuesta ≈ 2 min: va
/// ignorado por coste, como pidió el escéptico, no por un defecto abierto.
#[test]
#[ignore = "COSTE (≈ 2 min en debug, 60 000 pares por nu_mle): pasa hoy; correr con --include-ignored"]
fn cf2_copula_gaussiana_no_fabrica_cola() {
    let mut rng = Lcg(0xB9B9_0009);
    for rho_verdad in [0.6, 0.77] {
        let mut publicados = 0usize;
        let mut nus = Vec::new();
        for _ in 0..20 {
            let (x, y) = muestra_t(&mut rng, 1500, rho_verdad, None);
            let rho = rho_desde_tau(tau_kendall(&x, &y).unwrap());
            let (nu, _) = nu_mle(&rangos(&x), &rangos(&y), rho).unwrap();
            nus.push(nu);
            if lambda_cola_t(rho, nu) >= 0.05 {
                publicados += 1;
            }
        }
        assert!(
            publicados <= 4,
            "cópula gaussiana con rho {rho_verdad}: {publicados}/20 pares publicados con lambda >= 0,05 (nu {nus:?})"
        );
    }
}
