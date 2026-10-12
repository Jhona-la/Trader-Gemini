//! T-3 — ESCALERA DE VENTAJA CONOCIDA (LH-1, Línea H).
//!
//! # Por qué existe
//!
//! El T-1 certifica que los genes mueven algo sobre una tendencia determinista
//! y el T-2 mide las puertas sobre una serie lateral sin ventaja. Ninguno dice
//! nada de la meta: el T-2 pierde lo que cuesta operar, y eso es correcto en
//! una serie sin ventaja. La pregunta que falta es la inversa: **si la serie
//! tiene una ventaja conocida, ¿la cadena de decisión la deja pasar y la
//! compone, o la mata?** Y si la mata, ¿qué puerta y con qué regla?
//!
//! # Fixture
//!
//! Cadena de Markov en el signo del retorno por vela de 1 min: el estado
//! oculto `s_t ∈ {+1, −1}` se conserva con probabilidad `a` y se invierte con
//! `1 − a`; el retorno logarítmico de la vela es `s_t·d + ruido` con ruido
//! uniforme centrado de desviación `σ`. Con `a = 0,5` no hay estructura (el
//! signo es independiente de vela a vela); con `a → 1` los regímenes duran
//! `1/(1 − a)` velas de media y la deriva acumulada domina al ruido. Máximos,
//! mínimos y volúmenes se sintetizan como en el T-2. La cuenta es la del
//! dueño (13 USD, régimen micro) y el lote es fino como en el T-2 (CL-98b).
//!
//! # La ventaja «conocida»
//!
//! Para cada peldaño el propio test calcula, SOBRE LA MISMA SERIE y sin el
//! motor, lo que ganaría quien leyera el estado oculto (oráculo) y lo que
//! ganaría quien sólo viera el signo de la vela anterior (el proxy que ven
//! las features de flujo del núcleo): retorno medio a 1, 5 y 20 velas en la
//! dirección de la lectura, y la probabilidad de que un bracket RR 2,25 con
//! stop del 0,5 % toque el TP antes que el SL (resuelto con máximos y mínimos
//! de vela; si ambos se tocan en la misma vela cuenta el SL). Eso convierte el
//! resultado del motor en una fracción de un techo medido, no en un número
//! suelto.
//!
//! # Qué mide
//!
//! Por peldaño: operaciones, acierto neto, PnL neto, capital final, drawdown
//! máximo y el reparto de los rechazos del riesgo (`risk_engine::REJECT_*`)
//! como diferencia de contadores antes y después de la corrida. No es un
//! trinquete: publica la tabla para que cada línea compare su cadena contra
//! la ventaja que la serie tenía.
//!
//! Uso: `cargo test --release -p backtest-engine --test t3_ventaja_conocida
//! -- --include-ignored --nocapture`.

use backtest_engine::{STATS_LEN, run_backtest_native};
use quantum_arena::genome::SuperGenotype;
use risk_engine::{REJECT_COUNTERS, REJECT_NAMES, REJECT_SLOTS};
use std::sync::atomic::Ordering;

/// Capital de la cuenta del dueño: régimen micro.
const CAPITAL_MICRO: f64 = 13.0;

/// Tasa base del predictor sintético (la de un bosque real, como en el T-2).
const BASE_PREDICTOR: f64 = 0.30;

/// Fricción de ida y vuelta de referencia para los techos: dos piernas taker
/// del spec del fixture (0,05 % cada una). El motor paga además la horquilla
/// del genoma; el techo es por tanto optimista en ese término.
const FRICCION_RT: f64 = 2.0 * 0.0005;

/// Geometría de referencia del bracket de los techos (RR 2,25, stop 0,5 %).
const SL_REF: f64 = 0.005;
const RR_REF: f64 = 2.25;

/// Un peldaño de la escalera.
#[derive(Clone, Copy, Debug)]
pub struct Peldano {
    /// Persistencia del signo: P(s_t = s_{t−1}).
    pub a: f64,
    /// Deriva por vela en fracción (0,001 = 0,1 %).
    pub d: f64,
    /// Desviación del ruido por vela en fracción.
    pub sigma: f64,
    /// Velas de 1 min.
    pub n: usize,
}

/// Serie sintética con su estado oculto.
pub struct Serie {
    pub closes: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub vols: Vec<f64>,
    /// Estado oculto por vela (+1/−1).
    pub estado: Vec<i8>,
}

/// Generador determinista (LCG de 64 bits, misma familia que el T-2) con una
/// sal por peldaño para que dos peldaños distintos no compartan el ruido.
pub fn serie_persistente(p: Peldano) -> Serie {
    let n = p.n;
    let mut closes = Vec::with_capacity(n);
    let mut highs = Vec::with_capacity(n);
    let mut lows = Vec::with_capacity(n);
    let mut vols = Vec::with_capacity(n);
    let mut estado = Vec::with_capacity(n);
    let sal = ((p.a * 1e6) as u64) ^ (((p.d * 1e8) as u64) << 20) ^ ((n as u64) << 44);
    let mut seed = 0x2545F4914F6CDD1Du64 ^ sal.wrapping_mul(0x9E3779B97F4A7C15);
    let mut uniforme = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    // Uniforme centrada en (−0,5, 0,5): σ = 1/√12.
    let escala_ruido = p.sigma / (1.0 / 12f64.sqrt());
    let mut ln_p = 60_000.0f64.ln();
    let mut s: i8 = 1;
    for _ in 0..n {
        if uniforme() > p.a {
            s = -s;
        }
        let u = uniforme() - 0.5;
        let ruido = u * escala_ruido;
        ln_p += f64::from(s) * p.d + ruido;
        let px = ln_p.exp();
        closes.push(px);
        highs.push(px * (1.0 + 0.0015 + u.abs() * 0.0020));
        lows.push(px * (1.0 - 0.0015 - u.abs() * 0.0020));
        vols.push(900.0 + u.abs() * 2_000.0);
        estado.push(s);
    }
    Serie {
        closes,
        highs,
        lows,
        vols,
        estado,
    }
}

/// Lo que la serie permitía ganar, medido sin el motor.
#[derive(Clone, Copy, Debug, Default)]
pub struct Techo {
    /// Persistencia medida del estado oculto.
    pub a_medida: f64,
    /// P(signo del retorno de la vela = estado oculto): cuánto deja ver el ruido.
    pub lectura: f64,
    /// Retorno medio neto de fricción a 1, 5 y 20 velas siguiendo al oráculo.
    pub oraculo: [f64; 3],
    /// Lo mismo siguiendo al signo de la vela anterior (el proxy del flujo).
    pub proxy: [f64; 3],
    /// P(TP antes que SL) del bracket de referencia siguiendo al oráculo.
    pub p_tp_oraculo: f64,
    /// P(TP antes que SL) siguiendo al proxy.
    pub p_tp_proxy: f64,
    /// Fracción de brackets que vencen (ni TP ni SL en 60 velas), oráculo.
    pub vencidos_oraculo: f64,
}

fn retorno_medio(serie: &Serie, direccion: impl Fn(usize) -> f64, k: usize) -> f64 {
    let n = serie.closes.len();
    if n <= k + 1 {
        return 0.0;
    }
    let mut suma = 0.0;
    let mut cuenta = 0usize;
    for t in 1..(n - k) {
        let dir = direccion(t);
        if dir == 0.0 {
            continue;
        }
        suma += dir * (serie.closes[t + k].ln() - serie.closes[t].ln());
        cuenta += 1;
    }
    if cuenta == 0 {
        0.0
    } else {
        suma / cuenta as f64 - FRICCION_RT
    }
}

/// Primer toque del bracket (SL_REF, RR_REF·SL_REF) abierto al cierre de `t`
/// en la dirección `dir`, resuelto con máximos y mínimos hasta `horizonte`
/// velas; devuelve (tp, sl, vencido).
fn primer_toque(serie: &Serie, t: usize, dir: f64, horizonte: usize) -> (bool, bool, bool) {
    let entrada = serie.closes[t];
    let tp = entrada * (1.0 + dir * RR_REF * SL_REF);
    let sl = entrada * (1.0 - dir * SL_REF);
    let n = serie.closes.len();
    for j in (t + 1)..(t + 1 + horizonte).min(n) {
        let (hi, lo) = (serie.highs[j], serie.lows[j]);
        let toca_sl = if dir > 0.0 { lo <= sl } else { hi >= sl };
        let toca_tp = if dir > 0.0 { hi >= tp } else { lo <= tp };
        if toca_sl {
            return (false, true, false);
        }
        if toca_tp {
            return (true, false, false);
        }
    }
    (false, false, true)
}

fn p_primer_toque(serie: &Serie, direccion: impl Fn(usize) -> f64) -> (f64, f64) {
    let n = serie.closes.len();
    let (mut tps, mut sls, mut vencidos, mut total) = (0usize, 0usize, 0usize, 0usize);
    for t in 1..n.saturating_sub(1) {
        let dir = direccion(t);
        if dir == 0.0 {
            continue;
        }
        let (tp, sl, v) = primer_toque(serie, t, dir, 60);
        tps += tp as usize;
        sls += sl as usize;
        vencidos += v as usize;
        total += 1;
    }
    if total == 0 {
        return (0.0, 0.0);
    }
    let resueltos = (tps + sls).max(1);
    (tps as f64 / resueltos as f64, vencidos as f64 / total as f64)
}

pub fn techo(serie: &Serie) -> Techo {
    let n = serie.closes.len();
    let mut iguales = 0usize;
    let mut lecturas = 0usize;
    for t in 1..n {
        iguales += (serie.estado[t] == serie.estado[t - 1]) as usize;
        let signo = (serie.closes[t] - serie.closes[t - 1]).signum();
        lecturas += (signo == f64::from(serie.estado[t])) as usize;
    }
    let oraculo = |t: usize| f64::from(serie.estado[t]);
    let proxy = |t: usize| (serie.closes[t] - serie.closes[t - 1]).signum();
    let (p_tp_oraculo, vencidos_oraculo) = p_primer_toque(serie, oraculo);
    let (p_tp_proxy, _) = p_primer_toque(serie, proxy);
    Techo {
        a_medida: iguales as f64 / (n - 1).max(1) as f64,
        lectura: lecturas as f64 / (n - 1).max(1) as f64,
        oraculo: [
            retorno_medio(serie, oraculo, 1),
            retorno_medio(serie, oraculo, 5),
            retorno_medio(serie, oraculo, 20),
        ],
        proxy: [
            retorno_medio(serie, proxy, 1),
            retorno_medio(serie, proxy, 5),
            retorno_medio(serie, proxy, 20),
        ],
        p_tp_oraculo,
        p_tp_proxy,
        vencidos_oraculo,
    }
}

/// Esperanza en R del bracket de referencia con una p de TP: p·RR − (1 − p) − f/SL.
fn ev_en_r(p_tp: f64) -> f64 {
    p_tp * RR_REF - (1.0 - p_tp) - FRICCION_RT / SL_REF
}

/// Lote fino del símbolo del fixture (ver el T-2).
fn registrar_lote_fino() {
    quantum_arena::symbol_registry::update_registry(vec![
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "BTCUSDT".into(),
            step_size: 0.00001,
            tick_size: 0.1,
            min_qty: 0.00001,
            min_notional: 5.0,
            max_leverage: 50,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        },
    ]);
}

/// El mismo predictor sintético del T-2: un tocón sobre la feature 0 del
/// vector del bosque (la separación EMA rápida/lenta del núcleo), con la
/// tasa base de un bosque real. En una serie persistente ese tocón lee el
/// régimen con el retardo de las EMAs: es un lector imperfecto, como lo
/// sería un bosque entrenado.
fn cargar_predictor() {
    registrar_lote_fino();
    // El runner nativo imprime sus contadores de puertas del núcleo (abiertas,
    // vetos del consejo, vetos del ML, cierres) sólo con esta variable; la
    // escalera los necesita para atribuir lo que el riesgo no ve.
    // SAFETY: se fija antes de lanzar cualquier hilo del motor, en el hilo
    // de la prueba.
    unsafe { std::env::set_var("TG_TRACE_NATIVO", "1") };
    let confident = god_engine_core::ml_inference::NanoForestData {
        children_left: vec![1, -1, -1],
        children_right: vec![2, -1, -1],
        feature: vec![0, -1, -1],
        threshold: vec![0.0, 0.0, 0.0],
        value: vec![0.0, -3.0, 3.0],
        tree_offsets: vec![0, 3],
        init_score: (BASE_PREDICTOR / (1.0 - BASE_PREDICTOR)).ln() as _,
    };
    let forest = god_engine_core::ml_inference::NanoForest::from_data(confident)
        .expect("forest sintético del T-3 fuera de contrato");
    god_engine_core::ml_inference::NanoForest::store_global("BTCUSDT_MOTOR", forest);
}

fn rechazos_ahora() -> [u64; REJECT_SLOTS] {
    let mut v = [0u64; REJECT_SLOTS];
    for (i, c) in REJECT_COUNTERS.iter().enumerate() {
        v[i] = c.load(Ordering::Relaxed);
    }
    v
}

fn rechazos_texto(antes: &[u64; REJECT_SLOTS], despues: &[u64; REJECT_SLOTS]) -> String {
    let partes: Vec<String> = (0..REJECT_SLOTS)
        .filter(|&i| despues[i] > antes[i])
        .map(|i| format!("{}={}", REJECT_NAMES[i], despues[i] - antes[i]))
        .collect();
    if partes.is_empty() {
        "sin rechazos".into()
    } else {
        partes.join(" ")
    }
}

/// Resultado del motor sobre un peldaño.
#[derive(Clone, Debug)]
pub struct Medida {
    pub peldano: Peldano,
    /// Capital inicial de la corrida (USD).
    pub capital: f64,
    pub techo: Techo,
    pub stats: [f64; STATS_LEN],
    pub rechazos: [u64; REJECT_SLOTS],
    pub segundos: f64,
}

/// Corre el motor canónico (`run_backtest_native`) sobre la serie del peldaño
/// en un hilo con pila amplia; los contadores de rechazo se toman como
/// diferencia, así que los peldaños deben correr EN SERIE.
pub fn medir(p: Peldano) -> Medida {
    medir_con_capital(p, CAPITAL_MICRO)
}

/// Como [`medir`], con el capital inicial dado: la meta es componer, así que
/// lo que vale con 13 USD tiene que seguir valiendo al doblar.
pub fn medir_con_capital(p: Peldano, capital: f64) -> Medida {
    let serie = serie_persistente(p);
    let techo = techo(&serie);
    let antes = rechazos_ahora();
    let inicio = std::time::Instant::now();
    let stats = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            let mut base = SuperGenotype::new_baseline(0.0002, 0.0005);
            // Intervención controlada sobre UN gen (para aislar una puerta):
            // `T3_MAX_FEE_PCT` sustituye el techo de fricción apalancada del
            // genoma base (taker × 20 = 1 %). Sólo diagnóstico.
            if let Some(v) = std::env::var("T3_MAX_FEE_PCT")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
            {
                base.max_fee_pct = v;
            }
            let mut pnl = vec![0.0f64; serie.closes.len().max(1)];
            let mut stats = [0.0f64; STATS_LEN];
            run_backtest_native(
                &serie.closes,
                &serie.highs,
                &serie.lows,
                &serie.vols,
                &base,
                &mut pnl,
                &mut stats,
                "BTCUSDT",
                capital,
            );
            stats
        })
        .expect("no se pudo lanzar el hilo del T-3")
        .join()
        .expect("el motor cayó en el T-3");
    let segundos = inicio.elapsed().as_secs_f64();
    let despues = rechazos_ahora();
    let mut rechazos = [0u64; REJECT_SLOTS];
    for i in 0..REJECT_SLOTS {
        rechazos[i] = despues[i] - antes[i];
    }
    Medida {
        peldano: p,
        capital,
        techo,
        stats,
        rechazos,
        segundos,
    }
}

fn linea_tsv(m: &Medida) -> String {
    let t = &m.techo;
    let s = &m.stats;
    let cero = [0u64; REJECT_SLOTS];
    format!(
        "{:.2}\t{:.4}\t{:.4}\t{}\t{:.2}\t{:.3}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.3}\t{:.3}\t{:.3}\t{:+.2}\t{}\t{:.4}\t{:.1}\t{:.4}\t{:.5}\t{:.4}\t{:.1}\t{}",
        m.peldano.a,
        m.peldano.d * 100.0,
        m.peldano.sigma * 100.0,
        m.peldano.n,
        m.capital,
        t.a_medida,
        t.lectura,
        t.oraculo[0] * 1e4,
        t.oraculo[1] * 1e4,
        t.oraculo[2] * 1e4,
        t.proxy[0] * 1e4,
        t.proxy[1] * 1e4,
        t.proxy[2] * 1e4,
        t.p_tp_oraculo,
        t.p_tp_proxy,
        t.vencidos_oraculo,
        ev_en_r(t.p_tp_oraculo),
        s[1] as u64,
        s[0],
        (s[6] / m.capital) * 100.0,
        s[2],
        s[6],
        s[3],
        m.segundos,
        rechazos_texto(&cero, &m.rechazos)
    )
}

const CABECERA_TSV: &str = "a\td_pct\tsigma_pct\tn\tcapital_inicial\ta_medida\tlectura\toraculo_1v_bps\toraculo_5v_bps\toraculo_20v_bps\tproxy_1v_bps\tproxy_5v_bps\tproxy_20v_bps\tp_tp_oraculo\tp_tp_proxy\tvencidos_oraculo\tev_R_oraculo\ttrades\twr_neto\tpnl_pct_capital\tcapital_final\tpnl_neto\tdd_max\tsegundos\trechazos";

fn imprimir(m: &Medida) {
    let t = &m.techo;
    let s = &m.stats;
    let cero = [0u64; REJECT_SLOTS];
    println!(
        "[T3] a={:.2} d={:.2}% σ={:.2}% n={} C0={:.0} USD ({:.0} s) | serie: â={:.3} lectura={:.3} oráculo(1/5/20 velas)={:+.1}/{:+.1}/{:+.1} pb proxy={:+.1}/{:+.1}/{:+.1} pb p_tp(oráculo/proxy)={:.3}/{:.3} vencidos={:.2} EV_R(oráculo)={:+.2} | motor: trades={} wr={:.3} pnl={:+.4} USD ({:+.2} % del capital) capital={:.4} dd={:.3} | rechazos: {}",
        m.peldano.a,
        m.peldano.d * 100.0,
        m.peldano.sigma * 100.0,
        m.peldano.n,
        m.capital,
        m.segundos,
        t.a_medida,
        t.lectura,
        t.oraculo[0] * 1e4,
        t.oraculo[1] * 1e4,
        t.oraculo[2] * 1e4,
        t.proxy[0] * 1e4,
        t.proxy[1] * 1e4,
        t.proxy[2] * 1e4,
        t.p_tp_oraculo,
        t.p_tp_proxy,
        t.vencidos_oraculo,
        ev_en_r(t.p_tp_oraculo),
        s[1] as u64,
        s[0],
        s[6],
        (s[6] / m.capital) * 100.0,
        s[2],
        s[3],
        rechazos_texto(&cero, &m.rechazos)
    );
}

/// Los peldaños de la escalera: el nulo (a = 0,5 sin deriva), la persistencia
/// creciente con deriva fija y la deriva creciente con persistencia fija.
pub fn escalera(n: usize) -> Vec<Peldano> {
    let sigma = 0.0025;
    let mut v = vec![Peldano {
        a: 0.5,
        d: 0.0,
        sigma,
        n,
    }];
    for a in [0.6, 0.7, 0.8, 0.9, 0.95] {
        v.push(Peldano {
            a,
            d: 0.0010,
            sigma,
            n,
        });
    }
    for d in [0.0005, 0.0020] {
        v.push(Peldano {
            a: 0.9,
            d,
            sigma,
            n,
        });
    }
    v
}

/// El generador hace lo que dice: persistencia medida ≈ a, y la ventaja del
/// oráculo crece con a. Es una propiedad de la serie, no del motor.
#[test]
fn t3_la_serie_tiene_la_ventaja_que_declara() {
    let n = 20_000;
    let nulo = techo(&serie_persistente(Peldano {
        a: 0.5,
        d: 0.0,
        sigma: 0.0025,
        n,
    }));
    assert!(
        (nulo.a_medida - 0.5).abs() < 0.02,
        "persistencia medida {:.3} lejos de 0,5",
        nulo.a_medida
    );
    assert!(
        nulo.oraculo[1].abs() < 0.0004 + FRICCION_RT && nulo.p_tp_oraculo < 0.45,
        "sin deriva el oráculo no debería tener ventaja: 5v={:+.2e} p_tp={:.3}",
        nulo.oraculo[1],
        nulo.p_tp_oraculo
    );
    let mut anterior = f64::NEG_INFINITY;
    for a in [0.6, 0.8, 0.95] {
        let t = techo(&serie_persistente(Peldano {
            a,
            d: 0.0010,
            sigma: 0.0025,
            n,
        }));
        assert!(
            (t.a_medida - a).abs() < 0.02,
            "a={a}: persistencia medida {:.3}",
            t.a_medida
        );
        assert!(
            t.oraculo[1] > anterior,
            "a={a}: la ventaja a 5 velas ({:+.2e}) no crece con la persistencia ({:+.2e})",
            t.oraculo[1],
            anterior
        );
        anterior = t.oraculo[1];
    }
}

/// Un peldaño fuerte sin `--ignored`: si el motor no opera sobre una serie
/// con regímenes de 20 velas y ventaja neta a 5 velas, la escalera no mide
/// nada. Imprime la fila completa.
#[test]
fn t3_diag_un_peldano_persistente_micro() {
    cargar_predictor();
    let m = medir(Peldano {
        a: 0.95,
        d: 0.0010,
        sigma: 0.0025,
        n: 2_000,
    });
    imprimir(&m);
    println!("[T3-TSV] {CABECERA_TSV}");
    println!("[T3-TSV] {}", linea_tsv(&m));
    assert!(
        m.techo.oraculo[1] > 0.0,
        "el fixture no tiene ventaja neta a 5 velas: {:+.2e}",
        m.techo.oraculo[1]
    );
    assert!(
        m.stats[1] >= 10.0,
        "el motor casi no opera sobre el peldaño persistente ({} cierres)",
        m.stats[1]
    );
}

/// La escalera completa, en serie (los contadores de rechazo son globales).
#[test]
#[ignore = "medición larga: cargo test --release -p backtest-engine --test t3_ventaja_conocida -- --ignored --nocapture"]
fn t3_escalera_de_ventaja_conocida() {
    cargar_predictor();
    let n = std::env::var("T3_VELAS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3_000);
    let mut filas = Vec::new();
    for p in escalera(n) {
        let m = medir(p);
        imprimir(&m);
        filas.push(linea_tsv(&m));
    }
    println!("\n[T3-TSV] {CABECERA_TSV}");
    for f in &filas {
        println!("[T3-TSV] {f}");
    }
}

/// Los dos peldaños con ventaja clara, con más capital: la meta es componer,
/// así que la fracción del capital que el motor arriesga y gana tiene que
/// seguir valiendo al doblar la cuenta. `T3_CAPITALES` (coma) fija los
/// capitales; por defecto 100 y 1 000 USD.
#[test]
#[ignore = "medición larga: cargo test --release -p backtest-engine --test t3_ventaja_conocida -- --ignored --nocapture t3_escalera_por_capital"]
fn t3_escalera_por_capital() {
    cargar_predictor();
    let n = std::env::var("T3_VELAS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3_000);
    let capitales: Vec<f64> = std::env::var("T3_CAPITALES")
        .ok()
        .map(|v| v.split(',').filter_map(|c| c.trim().parse::<f64>().ok()).collect())
        .filter(|v: &Vec<f64>| !v.is_empty())
        .unwrap_or_else(|| vec![100.0, 1_000.0]);
    let sigma = 0.0025;
    let peldanos = [
        Peldano { a: 0.95, d: 0.0010, sigma, n },
        Peldano { a: 0.90, d: 0.0020, sigma, n },
    ];
    let mut filas = Vec::new();
    for &capital in &capitales {
        for p in peldanos {
            let m = medir_con_capital(p, capital);
            imprimir(&m);
            filas.push(linea_tsv(&m));
        }
    }
    println!("\n[T3-TSV] {CABECERA_TSV}");
    for f in &filas {
        println!("[T3-TSV] {f}");
    }
}
