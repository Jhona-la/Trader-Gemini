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
//! # Scope of the references
//!
//! V1 retains the historical uniform generator; V2 uses Gaussian close noise
//! and separate wick/volume draws. Bayes estimates latent-state probabilities
//! from CLOSED bars under known DGP parameters. It is a diagnostic reference,
//! not a universal bound on selective net PnL or on the synthetic intrabar engine.
//! The runner still bridges toward the current OHLC close (H1-01 iii).
//!
//! Legacy `filtro_*` columns use always-active directions; `filtro_neto_*`
//! abstains unless expected log return pays the declared 10 bp fee, with
//! coverage reported separately. Returns are missing (NaN) when no order is
//! selected. Reference windows crossing a declared segment boundary are omitted.
//! Legacy `ev_R_*` columns are conditional on barrier resolution. The appended
//! `bracket_ev_R_*` includes expired brackets marked to the horizon close and
//! excludes right-censored entries. These reference fills assume exact barriers.
//! Gaussian wick scales and return tails differ from V1, so a V1/V2 comparison
//! changes several inputs; it does not isolate the effect of removing leakage.
//!
//! Run measurements in series: cargo test --release -p backtest-engine
//! --test t3_ventaja_conocida -- --include-ignored --nocapture --test-threads=1
//! T3_GENERADOR=v1|v2 selects the generator (v1 is the historical default).
//! Standalone model tests: rustc --edition=2024 --test
//! crates/backtest-engine/tests/support/t3_synthetic.rs -o model-tests.exe
//!
use backtest_engine::{STATS_LEN, run_backtest_native};
use quantum_arena::genome::SuperGenotype;
use risk_engine::{REJECT_COUNTERS, REJECT_NAMES, REJECT_SLOTS};
use std::sync::atomic::Ordering;

/// Capital de la cuenta del dueño: régimen micro.
const CAPITAL_MICRO: f64 = 13.0;

/// Tasa base del predictor sintético (la de un bosque real, como en el T-2).
const BASE_PREDICTOR: f64 = 0.30;

#[path = "support/t3_synthetic.rs"]
mod synthetic;
pub use synthetic::*;

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
    pub generador: Generador,
    pub techo: Techo,
    pub stats: [f64; STATS_LEN],
    pub rechazos: [u64; REJECT_SLOTS],
    /// Pago realizado por cierre, del `out_pnl` del runner.
    pub pago: Pago,
    /// PnL neto de cada cierre, en orden de cierre (`out_pnl` del runner).
    pub pnl: Vec<f64>,
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
    medir_serie(serie_de(p, generador_env()), p, capital)
}

/// Como [`medir_con_capital`] sobre una serie ya construida (p. ej. una
/// compuesta de dos tramos); `p` sólo rotula la fila y `n` se toma de la serie.
pub fn medir_serie(serie: Serie, p: Peldano, capital: f64) -> Medida {
    let p = Peldano {
        n: serie.closes.len(),
        ..p
    };
    let techo = techo(&serie);
    let generador = serie.generador;
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
            // Las otras dos entradas de la fricción del gate (XLIV-8):
            // latencia y suelo de deslizamiento; 0 las anula para aislar el
            // término del apalancamiento en `fee_impact`.
            if let Some(v) = std::env::var("T3_LATENCY_MS")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v >= 0.0)
            {
                base.latency_penalty_ms = v;
            }
            if let Some(v) = std::env::var("T3_SLIP_FLOOR")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v >= 0.0)
            {
                base.base_slippage_floor = v;
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
            (stats, pnl)
        })
        .expect("no se pudo lanzar el hilo del T-3")
        .join()
        .expect("el motor cayó en el T-3");
    let (stats, pnl) = stats;
    let segundos = inicio.elapsed().as_secs_f64();
    let despues = rechazos_ahora();
    let mut rechazos = [0u64; REJECT_SLOTS];
    for i in 0..REJECT_SLOTS {
        rechazos[i] = despues[i] - antes[i];
    }
    let cierres = (stats[1] as usize).min(pnl.len());
    let pago = Pago::de_cierres(&pnl[..cierres]);
    Medida {
        peldano: p,
        capital,
        generador,
        techo,
        stats,
        rechazos,
        pago,
        pnl: pnl[..cierres].to_vec(),
        segundos,
    }
}

/// Lo que el motor PAGÓ por cierre, leído del PnL neto de cada operación
/// (`out_pnl` del runner): es la variable que la envolvente de Kelly consume
/// fuera de micro (el pago realizado b, no la probabilidad) y la que dice si
/// el tamaño por operación se mueve o es siempre la orden mínima.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pago {
    pub n_gana: usize,
    pub n_pierde: usize,
    /// Media de los cierres netos positivos (USD).
    pub media_gana: f64,
    /// Media de |cierres netos negativos| (USD).
    pub media_pierde: f64,
    /// b realizado = media_gana / media_pierde.
    pub b_real: f64,
    /// Equilibrio implícito 1 / (1 + b_real).
    pub p_estrella_real: f64,
    /// |PnL| máximo de un cierre (USD): cota del tamaño que llegó a operar.
    pub max_abs: f64,
}

impl Pago {
    pub fn de_cierres(pnl: &[f64]) -> Self {
        let mut p = Pago::default();
        let (mut sg, mut sp) = (0.0f64, 0.0f64);
        for &x in pnl {
            if !x.is_finite() {
                continue;
            }
            if x > 0.0 {
                p.n_gana += 1;
                sg += x;
            } else if x < 0.0 {
                p.n_pierde += 1;
                sp += -x;
            }
            p.max_abs = p.max_abs.max(x.abs());
        }
        p.media_gana = if p.n_gana > 0 {
            sg / p.n_gana as f64
        } else {
            0.0
        };
        p.media_pierde = if p.n_pierde > 0 {
            sp / p.n_pierde as f64
        } else {
            0.0
        };
        p.b_real = if p.media_pierde > 0.0 {
            p.media_gana / p.media_pierde
        } else {
            f64::NAN
        };
        p.p_estrella_real = if p.b_real.is_finite() && p.b_real > 0.0 {
            1.0 / (1.0 + p.b_real)
        } else {
            f64::NAN
        };
        p
    }
}

fn linea_tsv(m: &Medida) -> String {
    let t = &m.techo;
    let s = &m.stats;
    let cero = [0u64; REJECT_SLOTS];
    format!(
        "{:.2}\t{:.4}\t{:.4}\t{}\t{:.2}\t{:.3}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.3}\t{:.3}\t{:.3}\t{:+.2}\t{}\t{:.4}\t{:.1}\t{:.4}\t{:.5}\t{:.4}\t{}\t{:.5}\t{:.5}\t{:.3}\t{:.3}\t{:.5}\t{:.1}\t{}",
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
        m.pago.n_gana,
        m.pago.media_gana,
        m.pago.media_pierde,
        m.pago.b_real,
        m.pago.p_estrella_real,
        m.pago.max_abs,
        m.segundos,
        rechazos_texto(&cero, &m.rechazos)
    ) + &format!(
        "\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.3}\t{:+.2}\t{:.4}\t{:.3}\t{:.3}\t{:.3}",
        m.generador.etiqueta(),
        t.filtro[0] * 1e4,
        t.filtro[1] * 1e4,
        t.filtro[2] * 1e4,
        t.p_tp_filtro,
        ev_en_r(t.p_tp_filtro),
        t.brier_filtro,
        t.decision_filtro,
        t.decodificable,
        t.seguro
    ) + &format!(
        "\t{:.2}\t{:.2}\t{:.2}\t{:.4}\t{:.4}\t{:.4}",
        t.filtro_neto[0] * 1e4,
        t.filtro_neto[1] * 1e4,
        t.filtro_neto[2] * 1e4,
        t.cobertura_neta[0],
        t.cobertura_neta[1],
        t.cobertura_neta[2]
    ) + &format!(
        "\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
        t.bracket_ev_r[0],
        t.bracket_ev_r[1],
        t.bracket_ev_r[2],
        t.bracket_vencidos[0],
        t.bracket_vencidos[1],
        t.bracket_vencidos[2]
    )
}

const CABECERA_TSV: &str = "a\td_pct\tsigma_pct\tn\tcapital_inicial\ta_medida\tlectura\toraculo_1v_bps\toraculo_5v_bps\toraculo_20v_bps\tproxy_1v_bps\tproxy_5v_bps\tproxy_20v_bps\tp_tp_oraculo\tp_tp_proxy\tvencidos_oraculo\tev_R_oraculo\ttrades\twr_neto\tpnl_pct_capital\tcapital_final\tpnl_neto\tdd_max\tn_gana\tmedia_gana_usd\tmedia_pierde_usd\tb_real\tpstar_real\tmax_abs_pnl_usd\tsegundos\trechazos\tgenerador\tfiltro_1v_bps\tfiltro_5v_bps\tfiltro_20v_bps\tp_tp_filtro\tev_R_filtro\tbrier_filtro\tdecision_filtro\tdecodificable\tseguro\tfiltro_neto_1v_bps\tfiltro_neto_5v_bps\tfiltro_neto_20v_bps\tcobertura_neta_1v\tcobertura_neta_5v\tcobertura_neta_20v\tbracket_ev_R_oraculo\tbracket_ev_R_proxy\tbracket_ev_R_filtro\tbracket_vencidos_oraculo\tbracket_vencidos_proxy\tbracket_vencidos_filtro";

#[test]
fn t3_tsv_header_matches_row_with_abstention() {
    let p = Peldano {
        a: 0.5,
        d: 0.0,
        sigma: 0.0025,
        n: 100,
    };
    let m = Medida {
        peldano: p,
        capital: CAPITAL_MICRO,
        generador: Generador::V2Gauss,
        techo: techo(&serie_persistente_v2(p)),
        stats: [0.0; STATS_LEN],
        rechazos: [0; REJECT_SLOTS],
        pago: Pago::de_cierres(&[]),
        pnl: vec![],
        segundos: 0.0,
    };
    let row = linea_tsv(&m);
    assert_eq!(CABECERA_TSV.split('\t').count(), row.split('\t').count());
    let values: std::collections::HashMap<_, _> =
        CABECERA_TSV.split('\t').zip(row.split('\t')).collect();
    assert_eq!(values["filtro_neto_5v_bps"], "NaN");
    assert_eq!(values["cobertura_neta_5v"], "0.0000");
}

fn imprimir(m: &Medida) {
    let t = &m.techo;
    let s = &m.stats;
    let cero = [0u64; REJECT_SLOTS];
    println!(
        "[T3] a={:.2} d={:.2}% σ={:.2}% n={} C0={:.0} USD gen={} ({:.0} s) | serie: â={:.3} lectura={:.3} oráculo(1/5/20 velas)={:+.1}/{:+.1}/{:+.1} pb proxy={:+.1}/{:+.1}/{:+.1} pb filtro={:+.1}/{:+.1}/{:+.1} pb p_tp(oráculo/proxy/filtro)={:.3}/{:.3}/{:.3} vencidos={:.2} EV_R(oráculo/filtro)={:+.2}/{:+.2} brier={:.4} decisión={:.3} decodificable={:.3} seguro={:.3} | motor: trades={} wr={:.3} pnl={:+.4} USD ({:+.2} % del capital) capital={:.4} dd={:.3} | pago: gana {:.4}/pierde {:.4} USD b={:.2} p*={:.3} max|pnl|={:.4} | rechazos: {}",
        m.peldano.a,
        m.peldano.d * 100.0,
        m.peldano.sigma * 100.0,
        m.peldano.n,
        m.capital,
        m.generador.etiqueta(),
        m.segundos,
        t.a_medida,
        t.lectura,
        t.oraculo[0] * 1e4,
        t.oraculo[1] * 1e4,
        t.oraculo[2] * 1e4,
        t.proxy[0] * 1e4,
        t.proxy[1] * 1e4,
        t.proxy[2] * 1e4,
        t.filtro[0] * 1e4,
        t.filtro[1] * 1e4,
        t.filtro[2] * 1e4,
        t.p_tp_oraculo,
        t.p_tp_proxy,
        t.p_tp_filtro,
        t.vencidos_oraculo,
        ev_en_r(t.p_tp_oraculo),
        ev_en_r(t.p_tp_filtro),
        t.brier_filtro,
        t.decision_filtro,
        t.decodificable,
        t.seguro,
        s[1] as u64,
        s[0],
        s[6],
        (s[6] / m.capital) * 100.0,
        s[2],
        s[3],
        m.pago.media_gana,
        m.pago.media_pierde,
        m.pago.b_real,
        m.pago.p_estrella_real,
        m.pago.max_abs,
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
        .map(|v| {
            v.split(',')
                .filter_map(|c| c.trim().parse::<f64>().ok())
                .collect()
        })
        .filter(|v: &Vec<f64>| !v.is_empty())
        .unwrap_or_else(|| vec![100.0, 1_000.0]);
    let sigma = 0.0025;
    let peldanos = [
        Peldano {
            a: 0.95,
            d: 0.0010,
            sigma,
            n,
        },
        Peldano {
            a: 0.90,
            d: 0.0020,
            sigma,
            n,
        },
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

/// El valle de capital: el peldaño a = 0,9 / d = 0,2 % (el de más ventaja)
/// con capitales entre la cuenta del dueño y 1 000 USD, para localizar el
/// tramo en que la cuenta deja de operar (el techo de fricción apalancada
/// vuelve al gen al salir del régimen micro: 3 a 10 órdenes mínimas) y el
/// tramo en que el tamaño por operación empieza a seguir al capital.
#[test]
#[ignore = "medición larga: cargo test --release -p backtest-engine --test t3_ventaja_conocida -- --ignored --nocapture t3_valle_de_capital"]
fn t3_valle_de_capital() {
    cargar_predictor();
    let n = std::env::var("T3_VELAS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3_000);
    let capitales: Vec<f64> = std::env::var("T3_CAPITALES")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|c| c.trim().parse::<f64>().ok())
                .collect()
        })
        .filter(|v: &Vec<f64>| !v.is_empty())
        .unwrap_or_else(|| {
            vec![
                13.0, 20.0, 26.0, 35.0, 50.0, 65.0, 100.0, 215.0, 500.0, 1_000.0,
            ]
        });
    let p = Peldano {
        a: 0.90,
        d: 0.0020,
        sigma: 0.0025,
        n,
    };
    let mut filas = Vec::new();
    for &capital in &capitales {
        let m = medir_con_capital(p, capital);
        imprimir(&m);
        filas.push(linea_tsv(&m));
    }
    println!("\n[T3-TSV] {CABECERA_TSV}");
    for f in &filas {
        println!("[T3-TSV] {f}");
    }
}

// ---------------------------------------------------------------------------
// LH-10 — LA VENTAJA CONOCIDA PUESTA EN EL TIEMPO: ¿se adapta el sistema?
//
// El motor es determinista y la serie compuesta de dos tramos tiene el
// mismo prefijo que el primer tramo solo: los cierres del primer tramo son
// los mismos en los dos casos. Lo que cambia en el segundo tramo frente al
// mismo tramo corrido en frío es, exactamente, lo que el estado aprendido en
// el primero hace con el segundo. Cuatro experimentos:
//   E1 ventaja estacionaria larga: R por operación por cuartiles del índice
//      de cierre (¿mejora con la experiencia?);
//   E2 ventaja → nulo: ¿el estado ganador agrava o reduce las pérdidas del
//      nulo frente al nulo en frío?;
//   E3 nulo → ventaja: ¿una primera mitad perdedora deja al sistema fuera de
//      la ventaja que llega después (estado absorbente)?;
//   E4 persistencia → antipersistencia: cuando el signo de la vela anterior
//      pasa a ser exactamente el contrario del siguiente, ¿invierte el motor
//      su conducta o sigue operando con la estructura vieja?
// ---------------------------------------------------------------------------

/// Serie compuesta de dos tramos: el segundo continúa el precio del primero
/// (se reescala por el cociente entre el último cierre del primero y el
/// precio base del generador, así conserva su primer retorno).
pub fn serie_concatenada(p1: Peldano, p2: Peldano) -> Serie {
    let g = generador_env();
    let mut a = serie_de(p1, g);
    let b = serie_de(p2, g);
    a.tramos.push((a.closes.len(), p2));
    let ratio = a.closes.last().copied().unwrap_or(60_000.0) / 60_000.0;
    a.closes.extend(b.closes.iter().map(|x| x * ratio));
    a.highs.extend(b.highs.iter().map(|x| x * ratio));
    a.lows.extend(b.lows.iter().map(|x| x * ratio));
    a.vols.extend(b.vols.iter().copied());
    a.estado.extend(b.estado.iter().copied());
    a
}

/// (cierres, acierto, PnL medio por cierre en USD) por cuartil del índice de
/// cierre: la experiencia acumulada, no el tiempo.
pub fn cuartiles(pnl: &[f64]) -> [(usize, f64, f64); 4] {
    let n = pnl.len();
    let mut out = [(0usize, 0.0f64, 0.0f64); 4];
    for (q, fila) in out.iter_mut().enumerate() {
        let (lo, hi) = (n * q / 4, n * (q + 1) / 4);
        let s = &pnl[lo..hi];
        if s.is_empty() {
            continue;
        }
        let ganadas = s.iter().filter(|x| **x > 0.0).count();
        *fila = (
            s.len(),
            ganadas as f64 / s.len() as f64,
            s.iter().sum::<f64>() / s.len() as f64,
        );
    }
    out
}

fn acierto(pnl: &[f64]) -> f64 {
    if pnl.is_empty() {
        return 0.0;
    }
    pnl.iter().filter(|x| **x > 0.0).count() as f64 / pnl.len() as f64
}

/// Los cuatro experimentos de adaptación (13 USD, `T3_VELAS` velas por tramo).
#[test]
#[ignore = "medición larga: cargo test --release -p backtest-engine --test t3_ventaja_conocida -- --ignored --nocapture t3_adaptacion_en_el_tiempo"]
fn t3_adaptacion_en_el_tiempo() {
    cargar_predictor();
    let n = std::env::var("T3_VELAS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3_000);
    let sigma = 0.0025;
    let ventaja = Peldano {
        a: 0.90,
        d: 0.0020,
        sigma,
        n,
    };
    let nulo = Peldano {
        a: 0.50,
        d: 0.0,
        sigma,
        n,
    };
    let anti = Peldano {
        a: 0.10,
        d: 0.0020,
        sigma,
        n,
    };
    let larga = Peldano {
        a: 0.90,
        d: 0.0020,
        sigma,
        n: 2 * n,
    };

    // Tramos en frío (referencia) y E1.
    let m_v = medir(ventaja);
    imprimir(&m_v);
    let m_n = medir(nulo);
    imprimir(&m_n);
    let m_k = medir(anti);
    imprimir(&m_k);
    let m_l = medir(larga);
    imprimir(&m_l);
    let q = cuartiles(&m_l.pnl);
    println!(
        "[T3-ADAPT] E1 ventaja larga ({} cierres): cuartiles por índice de cierre (cierres, acierto, PnL medio USD) = Q1 {:?} Q2 {:?} Q3 {:?} Q4 {:?}",
        m_l.pnl.len(),
        q[0],
        q[1],
        q[2],
        q[3]
    );

    // E5 (ola 3): antipersistencia larga por cuartiles, la prueba directa de
    // inversión: los cuartiles son descriptivos; superar ½ no demuestra inversión ni aprendizaje.
    let m_a = medir(Peldano {
        a: 0.10,
        d: 0.0020,
        sigma,
        n: 2 * n,
    });
    imprimir(&m_a);
    let qa = cuartiles(&m_a.pnl);
    println!(
        "[T3-ADAPT] E5 antipersistencia larga ({} cierres): cuartiles por índice de cierre (cierres, acierto, PnL medio USD) = Q1 {:?} Q2 {:?} Q3 {:?} Q4 {:?}",
        m_a.pnl.len(),
        qa[0],
        qa[1],
        qa[2],
        qa[3]
    );

    // Compuestos: E2, E3, E4 y E4' (anti → anti: separa calentamiento de
    // herencia; A2/A3 de la ola 2). El pago del segundo tramo y del fresco
    // va en la fila (A2-N4).
    let cero = [0u64; REJECT_SLOTS];
    println!(
        "[T3-ADAPT-TSV] experimento\ttrades\twr\tpnl_usd\tb_real\tprimer_tramo_cierres_iguales\tsegundo_tramo_trades\tsegundo_tramo_pnl_usd\tsegundo_tramo_wr\tfresco_trades\tfresco_pnl_usd\tfresco_wr\trechazos_compuesto\trechazos_fresco\tsegundo_tramo_gana_usd\tsegundo_tramo_pierde_usd\tsegundo_tramo_b_real\tfresco_gana_usd\tfresco_pierde_usd\tfresco_b_real\tgenerador"
    );
    let compuestos: [(&str, Peldano, Peldano, &Medida, &Medida); 4] = [
        ("E2 ventaja→nulo", ventaja, nulo, &m_v, &m_n),
        ("E3 nulo→ventaja", nulo, ventaja, &m_n, &m_v),
        ("E4 ventaja→anti", ventaja, anti, &m_v, &m_k),
        ("E4' anti→anti", anti, anti, &m_k, &m_k),
    ];
    println!("[T3-REF-TSV] experimento\t{CABECERA_TSV}");
    for (nombre, p1, p2, primero, fresco) in compuestos {
        let m = medir_serie(serie_concatenada(p1, p2), p1, CAPITAL_MICRO);
        imprimir(&m);
        println!("[T3-REF-TSV] {nombre}\t{}", linea_tsv(&m));
        let n1 = primero.pnl.len().min(m.pnl.len());
        let iguales = (0..n1)
            .filter(|&i| (m.pnl[i] - primero.pnl[i]).abs() < 1e-9)
            .count();
        let segundo = &m.pnl[n1..];
        let pago2 = Pago::de_cierres(segundo);
        println!(
            "[T3-ADAPT-TSV] {nombre}\t{}\t{:.4}\t{:.5}\t{:.3}\t{iguales}/{n1}\t{}\t{:.5}\t{:.4}\t{}\t{:.5}\t{:.4}\t{}\t{}\t{:.5}\t{:.5}\t{:.3}\t{:.5}\t{:.5}\t{:.3}\t{}",
            m.stats[1] as u64,
            m.stats[0],
            m.stats[6],
            m.pago.b_real,
            segundo.len(),
            segundo.iter().sum::<f64>(),
            acierto(segundo),
            fresco.stats[1] as u64,
            fresco.stats[6],
            fresco.stats[0],
            rechazos_texto(&cero, &m.rechazos),
            rechazos_texto(&cero, &fresco.rechazos),
            pago2.media_gana,
            pago2.media_pierde,
            pago2.b_real,
            fresco.pago.media_gana,
            fresco.pago.media_pierde,
            fresco.pago.b_real,
            m.generador.etiqueta()
        );
    }
}

/// E1b — LA MISMA SERIE DOS VECES. El segundo tramo es la MISMA realización
/// del ruido que el primero (la sal del generador depende de a, d y n), así
/// que la única diferencia entre las dos pasadas es el estado que el motor
/// lleva aprendido al entrar en la segunda (más el nivel de precio). Si el
/// sistema se adaptara, la segunda pasada tendría que ser distinta de la
/// primera, y a mejor; si es idéntica, lo aprendido no decide nada.
#[test]
#[ignore = "medición larga: cargo test --release -p backtest-engine --test t3_ventaja_conocida -- --ignored --nocapture t3_adaptacion_misma_serie_dos_veces"]
fn t3_adaptacion_misma_serie_dos_veces() {
    cargar_predictor();
    let n = std::env::var("T3_VELAS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3_000);
    let sigma = 0.0025;
    let cero = [0u64; REJECT_SLOTS];
    println!(
        "[T3-ADAPT2-TSV] serie\tprimera_pasada_trades\tprimera_pasada_pnl_usd\tprimera_pasada_wr\tsegunda_pasada_trades\tsegunda_pasada_pnl_usd\tsegunda_pasada_wr\tcierres_iguales_entre_pasadas\tfresco_trades\tfresco_pnl_usd\trechazos_compuesto\trechazos_fresco"
    );
    for (nombre, p) in [
        (
            "ventaja a=0,9 d=0,2%",
            Peldano {
                a: 0.90,
                d: 0.0020,
                sigma,
                n,
            },
        ),
        (
            "regimen medio a=0,9 d=0,1%",
            Peldano {
                a: 0.90,
                d: 0.0010,
                sigma,
                n,
            },
        ),
        (
            "nulo a=0,5",
            Peldano {
                a: 0.50,
                d: 0.0,
                sigma,
                n,
            },
        ),
    ] {
        let fresco = medir(p);
        imprimir(&fresco);
        let m = medir_serie(serie_concatenada(p, p), p, CAPITAL_MICRO);
        imprimir(&m);
        let n1 = fresco.pnl.len().min(m.pnl.len());
        let primera = &m.pnl[..n1];
        let segunda = &m.pnl[n1..];
        let comparables = primera.len().min(segunda.len());
        let iguales = (0..comparables)
            .filter(|&i| (primera[i] - segunda[i]).abs() < 1e-9)
            .count();
        println!(
            "[T3-ADAPT2-TSV] {nombre}\t{}\t{:.5}\t{:.4}\t{}\t{:.5}\t{:.4}\t{iguales}/{comparables}\t{}\t{:.5}\t{}\t{}",
            primera.len(),
            primera.iter().sum::<f64>(),
            acierto(primera),
            segunda.len(),
            segunda.iter().sum::<f64>(),
            acierto(segunda),
            fresco.stats[1] as u64,
            fresco.stats[6],
            rechazos_texto(&cero, &m.rechazos),
            rechazos_texto(&cero, &fresco.rechazos)
        );
    }
}
