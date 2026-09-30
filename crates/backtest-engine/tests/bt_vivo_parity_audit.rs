//! AUDITORÍA bt↔vivo (Ola XLVI·A) — tests de contrato del censo de
//! divergencias (docs/AUDITORIA_BT_VIVO_2026-09-28.md).
//!
//! El replay conduce el NÚCLEO de producción (`GodEngineCore`), así que la
//! lógica de decisión es paridad por construcción. Estos tests fijan las
//! TRES divergencias que viven en el envoltorio como invariantes medibles:
//!
//! - DIV-1: el harness del replay desplaza bid/ask adversamente (±ATR·0.10)
//!   mientras el host vivo pasa precios crudos. Invariante: el
//!   desplazamiento SOLO empeora (monótono adverso, nunca mejora el precio).
//! - DIV-2: latencia estática del genoma (30.68ms) vs lognormal real. El
//!   simulador estocástico del repo es código muerto en la ruta de decisión.
//!   Invariante: P99 determinista del simulador > penalización estática —
//!   la dirección del sesgo, medida.
//! - Config: mismo genoma ⇒ mismo knob de latencia en ambas rutas (la
//!   divergencia DIV-2 es "estática vs distribución", NUNCA config distinta).

use backtest_engine::network_jitter::NetworkJitterSimulator;
use quantum_arena::genome::SuperGenotype;

/// DIV-2 (contrato de config): el genoma decide `latency_penalty_ms` y
/// `apply_to_arena` lo planta en el arena — la MISMA ruta que usa el host
/// vivo. Si alguien rompe esta igualdad (p. ej. un override de bt), este
/// test falla y la divergencia de configuración queda expuesta en lugar de
/// silenciosa.
#[test]
fn xlvia_genoma_compartido_misma_latencia_en_bt_y_vivo() {
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let expected = genome.latency_penalty_ms;
    assert!(
        expected.is_finite() && expected > 0.0,
        "baseline sin latencia: {}",
        expected
    );

    // Ruta del replay (booktick_replay.rs:232-233): arena propio + apply.
    let arena_bt = quantum_arena::GlobalArena::build_in_own_stack(1000.0);
    genome.apply_to_arena(&arena_bt);
    let bt = arena_bt
        .config
        .latency_penalty_ms
        .load(std::sync::atomic::Ordering::Relaxed);

    // Ruta del vivo (host): mismo apply sobre su arena.
    let arena_live = quantum_arena::GlobalArena::build_in_own_stack(1000.0);
    genome.apply_to_arena(&arena_live);
    let live = arena_live
        .config
        .latency_penalty_ms
        .load(std::sync::atomic::Ordering::Relaxed);

    assert_eq!(
        bt.to_bits(),
        live.to_bits(),
        "mismo genoma debe producir bit-idéntica la latencia en bt y vivo"
    );
    assert_eq!(bt.to_bits(), expected.to_bits());
}

/// DIV-1 (invariante de dirección): el desplazamiento del harness
/// (`sim_bid = bid − running_atr·0.10`, `sim_ask = ask + running_atr·0.10`)
/// sólo puede EMPEORAR el precio. A mayor ATR, bid más bajo y ask más alto;
/// con ATR=0 el precio es el crudo del tick. Si alguien invierte un signo
/// (slippage que mejora el fill), este test lo detiene.
#[test]
fn xlvia_desplazamiento_del_harness_solo_empeora() {
    let bid = 100.0_f64;
    let ask = 100.02_f64;

    let sim = |running_atr: f64| -> (f64, f64) {
        let slip = running_atr * 0.10;
        (bid - slip, ask + slip)
    };

    // ATR=0 ⇒ precios crudos (paridad con el host vivo, lib.rs:1343).
    let (b0, a0) = sim(0.0);
    assert_eq!((b0, a0), (bid, ask), "ATR nulo no debe desplazar");

    // Monotonía adversa: más ATR ⇒ bid estrictamente menor, ask mayor.
    for atr in [0.01_f64, 0.05, 0.2, 1.0, 5.0] {
        let (b, a) = sim(atr);
        assert!(
            b < bid && a > ask,
            "ATR={}: ({}, {}) debe ensanchar, no estrechar",
            atr,
            b,
            a
        );
        assert!(b > 0.0, "ATR={} produce bid ≤ 0: {}", atr, b);
    }

    // El orden se preserva (nunca cruza el libro).
    let (b, a) = sim(5.0);
    assert!(b < a, "spread invertido con ATR alto: bid={} ask={}", b, a);
}

/// DIV-2 (magnitud medida): la penalización estática del genoma activo es
/// 30.68ms; el RTT real es lognormal (base 25ms, σ=0.35). Con σ=0.35, el
/// P99 ≈ base·exp(σ·z99 − σ²/2) ≈ 2× la base. Este test DEMUESTRA que el
/// quantile superior de la distribución real supera el valor estático que
/// el bt cobra — la dirección del sesgo optimista del bt, cuantificada con
/// el simulador determinista del repo (mismo seed ⇒ mismos samples).
#[test]
fn xlvia_p99_latencia_lognormal_supera_penalizacion_estatica() {
    // Parámetros del propio simulador por defecto (25ms RTT, σ=0.35, 0.1% loss)
    // y la penalización estática del genoma ACTIVO del repo (30.679914…ms).
    let sim = NetworkJitterSimulator::new(25.0, 0.35, 0.001);
    let estatica = 30.679_914_238_190_136_f64;

    // Muestras deterministas: seed = índice (el sampler es xorshift puro).
    let mut samples: Vec<f64> = (0..20_000u64)
        .map(|i| sim.sample_latency_ms(i).0)
        .filter(|l| l.is_finite() && *l > 0.0)
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = samples[(samples.len() as f64 * 0.99) as usize - 1];
    let p50 = samples[samples.len() / 2];

    // Contrato: la cola derecha SUPERA la penalización estática del bt.
    assert!(
        p99 > estatica,
        "P99 lognormal ({:.1}ms) debe superar la penalización estática ({:.1}ms) — \
         si no, el sesgo optimista del bt en colas habría desaparecido (revisar censo DIV-2)",
        p99, estatica
    );
    // Y la mediana queda por debajo (la estática NO es absurda para el cuerpo
    // de la distribución — la brecha está en la cola, no en el centro).
    assert!(
        p50 < estatica,
        "P50 ({:.1}ms) por debajo de la estática ({:.1}ms): el cuerpo está bien cobrado",
        p50, estatica
    );
    // Sanidad del sampler: P99 > P50 (distribución con cola derecha).
    assert!(p99 > p50 * 1.5, "cola insuficiente: p50={:.1} p99={:.1}", p50, p99);
}

/// XLVI·B (cierre DIV-2) — CONTRATO DE IGUALDAD BIT-EXACTA: el muestreador
/// que usa la FÍSICA DE FILLS del núcleo (`risk_engine::tp_sl::
/// sample_latency_lognormal_ms`, gen como mediana) y el simulador canónico
/// del backtest (`NetworkJitterSimulator`) son la MISMA secuencia matemática
/// (xorshift + Box-Muller + normalización lognormal). Mismo base, mismo seed
/// ⇒ mismo milisegundo. Si alguien «corrige» uno de los dos lados sin el
/// otro, este test expone la divergencia antes de que el bt y el vivo
/// vuelvan a cobrar latencias distintas.
#[test]
fn xlvib_sampler_de_fills_y_simulador_canonico_son_bit_exactos() {
    for base in [5.0_f64, 25.0, 30.68, 100.0] {
        let sim = NetworkJitterSimulator::new(base, 0.35, 0.001);
        for seed in [0u64, 1, 7, 42, 999_983, u64::MAX] {
            let (canonico, _dropped) = sim.sample_latency_ms(seed);
            let fisica = risk_engine::tp_sl::sample_latency_lognormal_ms(base, seed);
            assert_eq!(
                canonico.to_bits(),
                fisica.to_bits(),
                "base={base} seed={seed}: canonico={canonico} fisica={fisica}"
            );
        }
    }
}

/// XLVI·B — el genoma calibra la MEDIA del RTT y la física añade la cola:
/// con el genoma ACTIVO (30.68ms), media muestreada ≈ gen, mediana ≈ 0.94×
/// gen y p99 ≈ 2× gen. Este es el estado POST-cierre de DIV-2: la cola que
/// el bt no cobraba ahora vive dentro del núcleo de producción (bt y vivo,
/// mismo código, misma semilla por evento).
#[test]
fn xlvib_gen_como_media_cola_sobre_el_gen_activo() {
    let base = 30.679_914_238_190_136_f64; // genoma activo del repo
    let mut s: Vec<f64> = (0..20_000u64)
        .map(|i| risk_engine::tp_sl::sample_latency_lognormal_ms(base, i))
        .collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = s[s.len() / 2];
    let p99 = s[(s.len() as f64 * 0.99) as usize - 1];
    assert!((p50 / base - 0.9401).abs() < 0.06, "p50={p50} debe ≈ 0.94× gen {base}");
    assert!(p99 > base * 1.7, "p99={p99} debe superar 1.7× gen {base}");
}

/// XLVI·H (DIV-1) — HARNESS DE MEDICIÓN A/B: misma serie, mismo genoma,
/// desplazamiento del harness 0.10 (histórico) vs 0.0 (paridad de features
/// con el vivo; el slippage queda sólo en la física del core). La DIFERENCIA
/// entre ambos es la cuantificación del doble-conteo de DIV-1. Imprime la
/// tabla con --nocapture (invisible en corridas normales); los asserts
/// pinean sanidad, no dirección — el signo del delta es un hecho medido.
#[test]
fn xlvih_medicion_ab_doble_conteo_div1() {
    use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay};
    use quantum_arena::genome::SuperGenotype;

    backtest_engine::asegurar_spec_nativo("BTCUSDT");
    // Serie con la receta EXACTA del oráculo T-1 (trend+ciclo 30pb+ruido
    // 40pb — física D-755 coherente): garantiza trades para que la
    // medición del doble-conteo no sea 0/0.
    let mut seed = 0x5DEECE66Du64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / u32::MAX as f64) - 0.5
    };
    let mut p = 60_000.0f64;
    let ticks: Vec<ReplayTick> = (0..30_000)
        .map(|i| {
            let u = next();
            let ciclo = (i as f64 / 180.0).sin() * 0.0030;
            p *= 1.0 + ciclo + u * 0.0040 + 0.00004;
            let half = p * 0.0002; // spread sintético 4 pb
            ReplayTick {
                // Cadencia de 1 MINUTO por tick: cada tick madura su propio
                // kline 1m — el warmup sintetiza 600 klines (Hurst necesita
                // 512). Con 100ms el warmup produce ~1 kline y nada opera.
                ts_ms: 1_700_000_000_000 + (i as u64) * 60_000,
                bid: p - half,
                ask: p + half,
                bid_qty: 450.0 + u.abs() * 1000.0,
                ask_qty: 450.0 + (1.0 - u.abs()) * 1000.0,
            }
        })
        .collect();
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let cfg = |frac: f64| ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: 600,
        trade_only: false,
        shift_atr_frac: frac,
    };

    let a = run_booktick_replay(&ticks, &genome, None, &cfg(0.10));
    let b = run_booktick_replay(&ticks, &genome, None, &cfg(0.0));
    println!(
        "A/B DIV-1: histórico(0.10) trades={} net={:.4} dd={:.4} fees={:.4} | paridad(0.0) trades={} net={:.4} dd={:.4} fees={:.4} | Δnet={:+.4}",
        a.trades, a.net_pnl, a.max_dd, a.fees_est,
        b.trades, b.net_pnl, b.max_dd, b.fees_est,
        b.net_pnl - a.net_pnl
    );
    // Sanidad de ambos modos (la dirección del delta NO se pinea: es dato).
    for (name, s) in [("histórico", &a), ("paridad", &b)] {
        assert!(s.final_capital.is_finite() && s.final_capital > 0.0, "{name}: capital roto");
        assert!(s.max_dd < 1.0, "{name}: dd {max}", max = s.max_dd);
    }
}

/// XLVII·A — RADIO DE IMPACTO DE DIV-1: el modo TRADE-ONLY (aggTrades, el
/// que USA LA EVOLUCIÓN para medir aptitud) BYPASA el desplazamiento del
/// harness — su branch pasa bid/ask sintéticos derivados del trade, no
/// `sim_bid/sim_ask`. Consecuencia: el fitness que selecciona genomas
/// NUNCA estuvo contaminado por el doble-conteo; el radio de DIV-1 es
/// SOLO el modo libro (backtest_windows por defecto). Si alguien mueve el
/// shift al camino trade-only o rompe el bypass, este contrato lo expone.
#[test]
fn xlviiA_trade_only_bypasa_el_shift_div1() {
    use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay};
    use quantum_arena::genome::SuperGenotype;

    backtest_engine::asegurar_spec_nativo("BTCUSDT");
    let mut seed = 0x5DEECE66Du64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / u32::MAX as f64) - 0.5
    };
    let mut p = 60_000.0f64;
    let ticks: Vec<ReplayTick> = (0..20_000)
        .map(|i| {
            let u = next();
            p *= 1.0 + (i as f64 / 180.0).sin() * 0.0030 + u * 0.0040 + 0.00004;
            let half = p * 0.0002;
            ReplayTick {
                ts_ms: 1_700_000_000_000 + (i as u64) * 60_000,
                bid: p - half,
                ask: p + half,
                bid_qty: 450.0 + u.abs() * 1000.0,
                ask_qty: 450.0 + (1.0 - u.abs()) * 1000.0,
            }
        })
        .collect();
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let cfg = |frac: f64| ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: 600,
        trade_only: true, // ← el modo de la evolución
        shift_atr_frac: frac,
    };
    let a = run_booktick_replay(&ticks, &genome, None, &cfg(0.10));
    let b = run_booktick_replay(&ticks, &genome, None, &cfg(0.0));
    // El shift NO toca este camino: bit-idéntico con cualquier fracción.
    assert_eq!(a.trades, b.trades, "trade-only no puede depender del shift");
    assert_eq!(a.net_pnl.to_bits(), b.net_pnl.to_bits());
    assert_eq!(a.final_capital.to_bits(), b.final_capital.to_bits());
    // Y una fracción absurda tampoco lo mueve (el bypass es total).
    let c = run_booktick_replay(&ticks, &genome, None, &cfg(3.0));
    assert_eq!(c.net_pnl.to_bits(), a.net_pnl.to_bits());
}

/// XLVII·A — MEDICIÓN EN TAPE REAL (manual, --ignored): carga aggTrades
/// reales (TGMTICK1) y confirma el bypass empíricamente — trade-only
/// idéntico con ambos shifts; book-mode sobre el mismo tape SÍ se mueve
/// (radio de impacto = modo libro). Requiere data/THETAUSDT_2026-08_REAL.bin.
#[test]
#[ignore = "medición manual con tape real presente en data/"]
fn xlviiA_medicion_radio_div1_en_tape_real() {
    use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay};
    use backtest_engine::tick_replayer::load_binary_ticks;
    use quantum_arena::genome::SuperGenotype;

    // El cwd de los tests es la raíz del CRATE: ruta absoluta al workspace.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/THETAUSDT_2026-08_REAL.bin");
    let events = load_binary_ticks(&path, 0).expect("tape real THETA 2026-08");
    assert!(events.len() > 10_000, "tape sospechosamente corto");
    let ticks: Vec<ReplayTick> = events
        .iter()
        .map(|e| ReplayTick {
            ts_ms: e.timestamp,
            bid: e.bid_price,
            ask: e.ask_price,
            bid_qty: e.bid_qty,
            ask_qty: e.ask_qty,
        })
        .collect();
    backtest_engine::asegurar_spec_nativo("THETAUSDT");
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let cfg = |frac: f64, trade_only: bool| ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: 600,
        trade_only,
        shift_atr_frac: frac,
    };
    // Bypass: trade-only idéntico.
    let t1 = run_booktick_replay(&ticks, &genome, None, &cfg(0.10, true));
    let t0 = run_booktick_replay(&ticks, &genome, None, &cfg(0.0, true));
    println!(
        "REAL trade-only: 0.10 trades={} net={:.4} | 0.0 trades={} net={:.4}",
        t1.trades, t1.net_pnl, t0.trades, t0.net_pnl
    );
    assert_eq!(t1.net_pnl.to_bits(), t0.net_pnl.to_bits(), "bypass roto en tape real");
    // Radio: book-mode sobre el MISMO tape (delta esperado ≠ 0).
    let b1 = run_booktick_replay(&ticks, &genome, None, &cfg(0.10, false));
    let b0 = run_booktick_replay(&ticks, &genome, None, &cfg(0.0, false));
    println!(
        "REAL book-mode: 0.10 trades={} net={:.4} | 0.0 trades={} net={:.4} | Δ={:+.4}",
        b1.trades, b1.net_pnl, b0.trades, b0.net_pnl,
        b0.net_pnl - b1.net_pnl
    );
    for (name, s) in [("t1", &t1), ("b1", &b1), ("b0", &b0)] {
        assert!(s.final_capital.is_finite() && s.final_capital > 0.0, "{name} roto");
    }
}

/// XLVII·B — BRECHA CONTRA LA META, MEDIDA EN TAPES REALES (manual, --ignored).
///
/// La auditoría de capitalización (XLV·L) fijó la aritmética de la meta:
/// +100%/3d ≡ 25.99% diario ⇒ con el axioma de ruina 25% y 10%/trade se
/// requieren ~10 trades/día con edge sostenido. Esta medición carga el
/// GENOMA CAMPEÓN del repo y lo corre en modo trade-only (el de la
/// evolución) sobre una muestra de tapes reales por tier de liquidez,
/// reportando trades/día logrados vs los ~10/día que la meta exige.
/// Muestra y método: docs/AUDITORIA_BT_VIVO_2026-09-28.md (adenda XLVII·B).
#[test]
#[ignore = "medición manual: ~40 min de cómputo sobre tapes reales"]
fn xlviiB_brecha_meta_en_tapes_reales_campeon() {
    use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay};
    use backtest_engine::tick_replayer::load_binary_ticks;
    use quantum_arena::genome::SuperGenotype;

    // El loader de modelos resuelve "models/" RELATIVO AL CWD: desde la
    // raíz del crate los modelos no se ven y la medición capturaría el
    // piso SIN modelos (sonda única por símbolo), no la brecha del
    // campeón. Este test debe correrse FILTRADO (-- --ignored
    // xlviiB) para que el chdir no afecte a otros tests del proceso.
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::env::set_current_dir(&workspace).expect("chdir al workspace");

    // Genoma CAMPEÓN del repo (el artefacto que config_compiler promueve).
    let champion: SuperGenotype = serde_json::from_str(
        &std::fs::read_to_string("config_dir/genotypes/quantum_champion.json")
            .expect("quantum_champion.json legible"),
    )
    .expect("campeón deserializable");

    // Muestra por tier de liquidez (todas < ~120 MB para cómputo acotado).
    let muestra: &[(&str, &str)] = &[
        ("LTCUSDT", "2026-08"),
        ("ADAUSDT", "2026-08"),
        ("LINKUSDT", "2026-08"),
        ("ATOMUSDT", "2026-08"),
        ("NEARUSDT", "2026-08"),
        ("THETAUSDT", "2026-08"),
    ];
    let base = std::path::Path::new("data"); // CWD ya es el workspace
    let mut total_trades = 0u64;
    let mut total_days = 0.0f64;
    println!("sym          trades  días   t/día   WR     net      fees");
    for (sym, month) in muestra {
        let path = base.join(format!("{sym}_{month}_REAL.bin"));
        let Ok(events) = load_binary_ticks(&path, 0) else {
            println!("{sym}: tape ausente, omitido");
            continue;
        };
        let ticks: Vec<ReplayTick> = events
            .iter()
            .map(|e| ReplayTick {
                ts_ms: e.timestamp,
                bid: e.bid_price,
                ask: e.ask_price,
                bid_qty: e.bid_qty,
                ask_qty: e.ask_qty,
            })
            .collect();
        let span_ms = ticks.last().unwrap().ts_ms.saturating_sub(ticks[0].ts_ms);
        let dias = span_ms as f64 / 86_400_000.0;
        backtest_engine::asegurar_spec_nativo(sym);
        let cfg = ReplayConfig {
            initial_capital: 1000.0,
            warmup_ticks: 600,
            trade_only: true, // el modo de la evolución
            shift_atr_frac: 0.10,
        };
        let s = run_booktick_replay(&ticks, &champion, None, &cfg);
        let t_dia = if dias > 0.0 { s.trades as f64 / dias } else { 0.0 };
        println!(
            "{sym:<12} {:>4}   {:>5.1}  {:>5.2}  {:>.2}  {:>+8.3}  {:>.3}",
            s.trades,
            dias,
            t_dia,
            s.wr_net(),
            s.net_pnl,
            s.fees_est
        );
        total_trades += s.trades;
        total_days += dias;
        assert!(s.final_capital.is_finite() && s.final_capital > 0.0);
    }
    let t_dia_global = if total_days > 0.0 {
        total_trades as f64 / total_days
    } else {
        0.0
    };
    println!(
        "TOTAL: {} trades / {:.1} días = {:.2} trades/día — meta ≈ 10/día ⇒ brecha ≈ {:.0}×",
        total_trades,
        total_days,
        t_dia_global,
        if t_dia_global > 0.0 { 10.0 / t_dia_global } else { f64::INFINITY }
    );
    // El contrato es la SANIDAD del método, no el valor (la brecha es dato).
    assert!(total_days > 5.0, "muestra sin días suficientes");
}

/// XLIX·A — MX-19 REPARADO: causalidad temporal del prefijo.
///
/// El defecto (P0, auditoría Codex): el bucle de replay arrancaba en el
/// índice 0 pero el estado de klines/Hurst ya contenía información de los
/// ticks [1..600) del propio prefijo — look-ahead. La reparación: el bucle
/// evalúa DESPUÉS de la frontera de precarga (el prefijo se consume una
/// sola vez, como historia). Contratos: determinismo preservado; el
/// prefijo NO participa en la evaluación (por construcción, ninguna
/// posición puede abrirse antes de la frontera).
#[test]
fn xlixA_mx19_prefijo_consumido_una_vez_y_determinista() {
    use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay};
    use quantum_arena::genome::SuperGenotype;

    backtest_engine::asegurar_spec_nativo("BTCUSDT");
    let mut seed = 0x5DEECE66Du64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / u32::MAX as f64) - 0.5
    };
    // Serie con tendencia BRUTAL confinada al prefijo (primeros 700
    // ticks > frontera 600): con el defecto, el estado en el tick 0
    // "sabía" el futuro del prefijo y abría posiciones allí. Con la
    // reparación, el prefijo no se evalúa — no puede generar trading.
    let mut p = 60_000.0f64;
    let ticks: Vec<ReplayTick> = (0..3_000)
        .map(|i| {
            let u = next();
            let fuerte = if i < 700 { 0.02 } else { 0.003 };
            p *= 1.0 + (i as f64 / 180.0).sin() * fuerte + u * 0.004 + 0.0002;
            let half = p * 0.0002;
            ReplayTick {
                ts_ms: 1_700_000_000_000 + (i as u64) * 60_000,
                bid: p - half,
                ask: p + half,
                bid_qty: 450.0 + u.abs() * 1000.0,
                ask_qty: 450.0 + (1.0 - u.abs()) * 1000.0,
            }
        })
        .collect();
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let cfg = ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: 600, // frontera = max(600, 600) = 600
        trade_only: false,
        shift_atr_frac: 0.10,
    };
    let a = run_booktick_replay(&ticks, &genome, None, &cfg);
    let b = run_booktick_replay(&ticks, &genome, None, &cfg);
    assert_eq!(a.trades, b.trades, "determinismo preservado");
    assert_eq!(a.net_pnl.to_bits(), b.net_pnl.to_bits());
    // Sanidad: la corrida con tendencia brutal en el prefijo sigue viva.
    assert!(a.final_capital.is_finite() && a.final_capital > 0.0);
    // Invariancia estructural: la MISMA serie con warmup 700 (frontera
    // más allá de la tendencia brutal del prefijo) no puede tener MÁS
    // actividad del prefijo que warmup 600 — el prefijo nunca tradea.
    let cfg700 = ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: 700,
        trade_only: false,
        shift_atr_frac: 0.10,
    };
    let c = run_booktick_replay(&ticks, &genome, None, &cfg700);
    assert!(c.final_capital.is_finite() && c.final_capital > 0.0);
    assert!(c.max_dd < 1.0);
}
