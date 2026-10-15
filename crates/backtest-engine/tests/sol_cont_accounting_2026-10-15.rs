//! SOL-CONT-2026-10-15 — caracterización de la contabilidad del replay de
//! ticks y testigo de los huecos ABIERTOS SOL-R5-02 y SOL-R4-01.
//!
//! QUÉ: fija, con ejecución real del motor de replay, la semántica HOY del
//! panel que la evolución usa para puntuar genomas: capital terminal,
//! drawdown, span del panel y gate de warmup.
//!
//! POR QUÉ (hallazgos, con ancla en `main = 78647b12`):
//! - **SOL-R5-02 [ABIERTO]** — `unified_capital` (cash) sólo recibe fees en
//!   el caliente: únicos sitios de escritura medidos con grep sobre todo
//!   `crates/god-engine-core/src/lib.rs`: líneas 9359 (`fetch_add(-fee_paid)`)
//!   y 9381 (`fetch_add(+fee_paid)`); el resto son lecturas. En
//!   `booktick_replay.rs` el `max_dd` (líneas 605-612) y
//!   `stats.final_capital` (línea 618) se leen de ese cash SIN marca de
//!   posiciones abiertas, mientras `continuous_evolution_backtest.rs`
//!   (`daily_report`, líneas 124-143, y la valuación de mutantes, líneas
//!   815-837) SÍ marca equity con la fórmula
//!   `qty·(precio−entrada)·±1 − qty·precio·0.0005`. El camino de fitness y
//!   el camino de reporte miden capitales distintos con posiciones abiertas
//!   («dos caras sin reconciliar», octava confirmación del consejo).
//! - **SOL-R4-01 [ABIERTO en su registro]** — el panel anualiza con
//!   `span = ticks.last().ts − ticks[0].ts` (línea 637): si el replay rompe
//!   por ruina (línea 613, `cap <= 0.0`), el span reportado sigue siendo el
//!   de la CINTA COMPLETA, no el del intervalo ejecutado. No existe campo
//!   `executed_*` en `ReplayStats`, y `continuous_evolution_backtest.rs`
//!   calcula `total_simulation_days = min(solicitado, disponible)` (líneas
//!   311-313) sin registrar solicitado/disponible/ejecutado ni causa de fin
//!   (el REKT de la línea 803 imprime el día; el agotamiento de cinta y el
//!   corte por `current_capital <= 1.0` de la línea 549 no se distinguen).
//!
//! PARA QUÉ: estos tests NO son un RED de corrección — fijan la semántica
//! actual para que cualquier cambio de marca/span futuro sea EXPLÍCITO y
//! rompa aquí. El testigo aritmético cuantificado vive en
//! `scripts/sol/testigo_contabilidad_replay_2026-10-15.rs` y los recibos en
//! `outputs/Sol_continuacion_replay_2026-10-15.*`.
//!
//! ALCANCE: cero cambios de runtime; cero entrenamiento; cero operación viva.

use backtest_engine::booktick_replay::{run_booktick_replay, ReplayConfig, ReplayTick};
use quantum_arena::genome::SuperGenotype;

/// Cinta determinista (misma receta que los tests del propio replay):
/// caminata suave sin direccionalidad falsa, 100 ms entre ticks.
fn synth_ticks(n: usize) -> Vec<ReplayTick> {
    let mut seed: u64 = 7;
    let mut p = 60_000.0f64;
    (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
            p *= 1.0 + noise * 0.0004;
            ReplayTick {
                ts_ms: 1_700_000_000_000 + i as u64 * 100,
                bid: p - 0.5,
                ask: p + 0.5,
                bid_qty: 1.0 + noise.abs(),
                ask_qty: 1.0 + (1.0 - noise.abs()),
            }
        })
        .collect()
}

const DIA_MS: u64 = 86_400_000;

/// Con warmup = longitud de cinta, `suppress_entries` (línea 342) bloquea
/// TODAS las entradas — pero antes, el guard de degeneración (líneas
/// 278-283: `ticks.len() - warmup <= 10`) devuelve `ReplayStats::default()`.
/// Caracterización EXACTA de ese camino: `final_capital = 0.0` es el
/// centinela «sin replay», NO una ruina medida. Los consumidores
/// (`evolver.rs:104`, `evolution.rs:347/478`, `backtest_windows.rs:270`)
/// leen `final_capital` sin filtrar este centinela — observación al consejo.
#[test]
fn sol_cont_2026_10_15_warmup_degenerado_devuelve_centinela_no_ruina() {
    let ticks = synth_ticks(20_000);
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let cfg = ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: ticks.len(), // 0 ticks tras warmup ⇒ guard 278-283
        trade_only: false,
        shift_atr_frac: 0.10,
    };
    let s = run_booktick_replay(&ticks, &genome, None, &cfg);

    // Centinela del camino degenerado: stats por defecto, NO ejecución.
    assert_eq!(s.trades, 0);
    assert_eq!(s.final_capital, 0.0, "centinela 'sin replay' (default), no capital medido");
    assert_eq!(s.net_pnl, 0.0);
    assert_eq!(s.max_dd, 0.0);
    assert_eq!(s.metrics.n_trades, 0);
    assert_eq!(s.metrics.span_days, 0.0, "camino degenerado: no hay span ejecutado que reportar");
}

/// Warmup válido (≥ 10 ticks de margen): el gate de entradas se respeta y
/// el panel reporta el span de la CINTA COMPLETA (línea 637), no un span
/// ejecutado — caracterización de SOL-R4-01. Con margen de sólo 11 ticks
/// tras el warmup, la caminata suave no abre posición: 0 trades
/// deterministas y capital plano.
#[test]
fn sol_cont_2026_10_15_sin_entradas_el_panel_reporta_la_cinta_no_la_ejecucion() {
    let ticks = synth_ticks(20_000);
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let cfg = ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: ticks.len() - 11, // margen mínimo del guard: 11 > 10
        trade_only: false,
        shift_atr_frac: 0.10,
    };
    let s = run_booktick_replay(&ticks, &genome, None, &cfg);

    // Gate de warmup: sin entradas no hay trades ni fricción.
    assert_eq!(s.trades, 0);
    assert_eq!(s.final_capital, 1000.0);
    assert_eq!(s.max_dd, 0.0);
    assert_eq!(s.net_pnl, 0.0);

    // SOL-R4-01 (caracterización): el span del panel es el de la CINTA
    // COMPLETA (línea 637). Con 20 000 ticks a 100 ms → 1 999 900 ms.
    let span_esperado_ms = ticks.last().unwrap().ts_ms - ticks[0].ts_ms;
    assert_eq!(span_esperado_ms, 1_999_900);
    let span_dias = span_esperado_ms as f64 / DIA_MS as f64;
    assert!(
        (s.metrics.span_days - span_dias).abs() < 1e-12,
        "span reportado {:.9} != span de cinta {:.9} — si cambia, avisar: \
         el contrato de SOL-R4-01 exige span EJECUTADO, no de cinta",
        s.metrics.span_days,
        span_dias
    );

    // Sin trades, el panel no fabrica ratios (NaN explícito, no cero).
    assert!(s.metrics.cagr == 0.0, "sin trades y capital plano, CAGR = 0 exacto");
    assert!(s.metrics.sharpe_ann.is_nan());
    assert!(s.metrics.profit_factor.is_nan());
}

/// Con warmup normal, el replay arranca y las estadísticas son finitas y
/// deterministas (bit a bit en dos corridas). Caracterización de conducta,
/// NO evidencia de crecimiento: cinta sintética, fixture, sin economía.
#[test]
fn sol_cont_2026_10_15_replay_es_determinista_y_finito() {
    let ticks = synth_ticks(20_000);
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let cfg = ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: 200,
        trade_only: false,
        shift_atr_frac: 0.10,
    };
    let a = run_booktick_replay(&ticks, &genome, None, &cfg);
    let b = run_booktick_replay(&ticks, &genome, None, &cfg);
    assert_eq!(a.trades, b.trades);
    assert!((a.net_pnl - b.net_pnl).abs() < 1e-12);
    assert!(a.final_capital.is_finite());
    assert!(a.max_dd < 1.0);
    // El span del panel sigue siendo el de la cinta completa incluso si el
    // motor hubiera roto antes por ruina (líneas 613-615 + 637).
    let span_esperado_ms = ticks.last().unwrap().ts_ms - ticks[0].ts_ms;
    assert!(
        (a.metrics.span_days - span_esperado_ms as f64 / DIA_MS as f64).abs() < 1e-12
    );
}
