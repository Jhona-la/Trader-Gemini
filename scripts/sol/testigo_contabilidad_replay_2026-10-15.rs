//! SOL-CONT-2026-10-15 — testigo aritmético de los huecos abiertos
//! SOL-R5-02 (marca de posiciones abiertas) y SOL-R4-01 (span ejecutado).
//!
//! QUÉ: ejecuta el panel `ex_post_metrics` REAL (blob congelado de
//! `crates/backtest-engine/src/metrics.rs` en `main = 78647b12`, sin
//! modificar) sobre dos escenarios y cuantifica la distorsión que introduce
//! cada hueco. NO es el motor: es aritmética de expresiones, con las
//! fórmulas citadas por archivo y línea.
//!
//! Escenario A (SOL-R4-01): una ruina en la mitad de la cinta. El replay
//! rompe en `if cap <= 0.0 { break; }` (booktick_replay.rs:613-615), pero el
//! panel anualiza con `span = ticks.last().ts − ticks[0].ts` (línea 637) —
//! la cinta COMPLETA. Comparo el panel con span de cinta vs span ejecutado.
//!
//! Escenario B (SOL-R5-02): cierre con una posición ABIERTA sin marca.
//! `stats.final_capital = arena.unified_capital.load(...)` (línea 618) lee
//! CASH (únicos escritores de unified_capital en todo god-engine-core:
//! lib.rs:9359 `-fee_paid` y lib.rs:9381 `+fee_paid`), mientras
//! `daily_report` (continuous_evolution_backtest.rs:124-143) marca
//! `qty·(precio−entrada)·±1 − qty·precio·0.0005`. Comparo ambos capitales
//! terminales con el mismo libro.
//!
//! Uso: rustc scripts/sol/testigo_contabilidad_replay_2026-10-15.rs
//!      -o target/testigo_sol_cont && ./target/testigo_sol_cont

#[path = "../../crates/backtest-engine/src/metrics.rs"]
mod metrics;

use metrics::ex_post_metrics;

const DIA_MS: u64 = 86_400_000;

fn main() {
    println!("=== SOL-CONT-2026-10-15 · testigo sobre main 78647b12 ===");
    println!("metricas: crates/backtest-engine/src/metrics.rs (blob congelado, sin modificar)");

    // ── Escenario A: ruina a mitad de cinta (SOL-R4-01) ──────────────────
    // Muestra: 40 trades con edge leve (se ganan 55 de cada 100 arriesgados),
    // cinta de 10 días, ejecución real hasta el día 5 (ruina).
    let mut pnls = Vec::new();
    for i in 0..40 {
        pnls.push(if i % 20 < 11 { 0.55 } else { -0.45 });
    }
    let capital_inicial = 13.0;
    // Crecimiento ≈ meta del operador: ln(2)/3 por día ≈ 0.2310/día
    // ≈ 1.604e-4 por minuto → ×3.16 en 5 días (ruina en día 5).
    let capital_final_ruina = 13.0 * (1.604e-4_f64 * 5.0 * 1440.0).exp();
    let notional = 5_000.0;
    let max_dd = 0.18;

    let span_cinta_ms: u64 = 10 * DIA_MS;      // ticks.last().ts − ticks[0].ts (línea 637)
    let span_ejecutado_ms: u64 = 5 * DIA_MS;   // día real del break (línea 613)

    let panel_cinta = ex_post_metrics(&pnls, notional, capital_inicial, capital_final_ruina, max_dd, span_cinta_ms);
    let panel_ejecutado = ex_post_metrics(&pnls, notional, capital_inicial, capital_final_ruina, max_dd, span_ejecutado_ms);

    println!("\n── A · SOL-R4-01 — span de cinta completa vs span ejecutado ──");
    println!("  capital: 13.0 -> {:.6} (ruina en día 5 de cinta de 10 días)", capital_final_ruina);
    println!("  span reportado (línea 637) : {:.4} días", panel_cinta.span_days);
    println!("  span ejecutado (break 613) : {:.4} días", panel_ejecutado.span_days);
    println!("  CAGR  con span de cinta    : {:+.4} %", panel_cinta.cagr * 100.0);
    println!("  CAGR  con span ejecutado   : {:+.4} %", panel_ejecutado.cagr * 100.0);
    println!("  turnover/día cinta         : {:.4}", panel_cinta.turnover_per_day);
    println!("  turnover/día ejecutado     : {:.4}", panel_ejecutado.turnover_per_day);
    let distorsion_cagr = (panel_ejecutado.cagr - panel_cinta.cagr) / panel_cinta.cagr.abs();
    let distorsion_turnover = (panel_ejecutado.turnover_per_day - panel_cinta.turnover_per_day)
        / panel_cinta.turnover_per_day;
    println!(
        "  distorsión: CAGR {:+.1} % relativo, turnover {:+.1} % relativo",
        distorsion_cagr * 100.0,
        distorsion_turnover * 100.0
    );
    // El log-crecimiento anualizado escala EXACTAMENTE con el span:
    // overstated_factor = span_cinta / span_ejecutado = 2.0 en este fixture.
    let factor = span_cinta_ms as f64 / span_ejecutado_ms as f64;
    let log_cinta = panel_cinta.cagr.ln_1p();
    let log_ejec = panel_ejecutado.cagr.ln_1p();
    println!(
        "  factor exacto de sobre-declaración: log-creciente_ejecutado / log-creciente_cinta \
         = {:.6} (span {}/{} = {:.1})",
        log_ejec / log_cinta,
        span_cinta_ms,
        span_ejecutado_ms,
        factor
    );
    assert!((log_ejec / log_cinta - factor).abs() < 1e-9, "la anualización debe escalar con el span");

    // ── Escenario B: posición abierta al cierre (SOL-R5-02) ──────────────
    // Libro: cash 13.0 − fee de entrada 0.0026; 1 posición larga
    // qty 0.13 @ entrada 100.0, marca 110.0 al cierre de la cinta.
    let cash = 13.0 - 0.0026; // unified_capital tras el fee (lib.rs:9359)
    let qty = 0.13_f64;
    let entrada = 100.0_f64;
    let marca = 110.0_f64;
    // Fórmula EXACTA de daily_report (continuous_evolution_backtest.rs:128-139).
    let marca_flotante = (marca - entrada) * qty - qty * marca * 0.0005;
    let equity_marcada = cash + marca_flotante;

    let panel_cash = ex_post_metrics(&pnls, notional, capital_inicial, cash, max_dd, span_cinta_ms);
    let panel_marcado = ex_post_metrics(&pnls, notional, capital_inicial, equity_marcada, max_dd, span_cinta_ms);

    println!("\n── B · SOL-R5-02 — terminal en cash (booktick_replay:618) vs equity marcada ──");
    println!("  unified_capital (cash)          : {:.6}", cash);
    println!("  marca flotante (daily_report)   : {:+.6}", marca_flotante);
    println!("  equity marcada                  : {:.6}", equity_marcada);
    println!("  CAGR con cash (camino fitness)  : {:+.4} %", panel_cash.cagr * 100.0);
    println!("  CAGR con equity (camino reporte): {:+.4} %", panel_marcado.cagr * 100.0);
    println!(
        "  brecha de crecimiento declarada  : {:+.6} USD de capital terminal no contabilizados \
         por el camino que PROMUEVE genomas",
        equity_marcada - cash
    );
    println!(
        "  brecha relativa                 : {:+.2} % del capital terminal",
        (equity_marcada - cash) / cash * 100.0
    );
    println!("\n  Nota: el drawdown por tick (booktick_replay:605-612) usa el mismo cash:");
    println!("  una posición abierta en -30 % no mueve max_dd mientras el cash no caiga.");

    // Aserciones de aritmética (el testigo falla si las fórmulas se mueven).
    assert!((panel_ejecutado.cagr > panel_cinta.cagr), "ruina más tardía no puede declarar más CAGR");
    assert!((panel_ejecutado.turnover_per_day > panel_cinta.turnover_per_day));
    assert!(equity_marcada > cash, "la marca positiva debe superar el cash");
    assert!((panel_cash.cagr < panel_marcado.cagr), "cash subestima crecimiento con marca positiva");

    println!("\n=== testigo OK (exit 0) — expresiones verificadas, motor NO ejecutado ===");
}
