//! BACKTEST MULTI-VENTANA (F7) — estadísticas por ventana temporal con el
//! motor HONESTO (booktick_replay sobre ticks reales).
//!
//! USO: cargo run --release --bin backtest_windows [-- --file data/BTCUSDT_ticks_REAL.bin]
//!
//! Reporta: ventana completa + sub-ventanas 3d/7d/15d (según data disponible),
//! con PnL cerrado bruto/neto, WR, trades, sharpe, drawdown y retorno terminal
//! de equity. Los equivalentes geométricos describen los extremos y la duración
//! suministrada de la cinta; no son un pronóstico ni una garantía de ganancias.

use backtest_engine::booktick_replay::{run_booktick_replay, ReplayConfig, ReplayTick};
use quantum_arena::genome::SuperGenotype;

// The caller supplies its existing duration; this does not validate tape coverage.

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct TerminalEquityMetrics {
    // Fraction of initial equity, not percent. May be below -1 for debt.
    equity_return: Option<f64>,
    // Log growth per observed day. None for zero/negative terminal equity,
    // invalid duration/capital, or a nonrepresentable derived rate.
    daily_log_growth: Option<f64>,
}

fn finite_equity_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn terminal_equity_metrics(
    initial: f64,
    final_capital: f64,
    elapsed_days: f64,
) -> TerminalEquityMetrics {
    if !initial.is_finite() || initial <= 0.0 || !final_capital.is_finite() {
        return TerminalEquityMetrics::default();
    }

    // Algebraically CT/C0 - 1. Difference first preserves small changes.
    // With finite opposite-sign extremes CT-C0 can overflow even though the
    // quotient, and therefore the simple return, remain representable.
    let change = final_capital - initial;
    let relative = if change.is_finite() {
        change / initial
    } else {
        final_capital / initial - 1.0
    };
    let equity_return = finite_equity_value(relative);

    let daily_log_growth = if final_capital > 0.0 && elapsed_days.is_finite() && elapsed_days > 0.0
    {
        let log_growth = if relative.is_finite() && relative.abs() <= 0.5 {
            relative.ln_1p()
        } else {
            // Do not take ln(CT/C0): that quotient may overflow or underflow
            // while the difference of these finite logarithms is valid.
            final_capital.ln() - initial.ln()
        };
        let rate = log_growth / elapsed_days;
        if !rate.is_finite() || (log_growth != 0.0 && rate == 0.0) {
            None // Do not present underflow of a nonzero rate as measured zero.
        } else {
            Some(rate)
        }
    } else {
        None
    };

    TerminalEquityMetrics {
        equity_return,
        daily_log_growth,
    }
}

impl TerminalEquityMetrics {
    // A geometric equivalent under this constant log rate, not a forecast,
    // nor proof that a window of this duration was actually observed.
    fn equivalent_return(self, horizon_days: f64) -> Option<f64> {
        if !horizon_days.is_finite() || horizon_days <= 0.0 {
            return None;
        }
        let rate = self.daily_log_growth?;
        let exponent = rate * horizon_days;
        if !exponent.is_finite() || (rate != 0.0 && exponent == 0.0) {
            return None;
        }
        // Large negative finite exponents may round this return to -1.
        // That does not establish that the observed terminal capital is zero.
        finite_equity_value(exponent.exp_m1())
    }

    // Capital under the same constant log rate. This is a geometric equivalent,
    // not a forecast. Do not reconstruct it as initial * (1 + equivalent_return):
    // exp_m1 can round a finite negative exponent to -1 while capital is positive.
    fn equivalent_capital(self, initial: f64, horizon_days: f64) -> Option<f64> {
        if !initial.is_finite()
            || initial <= 0.0
            || !horizon_days.is_finite()
            || horizon_days <= 0.0
        {
            return None;
        }
        let rate = self.daily_log_growth?;
        let exponent = rate * horizon_days;
        if !rate.is_finite() || !exponent.is_finite() || (rate != 0.0 && exponent == 0.0) {
            return None;
        }
        if exponent == 0.0 {
            return Some(initial);
        }

        // Preserve the initial scale and near-zero precision when the factor
        // itself is normal. A subnormal factor may have poor relative precision
        // even when the rescaled capital is normal, so use the log path there.
        let factor = exponent.exp();
        if factor.is_normal() {
            let capital = initial * factor;
            // With a normal factor, a nonrepresentable product stays missing.
            // A log fallback could round away growth and fabricate a finite result.
            return (capital.is_finite() && capital > 0.0).then_some(capital);
        }

        // The factor alone may overflow/underflow although the capital does not.
        // Finite positive inputs have finite logarithms; a rounded zero result
        // is unrepresentable positive capital, never evidence of actual ruin.
        let capital = (initial.ln() + exponent).exp();
        (capital.is_finite() && capital > 0.0).then_some(capital)
    }
}

// Signed percent display; use e.g. format!("{:.3}", EquityPct(value)).
// Missing/unrepresentable values are explicit, including overflow in *100.
// The surrounding report should distinguish the actual capital/duration state.
struct EquityPct(Option<f64>);

impl std::fmt::Display for EquityPct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = f.precision().unwrap_or(2);
        match self.0.and_then(|value| finite_equity_value(value * 100.0)) {
            Some(percent) => write!(f, "{:+.*}%", precision, percent),
            None => f.write_str("N/D"),
        }
    }
}

#[tokio::main]
async fn main() {
    // El motor necesita stack grande (producción usa 32MB) — el hilo main
    // de Rust solo tiene 2MB y GodEngineCore::process_event con su anidamiento
    // provoca stack overflow en backtests largos.
    let child = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run_backtest)
        .expect("spawn worker");
    match child.join() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("❌ worker panic: {:?}", e);
            std::process::exit(1);
        }
    }
}

fn run_backtest() {
    let args: Vec<String> = std::env::args().collect();
    run_backtest_from_args(&args);
}

fn run_backtest_from_args(args: &[String]) {
    let file = args
        .iter()
        .position(|a| a == "--file")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "data/BTCUSDT_ticks_REAL.bin".to_string());
    let max_ticks: usize = args
        .iter()
        .position(|a| a == "--max-ticks")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000_000); // 2M default: suficiente densidad sin stack overflow

    println!("══════════════════════════════════════════════════════════════════");
    println!("📊 BACKTEST MULTI-VENTANA — MOTOR HONESTO (ticks reales)");
    println!("   Fuente: {} ", file);
    println!("   Motor: GodEngineCore::process_event (mismo camino que producción)");
    println!("   Slippage: ATR real · Fees: del genoma (pisos realistas)");
    println!("══════════════════════════════════════════════════════════════════");

    // Cargar ticks
    let mmap = match (|| -> Result<memmap2::Mmap, String> {
        let f = std::fs::File::open(&file).map_err(|e| e.to_string())?;
        unsafe {
            memmap2::MmapOptions::new()
                .map(&f)
                .map_err(|e| e.to_string())
        }
    })() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ No se pudo abrir {}: {}", file, e);
            std::process::exit(1);
        }
    };
    #[repr(C)]
    struct BinTick {
        ts: u64,
        bid: f64,
        ask: f64,
        bq: f64,
        aq: f64,
    }
    let sz = std::mem::size_of::<BinTick>();
    // Los archivos con magic "TGMTICK1" tienen un header de 8 bytes antes
    // de los ticks estándar. Detectar y saltar.
    let header_off = if mmap.len() >= 8 && &mmap[..8] == b"TGMTICK1" {
        println!("   📎 Formato TGMTICK1 detectado (header 8 bytes saltado)");
        8usize
    } else {
        0
    };
    let n_total = (mmap.len() - header_off) / sz;
    if n_total < 10_000 {
        eprintln!("❌ Datos insuficientes: {} ticks", n_total);
        std::process::exit(1);
    }
    // Limitar a los últimos max_ticks (densidad preservada del final)
    let start_idx = n_total.saturating_sub(max_ticks);
    let n_used = n_total - start_idx;
    let ptr = unsafe { mmap.as_ptr().add(header_off + start_idx * sz) } as *const BinTick;
    let raw = unsafe { std::slice::from_raw_parts(ptr, n_used) };

    // Convertir a ReplayTick (filtrando inválidos)
    let ticks: Vec<ReplayTick> = raw
        .iter()
        .filter(|t| t.bid > 0.0 && t.ask > 0.0 && t.bid <= t.ask && t.ts > 0)
        .map(|t| ReplayTick {
            ts_ms: t.ts,
            bid: t.bid,
            ask: t.ask,
            bid_qty: t.bq,
            ask_qty: t.aq,
        })
        .collect();

    let initial_capital = 1000.0; // USDT estándar para comparabilidad
    let trade_only = args.iter().any(|a| a == "--trade-only");
    let cfg = ReplayConfig {
        initial_capital,
        warmup_ticks: 500,
        trade_only,
        shift_atr_frac: 0.10, // histórico (DIV-1: A/B con 0.0 documentado)
    };
    // This count is after both the input limit and the invalid-row filter.
    // Do not invent a data span/B&H or load dependencies for omitted replay.
    if report_replay_omitted("COMPLETA", ticks.len(), &cfg) {
        return;
    }

    if trade_only {
        println!("   🔀 MODO TRADE-ONLY (sin bid/ask sintético)");
    }

    // B3.15 — MODELOS PROMOVIDOS al replay: sin esto el backtest mide el
    // motor SIN la parte ML (los forests viven en GLOBAL_FORESTS, que en
    // producción carga god_engine al arranque). El cargador valida el
    // contrato ML_VECTOR_DIM: modelos incompatibles se rechazan ruidoso.
    // Se carga SIEMPRE el .json cuando existe (fuente de verdad); el .bin
    // es caché que el propio loader regenera.
    if let Ok(entries) = std::fs::read_dir("models") {
        let mut loaded = 0usize;
        let mut rejected = 0usize;
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            match god_engine_core::ml_inference::NanoForest::load_global(
                stem,
                &path.to_string_lossy(),
            ) {
                Ok(()) => loaded += 1,
                Err(e) => {
                    rejected += 1;
                    println!("   ⚠️ modelo {stem} rechazado: {e}");
                }
            }
        }
        println!(
            "   🧠 Modelos: {loaded} cargados, {rejected} rechazados (contrato ML_VECTOR_DIM)"
        );
    }

    let first_ts = ticks.first().map(|t| t.ts_ms).unwrap_or(0);
    let last_ts = ticks.last().map(|t| t.ts_ms).unwrap_or(0);
    let total_days = (last_ts.saturating_sub(first_ts)) as f64 / 86_400_000.0;
    let p_start = ticks.first().map(|t| t.mid()).unwrap_or(0.0);
    let p_end = ticks.last().map(|t| t.mid()).unwrap_or(0.0);
    let bh_ret = if p_start > 0.0 {
        (p_end / p_start - 1.0) * 100.0
    } else {
        0.0
    };

    println!(
        "\n🗂️  Data: {} ticks válidos (últimos {} de {} crudos)",
        ticks.len(),
        n_used,
        n_total
    );
    println!("   Período: {:.1} días", total_days);
    println!(
        "   BTC: ${:.0} → ${:.0} (buy&hold: {:+.1}%)",
        p_start, p_end, bh_ret
    );

    // Genoma: el activo del almacén o baseline
    let genome = quantum_arena::genome_store::GenomeEnvelope::load_active()
        .map(|e| e.genome)
        .unwrap_or_else(|| SuperGenotype::new_baseline(0.0002, 0.0005));

    // CRÍTICO: el RiskEngine RECHAZA toda orden sin spec del símbolo
    // (try_spec=None → rej(3)). Y B3.25 (disciplina de roster) exige que
    // el símbolo del replay TENGA un modelo en models/ — si el archivo
    // es BNBUSDT pero registramos BTCUSDT, el gate veta TODO.
    // Derivar el símbolo del nombre del archivo de datos.
    let replay_symbol = file
        .split('/')
        .next_back()
        .unwrap_or("BTCUSDT")
        .split('_')
        .next()
        .unwrap_or("BTCUSDT")
        .to_string();
    println!("   📋 Símbolo del replay: {replay_symbol} (derivado de {file})");
    quantum_arena::symbol_registry::update_registry(vec![
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: replay_symbol.clone(),
            step_size: 0.001,
            tick_size: 0.01,
            min_qty: 0.001,
            min_notional: 5.0,
            max_leverage: 20,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        },
    ]);
    println!("   📋 Symbol specs registradas (RiskEngine puede evaluar)");

    // Cargar omni FRED (si hay red)
    println!("\n🌐 Cargando macro FRED...");
    let omni = backtest_engine::booktick_replay::OmniHistory::fetch();
    match &omni {
        Some(_) => println!("   ✅ 6 series cargadas"),
        None => println!("   ⚠️ Sin red — omni neutro"),
    }

    // ===== VENTANA COMPLETA =====
    let sep = "═".repeat(60);
    let sep2 = "─".repeat(60);
    println!("\n{}", sep);
    println!("📈 VENTANA COMPLETA ({:.1} días)", total_days);
    println!("{}", sep);
    let Some(full) = report_replay_window("COMPLETA", ticks.len(), &cfg, total_days, || {
        run_booktick_replay(&ticks, &genome, omni.as_ref(), &cfg)
    }) else {
        return;
    };

    // ===== SUB-VENTANAS (desde el final hacia atrás) =====
    let ms_per_day: u64 = 86_400_000;
    for &days in &[3u64, 7u64, 15u64] {
        if (days as f64) > total_days {
            continue;
        }
        let cutoff = last_ts.saturating_sub(days * ms_per_day);
        let start_idx = ticks.partition_point(|t| t.ts_ms < cutoff);
        if start_idx >= ticks.len() || ticks.len() - start_idx < 1000 {
            continue;
        }
        let window = &ticks[start_idx..];
        let wdays =
            (window.last().unwrap().ts_ms - window.first().unwrap().ts_ms) as f64 / 86_400_000.0;
        println!("\n{}", sep2);
        println!("📅 ÚLTIMOS {} DÍAS ({:.1} días reales)", days, wdays);
        println!("{}", sep2);
        let _ = report_replay_window(&format!("{}D", days), window.len(), &cfg, wdays, || {
            run_booktick_replay(window, &genome, omni.as_ref(), &cfg)
        });
    }

    // ===== EQUIVALENTES GEOMÉTRICOS DE EQUITY =====
    println!("\n{}", sep);
    println!("🎯 EQUIVALENTES GEOMÉTRICOS (equity con MTM estimado — no pronóstico)");
    println!("{}", sep);
    print_full_projection(&full, initial_capital, total_days);
    println!("\n══════════════════════════════════════════════════════════════════");
}

// Shared omission policy for the early input barrier and every window.
// row_count is the count after the caller's max-ticks limit and filter.
// A known starting ledger is not evidence of observed flat performance.
fn report_replay_omitted(label: &str, row_count: usize, cfg: &ReplayConfig) -> bool {
    if !cfg.initial_capital.is_finite() || cfg.initial_capital <= 0.0 {
        println!("   Replay omitido ({label}): capital inicial inválido (N/D); no se calculó rendimiento.");
        return true;
    }
    if row_count.saturating_sub(cfg.warmup_ticks) <= 10 {
        println!(
            "   Replay omitido ({label}): datos insuficientes tras límite/filtro ({row_count} filas, warmup={}); se requieren más de 10 filas posteriores al warmup.",
            cfg.warmup_ticks
        );
        return true;
    }
    false
}

// None stops the full-window projection. A zero terminal equity returned
// by an invoked replay remains a total loss, not a missing value.
fn report_replay_window(
    label: &str,
    row_count: usize,
    cfg: &ReplayConfig,
    days: f64,
    replay: impl FnOnce() -> backtest_engine::booktick_replay::ReplayStats,
) -> Option<backtest_engine::booktick_replay::ReplayStats> {
    if report_replay_omitted(label, row_count, cfg) {
        return None;
    }
    let stats = replay();
    print_window(label, &stats, cfg.initial_capital, days);
    Some(stats)
}

fn capital_text(value: f64) -> String {
    match finite_equity_value(value) {
        Some(value) if value != 0.0 && (value.abs() < 0.005 || value.abs() >= 1e12) => {
            format!("${value:.6e}")
        }
        Some(value) => format!("${value:.2}"),
        None => "N/D".to_string(),
    }
}

fn daily_equivalent_text(equity: TerminalEquityMetrics) -> String {
    let percent = format!("{:.3}", EquityPct(equity.equivalent_return(1.0)));
    if percent == "N/D" {
        percent
    } else {
        format!("{percent}/d")
    }
}

fn print_full_projection(
    full: &backtest_engine::booktick_replay::ReplayStats,
    cap: f64,
    days: f64,
) {
    let equity = terminal_equity_metrics(cap, full.final_capital, days);
    println!(
        "   Equivalente geométrico diario: {}",
        daily_equivalent_text(equity)
    );
    match equity.daily_log_growth {
        Some(growth) => println!("   Crecimiento logarítmico diario: {:+.6e}/d", growth),
        None => println!(
            "   Crecimiento logarítmico diario: N/D (requiere equity positiva, duración válida y cálculo representable)"
        ),
    }
    println!(
        "   Extremos de equity y duración suministrada; mantener esta tasa es un supuesto, no un pronóstico."
    );
    for &d in &[3u64, 7u64, 15u64, 30u64, 365u64] {
        let horizon = d as f64;
        let equivalent_return = EquityPct(equity.equivalent_return(horizon));
        match equity.equivalent_capital(cap, horizon) {
            Some(capital) => println!(
                "   {} días: {:.1} (capital {} → {})",
                d,
                equivalent_return,
                capital_text(cap),
                capital_text(capital)
            ),
            None => println!(
                "   {} días: {:.1} (capital equivalente N/D)",
                d, equivalent_return
            ),
        }
    }
    // La meta algebraica tampoco demuestra que exista un periodo observado de
    // tres días, ni una probabilidad de duplicación, ni ganancias futuras.
    let needed_log_daily = 2.0f64.ln() / 3.0;
    println!(
        "\n   ⚖️  META +100%/3d: equivalente diario {:.3}/d · crecimiento logarítmico {:+.6e}/d",
        EquityPct(Some(needed_log_daily.exp_m1())),
        needed_log_daily
    );
}
fn print_window(
    _label: &str,
    s: &backtest_engine::booktick_replay::ReplayStats,
    cap: f64,
    days: f64,
) {
    let equity = terminal_equity_metrics(cap, s.final_capital, days);
    println!(
        "   Capital: {} → {} ({:.2})",
        capital_text(cap),
        capital_text(s.final_capital),
        EquityPct(equity.equity_return)
    );
    println!(
        "   PnL cerrado bruto: ${:+.4} · fees est: ${:.4} · neto: ${:+.4}",
        s.gross_pnl, s.fees_est, s.net_pnl
    );
    // roi_net conserva el significado de cierres; no incluye el MTM abierto.
    let closed_roi = if cap.is_finite() && cap > 0.0 {
        finite_equity_value(s.roi_net(cap))
    } else {
        None
    };
    println!("   ROI neto de cierres: {:.2}", EquityPct(closed_roi));
    println!(
        "   WR neto: {:.1}% · WR bruto: {:.1}% · trades: {}",
        s.wr_net() * 100.0,
        if s.trades > 0 {
            s.wins_gross as f64 / s.trades as f64 * 100.0
        } else {
            0.0
        },
        s.trades
    );
    println!(
        "   Sharpe/trade (cierres): {:.4} · MaxDD (caja): {:.1}%",
        s.sharpe,
        s.max_dd * 100.0
    );
    println!(
        "   Equivalente geométrico diario: {} · equivalente 3d: {:.1}",
        daily_equivalent_text(equity),
        EquityPct(equity.equivalent_return(3.0))
    );
    if s.omni_neutral {
        println!("   ⚠️ Omni neutro (sin FRED)");
    }
}
#[cfg(test)]
mod si_b15_report_contract {
    use super::*;
    use backtest_engine::booktick_replay::ReplayStats;
    use std::process::Command;

    const CHILD_CASE: &str = "TRADER_GEMINI_B15_REPORT_CHILD_CASE";

    // Exercise print_window itself: an unused replacement formula must not
    // make this contract green. The child runs only the named Rust test,
    // never main, tape loading, training or promotion.
    fn captured_report(test_name: &str, case: &str) -> String {
        let output = Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", test_name, "--nocapture"])
            .env(CHILD_CASE, case)
            .output()
            .expect("run report fixture");
        assert!(
            output.status.success(),
            "fixture failed: {:?}",
            output.status
        );
        String::from_utf8(output.stdout).expect("report must be UTF-8")
    }

    #[test]
    fn capital_and_rates_follow_equity_despite_positive_closed_pnl() {
        let case = "closed_profit_open_loss";
        if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
            let stats = ReplayStats {
                final_capital: 64.0,
                net_pnl: 32.0,
                trades: 1,
                ..Default::default()
            };
            assert_eq!(stats.roi_net(128.0), 0.25, "closed PnL keeps its meaning");
            print_window("fixture", &stats, 128.0, 3.0);
            return;
        }
        let report = captured_report(
            "si_b15_report_contract::capital_and_rates_follow_equity_despite_positive_closed_pnl",
            case,
        );
        let capital = report
            .lines()
            .find(|line| line.contains("Capital:"))
            .expect("capital line");
        assert!(capital.contains("$128.00 → $64.00 (-50.00%)"), "{report}");
        let rate = report
            .lines()
            .find(|line| line.contains("equivalente 3d:"))
            .expect("rate line");
        assert!(rate.contains("-20.630%/d"), "{report}");
        assert!(rate.contains("equivalente 3d: -50.0%"), "{report}");
    }

    #[test]
    fn capital_return_does_not_require_closed_trades() {
        let case = "open_equity_without_closes";
        if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
            let stats = ReplayStats {
                final_capital: 256.0,
                net_pnl: 0.0,
                trades: 0,
                ..Default::default()
            };
            assert_eq!(stats.roi_net(128.0), 0.0, "no realized PnL is invented");
            print_window("fixture", &stats, 128.0, 3.0);
            return;
        }
        let report = captured_report(
            "si_b15_report_contract::capital_return_does_not_require_closed_trades",
            case,
        );
        let capital = report
            .lines()
            .find(|line| line.contains("Capital:"))
            .expect("capital line");
        assert!(capital.contains("$128.00 → $256.00 (+100.00%)"), "{report}");
        let rate = report
            .lines()
            .find(|line| line.contains("equivalente 3d:"))
            .expect("rate line");
        assert!(rate.contains("+25.992%/d"), "{report}");
        assert!(rate.contains("equivalente 3d: +100.0%"), "{report}");
    }

    #[test]
    fn full_projection_uses_equity_without_closed_trades() {
        let case = "full_equity_without_closes";
        if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
            let stats = ReplayStats {
                final_capital: 256.0,
                trades: 0,
                net_pnl: 0.0,
                ..Default::default()
            };
            print_full_projection(&stats, 128.0, 3.0);
            return;
        }
        let report = captured_report(
            "si_b15_report_contract::full_projection_uses_equity_without_closed_trades",
            case,
        );
        let daily = report
            .lines()
            .find(|line| line.contains("Equivalente geométrico diario:"))
            .expect("daily equivalent");
        assert!(daily.contains("+25.992%/d"), "{report}");
        let three_days = report
            .lines()
            .find(|line| line.contains("3 días:"))
            .expect("three-day equivalent");
        assert!(
            three_days.contains("+100.0% (capital $128.00 → $256.00)"),
            "{report}"
        );
        assert!(report.contains("no un pronóstico"), "{report}");
    }

    #[test]
    fn nonpositive_terminal_equity_keeps_loss_and_missing_geometric_equivalents() {
        let case = "nonpositive_equity_domain";
        if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
            for end in [0.0, -128.0] {
                let stats = ReplayStats {
                    final_capital: end,
                    ..Default::default()
                };
                print_window("fixture", &stats, 128.0, 3.0);
                print_full_projection(&stats, 128.0, 3.0);
            }
            return;
        }
        let report = captured_report(
            "si_b15_report_contract::nonpositive_terminal_equity_keeps_loss_and_missing_geometric_equivalents",
            case,
        );
        assert!(
            report.contains("Capital: $128.00 → $0.00 (-100.00%)"),
            "{report}"
        );
        assert!(
            report.contains("Capital: $128.00 → $-128.00 (-200.00%)"),
            "{report}"
        );
        let rates: Vec<_> = report
            .lines()
            .filter(|line| line.contains("equivalente 3d:"))
            .collect();
        assert_eq!(rates.len(), 2, "{report}");
        assert!(
            rates
                .iter()
                .all(|line| line.contains("diario: N/D · equivalente 3d: N/D")),
            "{report}"
        );
        let projections: Vec<_> = report
            .lines()
            .filter(|line| line.contains("3 días:"))
            .collect();
        assert_eq!(projections.len(), 2, "{report}");
        assert!(
            projections
                .iter()
                .all(|line| line.contains("N/D (capital equivalente N/D)")),
            "{report}"
        );
    }

    #[test]
    fn invalid_duration_keeps_observed_equity_without_fabricating_rates() {
        let case = "invalid_duration_domain";
        if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
            for days in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let stats = ReplayStats {
                    final_capital: 256.0,
                    ..Default::default()
                };
                print_window("fixture", &stats, 128.0, days);
                print_full_projection(&stats, 128.0, days);
            }
            return;
        }
        let report = captured_report(
            "si_b15_report_contract::invalid_duration_keeps_observed_equity_without_fabricating_rates",
            case,
        );
        let capitals: Vec<_> = report
            .lines()
            .filter(|line| line.contains("Capital:"))
            .collect();
        assert_eq!(capitals.len(), 5, "{report}");
        assert!(
            capitals
                .iter()
                .all(|line| line.contains("$128.00 → $256.00 (+100.00%)")),
            "{report}"
        );
        let rates: Vec<_> = report
            .lines()
            .filter(|line| line.contains("equivalente 3d:"))
            .collect();
        assert_eq!(rates.len(), 5, "{report}");
        assert!(
            rates
                .iter()
                .all(|line| line.contains("diario: N/D · equivalente 3d: N/D")),
            "{report}"
        );
        let logs: Vec<_> = report
            .lines()
            .filter(|line| line.contains("Crecimiento logarítmico diario:"))
            .collect();
        assert_eq!(logs.len(), 5, "{report}");
        assert!(logs.iter().all(|line| line.contains("N/D")), "{report}");
    }

    #[cfg(test)]
    mod terminal_equity_domain_contract {
        use super::*;

        fn near(actual: Option<f64>, expected: f64, tolerance: f64) {
            let actual = actual.expect("a representable measurement is required");
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} != {expected}"
            );
        }

        #[test]
        fn powers_of_two_fix_the_equity_and_geometric_oracles() {
            let doubled = terminal_equity_metrics(128.0, 256.0, 3.0);
            assert_eq!(doubled.equity_return, Some(1.0));
            near(doubled.daily_log_growth, 0.23104906018664842, 1e-15);
            near(doubled.equivalent_return(1.0), 0.2599210498948732, 1e-14);
            near(doubled.equivalent_return(3.0), 1.0, 1e-14);

            let halved = terminal_equity_metrics(128.0, 64.0, 3.0);
            assert_eq!(halved.equity_return, Some(-0.5));
            near(halved.equivalent_return(1.0), -0.2062994740159002, 1e-14);
            near(halved.equivalent_return(3.0), -0.5, 1e-14);

            let unchanged = terminal_equity_metrics(128.0, 128.0, 3.0);
            assert_eq!(unchanged.equity_return, Some(0.0));
            assert_eq!(unchanged.daily_log_growth, Some(0.0));
            assert_eq!(unchanged.equivalent_return(3.0), Some(0.0));
        }

        #[test]
        fn nonpositive_equity_preserves_loss_without_fabricating_a_log_rate() {
            for (end, expected) in [(0.0, -1.0), (-128.0, -2.0)] {
                let metrics = terminal_equity_metrics(128.0, end, 3.0);
                assert_eq!(metrics.equity_return, Some(expected));
                assert_eq!(metrics.daily_log_growth, None);
                assert_eq!(metrics.equivalent_return(3.0), None);
            }
            // CT-C0 overflows, but the return is exactly a 200% loss.
            assert_eq!(
                terminal_equity_metrics(f64::MAX, -f64::MAX, 3.0).equity_return,
                Some(-2.0)
            );
        }

        #[test]
        fn invalid_capital_and_duration_have_separate_domains() {
            for initial in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(
                    terminal_equity_metrics(initial, 256.0, 3.0),
                    TerminalEquityMetrics::default()
                );
            }
            for end in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(
                    terminal_equity_metrics(128.0, end, 3.0),
                    TerminalEquityMetrics::default()
                );
            }
            for days in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let metrics = terminal_equity_metrics(128.0, 256.0, days);
                assert_eq!(metrics.equity_return, Some(1.0));
                assert_eq!(metrics.daily_log_growth, None);
            }
        }

        #[test]
        fn quotient_extremes_do_not_erase_finite_log_information() {
            let up = terminal_equity_metrics(1e-300, 1e300, 3.0);
            assert_eq!(up.equity_return, None);
            near(up.daily_log_growth, 460.51701859880916, 1e-12);
            assert_eq!(up.equivalent_return(3.0), None);
            let down = terminal_equity_metrics(1e300, 1e-300, 3.0);
            assert_eq!(down.equity_return, Some(-1.0));
            near(down.daily_log_growth, -460.51701859880916, 1e-12);
            assert_eq!(down.equivalent_return(3.0), Some(-1.0));
        }

        #[test]
        fn a_small_observed_change_is_not_rounded_away_before_logging() {
            let initial = 1.5f64;
            let end = f64::from_bits(initial.to_bits() + 1);
            let metrics = terminal_equity_metrics(initial, end, 1.0);
            near(metrics.equity_return, 1.4802973661668753e-16, 1e-30);
            near(metrics.daily_log_growth, 1.4802973661668753e-16, 1e-30);
            assert_eq!(
                terminal_equity_metrics(initial, end, f64::MAX).daily_log_growth,
                None
            );
            assert_eq!(
                terminal_equity_metrics(128.0, 256.0, f64::from_bits(1)).daily_log_growth,
                None
            );
        }

        #[test]
        fn percentages_do_not_print_overflow_as_infinite_performance() {
            assert_eq!(
                format!("{:.3}", EquityPct(Some(0.2599210498948732))),
                "+25.992%"
            );
            assert_eq!(format!("{}", EquityPct(Some(-0.5))), "-50.00%");
            for value in [None, Some(f64::MAX), Some(f64::NAN), Some(f64::INFINITY)] {
                assert_eq!(format!("{}", EquityPct(value)), "N/D");
            }
        }

        fn relatively_near(actual: Option<f64>, expected: f64) {
            let actual = actual.expect("positive finite equivalent capital is required");
            assert!(actual.is_finite() && actual > 0.0);
            assert!(
                (actual / expected - 1.0).abs() <= 1e-12,
                "{actual} != {expected}"
            );
        }

        #[test]
        fn equivalent_capital_survives_a_return_rounded_to_total_loss() {
            let metrics = TerminalEquityMetrics {
                daily_log_growth: Some(-10.0),
                ..Default::default()
            };
            assert_eq!(metrics.equivalent_return(5.0), Some(-1.0));
            relatively_near(metrics.equivalent_capital(1e100, 5.0), 1.928749847963918e78);
        }

        #[test]
        fn equivalent_capital_rescales_unrepresentable_exponential_factors() {
            let growth = TerminalEquityMetrics {
                daily_log_growth: Some(1000.0),
                ..Default::default()
            };
            let decay = TerminalEquityMetrics {
                daily_log_growth: Some(-1000.0),
                ..Default::default()
            };
            // Independent numerical oracles: these are the squared, representable
            // half-exponent products (exp(500)*1e-150) and (exp(-500)*1e150).
            relatively_near(
                growth.equivalent_capital(1e-300, 1.0),
                1.970071114017047e134,
            );
            relatively_near(decay.equivalent_capital(1e300, 1.0), 5.075958897549456e-135);
            assert_eq!(growth.equivalent_capital(1.0, 1.0), None);
            assert_eq!(decay.equivalent_capital(1.0, 1.0), None);
        }

        #[test]
        fn equivalent_capital_does_not_rescue_overflow_of_a_normal_factor_product() {
            let initial = f64::from_bits(f64::MAX.to_bits() - 2);
            let metrics = terminal_equity_metrics(initial, f64::MAX, 3.0);
            let rate = metrics.daily_log_growth.expect("positive finite growth");
            assert!(rate > 0.0);
            let factor = (rate * 7.0).exp();
            assert!(factor.is_normal(), "fixture requires a normal factor");
            assert!(
                !(initial * factor).is_finite(),
                "fixture requires capital overflow, not factor overflow"
            );
            assert!(
                metrics
                    .equivalent_return(7.0)
                    .expect("finite equivalent return")
                    > 0.0
            );
            assert_eq!(
                metrics.equivalent_capital(initial, 7.0),
                None,
                "a log fallback must not fabricate finite capital after product overflow"
            );
        }

        #[test]
        fn equivalent_capital_validates_its_domain_and_preserves_zero_growth() {
            let metrics = terminal_equity_metrics(128.0, 256.0, 3.0);
            relatively_near(metrics.equivalent_capital(128.0, 3.0), 256.0);
            for initial in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(metrics.equivalent_capital(initial, 3.0), None);
            }
            for horizon in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(metrics.equivalent_capital(128.0, horizon), None);
            }
            for rate in [
                None,
                Some(f64::NAN),
                Some(f64::INFINITY),
                Some(f64::NEG_INFINITY),
            ] {
                let invalid = TerminalEquityMetrics {
                    daily_log_growth: rate,
                    ..Default::default()
                };
                assert_eq!(invalid.equivalent_capital(128.0, 3.0), None);
            }
            let zero = terminal_equity_metrics(128.0, 128.0, 3.0);
            for initial in [f64::from_bits(1), 128.0, f64::MAX] {
                assert_eq!(zero.equivalent_capital(initial, 3.0), Some(initial));
            }
            let tiny = TerminalEquityMetrics {
                daily_log_growth: Some(f64::from_bits(1)),
                ..Default::default()
            };
            assert_eq!(tiny.equivalent_capital(128.0, 0.5), None);
            let huge = TerminalEquityMetrics {
                daily_log_growth: Some(f64::MAX),
                ..Default::default()
            };
            assert_eq!(huge.equivalent_capital(128.0, 2.0), None);
        }
    }
    mod initial_ledger {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        fn assert_no_performance_lines(report: &str) {
            for forbidden in [
                "   Capital:",
                "   PnL cerrado bruto:",
                "   ROI neto de cierres:",
                "   WR neto:",
                "   Sharpe/trade",
                "   Equivalente geométrico diario:",
                "   Crecimiento logarítmico diario:",
                "equivalente 3d:",
            ] {
                assert!(
                    !report.contains(forbidden),
                    "omitted replay printed {forbidden}: {report}"
                );
            }
        }

        #[test]
        fn post_limit_and_filter_short_window_omits_replay_and_performance() {
            let case = "mx06-short-window";
            if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
                let cfg = ReplayConfig {
                    initial_capital: 1_000.0,
                    warmup_ticks: 500,
                    ..ReplayConfig::default()
                };
                // Values are counts AFTER max-ticks and the input filter. Even
                // if the raw file exceeded 10_000 rows, 500 cannot pass W=500.
                for rows in [0, 500, 510] {
                    let calls = Arc::new(AtomicUsize::new(0));
                    let retained = Arc::clone(&calls);
                    let result = report_replay_window("COMPLETA", rows, &cfg, 3.0, move || {
                        calls.fetch_add(1, Ordering::Relaxed);
                        panic!("a short window must never invoke replay");
                    });
                    assert!(result.is_none());
                    assert_eq!(retained.load(Ordering::Relaxed), 0);
                    // This mirrors the caller's explicit full-window handling:
                    // no returned stats means no projection is evaluated.
                    if let Some(stats) = result {
                        print_full_projection(&stats, cfg.initial_capital, 3.0);
                    }
                }
                return;
            }
            let report = captured_report(
            "si_b15_report_contract::initial_ledger::post_limit_and_filter_short_window_omits_replay_and_performance",
            case,
        );
            assert_eq!(
                report
                    .matches("Replay omitido (COMPLETA): datos insuficientes tras límite/filtro")
                    .count(),
                3
            );
            for rows in [0, 500, 510] {
                assert!(report.contains(&format!("{rows} filas, warmup=500")));
            }
            assert_no_performance_lines(&report);
        }

        #[test]
        fn invalid_initial_capital_omits_replay_without_fabricated_zero() {
            let case = "mx06-invalid-initial";
            if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
                for initial_capital in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
                    let cfg = ReplayConfig {
                        initial_capital,
                        warmup_ticks: 64,
                        ..ReplayConfig::default()
                    };
                    assert!(80usize.saturating_sub(cfg.warmup_ticks) > 10);
                    let result = report_replay_window("COMPLETA", 80, &cfg, 3.0, || {
                        panic!("invalid initial capital must never invoke replay");
                    });
                    assert!(result.is_none());
                }
                return;
            }
            let report = captured_report(
            "si_b15_report_contract::initial_ledger::invalid_initial_capital_omits_replay_without_fabricated_zero",
            case,
        );
            assert_eq!(
                report
                    .matches("Replay omitido (COMPLETA): capital inicial inválido (N/D)")
                    .count(),
                4
            );
            assert_no_performance_lines(&report);
            assert!(!report.contains("$0.00"));
            assert!(!report.contains("-100.00%"));
        }

        #[test]
        fn first_eligible_window_reports_supplied_zero_terminal_equity_as_total_loss() {
            let case = "mx06-executed-zero-terminal";
            if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
                let cfg = ReplayConfig {
                    initial_capital: 1_000.0,
                    warmup_ticks: 64,
                    ..ReplayConfig::default()
                };
                let calls = Arc::new(AtomicUsize::new(0));
                let retained = Arc::clone(&calls);
                // A controlled report fixture, not Core or an executed strategy.
                // N=75 must invoke the supplied replay path exactly once.
                let stats = report_replay_window("COMPLETA", 75, &cfg, 3.0, move || {
                    calls.fetch_add(1, Ordering::Relaxed);
                    ReplayStats {
                        final_capital: 0.0,
                        ..ReplayStats::default()
                    }
                })
                .expect("11 post-warmup rows pass this guard");
                assert_eq!(retained.load(Ordering::Relaxed), 1);
                assert_eq!(stats.final_capital.to_bits(), 0.0f64.to_bits());
                return;
            }
            let report = captured_report(
            "si_b15_report_contract::initial_ledger::first_eligible_window_reports_supplied_zero_terminal_equity_as_total_loss",
            case,
        );
            assert!(
                report.contains("Capital: $1000.00 → $0.00 (-100.00%)"),
                "{report}"
            );
            assert!(report.contains("equivalente 3d: N/D"), "{report}");
            assert!(!report.contains("Replay omitido"));
        }

        // These contracts call the actual input consumer, not a mirrored eligibility
        // helper. The child owns a private working directory with a real tape and
        // a model probe, and must return before models, genome, FRED or Core.
        struct InputFixture(std::path::PathBuf);

        impl InputFixture {
            fn new(filtered_empty: bool) -> Self {
                let nonce = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("fixture clock")
                    .as_nanos();
                let root = std::env::temp_dir().join(format!(
                    "trader-gemini-b15-input-{}-{nonce}",
                    std::process::id()
                ));
                // create_dir fails if an unrelated directory already exists. No
                // existing project files are used, overwritten or recursively cleaned.
                std::fs::create_dir(&root).expect("create isolated fixture directory");
                let fixture = Self(root);
                std::fs::create_dir(fixture.0.join("models"))
                    .expect("create model probe directory");
                std::fs::write(fixture.0.join("models/guard_probe.json"), b"not a model")
                    .expect("write model probe");

                const RAW_ROWS: usize = 10_000;
                let mut bytes = Vec::with_capacity(8 + RAW_ROWS * 40);
                bytes.extend_from_slice(b"TGMTICK1");
                for i in 0..RAW_ROWS {
                    let ts_ms = 1_700_000_000_000_u64 + i as u64 * 100;
                    let bid: f64 = if filtered_empty { 0.0 } else { 99.99 };
                    // Native endian matches the consumer's existing repr(C) mmap
                    // format: u64 plus four f64 fields, each eight bytes, no padding.
                    bytes.extend_from_slice(&ts_ms.to_ne_bytes());
                    bytes.extend_from_slice(&bid.to_ne_bytes());
                    bytes.extend_from_slice(&100.01_f64.to_ne_bytes());
                    bytes.extend_from_slice(&2.0_f64.to_ne_bytes());
                    bytes.extend_from_slice(&2.0_f64.to_ne_bytes());
                }
                assert_eq!(
                    bytes.len(),
                    400_008,
                    "10k records plus the supported 8-byte header"
                );
                std::fs::write(fixture.0.join("BTCUSDT_ticks_fixture.bin"), bytes)
                    .expect("write real binary tape fixture");
                fixture
            }
        }

        impl Drop for InputFixture {
            fn drop(&mut self) {
                // This exact unique child was created above. The child process has
                // exited and released its mmap before the owning fixture is dropped.
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        fn captured_input_consumer(test_name: &str, case: &str, filtered_empty: bool) -> String {
            let fixture = InputFixture::new(filtered_empty);
            let output = Command::new(std::env::current_exe().expect("test executable"))
                .args(["--exact", test_name, "--nocapture"])
                .env(CHILD_CASE, case)
                .current_dir(&fixture.0)
                .output()
                .expect("run real input consumer fixture");
            assert!(
                output.status.success(),
                "input fixture failed: {:?}; stderr={}",
                output.status,
                String::from_utf8_lossy(&output.stderr),
            );
            let stdout = String::from_utf8(output.stdout).expect("report must be UTF-8");
            let stderr = String::from_utf8(output.stderr).expect("diagnostics must be UTF-8");
            format!("{stdout}\n{stderr}")
        }

        fn assert_input_omission_precedes_data_and_dependencies(report: &str, rows: usize) {
            assert_eq!(
                report
                    .matches("Replay omitido (COMPLETA): datos insuficientes tras límite/filtro")
                    .count(),
                1,
                "the actual input consumer must stop once: {report}",
            );
            assert!(
                report.contains(&format!("{rows} filas, warmup=500")),
                "{report}"
            );
            assert!(
                report.contains("Fuente: BTCUSDT_ticks_fixture.bin"),
                "{report}"
            );
            assert!(
                report.contains("TGMTICK1 detectado"),
                "the real mmap/header path must run: {report}"
            );
            assert_no_performance_lines(report);
            for forbidden in [
                "🗂️  Data:",
                "   Período:",
                "   BTC:",
                "buy&hold:",
                "   🧠 Modelos:",
                "modelo guard_probe",
                "Cargando macro FRED",
                "Símbolo del replay:",
                "Symbol specs registradas",
                "GENOME-STORE",
                "No se pudo abrir",
                "❌ Datos insuficientes:",
                "VENTANA COMPLETA (",
                "🎯 EQUIVALENTES GEOMÉTRICOS",
            ] {
                assert!(
                    !report.contains(forbidden),
                    "omitted input reached {forbidden}: {report}"
                );
            }
        }

        #[test]
        fn actual_input_consumer_stops_after_max_ticks_before_data_and_models() {
            let case = "mx06-actual-max-ticks";
            if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
                let args = [
                    "backtest_windows",
                    "--file",
                    "BTCUSDT_ticks_fixture.bin",
                    "--max-ticks",
                    "500",
                ]
                .map(str::to_owned);
                // Match the production worker even on early returns: release/LTO
                // may reserve large frames before an omitted path returns.
                std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || run_backtest_from_args(&args))
                    .expect("spawn input consumer worker")
                    .join()
                    .expect("input consumer worker panicked");
                return;
            }
            let report = captured_input_consumer(
        "si_b15_report_contract::initial_ledger::actual_input_consumer_stops_after_max_ticks_before_data_and_models",
        case,
        false,
    );
            assert_input_omission_precedes_data_and_dependencies(&report, 500);
        }

        #[test]
        fn actual_input_consumer_stops_when_filter_empties_tape_before_data_and_models() {
            let case = "mx06-actual-filter-empty";
            if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
                let args = [
                    "backtest_windows",
                    "--file",
                    "BTCUSDT_ticks_fixture.bin",
                    "--max-ticks",
                    "10000",
                ]
                .map(str::to_owned);
                // Match the production worker even on early returns: release/LTO
                // may reserve large frames before an omitted path returns.
                std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || run_backtest_from_args(&args))
                    .expect("spawn input consumer worker")
                    .join()
                    .expect("input consumer worker panicked");
                return;
            }
            let report = captured_input_consumer(
        "si_b15_report_contract::initial_ledger::actual_input_consumer_stops_when_filter_empties_tape_before_data_and_models",
        case,
        true,
    );
            assert_input_omission_precedes_data_and_dependencies(&report, 0);
        }
    }
}
