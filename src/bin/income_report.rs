//! INFORME CONTABLE REAL (F3.5) — pre/post fees desde /fapi/v1/income.
//!
//! QUÉ: reporte por símbolo con PnL bruto (REALIZED_PNL), comisiones
//!      (COMMISSION), funding (FUNDING_FEE) y PnL neto — TODO desde la
//!      contabilidad del exchange, no de simulaciones.
//! USO:
//!   income_report                    → últimos 7 días (demo/testnet según USE_TESTNET)
//!   income_report --days 30          → ventana custom
//!   income_report --live             → mainnet (requiere .env con llave activa)
//! Directriz de informes: ROI/PnL/WR antes Y después de fees, con el período
//! temporal usado siempre visible.

use execution_engine::executor::OrderExecutor;
use std::collections::BTreeMap;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let days: u64 = args
        .iter()
        .position(|a| a == "--days")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let is_testnet = !args.contains(&"--live".to_string());

    // .env local si existe (dotenv ligero sin dependencia extra) — ANTES de
    // leer credenciales: el chequeo solía correr primero y el binario moría
    // con "Credenciales ausentes" aunque el .env las tuviera.
    if let Ok(env_src) = std::fs::read_to_string(".env") {
        for line in env_src.lines() {
            if let Some((k, v)) = line.split_once('=') {
                if std::env::var(k).is_err() {
                    std::env::set_var(k.trim(), v.trim());
                }
            }
        }
    }

    // Credenciales canónicas + alias (F0.2)
    let (key, secret) = if is_testnet {
        let k = std::env::var("BINANCE_DEMO_API_KEY")
            .or_else(|_| std::env::var("BINANCE_TESTNET_API_KEY"))
            .unwrap_or_default();
        let s = std::env::var("BINANCE_DEMO_SECRET_KEY")
            .or_else(|_| std::env::var("BINANCE_TESTNET_SECRET_KEY"))
            .unwrap_or_default();
        (k, s)
    } else {
        (
            std::env::var("BINANCE_API_KEY").unwrap_or_default(),
            std::env::var("BINANCE_SECRET_KEY").unwrap_or_default(),
        )
    };
    if key.is_empty() || secret.is_empty() {
        eprintln!("❌ Credenciales ausentes (BINANCE_DEMO_* o BINANCE_API_KEY).");
        std::process::exit(1);
    }

    let mut exec = OrderExecutor::new(key, secret, is_testnet);
    exec.set_paper_trading(false);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let start_ms = now_ms - days * 86_400_000;

    println!("══════════════════════════════════════════════════════════════");
    println!(
        "📊 INFORME CONTABLE REAL — Binance Futures {}",
        if is_testnet { "TESTNET" } else { "MAINNET" }
    );
    println!(
        "   Ventana: últimos {} días ({} → {})",
        days,
        fmt_ms(start_ms),
        fmt_ms(now_ms)
    );
    println!("   Fuente: /fapi/v1/income (contabilidad del exchange — hechos, no estimaciones)");
    println!("══════════════════════════════════════════════════════════════");

    // B3.6b (auditoría): paginado — 1000 entradas por llamada truncaba
    // ventanas de 7/30 días sin aviso (sub-contando fees). 20 páginas =
    // hasta 20k entradas.
    let entries = match exec.fetch_income_paged(&[], start_ms, 20).await {
        Ok(e) => e,
        Err(err) => {
            eprintln!("❌ Income API falló: {}", err);
            std::process::exit(1);
        }
    };

    // Agregación por símbolo: pnl bruto, comisiones, funding
    #[derive(Default)]
    struct Agg {
        realized_pnl: f64,
        commission: f64,
        funding: f64,
        pnl_trades: u64,
        win_trades: u64,
    }
    let mut by_symbol: BTreeMap<String, Agg> = BTreeMap::new();
    let mut total = Agg::default();
    let mut first_ts = u64::MAX;
    let mut last_ts = 0u64;

    for e in &entries {
        let agg = by_symbol.entry(e.symbol.clone()).or_default();
        first_ts = first_ts.min(e.time);
        last_ts = last_ts.max(e.time);
        match e.income_type.as_str() {
            "REALIZED_PNL" => {
                agg.realized_pnl += e.income;
                agg.pnl_trades += 1;
                if e.income > 0.0 {
                    agg.win_trades += 1;
                }
            }
            "COMMISSION" => agg.commission += e.income, // negativo = pagado
            "FUNDING_FEE" => agg.funding += e.income,   // con signo
            _ => {}
        }
    }
    for agg in by_symbol.values() {
        total.realized_pnl += agg.realized_pnl;
        total.commission += agg.commission;
        total.funding += agg.funding;
        total.pnl_trades += agg.pnl_trades;
        total.win_trades += agg.win_trades;
    }

    if entries.is_empty() {
        println!("\n∅ Sin movimientos en la ventana.");
        return;
    }

    println!(
        "\n{:<14}{:>12}{:>11}{:>11}{:>12}{:>9}{:>8}",
        "SÍMBOLO", "PnL BRUTO", "FEES", "FUNDING", "PnL NETO", "WR", "TRDES"
    );
    println!("{}", "-".repeat(80));
    for (sym, a) in &by_symbol {
        let net = a.realized_pnl + a.commission + a.funding;
        let wr = if a.pnl_trades > 0 {
            (a.win_trades as f64 / a.pnl_trades as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "{:<14}{:>12.4}{:>11.4}{:>11.4}{:>12.4}{:>8.1}%{:>7}",
            if sym.is_empty() { "(global)" } else { sym },
            a.realized_pnl,
            a.commission,
            a.funding,
            net,
            wr,
            a.pnl_trades
        );
    }
    println!("{}", "-".repeat(80));
    let net = total.realized_pnl + total.commission + total.funding;
    let wr = if total.pnl_trades > 0 {
        (total.win_trades as f64 / total.pnl_trades as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "{:<14}{:>12.4}{:>11.4}{:>11.4}{:>12.4}{:>8.1}%{:>7}",
        "TOTAL", total.realized_pnl, total.commission, total.funding, net, wr, total.pnl_trades
    );

    let fee_drag_pct = if total.realized_pnl.abs() > 1e-9 {
        ((total.commission + total.funding) / total.realized_pnl.abs()) * 100.0
    } else {
        0.0
    };
    println!(
        "\n💸 Arrastre de costos: {:.1}% del volumen de PnL bruto (comisiones+funding / |bruto|)",
        fee_drag_pct
    );
    println!(
        "   WR bruto = WR neto por trade (Binance ya descuenta fees del REALIZED_PNL: {} entradas)",
        total.pnl_trades
    );
    if last_ts > 0 {
        println!(
            "   Datos reales entre {} → {}",
            fmt_ms(first_ts),
            fmt_ms(last_ts)
        );
    }
}

fn fmt_ms(ms: u64) -> String {
    // Formato local legible sin dependencia pesada: día/mes hora UTC.
    let secs = (ms / 1000) as i64;
    let days = secs / 86400;
    let rem = secs % 86400;
    let (y, mo, d) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}Z",
        y,
        mo,
        d,
        rem / 3600,
        (rem % 3600) / 60
    )
}

/// Conversión días-epoch → civil (algoritmo de Howard Hinnant, sin deps).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}
