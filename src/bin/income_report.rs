//! INFORME CONTABLE REAL (F3.5) — pre/post fees desde /fapi/v1/income.
//!
//! QUÉ: reporte por símbolo con PnL bruto (REALIZED_PNL), comisiones
//!      (COMMISSION), funding (FUNDING_FEE) y subtotal seleccionado, por moneda.
//!      Filas de income no son operaciones independientes ni un WR neto.
//! USO:
//!   income_report                    → últimos 7 días (demo/testnet)
//!   income_report --days 30          → ventana custom
//!   income_report --live             → mainnet (requiere .env con llave activa)
//! Período y unidades explícitos. ROI requiere capital/flujos; WR neto requiere
//! operaciones identificadas y costes vinculados: no se inventan con filas.

use execution_engine::executor::OrderExecutor;
use execution_engine::income_evidence::{aggregate_income_by_asset, income_lookback_start};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let days: u64 = match args.iter().position(|a| a == "--days") {
        None => 7,
        Some(i) => match args.get(i + 1).and_then(|v| v.parse().ok()) {
            Some(days) => days,
            None => {
                eprintln!("--days requiere un entero positivo");
                std::process::exit(2);
            }
        },
    };
    let now_ms = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(t) => t.as_millis() as u64,
        Err(_) => {
            eprintln!("Reloj anterior al epoch");
            std::process::exit(2);
        }
    };
    let start_ms = match income_lookback_start(now_ms, days) {
        Ok(start) => start,
        Err(reason) => {
            eprintln!("Ventana inválida: {reason:?}");
            std::process::exit(2);
        }
    };
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
    println!("   Fuente: /fapi/v1/income; agotamiento de páginas no garantiza snapshot ni retención histórica completa");
    println!("══════════════════════════════════════════════════════════════");

    // Freeze the exact interval displayed above. The page cap is a resource
    // budget, not permission to present a truncated traversal as complete.
    let window = match exec.fetch_income_window(&[], start_ms, now_ms, 20).await {
        Ok(window) => window,
        Err(err) => {
            eprintln!("❌ Income API falló: {}", err);
            std::process::exit(1);
        }
    };
    println!(
        "   Páginas leídas: {}; cobertura observada: {:?}",
        window.pages_read, window.coverage
    );
    let entries = match window.into_exhausted_entries() {
        Ok(entries) => entries,
        Err(reason) => {
            eprintln!("Informe no emitido como completo: {reason:?}");
            std::process::exit(1);
        }
    };
    if entries.is_empty() {
        println!("\n∅ El endpoint no devolvió movimientos; no demuestra cobertura fuera de su retención.");
        return;
    }
    let summary = match aggregate_income_by_asset(&entries) {
        Ok(summary) => summary,
        Err(reason) => {
            eprintln!("Agregación inválida: {reason:?}");
            std::process::exit(1);
        }
    };
    println!(
        "\n{:<9}{:<14}{:>12}{:>12}{:>12}{:>12}{:>12}{:>9}{:>8}",
        "MONEDA",
        "SÍMBOLO",
        "BRUTO",
        "COMISIÓN",
        "FUNDING",
        "SUBTOTAL",
        "OTROS",
        "% FILAS+",
        "FILAS"
    );
    for ((asset, symbol), totals) in &summary.by_symbol {
        print_totals(
            asset,
            if symbol.is_empty() {
                "(global)"
            } else {
                symbol
            },
            totals,
        );
    }
    for (asset, totals) in &summary.by_asset {
        print_totals(asset, "TOTAL MONEDA", totals);
        println!(
            "   {asset}: razón firmada (comisión+funding)/|bruto| = {}; filas de otras clases = {}",
            totals
                .cost_to_abs_realized_ratio()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "N/D".into()),
            totals.other_rows
        );
    }
    println!("   SUBTOTAL = REALIZED_PNL + COMMISSION + FUNDING_FEE; OTROS conserva las demás clases, incluidas transferencias: no son automáticamente beneficio.");
    println!("   % FILAS+ cuenta filas REALIZED_PNL positivas, no WR de operaciones ni WR neto. Requiere enlazar fills y costes para medirlos.");
    println!("   No se suman monedas distintas. N/D indica razón no definida o no representable, no cero.");
    println!(
        "   Eventos recibidos entre {} → {}",
        fmt_ms(entries.iter().map(|e| e.time).min().unwrap()),
        fmt_ms(entries.iter().map(|e| e.time).max().unwrap())
    );
}

fn print_totals(
    asset: &str,
    symbol: &str,
    totals: &execution_engine::income_evidence::IncomeTotals,
) {
    let net = totals
        .selected_net()
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "INVÁLIDO".into());
    let positive = totals
        .positive_row_fraction()
        .map(|v| format!("{:.1}", v * 100.0))
        .unwrap_or_else(|| "N/D".into());
    // Default Display preserves small nonzero values; fixed decimals are not evidence.
    println!(
        "{:<9}{:<14}{:>12}{:>12}{:>12}{:>12}{:>12}{:>9}{:>8}",
        asset,
        symbol,
        totals.realized_pnl,
        totals.commission,
        totals.funding,
        net,
        totals.other_income,
        positive,
        totals.pnl_rows
    );
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
