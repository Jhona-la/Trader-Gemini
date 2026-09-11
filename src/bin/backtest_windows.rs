//! BACKTEST MULTI-VENTANA (F7) — estadísticas por ventana temporal con el
//! motor HONESTO (booktick_replay sobre ticks reales).
//!
//! USO: cargo run --release --bin backtest_windows [-- --file data/BTCUSDT_ticks_REAL.bin]
//!
//! Reporta: ventana completa + sub-ventanas 3d/7d/15d (según data disponible),
//! con PnL bruto/neto, WR, trades, sharpe, drawdown, y la TASA REAL lograda
//! vs la tasa que el operador busca (para confrontar expectativas con datos).

use backtest_engine::booktick_replay::{run_booktick_replay, ReplayConfig, ReplayTick};
use quantum_arena::genome::SuperGenotype;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let file = args
        .iter()
        .position(|a| a == "--file")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "data/BTCUSDT_ticks_REAL.bin".to_string());

    println!("══════════════════════════════════════════════════════════════════");
    println!("📊 BACKTEST MULTI-VENTANA — MOTOR HONESTO (ticks reales)");
    println!("   Fuente: {} ", file);
    println!("   Motor: GodEngineCore::process_event (mismo camino que producción)");
    println!("   Slippage: ATR real · Fees: del genoma (pisos realistas)");
    println!("══════════════════════════════════════════════════════════════════");

    // Cargar ticks
    let mmap = match (|| -> Result<memmap2::Mmap, String> {
        let f = std::fs::File::open(&file).map_err(|e| e.to_string())?;
        unsafe { memmap2::MmapOptions::new().map(&f).map_err(|e| e.to_string()) }
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
    let n_total = mmap.len() / sz;
    if n_total < 10_000 {
        eprintln!("❌ Datos insuficientes: {} ticks", n_total);
        std::process::exit(1);
    }
    let ptr = mmap.as_ptr() as *const BinTick;
    let raw = unsafe { std::slice::from_raw_parts(ptr, n_total) };

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

    let first_ts = ticks.first().map(|t| t.ts_ms).unwrap_or(0);
    let last_ts = ticks.last().map(|t| t.ts_ms).unwrap_or(0);
    let total_days = (last_ts.saturating_sub(first_ts)) as f64 / 86_400_000.0;
    let p_start = ticks.first().map(|t| t.mid()).unwrap_or(0.0);
    let p_end = ticks.last().map(|t| t.mid()).unwrap_or(0.0);
    let bh_ret = if p_start > 0.0 { (p_end / p_start - 1.0) * 100.0 } else { 0.0 };

    println!("\n🗂️  Data: {} ticks válidos de {} crudos", ticks.len(), n_total);
    println!("   Período: {:.1} días", total_days);
    println!("   BTC: ${:.0} → ${:.0} (buy&hold: {:+.1}%)", p_start, p_end, bh_ret);

    // Genoma: el activo del almacén o baseline
    let genome = quantum_arena::genome_store::GenomeEnvelope::load_active()
        .map(|e| e.genome)
        .unwrap_or_else(|| SuperGenotype::new_baseline(0.0002, 0.0005));

    let initial_capital = 1000.0; // USDT estándar para comparabilidad
    let cfg = ReplayConfig {
        initial_capital,
        warmup_ticks: 500,
    };

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
    let full = run_booktick_replay(&ticks, &genome, omni.as_ref(), &cfg);
    print_window("COMPLETA", &full, initial_capital, total_days);

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
        let wdays = (window.last().unwrap().ts_ms - window.first().unwrap().ts_ms) as f64 / 86_400_000.0;
        println!("\n{}", sep2);
        println!("📅 ÚLTIMOS {} DÍAS ({:.1} días reales)", days, wdays);
        println!("{}", sep2);
        let stats = run_booktick_replay(window, &genome, omni.as_ref(), &cfg);
        print_window(&format!("{}D", days), &stats, initial_capital, wdays);
    }

    // ===== PROYECCIÓN HONESTA =====
    println!("\n{}", sep);
    println!("🎯 PROYECCIÓN (EXTRAPOLACIÓN de la tasa REAL — no promesa)");
    println!("{}", sep);
    if full.trades > 0 && full.net_pnl != 0.0 {
        let daily_ret = full.net_pnl / initial_capital / total_days;
        println!("   Tasa diaria REAL: {:+.3}%", daily_ret * 100.0);
        for &d in &[3u64, 7u64, 15u64, 30u64, 365u64] {
            let compounded = (1.0 + daily_ret).powi(d as i32);
            println!(
                "   {} días: {:+.1}% (capital ${:.0} → ${:.0})",
                d,
                (compounded - 1.0) * 100.0,
                initial_capital,
                initial_capital * compounded
            );
        }
        // Contexto: qué se necesitaría para 100%/3d
        let needed_daily = 2.0f64.powf(1.0 / 3.0) - 1.0; // 26%/día
        println!(
            "\n   ⚖️  META 100%/3d requiere {:+.1}%/día — la tasa real es {:+.3}%/día",
            needed_daily * 100.0,
            daily_ret * 100.0
        );
        if daily_ret.abs() < needed_daily {
            println!("   La brecha es {:.0}x la tasa actual — requiere revisar estrategia, no solo optimizar.", needed_daily / daily_ret.abs().max(1e-9));
        }
    } else {
        println!("   Sin trades en la ventana completa — no hay tasa que proyectar.");
    }
    println!("\n══════════════════════════════════════════════════════════════════");
}

fn print_window(_label: &str, s: &backtest_engine::booktick_replay::ReplayStats, cap: f64, days: f64) {
    let roi = s.roi_net(cap) * 100.0;
    let daily = if days > 0.0 { roi / days } else { 0.0 };
    let compound_3d = if daily != 0.0 { (1.0 + daily / 100.0).powi(3) } else { 1.0 };
    println!(
        "   Capital: ${:.2} → ${:.2} ({:+.2}%)",
        cap,
        s.final_capital,
        roi
    );
    println!(
        "   PnL bruto: ${:+.4} · fees est: ${:.4} · neto: ${:+.4}",
        s.gross_pnl,
        s.fees_est,
        s.net_pnl
    );
    println!(
        "   WR neto: {:.1}% · WR bruto: {:.1}% · trades: {}",
        s.wr_net() * 100.0,
        if s.trades > 0 { s.wins_gross as f64 / s.trades as f64 * 100.0 } else { 0.0 },
        s.trades
    );
    println!(
        "   Sharpe/trade: {:.4} · MaxDD: {:.1}%",
        s.sharpe,
        s.max_dd * 100.0
    );
    println!(
        "   Tasa diaria: {:+.3}%/d · equivalente 3d: {:+.1}%",
        daily,
        (compound_3d - 1.0) * 100.0
    );
    if s.omni_neutral {
        println!("   ⚠️ Omni neutro (sin FRED)");
    }
}
