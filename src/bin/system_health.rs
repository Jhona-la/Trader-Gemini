//! Diagnóstico parcial (F6): consultas de sólo lectura, con procedencia explícita.
//!
//! QUÉ: reúne las fuentes disponibles en un reporte legible:
//!      genoma activo (linaje), envolvente Kelly bayesiana, espectro temporal,
//!      ensamble ML, contabilidad del exchange (income API), feed health,
//!      registro local. No accede al estado privado de otro proceso ni certifica salud.
//! POR QUÉ: directriz — "auditoría y visualización diagnóstica" + "identifica
//!      las áreas donde el sistema es mudo, ciego y sordo". Antes cada métrica
//!      vivía en su propio log/binario; el operador no tenía UNA vista.
//! USO: cargo run --bin system_health           (demo/testnet)
//!      cargo run --bin system_health -- --live (mainnet — requiere IP en whitelist)

use execution_engine::executor::{ExecutionProvider, OrderExecutor};
use quantum_arena::genome_store::GenomeEnvelope;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let is_testnet = !args.contains(&"--live".to_string());

    // .env local (compatibilidad con lanzadores)
    if let Ok(env_src) = std::fs::read_to_string(".env") {
        for line in env_src.lines() {
            if let Some((k, v)) = line.split_once('=') {
                if std::env::var(k).is_err() {
                    std::env::set_var(k.trim(), v.trim());
                }
            }
        }
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           🩺 SYSTEM HEALTH — DIAGNÓSTICO PARCIAL                 ║");
    println!(
        "║  Entorno: {}                                    ║",
        if is_testnet {
            "DEMO/TESTNET"
        } else {
            "MAINNET     "
        }
    );
    println!("║  Momento: {}                             ║", chrono_now());
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ═══ 1. GENOMA ACTIVO (linaje del almacén) ═══
    println!("─── 🧬 GENOMA DEL ALMACÉN (no acredita carga del motor) ─────────────");
    match GenomeEnvelope::load_active() {
        Some(env) => {
            let age_min = (now_ms().saturating_sub(env.created_ms)) / 60_000;
            println!(
                "   Generación {} (padre {}) · fuente: {} · edad: {} min",
                env.generation, env.parent_generation, env.source, age_min
            );
            println!("   Razón: {}", truncate_str(&env.promotion_reason, 70));
            // Curvas de horizonte: los genes que gobiernan TP/SL en todo el espectro
            let tp_fast = env
                .genome
                .tp_horizon_curve
                .eval(quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS);
            let tp_slow = env
                .genome
                .tp_horizon_curve
                .eval(quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS);
            let sl_fast = env
                .genome
                .sl_horizon_curve
                .eval(quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS);
            let sl_slow = env
                .genome
                .sl_horizon_curve
                .eval(quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS);
            println!(
                "   Curva TP: {:.2}% (τ=30s) → {:.2}% (τ=12h) · SL: {:.2}% → {:.2}%",
                tp_fast * 100.0,
                tp_slow * 100.0,
                sl_fast * 100.0,
                sl_slow * 100.0
            );
            println!(
                "   RR en anclas: {:.1}x / {:.1}x · leverage global: {:.0}x · DD máx: {:.0}%",
                tp_fast / sl_fast.max(1e-9),
                tp_slow / sl_slow.max(1e-9),
                env.genome.global_leverage,
                env.genome.global_max_drawdown * 100.0
            );
            // Historia reciente
            let hist = GenomeEnvelope::recent_history(3);
            if hist.len() > 1 {
                println!("   Linaje reciente:");
                for (g, src, reason) in hist.iter().take(3) {
                    println!("     gen {} ← {}: {}", g, src, truncate_str(reason, 55));
                }
            }
        }
        None => println!("   ⚠️ Sin genoma legible en el almacén; estado del motor desconocido"),
    }
    println!();

    // ═══ 2. FEED HEALTH ═══
    println!("─── 📡 FEED ─────────────────────────────────────────────────────────");
    // El átomo de feed_health es local al proceso; no observa god_engine.
    println!("   Estado del motor: DESCONOCIDO (sin snapshot IPC del watchdog)");
    println!("   Consultar telemetría del proceso y su timestamp; no inferir salud desde este binario.");
    println!();

    // ═══ 3. CUENTA Y POSICIONES (verdad del exchange) ═══
    let (key, secret) = if is_testnet {
        (
            std::env::var("BINANCE_DEMO_API_KEY")
                .or_else(|_| std::env::var("BINANCE_TESTNET_API_KEY"))
                .unwrap_or_default(),
            std::env::var("BINANCE_DEMO_SECRET_KEY")
                .or_else(|_| std::env::var("BINANCE_TESTNET_SECRET_KEY"))
                .unwrap_or_default(),
        )
    } else {
        (
            std::env::var("BINANCE_API_KEY").unwrap_or_default(),
            std::env::var("BINANCE_SECRET_KEY").unwrap_or_default(),
        )
    };
    if key.is_empty() {
        println!("─── 💰 CUENTA ── ⚠️ sin credenciales ──");
        return;
    }
    let mut exec = OrderExecutor::new(key, secret, is_testnet);
    exec.set_paper_trading(false);

    println!("─── 💰 CUENTA Y POSICIONES (verdad del exchange) ────────────────────");
    match exec.fetch_account_balance().await {
        Ok(bal) => println!("   Balance USDT: ${:.2}", bal),
        Err(e) => {
            println!("   ❌ Balance: {}", truncate_str(&e, 60));
            if e.contains("-2015") {
                println!("      (IP no registrada en whitelist de Binance — normal fuera del host autorizado)");
            }
        }
    }
    match exec.fetch_position_risk().await {
        Ok(entries) => {
            let open: Vec<_> = entries.iter().filter(|p| p.is_open()).collect();
            if open.is_empty() {
                println!("   Posiciones: FLAT (0 abiertas)");
            } else {
                println!("   Posiciones abiertas: {}", open.len());
                for p in open.iter().take(10) {
                    println!(
                        "     {} {} {} @ {:.4} (uPnL {:+.4}, liq {:.2}, lev {:.0}x)",
                        p.symbol,
                        if p.is_long() { "LONG" } else { "SHORT" },
                        p.position_amt,
                        p.entry_price,
                        p.unrealized_pnl,
                        p.liquidation_price,
                        p.leverage
                    );
                }
            }
        }
        Err(e) => println!("   ⚠️ positionRisk: {}", truncate_str(&e, 60)),
    }

    // ═══ 4. CONTABILIDAD REAL (income API — la verdad pre/post fees) ═══
    println!();
    println!("─── 💸 INCOME (desde hace 7 días; una página, cobertura no certificada) ──");
    let week_ms = now_ms().saturating_sub(7 * 86_400_000);
    match exec.fetch_income(&[], week_ms, 1000).await {
        Ok(entries) => {
            println!("   {} filas recibidas; las sumas siguientes sólo cubren esta respuesta.", entries.len());
            if entries.len() >= 1000 {
                println!("   ⚠️ Límite de página alcanzado: puede faltar información de la ventana.");
            }
            let mut pnl_gross = 0.0;
            let mut commissions = 0.0;
            let mut funding = 0.0;
            let mut trades = 0u64;
            let mut wins = 0u64;
            for e in &entries {
                match e.income_type.as_str() {
                    "REALIZED_PNL" => {
                        pnl_gross += e.income;
                        trades += 1;
                        if e.income > 0.0 {
                            wins += 1;
                        }
                    }
                    "COMMISSION" => commissions += e.income,
                    "FUNDING_FEE" => funding += e.income,
                    _ => {}
                }
            }
            let net = pnl_gross + commissions + funding;
            if trades > 0 {
                println!(
                    "   PnL bruto: ${:+.4} · fees: ${:.4} · funding: ${:.4}",
                    pnl_gross, commissions, funding
                );
                println!(
                    "   Neto de filas: ${:+.4} · filas positivas: {:.1}% · filas REALIZED_PNL: {}",
                    net,
                    (wins as f64 / trades as f64) * 100.0,
                    trades
                );
                let drag = if pnl_gross.abs() > 1e-9 {
                    ((commissions + funding) / pnl_gross.abs()) * 100.0
                } else {
                    0.0
                };
                let funding_share = if (commissions + funding).abs() > 1e-9 {
                    funding / (commissions + funding).abs() * 100.0
                } else {
                    0.0
                };
                println!(
                    "   Arrastre de costos: {:.1}% del bruto (funding = {:.0}% del costo total)",
                    drag, funding_share
                );
            } else {
                println!("   ∅ Sin filas REALIZED_PNL en esta respuesta; no acredita ausencia de operaciones.");
                println!("   Fees: ${:+.4} · funding: ${:+.4} · neto de filas: ${:+.4}", commissions, funding, net);
            }
        }
        Err(e) => println!("   ⚠️ income: {}", truncate_str(&e, 60)),
    }

    // ═══ 5. REGISTRO DE ÓRDENES (máquina de estados local) ═══
    println!();
    println!("─── 📋 REGISTRO DE ESTA CONSULTA (no es el del motor) ───────────────");
    println!("   Sin snapshot IPC ni consulta de órdenes abiertas: estado operativo DESCONOCIDO.");
    let stats = exec.registry().stats();
    println!(
        "   Total: {} · activas: {} · llenas: {} · parciales: {} · canceladas: {} · rechazadas: {}",
        stats.total,
        stats.active,
        stats.filled,
        stats.partially_filled,
        stats.canceled,
        stats.rejected
    );
    if stats.active > 0 {
        println!(
            "   ⚠️ {} registros activos locales; no es confirmación de órdenes vivas en exchange",
            stats.active
        );
    }
    println!();

    // ═══ 6. MODO DE LA CUENTA ═══
    println!("─── 🔀 MODO DE POSICIÓN ─────────────────────────────────────────────");
    match exec.fetch_hedge_mode().await {
        Ok(true) => println!("   HEDGE (dualSidePosition) — consulta de sólo lectura"),
        Ok(false) => println!("   ONE-WAY — consulta de sólo lectura; no se cambió el modo"),
        Err(e) => println!(
            "   ⚠️ {}: {}",
            if is_testnet { "testnet" } else { "mainnet" },
            truncate_str(&e, 60)
        ),
    }
    println!();

    println!("══════════════════════════════════════════════════════════════════");
    println!("  Diagnóstico parcial, no certificación de salud. Para estado vivo:");
    println!("  telemetry_server (per-100 ticks) · income_report --days N       ");
    println!("══════════════════════════════════════════════════════════════════");
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn chrono_now() -> String {
    // UTC legible sin dependencia: días→civil (Hinnant)
    let secs = (now_ms() / 1000) as i64;
    let days = secs / 86400;
    let rem = secs % 86400;
    let (y, mo, d) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        y,
        mo,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

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

fn truncate_str(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}
