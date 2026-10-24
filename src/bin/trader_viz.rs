//! Trader Gemini V5 — Decoupled Zero-Latency Visual Suite (`trader_viz`) (FASE 5)
//!
//! Terminal UI (TUI) y monitor desacoplado de alto rendimiento que lee
//! exclusivamente vía memoria compartida (`mmap` lock-free) sin realizar llamadas
//! HTTP ni interferir con el bucle de eventos del motor de trading en producción.
//!
//! # QUÉ:
//! Aplicación interactiva de consola para la monitorización de estado unificado,
//! simulación probabilística de Mirofish, estado del grafo hiperdimensional y
//! atribución causal de Aumann-Shapley.
//!
//! # POR QUÉ:
//! Permite al operador auditar la ejecución sin penalizar la latencia del trading hot path.
//!
//! # PARA QUÉ:
//! Inspeccionar el capital, las comisiones reales, el avance del árbol de Mirofish,
//! la salud de los pods de aislamiento y las alarmas del detector Shiryaev-Roberts-Pollak.
//!
//! # CÓMO:
//! Sondeo periódico no bloqueante sobre el archivo binario mapeado en RAM mediante
//! `MmapVizStreamReader` y renderizado optimizado con `crossterm`.

use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use std::io::{stdout, Write};
use std::time::{Duration, Instant};
use telemetry_server::mmap_viz_stream::MmapVizStreamReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let once_mode = args.iter().any(|a| a == "--once");

    // Buscar ruta de telemetría mmap (prod, demo o fallback temporal)
    let possible_paths = [
        "data/telemetry_mmap.bin",
        "data/demo/telemetry_mmap.bin",
        "data/prod/telemetry_mmap.bin",
        "telemetry_mmap.bin",
    ];

    let mut target_path = possible_paths[0];
    for p in &possible_paths {
        if std::path::Path::new(p).exists() {
            target_path = p;
            break;
        }
    }

    let mut reader = MmapVizStreamReader::new(target_path);

    let active_symbols = quantum_arena::symbols::get_active_universe();
    let n = active_symbols.len().max(1);
    let vn = vec![0.45; n];
    let tsallis = vec![0.50; n];
    let gauge = vec![0.2; n];
    let void_risks = vec![0.0; n];
    let betti_1 = vec![0.0; n];

    if once_mode {
        let packet = reader.build_unified_packet(
            &active_symbols,
            &vn,
            &tsallis,
            &gauge,
            &void_risks,
            &betti_1,
        );
        println!("============================================================");
        println!("TRADER GEMINI V5 - DECOUPLED VISUAL SUITE (ONCE MODE)");
        println!("Target mmap: {}", target_path);
        println!("Tick Counter:       {}", packet.tick_counter);
        println!("Unified Capital:    ${:.2} USD", packet.unified_capital);
        println!("PnL Realized (Net): ${:.2} USD", packet.pnl_realized);
        println!("Win Rate:           {:.1}%", packet.win_rate * 100.0);
        println!("Mirofish P(UP):     {:.1}%", packet.lattice.prob_up_total * 100.0);
        println!("Mirofish P(DOWN):   {:.1}%", packet.lattice.prob_down_total * 100.0);
        println!("HyperGraph Nodes:   {}", packet.hypergraph.nodes.len());
        println!("Isolation Pods:     {}", packet.hypergraph.pods.len());
        println!("Causal Pillars:     {}", packet.causal_report.attributions.len());
        println!("SRP Alarms Active:  {}", packet.causal_report.srp_alarm_active);
        println!("============================================================");
        return Ok(());
    }

    // Configuración de terminal interactiva
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, Hide)?;

    let mut last_draw = Instant::now();
    let draw_interval = Duration::from_millis(200);

    loop {
        // Manejo de eventos de teclado no bloqueante
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q')
                    || key.code == KeyCode::Esc
                    || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
                {
                    break;
                }
            }
        }

        if last_draw.elapsed() >= draw_interval {
            last_draw = Instant::now();
            let packet = reader.build_unified_packet(
                &active_symbols,
                &vn,
                &tsallis,
                &gauge,
                &void_risks,
                &betti_1,
            );

            render_tui(&mut stdout, &packet, target_path)?;
        }
    }

    execute!(stdout, Show, LeaveAlternateScreen)?;
    disable_raw_mode()?;
    println!("Trader Viz desacoplado finalizado limpiamente.");
    Ok(())
}

fn render_tui(
    out: &mut std::io::Stdout,
    pkt: &telemetry_server::mmap_viz_stream::UnifiedVizPacket,
    path: &str,
) -> Result<(), std::io::Error> {
    execute!(out, MoveTo(0, 0), Clear(ClearType::All))?;

    // Cabecera Principal
    execute!(
        out,
        SetForegroundColor(Color::Cyan),
        Print("╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n"),
        Print("║            TRADER GEMINI V5 — SUITE VISUAL DESACOPLADA (ZERO-LATENCY MMAP)                   ║\n"),
        Print("╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n"),
        ResetColor
    )?;

    execute!(
        out,
        SetForegroundColor(Color::DarkGrey),
        Print(format!(" Source: {} | Ticks: {} | RAM: {:.1} MB\n\n", path, pkt.tick_counter, pkt.memory_used_mb)),
        ResetColor
    )?;

    // 1. Panel de Métricas Financieras Pre & Post-Fees
    let net_pnl = pkt.pnl_realized;
    let gross_pnl = pkt.causal_report.total_gross_pnl;
    let fees = pkt.causal_report.total_fees;
    let pnl_color = if net_pnl >= 0.0 { Color::Green } else { Color::Red };

    execute!(
        out,
        SetForegroundColor(Color::Yellow),
        Print("─── [1] ESTADO FINANCIERO INTEGRAL (PRE & POST FEES) ─────────────────────────────────────────\n"),
        ResetColor,
        Print(format!(" Capital Global:    ${:.2} USD         Apalancamiento: {:.1}x\n", pkt.unified_capital, pkt.global_leverage)),
        SetForegroundColor(pnl_color),
        Print(format!(" PnL Neto (Post):   ${:.2} USD         ROI Neto:       {:.2}%\n", net_pnl, pkt.global_roi)),
        ResetColor,
        Print(format!(" PnL Bruto (Pre):   ${:.2} USD         Comisiones:     ${:.2} USD\n", gross_pnl, fees)),
        Print(format!(" Win Rate Global:   {:.1}%              Drawdown:       0.0%\n\n", pkt.win_rate * 100.0)),
    )?;

    // 2. Panel Mirofish Probability Lattice
    let p_up = pkt.lattice.prob_up_total * 100.0;
    let p_down = pkt.lattice.prob_down_total * 100.0;
    let ofi = pkt.lattice.ofi_drift;
    let expected_price = pkt.lattice.levels.last().map(|l| l.expected_price).unwrap_or(pkt.lattice.root_price);

    execute!(
        out,
        SetForegroundColor(Color::Magenta),
        Print("─── [2] SIMULACIÓN PROBABILÍSTICA MIROFISH (GALTON BOARD & CONFORMAL) ───────────────────────\n"),
        ResetColor,
        Print(format!(" P(UP):             {:.1}%              P(DOWN):        {:.1}%\n", p_up, p_down)),
        Print(format!(" Drift OFI:         {:.3}              Precio Esperado: ${:.2}\n", ofi, expected_price)),
        Print(format!(" Escalas Espectro:  32 diádicas        Riesgo CVaR:    Max @ {:.2} ms\n\n", pkt.tail_ridge.max_risk_scale_ms)),
    )?;

    // 3. Panel Grafo Hiperdimensional & Pods
    let nodes_count = pkt.hypergraph.nodes.len();
    let edges_count = pkt.hypergraph.edges.len();
    let pods_count = pkt.hypergraph.pods.len();
    let quarantined = pkt.hypergraph.pods.iter().filter(|p| p.status == telemetry_server::hyperdimensional_graph::PodIsolationState::Quarantined).count();

    execute!(
        out,
        SetForegroundColor(Color::Blue),
        Print("─── [3] GRAFO HIPERDIMENSIONAL 3D & PODS DE AISLAMIENTO ──────────────────────────────────────\n"),
        ResetColor,
        Print(format!(" Nodos Activos:     {}                  Aristas Vivas:  {}\n", nodes_count, edges_count)),
        Print(format!(" Pods de Red:       {}                  En Cuarentena:  {}\n", pods_count, quarantined)),
        Print(format!(" Energía Libre:     {:.2}              Ciclos Betti 1: {}\n\n", pkt.hypergraph.total_energy, pkt.hypergraph.global_betti_1_cycles)),
    )?;

    // 4. Panel Causal Meta-Analytics (Aumann-Shapley & Owen)
    execute!(
        out,
        SetForegroundColor(Color::Cyan),
        Print("─── [4] ATRIBUCIÓN CAUSAL AUMANN-SHAPLEY & EFICIENCIA SILICIO-ALPHA ─────────────────────────\n"),
        Print(" Pilar              Net PnL       Owen PnL    Net Share   Fees       η_CPU       SRP Status\n"),
        ResetColor,
    )?;

    for a in &pkt.causal_report.attributions {
        let status_str = if a.is_quarantined { "ALARM" } else { "OK" };
        let status_color = if a.is_quarantined { Color::Red } else { Color::Green };
        let name_trimmed = format!("{:18}", a.name);

        execute!(
            out,
            Print(format!(" {} ${:7.2}     ${:7.2}   {:5.1}%      ${:5.2}   {:7.3}     ",
                &name_trimmed[..18], a.aumann_shapley_pnl, a.owen_coalition_pnl, a.net_share_pct, a.attributed_fees, a.silicon_efficiency_eta)),
            SetForegroundColor(status_color),
            Print(format!("[{}]\n", status_str)),
            ResetColor,
        )?;
    }

    execute!(
        out,
        SetForegroundColor(Color::DarkGrey),
        Print("\n [q] Salir | [Ctrl+C] Interrumpir | Frecuencia de Sondeo: 5 Hz (Cero Contención)\n"),
        ResetColor
    )?;

    out.flush()?;
    Ok(())
}
