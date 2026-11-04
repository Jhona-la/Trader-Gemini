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
    terminal::{
        disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
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
    let vn = Vec::new();
    let tsallis = Vec::new();
    let gauge = Vec::new();
    let void_risks = Vec::new();
    let betti_1 = Vec::new();

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
        match packet {
            Some(packet) => {
                let (p_up, p_down) = match &packet.lattice {
                    Some(lattice) => (
                        format!("{:.1}%", lattice.prob_up_total * 100.0),
                        format!("{:.1}%", lattice.prob_down_total * 100.0),
                    ),
                    None => ("N/D".to_string(), "N/D".to_string()),
                };
                let (nodes, pods) = match &packet.hypergraph {
                    Some(graph) => (graph.nodes.len().to_string(), graph.pods.len().to_string()),
                    None => ("N/D".to_string(), "N/D".to_string()),
                };
                let (pillars, alarms) = match &packet.causal_report {
                    Some(report) => (
                        report.attributions.len().to_string(),
                        report.srp_alarm_active.to_string(),
                    ),
                    None => ("N/D".to_string(), "N/D".to_string()),
                };
                println!("Tick Counter:       {}", packet.tick_counter);
                println!("Unified Capital:    ${:.2} USD", packet.unified_capital);
                println!("PnL Realized (Net): ${:.2} USD", packet.pnl_realized);
                println!("Win Rate:           {:.1}%", packet.win_rate * 100.0);
                println!("Mirofish P(UP):     {}", p_up);
                println!("Mirofish P(DOWN):   {}", p_down);
                println!("HyperGraph Nodes:   {}", nodes);
                println!("Isolation Pods:     {}", pods);
                println!("Causal Pillars:     {}", pillars);
                println!("SRP Alarms Active:  {}", alarms);
            }
            None => {
                println!("Tick Counter:       N/D");
                println!("Unified Capital:    N/D");
                println!("PnL Realized (Net): N/D");
                println!("Win Rate:           N/D");
                println!("Mirofish P(UP):     N/D");
                println!("Mirofish P(DOWN):   N/D");
                println!("HyperGraph Nodes:   N/D");
                println!("Isolation Pods:     N/D");
                println!("Causal Pillars:     N/D");
                println!("SRP Alarms Active:  N/D");
            }
        }
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
                    || (key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL))
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

            render_tui(&mut stdout, packet.as_ref(), target_path)?;
        }
    }

    execute!(stdout, Show, LeaveAlternateScreen)?;
    disable_raw_mode()?;
    println!("Trader Viz desacoplado finalizado limpiamente.");
    Ok(())
}

fn render_tui(
    out: &mut std::io::Stdout,
    pkt: Option<&telemetry_server::mmap_viz_stream::UnifiedVizPacket>,
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

    let Some(pkt) = pkt else {
        execute!(
            out,
            SetForegroundColor(Color::DarkGrey),
            Print(format!(" Source: {} | Ticks: N/D | RAM: N/D\n\n", path)),
            Print(" Capital: N/D | PnL: N/D | Lattice: N/D | Hypergraph: N/D | Causalidad: N/D\n"),
            ResetColor
        )?;
        out.flush()?;
        return Ok(());
    };

    execute!(
        out,
        SetForegroundColor(Color::DarkGrey),
        Print(format!(
            " Source: {} | Ticks: {} | RAM: {:.1} MB\n\n",
            path, pkt.tick_counter, pkt.memory_used_mb
        )),
        ResetColor
    )?;

    // 1. Panel de Métricas Financieras Pre & Post-Fees
    let net_pnl = pkt.pnl_realized;
    let (gross_pnl, fees) = match &pkt.causal_report {
        Some(report) => (
            format!("{:.2}", report.total_gross_pnl),
            format!("{:.2}", report.total_fees),
        ),
        None => ("N/D".to_string(), "N/D".to_string()),
    };
    let pnl_color = if net_pnl >= 0.0 {
        Color::Green
    } else {
        Color::Red
    };

    execute!(
        out,
        SetForegroundColor(Color::Yellow),
        Print("─── [1] ESTADO FINANCIERO INTEGRAL (PRE & POST FEES) ─────────────────────────────────────────\n"),
        ResetColor,
        Print(format!(" Capital Global:    ${:.2} USD         Apalancamiento: {:.1}x\n", pkt.unified_capital, pkt.global_leverage)),
        SetForegroundColor(pnl_color),
        Print(format!(" PnL Neto (Post):   ${:.2} USD         ROI Neto:       {:.2}%\n", net_pnl, pkt.global_roi)),
        ResetColor,
        Print(format!(" PnL Bruto (Pre):   {} USD         Comisiones:     {} USD\n", gross_pnl, fees)),
        Print(format!(" Win Rate Global:   {:.1}%              Drawdown:       N/D\n\n", pkt.win_rate * 100.0)),
    )?;

    // 2. Panel Mirofish Probability Lattice
    let (p_up, p_down, ofi, expected_price) = match &pkt.lattice {
        Some(lattice) => {
            let expected = lattice
                .levels
                .last()
                .map(|level| level.expected_price)
                .unwrap_or(lattice.root_price);
            (
                format!("{:.1}%", lattice.prob_up_total * 100.0),
                format!("{:.1}%", lattice.prob_down_total * 100.0),
                format!("{:.3}", lattice.ofi_drift),
                format!("${:.2}", expected),
            )
        }
        None => (
            "N/D".to_string(),
            "N/D".to_string(),
            "N/D".to_string(),
            "N/D".to_string(),
        ),
    };
    let max_risk = match &pkt.tail_ridge {
        Some(ridge) => format!("{:.2} ms", ridge.max_risk_scale_ms),
        None => "N/D".to_string(),
    };

    execute!(
        out,
        SetForegroundColor(Color::Magenta),
        Print("─── [2] SIMULACIÓN PROBABILÍSTICA MIROFISH (GALTON BOARD & CONFORMAL) ───────────────────────\n"),
        ResetColor,
        Print(format!(" P(UP):             {}              P(DOWN):        {}\n", p_up, p_down)),
        Print(format!(" Drift OFI:         {}              Precio Esperado: {}\n", ofi, expected_price)),
        Print(format!(" Escalas Espectro:  32 diádicas        Riesgo CVaR:    Max @ {}\n\n", max_risk)),
    )?;

    // 3. Panel Grafo Hiperdimensional & Pods
    let (nodes_count, edges_count, pods_count, quarantined, total_energy, cycles) =
        match &pkt.hypergraph {
            Some(graph) => (
                graph.nodes.len().to_string(),
                graph.edges.len().to_string(),
                graph.pods.len().to_string(),
                graph
                    .pods
                    .iter()
                    .filter(|pod| {
                        pod.status
                        == telemetry_server::hyperdimensional_graph::PodIsolationState::Quarantined
                    })
                    .count()
                    .to_string(),
                format!("{:.2}", graph.total_energy),
                graph.global_betti_1_cycles.to_string(),
            ),
            None => (
                "N/D".to_string(),
                "N/D".to_string(),
                "N/D".to_string(),
                "N/D".to_string(),
                "N/D".to_string(),
                "N/D".to_string(),
            ),
        };

    execute!(
        out,
        SetForegroundColor(Color::Blue),
        Print("─── [3] GRAFO HIPERDIMENSIONAL 3D & PODS DE AISLAMIENTO ──────────────────────────────────────\n"),
        ResetColor,
        Print(format!(" Nodos Activos:     {}                  Aristas Vivas:  {}\n", nodes_count, edges_count)),
        Print(format!(" Pods de Red:       {}                  En Cuarentena:  {}\n", pods_count, quarantined)),
        Print(format!(" Energía Libre:     {}              Ciclos Betti 1: {}\n\n", total_energy, cycles)),
    )?;

    // 4. Panel Causal Meta-Analytics (Aumann-Shapley & Owen)
    execute!(
        out,
        SetForegroundColor(Color::Cyan),
        Print("─── [4] ATRIBUCIÓN CAUSAL AUMANN-SHAPLEY & EFICIENCIA SILICIO-ALPHA ─────────────────────────\n"),
        Print(" Pilar              Net PnL       Owen PnL    Net Share   Fees       η_CPU       SRP Status\n"),
        ResetColor,
    )?;

    if let Some(report) = &pkt.causal_report {
        for a in &report.attributions {
            let status_str = if a.is_quarantined { "ALARM" } else { "OK" };
            let status_color = if a.is_quarantined {
                Color::Red
            } else {
                Color::Green
            };
            let name_trimmed = format!("{:18}", a.name);

            execute!(
                out,
                Print(format!(
                    " {} ${:7.2}     ${:7.2}   {:5.1}%      ${:5.2}   {:7.3}     ",
                    &name_trimmed[..18],
                    a.aumann_shapley_pnl,
                    a.owen_coalition_pnl,
                    a.net_share_pct,
                    a.attributed_fees,
                    a.silicon_efficiency_eta
                )),
                SetForegroundColor(status_color),
                Print(format!("[{}]\n", status_str)),
                ResetColor,
            )?;
        }
    } else {
        execute!(
            out,
            Print(" N/D: no hay actividad, ciclos y misses auténticos para atribuir.\n")
        )?;
    }

    execute!(
        out,
        SetForegroundColor(Color::DarkGrey),
        Print(
            "\n [q] Salir | [Ctrl+C] Interrumpir | Frecuencia de Sondeo: 5 Hz (Cero Contención)\n"
        ),
        ResetColor
    )?;

    out.flush()?;
    Ok(())
}
