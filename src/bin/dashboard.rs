use crossterm::{
    cursor::{Hide, MoveTo},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{Clear, ClearType, EnterAlternateScreen},
};
use reqwest;
use serde::Deserialize;
use std::io::stdout;
use std::time::Duration;
use tokio::time::sleep;

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct SystemState {
    tick_counter: u64,
    unified_capital: f64,
    pnl_realized_scalp: f64,
    pnl_gross_scalp: f64,
    pnl_unrealized_scalp: f64,
    win_rate_scalp: f64,
    pnl_realized_swing: f64,
    pnl_gross_swing: f64,
    pnl_unrealized_swing: f64,
    win_rate_swing: f64,
    global_leverage: f64,
    global_max_drawdown: f64,
    #[serde(default)]
    cpu_usage: f32,
    #[serde(default)]
    memory_used_mb: f64,
    #[serde(default)]
    total_memory_mb: f64,
    #[serde(default)]
    net_roi_pct: f64,
    #[serde(default)]
    gross_roi_pct: f64,
    #[serde(default)]
    fees_paid: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = stdout();

    // Configurar terminal interactiva
    execute!(stdout, EnterAlternateScreen, Hide)?;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()?;

    let url = "http://127.0.0.1:3000/api/state";
    let initial_capital = std::env::var("INITIAL_CAPITAL")
        .unwrap_or_else(|_| "0.0".to_string())
        .parse()
        .unwrap_or(0.0);

    loop {
        match client.get(url).send().await {
            Ok(response) => {
                if let Ok(state) = response.json::<SystemState>().await {
                    execute!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;

                    // Render Header
                    execute!(
                        stdout,
                        SetForegroundColor(Color::Cyan),
                        Print("====================================================\n"),
                        Print("🌌 TRADER GEMINI V5 - FLIGHT RECORDER & DASHBOARD 🌌\n"),
                        Print("====================================================\n"),
                        ResetColor
                    )?;

                    // Render Metrics
                    let pnl_total = if initial_capital > 0.0 {
                        state.unified_capital - initial_capital
                    } else {
                        state.pnl_realized_scalp + state.pnl_realized_swing
                    };
                    let pnl_color = if pnl_total >= 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    };
                    let un_color = if state.pnl_unrealized_scalp >= 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    };
                    let growth = if initial_capital > 0.0 {
                        ((state.unified_capital - initial_capital) / initial_capital) * 100.0
                    } else {
                        state.net_roi_pct
                    };

                    execute!(
                        stdout,
                        Print("\n"),
                        SetForegroundColor(Color::Magenta),
                        Print(format!(">> CAPITAL:       ${:.4}\n", state.unified_capital)),
                        SetForegroundColor(Color::White),
                        Print(format!(">> COMPOUND GRO.: {:.2}%\n", growth)),
                        SetForegroundColor(pnl_color),
                        Print(format!(">> PNL NETO TOT.: ${:.4}\n", pnl_total)),
                        SetForegroundColor(Color::Yellow),
                        Print(format!(">> FEES PAGADOS:  ${:.4}\n", state.fees_paid)),
                        Print("\n"),
                        SetForegroundColor(Color::Cyan),
                        Print("--- ⚡ SCALP ENGINE ---\n"),
                        SetForegroundColor(Color::Yellow),
                        Print(format!(
                            ">> WIN RATE:       {:.2}%\n",
                            state.win_rate_scalp * 100.0
                        )),
                        SetForegroundColor(Color::White),
                        Print(format!(
                            ">> GROSS PNL:      ${:.4}\n",
                            state.pnl_gross_scalp
                        )),
                        SetForegroundColor(Color::White),
                        Print(format!(
                            ">> NET PNL (FEES): ${:.4}\n",
                            state.pnl_realized_scalp
                        )),
                        SetForegroundColor(un_color),
                        Print(format!(
                            ">> UNREALIZED PNL: ${:.4}\n",
                            state.pnl_unrealized_scalp
                        )),
                        Print("\n"),
                        SetForegroundColor(Color::Magenta),
                        Print("--- 🚀 SWING ENGINE ---\n"),
                        SetForegroundColor(Color::Yellow),
                        Print(format!(
                            ">> WIN RATE:       {:.2}%\n",
                            state.win_rate_swing * 100.0
                        )),
                        SetForegroundColor(Color::White),
                        Print(format!(
                            ">> GROSS PNL:      ${:.4}\n",
                            state.pnl_gross_swing
                        )),
                        SetForegroundColor(Color::White),
                        Print(format!(
                            ">> NET PNL (FEES): ${:.4}\n",
                            state.pnl_realized_swing
                        )),
                        SetForegroundColor(if state.pnl_unrealized_swing >= 0.0 {
                            Color::Green
                        } else {
                            Color::Red
                        }),
                        Print(format!(
                            ">> UNREALIZED PNL: ${:.4}\n",
                            state.pnl_unrealized_swing
                        )),
                        Print("\n"),
                        SetForegroundColor(Color::DarkGrey),
                        Print(format!(
                            ">> LEVERAGE:       {:.1}x\n",
                            state.global_leverage
                        )),
                        SetForegroundColor(Color::DarkGrey),
                        Print(format!(">> ENGINE TICKS:   {}\n", state.tick_counter)),
                        SetForegroundColor(Color::DarkGrey),
                        Print(format!(
                            ">> HOST RAM/CPU:   {:.1}MB / {:.1}% CPU\n",
                            state.memory_used_mb, state.cpu_usage
                        )),
                        Print("\n"),
                        SetForegroundColor(Color::Cyan),
                        Print("====================================================\n"),
                        ResetColor
                    )?;
                }
            }
            Err(_) => {
                execute!(
                    stdout,
                    MoveTo(0, 0),
                    Clear(ClearType::All),
                    SetForegroundColor(Color::Red),
                    Print("📡 Esperando a que God Engine encienda en puerto 3000...\n"),
                    ResetColor
                )?;
            }
        }

        // 10 Hz refresh rate
        sleep(Duration::from_millis(100)).await;
    }

    // #[allow(unreachable_code)]
    // execute!(stdout, Show, LeaveAlternateScreen)?;
    // Ok(())
}
