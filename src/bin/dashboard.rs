use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use reqwest;
use serde::Deserialize;
use std::io::stdout;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Deserialize, Debug)]
struct SystemState {
    tick_counter: u64,
    unified_capital: f64,
    pnl_realized_scalp: f64,
    pnl_unrealized_scalp: f64,
    win_rate_scalp: f64,
    global_leverage: f64,
    global_max_drawdown: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = stdout();
    
    // Configurar terminal interactiva
    execute!(stdout, EnterAlternateScreen, Hide)?;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(50))
        .build()?;
    
    let url = "http://127.0.0.1:3000/api/state";
    let initial_capital = 13.0;

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
                    let pnl_total = state.unified_capital - initial_capital;
                    let pnl_color = if pnl_total >= 0.0 { Color::Green } else { Color::Red };
                    let un_color = if state.pnl_unrealized_scalp >= 0.0 { Color::Green } else { Color::Red };
                    let growth = (state.unified_capital / initial_capital) * 100.0;
                    
                    execute!(
                        stdout,
                        Print("\n"),
                        SetForegroundColor(Color::Magenta), Print(format!(">> CAPITAL:       ${:.4}\n", state.unified_capital)),
                        SetForegroundColor(Color::White), Print(format!(">> COMPOUND GRO.: {:.2}%\n", growth)),
                        SetForegroundColor(pnl_color), Print(format!(">> PNL NETO:      ${:.4}\n", pnl_total)),
                        Print("\n"),
                        SetForegroundColor(Color::Yellow), Print(format!(">> SCALP WIN RATE: {:.2}%\n", state.win_rate_scalp * 100.0)),
                        SetForegroundColor(Color::White), Print(format!(">> REALIZED PNL:   ${:.4}\n", state.pnl_realized_scalp)),
                        SetForegroundColor(un_color), Print(format!(">> UNREALIZED PNL: ${:.4}\n", state.pnl_unrealized_scalp)),
                        Print("\n"),
                        SetForegroundColor(Color::DarkGrey), Print(format!(">> LEVERAGE:       {:.1}x\n", state.global_leverage)),
                        SetForegroundColor(Color::DarkGrey), Print(format!(">> ENGINE TICKS:   {}\n", state.tick_counter)),
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
