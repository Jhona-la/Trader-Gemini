use feature_engine::{OFIModel, order_book_imbalance};
use polars::prelude::*;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("===========================================================");
    println!("🔬 TRADER GEMINI V5 - FASE 6: POC (RENTABILIDAD DE FEATURE AISLADA)");
    println!("===========================================================");

    let file_path = Path::new("data/historical/BTCUSDT_6M.parquet");
    if !file_path.exists() {
        println!("❌ Archivo no encontrado: {:?}. Ejecuta 'cargo run --bin download_history' primero.", file_path);
        return Ok(());
    }

    println!("📥 Cargando Dataset HFT (Parquet)...");
    let start_load = Instant::now();
    let mut file = File::open(file_path)?;
    let df = ParquetReader::new(&mut file).finish()?;
    
    let _opens = df.column("open")?.f64()?;
    let closes = df.column("close")?.f64()?;
    let highs = df.column("high")?.f64()?;
    let lows = df.column("low")?.f64()?;
    let volumes = df.column("volume")?.f64()?;

    let rows = df.height();
    println!("✅ Cargadas {} filas en {:?}", rows, start_load.elapsed());

    let mut ofi_model = OFIModel::new();
    
    // Variables de backtest simplificado (Sin comisiones por ahora, puramente capacidad predictiva del feature)
    let mut position = 0; // 1 = Long, -1 = Short, 0 = Flat
    let mut entry_price = 0.0;
    let mut pnl = 0.0;
    let mut total_trades = 0;
    let mut winning_trades = 0;

    let take_profit_pct = 0.002; // 0.2%
    let stop_loss_pct = 0.002;   // 0.2%
    let obi_threshold = 0.7;

    println!("🧪 Feature a evaluar: Order Book Imbalance (OBI) puro.");
    println!("🧪 Hipótesis: OBI > {} augura subida, OBI < -{} augura bajada a micro-escala.", obi_threshold, obi_threshold);
    println!("⚙️ Parámetros de prueba: TP {:.2}% | SL {:.2}%", take_profit_pct * 100.0, stop_loss_pct * 100.0);

    let start_sim = Instant::now();

    for i in 1..rows {
        let current_close = closes.get(i).unwrap_or(0.0);
        let prev_close = closes.get(i-1).unwrap_or(0.0);
        let current_high = highs.get(i).unwrap_or(0.0);
        let current_low = lows.get(i).unwrap_or(0.0);
        let volume = volumes.get(i).unwrap_or(0.0);

        // Simulamos Bid/Ask y Volúmenes usando Klines (Acercamiento tosco para POC ya que falta L2 real, pero útil como proxy direccional)
        // Asumimos bid = low, ask = high temporalmente
        let bid = current_low;
        let ask = current_high;
        // Volumen Bid vs Ask (Si vela verde = más volumen al ask, roja = más volumen al bid)
        let (bid_qty, ask_qty) = if current_close >= prev_close {
            (volume * 0.7, volume * 0.3)
        } else {
            (volume * 0.3, volume * 0.7)
        };

        // Extraer Feature OBI
        let current_obi = order_book_imbalance(bid_qty, ask_qty);
        let _current_ofi = ofi_model.update(bid, ask, bid_qty, ask_qty);

        // Lógica de Ejecución Aislada
        if position == 0 {
            if current_obi > obi_threshold {
                position = 1;
                entry_price = ask;
            } else if current_obi < -obi_threshold {
                position = -1;
                entry_price = bid;
            }
        } else if position == 1 {
            let unrealized = (current_high - entry_price) / entry_price;
            let max_loss = (current_low - entry_price) / entry_price;
            
            if unrealized >= take_profit_pct {
                pnl += take_profit_pct;
                total_trades += 1;
                winning_trades += 1;
                position = 0;
            } else if max_loss <= -stop_loss_pct {
                pnl -= stop_loss_pct;
                total_trades += 1;
                position = 0;
            }
        } else if position == -1 {
            let unrealized = (entry_price - current_low) / entry_price;
            let max_loss = (entry_price - current_high) / entry_price;
            
            if unrealized >= take_profit_pct {
                pnl += take_profit_pct;
                total_trades += 1;
                winning_trades += 1;
                position = 0;
            } else if max_loss <= -stop_loss_pct {
                pnl -= stop_loss_pct;
                total_trades += 1;
                position = 0;
            }
        }
    }

    let elapsed = start_sim.elapsed();
    let win_rate = if total_trades > 0 { (winning_trades as f64 / total_trades as f64) * 100.0 } else { 0.0 };

    println!("\n📊 === RESULTADOS DEL POC (FASE 6) ===");
    println!("⏱️ Tiempo de evaluación: {:?}", elapsed);
    println!("📈 PnL Acumulado (Sin apalancamiento): {:.2}%", pnl * 100.0);
    println!("🔄 Total Trades Ejecutados: {}", total_trades);
    println!("🏆 Win Rate de Feature Aislado: {:.2}%", win_rate);
    
    if win_rate > 55.0 && pnl > 0.0 {
        println!("✅ VEREDICTO: El Feature OBI posee Ventaja Estadística Independiente.");
    } else {
        println!("⚠️ VEREDICTO: El Feature OBI carece de Ventaja Aislada en esta temporalidad. Se requiere Fusión ML u otros factores.");
    }
    println!("===========================================================");

    Ok(())
}
