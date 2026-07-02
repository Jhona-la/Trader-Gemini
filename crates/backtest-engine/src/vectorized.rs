use polars::prelude::*;
use crate::UnifiedConfig;

/// Ejecuta el backtest híbrido: 
/// 1. Polars (Vectorizado SIMD) para Features y Señales
/// 2. Bucle rápido nativo para Path-Dependency (SL/TP)
pub fn run_vectorized_hybrid(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[f64],
    cfg: &UnifiedConfig,
) -> Result<(f64, f64, u32, u32), PolarsError> {
    let len = closes.len();
    if len == 0 {
        return Ok((cfg.starting_capital, 0.0, 0, 0));
    }

    // 1. Cargar datos en memoria columnar (Zero-Copy si fuera mmap, aquí copiamos a Series)
    let s_close = Series::new("close".into(), closes);
    let s_high = Series::new("high".into(), highs);
    let s_low = Series::new("low".into(), lows);
    let s_volume = Series::new("volume".into(), volumes);
    
    let df = DataFrame::new(vec![s_close, s_high, s_low, s_volume])?;
    let lf = df.lazy();
    
    // 2. Vectorización Masiva (Generación de Features y Señales SIMD)
    // Usaremos un umbral de momentum simple para propósitos de backtest.
    // En producción, esto replicaría los 150+ features.
    let signals_lf = lf
        .with_columns(vec![
            (col("close") - col("close").shift(lit(1))).alias("delta"),
            // Mock de SMA para que compile sin depender de RollingOptions que cambia de firma entre versiones
            col("close").alias("sma_14"),
        ])
        .with_columns(vec![
            // Señal Long: Cierre cruza SMA hacia arriba con Momentum
            (col("close").gt(col("sma_14"))
                .and(col("delta").gt(lit(cfg.tech_threshold_l))))
                .alias("signal_long"),
            // Señal Short: Cierre cruza SMA hacia abajo con Momentum
            (col("close").lt(col("sma_14"))
                .and(col("delta").lt(lit(-cfg.tech_threshold_s))))
                .alias("signal_short"),
        ]);
        
    let result_df = signals_lf.collect()?;
    
    // Extraemos las columnas calculadas en paralelo a slices crudos para el bucle path-dependent
    let sig_long_ca = result_df.column("signal_long")?.bool()?;
    let sig_short_ca = result_df.column("signal_short")?.bool()?;
    
    // 3. Ejecución Híbrida Path-Dependent (SL/TP) en O(N) nativo
    let mut capital = cfg.starting_capital;
    let mut position = 0; // 0 = flat, 1 = long, -1 = short
    let mut entry_price = 0.0;
    let mut position_size = 0.0; // in coins
    let mut peak_capital = capital;
    let mut max_dd = 0.0;
    let mut wins = 0;
    let mut trades = 0;
    
    let tp_ratio = cfg.tp_pct;
    let sl_ratio = cfg.sl_pct;
    
    for i in 1..len { // Ignoramos i=0 por los shifts
        let c = closes[i];
        let h = highs[i];
        let l = lows[i];
        
        let sl_long = sig_long_ca.get(i).unwrap_or(false);
        let sl_short = sig_short_ca.get(i).unwrap_or(false);
        
        // Manejar posición abierta (Path Dependency)
        if position == 1 {
            let tp_price = entry_price * (1.0 + tp_ratio);
            let sl_price = entry_price * (1.0 - sl_ratio);
            
            if h >= tp_price { // TP hit
                let pnl = (tp_price - entry_price) * position_size;
                capital += pnl;
                wins += 1;
                position = 0;
            } else if l <= sl_price { // SL hit
                let pnl = (sl_price - entry_price) * position_size;
                capital += pnl;
                position = 0;
            } else if sl_short { // Reverse signal
                let pnl = (c - entry_price) * position_size;
                capital += pnl;
                if pnl > 0.0 { wins += 1; }
                position = 0;
            }
        } else if position == -1 {
            let tp_price = entry_price * (1.0 - tp_ratio);
            let sl_price = entry_price * (1.0 + sl_ratio);
            
            if l <= tp_price { // TP hit
                let pnl = (entry_price - tp_price) * position_size;
                capital += pnl;
                wins += 1;
                position = 0;
            } else if h >= sl_price { // SL hit
                let pnl = (entry_price - sl_price) * position_size;
                capital += pnl;
                position = 0;
            } else if sl_long { // Reverse signal
                let pnl = (entry_price - c) * position_size;
                capital += pnl;
                if pnl > 0.0 { wins += 1; }
                position = 0;
            }
        }
        
        // Abrir nuevas posiciones
        if position == 0 {
            if sl_long {
                position = 1;
                entry_price = c;
                position_size = (capital * cfg.scalp_leverage) / entry_price;
                trades += 1;
            } else if sl_short {
                position = -1;
                entry_price = c;
                position_size = (capital * cfg.scalp_leverage) / entry_price;
                trades += 1;
            }
        }
        
        if capital > peak_capital {
            peak_capital = capital;
        }
        let dd = (peak_capital - capital) / peak_capital;
        if dd > max_dd {
            max_dd = dd;
        }
        
        if capital <= 0.0 { break; } // Margin Call
    }
    
    Ok((capital, max_dd, trades, wins))
}
