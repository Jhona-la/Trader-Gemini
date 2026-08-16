use polars::prelude::*;
use quantum_arena::genome::SuperGenotype;

/// Ejecuta el backtest híbrido:
/// 1. Polars (Vectorizado SIMD) para Features y Señales
/// 2. Bucle rápido nativo para Path-Dependency (SL/TP)
pub fn run_vectorized_hybrid(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[f64],
    cfg: &SuperGenotype,
) -> Result<(f64, f64, u32, u32), PolarsError> {
    let len = closes.len();
    let initial_capital = std::env::var("INITIAL_CAPITAL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    // F3.4: sin capital real no hay backtest — basura silenciosa fuera.
    if len == 0 || !initial_capital.is_finite() || initial_capital <= 0.0 {
        return Ok((initial_capital.max(0.0), 0.0, 0, 0));
    }

    // F3.2 — FIX SMA REAL: el "mock" anterior era sma_14 = close ⇒
    // close > close = siempre falso ⇒ CERO trades para siempre (motor muerto).
    // SMA-14 calculada en Rust puro (rolling_mean de polars cambia de firma
    // entre versiones; esto es estable y O(N)).
    let sma_window = 14usize;
    let mut sma14 = vec![f64::NAN; len];
    let mut running_sum = 0.0;
    for i in 0..len {
        running_sum += closes[i];
        if i >= sma_window {
            running_sum -= closes[i - sma_window];
            sma14[i] = running_sum / sma_window as f64;
        }
    }

    // 1. Cargar datos en memoria columnar (Zero-Copy si fuera mmap, aquí copiamos a Series)
    let s_close = Series::new("close".into(), closes);
    let s_high = Series::new("high".into(), highs);
    let s_low = Series::new("low".into(), lows);
    let s_volume = Series::new("volume".into(), volumes);
    let s_sma = Series::new("sma_14".into(), sma14);

    let df = DataFrame::new(vec![s_close, s_high, s_low, s_volume, s_sma])?;
    let lf = df.lazy();

    // 2. Vectorización Masiva (Generación de Features y Señales SIMD)
    let signals_lf = lf
        .with_columns(vec![
            (col("close") - col("close").shift(lit(1))).alias("delta"),
        ])
        .with_columns(vec![
            // Señal Long: Cierre cruza SMA hacia arriba con Momentum
            (col("close")
                .gt(col("sma_14"))
                .and(col("delta").gt(lit(cfg.trend_threshold))))
            .alias("signal_long"),
            // Señal Short: Cierre cruza SMA hacia abajo con Momentum
            (col("close")
                .lt(col("sma_14"))
                .and(col("delta").lt(lit(-cfg.trend_threshold))))
            .alias("signal_short"),
        ]);

    let result_df = signals_lf.collect()?;

    // Extraemos las columnas calculadas en paralelo a slices crudos para el bucle path-dependent
    let sig_long_ca = result_df.column("signal_long")?.bool()?;
    let sig_short_ca = result_df.column("signal_short")?.bool()?;

    // 3. Ejecución Híbrida Path-Dependent (SL/TP) en O(N) nativo
    let mut capital = initial_capital;
    let mut position = 0; // 0 = flat, 1 = long, -1 = short
    let mut entry_price = 0.0;
    let mut position_size = 0.0; // in coins
    let mut peak_capital = capital;
    let mut max_dd = 0.0;
    let mut wins = 0;
    let mut trades = 0;

    let tp_ratio = cfg.scalp_tp_base;
    let sl_ratio = cfg.scalp_sl_base;

    // F3.4/F4.5 — LOS COSTOS NO SE EVOLUCIONAN: fee con PISO realista
    // (taker mínimo 0.04% en Binance futures). El genoma podía evolucionar
    // max_fee_pct → 0 y "ganar" cobrándose a sí mismo cero comisión.
    // El techo 0.1% cubre VIP negativo. Rango real, no superstición.
    let taker_fee = cfg.max_fee_pct.clamp(0.0004, 0.001);

    for i in 1..len {
        // Ignoramos i=0 por los shifts
        let c = closes[i];
        let h = highs[i];
        let l = lows[i];

        let sl_long = sig_long_ca.get(i).unwrap_or(false);
        let sl_short = sig_short_ca.get(i).unwrap_or(false);

        // Simulación de Funding Rates cada 480 velas (≈8h en TF 1m).
        // F4.5: sensibilidad del genoma acotada a banda REALISTA [0.5, 2.0]×
        // (0.005%-0.02% por periodo de 8h) — el costo de carry no se evoluciona a cero.
        if i % 480 == 0 && position != 0 {
            let notional = entry_price * position_size;
            let funding_rate = cfg.funding_rate_sensitivity.clamp(0.5, 2.0) * 0.0001;
            let funding_drag = notional * funding_rate;
            capital -= funding_drag;
        }

        // Manejar posición abierta (Path Dependency)
        if position == 1 {
            let tp_price = entry_price * (1.0 + tp_ratio);
            let sl_price = entry_price * (1.0 - sl_ratio);

            if h >= tp_price {
                // TP hit
                let exit_fee = (tp_price * position_size) * taker_fee;
                let pnl = (tp_price - entry_price) * position_size;
                capital += pnl - exit_fee;
                wins += 1;
                position = 0;
            } else if l <= sl_price {
                // SL hit
                let exit_fee = (sl_price * position_size) * taker_fee;
                let pnl = (sl_price - entry_price) * position_size;
                capital += pnl - exit_fee;
                position = 0;
            } else if sl_short {
                // Reverse signal
                let exit_fee = (c * position_size) * taker_fee;
                let pnl = (c - entry_price) * position_size;
                capital += pnl - exit_fee;
                if pnl > exit_fee {
                    wins += 1;
                }
                position = 0;
            }
        } else if position == -1 {
            let tp_price = entry_price * (1.0 - tp_ratio);
            let sl_price = entry_price * (1.0 + sl_ratio);

            if l <= tp_price {
                // TP hit
                let exit_fee = (tp_price * position_size) * taker_fee;
                let pnl = (entry_price - tp_price) * position_size;
                capital += pnl - exit_fee;
                wins += 1;
                position = 0;
            } else if h >= sl_price {
                // SL hit
                let exit_fee = (sl_price * position_size) * taker_fee;
                let pnl = (entry_price - sl_price) * position_size;
                capital += pnl - exit_fee;
                position = 0;
            } else if sl_long {
                // Reverse signal
                let exit_fee = (c * position_size) * taker_fee;
                let pnl = (entry_price - c) * position_size;
                capital += pnl - exit_fee;
                if pnl > exit_fee {
                    wins += 1;
                }
                position = 0;
            }
        }

        // Abrir nuevas posiciones
        if position == 0 {
            if sl_long {
                position = 1;
                entry_price = c;
                let notional = capital * cfg.global_leverage;
                let entry_fee = notional * taker_fee;
                capital -= entry_fee; // Cobrar comisión de entrada
                position_size = notional / entry_price;
                trades += 1;
            } else if sl_short {
                position = -1;
                entry_price = c;
                let notional = capital * cfg.global_leverage;
                let entry_fee = notional * taker_fee;
                capital -= entry_fee; // Cobrar comisión de entrada
                position_size = notional / entry_price;
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

        if capital <= 0.0 {
            break;
        } // Margin Call
    }

    Ok((capital, max_dd, trades, wins))
}
