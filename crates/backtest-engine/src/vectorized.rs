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
        .unwrap_or(13.0);
    // F3.4: sin capital real no hay backtest — basura silenciosa fuera.
    if len == 0 || !initial_capital.is_finite() || initial_capital <= 0.0 {
        return Ok((initial_capital.max(0.0), 0.0, 0, 0));
    }

    // F3.2 — CÁLCULO DE DUAL-EMA Y VOLATILIDAD NATIVA EN O(N)
    // FIX #944 / FIX #1487: Prevenir cascada de NaN inicializando con el primer valor finito válido
    // D-675 (DÉCIMA OLA): el ciclo de funding son 8 h. Cuántas barras son
    // depende de la resolución de la serie, que antes se presuponía de 1 min
    // mediante el literal 480. `TG_BAR_INTERVAL_MS` lo declara explícitamente.
    let bar_interval_ms = std::env::var("TG_BAR_INTERVAL_MS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(60_000.0);
    let bars_per_funding = ((8.0 * 3_600_000.0) / bar_interval_ms).round().max(1.0) as usize;

    // D-673 (DÉCIMA OLA) — LOS PERÍODOS DE EMA SALEN DEL GENOMA.
    //
    // Estaban fijados por literal a 7 y 21 mientras el genoma llevaba
    // `ema_fast_period` y `ema_slow_period`. El indicador central de tendencia
    // del sistema se evaluaba con unos parámetros y se desplegaba con otros —
    // y de hecho los genes no se leían en NINGÚN sitio (D-649), de modo que
    // toda la señal primaria era un cruce de medias de manual.
    let p_fast = if cfg.ema_fast_period.is_finite() {
        cfg.ema_fast_period.clamp(2.0, 400.0)
    } else {
        7.0
    };
    let p_slow_raw = if cfg.ema_slow_period.is_finite() {
        cfg.ema_slow_period.clamp(2.0, 2000.0)
    } else {
        21.0
    };
    // La lenta debe ser realmente más lenta: si el genoma las cruza, se separan
    // manteniendo el orden (un cruce de medias con fast >= slow no tiene señal).
    let p_slow = p_slow_raw.max(p_fast * 1.2);
    let alpha_fast = 2.0 / (p_fast + 1.0);
    let alpha_slow = 2.0 / (p_slow + 1.0);
    let mut ema_fast = vec![0.0; len];
    let mut ema_slow = vec![0.0; len];

    let first_valid = closes
        .iter()
        .copied()
        .find(|c| c.is_finite() && *c > 0.0)
        .unwrap_or(1.0);
    let mut cur_fast = first_valid;
    let mut cur_slow = first_valid;
    for i in 0..len {
        let c = if closes[i].is_finite() && closes[i] > 0.0 {
            closes[i]
        } else {
            cur_fast
        };
        cur_fast = (alpha_fast * c + (1.0 - alpha_fast) * cur_fast).max(1e-6);
        cur_slow = (alpha_slow * c + (1.0 - alpha_slow) * cur_slow).max(1e-6);
        ema_fast[i] = if cur_fast.is_finite() {
            cur_fast
        } else {
            first_valid
        };
        ema_slow[i] = if cur_slow.is_finite() {
            cur_slow
        } else {
            first_valid
        };
    }

    // 1. Cargar datos en memoria columnar Polars
    let s_close = Series::new("close".into(), closes);
    let s_high = Series::new("high".into(), highs);
    let s_low = Series::new("low".into(), lows);
    let s_volume = Series::new("volume".into(), volumes);
    let s_fast = Series::new("ema_fast".into(), ema_fast);
    let s_slow = Series::new("ema_slow".into(), ema_slow);

    let df = DataFrame::new(vec![s_close, s_high, s_low, s_volume, s_fast, s_slow])?;
    let lf = df.lazy();

    // 2. Vectorización Masiva (Generación de Features y Señales SIMD)
    let signals_lf = lf
        .with_columns(vec![
            (col("close") - col("close").shift(lit(1))).alias("delta"),
            (col("ema_fast") - col("ema_slow")).alias("ema_diff"),
        ])
        .with_columns(vec![
            // Señal Long: EMA Rápida > Lenta con Momentum y Delta Positivo
            (col("ema_fast")
                .gt(col("ema_slow"))
                .and(col("delta").gt(lit(cfg.trend_threshold * 0.5)))
                .and(col("ema_diff").gt(lit(0.0))))
            .alias("signal_long"),
            // Señal Short: EMA Rápida < Lenta con Momentum y Delta Negativo
            (col("ema_fast")
                .lt(col("ema_slow"))
                .and(col("delta").lt(lit(-cfg.trend_threshold * 0.5)))
                .and(col("ema_diff").lt(lit(0.0))))
            .alias("signal_short"),
        ]);

    let result_df = signals_lf.collect()?;

    // Extraemos las columnas calculadas en paralelo a slices crudos para el bucle path-dependent
    let sig_long_ca = result_df.column("signal_long")?.bool()?;
    let sig_short_ca = result_df.column("signal_short")?.bool()?;

    // 3. Ejecución Híbrida Path-Dependent (SL/TP) en O(N) nativo
    let mut capital = if initial_capital.is_finite() && initial_capital > 0.0 {
        initial_capital
    } else {
        13.0
    };
    let mut position = 0; // 0 = flat, 1 = long, -1 = short
    let mut entry_price = 0.0;
    let mut position_size = 0.0; // in coins
    // D-669: margen BLOQUEADO por la posición abierta. `capital` sigue siendo
    // el patrimonio total; `capital - used_margin` es lo realmente disponible
    // para abrir. Antes esta distinción no existía y el apalancamiento salía
    // gratis.
    let mut used_margin = 0.0f64;
    let mut entry_margin = 0.0f64;
    let mut peak_capital = capital;
    let mut max_dd = 0.0;
    let mut wins = 0;
    let mut trades = 0;

    // FIX #621 & #1515: Acotamiento inferior y verificación de finitud de TP y SL
    let tp_ratio = if cfg.scalp_tp_base.is_finite() && cfg.scalp_tp_base > 0.0 {
        cfg.scalp_tp_base.clamp(0.001, 0.50)
    } else {
        0.005
    };
    let sl_ratio = if cfg.scalp_sl_base.is_finite() && cfg.scalp_sl_base > 0.0 {
        cfg.scalp_sl_base.clamp(0.001, 0.50)
    } else {
        0.002
    };

    // F3.4/F4.5 — LOS COSTOS NO SE EVOLUCIONAN: fee con PISO realista
    let taker_fee = if cfg.max_fee_pct.is_finite() && cfg.max_fee_pct >= 0.0 {
        cfg.max_fee_pct.clamp(0.0004, 0.001)
    } else {
        0.0005
    };

    // FIX #942: Modelo de slippage dinámico integrado (Kyle Impact Model)
    let slippage_model = OrderBookL2DepthSlippageModel::default();

    for i in 1..len {
        // Ignoramos i=0 por los shifts
        let c = closes[i];
        let h = highs[i];
        let l = lows[i];

        // FIX #657: Saltar barras con precios corruptos o no finitos
        if !c.is_finite() || c <= 0.0 || !h.is_finite() || !l.is_finite() {
            continue;
        }

        // FIX #1301: Eliminar Look-Ahead Bias leyendo la señal cerrada en la vela anterior (i-1)
        // La señal generada en close[i-1] se ejecuta en la vela actual [i]
        let sl_long = if i > 0 {
            sig_long_ca.get(i - 1).unwrap_or(false)
        } else {
            false
        };
        let sl_short = if i > 0 {
            sig_short_ca.get(i - 1).unwrap_or(false)
        } else {
            false
        };

        // Simulación de Funding Rates cada 480 velas (≈8h en TF 1m).
        // F4.5: sensibilidad del genoma acotada a banda REALISTA [0.5, 2.0]×
        // (0.005%-0.02% por periodo de 8h) — el costo de carry no se evoluciona a cero.
        // D-671 (DÉCIMA OLA) — EL FUNDING TIENE SIGNO.
        //
        // Antes se restaba SIEMPRE, para largos y para cortos. En futuros
        // perpetuos el funding es una TRANSFERENCIA entre lados: con tasa
        // positiva los largos pagan y los cortos COBRAN. Penalizar a ambos
        // enseñaba a la evolución a evitar los cortos por una razón inexistente
        // — y en un mercado bajista le impedía descubrir la estrategia correcta.
        //
        // D-675: el período deja de presuponer velas de 1 minuto. Se deriva del
        // espaciado real de las barras, que el llamador conoce.
        if bars_per_funding > 0 && i % bars_per_funding == 0 && position != 0 {
            let notional = entry_price * position_size;
            let funding_rate = cfg.funding_rate_sensitivity.clamp(0.5, 2.0) * 0.0001;
            // position = +1 (largo) paga; -1 (corto) cobra.
            capital -= notional * funding_rate * (position as f64);
        }

        // Manejar posición abierta (Path Dependency con modelado realista de Liquidación)
        let maint_margin_rate = 0.005; // 0.5% Binance Futures maintenance margin
        let leverage = cfg.global_leverage.clamp(1.0, 100.0);

        if position == 1 {
            // FIX #578: Cálculo analítico del precio de liquidación Binance
            let liq_drop = (1.0 / leverage) - maint_margin_rate;
            let liq_price = entry_price * (1.0 - liq_drop.max(0.005));
            let hit_liq = l <= liq_price;

            let tp_price = entry_price * (1.0 + tp_ratio);
            let sl_price = entry_price * (1.0 - sl_ratio);

            let hit_tp = h >= tp_price;
            let hit_sl = l <= sl_price;

            if hit_liq {
                // Liquidación forzosa por el Exchange
                // D-669: lo que se pierde en la liquidación es el margen
                // efectivamente bloqueado, no una reconstrucción a partir del
                // nocional (que ignoraba el capital libre real).
                let margin_lost = entry_margin;
                let liquidation_fee = (liq_price * position_size) * (taker_fee * 1.5);
                capital = (capital - margin_lost - liquidation_fee).max(0.0);
                // D-669: devolver el margen bloqueado al capital disponible.
                used_margin = 0.0;
                entry_margin = 0.0;
                position = 0;
            } else if hit_sl {
                // SL hit (priorizado en barras de doble ruptura para eliminar sesgo optimista)
                let exit_fee = (sl_price * position_size) * taker_fee;
                let pnl = (sl_price - entry_price) * position_size;
                capital = (capital + pnl - exit_fee).max(0.0);
                // D-669: devolver el margen bloqueado al capital disponible.
                used_margin = 0.0;
                entry_margin = 0.0;
                position = 0;
            } else if hit_tp {
                // TP hit
                let exit_fee = (tp_price * position_size) * taker_fee;
                let pnl = (tp_price - entry_price) * position_size;
                capital += pnl - exit_fee;
                wins += 1;
                // D-669: devolver el margen bloqueado al capital disponible.
                used_margin = 0.0;
                entry_margin = 0.0;
                position = 0;
            } else if sl_short {
                // FIX #1302: Aplicar slippage de salida por reversión de posición (Long vende en Bid)
                let notional = position_size * c;
                let slip_pct = slippage_model.compute_slippage_pct(notional);
                let exit_price = c * (1.0 - slip_pct);
                let exit_fee = (exit_price * position_size) * taker_fee;
                let pnl = (exit_price - entry_price) * position_size;
                capital = (capital + pnl - exit_fee).max(0.0);
                if pnl > exit_fee {
                    wins += 1;
                }
                // D-669: devolver el margen bloqueado al capital disponible.
                used_margin = 0.0;
                entry_margin = 0.0;
                position = 0;
            }
        } else if position == -1 {
            // FIX #578: Cálculo analítico del precio de liquidación Binance Short
            let liq_rise = (1.0 / leverage) - maint_margin_rate;
            let liq_price = entry_price * (1.0 + liq_rise.max(0.005));
            let hit_liq = h >= liq_price;

            let tp_price = entry_price * (1.0 - tp_ratio);
            let sl_price = entry_price * (1.0 + sl_ratio);

            let hit_tp = l <= tp_price;
            let hit_sl = h >= sl_price;

            if hit_liq {
                // Liquidación forzosa por el Exchange
                // D-669: lo que se pierde en la liquidación es el margen
                // efectivamente bloqueado, no una reconstrucción a partir del
                // nocional (que ignoraba el capital libre real).
                let margin_lost = entry_margin;
                let liquidation_fee = (liq_price * position_size) * (taker_fee * 1.5);
                capital = (capital - margin_lost - liquidation_fee).max(0.0);
                // D-669: devolver el margen bloqueado al capital disponible.
                used_margin = 0.0;
                entry_margin = 0.0;
                position = 0;
            } else if hit_sl {
                // SL hit (priorizado en barras de doble ruptura para eliminar sesgo optimista)
                let exit_fee = (sl_price * position_size) * taker_fee;
                let pnl = (entry_price - sl_price) * position_size;
                capital = (capital + pnl - exit_fee).max(0.0);
                // D-669: devolver el margen bloqueado al capital disponible.
                used_margin = 0.0;
                entry_margin = 0.0;
                position = 0;
            } else if hit_tp {
                // TP hit
                let exit_fee = (tp_price * position_size) * taker_fee;
                let pnl = (entry_price - tp_price) * position_size;
                capital += pnl - exit_fee;
                wins += 1;
                // D-669: devolver el margen bloqueado al capital disponible.
                used_margin = 0.0;
                entry_margin = 0.0;
                position = 0;
            } else if sl_long {
                // FIX #1302: Aplicar slippage de salida por reversión de posición (Short compra en Ask)
                let notional = position_size * c;
                let slip_pct = slippage_model.compute_slippage_pct(notional);
                let exit_price = c * (1.0 + slip_pct);
                let exit_fee = (exit_price * position_size) * taker_fee;
                let pnl = (entry_price - exit_price) * position_size;
                capital = (capital + pnl - exit_fee).max(0.0);
                if pnl > exit_fee {
                    wins += 1;
                }
                // D-669: devolver el margen bloqueado al capital disponible.
                used_margin = 0.0;
                entry_margin = 0.0;
                position = 0;
            }
        }

        // 4. Apertura de Posiciones (Enforce MIN_NOTIONAL $5.00)
        // FIX #700: Reservar margen libre para comisión de entrada en microcuentas de $13 USD
        if position == 0 && (sl_long || sl_short) {
            // D-669 (DÉCIMA OLA) — EL BACKTEST RESERVA MARGEN.
            //
            // Antes al abrir sólo se descontaba la COMISIÓN: `capital` quedaba
            // íntegro y disponible mientras el PnL se calculaba sobre nocional
            // apalancado 30×. La recurrencia resultante era
            //
            //     cap(n+1) = cap(n) · (1 + 0,99·L·r(n))
            //
            // es decir, la curva de equity era el precio del activo compuesto
            // a 30× SIN colateral. Un movimiento favorable del 1 % producía un
            // 29,7 % de crecimiento. Y la rama de liquidación no compensaba
            // nada: con stops de 40–60 bps, el 2,83 % de la liquidación era
            // inalcanzable, de modo que el apalancamiento no tenía coste
            // simulado alguno.
            //
            // Combinado con D-653 (el fitness crece linealmente con el
            // apalancamiento) y D-644 (la mutación impedía bajar de 25×), el
            // sistema tenía TRES presiones hacia el apalancamiento máximo y
            // ninguna fuerza compensatoria.
            let leverage = cfg.global_leverage.clamp(1.0, 125.0);
            // El margen disponible es el capital LIBRE, no el total.
            let free_capital = (capital - used_margin).max(0.0);
            // Reserva del 1 % para la comisión de entrada.
            let margin = (free_capital * 0.99).max(0.0);
            let notional = margin * leverage;
            if margin > 0.0 && notional >= 5.0 && c > 0.0 && c.is_finite() {
                let slip_pct = slippage_model.compute_slippage_pct(notional);
                let entry_fee = notional * taker_fee;
                if entry_fee < free_capital {
                    position = if sl_long { 1 } else { -1 };
                    entry_price = if sl_long {
                        c * (1.0 + slip_pct)
                    } else {
                        c * (1.0 - slip_pct)
                    };
                    capital -= entry_fee;
                    // BLOQUEAR el margen: deja de estar disponible hasta cerrar.
                    used_margin = margin;
                    entry_margin = margin;
                    position_size = notional / entry_price;
                    trades += 1;
                }
            }
        }

        if capital > peak_capital {
            peak_capital = capital;
        }
        let dd = if peak_capital > 0.0 {
            (peak_capital - capital) / peak_capital
        } else {
            0.0
        };
        if dd > max_dd && dd.is_finite() {
            max_dd = dd;
        }

        if capital <= 0.0 {
            break;
        } // Margin Call
    }

    Ok((capital, max_dd, trades, wins))
}

/// Modelo de Slippage Dinámico basado en Reconstrucción de Profundidad L2 (#296-#305)
#[derive(Debug, Clone, Copy)]
pub struct OrderBookL2DepthSlippageModel {
    pub base_spread_bps: f64,
    pub depth_notional_l2: f64,
    pub market_impact_gamma: f64,
}

impl OrderBookL2DepthSlippageModel {
    pub fn new(base_spread_bps: f64, depth_notional_l2: f64, market_impact_gamma: f64) -> Self {
        Self {
            base_spread_bps: base_spread_bps.clamp(0.5, 50.0),
            depth_notional_l2: depth_notional_l2.max(100.0),
            market_impact_gamma: market_impact_gamma.clamp(0.01, 2.0),
        }
    }

    /// Calcula el slippage total (spread + impacto de mercado no lineal de Kyle)
    /// $\text{Slippage}(Q) = \frac{\text{Spread}}{2} + \gamma \sqrt{\frac{Q}{\text{Depth}}}$
    #[inline(always)]
    pub fn compute_slippage_pct(&self, order_notional: f64) -> f64 {
        if !order_notional.is_finite() || order_notional <= 0.0 {
            return (self.base_spread_bps * 0.0001) / 2.0;
        }
        let half_spread = (self.base_spread_bps * 0.0001) / 2.0;
        let depth_ratio = (order_notional / self.depth_notional_l2).clamp(0.0, 10.0);
        let impact = self.market_impact_gamma * depth_ratio.sqrt() * 0.001;
        (half_spread + impact).clamp(0.00005, 0.05)
    }

    /// Calcula la comisión efectiva combinada Maker/Taker según ejecución pasiva/agresiva
    #[inline(always)]
    pub fn compute_effective_fee(&self, order_notional: f64, is_post_only: bool) -> f64 {
        let rate = if is_post_only { 0.0002 } else { 0.0004 }; // 0.02% maker, 0.04% taker Binance Futures
        order_notional * rate
    }
}

impl Default for OrderBookL2DepthSlippageModel {
    fn default() -> Self {
        Self::new(2.0, 50000.0, 0.5)
    }
}

#[cfg(test)]
mod tests {
    /// D-669 (DÉCIMA OLA) — EL APALANCAMIENTO DEBE TENER COSTE.
    ///
    /// Antes el backtest no reservaba margen: `capital` quedaba íntegro tras
    /// abrir y el PnL se calculaba sobre nocional apalancado, de modo que la
    /// curva de equity era el precio compuesto a 30× sin colateral. Con tres
    /// presiones evolutivas hacia el apalancamiento máximo (D-653, D-644) y
    /// ninguna compensatoria, el genoma promovido era necesariamente el más
    /// apalancado que el sistema permitía.
    ///
    /// Contrato: en una serie adversa, más apalancamiento debe producir PEOR
    /// resultado. Es la fuerza compensatoria que faltaba.
    #[test]
    fn d669_mas_apalancamiento_penaliza_en_serie_adversa() {
        // Serie genuinamente adversa para un sistema bidireccional (long/short):
        // Falsos breakouts con reversiones bruscas que barren los Stop Losses en ambas direcciones.
        let n = 400usize;
        let mut closes = Vec::with_capacity(n);
        let mut highs = Vec::with_capacity(n);
        let mut lows = Vec::with_capacity(n);
        let mut vols = Vec::with_capacity(n);
        let mut p = 60_000.0f64;
        for i in 0..n {
            let phase = i % 20;
            if phase == 10 {
                p *= 0.988; // Ruptura falsa hacia abajo: barre el SL del Long
            } else if phase == 19 {
                p *= 1.012; // Ruptura falsa hacia arriba: barre el SL del Short
            } else if phase < 10 {
                p *= 1.001; // Impulso alcista suave que genera señal Long
            } else {
                p *= 0.999; // Impulso bajista suave que genera señal Short
            }
            closes.push(p);
            highs.push(p * 1.001);
            lows.push(p * 0.999);
            vols.push(1_000.0);
        }

        let run = |lev: f64| -> f64 {
            let mut cfg = SuperGenotype::new_baseline(0.0002, 0.0005);
            cfg.global_leverage = lev;
            let (cap, _dd, _t, _w) =
                run_vectorized_hybrid(&closes, &highs, &lows, &vols, &cfg).expect("backtest");
            cap
        };

        let cap_bajo = run(3.0);
        let cap_alto = run(50.0);
        assert!(
            cap_alto < cap_bajo,
            "el apalancamiento debe tener coste en serie adversa: 50x dejó {cap_alto}              y 3x dejó {cap_bajo} — si 50x no es peor, el margen no se está reservando"
        );
    }

    /// D-669: el margen bloqueado impide abrir con más capital del que hay.
    /// El capital final nunca puede superar lo que el nocional permite.
    #[test]
    fn d669_el_capital_nunca_es_negativo_ni_infinito() {
        let n = 2_000usize;
        let closes: Vec<f64> = (0..n).map(|i| 60_000.0 * (1.0 + 0.0001 * i as f64)).collect();
        let highs: Vec<f64> = closes.iter().map(|c| c * 1.001).collect();
        let lows: Vec<f64> = closes.iter().map(|c| c * 0.999).collect();
        let vols = vec![1_000.0; n];
        let cfg = SuperGenotype::new_baseline(0.0002, 0.0005);
        let (cap, dd, _t, _w) =
            run_vectorized_hybrid(&closes, &highs, &lows, &vols, &cfg).expect("backtest");
        assert!(cap.is_finite() && cap >= 0.0, "capital inválido: {cap}");
        assert!((0.0..=1.0).contains(&dd), "drawdown fuera de rango: {dd}");
    }

    use super::*;

    #[test]
    fn test_vectorized_hybrid_backtest() {
        let n = 50;
        let mut closes = Vec::with_capacity(n);
        let mut highs = Vec::with_capacity(n);
        let mut lows = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);

        let mut p = 60000.0;
        for i in 0..n {
            p += (i as f64 * 0.2).sin() * 20.0;
            closes.push(p);
            highs.push(p + 10.0);
            lows.push(p - 10.0);
            volumes.push(50.0);
        }

        let cfg = SuperGenotype::default();
        let res = run_vectorized_hybrid(&closes, &highs, &lows, &volumes, &cfg);
        assert!(res.is_ok());
        let (cap, max_dd, _, _) = res.unwrap();
        assert!(cap > 0.0);
        assert!(max_dd >= 0.0 && max_dd <= 1.0);

        let slip_model = OrderBookL2DepthSlippageModel::default();
        let fee = slip_model.compute_effective_fee(13.0, false);
        assert!(fee > 0.0);
    }

    #[test]
    fn test_orderbook_slippage_nan_immunity() {
        let slip_model = OrderBookL2DepthSlippageModel::default();
        let slip_nan = slip_model.compute_slippage_pct(f64::NAN);
        assert!(slip_nan.is_finite() && slip_nan > 0.0);

        let slip_inf = slip_model.compute_slippage_pct(f64::INFINITY);
        assert!(slip_inf.is_finite() && slip_inf > 0.0);

        let slip_neg = slip_model.compute_slippage_pct(-50.0);
        assert!(slip_neg.is_finite() && slip_neg > 0.0);
    }

    #[test]
    fn test_multi_coin_event_priority() {
        let t = 1700000000000000000;
        let p_stop = compute_multi_coin_event_priority(t, true);
        let p_normal = compute_multi_coin_event_priority(t, false);
        assert!(
            p_stop < p_normal,
            "Stop loss/liquidación debe tener mayor prioridad (menor valor)"
        );
    }

    #[test]
    fn test_multi_coin_event_priority_ordering() {
        let t1 = 1000;
        let t2 = 2000;

        let p_t1_normal = compute_multi_coin_event_priority(t1, false);
        let p_t2_stop = compute_multi_coin_event_priority(t2, true);

        // Eventos anteriores en el tiempo siempre preceden a eventos futuros sin importar el tipo
        assert!(
            p_t1_normal < p_t2_stop,
            "Causalidad temporal estricta garantizada"
        );
    }

    #[test]
    fn test_orderbook_slippage_massive_volume_bound() {
        let slip_model = OrderBookL2DepthSlippageModel::default();
        let slip_100k = slip_model.compute_slippage_pct(100_000.0);
        let slip_micro = slip_model.compute_slippage_pct(13.0);

        assert!(
            slip_100k > slip_micro,
            "Órdenes grandes deben sufrir mayor slippage por impacto de mercado"
        );
        assert!(
            slip_100k <= 0.05,
            "Slippage debe permanecer acotado al hardcap"
        );
    }

    #[test]
    fn test_run_vectorized_hybrid_empty_and_nan_inputs() {
        let cfg = SuperGenotype::default();
        let empty_closes: Vec<f64> = vec![];
        let empty_highs: Vec<f64> = vec![];
        let empty_lows: Vec<f64> = vec![];
        let empty_vols: Vec<f64> = vec![];

        let res_empty =
            run_vectorized_hybrid(&empty_closes, &empty_highs, &empty_lows, &empty_vols, &cfg);
        assert!(res_empty.is_ok());
        let (cap, max_dd, wins, trades) = res_empty.unwrap();
        assert_eq!(trades, 0);
        assert_eq!(wins, 0);
        assert_eq!(max_dd, 0.0);
        assert!(cap >= 0.0);
    }

    #[test]
    fn test_run_vectorized_hybrid_synthetic_trend() {
        let cfg = SuperGenotype::default();
        let n = 50;
        let mut closes = Vec::with_capacity(n);
        let mut highs = Vec::with_capacity(n);
        let mut lows = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);

        let mut p = 100.0;
        for i in 0..n {
            p += (i as f64 * 0.2).sin() * 2.0;
            closes.push(p);
            highs.push(p + 1.0);
            lows.push(p - 1.0);
            volumes.push(50.0);
        }

        let res = run_vectorized_hybrid(&closes, &highs, &lows, &volumes, &cfg);
        assert!(res.is_ok());
        let (cap, max_dd, _wins, _trades) = res.unwrap();
        assert!(cap > 0.0, "Capital final debe ser positivo");
        assert!(max_dd >= 0.0, "Max drawdown no puede ser negativo");
    }
}

/// Cola de prioridad de eventos temporales multi-moneda (Puntos #301 y #305)
#[inline(always)]
pub fn compute_multi_coin_event_priority(timestamp_ns: u64, is_liquidation_or_stop: bool) -> u64 {
    let base = timestamp_ns << 1;
    if is_liquidation_or_stop {
        base // Mayor prioridad (menor valor)
    } else {
        base | 1
    }
}
