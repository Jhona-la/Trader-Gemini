//! REPLAY DE BOOK-TICKS REALES (REHAB-2b / X-006) — el motor honesto del GA.
//!
//! QUÉ: reproduce el núcleo de `audit_forensic_backtest` como función de
//!      biblioteca para que la EVOLUCIÓN promocione genomas evaluados contra
//!      microestructura REAL y el MISMO camino de entrada que producción.
//! POR QUÉ (hallazgo X-006): el GA seleccionaba contra micro-ticks
//!      sintéticos (puente browniano al cierre conocido, OFI fabricado,
//!      omni congelado, timestamps uniformes, maker flag inventado) — el
//!      "edge" evolucionado era ajuste a artefactos de la síntesis.
//! FIDELIDAD con el forense y con producción:
//!      - cada tick de disco (ts, bid, ask, bq, aq) → DOS eventos del core
//!        (depth + trade con maker heurístico price<=bid) — idéntico al
//!        forense; el core es el MISMO GodEngineCore::process_event de live.
//!      - slippage penalizado por ATR real (no fijo).
//!      - omni: 6 series FRED históricas reales por fecha (step-function) —
//!        las mismas de producción (F4.1); sin red ⇒ omni neutro documentado.
//!      - fees: los del genoma/arena (ya piso-realistas post-F3.4).
//!      - sin relojes falsos: event_time = ts REAL del tick de disco.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use quantum_arena::genome::SuperGenotype;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Tick de libro en disco (repr exacto de *_ticks.bin de parquet_to_bin).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ReplayTick {
    pub ts_ms: u64,
    pub bid: f64,
    pub ask: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

/// Historia macro FRED: series (día-desde-epoch, valor) por índice de slot.
pub struct OmniHistory {
    series: [Vec<(i64, f64)>; 6],
}

impl OmniHistory {
    /// Descarga las 6 series FRED (histórico COMPLETO). Falla suave: None si
    /// no hay red — el replay corre con omni neutro (documentado en stats).
    pub fn fetch() -> Option<OmniHistory> {
        let ids = [
            "SP500",
            "NASDAQCOM",
            "VIXCLS",
            "DGS10",
            "DTWEXBGS",
            "DCOILWTICO",
        ];
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .ok()?;
        let mut series: [Vec<(i64, f64)>; 6] = Default::default();
        for (i, id) in ids.iter().enumerate() {
            let url = format!("https://fred.stlouisfed.org/graph/fredgraph.csv?id={}", id);
            let csv = client.get(&url).send().ok()?.text().ok()?;
            let mut parsed: Vec<(i64, f64)> = csv
                .lines()
                .skip(1)
                .filter_map(|l| {
                    let mut p = l.split(',');
                    let d = p.next()?.trim();
                    let v = p.next()?.trim().parse::<f64>().ok()?;
                    // fecha → días desde epoch ( YYYY-MM-DD )
                    let mut it = d.split('-');
                    let y: i64 = it.next()?.parse().ok()?;
                    let m: i64 = it.next()?.parse().ok()?;
                    let dd: i64 = it.next()?.parse().ok()?;
                    // días civiles → epoch (Hinnant, sin deps)
                    let yy = if m <= 2 { y - 1 } else { y };
                    let era = if yy >= 0 { yy } else { yy - 399 } / 400;
                    let yoe = yy - era * 400;
                    let mp = (m + 9) % 12;
                    let doy = (153 * mp + 2) / 5 + dd - 1;
                    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
                    let days = era * 146_097 + doe - 719_468;
                    Some((days, v))
                })
                .collect();
            parsed.sort_by_key(|(d, _)| *d);
            series[i] = parsed;
        }
        Some(OmniHistory { series })
    }

    fn value_at(&self, slot: usize, day: i64) -> f64 {
        let s = &self.series[slot];
        match s.binary_search_by_key(&day, |&(d, _)| d) {
            Ok(i) => s[i].1,
            Err(0) => s.first().map(|v| v.1).unwrap_or(0.0),
            Err(i) => s[i - 1].1,
        }
    }

    fn day_of(ts_ms: u64) -> i64 {
        (ts_ms / 86_400_000) as i64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReplayConfig {
    pub initial_capital: f64,
    /// Ticks de calentamiento de features (sin evaluación de PnL).
    pub warmup_ticks: usize,
    /// MODO TRADE-ONLY: no enviar eventos depth con bid/ask sintético (las
    /// features de microestructura del motor se diseñaron para bookTicker
    /// REAL; con bid/ask derivado de trades producen OBI/OFI plano/artificial
    /// que contamina las señales). En trade-only, SOLO se envía el evento de
    /// trade (is_trade=true) — las features de precio/volumen/ATR/Hurst/
    /// espectral funcionan correctamente sin el libro.
    pub trade_only: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            initial_capital: 13.0,
            warmup_ticks: 200,
            trade_only: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReplayStats {
    pub trades: u64,
    pub wins_net: u64,
    pub wins_gross: u64,
    pub net_pnl: f64,
    pub gross_pnl: f64,
    pub fees_est: f64,
    pub final_capital: f64,
    pub max_dd: f64,
    /// Sharpe por trade (media/σ de PnL neto).
    pub sharpe: f64,
    /// true si el omni fue neutro (sin red al cargar FRED).
    pub omni_neutral: bool,
}

impl ReplayStats {
    pub fn wr_net(&self) -> f64 {
        if self.trades > 0 {
            self.wins_net as f64 / self.trades as f64
        } else {
            0.0
        }
    }
    pub fn roi_net(&self, capital: f64) -> f64 {
        if capital > 0.0 {
            self.net_pnl / capital
        } else {
            0.0
        }
    }
}

/// Evalúa un genoma contra ticks REALES con el core de producción.
/// Un solo símbolo (coin 0) — el GA por-símbolo evoluciona contra SU data.
pub fn run_booktick_replay(
    ticks: &[ReplayTick],
    genome: &SuperGenotype,
    omni: Option<&OmniHistory>,
    cfg: &ReplayConfig,
) -> ReplayStats {
    let mut stats = ReplayStats::default();
    if ticks.len() <= cfg.warmup_ticks + 10
        || !cfg.initial_capital.is_finite()
        || cfg.initial_capital <= 0.0
    {
        return stats;
    }

    let arena = Arc::new(GlobalArena::new(cfg.initial_capital));
    genome.apply_to_arena(&arena);
    let mut core = GodEngineCore::new(arena.clone());

    // FIX AUDIT: Hurst necesita 512 cierres 1m para producir valores ≠ 0.5.
    // Sin esto, el estimador DFA queda clavado en neutral y TODO el canal
    // price-action (que depende de hurst > 0.52 o < 0.45) es inalcanzable.
    // Sintetizamos los klines 1m agregando ticks por minutos:
    let mut last_minute = 0u64;
    let mut minute_open = 0.0f64;
    let mut minute_high = 0.0f64;
    let mut minute_low = f64::MAX;
    let mut minute_close = 0.0f64;
    let mut minute_vol = 0.0f64;
    for t in ticks.iter().take(cfg.warmup_ticks.max(600).min(ticks.len())) {
        let minute = t.ts_ms / 60_000;
        if minute != last_minute && last_minute > 0 {
            // Cerrar el kline anterior
            core.feature_engines[0].process_kline(
                minute_open,
                minute_high,
                minute_low,
                minute_close,
                minute_vol,
            );
            minute_high = 0.0;
            minute_low = f64::MAX;
            minute_vol = 0.0;
        }
        let mid = t.mid();
        if minute != last_minute {
            last_minute = minute;
            minute_open = mid;
        }
        minute_high = minute_high.max(mid);
        minute_low = minute_low.min(mid);
        minute_close = mid;
        minute_vol += t.bid_qty + t.ask_qty;
    }

    // LECCIÓN DE CARRERA (golden X-test): las funciones de biblioteca NO
    // mutan estado global (symbol_registry) — los tests corren en paralelo
    // en el mismo proceso y el golden mide determinismo. El LLAMADOR registra
    // specs si las necesita (los bins lo hacen; sin registro, try_spec=None
    // y el core usa sus floors internos — replay válido igualmente).

    let maker_fee = arena
        .config
        .live_maker_fee
        .load(Ordering::Relaxed)
        .max(0.0002);
    let taker_fee = arena
        .config
        .live_taker_fee
        .load(Ordering::Relaxed)
        .max(0.0004);
    let roundtrip_fee = maker_fee + taker_fee;

    let omni_state = data_pipeline::omni_multiplexer::OmniState::new();
    let mut last_day = i64::MIN;
    let mut running_atr = 0.001 * ticks[0].mid();
    let mut prev_mid = ticks[0].mid();
    const ATR_ALPHA: f64 = 0.02;

    let mut pnl_list: Vec<f64> = Vec::new();
    let mut peak = cfg.initial_capital;
    let warmup = cfg.warmup_ticks.min(ticks.len() / 10);

    for (i, t) in ticks.iter().enumerate() {
        let mid = t.mid();
        if !mid.is_finite() || mid <= 0.0 || t.bid <= 0.0 || t.ask <= 0.0 || t.bid > t.ask {
            continue; // aduana F2.1 (misma regla que producción)
        }

        // ATR real del stream (TR = rango efectivo del tick).
        // FIX AUDIT: la fórmula anterior `max(spread, mid - running_atr)` era
        // un NIVEL DE PRECIO ($63K), no un rango → running_atr divergía a
        // mid/2 ≈ $31,500 → slippage del 5% por lado. TR correcto: el spread
        // o el cambio absoluto del precio vs el tick anterior.
        let tr = (t.ask - t.bid).max((mid - prev_mid).abs());
        running_atr = ATR_ALPHA * tr + (1.0 - ATR_ALPHA) * running_atr;
        prev_mid = mid;
        // Slippage institucional: castigo de fills según ATR vivo.
        let slip = running_atr * 0.10;
        let sim_bid = t.bid - slip;
        let sim_ask = t.ask + slip;
        let vol = t.bid_qty + t.ask_qty; // liquidez del libro (doc: no volumen operado)
        let obi = if vol > 0.0 {
            (t.bid_qty - t.ask_qty) / vol
        } else {
            0.0
        };

        // omni REAL por día (FRED); sin red ⇒ neutro.
        if let Some(hist) = omni {
            let day = OmniHistory::day_of(t.ts_ms);
            if day != last_day {
                last_day = day;
                omni_state
                    .sp500
                    .store(hist.value_at(0, day).to_bits(), Ordering::Relaxed);
                omni_state
                    .nasdaq
                    .store(hist.value_at(1, day).to_bits(), Ordering::Relaxed);
                omni_state
                    .vix
                    .store(hist.value_at(2, day).to_bits(), Ordering::Relaxed);
                omni_state
                    .us10y
                    .store(hist.value_at(3, day).to_bits(), Ordering::Relaxed);
                omni_state
                    .dxy
                    .store(hist.value_at(4, day).to_bits(), Ordering::Relaxed);
                omni_state
                    .oil_wti
                    .store(hist.value_at(5, day).to_bits(), Ordering::Relaxed);
            }
        }
        let omni_features = omni_state.get_features();

        let is_minute_kline =
            i > 0 && OmniHistory::day_of(ticks[i - 1].ts_ms) != OmniHistory::day_of(t.ts_ms);
        // (kline-close aproximado por frontera de día: suficiente para
        // calibración del ensamble en replay; los klines 1m reales viven en
        // producción por WS.)

        let (c1, c2): (Option<(bool, f64, f64)>, Option<(bool, f64, f64)>);
        if cfg.trade_only {
            // MODO TRADE-ONLY (para datos de trades/aggTrades sin libro real):
            // UN solo evento de trade — el precio/volumen del trade alimenta
            // las features de precio (ATR, momentum, Hurst, espectral) sin
            // contaminar OBI/OFI con bid/ask sintético.
            let (o2, closed) = core.process_event(
                0,
                true,          // is_trade
                is_minute_kline,
                false,         // is_depth
                mid,           // precio del trade
                vol,           // volumen
                mid * 0.9999,  // bid ~ precio (sin libro real)
                mid * 1.0001,  // ask ~ precio
                vol * 0.5,     // qty neutra (sin libro)
                vol * 0.5,
                0.0,           // OBI neutro
                0.0,           // micro_div neutro
                t.ts_ms,
                false,
                &omni_features,
                t.bid_qty > t.ask_qty, // maker heurístico del propio dato
            );
            let _ = o2;
            c1 = None;
            c2 = closed;
        } else {
            // Doble evento — patrón del forense (depth + trade).
            let (o1, closed1) = core.process_event(
                0,
                false,
                is_minute_kline,
                true,
                mid,
                vol,
                sim_bid,
                sim_ask,
                t.bid_qty,
                t.ask_qty,
                obi,
                0.0,
                t.ts_ms,
                false,
                &omni_features,
                false,
            );
            let maker_flag = mid <= sim_bid;
            let (o2, closed2) = core.process_event(
                0,
                true,
                false,
                false,
                mid,
                vol,
                sim_bid,
                sim_ask,
                t.bid_qty,
                t.ask_qty,
                obi,
                0.0,
                t.ts_ms,
                false,
                &omni_features,
                maker_flag,
            );
            let _ = (o1, o2);
            c1 = closed1;
            c2 = closed2;
        }

        if i >= warmup {
            if let Some((_, pnl_net, qty)) = c1.or(c2) {
                let notional = qty * mid;
                let fee = notional * roundtrip_fee;
                stats.trades += 1;
                stats.net_pnl += pnl_net; // el core ya cobra fees físicos
                stats.fees_est += fee;
                stats.gross_pnl += pnl_net + fee;
                if pnl_net > 0.0 {
                    stats.wins_net += 1;
                }
                if pnl_net + fee > 0.0 {
                    stats.wins_gross += 1;
                }
                pnl_list.push(pnl_net);
            }
        }

        let cap = arena.unified_capital.load(Ordering::Relaxed);
        if cap > peak {
            peak = cap;
        }
        let dd = if peak > 0.0 { (peak - cap) / peak } else { 0.0 };
        if dd > stats.max_dd {
            stats.max_dd = dd;
        }
        if cap <= 0.0 {
            break;
        }
    }

    stats.final_capital = arena.unified_capital.load(Ordering::Relaxed);
    if !pnl_list.is_empty() {
        let mean = stats.net_pnl / pnl_list.len() as f64;
        let var = pnl_list
            .iter()
            .map(|p| (p - mean) * (p - mean))
            .sum::<f64>()
            / pnl_list.len().max(1) as f64;
        let sd = var.sqrt();
        stats.sharpe = if sd > 1e-12 { mean / sd } else { 0.0 };
    }
    stats.omni_neutral = omni.is_none();
    stats
}

impl ReplayTick {
    #[inline]
    pub fn mid(&self) -> f64 {
        (self.bid + self.ask) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_ticks(n: usize) -> Vec<ReplayTick> {
        // Caminata determinista suave + spread — SIN falsa directionalidad.
        let mut seed: u64 = 7;
        let mut p = 60_000.0f64;
        (0..n)
            .map(|i| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
                p *= 1.0 + noise * 0.0004;
                ReplayTick {
                    ts_ms: 1_700_000_000_000 + i as u64 * 100,
                    bid: p - 0.5,
                    ask: p + 0.5,
                    bid_qty: 1.0 + noise.abs(),
                    ask_qty: 1.0 + (1.0 - noise.abs()),
                }
            })
            .collect()
    }

    #[test]
    fn replay_sano_es_determinista_y_no_explota() {
        let ticks = synth_ticks(20_000);
        let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
        let cfg = ReplayConfig {
            initial_capital: 1000.0,
            warmup_ticks: 200,
        };
        let a = run_booktick_replay(&ticks, &genome, None, &cfg);
        let b = run_booktick_replay(&ticks, &genome, None, &cfg);
        // Determinismo bit a bit del motor (mismo input, mismo core).
        assert_eq!(a.trades, b.trades);
        assert!((a.net_pnl - b.net_pnl).abs() < 1e-12);
        // Sanity: capital no NaN/inf y DD acotado en ruido suave.
        assert!(a.final_capital.is_finite());
        assert!(a.max_dd < 1.0);
    }
}
