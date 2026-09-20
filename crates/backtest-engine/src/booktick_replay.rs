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
//!      - B3.19 SIZING: cada entrada pasa por la MISMA envolvente Kelly
//!        bayesiana del host (D-442 stop real + D-116 bootstrap/autoridad +
//!        D-382 leverage-adapt/margin-guard) — ver `live_envelope_gate`.
//!
//! SEMÁNTICA DE SL (auditoría B3.19, tarea 2): el exit del core se evalúa
//! en CADA `process_event` (depth y trade — dos veces por tick en modo
//! doble): `pnl_pct <= -sl` corta y el fill se ACOTA a `sl_price`
//! (`mid.min(sl)` para long) ANTES de la física; `calculate_exit` después
//! empeora el fill taker (slippage + latencia), igual que un bracket
//! stop-market real resbala en el exchange. Los ZOMBIE tampoco exceden el
//! SL en bruto (mismo cap). Única diferencia estructural con el vivo: en
//! huecos SIN ticks el replay no puede cortar (no hay dato) mientras el
//! bracket del exchange sí dispararía — inherente al replay por datos, no
//! un hueco de código (no se "corrige": sería inventar precios).

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use quantum_arena::genome::SuperGenotype;
use risk_engine::kelly_envelope::RiskEnvelope;
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
    /// Descarga las 6 series FRED (histórico COMPLETO). B3.16 — FALLBACK A
    /// ARCHIVO: con FRED bloqueado (fingerprint TLS de esta red), el replay
    /// corría con omni NEUTRO mientras los modelos entrenaron con macro
    /// REAL — los splits 44-47 evaluaban en 0. Si la red falla, se leen
    /// data/macro/{SP500,NASDAQ,VIX,DXY}.csv (macro_history_sync, Yahoo
    /// v8: mismos cierres). DGS10/OIL quedan neutros (sin fuente local).
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
        let mut net_ok = true;
        for (i, id) in ids.iter().enumerate() {
            let url = format!("https://fred.stlouisfed.org/graph/fredgraph.csv?id={}", id);
            match client.get(&url).send().and_then(|r| r.text()) {
                Ok(csv) => {
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
                Err(_) => {
                    net_ok = false;
                    continue;
                }
            }
        }
        if !net_ok {
            // Fallback local: (slot FRED, tag de archivo en data/macro).
            let local: [(usize, &str); 4] = [(0, "SP500"), (1, "NASDAQ"), (2, "VIX"), (4, "DXY")];
            for (slot, tag) in local {
                if series[slot].is_empty() {
                    if let Ok(content) = std::fs::read_to_string(format!("data/macro/{tag}.csv")) {
                        let parsed: Vec<(i64, f64)> = content
                            .lines()
                            .skip(1)
                            .filter_map(|l| {
                                let mut p = l.split(',');
                                let ms: i64 = p.next()?.trim().parse().ok()?;
                                let v: f64 = p.next()?.trim().parse().ok()?;
                                Some((ms / 86_400_000, v))
                            })
                            .collect();
                        if !parsed.is_empty() {
                            series[slot] = parsed;
                        }
                    }
                }
            }
            let filled = series.iter().filter(|s| !s.is_empty()).count();
            println!(
                "   📎 FRED sin red — omni desde data/macro (Yahoo): {filled}/6 series reales, resto neutro"
            );
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
    /// B3.19 — entradas VETADAS por la envolvente D-442/margin-guards
    /// (rollback inmediato, paridad con los aborts del host en vivo).
    pub envelope_vetoes: u64,
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

    let arena = GlobalArena::build_in_own_stack(cfg.initial_capital);
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

    // B3.19 — envolvente del host replicada (D-442/D-116/D-382): misma
    // maquinaria que god_engine mantiene vivo entre eventos.
    let mut risk_envelope = RiskEnvelope::new();
    let mut avg_win_abs = 0.0f64;
    let mut avg_loss_abs = 0.0f64;
    let mut pos_was_open = arena.coins[0].positions.position.is_open();

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

        // omni REAL por día; sin red ⇒ neutro. B3.16 — corte t-1: el MISMO
        // día NO (su cierre no existe intradía — lookahead contra el
        // contrato del trainer y del poller vivo).
        if let Some(hist) = omni {
            let day = OmniHistory::day_of(t.ts_ms);
            if day != last_day {
                last_day = day;
                let prev_day = (day - 1).max(0);
                omni_state
                    .sp500
                    .store(hist.value_at(0, prev_day).to_bits(), Ordering::Relaxed);
                omni_state
                    .nasdaq
                    .store(hist.value_at(1, prev_day).to_bits(), Ordering::Relaxed);
                omni_state
                    .vix
                    .store(hist.value_at(2, prev_day).to_bits(), Ordering::Relaxed);
                omni_state
                    .us10y
                    .store(hist.value_at(3, prev_day).to_bits(), Ordering::Relaxed);
                omni_state
                    .dxy
                    .store(hist.value_at(4, prev_day).to_bits(), Ordering::Relaxed);
                omni_state
                    .oil_wti
                    .store(hist.value_at(5, prev_day).to_bits(), Ordering::Relaxed);
            }
        }
        let omni_features = omni_state.get_features();

        // D-705/D-708 (DÉCIMA OLA · auditoría integral): la frontera de vela es
        // por MINUTO, como en el forense, en el calentamiento (`interval=1m`) y
        // ahora en el vivo (`@kline_1m`). Aquí se aproximaba por frontera de DÍA:
        // el ensamble se calibraba una vez cada 24 h de datos y las EMAs de kline
        // —que gobiernan el escudo macro— avanzaban un paso por día. La aptitud
        // que este replay produce decidía promociones de genoma sobre un motor
        // cuyo reloj de calibración iba 1440 veces más lento que el de decisión.
        let is_minute_kline = i > 0 && (ticks[i - 1].ts_ms / 60_000) != (t.ts_ms / 60_000);

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
                // D-717 (DÉCIMA OLA · auditoría integral): el lado agresor viaja
                // en el dato con el convenio de `binance_vision_sync` —maker ⇒
                // (bid = base, ask = qty + base), es decir `bid_qty < ask_qty`—.
                // Aquí se pasaba la NEGACIÓN de ese convenio: cada compra
                // agresiva se contabilizaba como venta, `rolling_cvd` salía con
                // el signo opuesto al flujo real y las ramas de price-action
                // abrían LARGOS cuando el mercado vendía. Sobre ese motor se
                // calcula la aptitud de cada genoma que este binario PROMUEVE.
                t.bid_qty < t.ask_qty,
            );
            let _ = o2;
            // B3.19: envolvente del host sobre la entrada recién abierta.
            let atr_now = core.feature_engines[0].get_atr_pct();
            live_envelope_gate(
                &arena,
                &mut risk_envelope,
                0,
                mid,
                atr_now,
                pos_was_open,
                &mut stats.envelope_vetoes,
            );
            pos_was_open = arena.coins[0].positions.position.is_open();
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
            let _ = o1;
            // B3.19: envolvente del host sobre la entrada recién abierta.
            let atr_now = core.feature_engines[0].get_atr_pct();
            live_envelope_gate(
                &arena,
                &mut risk_envelope,
                0,
                mid,
                atr_now,
                pos_was_open,
                &mut stats.envelope_vetoes,
            );
            pos_was_open = arena.coins[0].positions.position.is_open();
            // D-717: en el modo con libro, `mid <= sim_bid` es una tautología
            // falsa (el mid nunca baja del bid simulado), de modo que el CVD
            // quedaba clavado en +1 y las dos ramas Short eran inalcanzables. El
            // lado agresor es el del dato, igual que en el modo trade-only.
            let maker_flag = t.bid_qty < t.ask_qty;
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
            let _ = o2;
            // B3.19: el evento de trade también puede abrir — misma envolvente.
            let atr_now = core.feature_engines[0].get_atr_pct();
            live_envelope_gate(
                &arena,
                &mut risk_envelope,
                0,
                mid,
                atr_now,
                pos_was_open,
                &mut stats.envelope_vetoes,
            );
            pos_was_open = arena.coins[0].positions.position.is_open();
            c1 = closed1;
            c2 = closed2;
        }

        // F5.1 (paridad host): alimentar el posterior del edge con CADA
        // cierre — mismos EWMAs y mismo record_trade que god_engine. Las
        // entradas vetadas nunca llegan aquí (rollback ⇒ sin cierre, B3.14).
        if let Some((_, pnl_net, _)) = c1.or(c2) {
            if pnl_net >= 0.0 {
                avg_win_abs = if avg_win_abs == 0.0 {
                    pnl_net.abs()
                } else {
                    avg_win_abs * 0.95 + pnl_net.abs() * 0.05
                };
            } else {
                avg_loss_abs = if avg_loss_abs == 0.0 {
                    pnl_net.abs()
                } else {
                    avg_loss_abs * 0.95 + pnl_net.abs() * 0.05
                };
            }
            risk_envelope.record_trade(
                pnl_net > 0.0,
                avg_win_abs.max(1e-9),
                -avg_loss_abs.max(1e-9),
            );
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

/// B3.19 — PARIDAD DE SIZING LIVE↔REPLAY (D-442 / D-116 / D-382).
///
/// HALLAZGO: el host en vivo (`god_engine.rs`, bloque ENVOLVENTE) recalcula
/// el leverage de CADA entrada que el core abre: envolvente Kelly bayesiana
/// sobre la distancia real del SL → `exec_leverage`, con bootstrap
/// exploratorio (leverage 1 mientras posterior.n() < 30) y abort
/// (`rollback_positions`) cuando la envolvente dice NO o el margin-guard
/// (-2019) no sostiene el notional. El replay NO hacía nada de esto: cada
/// orden del core vivía con SU leverage (hasta 50×) → el replay sobrestima
/// notional/nº de trades en stops anchos y capital chico.
///
/// Esta función replica FIELMENTE la cadena del host, en el mismo orden:
///   1. `stop_pct` = distancia entry→SL real (piso 15 bps), fallback
///      `scalp_sl_base.max(ATR×1.5).max(0.0015)` — idéntico al vivo.
///   2. `cap_now` = capital − Σ margins de posiciones ABIERTAS (la posición
///      recién abierta por el core YA reservó su margen: mismo instante que
///      el vivo, que decide después de `process_event`).
///   3. `(env_lev, operable)` = `max_leverage(cap_now, stop_pct, 5, z, k)`
///      con z/k del régimen micro (D-641).
///   4. D-116: n<30 ⇒ bootstrap leverage 1; operable ⇒
///      `min(core_leverage, env_cap)`; no-operable ⇒ VETO.
///   5. D-382: LEVERAGE-ADAPT (sube a ceil(notional/(0.8·free)) si el
///      requerido > 85% del margen libre) + MARGIN-GUARD (abort si el
///      requerido final > 95% del margen libre).
///   6. Veto = rollback EXACTO del vivo: `close_with_fee`, liberar
///      `used_margin`, reembolsar `entry_fee` (la entrada nunca existió ⇒
///      sin PnL, sin fee neto — B3.14: papel que no contamina).
///
/// El notional enviado "al exchange" es el qty del core en AMBOS mundos (el
/// vivo nunca encoge qty); la paridad consiste en que las entradas que el
/// vivo abortaría NO existan en el replay. Retorna false si la entrada fue
/// vetada (rollback aplicado).
#[allow(clippy::too_many_arguments)]
pub fn live_envelope_gate(
    arena: &Arc<GlobalArena>,
    envelope: &mut RiskEnvelope,
    coin_id: usize,
    mid: f64,
    atr_pct: f64,
    prev_open: bool,
    veto_counter: &mut u64,
) -> bool {
    // CERT-M8-C05: coin_id parametrizado — run_backtest_native opera sobre
    // target_coin_id (registro dinámico), no sobre el asiento 0.
    let coin = match arena.coins.get(coin_id) {
        Some(c) => c,
        None => return true,
    };
    let pos = &coin.positions.position;
    if prev_open || !pos.is_open() {
        return true; // nada nuevo que dictaminar
    }

    // F5.1/BUG-598: margen libre real = equity − margen retenido en vivas.
    let total_margin_used: f64 = arena
        .coins
        .iter()
        .map(|c| {
            if c.positions.position.is_open() {
                c.positions.position.margin_used.load(Ordering::Relaxed)
            } else {
                0.0
            }
        })
        .sum();
    let cap_now = (arena.unified_capital.load(Ordering::Relaxed) - total_margin_used).max(0.0);

    let entry_price = pos.entry_price.load(Ordering::Relaxed);
    let qty = pos.quantity.load(Ordering::Relaxed);
    let core_sl = pos.sl_price.load(Ordering::Relaxed);
    let pos_margin = pos.margin_used.load(Ordering::Relaxed);

    // D-442: stop real de la orden; fallback = scalp_sl_base ⊔ ATR×1.5, piso 15 bps.
    let stop_pct = if core_sl > 0.0 && entry_price > 0.0 {
        ((entry_price - core_sl).abs() / entry_price).max(0.0015)
    } else {
        arena
            .config
            .scalp_sl_base
            .load(Ordering::Relaxed)
            .max(atr_pct * 1.5)
            .max(0.0015)
    };

    // D-641: z/k continuos por régimen de capital (mismo literal 5.0 del vivo).
    let env_min_notional = 5.0;
    let env_w = risk_engine::capital_regime::micro_weight(cap_now, env_min_notional);
    let env_z = risk_engine::capital_regime::lerp(1.64, 0.85, env_w);
    let env_k = risk_engine::capital_regime::log_lerp(50.0, 10.0, env_w);
    let (env_lev, operable) = envelope.max_leverage(cap_now, stop_pct, env_min_notional, env_z, env_k);

    // D-116: envolvente AUTORITATIVA + bootstrap exploratorio (leverage 1).
    let envelope_n = envelope.posterior.n();
    let notional_ord = qty.abs() * entry_price;
    let core_leverage = if pos_margin > 0.0 && notional_ord > 0.0 {
        (notional_ord / pos_margin).round().clamp(1.0, 50.0) as u32
    } else {
        (5.05 / cap_now.max(1.0)).ceil().clamp(1.0, 20.0) as u32
    };
    let exec_leverage: u32 = if envelope_n < 30.0 {
        1
    } else if operable {
        let cap = env_lev.floor().clamp(1.0, 20.0) as u32;
        // CERT-M8-C01 — PARIDAD SIZING BT↔VIVO: el host (god_engine.rs
        // ~3857) computa `lev_from_risk = (0.05 · kelly_frac · vol_brake)
        // / sl_at_tau(τ_entry)` y luego `.min(cap)`. El replay ANTES usaba
        // `core_leverage.clamp(1,20).min(cap)` (pre-C-08) — sin kelly_frac
        // scaling, sin stop-distance normalization: un genoma certificado
        // a leverage L tradearía a OTRO L en demo. Ahora la MISMA fórmula,
        // INCLUIDO el vol_brake (P-1b) que antes faltaba — ver abajo.
        let kelly_frac = envelope.risk_fraction(env_z, env_k);
        let tau_entry = pos.entry_tau_ms.load(Ordering::Relaxed) as f64;
        let sl_frac = arena
            .config
            .sl_at_tau(if tau_entry > 0.0 { tau_entry } else { 30_000.0 });
        let risk_budget = 0.05 * kelly_frac;
        // P-1b — VOL-BRAKE (paridad con god_engine.rs:3838-3858): el predictor
        // {SYM}_VOL encoge el presupuesto cuando la σ pronosticada supera ×1.25
        // la base del régimen de entrenamiento. Sin modelo ⇒ brake = 1.0 (paridad
        // trivial). MISMAS claves de registry y MISMA fórmula que el host vivo.
        let vol_fc = arena
            .registry
            .get_for_coin_or(coin_id, "vol_forecast_pct", 0.0);
        let vol_base = arena
            .registry
            .get_for_coin_or(coin_id, "vol_forecast_base", 0.0);
        let vol_brake = vol_brake_factor(vol_fc, vol_base);
        let lev_from_risk = (risk_budget * vol_brake / sl_frac.max(1e-4)).clamp(1.0, 20.0);
        ((lev_from_risk as u32).min(cap)).max(1)
    } else {
        0
    };

    // Veto puro de la envolvente (el host retorna ANTES de los margin guards).
    if exec_leverage == 0 {
        *veto_counter += 1;
        rollback_local_position(arena, coin_id);
        return false;
    }

    // D-382 — LEVERAGE-ADAPT + MARGIN-GUARD (notional al precio de decisión).
    let notional_volume = qty.abs() * mid;
    // MOD6/8-010: lector saturado — ver GlobalArena::used_margin_saturated.
    let used_margin = arena.used_margin.load(Ordering::Relaxed).max(0.0);
    let free_margin = (arena.unified_capital.load(Ordering::Relaxed) - used_margin).max(0.0);
    let mut effective_leverage = exec_leverage;
    let required_margin = notional_volume / effective_leverage as f64;
    if required_margin > free_margin * 0.85 && free_margin > 0.0 {
        let needed_leverage =
            (notional_volume / (free_margin * 0.80)).ceil().clamp(1.0, 20.0) as u32;
        if needed_leverage > effective_leverage {
            effective_leverage = needed_leverage;
        }
    }
    let final_required_margin = notional_volume / effective_leverage as f64;
    if final_required_margin > free_margin * 0.95 {
        *veto_counter += 1;
        rollback_local_position(arena, coin_id);
        return false;
    }
    true
}

/// P-1b — VOL-BRAKE: réplica EXACTA de god_engine.rs:3846-3858 (el host vivo).
/// Cuando la σ pronosticada por {SYM}_VOL supera ×1.25 la base del régimen de
/// entrenamiento, encoge el presupuesto de riesgo UNILATERALMENTE (piso ×0.4,
/// nunca amplifica). Sin modelo (vol_fc o vol_base ≈ 0) ⇒ 1.0 = paridad trivial.
/// Ambos lados DEBEN usar esta misma curva; si el vivo cambia, cambiar aquí.
fn vol_brake_factor(vol_fc: f64, vol_base: f64) -> f64 {
    if vol_base > 1e-9 && vol_fc > 1e-9 {
        let ratio = vol_fc / vol_base;
        if ratio > 1.25 {
            return ((1.25 / ratio).max(0.4)).min(1.0);
        }
    }
    1.0
}

/// Rollback EXACTO de `rollback_positions` del host (god_engine.rs): cierra
/// la posición local, devuelve el margen al pool y REEMBOLSA el entry_fee
/// (nunca fue coste real). Sin PnL — la entrada vetada no contabiliza (B3.14).
pub fn rollback_local_position(arena: &Arc<GlobalArena>, coin_id: usize) {
    if let Some(coin) = arena.coins.get(coin_id) {
        let pos = &coin.positions.position;
        if pos.is_open() {
            let (_, _, _, margin_used, entry_fee) = pos.close_with_fee();
            if margin_used > 0.0 {
                // MOD6/8-010: resta atómica — mismo criterio que el host:
                // el RMW load→store no es atómico.
                arena
                    .used_margin
                    .fetch_sub(margin_used, Ordering::Relaxed);
            }
            if entry_fee > 0.0 {
                arena
                    .unified_capital
                    .fetch_add(entry_fee, Ordering::Relaxed);
            }
        }
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
            trade_only: false,
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

    /// TEST DE WIRING genoma→arena→signal: verifica que los parámetros
    /// del genoma SÍ llegan al arena config (donde el signal generation
    /// los lee). La sensibilidad COMPLETA solo es observable con datos
    /// reales donde el forest produce predicciones direccionales ≠0.5.
    #[test]
    fn wiring_genoma_llega_al_signal_generation() {
        let arena = GlobalArena::build_in_own_stack(1000.0);

        // Genoma A: umbral amplio
        let mut g_a = SuperGenotype::new_baseline(0.0002, 0.0005);
        g_a.ml_threshold_long = 0.52;
        g_a.ml_threshold_short = 0.48;
        g_a.apply_to_arena(&arena);
        let stored_long_a = arena.config.ml_threshold_long.load(Ordering::Relaxed);
        let stored_short_a = arena.config.ml_threshold_short.load(Ordering::Relaxed);

        // Genoma B: umbral estricto
        let mut g_b = SuperGenotype::new_baseline(0.0002, 0.0005);
        g_b.ml_threshold_long = 0.70;
        g_b.ml_threshold_short = 0.30;
        g_b.apply_to_arena(&arena);
        let stored_long_b = arena.config.ml_threshold_long.load(Ordering::Relaxed);
        let stored_short_b = arena.config.ml_threshold_short.load(Ordering::Relaxed);

        // WIRING: los valores DEL GENOMA deben estar EN EL AREA
        assert!((stored_long_a - 0.52).abs() < 1e-6, "genome A ml_long {} != 0.52", stored_long_a);
        assert!((stored_short_a - 0.48).abs() < 1e-6, "genome A ml_short {} != 0.48", stored_short_a);
        assert!((stored_long_b - 0.70).abs() < 1e-6, "genome B ml_long {} != 0.70", stored_long_b);
        assert!((stored_short_b - 0.30).abs() < 1e-6, "genome B ml_short {} != 0.30", stored_short_b);
        // Y deben ser DIFERENTES entre sí
        assert!(stored_long_a != stored_long_b, "wiring roto: A y B almacenan el mismo valor");
    }

    // ── B3.19: PARIDAD DE SIZING (envolvente D-442/D-116/D-382) ──────────

    fn open_test_position(arena: &Arc<GlobalArena>, entry: f64, qty: f64, margin: f64, fee: f64) {
        use quantum_arena::position::PositionHorizon;
        arena.coins[0].positions.position.open_with_fee(
            true,
            entry,
            qty,
            margin,
            1_700_000_000_000,
            entry * 1.01, // tp
            entry * 0.99, // sl (stop_pct = 1%)
            PositionHorizon::Continuous,
            0.6,
            0.5,
            fee,
        );
        arena.used_margin.fetch_add(margin, Ordering::Relaxed);
        arena.unified_capital.fetch_add(-fee, Ordering::Relaxed);
    }

    #[test]
    fn envelope_bootstrap_mantiene_entrada_que_sostiene_margen() {
        // n=0 < 30 ⇒ bootstrap exploratorio leverage 1 (D-116): una entrada
        // cuyo notional cabe en el margen libre NO se veta.
        // D-714: pila suficiente para construir el arena.
        let arena = GlobalArena::build_in_own_stack(1000.0);
        let mut env = RiskEnvelope::new();
        let mut vetoes = 0u64;
        open_test_position(&arena, 100.0, 1.0, 10.0, 0.05);
        let kept = live_envelope_gate(&arena, &mut env, 0, 100.0, 0.001, false, &mut vetoes);
        assert!(kept, "bootstrap no veta entrada sostenible");
        assert_eq!(vetoes, 0);
        assert!(arena.coins[0].positions.position.is_open());
        // El rollback NÓN tocó contabilidad: margen y capital intactos.
        assert!((arena.used_margin.load(Ordering::Relaxed) - 10.0).abs() < 1e-9);
        assert!((arena.unified_capital.load(Ordering::Relaxed) - 999.95).abs() < 1e-9);
    }

    #[test]
    fn envelope_autoritativa_veta_cuando_no_hay_edge() {
        // n≥30 y LCB sin edge ⇒ operable=false ⇒ exec=0 ⇒ VETO con rollback
        // exacto del host: posición cerrada, margen devuelto, fee reembolsado.
        // D-714: pila suficiente para construir el arena.
        let arena = GlobalArena::build_in_own_stack(1000.0);
        let mut env = RiskEnvelope::new();
        for i in 0..500 {
            env.record_trade(i % 2 == 0, 10.0, -10.0); // 50% WR 1:1 = sin edge
        }
        assert!(env.posterior.n() >= 30.0);
        let mut vetoes = 0u64;
        open_test_position(&arena, 100.0, 1.0, 10.0, 0.05);
        let kept = live_envelope_gate(&arena, &mut env, 0, 100.0, 0.001, false, &mut vetoes);
        assert!(!kept, "la envolvente autoritativa debía vetar");
        assert_eq!(vetoes, 1);
        assert!(!arena.coins[0].positions.position.is_open());
        assert!((arena.used_margin.load(Ordering::Relaxed) - 0.0).abs() < 1e-9);
        // Reembolso del entry_fee: capital vuelve a 1000 exacto.
        assert!((arena.unified_capital.load(Ordering::Relaxed) - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn margin_guard_veta_notional_imposible_en_capital_micro() {
        // Capital 13, margen 12, leverage del core 50 (notional 600): ni a
        // 20× el margen requerido cabe en el 95% del margen libre ⇒ abort
        // (paridad con el MARGIN-GUARD -2019 del vivo).
        let arena = GlobalArena::build_in_own_stack(13.0);
        let mut env = RiskEnvelope::new();
        let mut vetoes = 0u64;
        open_test_position(&arena, 100.0, 6.0, 12.0, 0.01); // notional 600
        let kept = live_envelope_gate(&arena, &mut env, 0, 100.0, 0.001, false, &mut vetoes);
        assert!(!kept, "margin-guard debía abortar (600/20 > 0.95·1)");
        assert_eq!(vetoes, 1);
        assert!(!arena.coins[0].positions.position.is_open());
        assert!((arena.unified_capital.load(Ordering::Relaxed) - 13.0).abs() < 1e-9);
    }

    #[test]
    fn envelope_gate_ignora_posicion_ya_evaluada() {
        // prev_open=true: la posición ya pasó por su dictamen — no se
        // re-evalúa (idempotencia por apertura, no por tick).
        // D-714: pila suficiente para construir el arena.
        let arena = GlobalArena::build_in_own_stack(1000.0);
        let mut env = RiskEnvelope::new();
        for i in 0..500 {
            env.record_trade(i % 2 == 0, 10.0, -10.0); // no-operable
        }
        let mut vetoes = 0u64;
        open_test_position(&arena, 100.0, 1.0, 10.0, 0.05);
        let kept = live_envelope_gate(&arena, &mut env, 0, 100.0, 0.001, true, &mut vetoes);
        assert!(kept && vetoes == 0);
        assert!(arena.coins[0].positions.position.is_open());
    }

    #[test]
    fn stop_pct_usa_distancia_real_del_sl() {
        // D-442: con SL al 4% y entry 100, el stop_pct debe ser 0.04 — un SL
        // amplio REDUCE el leverage de la envolvente (f/stop), no lo ignora.
        // Lo verificamos a través de max_leverage directamente (misma fórmula).
        let mut env = RiskEnvelope::new();
        for i in 0..200 {
            env.record_trade(i % 3 != 0, 12.0, -10.0); // 67% WR
        }
        let (lev_narrow, ok_n) = env.max_leverage(5_000.0, 0.01, 5.0, 1.64, 50.0);
        let (lev_wide, ok_w) = env.max_leverage(5_000.0, 0.04, 5.0, 1.64, 50.0);
        assert!(ok_n && ok_w);
        assert!(
            lev_wide < lev_narrow,
            "stop ancho debe reducir leverage: {} !< {}",
            lev_wide,
            lev_narrow
        );
    }

    #[test]
    fn vol_brake_factor_replica_curva_del_vivo() {
        // M8-C01 — paridad BT↔vivo: esta curva DEBE coincidir con
        // god_engine.rs:3846-3858. Si un lado deriva, este test falla.
        // Sin modelo {SYM}_VOL ⇒ 1.0 (paridad trivial: símbolos sin VOL).
        assert_eq!(vol_brake_factor(0.0, 0.0), 1.0);
        assert_eq!(vol_brake_factor(0.0, 2.0), 1.0);
        assert_eq!(vol_brake_factor(2.0, 0.0), 1.0);
        // ratio ≤ 1.25 ⇒ sin freno (nunca amplifica).
        assert_eq!(vol_brake_factor(1.0, 1.0), 1.0);
        assert_eq!(vol_brake_factor(1.25, 1.0), 1.0);
        assert_eq!(vol_brake_factor(0.5, 1.0), 1.0);
        // ratio = 2.5 ⇒ 1.25/2.5 = 0.5.
        assert!((vol_brake_factor(2.5, 1.0) - 0.5).abs() < 1e-12);
        // piso ×0.4: ratio = 10 ⇒ 1.25/10 = 0.125 → clamp 0.4.
        assert!((vol_brake_factor(10.0, 1.0) - 0.4).abs() < 1e-12);
        assert!((vol_brake_factor(1000.0, 1.0) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn replay_con_envolvente_sigue_determinista() {
        let ticks = synth_ticks(20_000);
        let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
        let cfg = ReplayConfig {
            initial_capital: 1000.0,
            warmup_ticks: 200,
            trade_only: false,
        };
        let a = run_booktick_replay(&ticks, &genome, None, &cfg);
        let b = run_booktick_replay(&ticks, &genome, None, &cfg);
        assert_eq!(a.trades, b.trades);
        assert_eq!(a.envelope_vetoes, b.envelope_vetoes);
        assert!((a.net_pnl - b.net_pnl).abs() < 1e-12);
        assert!(a.final_capital.is_finite() && a.final_capital > 0.0);
        assert!(a.max_dd < 1.0);
    }

    // ── B3.16: bordes de OmniHistory (parse, step-function, corte t-1) ────

    fn hist_with(series: Vec<(i64, f64)>) -> OmniHistory {
        let mut s: [Vec<(i64, f64)>; 6] = Default::default();
        s[0] = series;
        OmniHistory { series: s }
    }

    #[test]
    fn value_at_primer_dia_exacto_y_hueco() {
        // Serie con hueco de fin de semana: vie 100, lun 103.
        let h = hist_with(vec![(100, 1.0), (103, 2.0)]);
        // Antes del primer dato → primer valor (sin pánico, sin 0 fantasma).
        assert_eq!(h.value_at(0, 50), 1.0);
        assert_eq!(h.value_at(0, 99), 1.0);
        // Golpe exacto.
        assert_eq!(h.value_at(0, 100), 1.0);
        // Hueco sábado(101)/domingo(102) → último conocido (viernes).
        assert_eq!(h.value_at(0, 101), 1.0);
        assert_eq!(h.value_at(0, 102), 1.0);
        // Lunes y después.
        assert_eq!(h.value_at(0, 103), 2.0);
        assert_eq!(h.value_at(0, 9999), 2.0);
    }

    #[test]
    fn corte_t1_no_usa_el_cierre_del_mismo_dia() {
        // El replay consulta day-1: para un lunes (103) pide el domingo (102)
        // → debe recibir el VIERNES (100), jamás el propio lunes (103).
        let h = hist_with(vec![(100, 1.0), (103, 2.0)]);
        let monday = 103i64;
        let prev_day = (monday - 1).max(0);
        assert_eq!(prev_day, 102);
        assert_eq!(h.value_at(0, prev_day), 1.0, "t-1 debe ver el viernes");
        // Día 0: el corte jamás baja de 0 (sin underflow de i64).
        assert_eq!((0i64 - 1).max(0), 0);
    }

    #[test]
    fn day_of_es_piso_utc_sin_desborde() {
        assert_eq!(OmniHistory::day_of(0), 0);
        assert_eq!(OmniHistory::day_of(86_399_999), 0);
        assert_eq!(OmniHistory::day_of(86_400_000), 1);
        assert_eq!(OmniHistory::day_of(1_700_000_000_000), 19_675);
    }

    #[test]
    fn serie_vacia_es_neutra_sin_pánico() {
        let h = hist_with(vec![]);
        assert_eq!(h.value_at(0, 19_675), 0.0);
    }
}
