use crossbeam_channel::{bounded, Receiver, Sender};
use once_cell::sync::Lazy;
use std::thread;

/// Telemetry payload. No zero-latency or lock-free serialization guarantee.
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    LatencyWarning {
        dns_ms: u64,
        ws_ms: u64,
    },
    TradeExecution {
        coin_id: usize,
        is_long: bool,
        pnl: f64,
        fees: f64,
        roi_pre_fees: f64,
        roi_post_fees: f64,
        trade_duration_ms: u64,
        current_wr: f64,
    },
    RegimeShift {
        coin_id: usize,
        old_regime: u8,
        new_regime: u8,
    },
    NanoForestPrediction {
        coin_id: usize,
        ml_prob: f64,
        threshold: f64,
        execution_us: u64,
    },
    MemoryDiagnostic {
        ram_used_mb: u64,
        total_alloc_mb: u64,
        cpu_cycles: u64,
    },
    ParityAlert {
        message: String,
        expected: f64,
        actual: f64,
        diff: f64,
    },
    UpdateCapital {
        real_capital: f64,
    },
}

pub static TELEMETRY: Lazy<TelemetryManager> = Lazy::new(TelemetryManager::new);

/// Recibo económico del worker (GLM 144, §52 del mandato del operador):
/// agrega POR TRADE lo que el informe integral necesita — ROI antes/después
/// de fees, PnL bruto/neto, WR antes/después de costos, Sharpe (Welford sobre
/// roi_post_fees), EV/trade y el periodo EXACTO de los trades recibidos.
/// Ausencia honesta (CL-15): lo no medible se serializa como null, jamás 0.
struct ReciboEconomico {
    trades: u64,
    pnl_bruto: f64,
    fees: f64,
    ganadoras_brutas: u64,
    ganadoras_netas: u64,
    media_roi_post: f64,
    m2_roi_post: f64,
    n_roi_post: u64,
    capital_inicial: Option<f64>,
    wr_actual: f64,
    primer_ns: Option<u128>,
    ultimo_ns: u128,
}

impl ReciboEconomico {
    fn nuevo() -> Self {
        Self {
            trades: 0,
            pnl_bruto: 0.0,
            fees: 0.0,
            ganadoras_brutas: 0,
            ganadoras_netas: 0,
            media_roi_post: 0.0,
            m2_roi_post: 0.0,
            n_roi_post: 0,
            capital_inicial: None,
            wr_actual: f64::NAN,
            primer_ns: None,
            ultimo_ns: 0,
        }
    }

    fn ahora_ns() -> Option<u128> {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_nanos())
    }

    fn registrar_capital(&mut self, real_capital: f64) {
        if self.capital_inicial.is_none() && real_capital.is_finite() && real_capital > 0.0 {
            self.capital_inicial = Some(real_capital);
        }
    }

    fn registrar_trade(&mut self, pnl: f64, fees: f64, roi_post_fees: f64, wr: f64) {
        self.trades += 1;
        if pnl.is_finite() {
            self.pnl_bruto += pnl;
            if pnl > 0.0 {
                self.ganadoras_brutas += 1;
            }
        }
        if fees.is_finite() {
            self.fees += fees;
        }
        let neto = pnl - fees;
        if neto.is_finite() && neto > 0.0 {
            self.ganadoras_netas += 1;
        }
        if wr.is_finite() {
            self.wr_actual = wr;
        }
        // Welford online sobre roi_post_fees (sólo muestras finitas).
        if roi_post_fees.is_finite() {
            self.n_roi_post += 1;
            let delta = roi_post_fees - self.media_roi_post;
            self.media_roi_post += delta / self.n_roi_post as f64;
            self.m2_roi_post += delta * (roi_post_fees - self.media_roi_post);
        }
        if let Some(t) = Self::ahora_ns() {
            if self.primer_ns.is_none() {
                self.primer_ns = Some(t);
            }
            self.ultimo_ns = t;
        }
    }

    /// Serializa a JSON escribiendo null en lo no medible (ausencia honesta).
    fn volcar(&self, ruta: &str) {
        let dias = match self.primer_ns {
            Some(p) if self.ultimo_ns > p => {
                ((self.ultimo_ns - p) as f64 / 86_400_000_000_000.0).max(1.0 / 24.0)
            }
            _ => 1.0 / 24.0,
        };
        let pnl_neto = self.pnl_bruto - self.fees;
        let (roi_pre, roi_post) = match self.capital_inicial {
            Some(c) if c > 0.0 && self.pnl_bruto.is_finite() && self.fees.is_finite() => (
                Some(self.pnl_bruto / c * 100.0),
                Some(pnl_neto / c * 100.0),
            ),
            _ => (None, None),
        };
        let sharpe = if self.n_roi_post > 1 {
            let var = (self.m2_roi_post / (self.n_roi_post - 1) as f64).max(0.0);
            let std = var.sqrt();
            if std > 1e-9 && std.is_finite() && self.media_roi_post.is_finite() {
                let trades_por_ano = (self.n_roi_post as f64 / dias) * 365.0;
                Some((self.media_roi_post / std) * trades_por_ano.sqrt().min(500.0))
            } else {
                None
            }
        } else {
            None
        };
        let opt = |v: Option<f64>| -> String {
            match v {
                Some(x) if x.is_finite() => format!("{x}"),
                _ => "null".to_string(),
            }
        };
        let json = format!(
            "{{\"trades\":{},\"pnl_bruto\":{},\"fees\":{},\"pnl_neto\":{},\
\"ganadoras_brutas\":{},\"ganadoras_netas\":{},\
\"wr_pre_costos\":{},\"wr_post_costos\":{},\
\"capital_inicial\":{},\"roi_pre_fees_pct\":{},\"roi_post_fees_pct\":{},\
\"ev_por_trade\":{},\"sharpe_post_fees\":{},\
\"periodo_ns\":[{},{}],\"actualizado_ns\":{}}}",
            self.trades,
            opt(Some(self.pnl_bruto)),
            opt(Some(self.fees)),
            opt(Some(pnl_neto)),
            self.ganadoras_brutas,
            self.ganadoras_netas,
            if self.trades > 0 {
                (self.ganadoras_brutas as f64 / self.trades as f64).to_string()
            } else {
                "null".into()
            },
            if self.trades > 0 {
                (self.ganadoras_netas as f64 / self.trades as f64).to_string()
            } else {
                "null".into()
            },
            opt(self.capital_inicial),
            opt(roi_pre),
            opt(roi_post),
            if self.trades > 0 && pnl_neto.is_finite() {
                (pnl_neto / self.trades as f64).to_string()
            } else {
                "null".into()
            },
            opt(sharpe),
            self.primer_ns.map(|v| v.to_string()).unwrap_or_else(|| "null".into()),
            if self.ultimo_ns > 0 { self.ultimo_ns.to_string() } else { "null".into() },
            Self::ahora_ns().map(|v| v.to_string()).unwrap_or_else(|| "null".into()),
        );
        if let Err(e) = std::fs::write(ruta, json) {
            eprintln!("⚠️ [TELEMETRÍA] recibo económico no escrito ({ruta}): {e}");
        }
    }
}

pub struct TelemetryManager {
    sender: Sender<TelemetryEvent>,
}

impl Default for TelemetryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetryManager {
    pub fn new() -> Self {
        // Un ring buffer grande para evitar dropear eventos en picos de volatilidad, pero limitado para proteger la RAM.
        // Memory depends on enum size, channel overhead and owned strings, not 32MB.
        let (tx, rx): (Sender<TelemetryEvent>, Receiver<TelemetryEvent>) = bounded(1_000_000);

        // XXXVI OPEN: spawn failure is ignored; no consumer health is exposed.
        let _ = thread::Builder::new()
            .name("TelemetryWorker".to_string())
            .spawn(move || {
                Self::worker_loop(rx);
            });

        Self { sender: tx }
    }

    #[inline(always)]
    pub fn send(&self, event: TelemetryEvent) {
        // Nonblocking send; this is not a measured zero-latency guarantee.
        // Si el buffer está lleno, el evento se descarta para no frenar el God Engine (prioridad: latencia de trading).
        let _ = self.sender.try_send(event);
    }

    fn worker_loop(rx: Receiver<TelemetryEvent>) {
        let mut total_pnl = 0.0;
        let mut total_fees = 0.0;
        let mut total_trades = 0;
        let mut compounding_capital = 0.0; // Se inicializará con UpdateCapital
        // GLM 144 (§52): recibo económico para el informe integral; se vuelca
        // en outputs/ cada 10 trades (cadencia existente) y con cada capital.
        let mut recibo = ReciboEconomico::nuevo();
        let ruta_recibo = "outputs/economia_live.json";

        while let Ok(event) = rx.recv() {
            match event {
                TelemetryEvent::UpdateCapital { real_capital } => {
                    compounding_capital = real_capital;
                    recibo.registrar_capital(real_capital);
                    recibo.volcar(ruta_recibo);
                    println!(
                        "🏦 [TELEMETRÍA] Capital Compuesto Inicializado Dinámicamente a: {:.4} USD",
                        compounding_capital
                    );
                }
                TelemetryEvent::LatencyWarning { dns_ms, ws_ms } => {
                    println!("⚠️ [TELEMETRÍA RED] DNS: {}ms | WS: {}ms", dns_ms, ws_ms);
                }
                TelemetryEvent::TradeExecution {
                    coin_id,
                    is_long,
                    pnl,
                    fees,
                    roi_pre_fees,
                    roi_post_fees,
                    trade_duration_ms,
                    current_wr,
                } => {
                    total_pnl += pnl;
                    total_fees += fees;
                    total_trades += 1;
                    compounding_capital += pnl;
                    recibo.registrar_trade(pnl, fees, roi_post_fees, current_wr);

                    if total_trades % 10 == 0 || pnl.abs() > compounding_capital * 0.05 {
                        recibo.volcar(ruta_recibo);
                        let avg_pnl = total_pnl / total_trades as f64;
                        let projected_trades_per_day = 50.0; // Fixed hypothetical TOTAL cadence, not measured per asset.
                        let projected_daily_pnl = avg_pnl * projected_trades_per_day;
                        let days_to_double = if projected_daily_pnl > 0.0 {
                            compounding_capital / projected_daily_pnl
                        } else {
                            f64::INFINITY
                        };

                        println!(
                            "📊 [ESCENARIO LINEAL NO CALIBRADO] Trade #{} | Coin: {} | Dir: {}",
                            total_trades,
                            coin_id,
                            if is_long { "LONG" } else { "SHORT" }
                        );
                        println!(
                            "   ➤ PnL Neto Trade: {:.4} | Fees: {:.4} | ROI Post-Fee: {:.4}%",
                            pnl, fees, roi_post_fees
                        );
                        println!("   ➤ Capital Compuesto: {:.2} USD | PnL Acumulado: {:.2} USD | WR Actual: {:.2}%", compounding_capital, total_pnl, current_wr * 100.0);
                        if days_to_double.is_finite() && days_to_double > 0.0 {
                            println!("   [ESCENARIO] Capital / (PnL medio × 50 trades/día) = {:.1} días; no es pronóstico ni interés compuesto.", days_to_double);
                        } else {
                            println!("   ⚠️ [PROYECCIÓN] El sistema necesita mejorar el Profit Factor para proyectar crecimiento compuesto.");
                        }
                    } else if total_trades % 5 == 0 {
                        // Usa variables que de otro modo estarían huérfanas en intervalos de baja relevancia
                        println!("   ➤ Trade en {} ms | ROI Bruto: {:.4}% | Drag por Fees: {:.4}% | Fees Acumuladas: {:.4}", 
                                 trade_duration_ms, roi_pre_fees, roi_pre_fees - roi_post_fees, total_fees);
                    }
                }
                TelemetryEvent::RegimeShift {
                    coin_id,
                    old_regime,
                    new_regime,
                } => {
                    println!(
                        "🔄 [TELEMETRÍA RÉGIMEN] Coin: {} | Shift: {} -> {}",
                        coin_id, old_regime, new_regime
                    );
                }
                TelemetryEvent::NanoForestPrediction {
                    coin_id: _,
                    ml_prob: _,
                    threshold: _,
                    execution_us: _,
                } => {}
                TelemetryEvent::MemoryDiagnostic {
                    ram_used_mb,
                    total_alloc_mb,
                    cpu_cycles,
                } => {
                    println!("💾 [TELEMETRÍA RENDIMIENTO] RAM Used: {}MB | Alloc: {}MB | Asymptotic CPU Cycles: {}", ram_used_mb, total_alloc_mb, cpu_cycles);
                }
                TelemetryEvent::ParityAlert {
                    message,
                    expected,
                    actual,
                    diff,
                } => {
                    println!(
                        "🚨 [PARIDAD FORENSE] {} | Esperado: {:.4} | Real: {:.4} | Dif: {:.6}",
                        message, expected, actual, diff
                    );
                }
            }
        }
    }
}

/// Nonblocking telemetry helpers; latency has not been certified here.
#[inline(always)]
pub fn send_latency_warning(dns_ms: u64, ws_ms: u64) {
    TELEMETRY.send(TelemetryEvent::LatencyWarning { dns_ms, ws_ms });
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn send_trade_execution(
    coin_id: usize,
    is_long: bool,
    pnl: f64,
    fees: f64,
    roi_pre_fees: f64,
    roi_post_fees: f64,
    trade_duration_ms: u64,
    current_wr: f64,
) {
    TELEMETRY.send(TelemetryEvent::TradeExecution {
        coin_id,
        is_long,
        pnl,
        fees,
        roi_pre_fees,
        roi_post_fees,
        trade_duration_ms,
        current_wr,
    });
}

#[inline(always)]
pub fn send_ml_prediction(coin_id: usize, ml_prob: f64, threshold: f64, execution_us: u64) {
    TELEMETRY.send(TelemetryEvent::NanoForestPrediction {
        coin_id,
        ml_prob,
        threshold,
        execution_us,
    });
}

#[inline(always)]
pub fn send_parity_alert(message: &str, expected: f64, actual: f64, diff: f64) {
    TELEMETRY.send(TelemetryEvent::ParityAlert {
        message: message.to_string(),
        expected,
        actual,
        diff,
    });
}

#[inline(always)]
pub fn update_dynamic_capital(real_capital: f64) {
    TELEMETRY.send(TelemetryEvent::UpdateCapital { real_capital });
}

/// Función auxiliar para capturar ciclos asintóticos de CPU en x86_64
#[inline(always)]
pub fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}
