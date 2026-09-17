use crossbeam_channel::{bounded, Receiver, Sender};
use once_cell::sync::Lazy;
use std::thread;

/// Un evento de telemetría hiper-rápido y compacto para serialización lock-free
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
        // 1_000_000 de eventos = ~32MB de memoria RAM pre-asignada.
        let (tx, rx): (Sender<TelemetryEvent>, Receiver<TelemetryEvent>) = bounded(1_000_000);

        // FIX #1494: Spawn resiliente de TelemetryWorker sin expect
        let _ = thread::Builder::new()
            .name("TelemetryWorker".to_string())
            .spawn(move || {
                Self::worker_loop(rx);
            });

        Self { sender: tx }
    }

    #[inline(always)]
    pub fn send(&self, event: TelemetryEvent) {
        // try_send para que sea estrictamente lock-free y sin latencia.
        // Si el buffer está lleno, el evento se descarta para no frenar el God Engine (prioridad: latencia de trading).
        let _ = self.sender.try_send(event);
    }

    fn worker_loop(rx: Receiver<TelemetryEvent>) {
        let mut total_pnl = 0.0;
        let mut total_fees = 0.0;
        let mut total_trades = 0;
        let mut compounding_capital = 0.0; // Se inicializará con UpdateCapital

        while let Ok(event) = rx.recv() {
            match event {
                TelemetryEvent::UpdateCapital { real_capital } => {
                    compounding_capital = real_capital;
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

                    if total_trades % 10 == 0 || pnl.abs() > compounding_capital * 0.05 {
                        let avg_pnl = total_pnl / total_trades as f64;
                        let projected_trades_per_day = 50.0; // Asumimos 50 trades por día por moneda activa
                        let projected_daily_pnl = avg_pnl * projected_trades_per_day;
                        let days_to_double = if projected_daily_pnl > 0.0 {
                            compounding_capital / projected_daily_pnl
                        } else {
                            f64::INFINITY
                        };

                        println!(
                            "📊 [PROYECCIÓN INSTITUCIONAL] Trade #{} | Coin: {} | Dir: {}",
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
                            println!("   🚀 [PROYECCIÓN] A este ritmo, el capital se duplicará en {:.1} días.", days_to_double);
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

/// Helper functions macro-like para enviar telemetría en nanosegundos
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
