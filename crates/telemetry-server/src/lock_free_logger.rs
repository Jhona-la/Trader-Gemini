use crossbeam::channel::{Receiver, Sender, bounded};
use std::fs::OpenOptions;
use std::io::Write;
use std::thread;

/// Métrica asíncrona de telemetría de ultra baja latencia.
pub struct TelemetryPayload {
    pub timestamp_ns: u64,
    pub module_id: u8,
    pub pnl_gross: f64,
    pub pnl_post_fees: f64,
    pub roi_gross: f64,
    pub roi_post_fees: f64,
    pub wr_gross: f64,
    pub wr_post_fees: f64,
    pub current_drawdown: f64,
    pub timeframe_ms: u64, // Período temporal de los datos evaluados
    pub meta: [f32; 4],    // Tensors/Probs
    pub is_demo: bool,
}

/// Logger Asíncrono Lock-Free
/// Envía telemetría desde el hilo principal sin I/O, depositándolo en un canal bounded de alta velocidad.
/// Un hilo dedicado (Background) lo consume y escribe en WAL / Disco.
pub struct LockFreeLogger {
    tx: Sender<TelemetryPayload>,
}

impl LockFreeLogger {
    pub fn new(buffer_size: usize, log_path: &str) -> Self {
        // FIX #1444: Clamping defensivo de capacidad para proteger memoria de 16GB
        let safe_capacity = buffer_size.clamp(1024, 500_000);
        let (tx, rx): (Sender<TelemetryPayload>, Receiver<TelemetryPayload>) =
            bounded(safe_capacity);
        let path = log_path.to_string();

        // FIX #1491: Spawn resiliente de hilo de logging sin unwrap
        let _ = thread::Builder::new()
            .name("Telemetry_IO_Thread".to_string())
            .spawn(move || {
                if let Some(parent) = std::path::Path::new(&path).parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let mut file = match OpenOptions::new().create(true).append(true).open(&path) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("⚠️ [LockFreeLogger] No se pudo abrir {}: {}", path, e);
                        return;
                    }
                };

                while let Ok(msg) = rx.recv() {
                    let env_mode = if msg.is_demo { "DEMO" } else { "PROD" };
                    // FIX #1531: Sanitización de flotantes en registro de log
                    let safe_pnl_gross = if msg.pnl_gross.is_finite() {
                        msg.pnl_gross
                    } else {
                        0.0
                    };
                    let safe_pnl_post = if msg.pnl_post_fees.is_finite() {
                        msg.pnl_post_fees
                    } else {
                        0.0
                    };
                    let safe_roi_gross = if msg.roi_gross.is_finite() {
                        msg.roi_gross
                    } else {
                        0.0
                    };
                    let safe_roi_post = if msg.roi_post_fees.is_finite() {
                        msg.roi_post_fees
                    } else {
                        0.0
                    };
                    let safe_wr_gross = if msg.wr_gross.is_finite() {
                        msg.wr_gross
                    } else {
                        0.0
                    };
                    let safe_wr_post = if msg.wr_post_fees.is_finite() {
                        msg.wr_post_fees
                    } else {
                        0.0
                    };
                    let safe_dd = if msg.current_drawdown.is_finite() {
                        msg.current_drawdown
                    } else {
                        0.0
                    };

                    let _ = writeln!(
                        file,
                        "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:?}",
                        msg.timestamp_ns,
                        env_mode,
                        msg.module_id,
                        safe_pnl_gross,
                        safe_pnl_post,
                        safe_roi_gross,
                        safe_roi_post,
                        safe_wr_gross,
                        safe_wr_post,
                        safe_dd,
                        msg.timeframe_ms,
                        msg.meta
                    );
                    let _ = file.flush();
                }
            });

        Self { tx }
    }

    #[inline(always)]
    pub fn log(&self, payload: TelemetryPayload) {
        // Envia sin bloquear si hay espacio. Si está lleno, dropea la métrica (preferimos perder logs que retrasar HFT).
        let _ = self.tx.try_send(payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_free_logger_send() {
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(12345);
        let log_path = temp_dir.join(format!("test_lock_free_logger_{}.log", unique_id));
        let log_path_str = log_path.to_string_lossy().to_string();

        let logger = LockFreeLogger::new(1024, &log_path_str);
        logger.log(TelemetryPayload {
            timestamp_ns: 1672531200000000000,
            module_id: 1,
            pnl_gross: 1.0,
            pnl_post_fees: 0.95,
            roi_gross: 7.69,
            roi_post_fees: 7.30,
            wr_gross: 0.80,
            wr_post_fees: 0.80,
            current_drawdown: 0.01,
            timeframe_ms: 60000,
            meta: [0.5, 0.6, 0.7, 0.8],
            is_demo: true,
        });

        // Let the worker flush
        let mut found = false;
        for _ in 0..40 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            if log_path.exists() {
                found = true;
                break;
            }
        }
        assert!(found, "Log file should have been created by worker thread");
        drop(logger);
        std::thread::sleep(std::time::Duration::from_millis(50));
        let _ = std::fs::remove_file(log_path);
    }

    #[test]
    fn test_lock_free_logger_nan_sanitization_and_prod_mode() {
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(54321);
        let log_path = temp_dir.join(format!("test_logger_nan_{}.log", unique_id));
        let log_path_str = log_path.to_string_lossy().to_string();

        let logger = LockFreeLogger::new(2048, &log_path_str);
        logger.log(TelemetryPayload {
            timestamp_ns: 2000000000,
            module_id: 3,
            pnl_gross: f64::NAN,
            pnl_post_fees: f64::NAN,
            roi_gross: f64::NAN,
            roi_post_fees: f64::NAN,
            wr_gross: f64::NAN,
            wr_post_fees: f64::NAN,
            current_drawdown: f64::NAN,
            timeframe_ms: 1000,
            meta: [0.1, 0.2, 0.3, 0.4],
            is_demo: false,
        });

        let mut found = false;
        for _ in 0..40 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            if log_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&log_path) {
                    if content.contains("PROD") && content.contains("0.0000") {
                        found = true;
                        break;
                    }
                }
            }
        }
        assert!(
            found,
            "Log file should contain sanitized PROD telemetry record"
        );
        drop(logger);
        std::thread::sleep(std::time::Duration::from_millis(50));
        let _ = std::fs::remove_file(log_path);
    }
}
