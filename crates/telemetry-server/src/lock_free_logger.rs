use crossbeam::channel::{bounded, Sender, Receiver};
use std::thread;
use std::fs::OpenOptions;
use std::io::Write;

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
    pub meta: [f32; 4], // Tensors/Probs
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
        let (tx, rx): (Sender<TelemetryPayload>, Receiver<TelemetryPayload>) = bounded(buffer_size);
        let path = log_path.to_string();
        
        thread::Builder::new()
            .name("Telemetry_IO_Thread".to_string())
            .spawn(move || {
                // Pin to Core 2 if possible to avoid disrupting Core 0/1 HFT
                if let Some(core_ids) = core_affinity::get_core_ids() {
                    if core_ids.len() > 2 {
                        core_affinity::set_for_current(core_ids[2]);
                    }
                }
                
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .unwrap();
                
                while let Ok(msg) = rx.recv() {
                    let env_mode = if msg.is_demo { "DEMO" } else { "PROD" };
                    let _ = writeln!(file, "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:?}", 
                        msg.timestamp_ns, env_mode, msg.module_id, msg.pnl_gross, msg.pnl_post_fees, msg.roi_gross, msg.roi_post_fees, msg.wr_gross, msg.wr_post_fees, msg.current_drawdown, msg.timeframe_ms, msg.meta);
                }
            }).unwrap();

        Self { tx }
    }

    #[inline(always)]
    pub fn log(&self, payload: TelemetryPayload) {
        // Envia sin bloquear si hay espacio. Si está lleno, dropea la métrica (preferimos perder logs que retrasar HFT).
        let _ = self.tx.try_send(payload);
    }
}
