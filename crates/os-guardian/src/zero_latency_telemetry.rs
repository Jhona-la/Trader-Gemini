use crossbeam_queue::ArrayQueue;
use once_cell::sync::Lazy;
use std::sync::Arc;
use std::thread;

/// Métrica de telemetría atómica sin bloqueo
#[derive(Debug, Clone, Copy)]
pub struct QuantumMetric {
    pub timestamp: u64,
    pub event_type: u8, // 0: ML Prob, 1: Latency, 2: Error, 3: RAM, 4: Leverage
    pub value_f64: f64,
    pub value_u64: u64,
}

/// Motor de telemetría Cero-Latencia (Lock-Free)
/// Utiliza una cola atómica ArrayQueue para extraer métricas del Hot-Path
/// en menos de 10 nanosegundos sin bloquear.
pub struct TelemetryEngine {
    queue: Arc<ArrayQueue<QuantumMetric>>,
}

// Singleton global para que cualquier parte del sistema haga push de telemetría sin pasar referencias
pub static GLOBAL_TELEMETRY: Lazy<TelemetryEngine> = Lazy::new(|| {
    let engine = TelemetryEngine::new(1_000_000); // 1 Millón de eventos de buffer
    engine.spawn_background_worker();
    engine
});

impl TelemetryEngine {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(ArrayQueue::new(capacity)),
        }
    }

    /// Método hiper-rápido para el Hot-Path. Push atómico.
    #[inline(always)]
    pub fn record_metric(&self, event_type: u8, value_f64: f64, value_u64: u64) {
        let metric = QuantumMetric {
            timestamp: unsafe { std::arch::x86_64::_rdtsc() },
            event_type,
            value_f64,
            value_u64,
        };
        // Intento de push sin bloqueo. Si la cola está llena, descartamos la métrica
        // para NUNCA bloquear el motor principal (filosofía HFT).
        let _ = self.queue.push(metric);
    }

    /// Hilo de fondo que drena la cola y la escribe/procesa asíncronamente
    pub fn spawn_background_worker(&self) {
        let queue = Arc::clone(&self.queue);
        thread::Builder::new()
            .name("Quantum-Telemetry-Worker".into())
            .spawn(move || {
                loop {
                    // Drenar la cola en lotes
                    let mut count = 0;
                    while let Some(_metric) = queue.pop() {
                        // Aquí se enviaría a una base de datos TSDB, SQLite WAL, o Socket UDP.
                        // Por ahora consumimos el evento.
                        count += 1;
                    }

                    if count == 0 {
                        // Si no hay métricas, dormimos el hilo para no quemar CPU (ahorrando RAM/CPU)
                        thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            })
            .expect("Fallo al crear hilo de telemetría");
    }
}
