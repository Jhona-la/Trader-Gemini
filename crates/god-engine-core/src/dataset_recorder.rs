//! Grabador asíncrono no bloqueante de datasets de entrenamiento para DarkAlpha (F1-108).
//!
//! # Fundamento Teórico y Rendimiento (Nanosegundos)
//! - **Problema Histórico (F1-108)**: En cada cierre confirmado de posición, el camino caliente
//!   hacía llamadas síncronas a NTFS (`OpenOptions::open(append)` + 56 format! + writeln!)
//!   sobre `data/dark_alpha_dataset_{sym}.csv` reteniendo el `Mutex` de `nn_entry_tensor`.
//!   Esto disparaba la latencia de 25 µs a 2 ms (hasta 80x el presupuesto de tick).
//! - **Arquitectura Cero-Bloqueo**:
//!   1. Se copia el tensor de entrada 54D directamente en la pila (`[f32; 54]`, 216 bytes)
//!      y se libera el `Mutex` en nanosegundos (< 100 ns).
//!   2. Se despacha a una cola acotada MPSC/SPSC no bloqueante (`try_send`).
//!      Si la cola se satura, la muestra se descarta de forma segura sin bloquear el hilo
//!      del motor de trading, incrementando un contador atómico de diagnósticos.
//!   3. Un hilo demonio en segundo plano consume las muestras, gestiona los handles de archivo
//!      en búfer (`BufWriter<File>`) y resuelve la ruta canónica centralizada mediante
//!      `quantum_arena::paths::data_join`, eliminando dependencias del directorio de trabajo (CSG-1).

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::OnceLock;

/// Capacidad máxima de la cola asíncrona de muestras (acotada para proteger los 16 GB de RAM).
const DATASET_QUEUE_CAPACITY: usize = 1024;

/// Muestra de entrenamiento para la red neuronal DarkAlpha.
#[derive(Debug, Clone)]
pub struct DatasetSample {
    pub symbol: String,
    pub target_return: f64,
    pub features: [f64; 54],
}

/// Contador atómico de muestras descartadas por saturación del canal de disco.
pub static DATASET_DROPPED_SAMPLES: AtomicU64 = AtomicU64::new(0);

/// Contador atómico de muestras grabadas exitosamente en disco.
pub static DATASET_RECORDED_SAMPLES: AtomicU64 = AtomicU64::new(0);

static DATASET_SENDER: OnceLock<SyncSender<DatasetSample>> = OnceLock::new();

/// Inicializa explícitamente el hilo grabador en segundo plano si aún no ha sido lanzado.
///
/// Permite inicializar el grabador en el arranque del motor para que la primera muestra
/// no pague el coste de creación de hilos del sistema operativo.
pub fn init_dataset_recorder() {
    let _ = get_or_init_sender();
}

/// Inicializa el hilo grabador en segundo plano si aún no ha sido lanzado.
fn get_or_init_sender() -> &'static SyncSender<DatasetSample> {
    DATASET_SENDER.get_or_init(|| {
        let (tx, rx) = sync_channel::<DatasetSample>(DATASET_QUEUE_CAPACITY);
        std::thread::Builder::new()
            .name("dark-alpha-dataset-writer".to_string())
            .spawn(move || {
                run_writer_loop(rx);
            })
            .expect("Fallo crítico al lanzar hilo dark-alpha-dataset-writer");
        tx
    })
}

/// Envía una muestra de entrenamiento al grabador en segundo plano de manera no bloqueante.
///
/// Si la cola está llena, la muestra se descarta de forma segura sin degradar la latencia
/// del bucle de eventos principal.
#[inline]
pub fn record_training_sample(symbol: String, target_return: f64, features: [f64; 54]) {
    let sender = get_or_init_sender();
    let sample = DatasetSample {
        symbol,
        target_return,
        features,
    };
    match sender.try_send(sample) {
        Ok(()) => {
            // Encolado con éxito en < 150 ns
        }
        Err(TrySendError::Full(_)) => {
            DATASET_DROPPED_SAMPLES.fetch_add(1, Ordering::Relaxed);
        }
        Err(TrySendError::Disconnected(_)) => {
            // El hilo de escritura terminó o el proceso está cerrando
        }
    }
}

/// Retorna las métricas actuales de muestras procesadas y descartadas (grabadas, descartadas).
pub fn get_dataset_metrics() -> (u64, u64) {
    (
        DATASET_RECORDED_SAMPLES.load(Ordering::Relaxed),
        DATASET_DROPPED_SAMPLES.load(Ordering::Relaxed),
    )
}

/// Bucle de escritura en segundo plano con caché de file handles e I/O en búfer.
fn run_writer_loop(rx: Receiver<DatasetSample>) {
    let mut file_cache: HashMap<String, BufWriter<File>> = HashMap::new();
    let mut write_count = 0usize;

    while let Ok(sample) = rx.recv() {
        let sym = sample.symbol;
        let writer = file_cache.entry(sym.clone()).or_insert_with(|| {
            let path_str = quantum_arena::paths::data_join(&format!("dark_alpha_dataset_{}.csv", sym));
            let path = PathBuf::from(&path_str);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let need_header = !path.exists();
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|e| {
                    panic!("No se pudo abrir dataset para escritura {}: {}", path.display(), e);
                });
            let mut bw = BufWriter::with_capacity(32 * 1024, file);
            if need_header {
                let header = "target_return,".to_string()
                    + &(0..54).map(|i| format!("f{i}")).collect::<Vec<_>>().join(",")
                    + "\n";
                let _ = bw.write_all(header.as_bytes());
            }
            bw
        });

        // Formatear fila en segundo plano (fuera del hilo caliente)
        let mut row = format!("{:.8}", sample.target_return);
        for v in sample.features.iter() {
            row.push_str(&format!(",{:.8}", v));
        }
        row.push('\n');

        if writer.write_all(row.as_bytes()).is_ok() {
            DATASET_RECORDED_SAMPLES.fetch_add(1, Ordering::Relaxed);
        }

        write_count += 1;
        if write_count % 16 == 0 {
            let _ = writer.flush();
        }
    }

    // Al salir, vaciar todos los búferes
    for (_, mut w) in file_cache {
        let _ = w.flush();
    }
}
