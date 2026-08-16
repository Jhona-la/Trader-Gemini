use rusqlite::Connection;
use crossbeam_channel::{Sender, Receiver};
use std::thread;
use std::path::Path;

/// FASE XLIV: Data Lakehouse Warehouse
/// Una estructura de compresión y persistencia profunda (Offline) para separar 
/// el motor HFT de la grabación pesada de datos analíticos, telemetría o histórico de mercado.
pub struct LakehouseWarehouse {
    event_tx: Sender<LakehouseEvent>,
}

pub enum LakehouseEvent {
    StoreMarketDepth { symbol: String, timestamp: u64, bid: f64, ask: f64, obi: f64 },
    StoreTelemetry { subsystem: u8, frame_type: u8, timestamp: u64, data: [f64; 6] },
    StoreTensor { symbol: String, timestamp: u64, features: Vec<f32>, prediction: f32, target: f32 },
    FlushAndOptimize,
}

impl LakehouseWarehouse {
    /// Inicializa la conexión SQLite en un hilo en background, configurado en modo WAL
    /// para máxima concurrencia y persistencia asíncrona.
    pub fn new<P: AsRef<Path>>(db_path: P) -> Self {
        let path = db_path.as_ref().to_path_buf();
        // FASE 21: RAM Protection. Unbounded channel can cause OOM on 16GB systems.
        // We use a bounded channel of 500,000 items (~30MB max). Backpressure will discard metrics instead of crashing OS.
        let (tx, rx): (Sender<LakehouseEvent>, Receiver<LakehouseEvent>) = crossbeam_channel::bounded(500_000);
        
        thread::spawn(move || {
            let mut conn = Connection::open(path).expect("Failed to open Lakehouse SQLite");
            
            // Optimizaciones institucionales para inserciones asíncronas masivas sin desbordar la memoria (16GB Host limit)
            conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA temp_store = MEMORY;
                 PRAGMA mmap_size = 1000000000; -- 1GB MMAP para acceso ultra-rápido sin swap
                 PRAGMA page_size = 32768;      -- 32KB pages optimizado para NVMe/SSD
                 PRAGMA cache_size = -500000;   -- 500MB max cache para SQLite
                 PRAGMA busy_timeout = 5000;    -- 5 segundos de timeout anti-lock
                 
                 CREATE TABLE IF NOT EXISTS market_depth (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     symbol TEXT NOT NULL,
                     timestamp INTEGER NOT NULL,
                     bid REAL NOT NULL,
                     ask REAL NOT NULL,
                     obi REAL NOT NULL
                 );
                 
                 CREATE TABLE IF NOT EXISTS telemetry (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     subsystem INTEGER NOT NULL,
                     frame_type INTEGER NOT NULL,
                     timestamp INTEGER NOT NULL,
                     d1 REAL, d2 REAL, d3 REAL, d4 REAL, d5 REAL, d6 REAL
                 );
                 
                 CREATE TABLE IF NOT EXISTS tensor_state (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     symbol TEXT NOT NULL,
                     timestamp INTEGER NOT NULL,
                     features BLOB NOT NULL,
                     prediction REAL NOT NULL,
                     target REAL NOT NULL
                 );
                "
            ).expect("Failed to initialize Lakehouse Schema");
            
            let mut batch_count = 0;
            let mut tx_transaction = conn.transaction().unwrap();
            
            while let Ok(event) = rx.recv() {
                match event {
                    LakehouseEvent::StoreMarketDepth { symbol, timestamp, bid, ask, obi } => {
                        let ts = timestamp as i64;
                        tx_transaction.execute(
                            "INSERT INTO market_depth (symbol, timestamp, bid, ask, obi) VALUES (?1, ?2, ?3, ?4, ?5)",
                            (&symbol, &ts, &bid, &ask, &obi),
                        ).unwrap_or_default();
                    }
                    LakehouseEvent::StoreTelemetry { subsystem, frame_type, timestamp, data } => {
                        let sub = subsystem as i64;
                        let ft = frame_type as i64;
                        let ts = timestamp as i64;
                        tx_transaction.execute(
                            "INSERT INTO telemetry (subsystem, frame_type, timestamp, d1, d2, d3, d4, d5, d6) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                            (&sub, &ft, &ts, &data[0], &data[1], &data[2], &data[3], &data[4], &data[5]),
                        ).unwrap_or_default();
                    }
                    LakehouseEvent::StoreTensor { symbol, timestamp, features, prediction, target } => {
                        let ts = timestamp as i64;
                        // Transformar Vec<f32> a bytes planos para el BLOB
                        let bytes: &[u8] = bytemuck::cast_slice(&features);
                        tx_transaction.execute(
                            "INSERT INTO tensor_state (symbol, timestamp, features, prediction, target) VALUES (?1, ?2, ?3, ?4, ?5)",
                            rusqlite::params![symbol, ts, bytes, prediction, target],
                        ).unwrap_or_default();
                    }
                    LakehouseEvent::FlushAndOptimize => {
                        tx_transaction.commit().unwrap_or_default();
                        conn.execute("PRAGMA optimize;", []).unwrap_or_default();
                        tx_transaction = conn.transaction().unwrap();
                        continue;
                    }
                }
                
                batch_count += 1;
                if batch_count >= 10_000 {
                    tx_transaction.commit().unwrap_or_default();
                    tx_transaction = conn.transaction().unwrap();
                    batch_count = 0;
                }
            }
        });
        
        Self {
            event_tx: tx,
        }
    }
    
    /// Evía métricas al Lakehouse sin bloquear el motor principal
    #[inline(always)]
    pub fn record_depth(&self, symbol: String, timestamp: u64, bid: f64, ask: f64, obi: f64) {
        let _ = self.event_tx.try_send(LakehouseEvent::StoreMarketDepth { symbol, timestamp, bid, ask, obi });
    }
    
    #[inline(always)]
    pub fn record_telemetry(&self, subsystem: u8, frame_type: u8, timestamp: u64, data: [f64; 6]) {
        let _ = self.event_tx.try_send(LakehouseEvent::StoreTelemetry { subsystem, frame_type, timestamp, data });
    }
    
    #[inline(always)]
    pub fn record_tensor(&self, symbol: String, timestamp: u64, features: Vec<f32>, prediction: f32, target: f32) {
        let _ = self.event_tx.try_send(LakehouseEvent::StoreTensor { symbol, timestamp, features, prediction, target });
    }
    
    pub fn flush(&self) {
        let _ = self.event_tx.try_send(LakehouseEvent::FlushAndOptimize);
    }
}

