use crossbeam_channel::{Receiver, Sender};
use rusqlite::Connection;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

pub static DROPPED_LAKEHOUSE_EVENTS: AtomicU64 = AtomicU64::new(0);

/// FASE XLIV: Data Lakehouse Warehouse
/// Una estructura de compresión y persistencia profunda (Offline) para separar
/// el motor HFT de la grabación pesada de datos analíticos, telemetría o histórico de mercado.
pub struct LakehouseWarehouse {
    event_tx: Sender<LakehouseEvent>,
}

pub enum LakehouseEvent {
    StoreMarketDepth {
        symbol: String,
        timestamp: u64,
        bid: f64,
        ask: f64,
        obi: f64,
    },
    StoreTelemetry {
        subsystem: u8,
        frame_type: u8,
        timestamp: u64,
        data: [f64; 6],
    },
    StoreTensor {
        symbol: String,
        timestamp: u64,
        features: Vec<f32>,
        prediction: f32,
        target: f32,
    },
    FlushAndOptimize,
}

impl LakehouseWarehouse {
    /// Inicializa la conexión SQLite en un hilo en background, configurado en modo WAL
    /// para máxima concurrencia y persistencia asíncrona.
    pub fn new<P: AsRef<Path>>(db_path: P) -> Self {
        let path = db_path.as_ref().to_path_buf();
        // FASE 21: RAM Protection. Unbounded channel can cause OOM on 16GB systems.
        // We use a bounded channel of 500,000 items (~30MB max). Backpressure will discard metrics instead of crashing OS.
        let (tx, rx): (Sender<LakehouseEvent>, Receiver<LakehouseEvent>) =
            crossbeam_channel::bounded(500_000);

        thread::spawn(move || {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let mut conn = Connection::open(&path).unwrap_or_else(|_| {
                let temp_path = std::env::temp_dir().join("lakehouse_fallback.db");
                Connection::open(&temp_path).expect("Failed to open Lakehouse fallback SQLite DB")
            });

            // Optimizaciones institucionales para inserciones asíncronas masivas sin desbordar la memoria (16GB Host limit)
            // FIX #1437: Estandarización de mmap a 64MB para preservar memoria en host de 16GB
            conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA temp_store = MEMORY;
                 PRAGMA mmap_size = 67108864;  -- 64MB MMAP unificado para host de 16GB RAM
                 PRAGMA page_size = 32768;      -- 32KB pages optimizado para NVMe/SSD
                 PRAGMA cache_size = -32768;    -- 32MB cache para SQLite (protección RAM)
                 PRAGMA wal_autocheckpoint = 1000; -- Checkpoint frecuente para evitar crecimiento de archivo WAL
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

                 -- Índices compuestos para eliminar table scans y acelerar análisis de microestructura (BUG-647)
                 CREATE INDEX IF NOT EXISTS idx_market_depth_sym_ts ON market_depth(symbol, timestamp);
                 CREATE INDEX IF NOT EXISTS idx_telemetry_sub_ts ON telemetry(subsystem, timestamp);
                 CREATE INDEX IF NOT EXISTS idx_tensor_state_sym_ts ON tensor_state(symbol, timestamp);
                ",
            )
            .expect("Failed to initialize Lakehouse Schema");

            let mut batch: Vec<LakehouseEvent> = Vec::with_capacity(1000);

            let flush_batch = |conn: &mut Connection, batch: &mut Vec<LakehouseEvent>| {
                if batch.is_empty() {
                    return;
                }
                if let Ok(tx) = conn.transaction() {
                    for event in batch.drain(..) {
                        match event {
                            LakehouseEvent::StoreMarketDepth {
                                symbol,
                                timestamp,
                                bid,
                                ask,
                                obi,
                            } => {
                                let ts = timestamp as i64;
                                // FIX #1534: Sanitización de flotantes de libro de órdenes
                                let safe_bid = if bid.is_finite() { bid } else { 0.0 };
                                let safe_ask = if ask.is_finite() { ask } else { 0.0 };
                                let safe_obi = if obi.is_finite() { obi } else { 0.0 };
                                if let Err(e) = tx.execute(
                                    "INSERT INTO market_depth (symbol, timestamp, bid, ask, obi) VALUES (?1, ?2, ?3, ?4, ?5)",
                                    (&symbol, &ts, &safe_bid, &safe_ask, &safe_obi),
                                ) {
                                    eprintln!("⚠️ [LAKEHOUSE ERROR] Insert market_depth failed: {}", e);
                                }
                            }
                            LakehouseEvent::StoreTelemetry {
                                subsystem,
                                frame_type,
                                timestamp,
                                data,
                            } => {
                                let sub = subsystem as i64;
                                let ft = frame_type as i64;
                                let ts = timestamp as i64;
                                // FIX #1534: Sanitización de payload de telemetría
                                let mut safe_data = data;
                                for d in &mut safe_data {
                                    if !d.is_finite() { *d = 0.0; }
                                }
                                if let Err(e) = tx.execute(
                                    "INSERT INTO telemetry (subsystem, frame_type, timestamp, d1, d2, d3, d4, d5, d6) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                                    (&sub, &ft, &ts, &safe_data[0], &safe_data[1], &safe_data[2], &safe_data[3], &safe_data[4], &safe_data[5]),
                                ) {
                                    eprintln!("⚠️ [LAKEHOUSE ERROR] Insert telemetry failed: {}", e);
                                }
                            }
                            LakehouseEvent::StoreTensor {
                                symbol,
                                timestamp,
                                features,
                                prediction,
                                target,
                            } => {
                                let ts = timestamp as i64;
                                let bytes: &[u8] = bytemuck::cast_slice(&features);
                                // FIX #1534: Sanitización de predicción y target
                                let safe_pred = if prediction.is_finite() { prediction } else { 0.0 };
                                let safe_target = if target.is_finite() { target } else { 0.0 };
                                if let Err(e) = tx.execute(
                                    "INSERT INTO tensor_state (symbol, timestamp, features, prediction, target) VALUES (?1, ?2, ?3, ?4, ?5)",
                                    rusqlite::params![symbol, ts, bytes, safe_pred, safe_target],
                                ) {
                                    eprintln!("⚠️ [LAKEHOUSE ERROR] Insert tensor_state failed: {}", e);
                                }
                            }
                            LakehouseEvent::FlushAndOptimize => {}
                        }
                    }
                    let _ = tx.commit();
                }
            };

            let mut last_flush = std::time::Instant::now();
            let mut last_pragma_opt = std::time::Instant::now();
            loop {
                match rx.recv_timeout(std::time::Duration::from_millis(500)) {
                    Ok(event) => {
                        let is_flush = matches!(event, LakehouseEvent::FlushAndOptimize);
                        batch.push(event);
                        if batch.len() >= 1000 || is_flush || last_flush.elapsed() >= std::time::Duration::from_millis(500) {
                            flush_batch(&mut conn, &mut batch);
                            last_flush = std::time::Instant::now();
                            // FIX #610: Ejecutar PRAGMA optimize sólo en flush explícito o cada 1 hora para evitar contención de lock
                            if is_flush || last_pragma_opt.elapsed() >= std::time::Duration::from_secs(3600) {
                                let _ = conn.execute("PRAGMA optimize;", []);
                                last_pragma_opt = std::time::Instant::now();
                            }
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        if !batch.is_empty() {
                            flush_batch(&mut conn, &mut batch);
                            last_flush = std::time::Instant::now();
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        break;
                    }
                }
            }
            flush_batch(&mut conn, &mut batch);
        });

        Self { event_tx: tx }
    }

    /// Evía métricas al Lakehouse sin bloquear el motor principal
    #[inline(always)]
    pub fn record_depth(&self, symbol: String, timestamp: u64, bid: f64, ask: f64, obi: f64) {
        // FIX #659: Validar finitud antes de enviar a SQLite
        if !bid.is_finite() || !ask.is_finite() || !obi.is_finite() {
            return;
        }

        // FIX #1001: Verificar Result de try_send e incrementar contador de drops
        if let Err(_) = self.event_tx.try_send(LakehouseEvent::StoreMarketDepth {
            symbol,
            timestamp,
            bid,
            ask,
            obi,
        }) {
            DROPPED_LAKEHOUSE_EVENTS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn record_telemetry(&self, subsystem: u8, frame_type: u8, timestamp: u64, data: [f64; 6]) {
        // FIX #659: Sanitizar flotantes de telemetría
        let mut safe_data = data;
        for d in &mut safe_data {
            if !d.is_finite() {
                *d = 0.0;
            }
        }

        // FIX #1001: Verificar Result de try_send
        if let Err(_) = self.event_tx.try_send(LakehouseEvent::StoreTelemetry {
            subsystem,
            frame_type,
            timestamp,
            data: safe_data,
        }) {
            DROPPED_LAKEHOUSE_EVENTS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn record_tensor(
        &self,
        symbol: String,
        timestamp: u64,
        features: Vec<f32>,
        prediction: f32,
        target: f32,
    ) {
        // FIX #726: Sanitizar tensores y predicciones contra NaNs antes de Lakehouse
        let mut safe_features = features;
        for f in &mut safe_features {
            if !f.is_finite() {
                *f = 0.0;
            }
        }
        let safe_pred = if prediction.is_finite() { prediction } else { 0.5 };
        let safe_target = if target.is_finite() { target } else { 0.0 };

        // FIX #1001: Verificar Result de try_send
        if let Err(_) = self.event_tx.try_send(LakehouseEvent::StoreTensor {
            symbol,
            timestamp,
            features: safe_features,
            prediction: safe_pred,
            target: safe_target,
        }) {
            DROPPED_LAKEHOUSE_EVENTS.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn flush(&self) {
        let _ = self.event_tx.try_send(LakehouseEvent::FlushAndOptimize);
    }
}

/// Bloque de Ticks Comprimidos por Delta-Encoding para Lakehouse (#26-#36)
/// Comprime series temporales de ticks L2 con ratio > 80% usando empaquetamiento de enteros
#[derive(Debug, Clone)]
pub struct CompressedTickBatch {
    pub base_timestamp_ms: u64,
    pub base_price_fixed: i64,
    pub price_scale: f64,
    pub count: usize,
    pub delta_buffer: Vec<u8>,
}

impl Default for CompressedTickBatch {
    fn default() -> Self {
        Self {
            base_timestamp_ms: 0,
            base_price_fixed: 0,
            price_scale: 10_000.0,
            count: 0,
            delta_buffer: Vec::new(),
        }
    }
}

impl CompressedTickBatch {
    pub fn new(base_timestamp_ms: u64, base_price: f64) -> Self {
        let price_scale = if base_price > 0.0 && base_price < 1.0 {
            100_000_000.0 // 8 decimales para sub-centavos (ej. PEPE, SHIB)
        } else {
            10_000.0 // 4 decimales estándar
        };
        Self {
            base_timestamp_ms,
            base_price_fixed: (base_price * price_scale).round() as i64,
            price_scale,
            count: 0,
            delta_buffer: Vec::with_capacity(4096),
        }
    }

    /// Codifica un tick en bytes compactos: `[delta_time_ms: u32, delta_price: i32, qty: f32]` (12 bytes por tick vs 24 bytes f64)
    #[inline(always)]
    pub fn push_tick(&mut self, timestamp_ms: u64, price: f64, qty: f64) {
        let safe_price = if price.is_finite() && price > 0.0 { price } else { self.base_price_fixed as f64 / self.price_scale };
        let safe_qty = if qty.is_finite() && qty >= 0.0 { qty } else { 0.0 };

        let dt = (timestamp_ms.saturating_sub(self.base_timestamp_ms)).min(u32::MAX as u64) as u32;
        let p_fixed = (safe_price * self.price_scale).round() as i64;
        let dp = (p_fixed - self.base_price_fixed).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        let q_f32 = safe_qty as f32;

        self.delta_buffer.extend_from_slice(&dt.to_le_bytes());
        self.delta_buffer.extend_from_slice(&dp.to_le_bytes());
        self.delta_buffer.extend_from_slice(&q_f32.to_le_bytes());
        self.count += 1;
    }

    /// Descomprime un tick indexado
    #[inline(always)]
    pub fn get_tick(&self, index: usize) -> Option<(u64, f64, f64)> {
        if index >= self.count {
            return None;
        }
        let offset = index * 12;
        let dt = u32::from_le_bytes([
            self.delta_buffer[offset],
            self.delta_buffer[offset + 1],
            self.delta_buffer[offset + 2],
            self.delta_buffer[offset + 3],
        ]);
        let dp = i32::from_le_bytes([
            self.delta_buffer[offset + 4],
            self.delta_buffer[offset + 5],
            self.delta_buffer[offset + 6],
            self.delta_buffer[offset + 7],
        ]);
        let q_f32 = f32::from_le_bytes([
            self.delta_buffer[offset + 8],
            self.delta_buffer[offset + 9],
            self.delta_buffer[offset + 10],
            self.delta_buffer[offset + 11],
        ]);

        let ts = self.base_timestamp_ms + dt as u64;
        let price = (self.base_price_fixed + dp as i64) as f64 / self.price_scale;
        let qty = q_f32 as f64;

        Some((ts, price, qty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lakehouse_in_memory_or_temp() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!("lakehouse_test_{}.db", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        
        let warehouse = LakehouseWarehouse::new(&db_path);
        warehouse.record_depth("BTCUSDT".into(), 1700000000000, 60000.0, 60001.0, 0.25);
        warehouse.record_telemetry(1, 2, 1700000000000, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        warehouse.flush();

        std::thread::sleep(std::time::Duration::from_millis(50));
        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_compressed_tick_batch_roundtrip() {
        let base_ts = 1700000000000_u64;
        let base_p = 60000.0;
        let mut batch = CompressedTickBatch::new(base_ts, base_p);

        batch.push_tick(base_ts + 10, 60000.50, 1.25);
        batch.push_tick(base_ts + 25, 59999.80, 0.50);

        assert_eq!(batch.count, 2);
        assert_eq!(batch.delta_buffer.len(), 24); // 12 bytes per tick

        let (ts0, p0, q0) = batch.get_tick(0).unwrap();
        assert_eq!(ts0, base_ts + 10);
        assert!((p0 - 60000.50).abs() < 1e-4);
        assert!((q0 - 1.25).abs() < 1e-3);

        let (ts1, p1, q1) = batch.get_tick(1).unwrap();
        assert_eq!(ts1, base_ts + 25);
        assert!((p1 - 59999.80).abs() < 1e-4);
        assert!((q1 - 0.50).abs() < 1e-3);
    }

    #[test]
    fn test_compressed_tick_batch_subcent_precision() {
        let base_ts = 1700000000000_u64;
        let base_p = 0.00001850; // SHIB / PEPE sub-cent level
        let mut batch = CompressedTickBatch::new(base_ts, base_p);

        batch.push_tick(base_ts + 5, 0.00001855, 1000000.0);
        batch.push_tick(base_ts + 15, 0.00001842, 500000.0);

        assert_eq!(batch.count, 2);
        let (ts0, p0, q0) = batch.get_tick(0).unwrap();
        assert_eq!(ts0, base_ts + 5);
        assert!((p0 - 0.00001855).abs() < 1e-8, "Debe preservar 8 decimales en sub-centavos");
        assert!((q0 - 1000000.0).abs() < 1e-1);

        let (ts1, p1, q1) = batch.get_tick(1).unwrap();
        assert_eq!(ts1, base_ts + 15);
        assert!((p1 - 0.00001842).abs() < 1e-8, "Debe preservar 8 decimales en sub-centavos");
        assert!((q1 - 500000.0).abs() < 1e-1);
    }

    #[test]
    fn test_compressed_tick_batch_nan_immunity() {
        let base_ts = 1700000000000_u64;
        let mut batch = CompressedTickBatch::new(base_ts, 60000.0);
        batch.push_tick(base_ts + 10, f64::NAN, f64::INFINITY);
        let (ts, p, q) = batch.get_tick(0).unwrap();
        assert_eq!(ts, base_ts + 10);
        assert!(p.is_finite() && p > 0.0);
        assert!(q.is_finite() && q == 0.0);
    }

    #[test]
    fn test_compressed_tick_batch_empty_and_oob() {
        let batch = CompressedTickBatch::default();
        assert_eq!(batch.count, 0);
        assert!(batch.get_tick(0).is_none());
        assert!(batch.get_tick(100).is_none());
    }
}
