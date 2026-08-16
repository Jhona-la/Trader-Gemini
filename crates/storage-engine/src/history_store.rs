use rusqlite::{params, Connection};
use std::path::Path;

/// 🚀 ALGORITMO #91: HISTORICAL ROLLING BUFFER (1-Year Treadmill)
/// Almacena velas históricas de 1 minuto para todos los activos activos.
/// Soporta inyección rápida y limpieza de datos obsoletos (más antiguos a 1 año).
pub struct HistoryStore {
    conn: Connection,
}

#[derive(Debug, Clone)]
pub struct KlineRow {
    pub coin_id: usize,
    pub timestamp: u64, // ms since epoch
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl HistoryStore {
    pub fn new<P: AsRef<Path>>(db_path: P) -> Self {
        let conn = Connection::open(&db_path).expect("Failed to open HistoryStore SQLite DB");

        conn.pragma_update(None, "journal_mode", "WAL").unwrap();
        conn.pragma_update(None, "synchronous", "NORMAL").unwrap();
        conn.pragma_update(None, "temp_store", "MEMORY").unwrap();
        conn.pragma_update(None, "mmap_size", "268435456").unwrap();
        conn.pragma_update(None, "cache_size", "-64000").unwrap();

        conn.execute(
            "CREATE TABLE IF NOT EXISTS klines (
                coin_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (coin_id, timestamp)
            ) WITHOUT ROWID;",
            [],
        )
        .expect("Failed to create klines table");

        // Índice por tiempo para facilitar el pruning de 1 año
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON klines(timestamp);",
            [],
        )
        .unwrap();

        Self { conn }
    }

    /// Inserta o actualiza una vela.
    pub fn upsert_kline(&self, k: &KlineRow) {
        self.conn.execute(
            "INSERT INTO klines (coin_id, timestamp, open, high, low, close, volume) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(coin_id, timestamp) DO UPDATE SET 
             open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume",
            params![k.coin_id as i64, k.timestamp as i64, k.open, k.high, k.low, k.close, k.volume],
        ).unwrap();
    }

    /// Inserta velas en batch masivo para la sincronización Delta
    pub fn insert_batch(&mut self, klines: &[KlineRow]) {
        let tx = self.conn.transaction().unwrap();
        {
            let mut stmt = tx.prepare(
                "INSERT INTO klines (coin_id, timestamp, open, high, low, close, volume) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(coin_id, timestamp) DO UPDATE SET 
                 open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume"
            ).unwrap();
            for k in klines {
                stmt.execute(params![
                    k.coin_id as i64,
                    k.timestamp as i64,
                    k.open,
                    k.high,
                    k.low,
                    k.close,
                    k.volume
                ])
                .unwrap();
            }
        }
        tx.commit().unwrap();
    }

    /// Borra datos más antiguos a 365 días
    pub fn prune_older_than(&self, timestamp_ms: u64) {
        let _ = self.conn.execute(
            "DELETE FROM klines WHERE timestamp < ?1",
            params![timestamp_ms as i64],
        );
    }

    /// Extrae las últimas N velas de una moneda, ordenadas cronológicamente
    pub fn get_recent_candles(&self, coin_id: usize, limit: usize) -> Vec<KlineRow> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT coin_id, timestamp, open, high, low, close, volume 
             FROM klines 
             WHERE coin_id = ?1 
             ORDER BY timestamp DESC 
             LIMIT ?2",
            )
            .unwrap();

        let rows = stmt
            .query_map(params![coin_id as i64, limit as i64], |row| {
                Ok(KlineRow {
                    coin_id: row.get::<_, i64>(0)? as usize,
                    timestamp: row.get::<_, i64>(1)? as u64,
                    open: row.get(2)?,
                    high: row.get(3)?,
                    low: row.get(4)?,
                    close: row.get(5)?,
                    volume: row.get(6)?,
                })
            })
            .unwrap();

        let mut klines = Vec::new();
        for k in rows.flatten() {
            klines.push(k);
        }
        // Están en orden DESC por la query, las invertimos para que queden en orden cronológico ascendente
        klines.reverse();
        klines
    }

    /// Retorna el timestamp de la vela más reciente para un coin_id (para hacer Delta Sync)
    pub fn get_latest_timestamp(&self, coin_id: usize) -> Option<u64> {
        self.conn
            .query_row(
                "SELECT MAX(timestamp) FROM klines WHERE coin_id = ?1",
                params![coin_id as i64],
                |row| row.get::<_, Option<i64>>(0).map(|v| v.map(|ts| ts as u64)),
            )
            .unwrap_or(None)
    }
}
