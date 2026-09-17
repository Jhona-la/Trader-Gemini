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
        if let Some(parent) = db_path.as_ref().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(&db_path).unwrap_or_else(|_| {
            let temp_path = std::env::temp_dir().join("history_store_fallback.db");
            Connection::open(&temp_path).expect("Failed to open HistoryStore fallback SQLite DB")
        });

        // FIX #1435: Pragmas resilientes sin unwrap() y mmap optimizado a 64MB para 16GB RAM
        let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
        let _ = conn.pragma_update(None, "journal_mode", "WAL");
        let _ = conn.pragma_update(None, "synchronous", "NORMAL");
        let _ = conn.pragma_update(None, "temp_store", "MEMORY");
        let _ = conn.pragma_update(None, "mmap_size", "67108864"); // 64 MB
        let _ = conn.pragma_update(None, "cache_size", "-32000"); // 32 MB

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
        let _ = conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON klines(timestamp);",
            [],
        );

        Self { conn }
    }

    /// Inserta o actualiza una vela de forma segura (sin panic).
    pub fn try_upsert_kline(&self, k: &KlineRow) -> rusqlite::Result<()> {
        // FIX #629: Sanitizar campos flotantes para evitar inserción de valores corruptos
        if !k.open.is_finite()
            || !k.high.is_finite()
            || !k.low.is_finite()
            || !k.close.is_finite()
            || !k.volume.is_finite()
            || k.open <= 0.0
            || k.close <= 0.0
        {
            return Ok(());
        }
        self.conn.execute(
            "INSERT INTO klines (coin_id, timestamp, open, high, low, close, volume) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(coin_id, timestamp) DO UPDATE SET 
             open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume",
            params![k.coin_id as i64, k.timestamp as i64, k.open, k.high, k.low, k.close, k.volume],
        )?;
        Ok(())
    }

    /// Inserta o actualiza una vela.
    pub fn upsert_kline(&self, k: &KlineRow) {
        if let Err(e) = self.try_upsert_kline(k) {
            eprintln!("⚠️ [HistoryStore] Error en upsert_kline: {}", e);
        }
    }

    /// Inserta velas en batch masivo para la sincronización Delta de forma atómica y segura.
    pub fn try_insert_batch(&mut self, klines: &[KlineRow]) -> rusqlite::Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO klines (coin_id, timestamp, open, high, low, close, volume) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(coin_id, timestamp) DO UPDATE SET 
                 open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume"
            )?;
            for k in klines {
                // FIX #827: Sanitizar campos individuales en vez de descartar la vela entera
                if !k.open.is_finite() || !k.close.is_finite() || k.open <= 0.0 || k.close <= 0.0 {
                    continue;
                }
                let safe_high = if k.high.is_finite() && k.high >= k.open.max(k.close) {
                    k.high
                } else {
                    k.open.max(k.close)
                };
                let safe_low = if k.low.is_finite() && k.low > 0.0 && k.low <= k.open.min(k.close) {
                    k.low
                } else {
                    k.open.min(k.close)
                };
                let safe_vol = if k.volume.is_finite() && k.volume >= 0.0 {
                    k.volume
                } else {
                    0.0
                };

                stmt.execute(params![
                    k.coin_id as i64,
                    k.timestamp as i64,
                    k.open,
                    safe_high,
                    safe_low,
                    k.close,
                    safe_vol
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Inserta velas en batch masivo para la sincronización Delta
    pub fn insert_batch(&mut self, klines: &[KlineRow]) {
        if let Err(e) = self.try_insert_batch(klines) {
            eprintln!("⚠️ [HistoryStore] Error en insert_batch: {}", e);
        }
    }

    /// Borra datos más antiguos a 365 días
    pub fn prune_older_than(&self, timestamp_ms: u64) {
        // FIX #701: Sanitizar timestamp_ms para prevenir números negativos
        let safe_ts = (timestamp_ms.min(i64::MAX as u64)) as i64;
        let _ = self
            .conn
            .execute("DELETE FROM klines WHERE timestamp < ?1", params![safe_ts]);
    }

    /// Extrae las últimas N velas de una moneda de forma segura (sin panic)
    pub fn try_get_recent_candles(
        &self,
        coin_id: usize,
        limit: usize,
    ) -> rusqlite::Result<Vec<KlineRow>> {
        // FIX #701: Clampeado defensivo de limit para evitar 0 o desbordamientos de memoria
        let safe_limit = limit.clamp(1, 100_000);
        let mut stmt = self.conn.prepare(
            "SELECT coin_id, timestamp, open, high, low, close, volume 
             FROM klines 
             WHERE coin_id = ?1 
             ORDER BY timestamp DESC 
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![coin_id as i64, safe_limit as i64], |row| {
            Ok(KlineRow {
                coin_id: row.get::<_, i64>(0)? as usize,
                timestamp: row.get::<_, i64>(1)? as u64,
                open: row.get(2)?,
                high: row.get(3)?,
                low: row.get(4)?,
                close: row.get(5)?,
                volume: row.get(6)?,
            })
        })?;

        let mut klines = Vec::new();
        for k in rows.flatten() {
            klines.push(k);
        }
        klines.reverse();
        Ok(klines)
    }

    /// Extrae las últimas N velas de una moneda, ordenadas cronológicamente
    pub fn get_recent_candles(&self, coin_id: usize, limit: usize) -> Vec<KlineRow> {
        self.try_get_recent_candles(coin_id, limit)
            .unwrap_or_default()
    }

    /// Retorna el timestamp de la vela más reciente para un coin_id (para hacer Delta Sync)
    pub fn get_latest_timestamp(&self, coin_id: usize) -> Option<u64> {
        self.conn
            .query_row(
                "SELECT timestamp FROM klines WHERE coin_id = ?1 ORDER BY timestamp DESC LIMIT 1",
                params![coin_id as i64],
                |row| row.get::<_, i64>(0).map(|ts| ts as u64),
            )
            .ok()
    }

    /// Extrae velas dentro de un rango de timestamps [start_ts, end_ts] ordenadas cronológicamente
    pub fn get_range(&self, coin_id: usize, start_ts: u64, end_ts: u64) -> Vec<KlineRow> {
        let mut stmt = match self.conn.prepare(
            "SELECT coin_id, timestamp, open, high, low, close, volume 
             FROM klines 
             WHERE coin_id = ?1 AND timestamp >= ?2 AND timestamp <= ?3 
             ORDER BY timestamp ASC",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = match stmt.query_map(
            params![coin_id as i64, start_ts as i64, end_ts as i64],
            |row| {
                Ok(KlineRow {
                    coin_id: row.get::<_, i64>(0)? as usize,
                    timestamp: row.get::<_, i64>(1)? as u64,
                    open: row.get(2)?,
                    high: row.get(3)?,
                    low: row.get(4)?,
                    close: row.get(5)?,
                    volume: row.get(6)?,
                })
            },
        ) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        rows.flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_history_store_upsert_and_recent() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!(
            "history_test_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let store = HistoryStore::new(&db_path);
        store.upsert_kline(&KlineRow {
            coin_id: 0,
            timestamp: 1000,
            open: 60000.0,
            high: 60100.0,
            low: 59900.0,
            close: 60050.0,
            volume: 10.5,
        });

        let recent = store.get_recent_candles(0, 10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].timestamp, 1000);
        assert_eq!(store.get_latest_timestamp(0), Some(1000));

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_history_store_batch_insert_and_nan_sanitization() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!(
            "history_test_batch_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut store = HistoryStore::new(&db_path);
        let batch = vec![
            KlineRow {
                coin_id: 1,
                timestamp: 2000,
                open: 3000.0,
                high: 3050.0,
                low: 2950.0,
                close: 3020.0,
                volume: 50.0,
            },
            KlineRow {
                coin_id: 1,
                timestamp: 2060,
                open: f64::NAN, // Invalid candle -> sanitized/skipped
                high: 3100.0,
                low: 3000.0,
                close: 3050.0,
                volume: 20.0,
            },
        ];

        store.insert_batch(&batch);
        let recent = store.get_recent_candles(1, 10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].timestamp, 2000);

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_history_store_prune_older_than() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!(
            "history_test_prune_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let store = HistoryStore::new(&db_path);
        store.upsert_kline(&KlineRow {
            coin_id: 0,
            timestamp: 1000,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 10.0,
        });
        store.upsert_kline(&KlineRow {
            coin_id: 0,
            timestamp: 5000,
            open: 102.0,
            high: 108.0,
            low: 101.0,
            close: 106.0,
            volume: 15.0,
        });

        // Prune candles older than timestamp 3000
        store.prune_older_than(3000);

        let recent = store.get_recent_candles(0, 10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].timestamp, 5000);

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_history_store_get_range_and_multi_coin_isolation() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!(
            "history_test_range_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut store = HistoryStore::new(&db_path);
        let batch = vec![
            KlineRow {
                coin_id: 0,
                timestamp: 1000,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 102.0,
                volume: 10.0,
            },
            KlineRow {
                coin_id: 0,
                timestamp: 2000,
                open: 102.0,
                high: 108.0,
                low: 101.0,
                close: 107.0,
                volume: 12.0,
            },
            KlineRow {
                coin_id: 0,
                timestamp: 3000,
                open: 107.0,
                high: 110.0,
                low: 106.0,
                close: 109.0,
                volume: 15.0,
            },
            KlineRow {
                coin_id: 1,
                timestamp: 2000,
                open: 2000.0,
                high: 2050.0,
                low: 1980.0,
                close: 2030.0,
                volume: 50.0,
            },
        ];
        store.insert_batch(&batch);

        // Range query for coin 0 between [1500, 3500]
        let range = store.get_range(0, 1500, 3500);
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].timestamp, 2000);
        assert_eq!(range[1].timestamp, 3000);

        // Coin 1 query must isolate coin 0 data
        let coin1_candles = store.get_recent_candles(1, 10);
        assert_eq!(coin1_candles.len(), 1);
        assert_eq!(coin1_candles[0].close, 2030.0);

        let _ = std::fs::remove_file(db_path);
    }
}
