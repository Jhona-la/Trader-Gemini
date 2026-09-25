use rusqlite::{params, Connection};
use std::path::Path;

#[cfg(test)]
mod audit_xiii_tests {
    use super::*;

    fn candle() -> KlineRow {
        KlineRow {
            coin_id: 0,
            timestamp: 1000,
            open: 10.0,
            high: 12.0,
            low: 9.0,
            close: 11.0,
            volume: 2.0,
        }
    }

    #[test]
    fn invalid_ohlcv_is_rejected_without_rewriting_existing_evidence() {
        let store = HistoryStore::new(":memory:");
        let good = candle();
        store.try_upsert_kline(&good).unwrap();
        for field in 0..5 {
            let mut bad = good.clone();
            match field {
                0 => bad.high = 8.0,
                1 => bad.low = 11.0,
                2 => bad.volume = -1.0,
                3 => bad.high = f64::INFINITY,
                _ => bad.open = f64::NAN,
            }
            assert!(store.try_upsert_kline(&bad).is_err(), "field {field}");
        }
        assert_eq!(store.try_get_recent_candles(0, 1).unwrap()[0].close, 11.0);
    }

    #[test]
    fn invalid_batch_is_not_partial_success_or_synthetic_ohlcv() {
        let mut store = HistoryStore::new(":memory:");
        let good = candle();
        let mut bad = good.clone();
        bad.timestamp = 2000;
        bad.high = f64::NAN;
        assert!(store.try_insert_batch(&[good, bad]).is_err());
        assert!(store.try_get_recent_candles(0, 10).unwrap().is_empty());
    }

    #[test]
    fn unsigned_sql_identifiers_must_not_wrap() {
        let store = HistoryStore::new(":memory:");
        let mut bad = candle();
        bad.timestamp = u64::MAX;
        assert!(store.try_upsert_kline(&bad).is_err());
        bad = candle();
        bad.coin_id = usize::MAX;
        if usize::BITS == 64 {
            assert!(store.try_upsert_kline(&bad).is_err());
        }
    }

    #[test]
    fn zero_limit_means_zero_rows_and_large_limit_is_not_silently_clipped() {
        let store = HistoryStore::new(":memory:");
        store.try_upsert_kline(&candle()).unwrap();
        assert!(store.try_get_recent_candles(0, 0).unwrap().is_empty());
        assert!(store.try_get_recent_candles(0, 100_001).is_err());
    }

    #[test]
    fn row_decode_errors_are_not_an_empty_or_partial_success() {
        let store = HistoryStore::new(":memory:");
        store
            .conn
            .execute("INSERT INTO klines VALUES (0,1000,'broken',12,9,11,2)", [])
            .unwrap();
        assert!(store.try_get_recent_candles(0, 10).is_err());
    }

    #[test]
    fn negative_persisted_timestamp_is_not_a_far_future_date() {
        let store = HistoryStore::new(":memory:");
        store
            .conn
            .execute("INSERT INTO klines VALUES (0,-1,10,12,9,11,2)", [])
            .unwrap();
        assert!(store.try_get_recent_candles(0, 10).is_err());
    }

    #[test]
    fn singleton_and_batch_agree_on_valid_data_and_order() {
        let mut one = HistoryStore::new(":memory:");
        let mut batch = HistoryStore::new(":memory:");
        let mut first = candle();
        first.timestamp = 2000;
        let second = candle();
        let data = [first, second];
        for row in &data {
            one.try_upsert_kline(row).unwrap();
        }
        batch.try_insert_batch(&data).unwrap();
        let a = one.try_get_recent_candles(0, 10).unwrap();
        let b = batch.try_get_recent_candles(0, 10).unwrap();
        assert_eq!(
            a.iter()
                .map(|r| (r.timestamp, r.high, r.low, r.volume))
                .collect::<Vec<_>>(),
            b.iter()
                .map(|r| (r.timestamp, r.high, r.low, r.volume))
                .collect::<Vec<_>>()
        );
        assert_eq!(a[0].timestamp, 1000);
        // Also exercise the empty transaction.
        one.try_insert_batch(&[]).unwrap();
    }

    #[test]
    fn failed_open_does_not_redirect_to_another_database() {
        // A directory is not a SQLite file; no fallback path may be opened.
        assert!(HistoryStore::try_new(std::env::temp_dir()).is_err());
    }

    #[test]
    fn range_and_latest_propagate_corruption_and_invalid_bounds() {
        let store = HistoryStore::new(":memory:");
        assert_eq!(store.try_get_latest_timestamp(0).unwrap(), None);
        store.try_upsert_kline(&candle()).unwrap();
        assert_eq!(store.try_get_latest_timestamp(0).unwrap(), Some(1000));
        assert_eq!(store.try_get_range(0, 1000, 1000).unwrap().len(), 1);
        assert!(store.try_get_range(0, 2000, 1000).is_err());
        assert!(store.try_get_range(0, 0, u64::MAX).is_err());
        store
            .conn
            .execute("INSERT INTO klines VALUES (0,2000,'broken',12,9,11,2)", [])
            .unwrap();
        assert!(store.try_get_latest_timestamp(0).is_err());
        assert!(store.try_get_range(0, 0, 3000).is_err());
    }

    #[test]
    fn invalid_prune_cutoff_cannot_delete_the_archive() {
        let store = HistoryStore::new(":memory:");
        store.try_upsert_kline(&candle()).unwrap();
        assert!(store.try_prune_older_than(u64::MAX).is_err());
        assert_eq!(store.try_get_recent_candles(0, 10).unwrap().len(), 1);
        assert_eq!(store.try_prune_older_than(1000).unwrap(), 0);
        assert_eq!(store.try_prune_older_than(1001).unwrap(), 1);
    }

    #[test]
    fn sql_failure_rolls_back_prior_updates_in_the_same_batch() {
        let mut store = HistoryStore::new(":memory:");
        store.try_upsert_kline(&candle()).unwrap();
        store.conn.execute_batch("CREATE TRIGGER fail_second BEFORE INSERT ON klines WHEN NEW.timestamp = 2000 BEGIN SELECT RAISE(ABORT,'injected'); END;").unwrap();
        let mut first = candle();
        first.close = 10.5;
        let mut second = candle();
        second.timestamp = 2000;
        assert!(store.try_insert_batch(&[first, second]).is_err());
        let rows = store.try_get_recent_candles(0, 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].close, 11.0);
    }
}

/// OHLCV archive keyed by coin_id and timestamp. This schema does not encode
/// interval, venue or universe generation; do not mix those domains implicitly.
/// Retention is chosen by the caller's explicit cutoff, not learned here.
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

fn input_error(message: &str) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message,
    )))
}

fn sql_coin_id(coin_id: usize) -> rusqlite::Result<i64> {
    i64::try_from(coin_id).map_err(|_| input_error("coin_id exceeds SQLite signed integer domain"))
}

fn sql_timestamp(timestamp: u64) -> rusqlite::Result<i64> {
    i64::try_from(timestamp)
        .map_err(|_| input_error("timestamp exceeds SQLite signed integer domain"))
}

impl KlineRow {
    /// Domain checks, not statistical outlier filtering. Never impute a bar.
    fn validate(&self) -> rusqlite::Result<()> {
        sql_coin_id(self.coin_id)?;
        sql_timestamp(self.timestamp)?;
        if ![self.open, self.high, self.low, self.close, self.volume]
            .iter()
            .all(|x| x.is_finite())
            || self.low <= 0.0
            || self.volume < 0.0
            || self.low > self.open
            || self.open > self.high
            || self.low > self.close
            || self.close > self.high
        {
            return Err(input_error("invalid OHLCV: require finite positive prices, low<=open,close<=high and volume>=0"));
        }
        Ok(())
    }
}

fn decode_kline(row: &rusqlite::Row<'_>) -> rusqlite::Result<KlineRow> {
    let id: i64 = row.get(0)?;
    let ts: i64 = row.get(1)?;
    let invalid = |column, message: &str| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            rusqlite::types::Type::Integer,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                message,
            )),
        )
    };
    let coin_id = usize::try_from(id).map_err(|_| invalid(0, "invalid stored coin_id"))?;
    let timestamp = u64::try_from(ts).map_err(|_| invalid(1, "negative stored timestamp"))?;
    let value = KlineRow {
        coin_id,
        timestamp,
        open: row.get(2)?,
        high: row.get(3)?,
        low: row.get(4)?,
        close: row.get(5)?,
        volume: row.get(6)?,
    };
    value.validate().map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(
            2,
            rusqlite::types::Type::Real,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            )),
        )
    })?;
    Ok(value)
}

impl HistoryStore {
    /// Compatibility constructor. Never redirects a failed path to a shared DB.
    /// # Panics
    /// If opening/configuring fails. Use try_new for fallible initialization.
    pub fn new<P: AsRef<Path>>(db_path: P) -> Self {
        Self::try_new(db_path).expect("Failed to open requested HistoryStore")
    }

    pub fn try_new<P: AsRef<Path>>(db_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        if let Some(parent) = db_path
            .as_ref()
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&db_path)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        // NORMAL is not a guarantee that the latest commit survives power loss.
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "temp_store", "MEMORY")?;
        conn.pragma_update(None, "mmap_size", "67108864")?;
        conn.pragma_update(None, "cache_size", "-32000")?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS klines (
                coin_id INTEGER NOT NULL, timestamp INTEGER NOT NULL,
                open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
                close REAL NOT NULL, volume REAL NOT NULL,
                PRIMARY KEY (coin_id, timestamp)
            ) WITHOUT ROWID;",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON klines(timestamp);",
            [],
        )?;
        Ok(Self { conn })
    }

    /// Stores exactly the validated observation or reports an error.
    pub fn try_upsert_kline(&self, k: &KlineRow) -> rusqlite::Result<()> {
        k.validate()?;
        self.conn.execute(
            "INSERT INTO klines (coin_id,timestamp,open,high,low,close,volume)
             VALUES (?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(coin_id,timestamp) DO UPDATE SET
             open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,volume=excluded.volume",
            params![k.coin_id as i64,k.timestamp as i64,k.open,k.high,k.low,k.close,k.volume],
        )?;
        Ok(())
    }

    /// Legacy wrapper: reports errors to stderr. Prefer the fallible method.
    pub fn upsert_kline(&self, k: &KlineRow) {
        if let Err(e) = self.try_upsert_kline(k) {
            eprintln!("HistoryStore upsert rejected: {e}");
        }
    }

    /// All rows must be valid. Invalid input or SQL error rolls back the batch.
    /// Does not replace missing extrema/volume with fabricated observations.
    pub fn try_insert_batch(&mut self, klines: &[KlineRow]) -> rusqlite::Result<()> {
        for k in klines {
            k.validate()?;
        }
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO klines (coin_id,timestamp,open,high,low,close,volume)
                 VALUES (?1,?2,?3,?4,?5,?6,?7)
                 ON CONFLICT(coin_id,timestamp) DO UPDATE SET
                 open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,volume=excluded.volume"
            )?;
            for k in klines {
                stmt.execute(params![
                    k.coin_id as i64,
                    k.timestamp as i64,
                    k.open,
                    k.high,
                    k.low,
                    k.close,
                    k.volume
                ])?;
            }
        }
        tx.commit()
    }

    pub fn insert_batch(&mut self, klines: &[KlineRow]) {
        if let Err(e) = self.try_insert_batch(klines) {
            eprintln!("HistoryStore batch rejected: {e}");
        }
    }

    /// Explicit cutoff in milliseconds, not an internally chosen one-year window.
    pub fn try_prune_older_than(&self, timestamp_ms: u64) -> rusqlite::Result<usize> {
        let ts = sql_timestamp(timestamp_ms)?;
        self.conn
            .execute("DELETE FROM klines WHERE timestamp < ?1", params![ts])
    }

    pub fn prune_older_than(&self, timestamp_ms: u64) {
        if let Err(e) = self.try_prune_older_than(timestamp_ms) {
            eprintln!("HistoryStore prune rejected: {e}");
        }
    }

    /// Returns all requested rows or an error; never discards a decode failure.
    /// Zero is an empty request. The existing 100,000-row budget is explicit.
    pub fn try_get_recent_candles(
        &self,
        coin_id: usize,
        limit: usize,
    ) -> rusqlite::Result<Vec<KlineRow>> {
        let id = sql_coin_id(coin_id)?;
        if limit > 100_000 {
            return Err(input_error("recent-candle limit exceeds 100000-row budget"));
        }
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            "SELECT coin_id,timestamp,open,high,low,close,volume FROM klines
             WHERE coin_id=?1 ORDER BY timestamp DESC LIMIT ?2",
        )?;
        let mut rows = stmt
            .query_map(params![id, limit as i64], decode_kline)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows.reverse();
        Ok(rows)
    }

    /// Legacy lossy wrapper: empty may mean error. Learning should use try_*.
    pub fn get_recent_candles(&self, coin_id: usize, limit: usize) -> Vec<KlineRow> {
        self.try_get_recent_candles(coin_id, limit)
            .unwrap_or_else(|e| {
                eprintln!("HistoryStore recent read failed: {e}");
                Vec::new()
            })
    }

    pub fn try_get_latest_timestamp(&self, coin_id: usize) -> rusqlite::Result<Option<u64>> {
        Ok(self
            .try_get_recent_candles(coin_id, 1)?
            .first()
            .map(|k| k.timestamp))
    }

    /// Legacy lossy wrapper. None is not proof of an empty archive.
    pub fn get_latest_timestamp(&self, coin_id: usize) -> Option<u64> {
        self.try_get_latest_timestamp(coin_id).unwrap_or_else(|e| {
            eprintln!("HistoryStore latest read failed: {e}");
            None
        })
    }

    /// Inclusive range; invalid coordinates and corrupt rows are errors.
    pub fn try_get_range(
        &self,
        coin_id: usize,
        start_ts: u64,
        end_ts: u64,
    ) -> rusqlite::Result<Vec<KlineRow>> {
        let id = sql_coin_id(coin_id)?;
        let start = sql_timestamp(start_ts)?;
        let end = sql_timestamp(end_ts)?;
        if start > end {
            return Err(input_error("range starts after its end"));
        }
        let mut stmt = self.conn.prepare(
            "SELECT coin_id,timestamp,open,high,low,close,volume FROM klines
             WHERE coin_id=?1 AND timestamp>=?2 AND timestamp<=?3 ORDER BY timestamp ASC",
        )?;
        let rows = stmt
            .query_map(params![id, start, end], decode_kline)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Legacy lossy wrapper: prefer try_get_range when coverage matters.
    pub fn get_range(&self, coin_id: usize, start_ts: u64, end_ts: u64) -> Vec<KlineRow> {
        self.try_get_range(coin_id, start_ts, end_ts)
            .unwrap_or_else(|e| {
                eprintln!("HistoryStore range read failed: {e}");
                Vec::new()
            })
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
    fn test_history_store_batch_insert_rejects_invalid_without_partial_commit() {
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
                open: f64::NAN, // Invalid candle rejects the whole batch.
                high: 3100.0,
                low: 3000.0,
                close: 3050.0,
                volume: 20.0,
            },
        ];

        assert!(store.try_insert_batch(&batch).is_err());
        let recent = store.get_recent_candles(1, 10);
        assert!(recent.is_empty());

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
