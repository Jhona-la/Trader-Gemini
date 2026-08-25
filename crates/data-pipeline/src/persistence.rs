use rusqlite::{Connection, OpenFlags, Result};
use std::path::Path;

pub struct WalStorage {
    conn: Connection,
}

impl WalStorage {
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open_with_flags(
            db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        // Configuración crítica de SQLite para HFT en Windows
        // 1. WAL mode permite lecturas y escrituras concurrentes
        // 2. SYNCHRONOUS = NORMAL relaja el fsync a disco sin riesgo crítico en WAL
        // 3. MMAP incrementa dramáticamente el I/O performance
        // 4. MEMORY_TEMP_STORE guarda temp files en RAM
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA mmap_size = 3000000000;
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = -64000;
            ",
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS tick_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                timestamp INTEGER NOT NULL
            )",
            [],
        )?;

        Ok(Self { conn })
    }

    // Persistencia ultrarrápida
    #[inline(always)]
    pub fn insert_tick(&self, symbol: &str, price: f64, volume: f64, timestamp: u64) -> Result<()> {
        let safe_price = if price.is_finite() && price > 0.0 { price } else { 0.0 };
        let safe_volume = if volume.is_finite() && volume >= 0.0 { volume } else { 0.0 };
        let mut stmt = self.conn.prepare_cached(
            "INSERT INTO tick_data (symbol, price, volume, timestamp) VALUES (?1, ?2, ?3, ?4)",
        )?;
        stmt.execute(rusqlite::params![symbol, safe_price, safe_volume, timestamp as i64])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_storage_creation_and_insertion() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_wal_storage.db");

        let storage = WalStorage::new(&path).unwrap();
        assert!(storage.insert_tick("BTCUSDT", 60000.0, 1.5, 1672531200000).is_ok());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_wal_storage_nan_and_negative_price_sanitization() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_wal_storage_nan.db");

        let storage = WalStorage::new(&path).unwrap();
        assert!(storage.insert_tick("ETHUSDT", f64::NAN, -10.0, 1672531200000).is_ok());

        let _ = std::fs::remove_file(path);
    }
}

