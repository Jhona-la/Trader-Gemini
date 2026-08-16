use rusqlite::{params, Connection, OpenFlags, Result};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizonIntent {
    Scalp,
    Swing,
}

/// Base de Datos de Estado con Atomicidad y Recuperación Rápida
pub struct StateDb {
    conn: Connection,
}

impl StateDb {
    /// Inicializa la conexión SQLite en modo WAL para nanosegundos de latencia y protección contra apagones.
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open_with_flags(
            db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        // Configuración de grado institucional para HFT/WAL
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA temp_store=MEMORY;
             PRAGMA mmap_size=3000000000;",
        )?;

        // Tabla de intenciones de posición
        conn.execute(
            "CREATE TABLE IF NOT EXISTS position_intent (
                coin_id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                horizon TEXT NOT NULL,
                is_long BOOLEAN NOT NULL,
                entry_price REAL NOT NULL,
                qty REAL NOT NULL,
                updated_at INTEGER NOT NULL
            )",
            [],
        )?;

        Ok(Self { conn })
    }

    /// Guarda o actualiza atómicamente la intención de una posición.
    #[inline]
    pub fn save_position_intent(
        &self,
        coin_id: usize,
        symbol: &str,
        horizon: HorizonIntent,
        is_long: bool,
        entry_price: f64,
        qty: f64,
        ts: u64,
    ) -> Result<()> {
        let horizon_str = match horizon {
            HorizonIntent::Scalp => "SCALP",
            HorizonIntent::Swing => "SWING",
        };

        self.conn.execute(
            "INSERT INTO position_intent (coin_id, symbol, horizon, is_long, entry_price, qty, updated_at) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(coin_id) DO UPDATE SET 
             symbol=excluded.symbol, horizon=excluded.horizon, is_long=excluded.is_long, entry_price=excluded.entry_price, qty=excluded.qty, updated_at=excluded.updated_at",
            params![coin_id as i64, symbol, horizon_str, is_long, entry_price, qty, ts as i64],
        )?;
        Ok(())
    }

    /// Elimina una posición (cuando se cierra).
    #[inline]
    pub fn clear_position(&self, coin_id: usize) -> Result<()> {
        self.conn.execute(
            "DELETE FROM position_intent WHERE coin_id = ?1",
            params![coin_id as i64],
        )?;
        Ok(())
    }

    /// Recupera la intención (si existe) para restaurar el Engine desde un reinicio.
    pub fn get_position_intent(
        &self,
        coin_id: usize,
    ) -> Result<Option<(HorizonIntent, bool, f64, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT horizon, is_long, entry_price, qty FROM position_intent WHERE coin_id = ?1",
        )?;
        let mut rows = stmt.query(params![coin_id as i64])?;

        if let Some(row) = rows.next()? {
            let horizon_str: String = row.get(0)?;
            let is_long: bool = row.get(1)?;
            let entry_price: f64 = row.get(2)?;
            let qty: f64 = row.get(3)?;

            let horizon = match horizon_str.as_str() {
                "SCALP" => HorizonIntent::Scalp,
                _ => HorizonIntent::Swing,
            };

            Ok(Some((horizon, is_long, entry_price, qty)))
        } else {
            Ok(None)
        }
    }
}
