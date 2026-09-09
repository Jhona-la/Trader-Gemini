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

        // FIX #1426: Configuración de grado institucional optimizada para host de 16GB RAM
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA temp_store=MEMORY;
             PRAGMA mmap_size=67108864; -- 64MB MMAP
             PRAGMA busy_timeout=5000;",
        )?;

        // Tabla de intenciones de posición con clave primaria compuesta (coin_id, horizon)
        // Permite coexistencia simultánea e independiente de posiciones Scalp y Swing en la misma moneda
        conn.execute(
            "CREATE TABLE IF NOT EXISTS position_intent (
                coin_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                horizon TEXT NOT NULL,
                is_long BOOLEAN NOT NULL,
                entry_price REAL NOT NULL,
                qty REAL NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (coin_id, horizon)
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
        // FIX #1427: Sanitización estricta contra NaNs o valores no positivos
        if !entry_price.is_finite() || entry_price <= 0.0 || !qty.is_finite() || qty <= 0.0 {
            return Ok(());
        }

        let horizon_str = match horizon {
            HorizonIntent::Scalp => "SCALP",
            HorizonIntent::Swing => "SWING",
        };

        self.conn.execute(
            "INSERT INTO position_intent (coin_id, symbol, horizon, is_long, entry_price, qty, updated_at) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(coin_id, horizon) DO UPDATE SET 
             symbol=excluded.symbol, is_long=excluded.is_long, entry_price=excluded.entry_price, qty=excluded.qty, updated_at=excluded.updated_at",
            params![coin_id as i64, symbol, horizon_str, is_long, entry_price, qty, ts as i64],
        )?;
        Ok(())
    }

    /// Elimina una posición específica por horizonte o todas las posiciones de la moneda.
    #[inline]
    pub fn clear_position_horizon(&self, coin_id: usize, horizon: HorizonIntent) -> Result<()> {
        let horizon_str = match horizon {
            HorizonIntent::Scalp => "SCALP",
            HorizonIntent::Swing => "SWING",
        };
        self.conn.execute(
            "DELETE FROM position_intent WHERE coin_id = ?1 AND horizon = ?2",
            params![coin_id as i64, horizon_str],
        )?;
        Ok(())
    }

    /// Elimina todas las posiciones de una moneda.
    #[inline]
    pub fn clear_position(&self, coin_id: usize) -> Result<()> {
        self.conn.execute(
            "DELETE FROM position_intent WHERE coin_id = ?1",
            params![coin_id as i64],
        )?;
        Ok(())
    }

    /// Recupera la intención por horizonte específico.
    pub fn get_position_intent_by_horizon(
        &self,
        coin_id: usize,
        horizon: HorizonIntent,
    ) -> Result<Option<(HorizonIntent, bool, f64, f64)>> {
        let horizon_str = match horizon {
            HorizonIntent::Scalp => "SCALP",
            HorizonIntent::Swing => "SWING",
        };
        let mut stmt = self.conn.prepare(
            "SELECT horizon, is_long, entry_price, qty FROM position_intent WHERE coin_id = ?1 AND horizon = ?2",
        )?;
        let mut rows = stmt.query(params![coin_id as i64, horizon_str])?;

        if let Some(row) = rows.next()? {
            let h_str: String = row.get(0)?;
            let is_long: bool = row.get(1)?;
            let entry_price: f64 = row.get(2)?;
            let qty: f64 = row.get(3)?;

            let h = match h_str.as_str() {
                "SCALP" => HorizonIntent::Scalp,
                _ => HorizonIntent::Swing,
            };

            Ok(Some((h, is_long, entry_price, qty)))
        } else {
            Ok(None)
        }
    }

    /// Recupera la intención (si existe) para restaurar el Engine desde un reinicio.
    pub fn get_position_intent(
        &self,
        coin_id: usize,
    ) -> Result<Option<(HorizonIntent, bool, f64, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT horizon, is_long, entry_price, qty FROM position_intent WHERE coin_id = ?1 ORDER BY updated_at DESC LIMIT 1",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_db_dual_horizon_crud() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_state_db_dual_horizon.db");

        let db = StateDb::new(&db_path).unwrap();

        // Save Scalp Long
        db.save_position_intent(
            0,
            "BTCUSDT",
            HorizonIntent::Scalp,
            true,
            60000.0,
            0.1,
            1672531200000,
        )
        .unwrap();

        // Save Swing Short on the same coin_id (coexistence in Hedge Mode)
        db.save_position_intent(
            0,
            "BTCUSDT",
            HorizonIntent::Swing,
            false,
            60000.0,
            0.5,
            1672531200000,
        )
        .unwrap();

        let scalp = db
            .get_position_intent_by_horizon(0, HorizonIntent::Scalp)
            .unwrap()
            .unwrap();
        assert_eq!(scalp.0, HorizonIntent::Scalp);
        assert!(scalp.1); // Long
        assert_eq!(scalp.2, 60000.0);
        assert_eq!(scalp.3, 0.1);

        let swing = db
            .get_position_intent_by_horizon(0, HorizonIntent::Swing)
            .unwrap()
            .unwrap();
        assert_eq!(swing.0, HorizonIntent::Swing);
        assert!(!swing.1); // Short
        assert_eq!(swing.2, 60000.0);
        assert_eq!(swing.3, 0.5);

        // Clear Scalp only
        db.clear_position_horizon(0, HorizonIntent::Scalp).unwrap();
        assert!(db
            .get_position_intent_by_horizon(0, HorizonIntent::Scalp)
            .unwrap()
            .is_none());
        assert!(db
            .get_position_intent_by_horizon(0, HorizonIntent::Swing)
            .unwrap()
            .is_some());

        // Clear all
        db.clear_position(0).unwrap();
        assert!(db
            .get_position_intent_by_horizon(0, HorizonIntent::Swing)
            .unwrap()
            .is_none());

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_state_db_nan_sanitization() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_state_db_nan.db");

        let db = StateDb::new(&db_path).unwrap();
        assert!(db
            .save_position_intent(
                1,
                "ETHUSDT",
                HorizonIntent::Scalp,
                true,
                f64::NAN,
                1.0,
                1000
            )
            .is_ok());
        assert!(db
            .save_position_intent(1, "ETHUSDT", HorizonIntent::Scalp, true, 2000.0, -1.0, 1000)
            .is_ok());
        assert!(db
            .get_position_intent_by_horizon(1, HorizonIntent::Scalp)
            .unwrap()
            .is_none());

        let _ = std::fs::remove_file(db_path);
    }
}
