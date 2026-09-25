use rusqlite::{params, Connection, OpenFlags, Result};
use std::path::Path;

#[cfg(test)]
mod audit_xiii_tests {
    use super::*;

    #[test]
    fn invalid_write_is_an_error_not_acknowledged_persistence() {
        let db = StateDb::new(":memory:").unwrap();
        assert!(db
            .save_position_intent(
                0,
                "BTCUSDT",
                HorizonIntent::Continuous,
                true,
                f64::NAN,
                1.0,
                10
            )
            .is_err());
        assert!(db
            .save_position_intent(0, "", HorizonIntent::Continuous, true, 10.0, 1.0, 10)
            .is_err());
        assert!(db.get_position_intent(0).unwrap().is_none());
    }

    #[test]
    fn unknown_horizon_is_not_reinterpreted_as_swing() {
        let db = StateDb::new(":memory:").unwrap();
        db.conn
            .execute(
                "INSERT INTO position_intent VALUES (0,'BTCUSDT','UNRECOGNIZED',1,10,1,100)",
                [],
            )
            .unwrap();
        assert!(db.get_position_intent(0).is_err());
    }

    #[test]
    fn persisted_invalid_side_and_quantity_are_rejected() {
        for (side, quantity) in [(2, 1.0), (1, -1.0)] {
            let db = StateDb::new(":memory:").unwrap();
            db.conn
                .execute(
                    "INSERT INTO position_intent VALUES (0,'BTCUSDT','CONTINUOUS',?1,10,?2,100)",
                    params![side, quantity],
                )
                .unwrap();
            assert!(db.get_position_intent(0).is_err());
            assert!(db
                .get_position_intent_by_horizon(0, HorizonIntent::Continuous)
                .is_err());
        }
    }

    #[test]
    fn integer_domains_do_not_alias_other_keys_or_times() {
        let db = StateDb::new(":memory:").unwrap();
        assert!(db
            .save_position_intent(
                0,
                "BTCUSDT",
                HorizonIntent::Continuous,
                true,
                10.0,
                1.0,
                u64::MAX
            )
            .is_err());
        if usize::BITS == 64 {
            assert!(db
                .save_position_intent(
                    usize::MAX,
                    "BTCUSDT",
                    HorizonIntent::Continuous,
                    true,
                    10.0,
                    1.0,
                    1
                )
                .is_err());
            assert!(db.clear_position(usize::MAX).is_err());
        }
    }

    #[test]
    fn all_declared_legacy_tags_remain_readable_without_inventing_a_tag() {
        let db = StateDb::new(":memory:").unwrap();
        for (i, h) in [
            HorizonIntent::Continuous,
            HorizonIntent::Scalp,
            HorizonIntent::Swing,
        ]
        .into_iter()
        .enumerate()
        {
            db.save_position_intent(i, "BTCUSDT", h, false, 10.0, 1.0, 100)
                .unwrap();
            assert_eq!(db.get_position_intent(i).unwrap().unwrap().0, h);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HorizonIntent {
    #[default]
    Continuous,
    Scalp,
    Swing,
}

fn invalid_input(message: &str) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message,
    )))
}

fn sql_coin_id(coin_id: usize) -> Result<i64> {
    i64::try_from(coin_id).map_err(|_| invalid_input("coin_id exceeds SQLite INTEGER domain"))
}

impl HorizonIntent {
    // Legacy tags remain readable for migration; they are not a spectral coordinate.
    fn storage_tag(self) -> &'static str {
        match self {
            Self::Continuous => "CONTINUOUS",
            Self::Scalp => "SCALP",
            Self::Swing => "SWING",
        }
    }
}

fn corrupt_column(index: usize, kind: rusqlite::types::Type, message: &str) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        index,
        kind,
        Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            message,
        )),
    )
}

fn decode_intent(row: &rusqlite::Row<'_>) -> Result<(HorizonIntent, bool, f64, f64)> {
    use rusqlite::types::Type;
    let tag: String = row.get(0)?;
    let horizon = match tag.as_str() {
        "CONTINUOUS" => HorizonIntent::Continuous,
        "SCALP" => HorizonIntent::Scalp,
        "SWING" => HorizonIntent::Swing,
        _ => {
            return Err(corrupt_column(
                0,
                Type::Text,
                "unknown horizon tag; migration required",
            ))
        }
    };
    let side: i64 = row.get(1)?;
    if side != 0 && side != 1 {
        return Err(corrupt_column(1, Type::Integer, "is_long must be 0 or 1"));
    }
    let entry_price: f64 = row.get(2)?;
    let qty: f64 = row.get(3)?;
    for (index, value) in [(2, entry_price), (3, qty)] {
        if !value.is_finite() || value <= 0.0 {
            return Err(corrupt_column(
                index,
                Type::Real,
                "price and quantity must be finite and positive",
            ));
        }
    }
    let symbol: String = row.get(4)?;
    if symbol.trim().is_empty() {
        return Err(corrupt_column(4, Type::Text, "symbol must not be blank"));
    }
    let updated_at: i64 = row.get(5)?;
    if updated_at < 0 {
        return Err(corrupt_column(
            5,
            Type::Integer,
            "timestamp must be nonnegative",
        ));
    }
    Ok((horizon, side == 1, entry_price, qty))
}

/// Atomic individual position intentions, not a complete engine recovery snapshot.
pub struct StateDb {
    conn: Connection,
}

impl StateDb {
    /// Opens SQLite in WAL mode. Latency is unmeasured; NORMAL does not guarantee
    /// survival of the latest committed transactions after power loss.
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
        // Compatibility key: one intention per legacy tag, NOT arbitrary spectral slots.
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
        // A successful return acknowledges a write, never silently discards invalid input.
        if !entry_price.is_finite() || entry_price <= 0.0 || !qty.is_finite() || qty <= 0.0 {
            return Err(invalid_input(
                "price and quantity must be finite and positive",
            ));
        }
        if symbol.trim().is_empty() {
            return Err(invalid_input("symbol must not be blank"));
        }
        let coin_id = sql_coin_id(coin_id)?;
        let ts = i64::try_from(ts)
            .map_err(|_| invalid_input("timestamp exceeds SQLite INTEGER domain"))?;
        let horizon_str = horizon.storage_tag();

        self.conn.execute(
            "INSERT INTO position_intent (coin_id, symbol, horizon, is_long, entry_price, qty, updated_at) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(coin_id, horizon) DO UPDATE SET 
             symbol=excluded.symbol, is_long=excluded.is_long, entry_price=excluded.entry_price, qty=excluded.qty, updated_at=excluded.updated_at",
            params![coin_id, symbol, horizon_str, is_long, entry_price, qty, ts],
        )?;
        Ok(())
    }

    /// Elimina una posición específica por horizonte o todas las posiciones de la moneda.
    #[inline]
    pub fn clear_position_horizon(&self, coin_id: usize, horizon: HorizonIntent) -> Result<()> {
        let horizon_str = horizon.storage_tag();
        self.conn.execute(
            "DELETE FROM position_intent WHERE coin_id = ?1 AND horizon = ?2",
            params![sql_coin_id(coin_id)?, horizon_str],
        )?;
        Ok(())
    }

    /// Elimina todas las posiciones de una moneda.
    #[inline]
    pub fn clear_position(&self, coin_id: usize) -> Result<()> {
        self.conn.execute(
            "DELETE FROM position_intent WHERE coin_id = ?1",
            params![sql_coin_id(coin_id)?],
        )?;
        Ok(())
    }

    /// Recupera la intención por horizonte específico.
    pub fn get_position_intent_by_horizon(
        &self,
        coin_id: usize,
        horizon: HorizonIntent,
    ) -> Result<Option<(HorizonIntent, bool, f64, f64)>> {
        let horizon_str = horizon.storage_tag();
        let mut stmt = self.conn.prepare(
            "SELECT horizon, is_long, entry_price, qty, symbol, updated_at FROM position_intent WHERE coin_id = ?1 AND horizon = ?2",
        )?;
        let mut rows = stmt.query(params![sql_coin_id(coin_id)?, horizon_str])?;

        if let Some(row) = rows.next()? {
            decode_intent(row).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Retrieves only the latest intention, not all positions or the full engine state.
    pub fn get_position_intent(
        &self,
        coin_id: usize,
    ) -> Result<Option<(HorizonIntent, bool, f64, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT horizon, is_long, entry_price, qty, symbol, updated_at FROM position_intent WHERE coin_id = ?1 ORDER BY updated_at DESC LIMIT 1",
        )?;
        let mut rows = stmt.query(params![sql_coin_id(coin_id)?])?;

        if let Some(row) = rows.next()? {
            decode_intent(row).map(Some)
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
    fn test_state_db_invalid_values_are_errors() {
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
            .is_err());
        assert!(db
            .save_position_intent(1, "ETHUSDT", HorizonIntent::Scalp, true, 2000.0, -1.0, 1000)
            .is_err());
        assert!(db
            .get_position_intent_by_horizon(1, HorizonIntent::Scalp)
            .unwrap()
            .is_none());

        let _ = std::fs::remove_file(db_path);
    }
}
