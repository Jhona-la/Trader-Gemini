use crossbeam_channel::{Receiver, Sender};
use rusqlite::{params, Connection, OpenFlags};
use std::thread;

/// Un evento de posesión para actualizar el Ledger Local
#[derive(Debug, Clone)]
pub struct LedgerEvent {
    pub symbol: String,
    pub position_side: String,      // "LONG" o "SHORT"
    pub strategy: String,           // Etiqueta histórica de procedencia, no partición del motor
    pub qty_delta: f64,             // Positivo (abrir) o Negativo (cerrar)
    pub price: f64,                 // Precio promedio
    pub is_absolute_override: bool, // Si es true, sobrescribe en vez de sumar (útil para conciliación)
}

pub struct PositionLedger {
    tx: Sender<LedgerEvent>,
}

/// A persisted ownership row, not proof of an exchange fill or a spectral model.
/// Keep the old label as provenance; it must not partition the public read API
/// into two trading engines or silently hide other labels.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnershipRecord {
    pub provenance_label: String,
    pub quantity: f64,
    pub entry_price: f64,
}

impl PositionLedger {
    /// Inicializa la BD en modo WAL y lanza el hilo de fondo.
    /// Encolar no acredita persistencia; no se garantiza latencia cero.
    pub fn new(db_path: &str) -> Self {
        // FIX #1438: Canal acotado a 100k eventos para protección de RAM
        let (tx, rx): (Sender<LedgerEvent>, Receiver<LedgerEvent>) =
            crossbeam_channel::bounded(100_000);
        let db_path = db_path.to_string();

        let _ = thread::Builder::new().name("ledger-wal-writer".into()).spawn(move || {
            if let Some(parent) = std::path::Path::new(&db_path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let mut conn = match Connection::open(&db_path) {
                Ok(c) => c,
                Err(e) => {
                    telemetry_engine::telemetry_err!("⚠️ [LEDGER] Fallo crítico al abrir DB: {}", e);
                    return;
                }
            };

            // Activar WAL mode para máximo rendimiento y concurrencia
            // FIX #1438: Estandarización de mmap a 64MB y cache a 32MB
            let _ = conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA temp_store = MEMORY;
                 PRAGMA mmap_size = 67108864;
                 PRAGMA cache_size = -32000;
                 CREATE TABLE IF NOT EXISTS position_ownership (
                     symbol TEXT NOT NULL,
                     position_side TEXT NOT NULL,
                     strategy TEXT NOT NULL,
                     qty REAL NOT NULL,
                     entry_price REAL NOT NULL,
                     updated_at INTEGER NOT NULL,
                     PRIMARY KEY(symbol, position_side, strategy)
                 );"
            );

            // Bucle infinito recibiendo eventos Lock-Free con Batching asíncrono
            while let Ok(event) = rx.recv() {
                let mut batch = vec![event];
                // Drenar el resto de la cola sin bloquear
                while let Ok(e) = rx.try_recv() {
                    batch.push(e);
                    // Límite de seguridad para no hacer transacciones gigantes
                    if batch.len() >= 1000 {
                        break;
                    }
                }

                // Iniciar transacción explícita para agrupar todas las escrituras (protege SSD y es ~10x más rápido)
                    if let Ok(tx) = conn.transaction() {
                        for event in batch {
                            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
                            // FIX #1535: Sanitización de cantidad y precio
                            let safe_qty = if event.qty_delta.is_finite() { event.qty_delta } else { 0.0 };
                            let safe_price = if event.price.is_finite() && event.price > 0.0 { event.price } else { 0.0 };

                            if event.is_absolute_override {
                                if safe_qty <= 1e-8 {
                                    if let Err(e) = tx.execute(
                                        "DELETE FROM position_ownership WHERE symbol=?1 AND position_side=?2 AND strategy=?3",
                                        params![event.symbol, event.position_side, event.strategy],
                                    ) {
                                        telemetry_engine::telemetry_err!("⚠️ [LEDGER] Delete failed: {}", e);
                                    }
                                } else {
                                    if let Err(e) = tx.execute(
                                        "INSERT INTO position_ownership (symbol, position_side, strategy, qty, entry_price, updated_at)
                                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                                         ON CONFLICT(symbol, position_side, strategy) DO UPDATE SET
                                             qty=excluded.qty,
                                             entry_price=excluded.entry_price,
                                             updated_at=excluded.updated_at",
                                        params![event.symbol, event.position_side, event.strategy, safe_qty, safe_price, ts as i64],
                                    ) {
                                        telemetry_engine::telemetry_err!("⚠️ [LEDGER] Insert override failed: {}", e);
                                    }
                                }
                            } else {
                                // Update del delta
                                // 1. Intentamos insertar nuevo si no existe
                                if let Err(e) = tx.execute(
                                    "INSERT OR IGNORE INTO position_ownership (symbol, position_side, strategy, qty, entry_price, updated_at)
                                     VALUES (?1, ?2, ?3, 0.0, ?4, ?5)",
                                    params![event.symbol, event.position_side, event.strategy, safe_price, ts as i64],
                                ) {
                                    telemetry_engine::telemetry_err!("⚠️ [LEDGER] Insert delta ignore failed: {}", e);
                                }

                                // 2. Actualizamos el delta de la cantidad y hacemos promedio del entry_price si qty sube
                                if let Err(e) = tx.execute(
                                    "UPDATE position_ownership SET
                                        entry_price = CASE WHEN ?4 > 0.0 AND (qty + ?4) > 0.0 AND ?5 > 0.0 THEN ((qty * entry_price) + (?4 * ?5)) / (qty + ?4) ELSE entry_price END,
                                        qty = qty + ?4,
                                        updated_at = ?6
                                     WHERE symbol=?1 AND position_side=?2 AND strategy=?3",
                                    params![event.symbol, event.position_side, event.strategy, safe_qty, safe_price, ts as i64],
                                ) {
                                    telemetry_engine::telemetry_err!("⚠️ [LEDGER] Update delta failed: {}", e);
                                }

                                // 3. Limpiamos si bajó a cero (o negativo por dust)
                                if let Err(e) = tx.execute(
                                    "DELETE FROM position_ownership WHERE qty <= 1e-8 AND symbol=?1 AND position_side=?2 AND strategy=?3",
                                    params![event.symbol, event.position_side, event.strategy],
                                ) {
                                    telemetry_engine::telemetry_err!("⚠️ [LEDGER] Delete dust failed: {}", e);
                                }
                            }
                        }
                        if let Err(e) = tx.commit() {
                            telemetry_engine::telemetry_err!("⚠️ [LEDGER] Commit failed: {}", e);
                        }
                    } else {
                        telemetry_engine::telemetry_err!("⚠️ [LEDGER] Transaction begin failed");
                    }
            }
        }).expect("Error iniciando hilo de Ledger WAL");

        Self { tx }
    }

    /// Enviar evento al hilo de escritura sin bloquear el Hot-Path
    pub fn push_event(&self, event: LedgerEvent) {
        // FIX #625 & #1548: Sanitización de valores no finitos para proteger la integridad de la base de datos
        if !event.qty_delta.is_finite()
            || event.qty_delta.abs() < 1e-12
            || !event.price.is_finite()
            || event.price <= 0.0
        {
            return;
        }
        let _ = self.tx.try_send(event); // No bloqueante para el hot path
    }

    /// Read all provenance labels without modifying or creating the database.
    /// Missing/corrupt schema or row values are errors, not an observed flat
    /// portfolio. The result is a local snapshot, NOT a fill/reservation ledger.
    pub fn read_ownership(
        db_path: &str,
        symbol: &str,
        position_side: &str,
    ) -> rusqlite::Result<Vec<OwnershipRecord>> {
        let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let mut stmt = conn.prepare(
            "SELECT strategy, qty, entry_price FROM position_ownership \
             WHERE symbol=?1 AND position_side=?2 ORDER BY strategy",
        )?;
        let rows = stmt.query_map(params![symbol, position_side], |row| {
            let record = OwnershipRecord {
                provenance_label: row.get(0)?,
                quantity: row.get(1)?,
                entry_price: row.get(2)?,
            };
            if record.provenance_label.is_empty()
                || !record.quantity.is_finite()
                || record.quantity < 0.0
                || !record.entry_price.is_finite()
                || record.entry_price < 0.0
                || (record.quantity > 0.0 && record.entry_price == 0.0)
            {
                return Err(rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Real,
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "invalid ownership label, quantity or entry price",
                    )),
                ));
            }
            Ok(record)
        })?;
        rows.collect()
    }

    /// Legacy two-label adapter. New consumers must use read_ownership.
    /// Returns None if any row cannot be represented without losing evidence;
    /// None is NOT a flat portfolio and must never authorize new exposure.
    pub fn get_ownership(
        db_path: &str,
        symbol: &str,
        position_side: &str,
    ) -> Option<(f64, f64, f64, f64)> {
        let records = Self::read_ownership(db_path, symbol, position_side).ok()?;
        // Retain historical serialization at this adapter only.
        let mut scalp_qty = 0.0;
        let mut scalp_price = 0.0;
        let mut swing_qty = 0.0;
        let mut swing_price = 0.0;
        let mut seen = std::collections::HashSet::new();
        for record in records {
            if !seen.insert(record.provenance_label.clone()) {
                return None;
            }
            match record.provenance_label.as_str() {
                "scalp" => {
                    scalp_qty = record.quantity;
                    scalp_price = record.entry_price;
                }
                "swing" => {
                    swing_qty = record.quantity;
                    swing_price = record.entry_price;
                }
                _ => return None,
            }
        }

        Some((scalp_qty, scalp_price, swing_qty, swing_price))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_ledger_crud() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!(
            "ledger_test_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db_str = db_path.to_str().unwrap();

        let ledger = PositionLedger::new(db_str);
        ledger.push_event(LedgerEvent {
            symbol: "BTCUSDT".into(),
            position_side: "LONG".into(),
            strategy: "scalp".into(),
            qty_delta: 0.1,
            price: 60000.0,
            is_absolute_override: true,
        });

        let mut ownership = None;
        for _ in 0..20 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            if let Some(res) = PositionLedger::get_ownership(db_str, "BTCUSDT", "LONG") {
                if (res.0 - 0.1).abs() < 1e-6 {
                    ownership = Some(res);
                    break;
                }
            }
        }
        assert!(
            ownership.is_some(),
            "PositionLedger debe haber persistido la propiedad"
        );
        let (scalp_qty, scalp_price, _, _) = ownership.unwrap();
        assert!((scalp_qty - 0.1).abs() < 1e-6);
        assert!((scalp_price - 60000.0).abs() < 1e-6);

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_position_ledger_incremental_delta_and_nan_immunity() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!(
            "ledger_delta_test_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db_str = db_path.to_str().unwrap();

        let ledger = PositionLedger::new(db_str);
        // Push event with NaN price (rejected by push_event safety filter)
        ledger.push_event(LedgerEvent {
            symbol: "ETHUSDT".into(),
            position_side: "SHORT".into(),
            strategy: "swing".into(),
            qty_delta: 0.5,
            price: f64::NAN,
            is_absolute_override: false,
        });

        // Push two valid delta events
        ledger.push_event(LedgerEvent {
            symbol: "ETHUSDT".into(),
            position_side: "SHORT".into(),
            strategy: "swing".into(),
            qty_delta: 0.5,
            price: 3000.0,
            is_absolute_override: false,
        });

        ledger.push_event(LedgerEvent {
            symbol: "ETHUSDT".into(),
            position_side: "SHORT".into(),
            strategy: "swing".into(),
            qty_delta: 0.5,
            price: 3100.0,
            is_absolute_override: false,
        });

        let mut persisted = false;
        for _ in 0..30 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            if let Some(res) = PositionLedger::get_ownership(db_str, "ETHUSDT", "SHORT") {
                if (res.2 - 1.0).abs() < 1e-5 && (res.3 - 3050.0).abs() < 1.0 {
                    persisted = true;
                    break;
                }
            }
        }
        assert!(persisted, "Debe acumular los deltas incrementales correctamente a 1.0 ETH y precio promedio 3050.0");
        let _ = std::fs::remove_file(db_path);
    }

    // XXX: passing diagnostics of OPEN write-side contracts. No worker/DB.
    fn pending_event(qty: f64) -> LedgerEvent {
        LedgerEvent {
            symbol: "XXXUSDT".into(),
            position_side: "LONG".into(),
            strategy: "continuous".into(),
            qty_delta: qty,
            price: 100.0,
            is_absolute_override: true,
        }
    }

    #[test]
    fn xxx_open_absolute_flat_snapshot_is_discarded_before_writer() {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let ledger = PositionLedger { tx };
        ledger.push_event(pending_event(0.0));
        assert!(rx.try_recv().is_err(), "known defect: absolute zero never reaches delete branch");
    }

    #[test]
    fn xxx_open_small_nonzero_exposure_is_discarded_without_reason() {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let ledger = PositionLedger { tx };
        ledger.push_event(pending_event(1e-13));
        assert!(rx.try_recv().is_err(), "known defect: absolute unitless dust veto");
    }

    #[test]
    fn xxx_open_queue_full_discards_a_valid_event_without_ack() {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let ledger = PositionLedger { tx };
        ledger.push_event(pending_event(1.0));
        ledger.push_event(pending_event(2.0));
        assert_eq!(rx.try_recv().unwrap().qty_delta, 1.0);
        assert!(rx.try_recv().is_err(), "known defect: second valid event vanished");
    }
}
