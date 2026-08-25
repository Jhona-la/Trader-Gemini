use crossbeam_channel::{Receiver, Sender};
use rusqlite::{params, Connection};
use std::thread;

/// Un evento de posesión para actualizar el Ledger Local
#[derive(Debug, Clone)]
pub struct LedgerEvent {
    pub symbol: String,
    pub position_side: String,      // "LONG" o "SHORT"
    pub strategy: String,           // "scalp" o "swing"
    pub qty_delta: f64,             // Positivo (abrir) o Negativo (cerrar)
    pub price: f64,                 // Precio promedio
    pub is_absolute_override: bool, // Si es true, sobrescribe en vez de sumar (útil para conciliación)
}

pub struct PositionLedger {
    tx: Sender<LedgerEvent>,
}

impl PositionLedger {
    /// Inicializa la BD en modo WAL y lanza el hilo de fondo (Zero-Latency)
    pub fn new(db_path: &str) -> Self {
        // FIX #1438: Canal acotado a 100k eventos para protección de RAM
        let (tx, rx): (Sender<LedgerEvent>, Receiver<LedgerEvent>) = crossbeam_channel::bounded(100_000);
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
        if !event.qty_delta.is_finite() || event.qty_delta.abs() < 1e-12 || !event.price.is_finite() || event.price <= 0.0 {
            return;
        }
        let _ = self.tx.try_send(event); // No bloqueante para el hot path
    }

    /// Método síncrono para inicialización: Leer estado del Ledger
    pub fn get_ownership(
        db_path: &str,
        symbol: &str,
        position_side: &str,
    ) -> Option<(f64, f64, f64, f64)> {
        let conn = Connection::open(db_path).ok()?;
        let _ = conn.busy_timeout(std::time::Duration::from_secs(5));

        // Retornamos (scalp_qty, scalp_price, swing_qty, swing_price)
        let mut scalp_qty = 0.0;
        let mut scalp_price = 0.0;
        let mut swing_qty = 0.0;
        let mut swing_price = 0.0;

        let mut stmt = conn.prepare("SELECT strategy, qty, entry_price FROM position_ownership WHERE symbol=?1 AND position_side=?2").ok()?;
        let mut rows = stmt.query(params![symbol, position_side]).ok()?;

        while let Ok(Some(row)) = rows.next() {
            let strat: String = row.get(0).unwrap_or_default();
            let q: f64 = row.get(1).unwrap_or(0.0);
            let p: f64 = row.get(2).unwrap_or(0.0);
            if strat == "scalp" {
                scalp_qty = q;
                scalp_price = p;
            } else if strat == "swing" {
                swing_qty = q;
                swing_price = p;
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
        let db_path = temp_dir.join(format!("ledger_test_{}.db", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
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
        assert!(ownership.is_some(), "PositionLedger debe haber persistido la propiedad");
        let (scalp_qty, scalp_price, _, _) = ownership.unwrap();
        assert!((scalp_qty - 0.1).abs() < 1e-6);
        assert!((scalp_price - 60000.0).abs() < 1e-6);

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_position_ledger_incremental_delta_and_nan_immunity() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!("ledger_delta_test_{}.db", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
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

}
