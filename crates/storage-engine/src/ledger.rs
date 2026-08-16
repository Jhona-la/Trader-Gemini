use crossbeam_channel::{unbounded, Receiver, Sender};
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
        let (tx, rx): (Sender<LedgerEvent>, Receiver<LedgerEvent>) = unbounded();
        let db_path = db_path.to_string();

        thread::Builder::new().name("ledger-wal-writer".into()).spawn(move || {
            let mut conn = match Connection::open(&db_path) {
                Ok(c) => c,
                Err(e) => {
                    telemetry_engine::telemetry_err!("⚠️ [LEDGER] Fallo crítico al abrir DB: {}", e);
                    return;
                }
            };

            // Activar WAL mode para máximo rendimiento y concurrencia
            let _ = conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA temp_store = MEMORY;
                 PRAGMA mmap_size = 268435456;
                 PRAGMA cache_size = -64000;
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
                        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;

                        if event.is_absolute_override {
                            if event.qty_delta <= 1e-8 {
                                let _ = tx.execute(
                                    "DELETE FROM position_ownership WHERE symbol=?1 AND position_side=?2 AND strategy=?3",
                                    params![event.symbol, event.position_side, event.strategy],
                                );
                            } else {
                                let _ = tx.execute(
                                    "INSERT INTO position_ownership (symbol, position_side, strategy, qty, entry_price, updated_at)
                                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                                     ON CONFLICT(symbol, position_side, strategy) DO UPDATE SET
                                        qty=excluded.qty,
                                        entry_price=excluded.entry_price,
                                        updated_at=excluded.updated_at",
                                    params![event.symbol, event.position_side, event.strategy, event.qty_delta, event.price, ts as i64],
                                );
                            }
                        } else {
                            // Update del delta
                            // 1. Intentamos insertar nuevo si no existe
                            let _ = tx.execute(
                                "INSERT OR IGNORE INTO position_ownership (symbol, position_side, strategy, qty, entry_price, updated_at)
                                 VALUES (?1, ?2, ?3, 0.0, ?4, ?5)",
                                params![event.symbol, event.position_side, event.strategy, event.price, ts as i64],
                            );

                            // 2. Actualizamos el delta de la cantidad y hacemos promedio del entry_price si qty sube
                            let _ = tx.execute(
                                "UPDATE position_ownership SET
                                    entry_price = CASE WHEN ?4 > 0.0 THEN ((qty * entry_price) + (?4 * ?5)) / (qty + ?4) ELSE entry_price END,
                                    qty = qty + ?4,
                                    updated_at = ?6
                                 WHERE symbol=?1 AND position_side=?2 AND strategy=?3",
                                params![event.symbol, event.position_side, event.strategy, event.qty_delta, event.price, ts as i64],
                            );

                            // 3. Limpiamos si bajó a cero (o negativo por dust)
                            let _ = tx.execute(
                                "DELETE FROM position_ownership WHERE qty <= 1e-8 AND symbol=?1 AND position_side=?2 AND strategy=?3",
                                params![event.symbol, event.position_side, event.strategy],
                            );
                        }
                    }
                    let _ = tx.commit();
                }
            }
        }).expect("Error iniciando hilo de Ledger WAL");

        Self { tx }
    }

    /// Enviar evento al hilo de escritura sin bloquear el Hot-Path
    pub fn push_event(&self, event: LedgerEvent) {
        let _ = self.tx.send(event); // Ignoramos si se cierra (al apagar el bot)
    }

    /// Método síncrono para inicialización: Leer estado del Ledger
    pub fn get_ownership(
        db_path: &str,
        symbol: &str,
        position_side: &str,
    ) -> Option<(f64, f64, f64, f64)> {
        let conn = Connection::open(db_path).ok()?;

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
