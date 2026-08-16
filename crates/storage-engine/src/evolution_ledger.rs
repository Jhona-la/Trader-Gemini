use crossbeam_channel::{unbounded, Receiver, Sender};
use rusqlite::{params, Connection};
use std::thread;

/// Evento atómico para guardar pesos/mutaciones del Genoma Evolutivo (Lock-Free)
#[derive(Debug, Clone)]
pub struct GenomeUpdateEvent {
    pub coin_id: usize,
    pub symbol: String,
    pub strategy: String, // "scalp" o "swing"
    pub weight: f64,
}

pub struct EvolutionLedger {
    tx: Sender<GenomeUpdateEvent>,
}

impl EvolutionLedger {
    pub fn new(db_path: &str) -> Self {
        let (tx, rx): (Sender<GenomeUpdateEvent>, Receiver<GenomeUpdateEvent>) = unbounded();
        let db_path = db_path.to_string();

        thread::Builder::new()
            .name("evolution-wal-writer".into())
            .spawn(move || {
                let mut conn = match Connection::open(&db_path) {
                    Ok(c) => c,
                    Err(e) => {
                        telemetry_engine::telemetry_err!(
                            "⚠️ [EVOLUTION-LEDGER] Fallo crítico al abrir DB: {}",
                            e
                        );
                        return;
                    }
                };

                // Activar WAL mode para zero blocking
                let _ = conn.execute_batch(
                    "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA temp_store = MEMORY;
                 CREATE TABLE IF NOT EXISTS genome_weights (
                     symbol TEXT NOT NULL,
                     strategy TEXT NOT NULL,
                     weight REAL NOT NULL,
                     updated_at INTEGER NOT NULL,
                     PRIMARY KEY(symbol, strategy)
                 );",
                );

                while let Ok(event) = rx.recv() {
                    let mut batch = vec![event];
                    while let Ok(e) = rx.try_recv() {
                        batch.push(e);
                        if batch.len() >= 1000 {
                            break;
                        }
                    }

                    if let Ok(tx) = conn.transaction() {
                        for event in batch {
                            let ts = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64;
                            let _ = tx.execute(
                                "INSERT INTO genome_weights (symbol, strategy, weight, updated_at)
                             VALUES (?1, ?2, ?3, ?4)
                             ON CONFLICT(symbol, strategy) DO UPDATE SET
                                weight=excluded.weight,
                                updated_at=excluded.updated_at",
                                params![event.symbol, event.strategy, event.weight, ts as i64],
                            );
                        }
                        let _ = tx.commit();
                    }
                }
            })
            .expect("Failed to spawn evolution-wal-writer");

        Self { tx }
    }

    #[inline]
    pub fn save_weight(&self, coin_id: usize, symbol: String, strategy: String, weight: f64) {
        let _ = self.tx.try_send(GenomeUpdateEvent {
            coin_id,
            symbol,
            strategy,
            weight,
        });
    }
}
