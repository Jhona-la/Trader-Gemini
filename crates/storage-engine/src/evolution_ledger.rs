use crossbeam_channel::{Receiver, Sender};
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
        // FIX #1434: Canal acotado (100k eventos) para evitar sobrecarga de memoria en host de 16GB
        let (tx, rx): (Sender<GenomeUpdateEvent>, Receiver<GenomeUpdateEvent>) = crossbeam_channel::bounded(100_000);
        let db_path = db_path.to_string();

        let _ = thread::Builder::new()
            .name("evolution-wal-writer".into())
            .spawn(move || {
                if let Some(parent) = std::path::Path::new(&db_path).parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
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

                let _ = conn.busy_timeout(std::time::Duration::from_secs(5));

                // Activar WAL mode para zero blocking
                let _ = conn.execute_batch(
                    "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA temp_store = MEMORY;
                 CREATE TABLE IF NOT EXISTS genome_weights (
                     coin_id INTEGER NOT NULL DEFAULT 0,
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
                            // FIX #724: Uso de unwrap_or_default para inmunidad ante saltos de reloj NTP
                            let ts = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64;
                            let safe_weight = if event.weight.is_finite() { event.weight } else { 0.0 };
                            if let Err(e) = tx.execute(
                                "INSERT INTO genome_weights (coin_id, symbol, strategy, weight, updated_at)
                             VALUES (?1, ?2, ?3, ?4, ?5)
                             ON CONFLICT(symbol, strategy) DO UPDATE SET
                                coin_id=excluded.coin_id,
                                weight=excluded.weight,
                                updated_at=excluded.updated_at",
                                params![event.coin_id as i64, event.symbol, event.strategy, safe_weight, (ts.min(i64::MAX as u64)) as i64],
                            ) {
                                telemetry_engine::telemetry_err!("⚠️ [EVOLUTION-LEDGER] Execute error: {}", e);
                            }
                        }
                        if let Err(e) = tx.commit() {
                            telemetry_engine::telemetry_err!("⚠️ [EVOLUTION-LEDGER] Commit failed: {}", e);
                        }
                    } else {
                        telemetry_engine::telemetry_err!("⚠️ [EVOLUTION-LEDGER] Transaction begin failed");
                    }
                }
            });

        Self { tx }
    }

    #[inline]
    pub fn save_weight(&self, coin_id: usize, symbol: String, strategy: String, weight: f64) {
        // FIX #725: Sanitizar peso genético antes de enviar al canal WAL
        let safe_weight = if weight.is_finite() { weight } else { 0.0 };
        let _ = self.tx.try_send(GenomeUpdateEvent {
            coin_id,
            symbol,
            strategy,
            weight: safe_weight,
        });
    }

    /// FIX #1415: Método síncrono para inicialización / bootstrap: Recupera todos los pesos aprendidos
    pub fn load_all_weights(db_path: &str) -> std::collections::HashMap<(String, String), f64> {
        let mut weights = std::collections::HashMap::new();
        let conn = match Connection::open(db_path) {
            Ok(c) => c,
            Err(_) => return weights,
        };
        let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
        let mut stmt = match conn.prepare("SELECT symbol, strategy, weight FROM genome_weights") {
            Ok(s) => s,
            Err(_) => return weights,
        };
        let rows = match stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        }) {
            Ok(r) => r,
            Err(_) => return weights,
        };

        for item in rows.flatten() {
            weights.insert((item.0, item.1), item.2);
        }
        weights
    }

    /// FIX #1415: Carga el peso específico para un símbolo y estrategia ("scalp" o "swing")
    pub fn load_weight(db_path: &str, symbol: &str, strategy: &str) -> Option<f64> {
        let conn = Connection::open(db_path).ok()?;
        let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
        let mut stmt = conn
            .prepare("SELECT weight FROM genome_weights WHERE symbol=?1 AND strategy=?2")
            .ok()?;
        let weight: f64 = stmt
            .query_row(params![symbol, strategy], |row| row.get(0))
            .ok()?;
        Some(weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolution_ledger_save_and_load_weight() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!("evolution_test_{}.db", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let db_str = db_path.to_str().unwrap();

        let ledger = EvolutionLedger::new(db_str);
        ledger.save_weight(0, "BTCUSDT".into(), "scalp".into(), 1.5);

        let mut loaded = None;
        for _ in 0..30 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            if let Some(w) = EvolutionLedger::load_weight(db_str, "BTCUSDT", "scalp") {
                if (w - 1.5).abs() < 1e-6 {
                    loaded = Some(w);
                    break;
                }
            }
        }

        assert_eq!(loaded, Some(1.5));

        let all = EvolutionLedger::load_all_weights(db_str);
        assert_eq!(all.get(&("BTCUSDT".to_string(), "scalp".to_string())), Some(&1.5));

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_evolution_ledger_dual_horizon_and_nan_immunity() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!("evolution_dual_{}.db", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let db_str = db_path.to_str().unwrap();

        let ledger = EvolutionLedger::new(db_str);
        // Save distinct weights for Scalp and Swing on SOLUSDT
        ledger.save_weight(2, "SOLUSDT".into(), "scalp".into(), 2.8);
        ledger.save_weight(2, "SOLUSDT".into(), "swing".into(), 4.2);
        // Save NaN weight for BNBUSDT (should sanitize to 0.0)
        ledger.save_weight(3, "BNBUSDT".into(), "scalp".into(), f64::NAN);

        let mut scalp_ok = false;
        let mut swing_ok = false;
        let mut nan_ok = false;

        for _ in 0..30 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            if let Some(w) = EvolutionLedger::load_weight(db_str, "SOLUSDT", "scalp") {
                if (w - 2.8).abs() < 1e-5 { scalp_ok = true; }
            }
            if let Some(w) = EvolutionLedger::load_weight(db_str, "SOLUSDT", "swing") {
                if (w - 4.2).abs() < 1e-5 { swing_ok = true; }
            }
            if let Some(w) = EvolutionLedger::load_weight(db_str, "BNBUSDT", "scalp") {
                if (w - 0.0).abs() < 1e-5 { nan_ok = true; }
            }
            if scalp_ok && swing_ok && nan_ok { break; }
        }

        assert!(scalp_ok, "SOL scalp weight must be 2.8");
        assert!(swing_ok, "SOL swing weight must be 4.2");
        assert!(nan_ok, "BNB NaN weight must be sanitized to 0.0");

        let _ = std::fs::remove_file(db_path);
    }
}

