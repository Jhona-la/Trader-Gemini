use crossbeam_channel::{Receiver, Sender};
use rusqlite::{params, Connection};
use std::thread;

/// Evento atómico para guardar pesos/mutaciones del Genoma Evolutivo (Lock-Free)
///
/// # U-ERR-8 (LA COLUMNA NO GUARDABA NINGUNA ESTRATEGIA)
///
/// El campo se llamaba `strategy` y estaba documentado como «"scalp" o
/// "swing"». Ese comentario era FALSO respecto del único escritor vivo: el
/// demonio evolutivo guarda aquí `"promote"` y `"rollback"`, es decir, el TIPO
/// DE SUCESO del linaje genético — no una banda de horizonte ni una
/// estrategia. La etiqueta de banda sólo vivía en el comentario y en los
/// tests, que la sostenían con datos inventados y por tanto verificaban una
/// semántica que el sistema no tenía.
///
/// El campo se llama ahora `kind`, igual que la columna, que es lo que de
/// verdad contiene. Aquí NO procede sustituirlo por un horizonte τ: no
/// describe una operación de mercado sino un acontecimiento del linaje.
#[derive(Debug, Clone)]
pub struct GenomeUpdateEvent {
    pub coin_id: usize,
    pub symbol: String,
    /// Tipo de suceso registrado (p. ej. "promote", "rollback").
    pub kind: String,
    pub weight: f64,
}

pub struct EvolutionLedger {
    tx: Sender<GenomeUpdateEvent>,
}

impl EvolutionLedger {
    pub fn new(db_path: &str) -> Self {
        // FIX #1434: Canal acotado (100k eventos) para evitar sobrecarga de memoria en host de 16GB
        let (tx, rx): (Sender<GenomeUpdateEvent>, Receiver<GenomeUpdateEvent>) =
            crossbeam_channel::bounded(100_000);
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
                     kind TEXT NOT NULL,
                     weight REAL NOT NULL,
                     updated_at INTEGER NOT NULL,
                     PRIMARY KEY(symbol, kind)
                 );",
                );

                // U-ERR-8 — COMPATIBILIDAD DE LECTURA. Una base ya escrita
                // conserva su columna `strategy` y el CREATE IF NOT EXISTS no
                // la toca: sin esta conversión toda escritura nueva apuntaría
                // a una columna inexistente y se perdería en silencio. El
                // RENAME COLUMN de SQLite conserva íntegras las filas y los
                // índices; sólo cambia el nombre al que describe su contenido.
                if conn
                    .prepare("SELECT strategy FROM genome_weights LIMIT 1")
                    .is_ok()
                {
                    if let Err(e) = conn.execute(
                        "ALTER TABLE genome_weights RENAME COLUMN strategy TO kind",
                        [],
                    ) {
                        telemetry_engine::telemetry_err!(
                            "⚠️ [EVOLUTION-LEDGER] Conversión del esquema legado fallida: {}",
                            e
                        );
                    }
                }

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
                                "INSERT INTO genome_weights (coin_id, symbol, kind, weight, updated_at)
                             VALUES (?1, ?2, ?3, ?4, ?5)
                             ON CONFLICT(symbol, kind) DO UPDATE SET
                                coin_id=excluded.coin_id,
                                weight=excluded.weight,
                                updated_at=excluded.updated_at",
                                params![event.coin_id as i64, event.symbol, event.kind, safe_weight, (ts.min(i64::MAX as u64)) as i64],
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

    /// Registra un suceso del linaje genético. `kind` es el TIPO de suceso
    /// (p. ej. "promote", "rollback"), no una banda de horizonte.
    #[inline]
    pub fn save_weight(&self, coin_id: usize, symbol: String, kind: String, weight: f64) {
        // FIX #725: Sanitizar peso genético antes de enviar al canal WAL
        let safe_weight = if weight.is_finite() { weight } else { 0.0 };
        let _ = self.tx.try_send(GenomeUpdateEvent {
            coin_id,
            symbol,
            kind,
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
        let mut stmt = match conn.prepare("SELECT symbol, kind, weight FROM genome_weights") {
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

    /// FIX #1415: Carga el peso registrado para un símbolo y un tipo de suceso.
    pub fn load_weight(db_path: &str, symbol: &str, kind: &str) -> Option<f64> {
        let conn = Connection::open(db_path).ok()?;
        let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
        let mut stmt = conn
            .prepare("SELECT weight FROM genome_weights WHERE symbol=?1 AND kind=?2")
            .ok()?;
        let weight: f64 = stmt
            .query_row(params![symbol, kind], |row| row.get(0))
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
        let db_path = temp_dir.join(format!(
            "evolution_test_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db_str = db_path.to_str().unwrap();

        let ledger = EvolutionLedger::new(db_str);
        // U-ERR-8: los tipos de suceso que el demonio evolutivo escribe de
        // verdad, no las etiquetas de banda que el comentario inventaba.
        ledger.save_weight(0, "gen_7".into(), "promote".into(), 1.5);

        let mut loaded = None;
        for _ in 0..30 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            if let Some(w) = EvolutionLedger::load_weight(db_str, "gen_7", "promote") {
                if (w - 1.5).abs() < 1e-6 {
                    loaded = Some(w);
                    break;
                }
            }
        }

        assert_eq!(loaded, Some(1.5));

        let all = EvolutionLedger::load_all_weights(db_str);
        assert_eq!(
            all.get(&("gen_7".to_string(), "promote".to_string())),
            Some(&1.5)
        );

        let _ = std::fs::remove_file(db_path);
    }

    /// U-ERR-8 — EL LEDGER REGISTRA TIPOS DE SUCESO, NO BANDAS.
    ///
    /// Sustituye a `test_evolution_ledger_dual_horizon_and_nan_immunity`, que
    /// guardaba pesos bajo "scalp" y "swing" y comprobaba que convivían. Esa
    /// prueba verdeaba sobre datos que NINGÚN escritor del sistema produce: el
    /// demonio evolutivo escribe "promote" y "rollback". El test verificaba
    /// una semántica inventada por su propio comentario.
    ///
    /// Aquí se comprueba lo que el sistema hace de verdad: dos sucesos
    /// distintos del mismo linaje conviven, y un peso no finito se sanea.
    #[test]
    fn u_err_8_sucesos_del_linaje_conviven_y_el_nan_se_sanea() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!(
            "evolution_sucesos_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db_str = db_path.to_str().unwrap();

        let ledger = EvolutionLedger::new(db_str);
        ledger.save_weight(2, "gen_11".into(), "promote".into(), 2.8);
        ledger.save_weight(2, "gen_11".into(), "rollback".into(), 4.2);
        // Peso no finito: debe sanearse a 0.0
        ledger.save_weight(3, "gen_12".into(), "promote".into(), f64::NAN);

        let mut promote_ok = false;
        let mut rollback_ok = false;
        let mut nan_ok = false;

        for _ in 0..30 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            if let Some(w) = EvolutionLedger::load_weight(db_str, "gen_11", "promote") {
                if (w - 2.8).abs() < 1e-5 {
                    promote_ok = true;
                }
            }
            if let Some(w) = EvolutionLedger::load_weight(db_str, "gen_11", "rollback") {
                if (w - 4.2).abs() < 1e-5 {
                    rollback_ok = true;
                }
            }
            if let Some(w) = EvolutionLedger::load_weight(db_str, "gen_12", "promote") {
                if w.abs() < 1e-5 {
                    nan_ok = true;
                }
            }
            if promote_ok && rollback_ok && nan_ok {
                break;
            }
        }

        assert!(promote_ok, "el peso de la promoción debe ser 2.8");
        assert!(rollback_ok, "el peso del rollback debe ser 4.2");
        assert!(nan_ok, "un peso no finito debe saneare a 0.0");

        let _ = std::fs::remove_file(db_path);
    }

    /// U-ERR-8 — COMPATIBILIDAD DE LECTURA CON BASES YA ESCRITAS.
    ///
    /// Falla con el código viejo por construcción: allí no había conversión
    /// alguna. Comprueba que una base con la columna antigua `strategy` se
    /// convierte conservando sus filas, y que las escrituras nuevas aterrizan.
    #[test]
    fn u_err_8_base_con_columna_antigua_se_convierte_sin_perder_filas() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!(
            "evolution_legacy_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let db_str = db_path.to_str().unwrap();

        {
            let conn = Connection::open(db_str).expect("crear base legada");
            conn.execute_batch(
                "CREATE TABLE genome_weights (
                     coin_id INTEGER NOT NULL DEFAULT 0,
                     symbol TEXT NOT NULL,
                     strategy TEXT NOT NULL,
                     weight REAL NOT NULL,
                     updated_at INTEGER NOT NULL,
                     PRIMARY KEY(symbol, strategy)
                 );
                 INSERT INTO genome_weights VALUES (0,'gen_3','promote',3.3,10);",
            )
            .expect("poblar base legada");
        }

        let ledger = EvolutionLedger::new(db_str);
        ledger.save_weight(0, "gen_4".into(), "rollback".into(), -1.0);

        let mut historico = None;
        let mut nuevo = None;
        for _ in 0..40 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            historico = EvolutionLedger::load_weight(db_str, "gen_3", "promote");
            nuevo = EvolutionLedger::load_weight(db_str, "gen_4", "rollback");
            if historico.is_some() && nuevo.is_some() {
                break;
            }
        }

        assert_eq!(
            historico,
            Some(3.3),
            "la fila histórica debe sobrevivir a la conversión de columna"
        );
        assert_eq!(nuevo, Some(-1.0), "la escritura nueva debe aterrizar");

        let _ = std::fs::remove_file(db_path);
    }
}
