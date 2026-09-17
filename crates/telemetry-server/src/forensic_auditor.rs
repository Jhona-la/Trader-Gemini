use crate::TelemetryEvent;
use rusqlite::{Connection, params};
use tokio::sync::broadcast::Receiver;

pub struct ForensicAuditor {
    conn: Connection,
}

impl ForensicAuditor {
    pub fn new(db_path: &str) -> Self {
        // Aseguramos que el directorio exista
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let conn = Connection::open(db_path).expect("Error al abrir DB forense");

        // FIX #1445: Habilitar modo WAL y optimizar pragmas de SQLite para concurrencia masiva
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA temp_store=MEMORY;
             PRAGMA mmap_size=67108864;
             PRAGMA cache_size=-32768;
             PRAGMA busy_timeout=5000;",
        )
        .expect("Fallo al configurar PRAGMA de SQLite");

        // Tabla de métricas globales (OmniUpdate)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS global_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                latency_ms INTEGER,
                latency_panic BOOLEAN,
                dark_alpha REAL,
                scalp_pnl REAL,
                swing_pnl REAL,
                gross_pnl REAL,
                net_pnl REAL,
                win_rate REAL,
                trade_duration_avg REAL
            )",
            [],
        )
        .expect("Fallo al crear tabla global_metrics");

        // Tabla de genomas (Guardar snapshots genéticos)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS genome_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                genome_json TEXT
            )",
            [],
        )
        .expect("Fallo al crear tabla genome_snapshots");

        // FASE 18: Tabla de operaciones cerradas
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                coin_id INTEGER,
                trade_type TEXT,
                pnl REAL,
                roi_pct REAL,
                duration_ms INTEGER,
                ml_prob REAL
            )",
            [],
        )
        .expect("Fallo al crear tabla trade_events");

        Self { conn }
    }

    pub async fn start(self, mut rx: Receiver<TelemetryEvent>) {
        println!("🔍 [FORENSIC] Auditoría en base de datos WAL iniciada en hilo independiente.");

        // Bucle asíncrono pasivo, no bloquea al motor HFT
        // A-2 infra: Lagged es RECUPERABLE (consumer lento) — continuar, no morir.
        loop {
            let event = match rx.recv().await {
                Ok(e) => e,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    eprintln!(
                        "[FORENSIC-AUDITOR] Lagged: {} eventos perdidos — continuando",
                        n
                    );
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            };
            match event {
                TelemetryEvent::OmniUpdate {
                    latency_ms,
                    latency_panic,
                    dark_alpha,
                    scalp_pnl,
                    swing_pnl,
                    gross_pnl,
                    net_pnl,
                    win_rate,
                    trade_duration_avg,
                } => {
                    let safe_dark = if dark_alpha.is_finite() {
                        dark_alpha
                    } else {
                        0.0
                    };
                    let safe_scalp = if scalp_pnl.is_finite() {
                        scalp_pnl
                    } else {
                        0.0
                    };
                    let safe_swing = if swing_pnl.is_finite() {
                        swing_pnl
                    } else {
                        0.0
                    };
                    let safe_gross = if gross_pnl.is_finite() {
                        gross_pnl
                    } else {
                        0.0
                    };
                    let safe_net = if net_pnl.is_finite() { net_pnl } else { 0.0 };
                    let safe_wr = if win_rate.is_finite() { win_rate } else { 0.0 };
                    let safe_avg = if trade_duration_avg.is_finite() {
                        trade_duration_avg
                    } else {
                        0.0
                    };

                    let _ = self.conn.execute(
                        "INSERT INTO global_metrics (
                            latency_ms, latency_panic, dark_alpha, scalp_pnl, swing_pnl, 
                            gross_pnl, net_pnl, win_rate, trade_duration_avg
                        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                        params![
                            latency_ms as i64,
                            latency_panic,
                            safe_dark,
                            safe_scalp,
                            safe_swing,
                            safe_gross,
                            safe_net,
                            safe_wr,
                            safe_avg
                        ],
                    );
                }
                TelemetryEvent::GenomeUpdate(genome) => {
                    if let Ok(json) = serde_json::to_string(&*genome) {
                        let _ = self.conn.execute(
                            "INSERT INTO genome_snapshots (genome_json) VALUES (?1)",
                            params![json],
                        );
                    }
                }
                TelemetryEvent::TradeClosed {
                    coin_id,
                    trade_type,
                    pnl,
                    roi_pct,
                    duration_ms,
                    ml_prob,
                } => {
                    let safe_pnl = if pnl.is_finite() { pnl } else { 0.0 };
                    let safe_roi = if roi_pct.is_finite() { roi_pct } else { 0.0 };
                    let safe_ml_prob = if ml_prob.is_finite() { ml_prob } else { 0.0 };

                    let _ = self.conn.execute(
                        "INSERT INTO trade_events (coin_id, trade_type, pnl, roi_pct, duration_ms, ml_prob) 
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![
                            coin_id as i64,
                            trade_type,
                            safe_pnl,
                            safe_roi,
                            duration_ms as i64,
                            safe_ml_prob
                        ],
                    );
                }
                _ => {} // Otros eventos se ignoran en DB
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forensic_auditor_db_creation() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_forensic_auditor.db");
        let db_path_str = db_path.to_string_lossy().to_string();

        let auditor = ForensicAuditor::new(&db_path_str);
        let _ = auditor;

        assert!(db_path.exists());
        let _ = std::fs::remove_file(db_path);
    }

    #[tokio::test]
    async fn test_forensic_auditor_event_processing_and_nan_immunity() {
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let db_path = temp_dir.join(format!("test_forensic_{}.db", unique_id));
        let db_path_str = db_path.to_string_lossy().to_string();

        let auditor = ForensicAuditor::new(&db_path_str);
        let (tx, rx) = tokio::sync::broadcast::channel(16);

        tokio::spawn(async move {
            auditor.start(rx).await;
        });

        // Send OmniUpdate with NaN
        let _ = tx.send(TelemetryEvent::OmniUpdate {
            latency_ms: 15,
            latency_panic: false,
            dark_alpha: f64::NAN,
            scalp_pnl: 0.5,
            swing_pnl: 1.2,
            gross_pnl: 1.7,
            net_pnl: 1.65,
            win_rate: 0.85,
            trade_duration_avg: 120.0,
        });

        // Send TradeClosed
        let _ = tx.send(TelemetryEvent::TradeClosed {
            coin_id: 1,
            trade_type: "SCALP".to_string(),
            pnl: 0.25,
            roi_pct: 1.92,
            duration_ms: 5000,
            ml_prob: 0.88,
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let conn = Connection::open(&db_path_str).expect("open audit db");
        let count_metrics: i64 = conn
            .query_row("SELECT count(*) FROM global_metrics", [], |r| r.get(0))
            .unwrap();
        let count_trades: i64 = conn
            .query_row("SELECT count(*) FROM trade_events", [], |r| r.get(0))
            .unwrap();

        assert_eq!(count_metrics, 1);
        assert_eq!(count_trades, 1);

        let _ = std::fs::remove_file(db_path);
    }
}
