use tokio::sync::broadcast::Receiver;
use rusqlite::{Connection, params};
use crate::TelemetryEvent;

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
        
        // Habilitar modo WAL para concurrencia masiva (lector/escritor paralelo)
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA temp_store=MEMORY;"
        ).expect("Fallo al configurar PRAGMA de SQLite");

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
        ).expect("Fallo al crear tabla global_metrics");

        // Tabla de genomas (Guardar snapshots genéticos)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS genome_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                genome_json TEXT
            )",
            [],
        ).expect("Fallo al crear tabla genome_snapshots");

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
        ).expect("Fallo al crear tabla trade_events");

        Self { conn }
    }

    pub async fn start(self, mut rx: Receiver<TelemetryEvent>) {
        println!("🔍 [FORENSIC] Auditoría en base de datos WAL iniciada en hilo independiente.");
        
        // Bucle asíncrono pasivo, no bloquea al motor HFT
        while let Ok(event) = rx.recv().await {
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
                    let _ = self.conn.execute(
                        "INSERT INTO global_metrics (
                            latency_ms, latency_panic, dark_alpha, scalp_pnl, swing_pnl, 
                            gross_pnl, net_pnl, win_rate, trade_duration_avg
                        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                        params![
                            latency_ms as i64,
                            latency_panic,
                            dark_alpha,
                            scalp_pnl,
                            swing_pnl,
                            gross_pnl,
                            net_pnl,
                            win_rate,
                            trade_duration_avg
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
                    let _ = self.conn.execute(
                        "INSERT INTO trade_events (coin_id, trade_type, pnl, roi_pct, duration_ms, ml_prob) 
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![
                            coin_id as i64,
                            trade_type,
                            pnl,
                            roi_pct,
                            duration_ms as i64,
                            ml_prob
                        ],
                    );
                }
                _ => {} // Otros eventos se ignoran en DB
            }
        }
    }
}
