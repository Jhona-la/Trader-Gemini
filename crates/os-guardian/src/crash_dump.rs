/// 🛡️ VOLCADO ATÓMICO DE EMERGENCIA Y RECUPERACIÓN DE ESTADO (CRASH DUMP RECOVERY)
/// Permite capturar y persistir el snapshot del estado del bot en JSON ante excepciones críticas (#226-#235).
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionDumpEntry {
    pub coin_id: usize,
    pub symbol: String,
    pub horizon: String,
    pub side: String,
    pub entry_price: f64,
    pub size_nominal_usd: f64,
    pub leverage: f64,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyCrashDump {
    pub process_uuid: String,
    pub timestamp_ms: u64,
    pub reason: String,
    pub unified_capital: f64,
    pub open_positions: Vec<PositionDumpEntry>,
}

impl EmergencyCrashDump {
    pub fn new(process_uuid: &str, reason: &str, capital: f64) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            process_uuid: process_uuid.to_string(),
            timestamp_ms: now_ms,
            reason: reason.to_string(),
            unified_capital: capital,
            open_positions: Vec::new(),
        }
    }

    pub fn add_position(&mut self, entry: PositionDumpEntry) {
        self.open_positions.push(entry);
    }

    /// Vuelca atómicamente el estado a un archivo JSON en disco
    pub fn dump_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let path = path.as_ref();
        let tmp_path = path.with_extension("tmp");
        let json_data = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let mut file = File::create(&tmp_path)?;
        file.write_all(json_data.as_bytes())?;
        file.sync_all()?;
        if path.exists() {
            let _ = std::fs::remove_file(path);
        }
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Lee y restaura un crash dump previo
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let dump: Self = serde_json::from_reader(file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(dump)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emergency_crash_dump_serialization() {
        let mut dump = EmergencyCrashDump::new("uuid-1234", "SIGTERM", 13.50);
        dump.add_position(PositionDumpEntry {
            coin_id: 0,
            symbol: "BTCUSDT".to_string(),
            horizon: "Scalp".to_string(),
            side: "Long".to_string(),
            entry_price: 65000.0,
            size_nominal_usd: 5.05,
            leverage: 10.0,
            timestamp_ms: 1700000000,
        });

        let json = serde_json::to_string(&dump).unwrap();
        assert!(json.contains("BTCUSDT"));
        assert!(json.contains("uuid-1234"));
        assert!(json.contains("13.5"));

        let restored: EmergencyCrashDump = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.open_positions.len(), 1);
        assert_eq!(restored.open_positions[0].symbol, "BTCUSDT");
    }

    #[test]
    fn test_emergency_crash_dump_atomic_file_write() {
        let temp_dir = std::env::temp_dir();
        let dump_path = temp_dir.join(format!(
            "crash_dump_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut dump = EmergencyCrashDump::new("uuid-test-5678", "OOM_PREEMPTION", 26.0);
        dump.add_position(PositionDumpEntry {
            coin_id: 1,
            symbol: "ETHUSDT".to_string(),
            horizon: "Swing".to_string(),
            side: "Short".to_string(),
            entry_price: 3400.0,
            size_nominal_usd: 10.0,
            leverage: 5.0,
            timestamp_ms: 1700000050,
        });

        assert!(dump.dump_to_file(&dump_path).is_ok());
        assert!(dump_path.exists());

        let restored = EmergencyCrashDump::load_from_file(&dump_path).unwrap();
        assert_eq!(restored.process_uuid, "uuid-test-5678");
        assert_eq!(restored.open_positions.len(), 1);
        assert_eq!(restored.open_positions[0].symbol, "ETHUSDT");

        let _ = std::fs::remove_file(dump_path);
    }

    #[test]
    fn test_emergency_crash_dump_empty_positions_and_zero_balance() {
        let dump = EmergencyCrashDump::new("uuid-clean", "MANUAL_HALT", 0.0);
        let json = serde_json::to_string(&dump).unwrap();
        let restored: EmergencyCrashDump = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.open_positions.len(), 0);
        assert_eq!(restored.unified_capital, 0.0);
    }
}
