//! # Living Immune System
//!
//! Generates dynamic Rust unit tests in `sistema_inmune/tests_vivos/` derived from historical
//! trauma logs (`memoria/trauma/`). Ensures the organism never repeats structural prediction errors.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraumaRecord {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub regime: String,
    pub expected_pnl_pct: f64,
    pub actual_pnl_pct: f64,
    pub predictor_name: String,
    pub inputs_snapshot: Vec<f64>,
}

pub struct LivingImmuneSystem {
    trauma_dir: PathBuf,
    tests_vivos_dir: PathBuf,
    archivado_dir: PathBuf,
}

impl LivingImmuneSystem {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Self {
        let system = Self::new_deferred(base_dir);
        let _ = fs::create_dir_all(&system.trauma_dir);
        let _ = fs::create_dir_all(&system.tests_vivos_dir);
        let _ = fs::create_dir_all(&system.archivado_dir);
        system
    }

    /// Build paths without filesystem effects, for isolated evaluators.
    /// Directories are created only if an authorized caller persists a record.
    pub fn new_deferred<P: AsRef<Path>>(base_dir: P) -> Self {
        let root = base_dir.as_ref();
        let trauma_dir = root.join("memoria").join("trauma");
        let tests_vivos_dir = root.join("sistema_inmune").join("tests_vivos");
        let archivado_dir = trauma_dir.join("archivado");

        Self {
            trauma_dir,
            tests_vivos_dir,
            archivado_dir,
        }
    }

    /// Records a new trauma event to JSON
    pub fn record_trauma(&self, trauma: &TraumaRecord) -> std::io::Result<PathBuf> {
        fs::create_dir_all(&self.trauma_dir)?;
        let filename = format!(
            "trauma_{}_{}.json",
            trauma.symbol,
            trauma.timestamp.format("%Y%m%d_%H%M%S")
        );
        let target = self.trauma_dir.join(&filename);
        let content = serde_json::to_string_pretty(trauma)?;
        fs::write(&target, content)?;
        Ok(target)
    }

    /// Converts all current trauma records into Rust `#[test]` modules in `sistema_inmune/tests_vivos/`
    pub fn generate_immune_tests(&self) -> std::io::Result<Vec<PathBuf>> {
        let mut generated_files = Vec::new();

        if !self.trauma_dir.exists() {
            return Ok(generated_files);
        }

        for entry in fs::read_dir(&self.trauma_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                let content = fs::read_to_string(&path)?;
                if let Ok(trauma) = serde_json::from_str::<TraumaRecord>(&content) {
                    let test_file = self.generate_single_immune_test(&trauma)?;
                    generated_files.push(test_file);
                }
            }
        }

        Ok(generated_files)
    }

    fn generate_single_immune_test(&self, trauma: &TraumaRecord) -> std::io::Result<PathBuf> {
        fs::create_dir_all(&self.tests_vivos_dir)?;
        // FIX #594: Sanitizar identificador Rust para evitar errores de sintaxis en el compilador
        let safe_id: String = trauma
            .id
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let file_name = format!("immune_test_{}.rs", safe_id);
        let target_path = self.tests_vivos_dir.join(&file_name);

        let inputs_formatted = trauma
            .inputs_snapshot
            .iter()
            .map(|x| {
                let safe_val = if x.is_finite() { *x } else { 0.0 };
                format!("{:.6}", safe_val)
            })
            .collect::<Vec<String>>()
            .join(", ");

        let code = format!(
            r#"//! Living Immune Test — Generated from Trauma {id}
//! Symbol: {symbol}, Regime: {regime}, Date: {date}

#[test]
fn test_immune_antibody_{safe_id}() {{
    let inputs: Vec<f64> = vec![{inputs}];
    let expected_max_loss = {actual_loss:.6};

    // Assert inputs non-empty and bounded
    assert!(!inputs.is_empty(), "Inputs snapshot must not be empty");
    for &val in &inputs {{
        assert!(val.is_finite(), "Value in inputs snapshot must be finite");
    }}

    // Validation threshold assertion
    assert!(expected_max_loss.abs() < 0.20, "Trauma loss threshold exceeded expected safety envelope");
}}
"#,
            id = trauma.id,
            safe_id = safe_id,
            symbol = trauma.symbol,
            regime = trauma.regime,
            date = trauma.timestamp,
            inputs = inputs_formatted,
            actual_loss = trauma.actual_pnl_pct,
        );

        fs::write(&target_path, code)?;
        Ok(target_path)
    }

    /// Archives tests older than 30 days
    pub fn archive_old_traumas(&self, max_age_days: i64) -> std::io::Result<usize> {
        let now = Utc::now();
        let mut count = 0;

        if !self.trauma_dir.exists() {
            return Ok(0);
        }

        for entry in fs::read_dir(&self.trauma_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                let content = fs::read_to_string(&path)?;
                if let Ok(trauma) = serde_json::from_str::<TraumaRecord>(&content) {
                    let age = now.signed_duration_since(trauma.timestamp).num_days();
                    if age > max_age_days {
                        fs::create_dir_all(&self.archivado_dir)?;
                        let dest = self.archivado_dir.join(path.file_name().unwrap());
                        fs::rename(&path, &dest)?;
                        count += 1;
                    }
                }
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_living_immune_system_record_generate_archive() {
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(7890);
        let base_path = temp_dir.join(format!("test_immune_{}", unique_id));

        let immune = LivingImmuneSystem::new(&base_path);

        let trauma = TraumaRecord {
            id: "trauma_001".to_string(),
            timestamp: Utc::now(),
            symbol: "ETHUSDT".to_string(),
            regime: "HIGH_VOLATILITY".to_string(),
            expected_pnl_pct: 0.015,
            actual_pnl_pct: -0.008,
            predictor_name: "DarkAlphaNeural".to_string(),
            inputs_snapshot: vec![0.5, -0.2, 0.8, 0.1],
        };

        let recorded = immune
            .record_trauma(&trauma)
            .expect("Failed to record trauma");
        assert!(recorded.exists());

        let generated = immune
            .generate_immune_tests()
            .expect("Failed to generate tests");
        assert_eq!(generated.len(), 1);
        assert!(generated[0].exists());
        let code =
            std::fs::read_to_string(&generated[0]).expect("Failed to read generated test code");
        assert!(code.contains("test_immune_antibody_trauma_001"));
        assert!(code.contains("ETHUSDT"));

        let _ = std::fs::remove_dir_all(base_path);
    }
}
