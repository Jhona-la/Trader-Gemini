use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub gen_id: u32,
    pub timestamp_ns: u64,
    pub teleonomia_score: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub active_regime: String,
    pub code_hash: u64,
}

#[derive(Debug, Clone)]
pub struct AdnBackupCatalog {
    pub base_dir: PathBuf,
}

impl AdnBackupCatalog {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Self {
        let dir = base_dir.as_ref().to_path_buf();
        let _ = fs::create_dir_all(&dir);
        Self { base_dir: dir }
    }

    /// Guarda una generación completa en la memoria genética inmutable
    pub fn archive_generation(
        &self,
        meta: &GenerationMetadata,
        code_content: &str,
    ) -> std::io::Result<()> {
        let gen_folder = self
            .base_dir
            .join(format!("generaciones/gen_{:04}", meta.gen_id));
        fs::create_dir_all(&gen_folder)?;

        let meta_json = serde_json::to_string_pretty(meta).unwrap_or_default();
        fs::write(gen_folder.join("metadata.json"), meta_json)?;
        fs::write(gen_folder.join("cortex_source.rs"), code_content)?;

        Ok(())
    }

    /// Carga el historial de generaciones archivadas
    pub fn list_archived_generations(&self) -> Vec<GenerationMetadata> {
        let mut list = Vec::new();
        let gen_root = self.base_dir.join("generaciones");
        if let Ok(entries) = fs::read_dir(gen_root) {
            for entry in entries.flatten() {
                let meta_path = entry.path().join("metadata.json");
                if meta_path.exists() {
                    if let Ok(content) = fs::read_to_string(meta_path) {
                        if let Ok(meta) = serde_json::from_str::<GenerationMetadata>(&content) {
                            if meta.teleonomia_score.is_finite()
                                && meta.sharpe_ratio.is_finite()
                                && meta.max_drawdown.is_finite()
                                && meta.win_rate.is_finite()
                            {
                                list.push(meta);
                            }
                        }
                    }
                }
            }
        }
        list.sort_by_key(|m| m.gen_id);
        list
    }
}

/// Módulo de Reminiscencia: revisita arquitecturas antiguas y las usa como semilla
#[derive(Debug, Clone)]
pub struct ReminiscenceModule {
    pub catalog: AdnBackupCatalog,
}

impl ReminiscenceModule {
    pub fn new(catalog: AdnBackupCatalog) -> Self {
        Self { catalog }
    }

    /// Evalúa si existe alguna generación previa archivada que coincida con el régimen actual
    /// y presente superioridad de Sharpe/Teleonomía para resucitar como semilla mutacional.
    pub fn find_ancestral_seed(
        &self,
        current_regime: &str,
        min_sharpe: f64,
    ) -> Option<GenerationMetadata> {
        let archived = self.catalog.list_archived_generations();
        archived
            .into_iter()
            .filter(|g| g.active_regime == current_regime && g.sharpe_ratio >= min_sharpe)
            .max_by(|a, b| {
                a.teleonomia_score
                    .partial_cmp(&b.teleonomia_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adn_backup_catalog_archive_and_list() {
        let temp_dir = std::env::temp_dir().join(format!(
            "adn_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let catalog = AdnBackupCatalog::new(&temp_dir);

        let meta = GenerationMetadata {
            gen_id: 1,
            timestamp_ns: 1000000,
            teleonomia_score: 0.95,
            sharpe_ratio: 2.5,
            max_drawdown: 0.02,
            win_rate: 0.70,
            active_regime: "TrendingBull".to_string(),
            code_hash: 123456789,
        };

        catalog.archive_generation(&meta, "// source code").unwrap();
        let list = catalog.list_archived_generations();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].gen_id, 1);
        assert_eq!(list[0].teleonomia_score, 0.95);

        let reminiscence = ReminiscenceModule::new(catalog);
        let seed = reminiscence.find_ancestral_seed("TrendingBull", 2.0);
        assert!(seed.is_some());
        assert_eq!(seed.unwrap().gen_id, 1);

        let no_seed = reminiscence.find_ancestral_seed("HighVolatilityBear", 2.0);
        assert!(no_seed.is_none());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
