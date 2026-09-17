//! # Hot-Swap Controller & Epigenoma Manager
//!
//! Manages dynamic library loading (`libloading`) and atomic state persistence in `memoria/epigenoma/`.

use memmap2::MmapMut;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigenomaSymbolParams {
    pub symbol: String,
    pub scalp_tp: f64,
    pub scalp_sl: f64,
    pub swing_tp: f64,
    pub swing_sl: f64,
    pub min_confidence: f64,
    pub max_leverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigenomaUtilityConfig {
    pub utility_function_name: String,
    pub sharpe_weight: f64,
    pub wr_weight: f64,
    pub max_drawdown_penalty: f64,
    pub target_3day_return: f64,
}

pub struct HotSwapController {
    epigenoma_dir: PathBuf,
    loaded_libraries: Vec<Arc<libloading::Library>>,
    temp_files: Vec<PathBuf>,
}

impl HotSwapController {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Self {
        let root = base_dir.as_ref();
        let epigenoma_dir = root.join("memoria").join("epigenoma");
        let _ = fs::create_dir_all(epigenoma_dir.join("activos"));
        let _ = fs::create_dir_all(epigenoma_dir.join("modelos"));
        let _ = fs::create_dir_all(epigenoma_dir.join("teleonomia"));

        Self {
            epigenoma_dir,
            loaded_libraries: Vec::new(),
            temp_files: Vec::new(),
        }
    }

    /// Saves symbol state to TOML inside `memoria/epigenoma/activos/`
    pub fn save_symbol_epigenoma(
        &self,
        params: &EpigenomaSymbolParams,
    ) -> std::io::Result<PathBuf> {
        let target = self
            .epigenoma_dir
            .join("activos")
            .join(format!("{}.toml", params.symbol));
        let scalp_tp = if params.scalp_tp.is_finite() && params.scalp_tp > 0.0 {
            params.scalp_tp
        } else {
            0.005
        };
        let scalp_sl = if params.scalp_sl.is_finite() && params.scalp_sl > 0.0 {
            params.scalp_sl
        } else {
            0.002
        };
        let swing_tp = if params.swing_tp.is_finite() && params.swing_tp > 0.0 {
            params.swing_tp
        } else {
            0.020
        };
        let swing_sl = if params.swing_sl.is_finite() && params.swing_sl > 0.0 {
            params.swing_sl
        } else {
            0.010
        };
        let min_confidence = if params.min_confidence.is_finite() {
            params.min_confidence.clamp(0.0, 1.0)
        } else {
            0.60
        };
        let max_leverage = if params.max_leverage.is_finite() && params.max_leverage >= 1.0 {
            params.max_leverage.clamp(1.0, 100.0)
        } else {
            10.0
        };

        let content = format!(
            r#"# Epigenoma State for {}
symbol = "{}"
scalp_tp = {:.6}
scalp_sl = {:.6}
swing_tp = {:.6}
swing_sl = {:.6}
min_confidence = {:.6}
max_leverage = {:.6}
"#,
            params.symbol,
            params.symbol,
            scalp_tp,
            scalp_sl,
            swing_tp,
            swing_sl,
            min_confidence,
            max_leverage,
        );
        let temp_target = target.with_extension("tmp");
        fs::write(&temp_target, content)?;
        fs::rename(&temp_target, &target)?;
        Ok(target)
    }

    /// Fast mmap state snapshot writer for high frequency state updates (< 1ms latency)
    pub fn write_mmap_state(&self, file_name: &str, data: &[u8]) -> std::io::Result<PathBuf> {
        let target = self.epigenoma_dir.join("modelos").join(file_name);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&target)?;

        file.set_len(data.len() as u64)?;
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        mmap.copy_from_slice(data);
        mmap.flush()?;

        Ok(target)
    }

    /// # Safety
    /// Dynamically loading a shared library is unsafe. The caller must ensure the library conforms to expected ABIs.
    pub unsafe fn load_dynamic_library<P: AsRef<Path>>(
        &mut self,
        library_path: P,
    ) -> Result<Arc<libloading::Library>, libloading::Error> {
        let path = library_path.as_ref();

        // Copia temporal versionada para evitar bloqueos de archivo (ERROR_SHARING_VIOLATION) en Windows (BUG-670)
        let load_path = if cfg!(windows) {
            // FIX #1103: Purgar archivos temporales huérfanos de sesiones previas en temp_dir
            if let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) {
                for entry in entries.flatten() {
                    if let Some(name) = entry.file_name().to_str() {
                        if name.starts_with("cortex_")
                            && (name.ends_with(".dll")
                                || name.ends_with(".so")
                                || name.ends_with(".dylib"))
                        {
                            let _ = std::fs::remove_file(entry.path());
                        }
                    }
                }
            }

            let file_stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("cortex");
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("dll");
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let temp_name = format!("{}_{}.{}", file_stem, nonce, ext);
            let temp_dir = std::env::temp_dir();
            let temp_path = temp_dir.join(temp_name);
            if std::fs::copy(path, &temp_path).is_ok() {
                self.temp_files.push(temp_path.clone());
                temp_path
            } else {
                path.to_path_buf()
            }
        } else {
            path.to_path_buf()
        };

        let lib = libloading::Library::new(&load_path)?;
        let arc_lib = Arc::new(lib);
        self.loaded_libraries.push(arc_lib.clone());
        Ok(arc_lib)
    }

    /// # Safety
    /// Returning a symbol relies on the library providing a compatible function signature.
    pub unsafe fn get_evaluate_tensor(
        &self,
    ) -> Option<libloading::Symbol<'_, unsafe extern "C" fn(*const f32, usize) -> f32>> {
        if let Some(lib) = self.loaded_libraries.last() {
            lib.get(b"evaluate_tensor\0").ok()
        } else {
            None
        }
    }

    /// # Safety
    /// Provee acceso directo a símbolos en doble precisión (f64) nativa para tensores financieros (54D / 34D).
    pub unsafe fn get_evaluate_tensor_f64(
        &self,
    ) -> Option<libloading::Symbol<'_, unsafe extern "C" fn(*const f64, usize) -> f64>> {
        if let Some(lib) = self.loaded_libraries.last() {
            lib.get(b"evaluate_tensor_f64\0").ok()
        } else {
            None
        }
    }
}

impl Drop for HotSwapController {
    fn drop(&mut self) {
        // Drop loaded libraries first to release Windows file handles
        self.loaded_libraries.clear();
        for path in &self.temp_files {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_swap_controller_save_symbol_and_mmap() {
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(4567);
        let base_path = temp_dir.join(format!("test_hotswap_{}", unique_id));

        let controller = HotSwapController::new(&base_path);

        let params = EpigenomaSymbolParams {
            symbol: "BTCUSDT".to_string(),
            scalp_tp: 0.005,
            scalp_sl: 0.003,
            swing_tp: 0.02,
            swing_sl: 0.01,
            min_confidence: 0.75,
            max_leverage: 10.0,
        };

        let saved_path = controller
            .save_symbol_epigenoma(&params)
            .expect("Failed to save epigenoma");
        assert!(saved_path.exists());
        let content = std::fs::read_to_string(&saved_path).expect("Failed to read saved file");
        assert!(content.contains("BTCUSDT"));
        assert!(content.contains("0.005000"));

        let mmap_data = b"TENSOR_WEIGHTS_V5_MOCK";
        let mmap_path = controller
            .write_mmap_state("model_weights.bin", mmap_data)
            .expect("Failed to write mmap state");
        assert!(mmap_path.exists());
        let read_back = std::fs::read(&mmap_path).expect("Failed to read mmap file");
        assert_eq!(read_back, mmap_data);

        let _ = std::fs::remove_dir_all(base_path);
    }
}
