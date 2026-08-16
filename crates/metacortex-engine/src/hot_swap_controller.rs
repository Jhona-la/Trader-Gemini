//! # Hot-Swap Controller & Epigenoma Manager
//!
//! Manages dynamic library loading (`libloading`) and atomic state persistence in `memoria/epigenoma/`.

use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use memmap2::MmapMut;
use serde::{Deserialize, Serialize};

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
        }
    }

    /// Saves symbol state to TOML inside `memoria/epigenoma/activos/`
    pub fn save_symbol_epigenoma(&self, params: &EpigenomaSymbolParams) -> std::io::Result<PathBuf> {
        let target = self.epigenoma_dir.join("activos").join(format!("{}.toml", params.symbol));
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
            params.scalp_tp,
            params.scalp_sl,
            params.swing_tp,
            params.swing_sl,
            params.min_confidence,
            params.max_leverage,
        );
        fs::write(&target, content)?;
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
    pub unsafe fn load_dynamic_library<P: AsRef<Path>>(&mut self, library_path: P) -> Result<Arc<libloading::Library>, libloading::Error> {
        let lib = libloading::Library::new(library_path.as_ref())?;
        let arc_lib = Arc::new(lib);
        self.loaded_libraries.push(arc_lib.clone());
        Ok(arc_lib)
    }

    /// # Safety
    /// Returning a symbol relies on the library providing a compatible function signature.
    pub unsafe fn get_evaluate_tensor(&self) -> Option<libloading::Symbol<'_, unsafe extern "C" fn(*const f32, usize) -> f32>> {
        if let Some(lib) = self.loaded_libraries.last() {
            lib.get(b"evaluate_tensor\0").ok()
        } else {
            None
        }
    }
}
