//! REGISTRO DE MODELOS (Ola XLVIII·G — prioridad #1 del marco del
//! operador: "model registry que permita reproducir cualquier señal").
//!
//! `models/` está en `.gitignore` (artefactos locales): un clon fresco
//! mide el piso sin modelos. Este módulo convierte esa laguna en un
//! contrato: un MANIFIESTO versionado (JSON, commit-able) con huella
//! SHA-256 por modelo, base del modelo (sigmoid(init_score), B3.36),
//! número de árboles y metadatos de archivo. Cualquier señal producida
//! en vivo puede rastrearse al hash del modelo que la generó.
//!
//! Contrato:
//! - **Variable**: cada `models/{KEY}.json` promovido (los `_CANDIDATE`
//!   y `.bin` cacheados NO se registran — el registro documenta lo que
//!   el roster carga, igual que `ml_coverage`).
//! - **Operador**: SHA-256 del archivo + campos extraídos del JSON
//!   (init_score → base; tree_offsets → nº árboles).
//! - **Unidades**: base ∈ [0,1] (sigmoid); hash hex 64 caracteres.
//! - **Contorno**: dir ausente ⇒ manifest vacío (no error); JSON
//!   ilegible ⇒ entrada con `legible: false` y deuda visible, no
//!   silencio.
//! - **Identificabilidad**: el hash identifica el archivo EXACTO; la
//!   base identifica la calibración que el gate consume (CL-21/FMT-159).
//! - **Coste**: O(bytes) al escanear; fuera del hot path.
//! - **Falsación**: mismo directorio ⇒ mismo manifest byte a byte
//!   (determinismo); tocar un modelo ⇒ cambia SU hash y sólo el suyo.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Un modelo promovido registrado.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Clave del roster ({SYM}_MOTOR).
    pub key: String,
    /// Archivo registrado (nombre, no ruta absoluta — reproducible).
    pub archivo: String,
    /// SHA-256 del archivo (hex). Identidad del artefacto.
    pub sha256: String,
    /// Bytes del archivo.
    pub bytes: u64,
    /// Base del modelo sigmoid(init_score) — la que el gate mide (B3.36).
    /// None si el JSON no se pudo leer (deuda visible).
    pub base: Option<f64>,
    /// Número de árboles (tree_offsets.len()−1); None si ilegible.
    pub n_arboles: Option<usize>,
    /// false ⇒ el archivo existe pero el JSON no se pudo parsear.
    pub legible: bool,
}

/// Manifiesto del inventario de modelos promovidos.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ModelManifest {
    /// Orden canónico (por clave) — el JSON es determinista.
    pub entries: Vec<ModelEntry>,
    /// Marca de generación (epoch ms) — informativa, NO parte de la
    /// identidad (los tests comparan entries).
    pub generado_ms: u64,
}

impl ModelManifest {
    pub fn escribir(&self, ruta: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(ruta, json)
    }

    pub fn cargar(ruta: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(ruta)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Escanea `models_dir` y produce el manifest de modelos promovidos
/// ({KEY}.json, sin `_CANDIDATE`). Determinista: entradas ordenadas por
/// clave; el campo `generado_ms` es la única parte no determinista y va
/// documentada como informativa.
pub fn escanear_models(models_dir: &Path, generado_ms: u64) -> ModelManifest {
    let mut entries: Vec<ModelEntry> = Vec::new();
    if let Ok(dir) = std::fs::read_dir(models_dir) {
        for entry in dir.filter_map(|e| e.ok()) {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str());
            if ext != Some("json") {
                continue; // .bin = caché derivada; el registro documenta fuentes
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if stem.ends_with("_CANDIDATE") {
                continue; // no promovido: el roster no lo carga
            }
            let Ok(bytes_vec) = std::fs::read(&path) else {
                continue;
            };
            let mut hasher = Sha256::new();
            hasher.update(&bytes_vec);
            let sha = format!("{:x}", hasher.finalize());
            // Campos del contrato NanoForest: init_score y tree_offsets.
            let (base, n_arboles, legible) =
                match serde_json::from_slice::<serde_json::Value>(&bytes_vec) {
                    Ok(v) => {
                        let init = v.get("init_score").and_then(|x| x.as_f64());
                        let base = init.map(|i| 1.0 / (1.0 + (-i).exp()));
                        let n = v
                            .get("tree_offsets")
                            .and_then(|x| x.as_array())
                            .map(|a| a.len().saturating_sub(1));
                        (base, n, true)
                    }
                    Err(_) => (None, None, false),
                };
            entries.push(ModelEntry {
                key: stem.to_string(),
                archivo: path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string(),
                sha256: sha,
                bytes: bytes_vec.len() as u64,
                base,
                n_arboles,
                legible,
            });
        }
    }
    entries.sort_by(|a, b| a.key.cmp(&b.key));
    ModelManifest { entries, generado_ms }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(dir: &Path, files: &[(&str, &str)]) {
        std::fs::create_dir_all(dir).unwrap();
        for (name, content) in files {
            std::fs::write(dir.join(name), content).unwrap();
        }
    }

    fn modelo_json(init: f64, arboles: usize) -> String {
        let offsets: Vec<usize> = (0..=arboles).collect();
        format!(
            "{{\"children_left\":[0],\"children_right\":[0],\"feature\":[0],\"threshold\":[0.0],\"value\":[0.5],\"tree_offsets\":{:?},\"init_score\":{}}}",
            offsets, init
        )
    }

    /// FALSACIÓN (a): mismo directorio ⇒ mismos entries byte a byte
    /// (determinismo del registro; sólo generado_ms es informativo).
    #[test]
    fn manifest_es_determinista() {
        let tmp = std::env::temp_dir().join(format!("tg_mr_{:x}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        setup(&tmp, &[
            ("BTCUSDT_MOTOR.json", &modelo_json(-1.38, 12)),
            ("ATOMUSDT_MOTOR.json", &modelo_json(-0.9, 8)),
        ]);
        let a = escanear_models(&tmp, 1_000);
        let b = escanear_models(&tmp, 9_999);
        assert_eq!(a.entries, b.entries, "entries deterministas");
        assert_ne!(a.generado_ms, b.generado_ms, "sello informativo");
        assert_eq!(a.entries.len(), 2);
        assert_eq!(a.entries[0].key, "ATOMUSDT_MOTOR", "orden canónico");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// FALSACIÓN (b): tocar UN modelo cambia SU hash y sólo el suyo —
    /// la identidad del artefacto es local al archivo.
    #[test]
    fn tocar_un_modelo_cambia_solo_su_hash() {
        let tmp = std::env::temp_dir().join(format!("tg_mr2_{:x}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        setup(&tmp, &[
            ("A_MOTOR.json", &modelo_json(-1.0, 4)),
            ("B_MOTOR.json", &modelo_json(-1.0, 4)),
        ]);
        let antes = escanear_models(&tmp, 0);
        // Mutar A (base distinta).
        std::fs::write(tmp.join("A_MOTOR.json"), modelo_json(-0.5, 4)).unwrap();
        let despues = escanear_models(&tmp, 0);
        let find = |m: &ModelManifest, k: &str| {
            m.entries.iter().find(|e| e.key == k).unwrap().clone()
        };
        let (a0, a1) = (find(&antes, "A_MOTOR"), find(&despues, "A_MOTOR"));
        let (b0, b1) = (find(&antes, "B_MOTOR"), find(&despues, "B_MOTOR"));
        assert_ne!(a0.sha256, a1.sha256, "A cambió");
        assert_ne!(a0.base, a1.base, "la base sigue al init_score");
        assert_eq!(b0.sha256, b1.sha256, "B intacto");
        assert_eq!(a1.n_arboles, Some(4));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Contornos: directorio ausente ⇒ manifest VACÍO (no error); JSON
    /// roto ⇒ entrada con legible=false (deuda visible, no silencio);
    /// _CANDIDATE y .bin NO se registran.
    #[test]
    fn contornos_candidatos_bin_y_rotos() {
        let vacio = escanear_models(Path::new("/no/existe/models"), 0);
        assert!(vacio.entries.is_empty());

        let tmp = std::env::temp_dir().join(format!("tg_mr3_{:x}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        setup(&tmp, &[
            ("ROTO_MOTOR.json", "{ esto no es json"),
            ("CAND_MOTOR_CANDIDATE.json", &modelo_json(-1.0, 2)),
            ("CACHE_MOTOR.bin", "binario"),
            ("BUENO_MOTOR.json", &modelo_json(-1.2, 6)),
        ]);
        let m = escanear_models(&tmp, 0);
        assert_eq!(m.entries.len(), 2, "sólo promovidos .json");
        let roto = m.entries.iter().find(|e| e.key == "ROTO_MOTOR").unwrap();
        assert!(!roto.legible, "deuda visible");
        assert!(roto.base.is_none() && roto.n_arboles.is_none());
        assert!(roto.sha256.len() == 64, "hash presente aunque roto");
        assert!(m.entries.iter().all(|e| !e.key.contains("CANDIDATE")));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// El manifest es serializable/parseable (el JSON commit-able que
    /// hace reproducible el inventario).
    #[test]
    fn manifest_roundtrip_json() {
        let tmp = std::env::temp_dir().join(format!("tg_mr4_{:x}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        setup(&tmp, &[("X_MOTOR.json", &modelo_json(-1.0, 3))]);
        let m = escanear_models(&tmp, 42);
        let ruta = tmp.join("manifest.json");
        m.escribir(&ruta).unwrap();
        let cargado = ModelManifest::cargar(&ruta).unwrap();
        assert_eq!(cargado, m);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
