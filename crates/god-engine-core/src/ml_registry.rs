//! REGISTRO DE MODELOS (Ola XLVIII·G — prioridad #1 del marco del
//! operador: "model registry que permita reproducir cualquier señal").
//!
//! `models/` está en `.gitignore` (artefactos locales): un clon fresco
//! mide el piso sin modelos. Este módulo convierte esa laguna en un
//! contrato: un MANIFIESTO versionado (JSON, commit-able) con huella
//! SHA-256 por modelo, base del modelo (sigmoid(init_score), B3.36),
//! número de árboles y metadatos de archivo. El hash identifica los bytes
//! escaneados, NO el predictor activado: el watcher no consulta este manifest
//! y una caché .bin puede ser la fuente de serving. MR-01 separa legibilidad
//! JSON de validez estructural sin cambiar activación ni certificar promoción.
//!
//! Contrato:
//! - **Variable**: cada `models/{KEY}.json` no candidato (los `_CANDIDATE`
//!   y `.bin` cacheados NO se registran). La presencia no prueba activación.
//! - **Operador**: SHA-256 del archivo + campos extraídos del JSON
//!   (init_score → base; tree_offsets → nº árboles).
//! - **Unidades**: base ∈ [0,1] (sigmoid); hash hex 64 caracteres.
//! - **Contorno**: dir ausente ⇒ manifest vacío (no error); JSON
//!   ilegible ⇒ entrada con `legible: false` y deuda visible, no
//!   silencio.
//! - **Identificabilidad**: el hash identifica el archivo EXACTO; base es
//!   el metadato legado sigmoid(init_score), no una prueba de calibración.
//! - **Coste**: O(bytes + nodos) al escanear; fuera del hot path.
//! - **Falsación**: mismo directorio ⇒ mismo manifest byte a byte
//!   (determinismo); tocar un modelo ⇒ cambia SU hash y sólo el suyo.

use crate::ml_inference::{NanoForest, NanoForestData};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Un archivo fuente inventariado; ni su nombre ni su hash prueban promoción.
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
    /// Metadato legado sigmoid(init_score), pertinente para clasificación.
    /// En regresión NO expresa la media cruda ni las unidades del label.
    /// None si el JSON no se pudo leer (deuda visible).
    pub base: Option<f64>,
    /// Número de árboles (tree_offsets.len()−1); None si ilegible.
    pub n_arboles: Option<usize>,
    /// false ⇒ el archivo existe pero el JSON no se pudo parsear.
    pub legible: bool,
    /// MR-01: contrato estructural del cargador aplicado a ESTOS bytes JSON.
    /// None en manifests históricos sin evaluación; Some(false) si no es un
    /// NanoForestData válido. Some(true) NO prueba esquema semántico, habilidad,
    /// holdout, autorización de promoción ni identidad del modelo en memoria.
    #[serde(default)]
    pub valido_estructuralmente: Option<bool>,
    /// Motivo de rechazo estructural, independiente de la legibilidad JSON.
    #[serde(default)]
    pub error_estructural: Option<String>,
}

/// Inventario de fuentes locales; no es un registro de aprobaciones ni activaciones.
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

/// Escanea `models_dir` y produce el inventario de fuentes no candidatas
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
            let (base, n_arboles, legible, error_estructural) =
                match serde_json::from_slice::<serde_json::Value>(&bytes_vec) {
                    Ok(v) => {
                        let init = v.get("init_score").and_then(|x| x.as_f64());
                        let base = init.map(|i| 1.0 / (1.0 + (-i).exp()));
                        let n = v
                            .get("tree_offsets")
                            .and_then(|x| x.as_array())
                            .map(|a| a.len().saturating_sub(1));
                        // Pure in-memory validation: do not call load_model,
                        // which may prefer/write a .bin cache. Hash, metadata
                        // and structural verdict must describe the same bytes.
                        let validation = serde_json::from_value::<NanoForestData>(v)
                            .map_err(|e| e.to_string())
                            .and_then(NanoForest::from_data);
                        (base, n, true, validation.err())
                    }
                    Err(e) => (None, None, false, Some(e.to_string())),
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
                valido_estructuralmente: Some(error_estructural.is_none()),
                error_estructural,
            });
        }
    }
    entries.sort_by(|a, b| a.key.cmp(&b.key));
    ModelManifest { entries, generado_ms }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_leaf() -> NanoForestData {
        NanoForestData {
            children_left: vec![-1],
            children_right: vec![-1],
            feature: vec![-1],
            threshold: vec![0.0],
            value: vec![0.5],
            tree_offsets: vec![0, 1],
            init_score: -1.0,
        }
    }

    #[test]
    fn mr_json_legible_no_certifica_estructura_y_no_escribe_cache() {
        let tmp = std::env::temp_dir().join(format!("tg_mr_valid_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let mut cycle = valid_leaf();
        cycle.children_left[0] = 0;
        cycle.children_right[0] = 0;
        cycle.feature[0] = 0;
        setup(&tmp, &[
            ("VALID_MOTOR.json", &serde_json::to_string(&valid_leaf()).unwrap()),
            ("CYCLE_MOTOR.json", &serde_json::to_string(&cycle).unwrap()),
            ("EMPTY_MOTOR.json", "{}"),
            ("SCALAR_MOTOR.json", "null"),
            ("BROKEN_MOTOR.json", "{"),
        ]);
        let m = escanear_models(&tmp, 1);
        assert_eq!(m.entries.len(), 5);
        assert_eq!(m.entries.iter().filter(|e| e.legible).count(), 4);
        assert_eq!(m.entries.iter().filter(|e| e.valido_estructuralmente == Some(true)).count(), 1);
        for e in &m.entries {
            assert_eq!(e.sha256.len(), 64);
            assert!(!tmp.join(format!("{}.bin", e.key)).exists(), "scanner must not create cache");
            if e.key == "VALID_MOTOR" {
                assert!(e.error_estructural.is_none());
            } else {
                assert_eq!(e.valido_estructuralmente, Some(false));
                assert!(e.error_estructural.is_some());
            }
        }
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn mr_veredicto_sigue_contrato_del_loader_sin_activar_modelos() {
        let tmp = std::env::temp_dir().join(format!("tg_mr_contract_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let mut lengths = valid_leaf();
        lengths.value.clear();
        let mut offsets = valid_leaf();
        offsets.tree_offsets = vec![0, 2];
        let mut wide = valid_leaf();
        wide.children_left = vec![1, -1, -1];
        wide.children_right = vec![2, -1, -1];
        wide.feature = vec![NanoForest::ML_VECTOR_DIM as i32, -1, -1];
        wide.threshold = vec![0.0; 3];
        wide.value = vec![0.0; 3];
        wide.tree_offsets = vec![0, 3];
        for (i, data) in [valid_leaf(), lengths, offsets, wide].into_iter().enumerate() {
            std::fs::write(tmp.join(format!("{i}_MOTOR.json")), serde_json::to_string(&data).unwrap()).unwrap();
            let entry = escanear_models(&tmp, 0).entries.into_iter()
                .find(|e| e.key == format!("{i}_MOTOR")).unwrap();
            let valid = NanoForest::from_data(data).is_ok();
            assert_eq!(valid, i == 0, "fixture must exercise loader rejection");
            assert_eq!(entry.valido_estructuralmente, Some(valid));
        }
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn mr_manifest_historico_no_inventa_validacion() {
        let legacy = r#"{"entries":[{"key":"X_MOTOR","archivo":"X_MOTOR.json","sha256":"legacy","bytes":2,"base":null,"n_arboles":null,"legible":true}],"generado_ms":0}"#;
        let m: ModelManifest = serde_json::from_str(legacy).unwrap();
        assert_eq!(m.entries[0].valido_estructuralmente, None);
        assert_eq!(m.entries[0].error_estructural, None);
        let roundtrip: ModelManifest = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(m, roundtrip);
    }

    /// MR-03 OPEN: witness of a provenance gap, not a certification of serving.
    /// A structurally valid source hash does not identify a newer binary cache.
    #[test]
    fn mr_hash_json_no_identifica_necesariamente_predictor_cargado() {
        let tmp = std::env::temp_dir().join(format!("tg_mr_identity_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let json = tmp.join("AUDIT_MOTOR.json");
        let bin = tmp.join("AUDIT_MOTOR.bin");
        let source = valid_leaf();
        let mut cached = source.clone();
        cached.init_score = 2.0;
        let source_bytes = serde_json::to_vec(&source).unwrap();
        let cached_bytes = bincode::serialize(&cached).unwrap();
        std::fs::write(&json, &source_bytes).unwrap();
        std::fs::write(&bin, &cached_bytes).unwrap();
        let t = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_600_000_000);
        std::fs::File::options().write(true).open(&json).unwrap().set_modified(t).unwrap();
        std::fs::File::options().write(true).open(&bin).unwrap()
            .set_modified(t + std::time::Duration::from_secs(60)).unwrap();
        let entry = escanear_models(&tmp, 0).entries.remove(0);
        assert_eq!(entry.valido_estructuralmente, Some(true));
        assert_eq!(entry.sha256, format!("{:x}", Sha256::digest(&source_bytes)));
        assert_eq!(std::fs::read(&bin).unwrap(), cached_bytes, "scanner must not rewrite cache");
        let loaded = NanoForest::load_model(json.to_str().unwrap()).unwrap();
        assert_eq!(loaded.init_value(), 2.0, "current loader selects the newer bin");
        assert!((entry.base.unwrap() - loaded.base_prob()).abs() > 0.5);
        assert_eq!(std::fs::read(&json).unwrap(), source_bytes);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

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
