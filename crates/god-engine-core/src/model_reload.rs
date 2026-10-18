//! Filesystem polling policy, separated from inference and model publication.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug)]
pub struct ReloadEvent {
    pub key: String,
    pub path: PathBuf,
    pub result: Result<(), String>,
    /// CL-141: el mismo rechazo (misma fuente y mismo error) ya se informó en
    /// un escaneo anterior. El reintento sigue; el aviso se omite.
    pub repetido: bool,
}

/// CL-141 (C-F6) — UN AVISO POR RECHAZO, NO UNO CADA 10 s.
///
/// QUÉ ESTABA MAL: un modelo inválido en `models/` se reintenta en cada
/// escaneo (cada 10 s, por si el fallo era transitorio) y el host avisaba en
/// cada uno: ≈ 1,5 MB/día por archivo, 8 640 líneas idénticas.
///
/// QUÉ GARANTIZA: el reintento no cambia (un fallo nunca avanza `applied`),
/// pero un rechazo se marca `repetido` mientras su fuente y su error sean
/// los del último aviso. Un cambio del archivo o del error vuelve a avisar.
#[derive(Default)]
pub struct ModelReloadTracker {
    applied: HashMap<String, SourceStamp>,
    /// Último rechazo informado por clave: fuente (si se pudo observar) y
    /// error. Acotado por los archivos de `models/`, como `applied`.
    rechazos: HashMap<String, (Option<SourceStamp>, String)>,
    ultimo_error_de_escaneo: Option<String>,
}

// A change detector, NOT a content hash, generation or promotion certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceStamp {
    path: PathBuf,
    modified: SystemTime,
    bytes: u64,
}

impl ModelReloadTracker {
    /// CL-141: `true` si este rechazo de `key` no es el último informado.
    fn rechazo_nuevo(&mut self, key: &str, fuente: Option<SourceStamp>, error: &str) -> bool {
        let huella = (fuente, error.to_owned());
        if self.rechazos.get(key) == Some(&huella) {
            return false;
        }
        self.rechazos.insert(key.to_owned(), huella);
        true
    }

    /// CL-141: `true` si el directorio no se puede leer por un error distinto
    /// del último informado. Un escaneo que lee el directorio lo olvida.
    pub fn error_de_escaneo_nuevo(&mut self, error: &str) -> bool {
        if self.ultimo_error_de_escaneo.as_deref() == Some(error) {
            return false;
        }
        self.ultimo_error_de_escaneo = Some(error.to_owned());
        true
    }

    /// One attempt per key per scan. A JSON takes precedence over its sibling
    /// BIN; standalone legacy BINs remain supported. Failed attempts never
    /// advance applied state, so transient failures retry on the next scan.
    /// The caller owns validated publication and logging. Deletion does not
    /// revoke a live model; that requires a separate explicit lifecycle policy.
    pub fn scan(
        &mut self,
        directory: &Path,
        mut load: impl FnMut(&str, &Path) -> Result<(), String>,
    ) -> std::io::Result<Vec<ReloadEvent>> {
        let mut events = Vec::new();
        let mut paths = std::fs::read_dir(directory)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        self.ultimo_error_de_escaneo = None;
        paths.sort();
        for path in paths {
            let ext = path.extension().and_then(|s| s.to_str());
            if ext != Some("json") && ext != Some("bin") {
                continue;
            }
            let Some(key) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if ext == Some("bin") {
                match path.with_extension("json").try_exists() {
                    Ok(true) => continue,
                    Ok(false) => {}
                    Err(error) => {
                        let error = format!("cannot establish source precedence: {error}");
                        let repetido = !self.rechazo_nuevo(key, None, &error);
                        events.push(ReloadEvent {
                            key: key.to_owned(),
                            path,
                            result: Err(error),
                            repetido,
                        });
                        continue;
                    }
                }
            }
            let stamp = match std::fs::metadata(&path).and_then(|metadata| {
                if !metadata.is_file() {
                    return Ok(None);
                }
                Ok(Some(SourceStamp {
                    path: path.clone(),
                    modified: metadata.modified()?,
                    bytes: metadata.len(),
                }))
            }) {
                Ok(Some(stamp)) => stamp,
                Ok(None) => continue,
                Err(error) => {
                    let error = format!("cannot observe model source: {error}");
                    let repetido = !self.rechazo_nuevo(key, None, &error);
                    events.push(ReloadEvent {
                        key: key.to_owned(),
                        path,
                        result: Err(error),
                        repetido,
                    });
                    continue;
                }
            };
            if self.applied.get(key) == Some(&stamp) {
                continue;
            }
            let result = load(key, &path);
            let repetido = match &result {
                Ok(()) => {
                    self.rechazos.remove(key);
                    self.applied.insert(key.to_owned(), stamp);
                    false
                }
                Err(error) => !self.rechazo_nuevo(key, Some(stamp), error),
            };
            events.push(ReloadEvent {
                key: key.to_owned(),
                path,
                result,
                repetido,
            });
        }
        Ok(events)
    }
}

/// Resuelve la ruta canónica del directorio de modelos buscando en los
/// directorios de ejecución habituales (raíz del workspace, subcrates o TG_DATA_ROOT).
pub fn resolve_models_directory() -> Option<PathBuf> {
    if let Ok(root) = std::env::var("TG_DATA_ROOT") {
        let p = Path::new(&root).parent().map(|p| p.join("models"));
        if let Some(ref path) = p {
            if path.is_dir() {
                return p;
            }
        }
    }
    for candidate in [
        Path::new("models"),
        Path::new("../../models"),
        Path::new("../../../models"),
    ] {
        if candidate.is_dir() {
            return Some(candidate.to_path_buf());
        }
    }
    None
}

/// Precarga todos los modelos válidos encontrados en el directorio `models/`
/// hacia el almacenamiento global lock-free `GLOBAL_FORESTS`.
/// Garantiza paridad absoluta entre producción, replay, backtest y censo forense.
pub fn preload_models_from_disk() -> usize {
    let Some(models_dir) = resolve_models_directory() else {
        return 0;
    };
    let mut tracker = ModelReloadTracker::default();
    let scan = tracker.scan(&models_dir, |key, path| {
        let source = path.to_str().ok_or_else(|| "model path is not UTF-8".to_string())?;
        crate::ml_inference::NanoForest::load_global(key, source)
            .map_err(|error| error.to_string())
    });
    match scan {
        Ok(events) => {
            let mut count = 0;
            for event in events {
                if event.result.is_ok() {
                    count += 1;
                }
            }
            count
        }
        Err(_) => 0,
    }
}

