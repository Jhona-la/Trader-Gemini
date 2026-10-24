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
    /// Último rechazo informado por archivo: fuente (si se pudo observar) y
    /// error. CL-141c: por ruta, no por clave (`X.bin` y `X.json` comparten
    /// clave y se pisaban la huella, avisando en cada escaneo); sólo quedan
    /// los archivos del último escaneo.
    rechazos: HashMap<PathBuf, (Option<SourceStamp>, String)>,
    ultimo_error_de_escaneo: Option<String>,
}

/// CL-141b (C-F6): artefactos de `models/` que no son bosques y tienen su
/// propio cargador. `DarkAlpha_{SÍMBOLO}.json` son los pesos de la red de
/// DarkAlpha, que el host vigila aparte y que el entrenador automático
/// reescribe en cada mejora. El escáner de bosques los intentaba cargar, el
/// rechazo era seguro y cada reescritura traía una fuente nueva, así que el
/// aviso volvía con cada promoción del entrenador.
pub const PREFIJOS_AJENOS: &[&str] = &["DarkAlpha_"];

// A change detector, NOT a content hash, generation or promotion certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceStamp {
    path: PathBuf,
    modified: SystemTime,
    bytes: u64,
}

impl ModelReloadTracker {
    /// CL-141: `true` si este rechazo de `ruta` no es el último informado.
    fn rechazo_nuevo(&mut self, ruta: &Path, fuente: Option<SourceStamp>, error: &str) -> bool {
        let huella = (fuente, error.to_owned());
        if self.rechazos.get(ruta) == Some(&huella) {
            return false;
        }
        self.rechazos.insert(ruta.to_path_buf(), huella);
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
        self.rechazos.retain(|ruta, _| paths.binary_search(ruta).is_ok());
        for path in paths {
            let ext = path.extension().and_then(|s| s.to_str());
            if ext != Some("json") && ext != Some("bin") {
                continue;
            }
            let Some(key) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if PREFIJOS_AJENOS.iter().any(|prefijo| key.starts_with(prefijo)) {
                continue;
            }
            if ext == Some("bin") {
                match path.with_extension("json").try_exists() {
                    Ok(true) => continue,
                    Ok(false) => {}
                    Err(error) => {
                        let error = format!("cannot establish source precedence: {error}");
                        let repetido = !self.rechazo_nuevo(&path, None, &error);
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
                    let repetido = !self.rechazo_nuevo(&path, None, &error);
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
                    self.rechazos.remove(&path.with_extension("bin"));
                    self.rechazos.remove(&path.with_extension("json"));
                    self.applied.insert(key.to_owned(), stamp);
                    false
                }
                Err(error) => !self.rechazo_nuevo(&path, Some(stamp), error),
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

#[cfg(test)]
mod tests_cl141c {
    use super::*;

    /// CL-141c — Esperado: dos rechazos de la misma clave desde archivos
    /// distintos (`A.bin`, `A.json`) avisan una vez cada uno y después son
    /// repetidos. Antes compartían la huella por clave y se alternaban: dos
    /// avisos cada 10 s.
    #[test]
    fn cl141c_dos_archivos_con_la_misma_clave_no_se_pisan() {
        let mut t = ModelReloadTracker::default();
        let (bin, json) = (Path::new("models/A.bin"), Path::new("models/A.json"));
        assert!(t.rechazo_nuevo(bin, None, "cannot establish source precedence: e"));
        assert!(t.rechazo_nuevo(json, None, "cannot observe model source: e"));
        for _ in 0..3 {
            assert!(!t.rechazo_nuevo(bin, None, "cannot establish source precedence: e"));
            assert!(!t.rechazo_nuevo(json, None, "cannot observe model source: e"));
        }
    }

    /// CL-141c — Esperado: un rechazo de un archivo que ya no está se olvida
    /// en el siguiente escaneo; si el archivo vuelve con el mismo error, se
    /// avisa de nuevo.
    #[test]
    fn cl141c_un_archivo_borrado_se_olvida() {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("cl141c_{}_{id}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ruta = dir.join("X_MOTOR.json");
        std::fs::write(&ruta, "{}").unwrap();
        let mut t = ModelReloadTracker::default();
        let rechaza = |_: &str, _: &Path| Err("inválido".to_string());
        let primero = t.scan(&dir, rechaza).unwrap();
        assert!(primero.iter().all(|e| !e.repetido));
        assert!(t.scan(&dir, rechaza).unwrap().iter().all(|e| e.repetido));
        std::fs::remove_file(&ruta).unwrap();
        assert!(t.scan(&dir, rechaza).unwrap().is_empty());
        assert!(t.rechazos.is_empty(), "sin archivo no queda huella");
        let _ = std::fs::remove_dir_all(&dir);
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

/// F1-106 (Ola Ω164): Carga explícitamente el modelo DarkAlpha para un símbolo (e.g. BTCUSDT)
/// desde models/ resolviendo la ruta canónica centralizada.
///
/// Cumple la doctrina LG-23 ("construir el núcleo NO carga modelos del disco") permitiendo
/// que los evaluadores de mutantes y oráculos no paguen E/S ni alocación JSON involuntaria.
pub fn load_dark_alpha_model(symbol: &str) -> Option<dark_alpha_engine::DarkAlphaEngine> {
    let filename = format!("DarkAlpha_{}.json", symbol);
    let path = resolve_models_directory()?.join(filename);
    if !path.is_file() {
        return None;
    }
    let json_data = std::fs::read_to_string(&path).ok()?;
    let mut model = serde_json::from_str::<dark_alpha_engine::DarkAlphaEngine>(&json_data).ok()?;
    model.init_buffers();
    model.sanitize_denormals();
    model.freeze();
    Some(model)
}

