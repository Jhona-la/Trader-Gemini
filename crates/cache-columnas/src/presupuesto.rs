//! Tope de la cache DERIVADO del presupuesto menos lo medido (pedido v3, P2). Ninguna constante propia:
//!
//! ```text
//! cache_max_bytes = presupuesto_total_gb*1e9 - reserva_suelo_gb*1e9 - reserva_target_gb*1e9
//!                 - medido_archivo - medido_proyecto
//! ```
//!
//! - Las tres claves salen de `cinta_config.json` (cada una con su `fuente_*` alli).
//! - `medido_archivo`: bytes de `<raiz>/cinta_datos/` y de `<raiz>/logs/captura-l2-*`.
//! - `medido_proyecto`: bytes de `<raiz>` entera (repo, `data/`, `models/`, `.git`...) SIN la cache, SIN el archivo y
//!   sin ninguna carpeta llamada `target` o `.venv` (se informan aparte, como manda `fuente_presupuesto_total`).
//! - Unidad: bytes decimales (GB = 1e9), la convencion del guion 214. Largo logico de cada archivo; los enlaces
//!   simbolicos y las uniones no se siguen. Un error de lectura aborta (contar de menos subiria el tope).

use std::path::{Path, PathBuf};

pub const BYTES_POR_GB: f64 = 1e9;
pub const CLAVES: [&str; 3] = ["presupuesto_total_gb", "reserva_suelo_gb", "reserva_target_gb"];
pub const CARPETA_ARCHIVO: &str = "cinta_datos";
pub const CARPETA_LOGS: &str = "logs";
pub const PREFIJO_CAPTURA: &str = "captura-l2-";
/// Carpetas regenerables que se informan aparte (no entran en `medido_proyecto`).
pub const EXCLUIDAS_POR_NOMBRE: [&str; 2] = ["target", ".venv"];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub presupuesto_total_gb: f64,
    pub reserva_suelo_gb: f64,
    pub reserva_target_gb: f64,
}

pub fn leer_config(p: &Path) -> Result<Config, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut x = [0f64; 3];
    for (i, k) in CLAVES.iter().enumerate() {
        x[i] = v
            .get(*k)
            .and_then(|n| n.as_f64())
            .filter(|n| n.is_finite() && *n >= 0.0)
            .ok_or_else(|| format!("{}: falta la clave numerica {k} (>= 0)", p.display()))?;
    }
    Ok(Config { presupuesto_total_gb: x[0], reserva_suelo_gb: x[1], reserva_target_gb: x[2] })
}

pub fn gb_a_bytes(gb: f64) -> i128 {
    (gb * BYTES_POR_GB).round() as i128
}

/// Donde medir y con que configuracion.
#[derive(Debug, Clone)]
pub struct Limite {
    pub config: Config,
    pub raiz_medicion: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct Medida {
    pub bytes: u64,
    pub rutas: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Sumandos {
    pub presupuesto_b: i128,
    pub suelo_b: i128,
    pub target_b: i128,
    pub archivo: Medida,
    pub proyecto: Medida,
    pub excluidas: Vec<String>,
    pub limite_b: i128,
}

impl Sumandos {
    pub fn texto(&self, uso_cache: u64) -> String {
        format!(
            "unidad: bytes decimales (GB = 1e9)\n\
             presupuesto_total   {:>16}  (cinta_config.json:presupuesto_total_gb)\n\
             - reserva_suelo     {:>16}  (cinta_config.json:reserva_suelo_gb)\n\
             - reserva_target    {:>16}  (cinta_config.json:reserva_target_gb)\n\
             - medido_archivo    {:>16}  {:?}\n\
             - medido_proyecto   {:>16}  {:?}\n\
             = limite_cache      {:>16}\n\
             uso_cache           {:>16}\n\
             excluidas (aparte): {:?}",
            self.presupuesto_b,
            self.suelo_b,
            self.target_b,
            self.archivo.bytes,
            self.archivo.rutas,
            self.proyecto.bytes,
            self.proyecto.rutas,
            self.limite_b,
            uso_cache,
            self.excluidas
        )
    }
}

fn tamano(p: &Path, excluir: &dyn Fn(&Path, &str) -> bool, excluidas: &mut Vec<String>) -> Result<u64, String> {
    let md = std::fs::symlink_metadata(p).map_err(|e| format!("{}: {e}", p.display()))?;
    if md.file_type().is_symlink() {
        return Ok(0);
    }
    if md.is_file() {
        return Ok(md.len());
    }
    let mut total = 0u64;
    let rd = std::fs::read_dir(p).map_err(|e| format!("{}: {e}", p.display()))?;
    for e in rd {
        let e = e.map_err(|x| format!("{}: {x}", p.display()))?;
        let hijo = e.path();
        let nombre = e.file_name().to_string_lossy().into_owned();
        if excluir(&hijo, &nombre) {
            excluidas.push(hijo.to_string_lossy().into_owned());
            continue;
        }
        total += tamano(&hijo, excluir, excluidas)?;
    }
    Ok(total)
}

fn canonica(p: &Path) -> Option<PathBuf> {
    std::fs::canonicalize(p).ok()
}

/// Mide los cinco sumandos y el limite.
pub fn medir(cfg: &Config, raiz: &Path, cache: &Path) -> Result<Sumandos, String> {
    let raiz = std::fs::canonicalize(raiz).map_err(|e| format!("{}: {e}", raiz.display()))?;
    let cache_c = canonica(cache);
    let mut archivo_rutas: Vec<PathBuf> = Vec::new();
    let cd = raiz.join(CARPETA_ARCHIVO);
    if std::fs::symlink_metadata(&cd).is_ok() {
        archivo_rutas.push(cd);
    }
    let logs = raiz.join(CARPETA_LOGS);
    if std::fs::symlink_metadata(&logs).map(|m| m.is_dir()).unwrap_or(false) {
        let rd = std::fs::read_dir(&logs).map_err(|e| format!("{}: {e}", logs.display()))?;
        for e in rd {
            let e = e.map_err(|x| format!("{}: {x}", logs.display()))?;
            if e.file_name().to_string_lossy().starts_with(PREFIJO_CAPTURA) {
                archivo_rutas.push(e.path());
            }
        }
    }
    archivo_rutas.sort();
    let mut excluidas = Vec::new();
    let nada = |_: &Path, _: &str| false;
    let mut archivo = Medida::default();
    for r in &archivo_rutas {
        archivo.bytes += tamano(r, &nada, &mut excluidas)?;
        archivo.rutas.push(r.to_string_lossy().into_owned());
    }
    let excluir = |p: &Path, nombre: &str| {
        EXCLUIDAS_POR_NOMBRE.contains(&nombre)
            || archivo_rutas.iter().any(|a| a == p)
            || cache_c.as_deref().map(|c| c == p).unwrap_or(false)
    };
    let proyecto =
        Medida { bytes: tamano(&raiz, &excluir, &mut excluidas)?, rutas: vec![raiz.to_string_lossy().into_owned()] };
    let presupuesto_b = gb_a_bytes(cfg.presupuesto_total_gb);
    let suelo_b = gb_a_bytes(cfg.reserva_suelo_gb);
    let target_b = gb_a_bytes(cfg.reserva_target_gb);
    let limite_b = presupuesto_b - suelo_b - target_b - archivo.bytes as i128 - proyecto.bytes as i128;
    excluidas.sort();
    Ok(Sumandos { presupuesto_b, suelo_b, target_b, archivo, proyecto, excluidas, limite_b })
}
