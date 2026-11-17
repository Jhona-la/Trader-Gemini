//! Cache de columnas de las cintas de ticks (pedido 20261116-2345-cachev3, especificacion
//! `glm_app_pedido_cache_columnas_v3.md`).
//!
//! - Es un DERIVADO 100 % regenerable: todo lo que hay bajo la raiz de la cache se rehace desde la cinta de origen
//!   (`X.bin` o `X.bin.tgb`, leida por la cinta logica de `tick-tape`: si existen las dos manda el `.bin`).
//! - Unidad de construccion: la HORA de evento (`ts / MS_POR_HORA`) de una cinta. Cada hora se escribe en
//!   `<raiz>/.tmp-...`, se relee y se verifica, y solo entonces se renombra a
//!   `<raiz>/<fuente>/<simbolo>/<AAAA-MM-DD>/<HH>/` y se anota en `MANIFIESTO.json`.
//! - Columnas: `ts` (u64 LE), `bid`, `ask`, `bq`, `aq` (bits IEEE-754 f64 LE, SIN escala: el origen es f64 y una
//!   escala decimal perderia bits) y `pub_ms` (i64 LE, momento de PUBLICACION segun [`publicacion::TABLA`]).
//! - Nada del motor vivo la lee; `train_forest` la consulta solo con `--cache columnas` (opt-in).

pub mod construir;
pub mod presupuesto;
pub mod publicacion;
pub mod verificar;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

pub const CONTRATO_INDICE: &str = "cache-columnas-indice/1";
pub const CONTRATO_MANIFIESTO: &str = "cache-columnas-manifiesto/1";
pub const NOMBRE_MANIFIESTO: &str = "MANIFIESTO.json";
pub const NOMBRE_CERROJO: &str = "MANIFIESTO.lock";
pub const NOMBRE_EVICCIONES: &str = "EVICCIONES.jsonl";
pub const NOMBRE_INDICE: &str = "index.json";
pub const PREFIJO_TMP: &str = ".tmp-";
/// Cerrojo de lectura por hora: la expulsion nunca toca una hora con alguno.
pub const PREFIJO_CERROJO_LECTURA: &str = ".lectura-";
/// Ruta por defecto de la cache, relativa a la raiz del repo (fuera de data/, models/ y config_dir/).
pub const RUTA_POR_DEFECTO: &str = "cache/columnas";

/// Cabecera opcional de las cintas de ticks (la misma que reconocen `train_forest` y `tick-tape`).
pub const CABECERA_TICKS: &[u8; 8] = b"TGMTICK1";
/// Bytes de una palabra del registro `BinTick` (`#[repr(C)]`: ts u64 y bid, ask, bq, aq f64).
pub const BYTES_PALABRA: usize = std::mem::size_of::<u64>();

/// Unidades de reloj (milisegundos), no parametros.
pub const MS_POR_SEGUNDO: u64 = 1_000;
pub const MS_POR_HORA: u64 = 60 * 60 * MS_POR_SEGUNDO;
pub const MS_POR_DIA: u64 = 24 * MS_POR_HORA;

/// Una columna del contrato: nombre del campo, archivo y tipo.
#[derive(Debug, Clone, Copy)]
pub struct DefColumna {
    pub field: &'static str,
    pub file: &'static str,
    pub tipo: &'static str,
}

/// Las 5 palabras del registro, en su orden, seguidas de `pub_ms`.
pub const COLUMNAS_REGISTRO: [DefColumna; 5] = [
    DefColumna { field: "ts", file: "ts.u64le", tipo: "u64le" },
    DefColumna { field: "bid", file: "bid.f64le", tipo: "f64le" },
    DefColumna { field: "ask", file: "ask.f64le", tipo: "f64le" },
    DefColumna { field: "bq", file: "bq.f64le", tipo: "f64le" },
    DefColumna { field: "aq", file: "aq.f64le", tipo: "f64le" },
];
pub const COLUMNA_PUB: DefColumna = DefColumna { field: "pub_ms", file: "pub_ms.i64le", tipo: "i64le" };
pub const PALABRAS_POR_REGISTRO: usize = COLUMNAS_REGISTRO.len();
pub const BYTES_REGISTRO: usize = PALABRAS_POR_REGISTRO * BYTES_PALABRA;

/// Todas las columnas de una hora, en el orden del indice.
pub fn columnas() -> Vec<DefColumna> {
    let mut v = COLUMNAS_REGISTRO.to_vec();
    v.push(COLUMNA_PUB);
    v
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnaIndice {
    pub file: String,
    pub field: String,
    #[serde(rename = "type")]
    pub tipo: String,
    /// Siempre `null`: bits exactos del origen, sin escala decimal.
    pub scale: Option<u32>,
    pub rows: u64,
    pub bytes: u64,
    pub sha256: String,
}

/// `index.json` de una hora. Sin fecha de construccion: reconstruir da los MISMOS bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Indice {
    pub contrato: String,
    pub fuente: String,
    pub simbolo: String,
    pub hora_ms: u64,
    pub rows: u64,
    pub sha256_original: String,
    pub bytes_original: u64,
    pub registro_desde: u64,
    pub registro_hasta: u64,
    /// SHA-256 de los bytes de los registros `[registro_desde, registro_hasta)` del original.
    pub sha256_tramo: String,
    pub regla_publicacion: String,
    pub pub_min_ms: i64,
    pub pub_max_ms: i64,
    pub columnas: Vec<ColumnaIndice>,
}

/// Una hora en el manifiesto global.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntradaHora {
    /// Ruta relativa a la raiz de la cache, con `/`.
    pub clave: String,
    pub fuente: String,
    pub simbolo: String,
    pub hora_ms: u64,
    pub origen_logica: String,
    pub origen_fisica: String,
    pub representacion: String,
    pub sha256_original: String,
    pub bytes_original: u64,
    pub sha256_contenedor: Option<String>,
    pub sha256_indice: String,
    pub construida_ms: u64,
    pub ultimo_acceso_ms: Option<u64>,
    pub bytes: u64,
    pub rows: u64,
    pub registro_desde: u64,
    pub registro_hasta: u64,
    pub pub_min_ms: i64,
    pub pub_max_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifiesto {
    pub contrato: String,
    pub horas: Vec<EntradaHora>,
}

impl Default for Manifiesto {
    fn default() -> Self {
        Manifiesto { contrato: CONTRATO_MANIFIESTO.to_string(), horas: Vec::new() }
    }
}

impl Manifiesto {
    pub fn uso_bytes(&self) -> u64 {
        self.horas.iter().map(|h| h.bytes).sum()
    }
    pub fn posicion(&self, clave: &str) -> Option<usize> {
        self.horas.iter().position(|h| h.clave == clave)
    }
}

pub fn hex(b: &[u8]) -> String {
    const D: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push(D[(x >> 4) as usize] as char);
        s.push(D[(x & 15) as usize] as char);
    }
    s
}

pub fn sha256_hex(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

pub fn ahora_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Fecha civil UTC (anio, mes, dia) de un dia desde la epoca (algoritmo de H. Hinnant, `civil_from_days`).
pub fn fecha_civil(dias: i64) -> (i64, u32, u32) {
    let z = dias + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `<fuente>/<simbolo>/<AAAA-MM-DD>/<HH>` de una hora de evento.
pub fn clave_hora(fuente: &str, simbolo: &str, hora_ms: u64) -> String {
    let dias = (hora_ms / MS_POR_DIA) as i64;
    let (y, m, d) = fecha_civil(dias);
    let hh = (hora_ms % MS_POR_DIA) / MS_POR_HORA;
    format!("{fuente}/{simbolo}/{y:04}-{m:02}-{d:02}/{hh:02}")
}

/// Nombres que pueden ser un componente de ruta de la cache (sin separadores, `..` ni vacios).
pub fn nombre_valido(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Ruta de una clave del manifiesto, solo si es relativa y sin `..` (nunca sale de la raiz).
pub fn ruta_de_clave(raiz: &Path, clave: &str) -> Result<PathBuf, String> {
    let rel = Path::new(clave);
    if clave.is_empty() || !rel.components().all(|c| matches!(c, Component::Normal(_))) {
        return Err(format!("clave del manifiesto fuera de la cache: {clave:?}"));
    }
    Ok(raiz.join(rel))
}

/// Reconstruye los bytes de los registros intercalando las 5 columnas del registro (la inversa exacta de la
/// construccion: cada palabra se copia tal cual, sin pasar por f64).
pub fn registros_desde_columnas(cols: &[&[u8]], rows: usize) -> Result<Vec<u8>, String> {
    if cols.len() != PALABRAS_POR_REGISTRO {
        return Err(format!("se esperaban {PALABRAS_POR_REGISTRO} columnas de registro, hay {}", cols.len()));
    }
    let largo = rows.checked_mul(BYTES_PALABRA).ok_or("filas fuera de rango")?;
    if let Some(c) = cols.iter().position(|c| c.len() != largo) {
        return Err(format!("columna {} con {} B, se esperaban {largo}", COLUMNAS_REGISTRO[c].field, cols[c].len()));
    }
    let mut v = Vec::new();
    v.try_reserve_exact(rows * BYTES_REGISTRO).map_err(|e| e.to_string())?;
    for i in 0..rows {
        for c in cols {
            v.extend_from_slice(&c[i * BYTES_PALABRA..(i + 1) * BYTES_PALABRA]);
        }
    }
    Ok(v)
}

pub fn leer_u64(b: &[u8], i: usize) -> u64 {
    let mut w = [0u8; BYTES_PALABRA];
    w.copy_from_slice(&b[i * BYTES_PALABRA..(i + 1) * BYTES_PALABRA]);
    u64::from_le_bytes(w)
}

pub fn leer_i64(b: &[u8], i: usize) -> i64 {
    let mut w = [0u8; BYTES_PALABRA];
    w.copy_from_slice(&b[i * BYTES_PALABRA..(i + 1) * BYTES_PALABRA]);
    i64::from_le_bytes(w)
}

pub fn leer_manifiesto(raiz: &Path) -> Result<Manifiesto, String> {
    let p = raiz.join(NOMBRE_MANIFIESTO);
    match std::fs::read(&p) {
        Ok(b) => {
            let m: Manifiesto = serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", p.display()))?;
            if m.contrato != CONTRATO_MANIFIESTO {
                return Err(format!("{}: contrato {:?} desconocido", p.display(), m.contrato));
            }
            Ok(m)
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Manifiesto::default()),
        Err(e) => Err(format!("{}: {e}", p.display())),
    }
}

/// Escribe el manifiesto entero en un temporal y lo renombra encima (reemplazo atomico).
pub fn escribir_manifiesto(raiz: &Path, m: &Manifiesto) -> Result<(), String> {
    let p = raiz.join(NOMBRE_MANIFIESTO);
    let tmp = raiz.join(format!("{NOMBRE_MANIFIESTO}.tmp-{}", std::process::id()));
    let bytes = serde_json::to_vec_pretty(m).map_err(|e| e.to_string())?;
    escribir_sincronizado(&tmp, &bytes)?;
    std::fs::rename(&tmp, &p).map_err(|e| format!("{} -> {}: {e}", tmp.display(), p.display()))
}

pub fn escribir_sincronizado(p: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut f = File::create(p).map_err(|e| format!("{}: {e}", p.display()))?;
    f.write_all(bytes).map_err(|e| format!("{}: {e}", p.display()))?;
    f.sync_all().map_err(|e| format!("{}: {e}", p.display()))
}

/// Cerrojo exclusivo del manifiesto (archivo creado con `create_new`); se suelta al soltar el valor.
pub struct Cerrojo {
    ruta: PathBuf,
}

impl Cerrojo {
    pub fn tomar(raiz: &Path) -> Result<Cerrojo, String> {
        std::fs::create_dir_all(raiz).map_err(|e| format!("{}: {e}", raiz.display()))?;
        let ruta = raiz.join(NOMBRE_CERROJO);
        match OpenOptions::new().write(true).create_new(true).open(&ruta) {
            Ok(mut f) => {
                let _ = writeln!(f, "pid={}", std::process::id());
                Ok(Cerrojo { ruta })
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Err(format!(
                "cache ocupada: existe {} (otro proceso la usa, o quedo de una corrida caida: comprobar y retirarlo a mano)",
                ruta.display()
            )),
            Err(e) => Err(format!("{}: {e}", ruta.display())),
        }
    }
}

impl Drop for Cerrojo {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.ruta);
    }
}
