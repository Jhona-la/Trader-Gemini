//! IBKR-I6 (integrador, 2026-11-16): escritura de barras IBKR a la cinta.
//!
//! - Un archivo por dia UTC: `ibkr_YYYYMMDD.csv.gz` (gzip de un CSV).
//! - Esquema: `timestamp_utc,symbol,open,high,low,close,volume,wap,count`.
//! - Firma: SHA-256 (hex) del CSV SIN comprimir, en el companero
//!   `ibkr_YYYYMMDD.csv.gz.sha256`.
//! - Escribir otra vez el mismo dia agrega filas: se lee el CSV del dia, se
//!   suman las filas nuevas y se reescriben el `.gz` y su firma (cada uno por
//!   un temporal y un renombrado).

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use sha2::{Digest, Sha256};

use super::IbkrBar;

/// Carpeta de la cinta de IBKR, relativa a la raiz de datos.
pub const IBKR_CINTA_SUBDIR: &str = "cinta_datos/externas/ibkr";

/// Cabecera del CSV de la cinta.
pub const IBKR_CINTA_CABECERA: &str = "timestamp_utc,symbol,open,high,low,close,volume,wap,count";

/// Escritor de la cinta de IBKR sobre una carpeta.
#[derive(Debug, Clone)]
pub struct TapeWriter {
    dir: PathBuf,
}

/// Dia civil UTC (anio, mes, dia) de `dias` desde 1970-01-01.
/// Algoritmo `civil_from_days` de H. Hinnant (calendario gregoriano proleptico).
fn civil_desde_dias(dias: i64) -> (i64, u32, u32) {
    let z = dias + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn segundos_unix(t: SystemTime) -> io::Result<i64> {
    let d = t
        .duration_since(UNIX_EPOCH)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "barra anterior a 1970-01-01"))?;
    Ok(d.as_secs() as i64)
}

/// `YYYYMMDD` del dia UTC de `t`.
pub fn dia_utc(t: SystemTime) -> io::Result<String> {
    let s = segundos_unix(t)?;
    let (y, m, d) = civil_desde_dias(s.div_euclid(86_400));
    Ok(format!("{y:04}{m:02}{d:02}"))
}

/// `YYYY-MM-DDTHH:MM:SSZ` de `t` (precision de segundo).
pub fn timestamp_utc(t: SystemTime) -> io::Result<String> {
    let s = segundos_unix(t)?;
    let (y, m, d) = civil_desde_dias(s.div_euclid(86_400));
    let r = s.rem_euclid(86_400);
    Ok(format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", r / 3600, (r % 3600) / 60, r % 60))
}

/// SHA-256 en hex de `datos`.
pub fn sha256_hex(datos: &[u8]) -> String {
    hex::encode(Sha256::digest(datos))
}

fn fila_csv(b: &IbkrBar) -> io::Result<String> {
    if b.symbol.contains([',', '\n', '\r', '"']) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("simbolo fuera de esquema: {:?}", b.symbol)));
    }
    Ok(format!(
        "{},{},{},{},{},{},{},{},{}\n",
        timestamp_utc(b.timestamp)?,
        b.symbol,
        b.open,
        b.high,
        b.low,
        b.close,
        b.volume,
        b.wap,
        b.count
    ))
}

fn escribir_atomico(destino: &Path, datos: &[u8]) -> io::Result<()> {
    let mut tmp = destino.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(datos)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, destino)
}

impl TapeWriter {
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self { dir: dir.as_ref().to_path_buf() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Ruta del `.csv.gz` del dia `yyyymmdd`.
    pub fn ruta_dia(&self, yyyymmdd: &str) -> PathBuf {
        self.dir.join(format!("ibkr_{yyyymmdd}.csv.gz"))
    }

    /// Ruta de la firma de `csv_gz` (`<archivo>.csv.gz.sha256`).
    pub fn ruta_firma(csv_gz: &Path) -> PathBuf {
        let mut s = csv_gz.as_os_str().to_owned();
        s.push(".sha256");
        PathBuf::from(s)
    }

    /// CSV sin comprimir de un `.csv.gz` de la cinta.
    pub fn leer_csv(csv_gz: &Path) -> io::Result<String> {
        let mut texto = String::new();
        GzDecoder::new(fs::File::open(csv_gz)?).read_to_string(&mut texto)?;
        Ok(texto)
    }

    /// Escribe las barras en el archivo de su dia UTC. Devuelve las rutas de
    /// los `.csv.gz` tocados.
    pub fn write_bars(&mut self, bars: &[IbkrBar]) -> io::Result<Vec<PathBuf>> {
        let mut por_dia: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
        for b in bars {
            por_dia.entry(dia_utc(b.timestamp)?).or_default().push_str(&fila_csv(b)?);
        }
        if por_dia.is_empty() {
            return Ok(Vec::new());
        }
        fs::create_dir_all(&self.dir)?;
        let mut tocados = Vec::new();
        for (dia, filas) in por_dia {
            let ruta = self.ruta_dia(&dia);
            let mut csv = if ruta.exists() {
                let previo = Self::leer_csv(&ruta)?;
                if !previo.starts_with(IBKR_CINTA_CABECERA) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{} no empieza por la cabecera de la cinta", ruta.display()),
                    ));
                }
                previo
            } else {
                format!("{IBKR_CINTA_CABECERA}\n")
            };
            csv.push_str(&filas);
            let mut gz = GzEncoder::new(Vec::new(), Compression::default());
            gz.write_all(csv.as_bytes())?;
            let comprimido = gz.finish()?;
            escribir_atomico(&ruta, &comprimido)?;
            escribir_atomico(&Self::ruta_firma(&ruta), format!("{}\n", sha256_hex(csv.as_bytes())).as_bytes())?;
            tocados.push(ruta);
        }
        Ok(tocados)
    }
}
