//! Cinta LOGICA: el motor nombra siempre `X.bin`; fisicamente puede estar como `X.bin` o como `X.bin.tgb`.
//!
//! Reglas (pedido CODEX-CLI, Parte 1 y Parte 2 paso 1; adenda (d) del coordinador):
//! - manda el `.bin` que se pueda abrir; si existen los dos, se usa el `.bin` y el `.tgb` no se mira;
//! - solo un `ErrorKind::NotFound` al abrir el `.bin` permite probar `X.bin.tgb` (la ruta entera con `.tgb`
//!   anadido, sin conversion con perdida del nombre). PermissionDenied, un directorio, un fallo de mmap o de formato
//!   NO cambian de fuente: el error se devuelve tal cual;
//! - si faltan los dos, `NotFound` con las dos rutas;
//! - no hay `exists()`: un error de acceso no se convierte en ausencia.

use crate::tgb::{fallo_de, restituir_tgb_con, ErrorTgb, FalloTgb, IdentidadEsperada, Presupuesto, Restituido};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// `X.bin` -> `X.bin.tgb` (anade `.tgb` al nombre completo; no usa `with_extension`).
pub fn ruta_tgb(logica: &Path) -> PathBuf {
    let mut s = logica.as_os_str().to_os_string();
    s.push(".tgb");
    PathBuf::from(s)
}

/// Representacion fisica de una cinta logica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representacion {
    Bin,
    Tgb,
}

/// Como llegaron los bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transporte {
    /// El `.bin` mapeado en memoria.
    Bin,
    /// El contenedor `.bin.tgb`: tamano y SHA-256 del ARCHIVO entero (el del manifiesto `sha256_tgb`, sello incluido)
    /// y modo del formato. No sustituyen la identidad del original.
    Tgb { bytes_contenedor: u64, sha256_contenedor: [u8; 32], modo: u8 },
}

enum Datos {
    Mapa(memmap2::Mmap),
    Restituido(Restituido),
}

/// Bytes de una cinta logica, ya verificados si venian en TGB. Inmutables mientras viva el objeto.
pub struct TapeBytes {
    ruta_logica: PathBuf,
    ruta_fisica: PathBuf,
    transporte: Transporte,
    datos: Datos,
}

impl TapeBytes {
    pub fn as_bytes(&self) -> &[u8] {
        match &self.datos {
            Datos::Mapa(m) => &m[..],
            Datos::Restituido(r) => r.as_bytes(),
        }
    }
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }
    pub fn is_empty(&self) -> bool {
        self.as_bytes().is_empty()
    }
    pub fn ruta_logica(&self) -> &Path {
        &self.ruta_logica
    }
    pub fn ruta_fisica(&self) -> &Path {
        &self.ruta_fisica
    }
    pub fn transporte(&self) -> Transporte {
        self.transporte
    }
    pub fn representacion(&self) -> Representacion {
        match self.transporte {
            Transporte::Bin => Representacion::Bin,
            Transporte::Tgb { .. } => Representacion::Tgb,
        }
    }
    /// SHA-256 del ORIGINAL (en TGB, el ya verificado; en `.bin`, calculado sobre los bytes mapeados).
    pub fn sha256_original(&self) -> [u8; 32] {
        match &self.datos {
            Datos::Mapa(m) => Sha256::digest(&m[..]).into(),
            Datos::Restituido(r) => r.sha256_original(),
        }
    }
}

fn abrir_archivo(p: &Path) -> io::Result<File> {
    File::open(p)
}

fn sin_ninguna(logica: &Path, tgb: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!("cinta ausente: no existe {} ni {}", logica.display(), tgb.display()),
    )
}

/// Abre la representacion que manda: el `.bin`, o el `.tgb` solo si el `.bin` da NotFound.
fn elegir(logica: &Path, abrir: &dyn Fn(&Path) -> io::Result<File>) -> io::Result<(Representacion, PathBuf, File)> {
    match abrir(logica) {
        Ok(f) => Ok((Representacion::Bin, logica.to_path_buf(), f)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let tgb = ruta_tgb(logica);
            match abrir(&tgb) {
                Ok(f) => Ok((Representacion::Tgb, tgb, f)),
                Err(e2) if e2.kind() == io::ErrorKind::NotFound => Err(sin_ninguna(logica, &tgb)),
                Err(e2) => Err(e2),
            }
        }
        Err(e) => Err(e),
    }
}

/// Que representacion de la cinta hay, sin leerla. `NotFound` solo si faltan las dos.
pub fn representacion(logica: &Path) -> io::Result<Representacion> {
    elegir(logica, &abrir_archivo).map(|(r, _, _)| r)
}

/// `Ok(false)` solo si faltan las dos representaciones; cualquier otro error se devuelve.
pub fn hay_cinta(logica: &Path) -> io::Result<bool> {
    match representacion(logica) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

fn con_ruta(e: io::Error, ruta: &Path) -> io::Error {
    // Conserva el tipo y la causa TGB; anade la ruta fisica al mensaje.
    match fallo_de(&e) {
        Some(f) => {
            let detalle = e
                .get_ref()
                .and_then(|x| x.downcast_ref::<ErrorTgb>())
                .map(|x| x.detalle.clone())
                .unwrap_or_default();
            io::Error::new(e.kind(), ErrorTgb { fallo: f, detalle: format!("{}: {detalle}", ruta.display()) })
        }
        None => io::Error::new(e.kind(), format!("{}: {e}", ruta.display())),
    }
}

fn presupuesto_bin(len: u64, p: Presupuesto, ruta: &Path) -> io::Result<()> {
    match p.tope() {
        Some(t) if len > t => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            ErrorTgb { fallo: FalloTgb::Presupuesto, detalle: format!("{}: {len} B > presupuesto de {t} B", ruta.display()) },
        )),
        _ => Ok(()),
    }
}

fn abrir_con(
    logica: &Path,
    presupuesto: Presupuesto,
    esperada: Option<&IdentidadEsperada>,
    abrir: &dyn Fn(&Path) -> io::Result<File>,
) -> io::Result<TapeBytes> {
    let (rep, fisica, mut archivo) = elegir(logica, abrir)?;
    match rep {
        Representacion::Bin => {
            // Mismo mapeo que hacian los lectores: el `.bin` no se copia.
            let mapa = unsafe { memmap2::MmapOptions::new().map(&archivo)? };
            presupuesto_bin(mapa.len() as u64, presupuesto, &fisica)?;
            if let Some(e) = esperada {
                let sha: [u8; 32] = Sha256::digest(&mapa[..]).into();
                if e.bytes != mapa.len() as u64 || e.sha256 != sha {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        ErrorTgb { fallo: FalloTgb::Identidad, detalle: format!("{}: no es el original esperado", fisica.display()) },
                    ));
                }
            }
            Ok(TapeBytes { ruta_logica: logica.to_path_buf(), ruta_fisica: fisica, transporte: Transporte::Bin, datos: Datos::Mapa(mapa) })
        }
        Representacion::Tgb => {
            // Instantanea privada del contenedor: el motor no ve ningun byte hasta que todo se verifico.
            let mut buf = Vec::new();
            let pista = archivo.metadata().map(|m| m.len()).unwrap_or(0);
            if let Ok(p) = usize::try_from(pista) {
                buf.try_reserve_exact(p)
                    .map_err(|e| io::Error::new(io::ErrorKind::OutOfMemory, format!("{}: {e}", fisica.display())))?;
            }
            archivo.read_to_end(&mut buf).map_err(|e| con_ruta(e, &fisica))?;
            let sha256_contenedor: [u8; 32] = Sha256::digest(&buf).into();
            let r = restituir_tgb_con(&buf, presupuesto, esperada).map_err(|e| con_ruta(e, &fisica))?;
            let transporte = Transporte::Tgb { bytes_contenedor: buf.len() as u64, sha256_contenedor, modo: r.modo() };
            drop(buf);
            Ok(TapeBytes { ruta_logica: logica.to_path_buf(), ruta_fisica: fisica, transporte, datos: Datos::Restituido(r) })
        }
    }
}

/// Abre la cinta logica `path` (`.bin`, o `.bin.tgb` si el `.bin` no existe) y devuelve sus bytes verificados.
pub fn open_logical_tape(path: &Path, presupuesto: Presupuesto) -> io::Result<TapeBytes> {
    abrir_con(path, presupuesto, None, &abrir_archivo)
}

/// Igual que [`open_logical_tape`], exigiendo el tamano y el SHA-256 del original (recibo o manifiesto).
pub fn open_logical_tape_con_identidad(
    path: &Path,
    presupuesto: Presupuesto,
    esperada: &IdentidadEsperada,
) -> io::Result<TapeBytes> {
    abrir_con(path, presupuesto, Some(esperada), &abrir_archivo)
}

/// Bytes de la cinta logica en un `Vec` propio (para quien antes hacia `std::fs::read`).
pub fn leer_logica(path: &Path) -> io::Result<Vec<u8>> {
    let t = open_logical_tape(path, Presupuesto::sin_tope())?;
    let mut v = Vec::new();
    v.try_reserve_exact(t.len())
        .map_err(|e| io::Error::new(io::ErrorKind::OutOfMemory, format!("{}: {e}", path.display())))?;
    v.extend_from_slice(t.as_bytes());
    Ok(v)
}

/// Tamano VERIFICADO del original (en TGB exige restituir y verificar el contenedor entero).
pub fn largo_logico(path: &Path) -> io::Result<u64> {
    open_logical_tape(path, Presupuesto::sin_tope()).map(|t| t.len() as u64)
}

#[cfg(test)]
pub(crate) fn abrir_con_inyectado(
    logica: &Path,
    abrir: &dyn Fn(&Path) -> io::Result<File>,
) -> io::Result<TapeBytes> {
    abrir_con(logica, Presupuesto::sin_tope(), None, abrir)
}

#[cfg(test)]
mod pruebas {
    use super::*;

    #[test]
    fn permiso_denegado_no_cambia_de_fuente() {
        let dir = std::env::temp_dir().join(format!("tick_tape_permiso_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let logica = dir.join("x.bin");
        std::fs::write(ruta_tgb(&logica), include_bytes!("../tests/fixtures/f1_real.bin.tgb")).unwrap();
        let abrir = |p: &Path| -> io::Result<File> {
            if p.as_os_str().to_string_lossy().ends_with(".bin") {
                Err(io::Error::new(io::ErrorKind::PermissionDenied, "inyectado"))
            } else {
                File::open(p)
            }
        };
        let e = abrir_con_inyectado(&logica, &abrir).err().unwrap();
        assert_eq!(e.kind(), io::ErrorKind::PermissionDenied);
        // Sin la inyeccion, el `.tgb` si se usa (el .bin no existe).
        let t = abrir_con_inyectado(&logica, &abrir_archivo).unwrap();
        assert_eq!(t.representacion(), Representacion::Tgb);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
