//! Decodificador del contenedor TGB00001, transaccional.
//!
//! Orden (pedido CODEX-CLI, Parte 2, pasos 2 a 7): magia, minimo y sello ANTES de descomprimir; recorrido completo
//! de la estructura con aritmetica comprobada y coherencia de largos ANTES de reservar; presupuesto del
//! consumidor; descompresion acotada al largo esperado con prueba de EOF; tiempo con zigzag inverso y suma
//! modular, reiniciado en cada bloque; SHA-256 del original entero e identidad esperada. Solo entonces se publica
//! el original, en memoria anonima de solo lectura. Cualquier fallo descarta la carga entera: quien llama no ve
//! ningun byte de un contenedor que no paso todas las comprobaciones.
//!
//! Cada comprobacion tiene su propio `FalloTgb`, para que las pruebas de corrupcion distingan cual la detecto.

use sha2::{Digest, Sha256};
use std::fmt;
use std::io::{self, Read};
use std::ops::Range;

/// Magia del contenedor.
pub const MAGIA_TGB: &[u8; 8] = b"TGB00001";
/// Registro de la cinta: u64 tiempo + 4 f64 (formato, `tgb/src/main.rs:14`).
pub const REGISTRO: usize = 40;
/// Campos de 8 bytes por registro, en orden tiempo, bid, ask, bid_qty, ask_qty.
pub const CAMPOS: usize = 5;
/// Registros maximos por bloque (formato, `tgb/src/main.rs:15`).
pub const BLOQUE_MAX: u64 = 1 << 22;
/// Ventana maxima de Zstandard del formato (`tgb/src/main.rs:49`).
pub const VENTANA_LOG_MAX: u32 = 27;
/// Cabecera original que guarda el modo 1.
pub const CABECERA_ORIGINAL: usize = 8;
const SELLO: usize = 32;
const PREFIJO: usize = 8 + 8 + 32 + 1;
/// Todo modo lleva al menos un u64 tras el byte de modo.
pub const MINIMO_CONTENEDOR: usize = PREFIJO + 8 + SELLO;

/// Que comprobacion rechazo el contenedor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalloTgb {
    /// No empieza por `TGB00001`.
    Magia,
    /// Mas corto que el minimo del formato.
    Cortado,
    /// El SHA-256 del cuerpo no es el sello final.
    Sello,
    /// Modo distinto de 0, 1 y 2.
    Modo,
    /// Un campo, una longitud o un desplazamiento se sale del cuerpo.
    FueraDelCuerpo,
    /// Faltan bloques: la suma de registros no llega al total.
    FaltanBloques,
    /// Bloque con cero registros.
    BloqueCero,
    /// Bloque con mas registros que el maximo del formato.
    BloqueMayorQueMax,
    /// Bloque con mas registros de los que quedan en el total.
    BloqueExcedeTotal,
    /// Columna comprimida de largo cero.
    ColumnaVacia,
    /// Quedan bytes entre la ultima estructura y el sello.
    Sobrante,
    /// El largo del original no cuadra con la cabecera y los registros.
    LargoOriginal,
    /// El original no cabe en el presupuesto del consumidor.
    Presupuesto,
    /// No se pudo reservar la memoria del original.
    Reserva,
    /// Frame Zstandard invalido.
    Zstd,
    /// Una columna (o el frame del modo 0) expande a otro largo.
    LargoColumna,
    /// El SHA-256 del original restituido no es el anotado.
    ShaOriginal,
    /// Tamano o SHA del original distintos de la identidad esperada.
    Identidad,
}

/// Error de TGB con su causa; viaja dentro de un `io::Error` de tipo `InvalidData`.
#[derive(Debug)]
pub struct ErrorTgb {
    pub fallo: FalloTgb,
    pub detalle: String,
}

impl fmt::Display for ErrorTgb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TGB00001 rechazado ({:?}): {}", self.fallo, self.detalle)
    }
}

impl std::error::Error for ErrorTgb {}

fn fallo(fallo: FalloTgb, detalle: impl Into<String>) -> io::Error {
    let tipo = match fallo {
        FalloTgb::Reserva => io::ErrorKind::OutOfMemory,
        _ => io::ErrorKind::InvalidData,
    };
    io::Error::new(tipo, ErrorTgb { fallo, detalle: detalle.into() })
}

/// La causa TGB de un error, si la tiene.
pub fn fallo_de(e: &io::Error) -> Option<FalloTgb> {
    e.get_ref().and_then(|x| x.downcast_ref::<ErrorTgb>()).map(|x| x.fallo)
}

/// Tope de lectura que fija quien consume la cinta (bytes del original restituido).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Presupuesto {
    max_bytes_original: Option<u64>,
}

impl Presupuesto {
    /// Sin tope propio: el consumidor va a retener la cinta entera de todos modos.
    pub fn sin_tope() -> Self {
        Presupuesto { max_bytes_original: None }
    }
    /// Tope derivado de la tarea del consumidor.
    pub fn hasta(bytes: u64) -> Self {
        Presupuesto { max_bytes_original: Some(bytes) }
    }
    pub fn tope(&self) -> Option<u64> {
        self.max_bytes_original
    }
}

/// Identidad del original que se espera (de un recibo o manifiesto).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentidadEsperada {
    pub bytes: u64,
    pub sha256: [u8; 32],
}

/// Memoria del original: anonima, alineada a pagina y de solo lectura una vez verificada.
pub(crate) enum Respaldo {
    Vacio,
    Mapa(memmap2::Mmap),
}

impl Respaldo {
    pub(crate) fn bytes(&self) -> &[u8] {
        match self {
            Respaldo::Vacio => &[],
            Respaldo::Mapa(m) => &m[..],
        }
    }
}

/// Original restituido y verificado.
pub struct Restituido {
    respaldo: Respaldo,
    sha256_original: [u8; 32],
    modo: u8,
}

impl Restituido {
    pub fn as_bytes(&self) -> &[u8] {
        self.respaldo.bytes()
    }
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }
    pub fn is_empty(&self) -> bool {
        self.as_bytes().is_empty()
    }
    pub fn sha256_original(&self) -> [u8; 32] {
        self.sha256_original
    }
    pub fn modo(&self) -> u8 {
        self.modo
    }
}

/// Cursor acotado al cuerpo (sin el sello): toda lectura comprueba que cabe.
struct Cursor<'a> {
    cuerpo: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn u64(&mut self, que: &str) -> io::Result<u64> {
        let r = self.rango(8, que)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(&self.cuerpo[r]);
        Ok(u64::from_le_bytes(a))
    }
    fn rango(&mut self, largo: usize, que: &str) -> io::Result<Range<usize>> {
        let fin = self
            .pos
            .checked_add(largo)
            .filter(|f| *f <= self.cuerpo.len())
            .ok_or_else(|| fallo(FalloTgb::FueraDelCuerpo, format!("{que}: {largo} B desde {} se salen del cuerpo", self.pos)))?;
        let r = self.pos..fin;
        self.pos = fin;
        Ok(r)
    }
    fn trozo(&mut self, largo: u64, que: &str) -> io::Result<Range<usize>> {
        let largo = usize::try_from(largo)
            .map_err(|_| fallo(FalloTgb::FueraDelCuerpo, format!("{que}: largo {largo} no cabe en usize")))?;
        self.rango(largo, que)
    }
}

struct PlanBloque {
    n: u64,
    columnas: [Range<usize>; CAMPOS],
}

enum Plan {
    Entero(Range<usize>),
    Registros { cabecera: Option<Range<usize>>, bloques: Vec<PlanBloque> },
}

/// Recorre la estructura entera sin descomprimir nada.
fn planear(cuerpo: &[u8], modo: u8, largo: u64) -> io::Result<Plan> {
    let mut c = Cursor { cuerpo, pos: PREFIJO };
    let plan = match modo {
        0 => {
            let zl = c.u64("largo comprimido del modo 0")?;
            if zl == 0 {
                return Err(fallo(FalloTgb::ColumnaVacia, "frame del modo 0 vacio"));
            }
            Plan::Entero(c.trozo(zl, "frame del modo 0")?)
        }
        1 | 2 => {
            let cabecera = if modo == 1 { Some(c.rango(CABECERA_ORIGINAL, "cabecera original")?) } else { None };
            let total = c.u64("total de registros")?;
            let mut bloques = Vec::new();
            let mut hechos = 0u64;
            while hechos < total {
                if c.pos == cuerpo.len() {
                    return Err(fallo(FalloTgb::FaltanBloques, format!("{hechos} de {total} registros")));
                }
                let n = c.u64("registros del bloque")?;
                if n == 0 {
                    return Err(fallo(FalloTgb::BloqueCero, format!("bloque en {}", c.pos - 8)));
                }
                if n > BLOQUE_MAX {
                    return Err(fallo(FalloTgb::BloqueMayorQueMax, format!("{n} > {BLOQUE_MAX}")));
                }
                if n > total - hechos {
                    return Err(fallo(FalloTgb::BloqueExcedeTotal, format!("{n} > {} restantes", total - hechos)));
                }
                let mut columnas: [Range<usize>; CAMPOS] = std::array::from_fn(|_| 0..0);
                for (f, col) in columnas.iter_mut().enumerate() {
                    let zl = c.u64("largo de columna")?;
                    if zl == 0 {
                        return Err(fallo(FalloTgb::ColumnaVacia, format!("campo {f}")));
                    }
                    *col = c.trozo(zl, "columna")?;
                }
                bloques.push(PlanBloque { n, columnas });
                hechos += n;
            }
            let regs = total
                .checked_mul(REGISTRO as u64)
                .and_then(|b| b.checked_add(if modo == 1 { CABECERA_ORIGINAL as u64 } else { 0 }));
            if regs != Some(largo) {
                return Err(fallo(FalloTgb::LargoOriginal, format!("anotado {largo}, estructura {regs:?}")));
            }
            Plan::Registros { cabecera, bloques }
        }
        otro => return Err(fallo(FalloTgb::Modo, format!("modo {otro}"))),
    };
    if c.pos != cuerpo.len() {
        return Err(fallo(FalloTgb::Sobrante, format!("{} B antes del sello", cuerpo.len() - c.pos)));
    }
    Ok(plan)
}

/// Descomprime `z` exactamente en `destino`: ni un byte de menos ni uno de mas.
fn descomprimir_exacto(z: &[u8], destino: &mut [u8]) -> io::Result<()> {
    let zerr = |e: io::Error| fallo(FalloTgb::Zstd, e.to_string());
    let mut d = zstd::stream::read::Decoder::with_buffer(z).map_err(zerr)?;
    d.window_log_max(VENTANA_LOG_MAX).map_err(zerr)?;
    let mut hecho = 0usize;
    while hecho < destino.len() {
        match d.read(&mut destino[hecho..]) {
            Ok(0) => return Err(fallo(FalloTgb::LargoColumna, format!("expande a {hecho} de {}", destino.len()))),
            Ok(k) => hecho += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(zerr(e)),
        }
    }
    let mut sonda = [0u8; 1];
    loop {
        match d.read(&mut sonda) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(fallo(FalloTgb::LargoColumna, format!("expande a mas de {}", destino.len()))),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(zerr(e)),
        }
    }
}

fn zigzag_inverso(t: u64) -> u64 {
    (t >> 1) ^ 0u64.wrapping_sub(t & 1)
}

fn u64_le(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(b);
    u64::from_le_bytes(a)
}

/// Un bloque: descomprime sus 5 columnas (en hilos) y las entrelaza en `salida` (n registros de 40 B).
fn restituir_bloque(cuerpo: &[u8], b: &PlanBloque, salida: &mut [u8]) -> io::Result<()> {
    let n = usize::try_from(b.n).map_err(|_| fallo(FalloTgb::Reserva, "bloque"))?;
    let largo_col = n
        .checked_mul(8)
        .ok_or_else(|| fallo(FalloTgb::Reserva, "columna"))?;
    let resultados: Vec<io::Result<Vec<u8>>> = std::thread::scope(|s| {
        let hilos: Vec<_> = b
            .columnas
            .iter()
            .map(|r| {
                let z = &cuerpo[r.clone()];
                s.spawn(move || -> io::Result<Vec<u8>> {
                    let mut v = Vec::new();
                    v.try_reserve_exact(largo_col)
                        .map_err(|e| fallo(FalloTgb::Reserva, e.to_string()))?;
                    v.resize(largo_col, 0);
                    descomprimir_exacto(z, &mut v)?;
                    Ok(v)
                })
            })
            .collect();
        hilos
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err(fallo(FalloTgb::Zstd, "hilo de columna caido"))))
            .collect()
    });
    let mut cols = Vec::with_capacity(CAMPOS);
    for r in resultados {
        cols.push(r?);
    }
    let mut previo = 0u64;
    for (k, reg) in salida.chunks_exact_mut(REGISTRO).enumerate() {
        for (f, col) in cols.iter().enumerate() {
            let a = k * 8;
            let t = u64_le(&col[a..a + 8]);
            let v = if f == 0 {
                previo = previo.wrapping_add(zigzag_inverso(t));
                previo
            } else {
                t
            };
            reg[f * 8..f * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
    }
    Ok(())
}

/// Restituye y verifica el original de un contenedor TGB00001 entero.
pub fn restituir_tgb(contenedor: &[u8], presupuesto: Presupuesto) -> io::Result<Restituido> {
    restituir_tgb_con(contenedor, presupuesto, None)
}

/// Igual que [`restituir_tgb`], exigiendo ademas el tamano y el SHA-256 del original esperado.
pub fn restituir_tgb_con(
    contenedor: &[u8],
    presupuesto: Presupuesto,
    esperada: Option<&IdentidadEsperada>,
) -> io::Result<Restituido> {
    // 2. Minimo, magia y sello antes de descomprimir.
    if contenedor.len() >= MAGIA_TGB.len() && &contenedor[..MAGIA_TGB.len()] != MAGIA_TGB {
        return Err(fallo(FalloTgb::Magia, "no empieza por TGB00001"));
    }
    if contenedor.len() < MINIMO_CONTENEDOR {
        return Err(fallo(FalloTgb::Cortado, format!("{} B < {MINIMO_CONTENEDOR}", contenedor.len())));
    }
    let (cuerpo, sello) = contenedor.split_at(contenedor.len() - SELLO);
    let h: [u8; 32] = Sha256::digest(cuerpo).into();
    if h[..] != sello[..] {
        return Err(fallo(FalloTgb::Sello, "el SHA-256 del contenedor no es su sello"));
    }
    let largo = u64_le(&cuerpo[8..16]);
    let mut sha_anotado = [0u8; 32];
    sha_anotado.copy_from_slice(&cuerpo[16..48]);
    let modo = cuerpo[48];

    // 3. Estructura completa y coherencia de largos.
    let plan = planear(cuerpo, modo, largo)?;

    // 4. Presupuesto del consumidor y reserva.
    if let Some(tope) = presupuesto.tope() {
        if largo > tope {
            return Err(fallo(FalloTgb::Presupuesto, format!("original de {largo} B > presupuesto de {tope} B")));
        }
    }
    if let Some(e) = esperada {
        if e.bytes != largo {
            return Err(fallo(FalloTgb::Identidad, format!("anotado {largo} B, esperado {} B", e.bytes)));
        }
    }
    let largo_us = usize::try_from(largo).map_err(|_| fallo(FalloTgb::Reserva, format!("{largo} B no caben en usize")))?;
    let mut salida = if largo_us == 0 {
        None
    } else {
        Some(memmap2::MmapMut::map_anon(largo_us).map_err(|e| fallo(FalloTgb::Reserva, format!("{largo} B: {e}")))?)
    };

    // 5. Restitucion.
    if let Some(sal) = salida.as_mut() {
        match &plan {
            Plan::Entero(r) => descomprimir_exacto(&cuerpo[r.clone()], &mut sal[..])?,
            Plan::Registros { cabecera, bloques } => {
                let mut pos = 0usize;
                if let Some(r) = cabecera {
                    sal[..CABECERA_ORIGINAL].copy_from_slice(&cuerpo[r.clone()]);
                    pos = CABECERA_ORIGINAL;
                }
                for b in bloques {
                    // n <= BLOQUE_MAX y la suma de n * 40 + cabecera == largo (comprobado en planear).
                    let bytes = (b.n as usize) * REGISTRO;
                    restituir_bloque(cuerpo, b, &mut sal[pos..pos + bytes])?;
                    pos += bytes;
                }
            }
        }
    } else if let Plan::Entero(r) = &plan {
        // Original vacio: el frame del modo 0 tiene que expandir a cero bytes.
        descomprimir_exacto(&cuerpo[r.clone()], &mut [])?;
    }

    // 6. SHA-256 del original entero, cabecera incluida, e identidad esperada.
    let restituido: &[u8] = salida.as_ref().map(|m| &m[..]).unwrap_or(&[]);
    let sha: [u8; 32] = Sha256::digest(restituido).into();
    if sha != sha_anotado {
        return Err(fallo(FalloTgb::ShaOriginal, "el original restituido no tiene el SHA-256 anotado"));
    }
    if let Some(e) = esperada {
        if e.sha256 != sha {
            return Err(fallo(FalloTgb::Identidad, "el SHA-256 del original no es el esperado"));
        }
    }

    // 7. Publicacion: solo lectura.
    let respaldo = match salida {
        None => Respaldo::Vacio,
        Some(m) => Respaldo::Mapa(m.make_read_only().map_err(|e| fallo(FalloTgb::Reserva, e.to_string()))?),
    };
    Ok(Restituido { respaldo, sha256_original: sha, modo })
}
