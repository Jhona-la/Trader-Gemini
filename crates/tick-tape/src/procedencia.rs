//! TGB-G — marca de generacion de una cinta REAL y admision para ENTRENAR (pedido CODEX-CLI, Parte 4).
//!
//! Dos recetas historicas del semi-spread con que `binance_vision_sync` fabrico el libro de cada aggTrade de precio
//! `p` (`bid = p - h`, `ask = p + h`):
//! - relativa (D-723, `src/bin/binance_vision_sync.rs:222`): `h = p * 0.00005`;
//! - antigua (`src/bin/tape_spread_fix.rs:4`): `h = max(p * 0.00005, 0.01)`.
//!
//! Son las constantes historicas de las recetas auditadas, no umbrales de trading. Por encima del cruce
//! `0.01 / 0.00005` las dos dan los mismos bytes: ningun hash, nombre o fecha distingue quien escribio esa cinta.
//!
//! FIRMA. Para cada registro se busca un TESTIGO EXACTO: un `p` f64 que reproduzca bit a bit el `bid` y el `ask`
//! con la receta. Si existe, el registro es compatible con esa receta (la compatibilidad queda probada por el
//! testigo). Si no existe ninguno en la ventana, es incompatible. La ventana sale de la cota del redondeo: el
//! `p` real dista del punto medio calculado `m = bid/2 + ask/2` a lo sumo `|bid - fl(p-h)|/2 + |ask - fl(p+h)|/2`
//! mas el redondeo de `m`, es decir menos de 2 ulp de `p` (bid, p y ask estan a menos de un factor 2 entre si
//! con `h <= p/2`). Se recorren `VENTANA_ULP` = 16 ulp a cada lado, ocho veces esa cota. No hay epsilon
//! decimal ni porcentaje de registros que coincidan. Nota (TGB-G): cuando bid, p y ask comparten binade el error
//! del medio es <= 3/8 ulp y redondea a `p`, asi que el testigo suele ser el propio medio; la ventana es una
//! defensa para los cruces de binade. No se encontro un caso de estas dos recetas en que haga falta (su control
//! de mutacion, ventana 0, sale ciego).
//!
//! ESTADO de la cinta (interseccion de todos los registros): `Relativa` si algun registro excluye la antigua y
//! ninguno excluye la relativa; `Antigua` al reves; `Mixta` si hay de los dos; `OtraOInvalida` si algun registro
//! no admite ninguna (incluidos no finitos, no positivos o cruzados, y la transformacion de medio tick de
//! `tape_spread_fix`); `Indeterminada` si todos admiten las dos o no hay registros.
//!
//! La firma no acredita QUIEN genero la cinta. Para entrenar se exige ademas un RECIBO lateral
//! (`X.bin.procedencia`) ligado al SHA-256 y al tamano exactos del original, con procedencia relativa acreditada
//! por el productor y cobertura completa. Se rechazan antigua, mixta, otra, indeterminada, recibo ausente, recibo
//! de otro SHA y analisis incompleto. Nada de esto reescribe la cinta ni corrige su spread al cargarla.

use crate::logica::TapeBytes;
use std::path::{Path, PathBuf};

/// Semi-spread relativo de la receta nueva (`binance_vision_sync.rs:222`).
pub const R_RELATIVA: f64 = 0.00005;
/// Suelo absoluto de la receta antigua (`tape_spread_fix.rs:4`).
pub const SUELO_ANTIGUO: f64 = 0.01;
/// Ulp recorridos a cada lado del punto medio (ver la cota en la cabecera del modulo).
pub const VENTANA_ULP: u32 = 16;
/// Cabecera REAL: los 8 bytes `TGMTICK1` (adenda (c) del coordinador; no basta el prefijo `TGM`).
pub const CABECERA_REAL: &[u8; 8] = b"TGMTICK1";
const REGISTRO: usize = 40;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstadoGeneracion {
    Antigua,
    Relativa,
    Indeterminada,
    Mixta,
    OtraOInvalida,
}

impl EstadoGeneracion {
    pub fn nombre(self) -> &'static str {
        match self {
            EstadoGeneracion::Antigua => "antigua",
            EstadoGeneracion::Relativa => "relativa",
            EstadoGeneracion::Indeterminada => "indeterminada",
            EstadoGeneracion::Mixta => "mixta",
            EstadoGeneracion::OtraOInvalida => "otra_o_invalida",
        }
    }
}

/// Contadores y primer testigo (indice de registro) de cada clase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Firma {
    pub registros: u64,
    pub ambas: u64,
    pub solo_relativa: u64,
    pub solo_antigua: u64,
    pub ninguna: u64,
    pub primer_solo_relativa: Option<u64>,
    pub primer_solo_antigua: Option<u64>,
    pub primer_ninguna: Option<u64>,
}

impl Firma {
    pub fn estado(&self) -> EstadoGeneracion {
        if self.ninguna > 0 {
            EstadoGeneracion::OtraOInvalida
        } else if self.solo_relativa > 0 && self.solo_antigua > 0 {
            EstadoGeneracion::Mixta
        } else if self.solo_antigua > 0 {
            EstadoGeneracion::Antigua
        } else if self.solo_relativa > 0 {
            EstadoGeneracion::Relativa
        } else {
            EstadoGeneracion::Indeterminada
        }
    }
}

fn semi_relativo(p: f64) -> f64 {
    p * R_RELATIVA
}

fn semi_antiguo(p: f64) -> f64 {
    (p * R_RELATIVA).max(SUELO_ANTIGUO)
}

fn reproduce(p: f64, semi: fn(f64) -> f64, bid: f64, ask: f64) -> bool {
    let h = semi(p);
    (p - h).to_bits() == bid.to_bits() && (p + h).to_bits() == ask.to_bits()
}

/// Siguiente f64 hacia arriba o hacia abajo de un positivo finito (por bits).
fn vecino(p: f64, arriba: bool) -> Option<f64> {
    let b = p.to_bits();
    let n = if arriba { b.checked_add(1)? } else { b.checked_sub(1)? };
    let q = f64::from_bits(n);
    (q.is_finite() && q > 0.0).then_some(q)
}

/// ¿Existe un `p` en la ventana que reproduzca `bid`/`ask` con la receta?
fn compatible(semi: fn(f64) -> f64, bid: f64, ask: f64) -> bool {
    let m = bid * 0.5 + ask * 0.5;
    if !(m.is_finite() && m > 0.0) {
        return false;
    }
    if reproduce(m, semi, bid, ask) {
        return true;
    }
    for arriba in [true, false] {
        let mut p = m;
        for _ in 0..VENTANA_ULP {
            match vecino(p, arriba) {
                Some(q) => p = q,
                None => break,
            }
            if reproduce(p, semi, bid, ask) {
                return true;
            }
        }
    }
    false
}

fn f64_en(r: &[u8], i: usize) -> f64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&r[i..i + 8]);
    f64::from_le_bytes(a)
}

/// Firma de los registros crudos de 40 B (sin cabecera), ANTES de cualquier saneamiento.
pub fn firma_de_registros(cuerpo: &[u8]) -> Result<Firma, String> {
    if cuerpo.len() % REGISTRO != 0 {
        return Err(format!("{} B no son registros de {REGISTRO} B", cuerpo.len()));
    }
    let mut f = Firma::default();
    for (k, r) in cuerpo.chunks_exact(REGISTRO).enumerate() {
        let k = k as u64;
        let (bid, ask) = (f64_en(r, 8), f64_en(r, 16));
        let valido = bid.is_finite() && ask.is_finite() && bid > 0.0 && ask > 0.0 && bid <= ask;
        let (rel, ant) = if valido {
            (compatible(semi_relativo, bid, ask), compatible(semi_antiguo, bid, ask))
        } else {
            (false, false)
        };
        f.registros += 1;
        match (rel, ant) {
            (true, true) => f.ambas += 1,
            (true, false) => {
                f.solo_relativa += 1;
                f.primer_solo_relativa.get_or_insert(k);
            }
            (false, true) => {
                f.solo_antigua += 1;
                f.primer_solo_antigua.get_or_insert(k);
            }
            (false, false) => {
                f.ninguna += 1;
                f.primer_ninguna.get_or_insert(k);
            }
        }
    }
    Ok(f)
}

/// Firma de una cinta REAL entera (cabecera `TGMTICK1` de 8 bytes exactos + registros).
pub fn firma_de_cinta(bytes: &[u8]) -> Result<Firma, String> {
    match bytes.get(..8) {
        Some(c) if c == CABECERA_REAL => firma_de_registros(&bytes[8..]),
        _ => Err("no es una cinta REAL: la cabecera no es TGMTICK1".to_string()),
    }
}

/// `X.bin` -> `X.bin.procedencia` (recibo lateral; mismo criterio de nombre que `ruta_tgb`).
pub fn ruta_recibo(logica: &Path) -> PathBuf {
    let mut s = logica.as_os_str().to_os_string();
    s.push(".procedencia");
    PathBuf::from(s)
}

/// Recibo lateral de procedencia, ligado a un original exacto.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recibo {
    pub sha256_original: [u8; 32],
    pub bytes_original: u64,
    pub procedencia_acreditada: String,
    pub evidencia_productor: String,
    pub cobertura: String,
}

fn hex32(s: &str) -> Result<[u8; 32], String> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return Err(format!("sha256_original de {} caracteres", b.len()));
    }
    let mut out = [0u8; 32];
    for (k, par) in b.chunks_exact(2).enumerate() {
        let txt = std::str::from_utf8(par).map_err(|e| e.to_string())?;
        out[k] = u8::from_str_radix(txt, 16).map_err(|e| format!("sha256_original: {e}"))?;
    }
    Ok(out)
}

/// Lee un recibo `formato=tg-procedencia-1` de lineas `clave=valor` (sin claves repetidas ni desconocidas).
pub fn leer_recibo(texto: &str) -> Result<Recibo, String> {
    let mut campos: Vec<(&str, &str)> = Vec::new();
    for linea in texto.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let (k, v) = linea.split_once('=').ok_or_else(|| format!("linea sin '=': {linea}"))?;
        let (k, v) = (k.trim(), v.trim());
        if campos.iter().any(|(c, _)| *c == k) {
            return Err(format!("clave repetida: {k}"));
        }
        campos.push((k, v));
    }
    let tomar = |k: &str| campos.iter().find(|(c, _)| *c == k).map(|(_, v)| *v).ok_or_else(|| format!("falta {k}"));
    const CLAVES: [&str; 6] = ["formato", "sha256_original", "bytes_original", "procedencia_acreditada", "evidencia_productor", "cobertura"];
    if let Some((k, _)) = campos.iter().find(|(c, _)| !CLAVES.contains(c)) {
        return Err(format!("clave desconocida: {k}"));
    }
    if tomar("formato")? != "tg-procedencia-1" {
        return Err("formato de recibo desconocido".to_string());
    }
    Ok(Recibo {
        sha256_original: hex32(tomar("sha256_original")?)?,
        bytes_original: tomar("bytes_original")?.parse().map_err(|e| format!("bytes_original: {e}"))?,
        procedencia_acreditada: tomar("procedencia_acreditada")?.to_string(),
        evidencia_productor: tomar("evidencia_productor")?.to_string(),
        cobertura: tomar("cobertura")?.to_string(),
    })
}

/// Cinta admitida para preparar muestras de entrenamiento.
#[derive(Debug, Clone)]
pub struct Admitida {
    pub firma: Firma,
    pub recibo: Recibo,
}

/// Admision para ENTRENAR: firma recalculada sobre la cinta entera y recibo con identidad vigente y procedencia
/// relativa acreditada. Cada rechazo dice su causa.
pub fn admitir_para_entrenamiento(tape: &TapeBytes, recibo: Option<&str>) -> Result<Admitida, String> {
    let ruta = tape.ruta_logica().display().to_string();
    let firma = firma_de_cinta(tape.as_bytes()).map_err(|e| format!("{ruta}: {e}"))?;
    let estado = firma.estado();
    if estado != EstadoGeneracion::Relativa {
        return Err(format!(
            "{ruta}: firma de generacion {} (solo relativa {}, solo antigua {}, ninguna {}, ambas {}; testigos rel {:?} ant {:?} ninguna {:?}): no entra a entrenamiento",
            estado.nombre(),
            firma.solo_relativa,
            firma.solo_antigua,
            firma.ninguna,
            firma.ambas,
            firma.primer_solo_relativa,
            firma.primer_solo_antigua,
            firma.primer_ninguna
        ));
    }
    let texto = recibo.ok_or_else(|| format!("{ruta}: marca de procedencia ausente ({})", ruta_recibo(tape.ruta_logica()).display()))?;
    let r = leer_recibo(texto).map_err(|e| format!("{ruta}: recibo invalido: {e}"))?;
    if r.sha256_original != tape.sha256_original() || r.bytes_original != tape.len() as u64 {
        return Err(format!("{ruta}: el recibo es de otro original (SHA-256 o tamano distintos)"));
    }
    if r.procedencia_acreditada != "relativa" || r.evidencia_productor.is_empty() {
        return Err(format!("{ruta}: procedencia relativa no acreditada por el productor"));
    }
    if r.cobertura != "completa" {
        return Err(format!("{ruta}: analisis de procedencia incompleto (cobertura {})", r.cobertura));
    }
    Ok(Admitida { firma, recibo: r })
}
