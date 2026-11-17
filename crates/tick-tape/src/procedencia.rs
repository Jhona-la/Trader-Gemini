//! TGB-G — marca de generacion de una cinta REAL y admision para ENTRENAR (pedido CODEX-CLI, Parte 4; cierre
//! `pedido_tgb_g_v1.md` §3).
//!
//! Dos recetas historicas del semi-spread con que `binance_vision_sync` fabrico el libro de cada aggTrade de precio
//! `p` (`bid = p - h`, `ask = p + h`):
//! - relativa (D-723, `src/bin/binance_vision_sync.rs:222`): `h = p * 0.00005`;
//! - antigua (`src/bin/tape_spread_fix.rs:4`): `h = max(p * 0.00005, 0.01)`.
//!
//! Son las constantes historicas de las recetas auditadas, no umbrales de trading. Por encima del cruce
//! `0.01 / 0.00005` las dos dan los mismos bytes: ningun hash, nombre o fecha distingue quien escribio esa cinta.
//!
//! FIRMA POR INTERVALOS (metodo conservador de la Parte 4). Si un `p` f64 produjo el registro, los reales
//! `x = p - h` e `y = p + h` redondearon a `bid` y `ask`, asi que `x` esta en `[bid⁻, bid⁺]` e `y` en
//! `[ask⁻, ask⁺]` (vecinos representables: cota exterior del redondeo al mas cercano), y `p = (x + y) / 2`
//! exactamente. El intervalo `P` de todos los `p` posibles se calcula con esas cotas y se ensancha hacia fuera
//! `ENSANCHE_ULP` pasos (cubre el redondeo de las mitades y de la suma, tambien al cruzar una binade). Despues se
//! RECORRE ENTERO: cada f64 de `P` se prueba bit a bit con las dos recetas. Un testigo exacto prueba la
//! compatibilidad; si ninguno de `P` reproduce el registro, la receta queda EXCLUIDA, porque fuera de `P` no hay
//! candidatos. Si `P` tuviera mas de `MAX_CANDIDATOS` f64 (no ocurre con datos positivos y finitos; es la guarda
//! de un intervalo inesperado), el registro queda SIN DECISION y la cinta no se acredita. No hay epsilon decimal,
//! porcentaje tolerado, muestreo ni ventana fija: la ventana de 16 ulp de H se retira.
//!
//! ESTADO de la cinta: `OtraOInvalida` si algun registro es invalido (no finito, no positivo, cruzado, cantidad no
//! finita o negativa) o no admite ninguna receta (incluida la transformacion de medio tick de `tape_spread_fix`);
//! `Mixta` si hay testigos exclusivos de las dos; `Indeterminada` si algun registro quedo sin decision, si todos
//! admiten las dos o si no hay registros; si no, `Antigua` o `Relativa`. Un resultado ambiguo nunca se fuerza.
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
/// Pasos hacia fuera con que se ensancha cada extremo del intervalo de `p` (ver la cabecera).
pub const ENSANCHE_ULP: u32 = 4;
/// Tope de f64 que se recorren por registro; por encima, el registro queda sin decision (nunca acredita).
pub const MAX_CANDIDATOS: u64 = 256;
/// Cabecera REAL: los 8 bytes `TGMTICK1` (adenda (c) del coordinador; no basta el prefijo `TGM`).
pub const CABECERA_REAL: &[u8; 8] = b"TGMTICK1";
/// Formato de cinta que firma este analizador.
pub const FORMATO_CINTA: &str = "TGMTICK1+40B_LE";
/// Version del analizador que debe declarar el recibo.
pub const VERSION_ANALIZADOR: &str = "tick-tape-procedencia-2";
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

/// Veredicto de una receta sobre un registro.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Veredicto {
    /// Hay un `p` f64 que reproduce bid y ask bit a bit (testigo exacto).
    Compatible,
    /// Ningun f64 del intervalo exterior de `p` lo reproduce: queda excluida.
    Excluida,
    /// El intervalo excede `MAX_CANDIDATOS`: no se decide.
    SinDecision,
}

/// Contadores, primer testigo y primer indice de exclusion (indice de registro; offset = 8 + 40 * indice).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Firma {
    pub registros: u64,
    pub ambas: u64,
    pub solo_relativa: u64,
    pub solo_antigua: u64,
    pub ninguna: u64,
    pub sin_decision: u64,
    pub invalidos: u64,
    pub primer_solo_relativa: Option<u64>,
    pub primer_solo_antigua: Option<u64>,
    pub primer_ninguna: Option<u64>,
    pub primer_sin_decision: Option<u64>,
    pub primer_invalido: Option<u64>,
    pub primer_excluye_relativa: Option<u64>,
    pub primer_excluye_antigua: Option<u64>,
}

impl Firma {
    pub fn estado(&self) -> EstadoGeneracion {
        if self.invalidos > 0 || self.ninguna > 0 {
            EstadoGeneracion::OtraOInvalida
        } else if self.solo_relativa > 0 && self.solo_antigua > 0 {
            EstadoGeneracion::Mixta
        } else if self.sin_decision > 0 {
            EstadoGeneracion::Indeterminada
        } else if self.solo_antigua > 0 {
            EstadoGeneracion::Antigua
        } else if self.solo_relativa > 0 {
            EstadoGeneracion::Relativa
        } else {
            EstadoGeneracion::Indeterminada
        }
    }

    /// Recetas compatibles con TODOS los registros, en orden fijo (`relativa`, `antigua`), separadas por coma.
    pub fn recetas_compatibles(&self) -> String {
        let mut v = Vec::new();
        if self.registros > 0 && self.invalidos == 0 && self.ninguna == 0 && self.sin_decision == 0 {
            if self.solo_antigua == 0 {
                v.push("relativa");
            }
            if self.solo_relativa == 0 {
                v.push("antigua");
            }
        }
        if v.is_empty() {
            "ninguna".to_string()
        } else {
            v.join(",")
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

fn hacia_fuera(mut x: f64, arriba: bool) -> f64 {
    for _ in 0..ENSANCHE_ULP {
        x = if arriba { x.next_up() } else { x.next_down() };
    }
    x
}

/// Intervalo exterior `[lo, hi]` (f64 positivos y finitos) de todo `p` que pudo producir `bid`/`ask`.
pub fn intervalo_de_p(bid: f64, ask: f64) -> (f64, f64) {
    let lo = hacia_fuera(bid.next_down() * 0.5 + ask.next_down() * 0.5, false);
    let hi = hacia_fuera(bid.next_up() * 0.5 + ask.next_up() * 0.5, true);
    let minimo = f64::from_bits(1);
    let lo = if lo.is_finite() && lo > 0.0 { lo } else { minimo };
    let hi = if hi.is_finite() { hi } else { f64::MAX };
    (lo, hi)
}

/// Veredictos (relativa, antigua) de un registro valido, recorriendo entero el intervalo de `p`.
pub fn veredictos(bid: f64, ask: f64) -> (Veredicto, Veredicto) {
    let (lo, hi) = intervalo_de_p(bid, ask);
    let (a, b) = (lo.to_bits(), hi.to_bits());
    if b < a || b - a >= MAX_CANDIDATOS {
        return (Veredicto::SinDecision, Veredicto::SinDecision);
    }
    let (mut rel, mut ant) = (false, false);
    for bits in a..=b {
        let p = f64::from_bits(bits);
        rel = rel || reproduce(p, semi_relativo, bid, ask);
        ant = ant || reproduce(p, semi_antiguo, bid, ask);
        if rel && ant {
            break;
        }
    }
    let v = |c: bool| if c { Veredicto::Compatible } else { Veredicto::Excluida };
    (v(rel), v(ant))
}

fn f64_en(r: &[u8], i: usize) -> f64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&r[i..i + 8]);
    f64::from_le_bytes(a)
}

/// Campos crudos validos para firmar: precios finitos y positivos sin cruce, cantidades finitas y no negativas.
fn registro_valido(bid: f64, ask: f64, bq: f64, aq: f64) -> bool {
    bid.is_finite() && ask.is_finite() && bid > 0.0 && ask > 0.0 && bid <= ask && bq.is_finite() && aq.is_finite() && bq >= 0.0 && aq >= 0.0
}

/// Firma de los registros crudos de 40 B (sin cabecera), ANTES de cualquier saneamiento. Recorre todos.
pub fn firma_de_registros(cuerpo: &[u8]) -> Result<Firma, String> {
    if cuerpo.len() % REGISTRO != 0 {
        return Err(format!("{} B no son registros de {REGISTRO} B", cuerpo.len()));
    }
    let mut f = Firma::default();
    for (k, r) in cuerpo.chunks_exact(REGISTRO).enumerate() {
        let k = k as u64;
        f.registros += 1;
        let (bid, ask, bq, aq) = (f64_en(r, 8), f64_en(r, 16), f64_en(r, 24), f64_en(r, 32));
        if !registro_valido(bid, ask, bq, aq) {
            f.invalidos += 1;
            f.primer_invalido.get_or_insert(k);
            continue;
        }
        let (rel, ant) = veredictos(bid, ask);
        if rel == Veredicto::Excluida {
            f.primer_excluye_relativa.get_or_insert(k);
        }
        if ant == Veredicto::Excluida {
            f.primer_excluye_antigua.get_or_insert(k);
        }
        match (rel, ant) {
            (Veredicto::Compatible, Veredicto::Compatible) => f.ambas += 1,
            (Veredicto::Compatible, Veredicto::Excluida) => {
                f.solo_relativa += 1;
                f.primer_solo_relativa.get_or_insert(k);
            }
            (Veredicto::Excluida, Veredicto::Compatible) => {
                f.solo_antigua += 1;
                f.primer_solo_antigua.get_or_insert(k);
            }
            (Veredicto::Excluida, Veredicto::Excluida) => {
                f.ninguna += 1;
                f.primer_ninguna.get_or_insert(k);
            }
            _ => {
                f.sin_decision += 1;
                f.primer_sin_decision.get_or_insert(k);
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
