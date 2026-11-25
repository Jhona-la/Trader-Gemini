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
//! La firma no acredita QUIEN genero la cinta. Para entrenar, [`admitir_original`] exige ademas un RECIBO
//! (`X.bin.procedencia`, formato `tg-procedencia-2`) ligado al SHA-256 y al tamano del original, con cuentas, firma y
//! testigo iguales a los recalculados, que apunta por SHA-256 a una EVIDENCIA DEL PRODUCTOR
//! (`tg-evidencia-productor-1`) reconocida en un INDICE REVISADO (`tg-indice-procedencia-1`) cuyo SHA-256 fija quien
//! llama (registro revisado por coordinador/integrador, nunca el propio recibo). Un texto libre, una ruta existente o
//! el hash del propio recibo no sustituyen esa cadena. Nada de esto reescribe la cinta ni corrige su spread.
//!
//! El nucleo es PURO: recibe bytes y un resolutor de evidencia por hash; no abre archivos ni lee el reloj.

use crate::logica::TapeBytes;
use sha2::{Digest, Sha256};
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
    if !cuerpo.len().is_multiple_of(REGISTRO) {
        return Err(format!("{} B no son registros de {REGISTRO} B", cuerpo.len()));
    }
    let mut f = Firma::default();
    for (k, r) in cuerpo.as_chunks::<REGISTRO>().0.iter().enumerate() {
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

// ───────────────────────────── documentos estrictos clave=valor ─────────────────────────────

/// Pares `clave=valor`, una por linea `\n`, sin espacios alrededor, sin `\r`, sin comentarios ni lineas vacias
/// intermedias. Solo `repetibles` puede aparecer mas de una vez; toda clave debe estar en `claves`.
fn campos<'a>(texto: &'a [u8], claves: &[&str], repetibles: &[&str]) -> Result<Vec<(&'a str, &'a str)>, String> {
    let texto = std::str::from_utf8(texto).map_err(|e| format!("no es UTF-8: {e}"))?;
    let cuerpo = texto.strip_suffix('\n').unwrap_or(texto);
    if cuerpo.is_empty() {
        return Err("documento vacio".to_string());
    }
    let mut out: Vec<(&str, &str)> = Vec::new();
    for linea in cuerpo.split('\n') {
        if linea.contains('\r') {
            return Err("retorno de carro en una linea".to_string());
        }
        let (k, v) = linea.split_once('=').ok_or_else(|| format!("linea sin '=': {linea}"))?;
        if k.is_empty() || k.trim() != k || v.trim() != v || v.is_empty() {
            return Err(format!("linea mal formada: {linea}"));
        }
        if !claves.contains(&k) {
            return Err(format!("clave desconocida: {k}"));
        }
        if !repetibles.contains(&k) && out.iter().any(|(c, _)| *c == k) {
            return Err(format!("clave repetida: {k}"));
        }
        out.push((k, v));
    }
    for k in claves {
        if !out.iter().any(|(c, _)| c == k) {
            return Err(format!("falta {k}"));
        }
    }
    Ok(out)
}

fn uno<'a>(c: &[(&'a str, &'a str)], k: &str) -> &'a str {
    c.iter().find(|(x, _)| *x == k).map(|(_, v)| *v).unwrap_or("")
}

/// Hex en minusculas de longitud exacta (sin signo, sin prefijo, sin mayusculas).
fn hex_exacto<const N: usize>(s: &str, campo: &str) -> Result<[u8; N], String> {
    let b = s.as_bytes();
    if b.len() != 2 * N || !b.iter().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(format!("{campo}: se esperan {} digitos hex en minusculas", 2 * N));
    }
    let nib = |c: u8| if c <= b'9' { c - b'0' } else { c - b'a' + 10 };
    let mut out = [0u8; N];
    for (k, par) in b.as_chunks::<2>().0.iter().enumerate() {
        out[k] = nib(par[0]) << 4 | nib(par[1]);
    }
    Ok(out)
}

/// Decimal sin signo, sin ceros a la izquierda (salvo "0").
fn decimal(s: &str, campo: &str) -> Result<u64, String> {
    let b = s.as_bytes();
    if b.is_empty() || !b.iter().all(u8::is_ascii_digit) || (b.len() > 1 && b[0] == b'0') {
        return Err(format!("{campo}: decimal mal formado"));
    }
    s.parse().map_err(|e| format!("{campo}: {e}"))
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Recibo de procedencia `tg-procedencia-2`, ligado a un original exacto.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recibo {
    pub ruta_logica: String,
    pub sha256_original: [u8; 32],
    pub bytes_original: u64,
    pub registros_esperados: u64,
    pub registros_analizados: u64,
    pub cobertura: String,
    pub recetas_compatibles: String,
    pub estado_firma: String,
    pub primer_testigo_relativa: u64,
    pub procedencia_acreditada: String,
    pub evidencia_productor_sha256: [u8; 32],
}

const CLAVES_RECIBO: [&str; 15] = [
    "formato",
    "version_analizador",
    "ruta_logica",
    "formato_cinta",
    "sha256_original",
    "bytes_original",
    "registros_esperados",
    "registros_analizados",
    "cobertura",
    "recetas_compatibles",
    "estado_firma",
    "primer_testigo_relativa",
    "procedencia_acreditada",
    "evidencia_productor_sha256",
    "analisis",
];

/// Lee un recibo `formato=tg-procedencia-2` con el parser estricto.
pub fn leer_recibo(texto: &[u8]) -> Result<Recibo, String> {
    let c = campos(texto, &CLAVES_RECIBO, &[])?;
    if uno(&c, "formato") != "tg-procedencia-2" {
        return Err("formato de recibo desconocido".to_string());
    }
    if uno(&c, "version_analizador") != VERSION_ANALIZADOR {
        return Err(format!("version de analizador no acreditable: {}", uno(&c, "version_analizador")));
    }
    if uno(&c, "formato_cinta") != FORMATO_CINTA {
        return Err(format!("formato de cinta no admitido: {}", uno(&c, "formato_cinta")));
    }
    if uno(&c, "analisis") != "intervalos_exhaustivos" {
        return Err(format!("metodo de analisis no admitido: {}", uno(&c, "analisis")));
    }
    Ok(Recibo {
        ruta_logica: uno(&c, "ruta_logica").to_string(),
        sha256_original: hex_exacto(uno(&c, "sha256_original"), "sha256_original")?,
        bytes_original: decimal(uno(&c, "bytes_original"), "bytes_original")?,
        registros_esperados: decimal(uno(&c, "registros_esperados"), "registros_esperados")?,
        registros_analizados: decimal(uno(&c, "registros_analizados"), "registros_analizados")?,
        cobertura: uno(&c, "cobertura").to_string(),
        recetas_compatibles: uno(&c, "recetas_compatibles").to_string(),
        estado_firma: uno(&c, "estado_firma").to_string(),
        primer_testigo_relativa: decimal(uno(&c, "primer_testigo_relativa"), "primer_testigo_relativa")?,
        procedencia_acreditada: uno(&c, "procedencia_acreditada").to_string(),
        evidencia_productor_sha256: hex_exacto(uno(&c, "evidencia_productor_sha256"), "evidencia_productor_sha256")?,
    })
}

/// Evidencia del productor `tg-evidencia-productor-1`: quien genero el original, con que fuentes, ejecutable,
/// receta, entradas, comando y salida.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenciaProductor {
    pub productor: String,
    pub commit_productor: [u8; 20],
    pub sha256_fuentes_productor: [u8; 32],
    pub sha256_ejecutable: [u8; 32],
    pub receta: String,
    pub constante_r_bits: u64,
    pub origen_receta: String,
    pub comando: String,
    pub entradas: Vec<[u8; 32]>,
    pub sha256_salida: [u8; 32],
    pub bytes_salida: u64,
    pub sha256_registro_salida: [u8; 32],
    pub transformaciones: String,
}

const CLAVES_EVIDENCIA: [&str; 14] = [
    "formato",
    "productor",
    "commit_productor",
    "sha256_fuentes_productor",
    "sha256_ejecutable",
    "receta",
    "constante_r_bits",
    "origen_receta",
    "comando",
    "entrada_sha256",
    "sha256_salida",
    "bytes_salida",
    "sha256_registro_salida",
    "transformaciones",
];

pub fn leer_evidencia(texto: &[u8]) -> Result<EvidenciaProductor, String> {
    let c = campos(texto, &CLAVES_EVIDENCIA, &["entrada_sha256"])?;
    if uno(&c, "formato") != "tg-evidencia-productor-1" {
        return Err("formato de evidencia desconocido".to_string());
    }
    let mut entradas = Vec::new();
    for (k, v) in &c {
        if *k == "entrada_sha256" {
            let h = hex_exacto::<32>(v, "entrada_sha256")?;
            if entradas.contains(&h) {
                return Err("entrada_sha256 repetida".to_string());
            }
            entradas.push(h);
        }
    }
    let r_bits = hex_exacto::<8>(uno(&c, "constante_r_bits"), "constante_r_bits")?;
    Ok(EvidenciaProductor {
        productor: uno(&c, "productor").to_string(),
        commit_productor: hex_exacto(uno(&c, "commit_productor"), "commit_productor")?,
        sha256_fuentes_productor: hex_exacto(uno(&c, "sha256_fuentes_productor"), "sha256_fuentes_productor")?,
        sha256_ejecutable: hex_exacto(uno(&c, "sha256_ejecutable"), "sha256_ejecutable")?,
        receta: uno(&c, "receta").to_string(),
        constante_r_bits: u64::from_be_bytes(r_bits),
        origen_receta: uno(&c, "origen_receta").to_string(),
        comando: uno(&c, "comando").to_string(),
        entradas,
        sha256_salida: hex_exacto(uno(&c, "sha256_salida"), "sha256_salida")?,
        bytes_salida: decimal(uno(&c, "bytes_salida"), "bytes_salida")?,
        sha256_registro_salida: hex_exacto(uno(&c, "sha256_registro_salida"), "sha256_registro_salida")?,
        transformaciones: uno(&c, "transformaciones").to_string(),
    })
}

/// Entrada del indice revisado: evidencia reconocida con su productor, commit y receta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntradaIndice {
    pub evidencia_sha256: [u8; 32],
    pub productor: String,
    pub commit_productor: [u8; 20],
    pub receta: String,
}

/// Indice revisado `tg-indice-procedencia-1`: `revisado_por=` y una linea
/// `evidencia=<sha256>;<productor>;<commit>;<receta>` por evidencia reconocida.
pub fn leer_indice(texto: &[u8]) -> Result<(String, Vec<EntradaIndice>), String> {
    let c = campos(texto, &["formato", "revisado_por", "evidencia"], &["evidencia"])?;
    if uno(&c, "formato") != "tg-indice-procedencia-1" {
        return Err("formato de indice desconocido".to_string());
    }
    let mut v: Vec<EntradaIndice> = Vec::new();
    for (k, linea) in &c {
        if *k != "evidencia" {
            continue;
        }
        let partes: Vec<&str> = linea.split(';').collect();
        let [sha, productor, commit, receta] = partes[..] else {
            return Err(format!("entrada de indice mal formada: {linea}"));
        };
        if productor.is_empty() || receta.is_empty() {
            return Err(format!("entrada de indice incompleta: {linea}"));
        }
        let e = EntradaIndice {
            evidencia_sha256: hex_exacto(sha, "evidencia")?,
            productor: productor.to_string(),
            commit_productor: hex_exacto(commit, "commit")?,
            receta: receta.to_string(),
        };
        if v.iter().any(|x| x.evidencia_sha256 == e.evidencia_sha256) {
            return Err("evidencia repetida en el indice".to_string());
        }
        v.push(e);
    }
    Ok((uno(&c, "revisado_por").to_string(), v))
}

// ───────────────────────────────────── nucleo de admision ─────────────────────────────────────

/// Causa estructurada de un rechazo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Causa {
    /// No es una cinta REAL bien formada (cabecera, registros truncados).
    Formato,
    /// La firma recalculada no es relativa.
    Firma(EstadoGeneracion),
    ReciboAusente,
    ReciboInvalido,
    /// El recibo es de otro original (SHA-256 o tamano).
    OtroOriginal,
    CoberturaIncompleta,
    /// Registros esperados/analizados distintos de los del original.
    CuentasDiscordantes,
    /// Estado, recetas o testigo del recibo distintos de los recalculados.
    FirmaDiscordante,
    ProcedenciaNoAcreditada,
    IndiceAusente,
    /// El indice no tiene el SHA-256 fijado por quien llama.
    IndiceNoFijado,
    IndiceInvalido,
    /// La evidencia citada no esta en el indice revisado.
    EvidenciaNoReconocida,
    EvidenciaAusente,
    EvidenciaInvalida,
    /// Hash, productor, commit, receta, constante o salida de la cadena no concuerdan.
    CadenaRota,
}

impl Causa {
    pub fn nombre(self) -> &'static str {
        match self {
            Causa::Formato => "formato",
            Causa::Firma(EstadoGeneracion::Antigua) => "firma_antigua",
            Causa::Firma(EstadoGeneracion::Relativa) => "firma_relativa",
            Causa::Firma(EstadoGeneracion::Indeterminada) => "firma_indeterminada",
            Causa::Firma(EstadoGeneracion::Mixta) => "firma_mixta",
            Causa::Firma(EstadoGeneracion::OtraOInvalida) => "firma_otra_o_invalida",
            Causa::ReciboAusente => "recibo_ausente",
            Causa::ReciboInvalido => "recibo_invalido",
            Causa::OtroOriginal => "recibo_de_otro_original",
            Causa::CoberturaIncompleta => "cobertura_incompleta",
            Causa::CuentasDiscordantes => "cuentas_discordantes",
            Causa::FirmaDiscordante => "firma_discordante",
            Causa::ProcedenciaNoAcreditada => "procedencia_no_acreditada",
            Causa::IndiceAusente => "indice_ausente",
            Causa::IndiceNoFijado => "indice_no_fijado",
            Causa::IndiceInvalido => "indice_invalido",
            Causa::EvidenciaNoReconocida => "evidencia_no_reconocida",
            Causa::EvidenciaAusente => "evidencia_ausente",
            Causa::EvidenciaInvalida => "evidencia_invalida",
            Causa::CadenaRota => "cadena_rota",
        }
    }

    /// La generacion no quedo acreditada por una cadena integra (todo lo posterior a la firma).
    pub fn es_procedencia_no_acreditada(self) -> bool {
        !matches!(self, Causa::Formato | Causa::Firma(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rechazo {
    pub causa: Causa,
    pub detalle: String,
}

impl std::fmt::Display for Rechazo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.causa.es_procedencia_no_acreditada() && self.causa != Causa::ProcedenciaNoAcreditada {
            write!(f, "procedencia_no_acreditada ({}): {}", self.causa.nombre(), self.detalle)
        } else {
            write!(f, "{}: {}", self.causa.nombre(), self.detalle)
        }
    }
}

fn rechazo<T>(causa: Causa, detalle: impl Into<String>) -> Result<T, Rechazo> {
    Err(Rechazo { causa, detalle: detalle.into() })
}

/// Indice revisado con el SHA-256 que fija quien llama (del registro revisado, no del recibo).
#[derive(Debug, Clone, Copy)]
pub struct IndiceFijado<'a> {
    pub bytes: &'a [u8],
    pub sha256_esperado: [u8; 32],
}

/// Original admitido para preparar muestras, con su identidad recalculada y la cadena que lo acredita.
#[derive(Debug, Clone)]
pub struct Admision {
    pub sha256_original: [u8; 32],
    pub bytes_original: u64,
    pub firma: Firma,
    pub recibo: Recibo,
    pub evidencia_sha256: [u8; 32],
    pub evidencia: EvidenciaProductor,
    pub sha256_indice: [u8; 32],
    pub revisado_por: String,
}

/// Resolutor de evidencia por SHA-256 (`Ok(None)` = no existe).
pub type ResolverEvidencia<'a> = dyn FnMut(&[u8; 32]) -> Result<Option<Vec<u8>>, String> + 'a;

/// NUCLEO PURO de admision para ENTRENAR sobre los bytes completos del original (los que se van a consumir).
/// El SHA-256 y el tamano se calculan aqui, nunca se toman del recibo.
pub fn admitir_original(
    original: &[u8],
    recibo: Option<&[u8]>,
    indice: Option<IndiceFijado<'_>>,
    evidencia: &mut ResolverEvidencia<'_>,
) -> Result<Admision, Rechazo> {
    let sha256_original: [u8; 32] = Sha256::digest(original).into();
    let bytes_original = original.len() as u64;
    let firma = match firma_de_cinta(original) {
        Ok(f) => f,
        Err(e) => return rechazo(Causa::Formato, e),
    };
    let estado = firma.estado();
    if estado != EstadoGeneracion::Relativa {
        return rechazo(
            Causa::Firma(estado),
            format!(
                "firma de generacion {} (solo relativa {}, solo antigua {}, ninguna {}, ambas {}, sin decision {}, invalidos {}; testigos rel {:?} ant {:?} ninguna {:?} invalido {:?}): no entra a entrenamiento",
                estado.nombre(),
                firma.solo_relativa,
                firma.solo_antigua,
                firma.ninguna,
                firma.ambas,
                firma.sin_decision,
                firma.invalidos,
                firma.primer_solo_relativa,
                firma.primer_solo_antigua,
                firma.primer_ninguna,
                firma.primer_invalido
            ),
        );
    }
    let Some(texto) = recibo else {
        return rechazo(Causa::ReciboAusente, "marca de procedencia ausente");
    };
    let r = match leer_recibo(texto) {
        Ok(r) => r,
        Err(e) => return rechazo(Causa::ReciboInvalido, e),
    };
    if r.sha256_original != sha256_original || r.bytes_original != bytes_original {
        return rechazo(Causa::OtroOriginal, "el recibo es de otro original (SHA-256 o tamano distintos)");
    }
    if r.cobertura != "completa" {
        return rechazo(Causa::CoberturaIncompleta, format!("analisis de procedencia incompleto (cobertura {})", r.cobertura));
    }
    if r.registros_esperados != firma.registros || r.registros_analizados != firma.registros {
        return rechazo(
            Causa::CuentasDiscordantes,
            format!(
                "el recibo declara {} esperados y {} analizados; el original tiene {}",
                r.registros_esperados, r.registros_analizados, firma.registros
            ),
        );
    }
    if r.estado_firma != estado.nombre()
        || r.recetas_compatibles != firma.recetas_compatibles()
        || Some(r.primer_testigo_relativa) != firma.primer_solo_relativa
    {
        return rechazo(Causa::FirmaDiscordante, "estado, recetas o testigo del recibo distintos de los recalculados");
    }
    if r.procedencia_acreditada != "relativa" {
        return rechazo(Causa::ProcedenciaNoAcreditada, format!("procedencia declarada {}", r.procedencia_acreditada));
    }
    let Some(ix) = indice else {
        return rechazo(Causa::IndiceAusente, "sin indice revisado de evidencia");
    };
    let sha256_indice: [u8; 32] = Sha256::digest(ix.bytes).into();
    if sha256_indice != ix.sha256_esperado {
        return rechazo(Causa::IndiceNoFijado, format!("el indice tiene SHA-256 {}", hex(&sha256_indice)));
    }
    let (revisado_por, entradas) = match leer_indice(ix.bytes) {
        Ok(x) => x,
        Err(e) => return rechazo(Causa::IndiceInvalido, e),
    };
    let Some(entrada) = entradas.iter().find(|e| e.evidencia_sha256 == r.evidencia_productor_sha256) else {
        return rechazo(
            Causa::EvidenciaNoReconocida,
            format!("la evidencia {} no esta en el indice revisado", hex(&r.evidencia_productor_sha256)),
        );
    };
    let bytes_ev = match evidencia(&r.evidencia_productor_sha256) {
        Ok(Some(b)) => b,
        Ok(None) => return rechazo(Causa::EvidenciaAusente, format!("no hay evidencia {}", hex(&r.evidencia_productor_sha256))),
        Err(e) => return rechazo(Causa::EvidenciaAusente, e),
    };
    let sha_ev: [u8; 32] = Sha256::digest(&bytes_ev).into();
    if sha_ev != r.evidencia_productor_sha256 {
        return rechazo(Causa::CadenaRota, "la evidencia resuelta no tiene el SHA-256 citado");
    }
    let ev = match leer_evidencia(&bytes_ev) {
        Ok(e) => e,
        Err(e) => return rechazo(Causa::EvidenciaInvalida, e),
    };
    if ev.sha256_salida != sha256_original || ev.bytes_salida != bytes_original {
        return rechazo(Causa::CadenaRota, "la salida del productor no es este original");
    }
    if ev.productor != entrada.productor || ev.commit_productor != entrada.commit_productor || ev.receta != entrada.receta {
        return rechazo(Causa::CadenaRota, "productor, commit o receta distintos de los del indice revisado");
    }
    if ev.receta != "relativa" || ev.constante_r_bits != R_RELATIVA.to_bits() {
        return rechazo(Causa::CadenaRota, "la receta del productor no es la relativa con su constante");
    }
    if ev.transformaciones != "ninguna" {
        return rechazo(Causa::CadenaRota, format!("transformaciones intermedias sin evidencia: {}", ev.transformaciones));
    }
    Ok(Admision {
        sha256_original,
        bytes_original,
        firma,
        recibo: r,
        evidencia_sha256: sha_ev,
        evidencia: ev,
        sha256_indice,
        revisado_por,
    })
}

/// Fachada sobre una cinta logica ya abierta: delega en [`admitir_original`] con sus bytes.
pub fn admitir_para_entrenamiento(
    tape: &TapeBytes,
    recibo: Option<&[u8]>,
    indice: Option<IndiceFijado<'_>>,
    evidencia: &mut ResolverEvidencia<'_>,
) -> Result<Admision, String> {
    admitir_original(tape.as_bytes(), recibo, indice, evidencia).map_err(|e| format!("{}: {e}", tape.ruta_logica().display()))
}
