//! Lectura por mmap de horas de la cache (pedido v3, P3/P4/P5).
//!
//! Montar una hora: cerrojo de lectura en su carpeta (la expulsion no la toca mientras viva), mapa de cada columna,
//! comprobacion de TODOS sus SHA-256 sobre los bytes mapeados ([`crate::verificar::comprobar_hora`]) y marca de
//! `ultimo_acceso_ms` en el manifiesto (la LRU de P3 es por uso real).

use crate::verificar::comprobar_hora;
use crate::{
    ahora_ms, columnas, escribir_manifiesto, leer_f64_bits, leer_i64, leer_manifiesto, leer_u64, registros_desde_columnas,
    ruta_de_clave, Cerrojo, EntradaHora, Indice, NOMBRE_INDICE, PALABRAS_POR_REGISTRO, PREFIJO_CERROJO_LECTURA,
};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static SECUENCIA: AtomicUsize = AtomicUsize::new(0);

/// Una hora montada y verificada. Sus columnas son mapas de solo lectura; el cerrojo se retira al soltarla.
pub struct HoraMontada {
    pub entrada: EntradaHora,
    pub indice: Indice,
    mapas: Vec<memmap2::Mmap>,
    cerrojo: PathBuf,
}

impl HoraMontada {
    pub fn rows(&self) -> usize {
        self.indice.rows as usize
    }
    /// Bytes de la columna `k` (orden de [`columnas`]).
    pub fn columna(&self, k: usize) -> &[u8] {
        &self.mapas[k][..]
    }
    pub fn ts(&self, i: usize) -> u64 {
        leer_u64(self.columna(0), i)
    }
    /// bid (1), ask (2), bq (3), aq (4): bits exactos del origen.
    pub fn valor_f64(&self, k: usize, i: usize) -> f64 {
        leer_f64_bits(self.columna(k), i)
    }
    pub fn pub_ms(&self, i: usize) -> i64 {
        leer_i64(self.columna(PALABRAS_POR_REGISTRO), i)
    }
    /// Los registros del original (`ts, bid, ask, bq, aq`) de esta hora, byte a byte.
    pub fn registros(&self) -> Result<Vec<u8>, String> {
        let cols: Vec<&[u8]> = (0..PALABRAS_POR_REGISTRO).map(|k| self.columna(k)).collect();
        registros_desde_columnas(&cols, self.rows())
    }
}

impl Drop for HoraMontada {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.cerrojo);
    }
}

fn mapear(p: &Path) -> Result<memmap2::Mmap, String> {
    let f = File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    unsafe { memmap2::MmapOptions::new().map(&f) }.map_err(|e| format!("{}: {e}", p.display()))
}

fn montar_una(raiz: &Path, e: &EntradaHora) -> Result<HoraMontada, String> {
    let dir = ruta_de_clave(raiz, &e.clave)?;
    let cerrojo = dir.join(format!(
        "{PREFIJO_CERROJO_LECTURA}{}-{}.lock",
        std::process::id(),
        SECUENCIA.fetch_add(1, Ordering::Relaxed)
    ));
    OpenOptions::new().write(true).create_new(true).open(&cerrojo).map_err(|x| format!("{}: {x}", cerrojo.display()))?;
    let montar = || -> Result<(Indice, Vec<memmap2::Mmap>), String> {
        let indice = std::fs::read(dir.join(NOMBRE_INDICE)).map_err(|x| format!("{}: {x}", dir.display()))?;
        let mut mapas = Vec::new();
        for d in columnas() {
            mapas.push(mapear(&dir.join(d.file))?);
        }
        let refs: Vec<&[u8]> = mapas.iter().map(|m| &m[..]).collect();
        let idx = comprobar_hora(e, &indice, &refs).map_err(|x| format!("{}: {x}", e.clave))?;
        Ok((idx, mapas))
    };
    match montar() {
        Ok((indice, mapas)) => Ok(HoraMontada { entrada: e.clone(), indice, mapas, cerrojo }),
        Err(x) => {
            let _ = std::fs::remove_file(&cerrojo);
            Err(x)
        }
    }
}

/// Monta las horas `claves` (en ese orden) y marca su acceso. Falla entera si alguna no verifica.
pub fn montar(raiz: &Path, claves: &[String]) -> Result<Vec<HoraMontada>, String> {
    let _c = Cerrojo::tomar(raiz)?;
    let mut man = leer_manifiesto(raiz)?;
    let mut horas = Vec::with_capacity(claves.len());
    for clave in claves {
        let i = man.posicion(clave).ok_or_else(|| format!("{clave}: no esta en el manifiesto"))?;
        horas.push(montar_una(raiz, &man.horas[i])?);
    }
    let ahora = ahora_ms();
    for h in &horas {
        if let Some(i) = man.posicion(&h.entrada.clave) {
            man.horas[i].ultimo_acceso_ms = Some(ahora);
        }
    }
    escribir_manifiesto(raiz, &man)?;
    Ok(horas)
}

/// Consulta por momento de PUBLICACION (P4): el rango `[desde_pub_ms, hasta_pub_ms)` se evalua sobre `pub_ms`,
/// nunca sobre la hora del evento. `series` = pares (fuente, simbolo).
#[derive(Debug, Clone)]
pub struct Consulta {
    pub series: Vec<(String, String)>,
    pub desde_pub_ms: i64,
    pub hasta_pub_ms: i64,
}

/// Una fila entregada: hora montada `hora` (indice en [`Entrega::horas`]) y fila `fila` dentro de ella.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fila {
    pub pub_ms: i64,
    pub ts: u64,
    pub hora: usize,
    pub fila: usize,
}

pub struct Entrega {
    pub horas: Vec<HoraMontada>,
    /// Ordenadas por `pub_ms`; a igualdad, por serie (orden de la consulta), hora y fila (orden estable).
    pub filas: Vec<Fila>,
}

/// Une las series pedidas y entrega SOLO lo publicado en `[desde, hasta)`, ordenado por `pub_ms`.
pub fn leer(raiz: &Path, c: &Consulta) -> Result<Entrega, String> {
    if c.desde_pub_ms >= c.hasta_pub_ms {
        return Err(format!("rango vacio: [{}, {})", c.desde_pub_ms, c.hasta_pub_ms));
    }
    let man = leer_manifiesto(raiz)?;
    let mut elegidas: Vec<(usize, u64, String)> = Vec::new();
    for (k, (fuente, simbolo)) in c.series.iter().enumerate() {
        for e in &man.horas {
            // Seleccion por el rango pub_ms del manifiesto (P4 c), sin escanear columnas.
            if &e.fuente == fuente && &e.simbolo == simbolo && e.pub_min_ms < c.hasta_pub_ms && e.pub_max_ms >= c.desde_pub_ms {
                elegidas.push((k, e.hora_ms, e.clave.clone()));
            }
        }
    }
    elegidas.sort();
    let claves: Vec<String> = elegidas.into_iter().map(|(_, _, clave)| clave).collect();
    let horas = montar(raiz, &claves)?;
    let mut filas = Vec::new();
    for (hi, h) in horas.iter().enumerate() {
        for i in 0..h.rows() {
            let t = h.pub_ms(i);
            if t >= c.desde_pub_ms && t < c.hasta_pub_ms {
                filas.push(Fila { pub_ms: t, ts: h.ts(i), hora: hi, fila: i });
            }
        }
    }
    filas.sort_by_key(|f| f.pub_ms);
    Ok(Entrega { horas, filas })
}

/// Rango `pub_ms` [min, max] de una serie segun el manifiesto (para que el entrenador sepa que pedir).
pub fn rango_pub(raiz: &Path, fuente: &str, simbolo: &str) -> Result<Option<(i64, i64)>, String> {
    let man = leer_manifiesto(raiz)?;
    let mut r: Option<(i64, i64)> = None;
    for e in man.horas.iter().filter(|e| e.fuente == fuente && e.simbolo == simbolo) {
        r = Some(match r {
            None => (e.pub_min_ms, e.pub_max_ms),
            Some((a, b)) => (a.min(e.pub_min_ms), b.max(e.pub_max_ms)),
        });
    }
    Ok(r)
}
/// P5: los registros (`ts, bid, ask, bq, aq`, LE, sin cabecera) del original con SHA-256 `sha256_original` (hex),
/// servidos desde la cache. `Ok(None)` si la cache no existe o ninguna serie cubre TODOS los registros
/// `[0, n_registros)` sin huecos; `Err` si la cubre pero alguna hora no verifica (quien llama cae a la cinta).
pub fn registros_de_original(raiz: &Path, sha256_original: &str, n_registros: u64) -> Result<Option<Vec<u8>>, String> {
    let man = leer_manifiesto(raiz)?;
    let mut series: Vec<(String, String)> = man
        .horas
        .iter()
        .filter(|e| e.sha256_original.eq_ignore_ascii_case(sha256_original))
        .map(|e| (e.fuente.clone(), e.simbolo.clone()))
        .collect();
    series.sort();
    series.dedup();
    for (fuente, simbolo) in series {
        let mut horas: Vec<&EntradaHora> = man
            .horas
            .iter()
            .filter(|e| e.sha256_original.eq_ignore_ascii_case(sha256_original) && e.fuente == fuente && e.simbolo == simbolo)
            .collect();
        horas.sort_by_key(|e| e.registro_desde);
        let mut siguiente = 0u64;
        let cubre = horas.iter().all(|e| {
            let ok = e.registro_desde == siguiente;
            siguiente = e.registro_hasta;
            ok
        }) && siguiente == n_registros;
        if !cubre {
            continue;
        }
        let claves: Vec<String> = horas.iter().map(|e| e.clave.clone()).collect();
        let montadas = montar(raiz, &claves)?;
        let mut v = Vec::new();
        for h in &montadas {
            v.extend_from_slice(&h.registros()?);
        }
        if v.len() as u64 != n_registros.saturating_mul(crate::BYTES_REGISTRO as u64) {
            return Err(format!("la cache entrego {} B para {n_registros} registros", v.len()));
        }
        return Ok(Some(v));
    }
    Ok(None)
}