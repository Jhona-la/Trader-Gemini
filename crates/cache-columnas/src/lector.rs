//! Lectura por mmap de horas de la cache (pedido v3, P3/P5).
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
