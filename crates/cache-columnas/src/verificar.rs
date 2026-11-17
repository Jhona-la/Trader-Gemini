//! `cache-columnas verificar`: rehace las comprobaciones de la construccion sin reconstruir (pedido v3, P1.3).
//!
//! Por hora: SHA-256 del `index.json` contra el manifiesto; cada columna contra el `sha256`, `bytes` y `rows` del
//! indice; los registros reconstruidos intercalando columnas contra `sha256_tramo`; `pub_ms` >= `ts` y su rango.
//! Con `origen`, ademas abre la cinta logica de origen y compara el SHA-256 del original y el del tramo.

use crate::{
    columnas, leer_i64, leer_manifiesto, leer_u64, registros_desde_columnas, ruta_de_clave, sha256_hex, EntradaHora,
    Indice, BYTES_PALABRA, BYTES_REGISTRO, CABECERA_TICKS, CONTRATO_INDICE, NOMBRE_INDICE, PALABRAS_POR_REGISTRO,
};
use std::collections::BTreeMap;
use std::path::Path;
use tick_tape::logica::open_logical_tape;
use tick_tape::tgb::Presupuesto;

#[derive(Debug, Clone)]
pub struct Hallazgo {
    pub clave: String,
    pub problema: String,
}

#[derive(Debug, Clone, Default)]
pub struct InformeVerificacion {
    pub horas: usize,
    pub rows: u64,
    pub origenes_comprobados: usize,
    pub hallazgos: Vec<Hallazgo>,
}

/// Comprueba una hora contra su entrada del manifiesto, con los bytes ya leidos (archivo o mapa).
/// `cols` va en el orden de [`columnas`].
pub fn comprobar_hora(e: &EntradaHora, indice: &[u8], cols: &[&[u8]]) -> Result<Indice, String> {
    if sha256_hex(indice) != e.sha256_indice {
        return Err("index.json no coincide con el SHA-256 del manifiesto".into());
    }
    let idx: Indice = serde_json::from_slice(indice).map_err(|x| format!("index.json ilegible: {x}"))?;
    let defs = columnas();
    if idx.contrato != CONTRATO_INDICE
        || idx.fuente != e.fuente
        || idx.simbolo != e.simbolo
        || idx.hora_ms != e.hora_ms
        || idx.rows != e.rows
        || idx.sha256_original != e.sha256_original
        || idx.registro_desde != e.registro_desde
        || idx.registro_hasta != e.registro_hasta
        || idx.registro_hasta.checked_sub(idx.registro_desde) != Some(idx.rows)
        || idx.pub_min_ms != e.pub_min_ms
        || idx.pub_max_ms != e.pub_max_ms
        || idx.columnas.len() != defs.len()
        || cols.len() != defs.len()
    {
        return Err("el indice no describe la hora del manifiesto".into());
    }
    let rows = idx.rows as usize;
    for ((d, c), b) in defs.iter().zip(&idx.columnas).zip(cols) {
        if c.file != d.file || c.field != d.field || c.tipo != d.tipo || c.scale.is_some() || c.rows != idx.rows {
            return Err(format!("columna {} no es la del contrato", d.field));
        }
        if b.len() as u64 != c.bytes || b.len() != rows * BYTES_PALABRA {
            return Err(format!("columna {}: {} B, el indice dice {}", d.field, b.len(), c.bytes));
        }
        if sha256_hex(b) != c.sha256 {
            return Err(format!("columna {}: SHA-256 distinto del indice", d.field));
        }
    }
    let tramo = registros_desde_columnas(&cols[..PALABRAS_POR_REGISTRO], rows)?;
    if sha256_hex(&tramo) != idx.sha256_tramo {
        return Err("las columnas no restituyen el tramo de registros del original".into());
    }
    let (ts, pubs) = (cols[0], cols[PALABRAS_POR_REGISTRO]);
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for i in 0..rows {
        let p = leer_i64(pubs, i);
        if (p as i128) < leer_u64(ts, i) as i128 {
            return Err(format!("fila {i}: pub_ms anterior al evento"));
        }
        lo = lo.min(p);
        hi = hi.max(p);
    }
    if rows > 0 && (lo != idx.pub_min_ms || hi != idx.pub_max_ms) {
        return Err("rango pub_ms distinto del indice".into());
    }
    Ok(idx)
}

/// Lee de disco y comprueba una hora.
pub fn verificar_hora(raiz: &Path, e: &EntradaHora) -> Result<Indice, String> {
    let dir = ruta_de_clave(raiz, &e.clave)?;
    let indice = std::fs::read(dir.join(NOMBRE_INDICE)).map_err(|x| format!("{}: {x}", dir.display()))?;
    let mut leidas = Vec::new();
    for d in columnas() {
        let p = dir.join(d.file);
        leidas.push(std::fs::read(&p).map_err(|x| format!("{}: {x}", p.display()))?);
    }
    let refs: Vec<&[u8]> = leidas.iter().map(|v| v.as_slice()).collect();
    comprobar_hora(e, &indice, &refs)
}

pub fn verificar(raiz: &Path, origen: bool) -> Result<InformeVerificacion, String> {
    let man = leer_manifiesto(raiz)?;
    let mut inf = InformeVerificacion::default();
    let mut por_origen: BTreeMap<String, Vec<(EntradaHora, Indice)>> = BTreeMap::new();
    for e in &man.horas {
        inf.horas += 1;
        inf.rows += e.rows;
        match verificar_hora(raiz, e) {
            Ok(idx) => por_origen.entry(e.origen_logica.clone()).or_default().push((e.clone(), idx)),
            Err(p) => inf.hallazgos.push(Hallazgo { clave: e.clave.clone(), problema: p }),
        }
    }
    if origen {
        for (logica, horas) in &por_origen {
            inf.origenes_comprobados += 1;
            let tape = match open_logical_tape(Path::new(logica), Presupuesto::sin_tope()) {
                Ok(t) => t,
                Err(x) => {
                    for (e, _) in horas {
                        inf.hallazgos.push(Hallazgo { clave: e.clave.clone(), problema: format!("origen {logica}: {x}") });
                    }
                    continue;
                }
            };
            let sha = crate::hex(&tape.sha256_original());
            let b = tape.as_bytes();
            let off = if b.len() >= CABECERA_TICKS.len() && &b[..CABECERA_TICKS.len()] == CABECERA_TICKS {
                CABECERA_TICKS.len()
            } else {
                0
            };
            for (e, idx) in horas {
                let (a, z) = (idx.registro_desde as usize, idx.registro_hasta as usize);
                let tramo = b.get(off + a * BYTES_REGISTRO..off + z * BYTES_REGISTRO);
                let problema = if sha != e.sha256_original {
                    Some(format!("origen {logica}: SHA-256 del original {sha}, el manifiesto dice {}", e.sha256_original))
                } else if tramo.map(sha256_hex).as_deref() != Some(idx.sha256_tramo.as_str()) {
                    Some(format!("origen {logica}: el tramo [{a},{z}) no coincide"))
                } else {
                    None
                };
                if let Some(problema) = problema {
                    inf.hallazgos.push(Hallazgo { clave: e.clave.clone(), problema });
                }
            }
        }
    }
    Ok(inf)
}
