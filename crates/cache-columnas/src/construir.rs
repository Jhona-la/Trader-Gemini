//! Construccion atomica por hora (pedido v3, P1).
//!
//! Por cada hora de evento de la cinta: columnas en memoria -> `<raiz>/.tmp-<clave>-<pid>/` -> relectura y
//! verificacion de cada SHA-256 (columnas, indice y el tramo de registros reconstruido intercalando columnas) ->
//! renombrado al destino -> anotacion en el manifiesto. Una hora que falla retira su `.tmp` y no queda publicada.

use crate::expulsion;
use crate::presupuesto;
use crate::publicacion;
use crate::{
    ahora_ms, clave_hora, columnas, escribir_manifiesto, escribir_sincronizado, leer_manifiesto, leer_u64, nombre_valido,
    registros_desde_columnas, ruta_de_clave, sha256_hex, Cerrojo, ColumnaIndice, EntradaHora, Indice, Manifiesto,
    BYTES_PALABRA, BYTES_REGISTRO, CABECERA_TICKS, COLUMNAS_REGISTRO, CONTRATO_INDICE, MS_POR_HORA, NOMBRE_INDICE,
    PALABRAS_POR_REGISTRO, PREFIJO_TMP,
};
use std::path::Path;
use tick_tape::logica::{open_logical_tape, Transporte};
use tick_tape::tgb::Presupuesto;

/// De donde salen los registros (queda en el manifiesto).
#[derive(Debug, Clone)]
pub struct Origen {
    pub logica: String,
    pub fisica: String,
    pub representacion: String,
    pub sha256_original: String,
    pub bytes_original: u64,
    pub sha256_contenedor: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Opciones {
    pub fuente: String,
    pub simbolo: String,
    /// SHA-256 del ORIGINAL que se exige (hex); si no coincide no se construye nada.
    pub sha256_esperado: Option<String>,
    /// Tope derivado (P2). `None` solo en pruebas de P1; el binario lo exige siempre (`--config`).
    pub limite: Option<presupuesto::Limite>,
    /// Solo pruebas: falla despues de escribir el `.tmp` de la hora numero `k` (0 = primera) y antes de publicarla.
    #[doc(hidden)]
    pub fallo_inyectado_en_hora: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct Informe {
    pub horas_nuevas: Vec<String>,
    pub horas_presentes: Vec<String>,
    pub rows: u64,
    pub bytes_nuevos: u64,
    /// Sumandos del tope (P2) con los que se decidio.
    pub sumandos: Option<String>,
}

/// Simbolo por defecto: lo que va antes del primer `_` del nombre de la cinta (`BTCUSDT_2026-09-14_REAL.bin`).
pub fn simbolo_de_ruta(logica: &Path) -> Option<String> {
    let nombre = logica.file_name()?.to_str()?;
    let s = nombre.split('_').next()?;
    nombre_valido(s).then(|| s.to_string())
}

/// Construye (o completa) en `raiz` las horas de la cinta logica `logica` (`.bin`, o `.bin.tgb` si el `.bin` no existe).
pub fn construir_desde_cinta(raiz: &Path, logica: &Path, op: &Opciones) -> Result<Informe, String> {
    let tape = open_logical_tape(logica, Presupuesto::sin_tope()).map_err(|e| format!("{}: {e}", logica.display()))?;
    let sha = crate::hex(&tape.sha256_original());
    if let Some(esp) = &op.sha256_esperado {
        if !esp.eq_ignore_ascii_case(&sha) {
            return Err(format!("{}: el original no es el esperado (sha256 {sha}, se exigia {esp})", logica.display()));
        }
    }
    let bytes = tape.as_bytes();
    let off = if bytes.len() >= CABECERA_TICKS.len() && &bytes[..CABECERA_TICKS.len()] == CABECERA_TICKS {
        CABECERA_TICKS.len()
    } else {
        0
    };
    let sha256_contenedor = match tape.transporte() {
        Transporte::Bin => None,
        Transporte::Tgb { sha256_contenedor, .. } => Some(crate::hex(&sha256_contenedor)),
    };
    let origen = Origen {
        logica: logica.to_string_lossy().into_owned(),
        fisica: tape.ruta_fisica().to_string_lossy().into_owned(),
        representacion: if sha256_contenedor.is_some() { "tgb" } else { "bin" }.to_string(),
        sha256_original: sha,
        bytes_original: bytes.len() as u64,
        sha256_contenedor,
    };
    construir_registros(raiz, &origen, &bytes[off..], None, op)
}

/// Construye desde los bytes de los registros (`ts u64, bid, ask, bq, aq f64`, LE, sin cabecera).
/// `pub_explicito` es obligatorio para las fuentes cuya regla no tiene formula y esta prohibido para las demas.
pub fn construir_registros(
    raiz: &Path,
    origen: &Origen,
    cuerpo: &[u8],
    pub_explicito: Option<&[i64]>,
    op: &Opciones,
) -> Result<Informe, String> {
    let fuente = publicacion::fuente(&op.fuente)
        .ok_or_else(|| format!("fuente {:?} fuera de la tabla de publicacion", op.fuente))?;
    if !nombre_valido(&op.simbolo) {
        return Err(format!("simbolo invalido: {:?}", op.simbolo));
    }
    if !cuerpo.len().is_multiple_of(BYTES_REGISTRO) {
        return Err(format!("{}: registro parcial ({} B no es multiplo de {BYTES_REGISTRO})", origen.logica, cuerpo.len()));
    }
    let n = cuerpo.len() / BYTES_REGISTRO;
    if n == 0 {
        return Err(format!("{}: cinta sin registros", origen.logica));
    }
    let ts = |i: usize| leer_u64(&cuerpo[i * BYTES_REGISTRO..], 0);
    if let Some(i) = (1..n).find(|&i| ts(i) < ts(i - 1)) {
        return Err(format!("{}: ts decreciente en el registro {i}; no se reordena", origen.logica));
    }
    let pubs: Vec<i64> = match pub_explicito {
        Some(p) => {
            if publicacion::por_regla(fuente.regla, ts(0))?.is_some() {
                return Err(format!("la fuente {} tiene formula de publicacion; no se acepta pub_ms del productor", fuente.nombre));
            }
            if p.len() != n {
                return Err(format!("pub_ms con {} filas para {n} registros", p.len()));
            }
            p.to_vec()
        }
        None => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let p = publicacion::por_regla(fuente.regla, ts(i))?.ok_or_else(|| {
                    format!("la fuente {} ({}) exige pub_ms del productor", fuente.nombre, fuente.regla.nombre())
                })?;
                v.push(p);
            }
            v
        }
    };
    if let Some(i) = (0..n).find(|&i| (pubs[i] as i128) < ts(i) as i128) {
        return Err(format!("{}: pub_ms {} anterior al evento {} en la fila {i}", origen.logica, pubs[i], ts(i)));
    }

    let _cerrojo = Cerrojo::tomar(raiz)?;
    let mut man = leer_manifiesto(raiz)?;
    let mut inf = Informe::default();
    // P2: el tope se mide en vivo al empezar (archivo y proyecto) y se compara contra el uso ACTUAL de la cache
    // antes de cada hora. Fail-closed: tope <= 0 -> no se construye nada.
    let sumandos = match &op.limite {
        Some(l) => {
            let s = presupuesto::medir(&l.config, &l.raiz_medicion, raiz)?;
            inf.sumandos = Some(s.texto(man.uso_bytes()));
            if s.limite_b <= 0 {
                return Err(format!("limite de la cache <= 0: no se construye nada\n{}", s.texto(man.uso_bytes())));
            }
            Some(s)
        }
        None => None,
    };
    let mut a = 0usize;
    let mut k = 0usize;
    while a < n {
        let hora = ts(a) / MS_POR_HORA;
        let mut b = a + 1;
        while b < n && ts(b) / MS_POR_HORA == hora {
            b += 1;
        }
        let tramo = Tramo { cuerpo, pubs: &pubs, desde: a, hasta: b, hora_ms: hora * MS_POR_HORA, numero: k };
        match construir_hora(raiz, &mut man, origen, fuente, &tramo, op, sumandos.as_ref())? {
            Estado::Nueva { clave, bytes } => {
                inf.bytes_nuevos += bytes;
                inf.horas_nuevas.push(clave);
            }
            Estado::Presente { clave } => inf.horas_presentes.push(clave),
        }
        escribir_manifiesto(raiz, &man)?;
        inf.rows += (b - a) as u64;
        a = b;
        k += 1;
    }
    Ok(inf)
}

struct Tramo<'a> {
    cuerpo: &'a [u8],
    pubs: &'a [i64],
    desde: usize,
    hasta: usize,
    hora_ms: u64,
    numero: usize,
}

enum Estado {
    Nueva { clave: String, bytes: u64 },
    Presente { clave: String },
}

/// Columnas e indice de una hora, en memoria.
struct HoraEnMemoria {
    columnas: Vec<(String, Vec<u8>)>,
    indice: Vec<u8>,
    sha256_indice: String,
    sha256_tramo: String,
    rows: u64,
    pub_min: i64,
    pub_max: i64,
    bytes: u64,
}

fn en_memoria(origen: &Origen, fuente: &publicacion::Fuente, op: &Opciones, t: &Tramo) -> Result<HoraEnMemoria, String> {
    let rows = t.hasta - t.desde;
    let defs = columnas();
    let mut cols: Vec<Vec<u8>> = Vec::with_capacity(defs.len());
    for _ in &defs {
        let mut v = Vec::new();
        v.try_reserve_exact(rows * BYTES_PALABRA).map_err(|e| e.to_string())?;
        cols.push(v);
    }
    for i in t.desde..t.hasta {
        let r = &t.cuerpo[i * BYTES_REGISTRO..(i + 1) * BYTES_REGISTRO];
        for (w, col) in cols.iter_mut().take(PALABRAS_POR_REGISTRO).enumerate() {
            col.extend_from_slice(&r[w * BYTES_PALABRA..(w + 1) * BYTES_PALABRA]);
        }
        cols[PALABRAS_POR_REGISTRO].extend_from_slice(&t.pubs[i].to_le_bytes());
    }
    let pubs = &t.pubs[t.desde..t.hasta];
    let pub_min = pubs.iter().copied().min().unwrap_or(0);
    let pub_max = pubs.iter().copied().max().unwrap_or(0);
    let sha256_tramo = sha256_hex(&t.cuerpo[t.desde * BYTES_REGISTRO..t.hasta * BYTES_REGISTRO]);
    let indice = Indice {
        contrato: CONTRATO_INDICE.to_string(),
        fuente: fuente.nombre.to_string(),
        simbolo: op.simbolo.clone(),
        hora_ms: t.hora_ms,
        rows: rows as u64,
        sha256_original: origen.sha256_original.clone(),
        bytes_original: origen.bytes_original,
        registro_desde: t.desde as u64,
        registro_hasta: t.hasta as u64,
        sha256_tramo: sha256_tramo.clone(),
        regla_publicacion: fuente.regla.nombre().to_string(),
        pub_min_ms: pub_min,
        pub_max_ms: pub_max,
        columnas: defs
            .iter()
            .zip(&cols)
            .map(|(d, c)| ColumnaIndice {
                file: d.file.to_string(),
                field: d.field.to_string(),
                tipo: d.tipo.to_string(),
                scale: None,
                rows: rows as u64,
                bytes: c.len() as u64,
                sha256: sha256_hex(c),
            })
            .collect(),
    };
    let indice = serde_json::to_vec_pretty(&indice).map_err(|e| e.to_string())?;
    let bytes = cols.iter().map(|c| c.len() as u64).sum::<u64>() + indice.len() as u64;
    Ok(HoraEnMemoria {
        columnas: defs.iter().map(|d| d.file.to_string()).zip(cols).collect(),
        sha256_indice: sha256_hex(&indice),
        indice,
        sha256_tramo,
        rows: rows as u64,
        pub_min,
        pub_max,
        bytes,
    })
}

/// Escribe la hora en `tmp` y la relee: cada columna, el indice y el tramo reconstruido deben dar su SHA-256.
fn escribir_y_verificar(tmp: &Path, h: &HoraEnMemoria, fallo: bool) -> Result<(), String> {
    std::fs::create_dir_all(tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    for (nombre, bytes) in &h.columnas {
        escribir_sincronizado(&tmp.join(nombre), bytes)?;
    }
    escribir_sincronizado(&tmp.join(NOMBRE_INDICE), &h.indice)?;
    if fallo {
        return Err(format!("{}: fallo inyectado (prueba) tras escribir el temporal", tmp.display()));
    }
    let mut leidas = Vec::with_capacity(h.columnas.len());
    for (nombre, bytes) in &h.columnas {
        let p = tmp.join(nombre);
        let r = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        if sha256_hex(&r) != sha256_hex(bytes) {
            return Err(format!("{}: la relectura no da el SHA-256 escrito", p.display()));
        }
        leidas.push(r);
    }
    let idx = std::fs::read(tmp.join(NOMBRE_INDICE)).map_err(|e| format!("{}: {e}", tmp.display()))?;
    if sha256_hex(&idx) != h.sha256_indice {
        return Err(format!("{}: la relectura del indice no da su SHA-256", tmp.display()));
    }
    let refs: Vec<&[u8]> = leidas.iter().take(COLUMNAS_REGISTRO.len()).map(|v| v.as_slice()).collect();
    let tramo = registros_desde_columnas(&refs, h.rows as usize)?;
    if sha256_hex(&tramo) != h.sha256_tramo {
        return Err(format!("{}: las columnas no restituyen los registros del original", tmp.display()));
    }
    Ok(())
}

fn construir_hora(
    raiz: &Path,
    man: &mut Manifiesto,
    origen: &Origen,
    fuente: &publicacion::Fuente,
    t: &Tramo,
    op: &Opciones,
    sumandos: Option<&presupuesto::Sumandos>,
) -> Result<Estado, String> {
    let clave = clave_hora(fuente.nombre, &op.simbolo, t.hora_ms);
    let previa = man.posicion(&clave);
    if let Some(p) = previa {
        if man.horas[p].sha256_original != origen.sha256_original {
            return Err(format!(
                "{clave}: ya esta en la cache desde otro original ({}); no se mezcla: expulsala antes",
                man.horas[p].sha256_original
            ));
        }
    }
    let h = en_memoria(origen, fuente, op, t)?;
    let destino = ruta_de_clave(raiz, &clave)?;
    let indice_en_destino = std::fs::read(destino.join(NOMBRE_INDICE)).ok().map(|b| sha256_hex(&b));
    if let (Some(p), Some(sha)) = (previa, &indice_en_destino) {
        if man.horas[p].sha256_indice == h.sha256_indice && *sha == h.sha256_indice {
            return Ok(Estado::Presente { clave });
        }
    }
    if let Some(s) = sumandos {
        let uso_sin_esta = man.uso_bytes() - previa.map(|p| man.horas[p].bytes).unwrap_or(0);
        if uso_sin_esta as i128 + h.bytes as i128 > s.limite_b {
            // P3: se expulsa lo minimo (LRU) para que quepa; nunca las horas de este mismo original (la
            // construccion en curso) ni las que estan en lectura. Si no alcanza, no se expulsa ni se construye nada.
            let uso_total = man.uso_bytes();
            let no_cabe = |motivo: String| {
                format!(
                    "{clave}: no cabe ({} B nuevos + {uso_sin_esta} B en uso > limite {} B): {motivo}; no se construye\n{}",
                    h.bytes,
                    s.limite_b,
                    s.texto(uso_total)
                )
            };
            let objetivo = s.limite_b - h.bytes as i128;
            if objetivo < 0 {
                return Err(no_cabe("la hora sola supera el limite".into()));
            }
            let protegidas: Vec<String> = man
                .horas
                .iter()
                .filter(|x| x.sha256_original == origen.sha256_original || x.clave == clave)
                .map(|x| x.clave.clone())
                .collect();
            let motivo = format!("hacer sitio a {clave} ({} B; limite {} B)", h.bytes, s.limite_b);
            if let Err(e) = expulsion::expulsar_en(raiz, man, objetivo as u64, &protegidas, &motivo) {
                return Err(no_cabe(e));
            }
        }
    }
    // La expulsion pudo mover posiciones del manifiesto (nunca retira esta clave, que esta protegida).
    let previa = man.posicion(&clave);
    let tmp = raiz.join(format!("{PREFIJO_TMP}{}-{}", clave.replace('/', "_"), std::process::id()));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    }
    if let Err(e) = escribir_y_verificar(&tmp, &h, op.fallo_inyectado_en_hora == Some(t.numero)) {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(e);
    }
    match &indice_en_destino {
        Some(sha) if *sha == h.sha256_indice => {
            // Hora ya publicada e identica (p. ej. corrida caida entre el renombrado y el manifiesto): se adopta.
            let _ = std::fs::remove_dir_all(&tmp);
        }
        Some(_) => {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(format!("{}: el destino existe con otro contenido; no se sobrescribe", destino.display()));
        }
        None => {
            let publicar = || -> Result<(), String> {
                if let Some(padre) = destino.parent() {
                    std::fs::create_dir_all(padre).map_err(|e| format!("{}: {e}", padre.display()))?;
                }
                std::fs::rename(&tmp, &destino).map_err(|e| format!("{} -> {}: {e}", tmp.display(), destino.display()))
            };
            if let Err(e) = publicar() {
                let _ = std::fs::remove_dir_all(&tmp);
                return Err(e);
            }
        }
    }
    let entrada = EntradaHora {
        clave: clave.clone(),
        fuente: fuente.nombre.to_string(),
        simbolo: op.simbolo.clone(),
        hora_ms: t.hora_ms,
        origen_logica: origen.logica.clone(),
        origen_fisica: origen.fisica.clone(),
        representacion: origen.representacion.clone(),
        sha256_original: origen.sha256_original.clone(),
        bytes_original: origen.bytes_original,
        sha256_contenedor: origen.sha256_contenedor.clone(),
        sha256_indice: h.sha256_indice.clone(),
        construida_ms: ahora_ms(),
        ultimo_acceso_ms: previa.and_then(|p| man.horas[p].ultimo_acceso_ms),
        bytes: h.bytes,
        rows: h.rows,
        registro_desde: t.desde as u64,
        registro_hasta: t.hasta as u64,
        pub_min_ms: h.pub_min,
        pub_max_ms: h.pub_max,
    };
    match previa {
        Some(p) => man.horas[p] = entrada,
        None => man.horas.push(entrada),
    }
    Ok(Estado::Nueva { clave, bytes: h.bytes })
}
