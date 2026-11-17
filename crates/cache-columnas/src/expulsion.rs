//! Expulsion sin perdida (pedido v3, P3).
//!
//! Todo lo de la cache se rehace desde la cinta de origen (manifiesto con su SHA-256): expulsar solo quita una copia
//! derivada. Solo se retiran rutas BAJO la raiz de la cache que figuran en el manifiesto, y cada una queda en
//! `EVICCIONES.jsonl`. Orden: ultimo acceso de entrenamiento mas viejo (nunca leida = la mas vieja), luego `pub_min_ms`
//! mas viejo. Jamas: una hora con cerrojo de lectura, una protegida por la construccion en curso, ni nada de fuera.
//! Primero se planifica: si el objetivo no se alcanza, no se expulsa nada.

use crate::{
    ahora_ms, escribir_manifiesto, leer_manifiesto, ruta_de_clave, Cerrojo, Manifiesto, NOMBRE_EVICCIONES,
    PREFIJO_CERROJO_LECTURA,
};
use std::io::Write;
use std::path::Path;

/// Variable de entorno SOLO de prueba: si existe, cada ruta retirada se anota (una por linea) en ese archivo.
pub const ENV_REGISTRO_BORRADOS: &str = "TG_CACHE_COLUMNAS_REGISTRO_BORRADOS";

/// Hay un cerrojo de lectura en la hora (o no se puede saber): no se toca.
fn en_uso(dir: &Path) -> bool {
    match std::fs::read_dir(dir) {
        Ok(rd) => rd.flatten().any(|e| e.file_name().to_string_lossy().starts_with(PREFIJO_CERROJO_LECTURA)),
        Err(e) => e.kind() != std::io::ErrorKind::NotFound,
    }
}

/// Indices del manifiesto a expulsar, en orden, para dejar el uso en `objetivo` bytes o menos.
pub fn planificar(raiz: &Path, man: &Manifiesto, objetivo: u64, protegidas: &[String]) -> Result<Vec<usize>, String> {
    let mut uso = man.uso_bytes();
    if uso <= objetivo {
        return Ok(Vec::new());
    }
    let mut cand: Vec<usize> = Vec::new();
    for (i, h) in man.horas.iter().enumerate() {
        if protegidas.contains(&h.clave) {
            continue;
        }
        let dir = ruta_de_clave(raiz, &h.clave)?;
        if !en_uso(&dir) {
            cand.push(i);
        }
    }
    cand.sort_by_key(|&i| {
        let h = &man.horas[i];
        (h.ultimo_acceso_ms.unwrap_or(0), h.pub_min_ms, h.hora_ms, h.clave.clone())
    });
    let mut plan = Vec::new();
    for i in cand {
        if uso <= objetivo {
            break;
        }
        uso -= man.horas[i].bytes;
        plan.push(i);
    }
    if uso > objetivo {
        return Err(format!(
            "expulsando todo lo expulsable el uso quedaria en {uso} B > objetivo {objetivo} B (horas en lectura o protegidas); no se expulsa nada"
        ));
    }
    Ok(plan)
}

fn retirar(raiz: &Path, clave: &str) -> Result<(), String> {
    let dir = ruta_de_clave(raiz, clave)?;
    let raiz_c = std::fs::canonicalize(raiz).map_err(|e| format!("{}: {e}", raiz.display()))?;
    let dir_c = match std::fs::canonicalize(&dir) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("{}: {e}", dir.display())),
    };
    if dir_c == raiz_c || !dir_c.starts_with(&raiz_c) {
        return Err(format!("{} no esta bajo la cache {}: no se borra", dir_c.display(), raiz_c.display()));
    }
    if let Ok(reg) = std::env::var(ENV_REGISTRO_BORRADOS) {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&reg).map_err(|e| format!("{reg}: {e}"))?;
        writeln!(f, "{}", dir_c.display()).map_err(|e| format!("{reg}: {e}"))?;
    }
    std::fs::remove_dir_all(&dir_c).map_err(|e| format!("{}: {e}", dir_c.display()))
}

/// Expulsa con el cerrojo del manifiesto YA tomado por quien llama; reescribe el manifiesto tras cada hora.
pub fn expulsar_en(
    raiz: &Path,
    man: &mut Manifiesto,
    objetivo: u64,
    protegidas: &[String],
    motivo: &str,
) -> Result<Vec<String>, String> {
    let plan = planificar(raiz, man, objetivo, protegidas)?;
    let claves: Vec<String> = plan.iter().map(|&i| man.horas[i].clave.clone()).collect();
    let mut hechas = Vec::new();
    for clave in claves {
        let Some(i) = man.posicion(&clave) else { continue };
        let h = man.horas[i].clone();
        retirar(raiz, &h.clave)?;
        let linea = serde_json::json!({
            "ms": ahora_ms(), "clave": h.clave, "motivo": motivo, "bytes": h.bytes, "rows": h.rows,
            "sha256_indice": h.sha256_indice, "sha256_original": h.sha256_original,
            "origen_logica": h.origen_logica, "ultimo_acceso_ms": h.ultimo_acceso_ms, "pub_min_ms": h.pub_min_ms,
        });
        let p = raiz.join(NOMBRE_EVICCIONES);
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&p)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        writeln!(f, "{linea}").map_err(|e| format!("{}: {e}", p.display()))?;
        man.horas.remove(i);
        escribir_manifiesto(raiz, man)?;
        hechas.push(h.clave);
    }
    Ok(hechas)
}

/// `cache-columnas expulsar --hasta <bytes>`: la misma politica, a mano.
pub fn expulsar_hasta(raiz: &Path, objetivo: u64, motivo: &str) -> Result<Vec<String>, String> {
    let _c = Cerrojo::tomar(raiz)?;
    let mut man = leer_manifiesto(raiz)?;
    expulsar_en(raiz, &mut man, objetivo, &[], motivo)
}
