//! P3 (pedido v3): expulsion LRU por uso real, sin tocar las fuentes, y reconstruccion identica.
mod comun;

use cache_columnas::construir::construir_desde_cinta;
use cache_columnas::lector::montar;
use cache_columnas::presupuesto::{leer_config, Limite};
use cache_columnas::verificar::verificar;
use cache_columnas::{leer_manifiesto, Indice};
use comun::*;

#[test]
fn expulsion_lru_reconstruye_identico() {
    let d = carpeta("p3_lru");
    let (cintas, cache, raiz) = (d.join("cintas"), d.join("cache"), d.join("repo_vacio"));
    std::fs::create_dir_all(&cintas).unwrap();
    std::fs::create_dir_all(&raiz).unwrap();
    // Cuatro cintas de una hora, mismo tamano de hora (simbolos del mismo largo).
    let mut rutas = Vec::new();
    for (k, s) in ["AUSDT", "BUSDT", "CUSDT", "DUSDT"].iter().enumerate() {
        let r = cintas.join(format!("{s}_REAL.bin"));
        escribir(&r, &cinta(DIA_BASE + (k as u64 + 1) * HORA, 200, 1_000, k as u64));
        rutas.push(r);
    }
    let sha_fuentes = || rutas.iter().map(|r| sha(&std::fs::read(r).unwrap())).collect::<Vec<_>>();
    let antes = sha_fuentes();

    // Tamano de una hora (sin tope) y una cache con el tope justo para tres.
    let sonda = d.join("sonda");
    let hb = construir_desde_cinta(&sonda, &rutas[0], &opciones("AUSDT")).unwrap().bytes_nuevos;
    let cfg = d.join("cinta_config.json");
    let j = serde_json::json!({"presupuesto_total_gb": (3 * hb) as f64 / 1e9, "reserva_suelo_gb": 0, "reserva_target_gb": 0});
    std::fs::write(&cfg, serde_json::to_vec(&j).unwrap()).unwrap();
    let op = |s: &str| {
        let mut o = opciones(s);
        o.limite = Some(Limite { config: leer_config(&cfg).unwrap(), raiz_medicion: raiz.clone() });
        o
    };
    let mut claves = Vec::new();
    for (k, s) in ["AUSDT", "BUSDT", "CUSDT"].iter().enumerate() {
        let i = construir_desde_cinta(&cache, &rutas[k], &op(s)).unwrap();
        assert_eq!(i.bytes_nuevos, hb);
        claves.push(i.horas_nuevas[0].clone());
    }
    assert_eq!(leer_manifiesto(&cache).unwrap().uso_bytes(), 3 * hb);
    let dir_c = cache.join(&claves[2]);
    let idx_c = std::fs::read(dir_c.join("index.json")).unwrap();
    let cols_c: Vec<Vec<u8>> = serde_json::from_slice::<Indice>(&idx_c)
        .unwrap()
        .columnas
        .iter()
        .map(|c| std::fs::read(dir_c.join(&c.file)).unwrap())
        .collect();

    // El lector sirve A y B (marca su acceso); C nunca se leyo.
    drop(montar(&cache, &claves[..2]).unwrap());
    let man = leer_manifiesto(&cache).unwrap();
    assert!(man.horas.iter().filter(|h| h.clave != claves[2]).all(|h| h.ultimo_acceso_ms.is_some()));

    // D no cabe: se expulsa C (la no accedida) y NO A ni B.
    let i = construir_desde_cinta(&cache, &rutas[3], &op("DUSDT")).unwrap();
    let clave_d = i.horas_nuevas[0].clone();
    let man = leer_manifiesto(&cache).unwrap();
    let mut presentes: Vec<String> = man.horas.iter().map(|h| h.clave.clone()).collect();
    presentes.sort();
    let mut esperadas = vec![claves[0].clone(), claves[1].clone(), clave_d.clone()];
    esperadas.sort();
    assert_eq!(presentes, esperadas);
    assert!(!dir_c.exists());
    let ev = std::fs::read_to_string(cache.join("EVICCIONES.jsonl")).unwrap();
    let lineas: Vec<serde_json::Value> = ev.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lineas.len(), 1);
    assert_eq!(lineas[0]["clave"], claves[2].as_str());
    assert_eq!(sha_fuentes(), antes, "la expulsion no toca las fuentes");

    // Volver a pedir C: se reconstruye (expulsando D, la otra nunca leida) con los MISMOS SHA-256.
    let i = construir_desde_cinta(&cache, &rutas[2], &op("CUSDT")).unwrap();
    assert_eq!(i.horas_nuevas, vec![claves[2].clone()]);
    assert_eq!(std::fs::read(dir_c.join("index.json")).unwrap(), idx_c, "index.json identico");
    let idx: Indice = serde_json::from_slice(&idx_c).unwrap();
    for (c, b) in idx.columnas.iter().zip(&cols_c) {
        assert_eq!(&std::fs::read(dir_c.join(&c.file)).unwrap(), b, "columna {} identica", c.file);
    }
    let man = leer_manifiesto(&cache).unwrap();
    assert!(man.posicion(&clave_d).is_none() && man.posicion(&claves[0]).is_some() && man.posicion(&claves[1]).is_some());
    assert!(verificar(&cache, true).unwrap().hallazgos.is_empty());
    assert_eq!(sha_fuentes(), antes);
    let _ = std::fs::remove_dir_all(&d);
}

/// Una cinta de 3 horas con sitio para 2 y media: la tercera no expulsa a las dos primeras de la MISMA construccion
/// (eso seria girar en vacio); no se construye y lo ya construido queda.
#[test]
fn expulsion_no_se_come_la_construccion_en_curso() {
    let d = carpeta("p3_en_curso");
    let (cache, raiz) = (d.join("cache"), d.join("repo_vacio"));
    std::fs::create_dir_all(&raiz).unwrap();
    let r = d.join("TRESUSDT_REAL.bin");
    escribir(&r, &cinta(DIA_BASE, 3 * 36, 100_000, 7));
    construir_desde_cinta(&d.join("sonda"), &r, &opciones("TRESUSDT")).unwrap();
    let b: Vec<u64> = leer_manifiesto(&d.join("sonda")).unwrap().horas.iter().map(|h| h.bytes).collect();
    assert_eq!(b.len(), 3);
    let cfg = d.join("cinta_config.json");
    let tope = b[0] + b[1] + b[2] / 2;
    let j = serde_json::json!({"presupuesto_total_gb": tope as f64 / 1e9, "reserva_suelo_gb": 0, "reserva_target_gb": 0});
    std::fs::write(&cfg, serde_json::to_vec(&j).unwrap()).unwrap();
    let mut o = opciones("TRESUSDT");
    o.limite = Some(Limite { config: leer_config(&cfg).unwrap(), raiz_medicion: raiz.clone() });
    let e = construir_desde_cinta(&cache, &r, &o).unwrap_err();
    assert!(e.contains("no cabe"), "{e}");
    let man = leer_manifiesto(&cache).unwrap();
    assert_eq!(man.horas.len(), 2, "las dos primeras horas de la cinta siguen");
    assert!(man.horas.iter().all(|h| h.registro_desde < 72));
    assert!(!cache.join("EVICCIONES.jsonl").exists());
    let _ = std::fs::remove_dir_all(&d);
}
