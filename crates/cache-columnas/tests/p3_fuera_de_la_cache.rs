//! P3 (pedido v3): la expulsion nunca sale de la cache. Binario de prueba propio: el registro de borrados se
//! instrumenta por la variable de entorno declarada (`TG_CACHE_COLUMNAS_REGISTRO_BORRADOS`), que es del proceso.
mod comun;

use cache_columnas::construir::construir_desde_cinta;
use cache_columnas::expulsion::{expulsar_hasta, ENV_REGISTRO_BORRADOS};
use cache_columnas::lector::montar;
use cache_columnas::{escribir_manifiesto, leer_manifiesto};
use comun::*;

#[test]
fn expulsion_nunca_sale_de_la_cache() {
    let d = carpeta("p3_fuera");
    let (cintas, cache, fuera) = (d.join("cintas"), d.join("cache"), d.join("fuera"));
    std::fs::create_dir_all(&cintas).unwrap();
    std::fs::create_dir_all(&fuera).unwrap();
    escribir(&fuera.join("modelo.bin"), b"no se toca");
    let registro = d.join("borrados.txt");
    std::env::set_var(ENV_REGISTRO_BORRADOS, &registro);

    let mut claves = Vec::new();
    for (k, s) in ["AUSDT", "BUSDT", "CUSDT"].iter().enumerate() {
        let r = cintas.join(format!("{s}_REAL.bin"));
        escribir(&r, &cinta(DIA_BASE + k as u64 * HORA, 100, 1_000, k as u64));
        claves.push(construir_desde_cinta(&cache, &r, &opciones(s)).unwrap().horas_nuevas[0].clone());
    }
    let hb = leer_manifiesto(&cache).unwrap().horas[0].bytes;

    // Un manifiesto alterado que apunta fuera de la cache: no se borra nada.
    let original = leer_manifiesto(&cache).unwrap();
    let mut alterado = original.clone();
    let mut mala = alterado.horas[0].clone();
    mala.clave = "../fuera".into();
    alterado.horas.insert(0, mala);
    escribir_manifiesto(&cache, &alterado).unwrap();
    assert!(expulsar_hasta(&cache, 0, "prueba").is_err());
    assert!(fuera.join("modelo.bin").exists());
    assert_eq!(listar(&cache).iter().filter(|p| p.ends_with("index.json")).count(), 3);
    escribir_manifiesto(&cache, &original).unwrap();

    // Una hora en lectura no se expulsa; si el objetivo exige tocarla, no se expulsa nada.
    let en_lectura = montar(&cache, &claves[..1]).unwrap();
    assert!(expulsar_hasta(&cache, 0, "prueba").is_err());
    assert_eq!(leer_manifiesto(&cache).unwrap().horas.len(), 3);
    let hechas = expulsar_hasta(&cache, hb, "prueba").unwrap();
    assert_eq!(hechas.len(), 2);
    assert!(!hechas.contains(&claves[0]));
    assert!(cache.join(&claves[0]).join("index.json").exists());
    drop(en_lectura);

    // Todo lo borrado esta bajo la cache y figura en EVICCIONES.jsonl.
    let raiz_c = std::fs::canonicalize(&cache).unwrap();
    let borradas: Vec<String> = std::fs::read_to_string(&registro).unwrap().lines().map(String::from).collect();
    assert_eq!(borradas.len(), 2);
    let ev = std::fs::read_to_string(cache.join("EVICCIONES.jsonl")).unwrap();
    let ev: Vec<serde_json::Value> = ev.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    for b in &borradas {
        let p = std::path::Path::new(b);
        assert!(p.starts_with(&raiz_c) && p != raiz_c, "{b} fuera de {}", raiz_c.display());
        assert!(
            ev.iter().any(|e| raiz_c.join(e["clave"].as_str().unwrap()) == p.to_path_buf()),
            "{b} no figura en EVICCIONES.jsonl"
        );
    }
    assert!(fuera.join("modelo.bin").exists());
    std::env::remove_var(ENV_REGISTRO_BORRADOS);
    let _ = std::fs::remove_dir_all(&d);
}
