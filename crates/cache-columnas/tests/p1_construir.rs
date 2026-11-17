//! P1 (pedido v3): directorio de cache por hora, construido atomicamente y verificado por SHA-256.
mod comun;

use cache_columnas::construir::construir_desde_cinta;
use cache_columnas::verificar::verificar;
use cache_columnas::{leer_manifiesto, Indice, MS_POR_DIA};
use comun::*;

#[test]
fn cache_una_hora_se_construye_y_verifica() {
    let d = carpeta("p1_ok");
    let (cintas, cache) = (d.join("cintas"), d.join("cache"));
    std::fs::create_dir_all(&cintas).unwrap();
    let a = cinta(DIA_BASE + 5 * HORA + 1_000, 500, 1_000, 1);
    let ruta = cintas.join("BTCUSDT_2026-09-14_REAL.bin");
    escribir(&ruta, &a);

    let inf = construir_desde_cinta(&cache, &ruta, &opciones("BTCUSDT")).unwrap();
    assert_eq!(inf.horas_nuevas, vec!["tgb-reales/BTCUSDT/2026-09-14/05".to_string()]);
    assert_eq!(inf.rows, 500);

    let man = leer_manifiesto(&cache).unwrap();
    assert_eq!(man.horas.len(), 1);
    let e = &man.horas[0];
    assert_eq!(e.sha256_original, sha(&a));
    assert_eq!(e.representacion, "bin");
    let dir = cache.join(&e.clave);
    let idx_bytes = std::fs::read(dir.join("index.json")).unwrap();
    assert_eq!(sha(&idx_bytes), e.sha256_indice, "SHA del indice recalculado contra el manifiesto");
    let idx: Indice = serde_json::from_slice(&idx_bytes).unwrap();
    let mut cols = Vec::new();
    for c in &idx.columnas {
        let b = std::fs::read(dir.join(&c.file)).unwrap();
        assert_eq!(sha(&b), c.sha256, "SHA recalculado de {}", c.file);
        assert_eq!(b.len() as u64, c.bytes);
        assert_eq!(c.rows, 500);
        cols.push(b);
    }
    // Bit a bit: intercalar las 5 columnas del registro devuelve los registros del original.
    let refs: Vec<&[u8]> = cols.iter().take(5).map(|v| v.as_slice()).collect();
    assert_eq!(cache_columnas::registros_desde_columnas(&refs, 500).unwrap(), a[8..].to_vec());
    // pub_ms de tgb-reales: el dia D se publica al empezar D+1.
    assert!(cols[5].chunks(8).all(|w| i64::from_le_bytes(w.try_into().unwrap()) == (DIA_BASE + MS_POR_DIA) as i64));
    let v = verificar(&cache, true).unwrap();
    assert_eq!((v.horas, v.origenes_comprobados, v.hallazgos.len()), (1, 1, 0), "{:?}", v.hallazgos);

    // La misma cinta solo como .bin.tgb da el MISMO indice (misma estructura y SHA por columna).
    let cache_tgb = d.join("cache_tgb");
    let ruta_t = cintas.join("T_REAL.bin");
    escribir(&cintas.join("T_REAL.bin.tgb"), &tick_tape::tgb::empaquetar(&a).unwrap());
    construir_desde_cinta(&cache_tgb, &ruta_t, &opciones("BTCUSDT")).unwrap();
    let mt = leer_manifiesto(&cache_tgb).unwrap();
    assert_eq!(mt.horas[0].representacion, "tgb");
    assert!(mt.horas[0].sha256_contenedor.is_some());
    assert_eq!(mt.horas[0].sha256_indice, e.sha256_indice);

    // Con .bin y .bin.tgb distintos manda el .bin.
    let cache_ambos = d.join("cache_ambos");
    let b = cinta(DIA_BASE + 5 * HORA, 50, 10, 9);
    let ruta_y = cintas.join("Y_REAL.bin");
    escribir(&ruta_y, &a);
    escribir(&cintas.join("Y_REAL.bin.tgb"), &tick_tape::tgb::empaquetar(&b).unwrap());
    construir_desde_cinta(&cache_ambos, &ruta_y, &opciones("BTCUSDT")).unwrap();
    assert_eq!(leer_manifiesto(&cache_ambos).unwrap().horas[0].sha256_original, sha(&a));

    // Una cinta de 3 horas da 3 horas; reconstruir no cambia nada (horas ya presentes).
    let tres = cinta(DIA_BASE + 22 * HORA, 3 * 360, 10_000, 2);
    let ruta3 = cintas.join("ETHUSDT_REAL.bin");
    escribir(&ruta3, &tres);
    let i3 = construir_desde_cinta(&cache, &ruta3, &opciones("ETHUSDT")).unwrap();
    assert_eq!(i3.horas_nuevas.len(), 3);
    let i3b = construir_desde_cinta(&cache, &ruta3, &opciones("ETHUSDT")).unwrap();
    assert_eq!((i3b.horas_nuevas.len(), i3b.horas_presentes.len()), (0, 3));
    assert!(verificar(&cache, true).unwrap().hallazgos.is_empty());

    // verificar detecta un bit cambiado en una columna.
    let bid = dir.join("bid.f64le");
    let mut x = std::fs::read(&bid).unwrap();
    x[3] ^= 1;
    escribir(&bid, &x);
    let v = verificar(&cache, false).unwrap();
    assert_eq!(v.hallazgos.len(), 1);
    assert_eq!(v.hallazgos[0].clave, e.clave);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn hora_fallida_no_deja_partials() {
    let d = carpeta("p1_fallo");
    let (cintas, cache) = (d.join("cintas"), d.join("cache"));
    std::fs::create_dir_all(&cintas).unwrap();
    let sin_rastro = |cache: &std::path::Path| {
        let l = listar(cache);
        assert!(!l.iter().any(|p| p.contains(".tmp-")), "temporal colgante: {l:?}");
        assert!(!l.iter().any(|p| p.ends_with("MANIFIESTO.lock")), "cerrojo colgante: {l:?}");
        l
    };

    // (a) Contenedor con el SHA del original alterado (fixture de tick-tape): no entra nada.
    escribir(
        &cintas.join("Z_REAL.bin.tgb"),
        include_bytes!("../../tick-tape/tests/fixtures/f1_real_sha_original_mal.bin.tgb"),
    );
    assert!(construir_desde_cinta(&cache, &cintas.join("Z_REAL.bin"), &opciones("ZUSDT")).is_err());
    assert!(!sin_rastro(&cache).iter().any(|p| p.starts_with("tgb-reales")));

    // (b) Original distinto del exigido.
    let a = cinta(DIA_BASE, 100, 1_000, 3);
    let ruta = cintas.join("A_REAL.bin");
    escribir(&ruta, &a);
    let mut op = opciones("AUSDT");
    op.sha256_esperado = Some(sha(b"otro"));
    assert!(construir_desde_cinta(&cache, &ruta, &op).is_err());
    assert!(!sin_rastro(&cache).iter().any(|p| p.starts_with("tgb-reales")));

    // (c) Fallo a mitad de la segunda hora: la primera queda publicada y verificada, la segunda y la tercera no.
    let tres = cinta(DIA_BASE + 10 * HORA, 3 * 36, 100_000, 4);
    let ruta3 = cintas.join("B_REAL.bin");
    escribir(&ruta3, &tres);
    let mut op = opciones("BUSDT");
    op.fallo_inyectado_en_hora = Some(1);
    assert!(construir_desde_cinta(&cache, &ruta3, &op).is_err());
    let l = sin_rastro(&cache);
    assert!(l.contains(&"tgb-reales/BUSDT/2026-09-14/10".to_string()));
    assert!(!l.iter().any(|p| p.contains("2026-09-14/11") || p.contains("2026-09-14/12")), "{l:?}");
    let man = leer_manifiesto(&cache).unwrap();
    assert_eq!(man.horas.len(), 1);
    assert!(verificar(&cache, true).unwrap().hallazgos.is_empty());

    // (d) Registro parcial y ts decreciente: rechazados sin rastro.
    let mut parcial = a.clone();
    parcial.push(0);
    escribir(&cintas.join("P_REAL.bin"), &parcial);
    assert!(construir_desde_cinta(&cache, &cintas.join("P_REAL.bin"), &opciones("PUSDT")).is_err());
    let mut desorden = cinta(DIA_BASE, 2, 1_000, 5);
    let (r0, r1) = (8..48, 48..88);
    let (x, y) = (desorden[r0.clone()].to_vec(), desorden[r1.clone()].to_vec());
    desorden[r0].copy_from_slice(&y);
    desorden[r1].copy_from_slice(&x);
    escribir(&cintas.join("D_REAL.bin"), &desorden);
    assert!(construir_desde_cinta(&cache, &cintas.join("D_REAL.bin"), &opciones("DUSDT")).is_err());
    let l = sin_rastro(&cache);
    assert!(!l.iter().any(|p| p.contains("PUSDT") || p.contains("DUSDT")));
    assert_eq!(leer_manifiesto(&cache).unwrap().horas.len(), 1);
    let _ = std::fs::remove_dir_all(&d);
}
