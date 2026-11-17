//! P2 (pedido v3): el tope de la cache se DERIVA del presupuesto menos lo medido; lo que no cabe no entra.
mod comun;

use cache_columnas::construir::construir_desde_cinta;
use cache_columnas::leer_manifiesto;
use cache_columnas::presupuesto::{leer_config, medir, Limite};
use comun::*;
use std::path::Path;

/// Raiz de medicion inyectada con contenidos de tamano conocido: archivo 3000 B, proyecto 1200 B; target, .venv y
/// la cache quedan fuera.
fn raiz_inyectada(d: &Path) -> (std::path::PathBuf, std::path::PathBuf, u64, u64) {
    let r = d.join("repo");
    for (p, n) in [
        ("cinta_datos/a.tgl4", 1000usize),
        ("logs/captura-l2-20261112/x.jsonl.gz", 2000),
        ("logs/otro.log", 300),
        ("data/d.csv", 400),
        ("models/m.bin", 500),
        ("target/debug/t", 99_999),
        ("crates/x/target/t", 77_777),
        (".venv/v", 88_888),
        ("cache/columnas/ya.bin", 66_666),
    ] {
        let f = r.join(p);
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, vec![7u8; n]).unwrap();
    }
    (r.clone(), r.join("cache/columnas"), 3000, 1200)
}

/// Mismo calculo que el pedido, hecho aqui desde las claves del JSON (no desde el crate).
fn esperado(cfg: &Path, archivo: u64, proyecto: u64) -> i128 {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(cfg).unwrap()).unwrap();
    let gb = |k: &str| (v[k].as_f64().unwrap() * 1e9).round() as i128;
    gb("presupuesto_total_gb") - gb("reserva_suelo_gb") - gb("reserva_target_gb") - archivo as i128 - proyecto as i128
}

fn config(d: &Path, total: f64, suelo: f64, target: f64) -> std::path::PathBuf {
    let p = d.join(format!("cinta_config_{total}.json"));
    let j = serde_json::json!({
        "presupuesto_total_gb": total, "fuente_presupuesto_total": "prueba",
        "reserva_suelo_gb": suelo, "reserva_target_gb": target, "captura_gib_dia": 1.929, "ventana_dias": 365
    });
    std::fs::write(&p, serde_json::to_vec(&j).unwrap()).unwrap();
    p
}

#[test]
fn formula_reproduce_sumandos() {
    let d = carpeta("p2_formula");
    let (raiz, cache, archivo, proyecto) = raiz_inyectada(&d);
    // Con TG_CINTA_CONFIG se usa el cinta_config.json real; sin ella, valores raros que delatan cualquier constante
    // escondida (ninguna cifra del presupuesto se copia a esta prueba).
    let cfg = match std::env::var("TG_CINTA_CONFIG") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => config(&d, 0.000_731_9, 0.000_011_3, 0.000_007_7),
    };
    let s = medir(&leer_config(&cfg).unwrap(), &raiz, &cache).unwrap();
    assert_eq!(s.archivo.bytes, archivo, "{:?}", s.archivo.rutas);
    assert_eq!(s.proyecto.bytes, proyecto);
    assert_eq!(s.limite_b, esperado(&cfg, archivo, proyecto));
    assert!(s.excluidas.iter().any(|e| e.ends_with(".venv")));
    assert_eq!(s.excluidas.iter().filter(|e| e.ends_with("target")).count(), 2);
    // Falta una clave -> error, nunca un valor por defecto.
    let roto = d.join("roto.json");
    std::fs::write(&roto, br#"{"presupuesto_total_gb": 200, "reserva_suelo_gb": 40}"#).unwrap();
    assert!(leer_config(&roto).is_err());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn limite_derivado_rechaza_lo_que_no_cabe() {
    let d = carpeta("p2_rechazo");
    let (raiz, cache, archivo, proyecto) = raiz_inyectada(&d);
    std::fs::create_dir_all(d.join("cintas")).unwrap();
    // 10 000 - 1000 - 1000 - 3000 - 1200 = 3800 B de hueco.
    let cfg = config(&d, 0.000_01, 0.000_001, 0.000_001);
    let hueco = esperado(&cfg, archivo, proyecto);
    assert_eq!(hueco, 3800);
    let op = |s: &str| {
        let mut o = opciones(s);
        o.limite = Some(Limite { config: leer_config(&cfg).unwrap(), raiz_medicion: raiz.clone() });
        o
    };
    // Una hora mas grande que el hueco: rechazada con los sumandos, y la cache sin la hora.
    let grande = d.join("cintas/G_REAL.bin");
    escribir(&grande, &cinta(DIA_BASE, 500, 1_000, 1));
    let e = construir_desde_cinta(&cache, &grande, &op("GUSDT")).unwrap_err();
    assert!(e.contains("no cabe") && e.contains("limite_cache") && e.contains("3800"), "{e}");
    assert!(leer_manifiesto(&cache).unwrap().horas.is_empty());
    assert!(!listar(&cache).iter().any(|p| p.contains("GUSDT") || p.contains(".tmp-")));
    // Una hora pequena cabe; la segunda ya no (el uso de la cache cuenta).
    let p1 = d.join("cintas/P1_REAL.bin");
    let p2 = d.join("cintas/P2_REAL.bin");
    escribir(&p1, &cinta(DIA_BASE, 20, 1_000, 2));
    escribir(&p2, &cinta(DIA_BASE, 20, 1_000, 3));
    let i1 = construir_desde_cinta(&cache, &p1, &op("PAUSDT")).unwrap();
    assert!(i1.bytes_nuevos as i128 <= hueco && 2 * i1.bytes_nuevos as i128 > hueco, "{}", i1.bytes_nuevos);
    assert!(construir_desde_cinta(&cache, &p2, &op("PBUSDT")).unwrap_err().contains("no cabe"));
    assert_eq!(leer_manifiesto(&cache).unwrap().horas.len(), 1);
    // Tope <= 0: no se construye NADA, ni lo que cabria.
    let cfg0 = config(&d, 0.000_002, 0.000_001, 0.000_001);
    let mut o = opciones("CEROUSDT");
    o.limite = Some(Limite { config: leer_config(&cfg0).unwrap(), raiz_medicion: raiz.clone() });
    let otra = d.join("cache_cero");
    assert!(construir_desde_cinta(&otra, &p1, &o).unwrap_err().contains("<= 0"));
    assert!(leer_manifiesto(&otra).unwrap().horas.is_empty());
    let _ = std::fs::remove_dir_all(&d);
}
