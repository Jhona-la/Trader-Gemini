//! `cache-columnas construir|verificar|medir` (pedido 20261116-2345-cachev3). Ver `src/lib.rs`.

use cache_columnas::expulsion::expulsar_hasta;
use cache_columnas::lector::{leer, Consulta};
use cache_columnas::construir::{construir_desde_cinta, simbolo_de_ruta, Opciones};
use cache_columnas::presupuesto::{leer_config, medir, Limite, BYTES_POR_GB};
use cache_columnas::verificar::verificar;
use cache_columnas::{leer_manifiesto, RUTA_POR_DEFECTO};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USO: &str = "uso:
  cache-columnas construir <cinta.bin> --config cinta_config.json [--raiz-medicion DIR] [--cache DIR]
                           [--fuente tgb-reales] [--simbolo S] [--sha256 HEX]
  cache-columnas verificar [--cache DIR] [--origen]
  cache-columnas expulsar --hasta BYTES [--cache DIR]
  cache-columnas leer --simbolo S --desde PUB_MS --hasta PUB_MS [--fuente F] [--cache DIR]
  cache-columnas medir --config cinta_config.json [--raiz-medicion DIR] [--cache DIR]
La cinta se lee por la cinta logica de tick-tape: X.bin, o X.bin.tgb si X.bin no existe (con las dos manda el .bin).
--raiz-medicion: raiz del proyecto que se mide para el tope (por defecto, la carpeta actual).";

fn valor(args: &[String], nombre: &str) -> Option<String> {
    args.iter().position(|a| a == nombre).and_then(|p| args.get(p + 1)).cloned()
}

fn limite(args: &[String]) -> Result<Limite, String> {
    let config = valor(args, "--config").ok_or("falta --config (el tope de la cache se deriva de cinta_config.json)")?;
    Ok(Limite {
        config: leer_config(Path::new(&config))?,
        raiz_medicion: PathBuf::from(valor(args, "--raiz-medicion").unwrap_or_else(|| ".".to_string())),
    })
}

fn principal(args: &[String]) -> Result<(), String> {
    let orden = args.first().ok_or(USO)?;
    let cache = PathBuf::from(valor(args, "--cache").unwrap_or_else(|| RUTA_POR_DEFECTO.to_string()));
    match orden.as_str() {
        "construir" => {
            let cinta = args.get(1).filter(|a| !a.starts_with("--")).ok_or(USO)?;
            let logica = Path::new(cinta);
            let simbolo = match valor(args, "--simbolo") {
                Some(s) => s,
                None => simbolo_de_ruta(logica).ok_or("no pude deducir el simbolo del nombre: usa --simbolo")?,
            };
            let op = Opciones {
                fuente: valor(args, "--fuente").unwrap_or_else(|| "tgb-reales".to_string()),
                simbolo,
                sha256_esperado: valor(args, "--sha256"),
                limite: Some(limite(args)?),
                ..Opciones::default()
            };
            let inf = construir_desde_cinta(&cache, logica, &op)?;
            if let Some(s) = &inf.sumandos {
                println!("{s}");
            }
            println!(
                "CONSTRUIR {} -> {} | horas nuevas {} | horas ya presentes {} | filas {} | bytes nuevos {}",
                cinta,
                cache.display(),
                inf.horas_nuevas.len(),
                inf.horas_presentes.len(),
                inf.rows,
                inf.bytes_nuevos
            );
            Ok(())
        }
        "verificar" => {
            let inf = verificar(&cache, args.iter().any(|a| a == "--origen"))?;
            for h in &inf.hallazgos {
                println!("HALLAZGO {}: {}", h.clave, h.problema);
            }
            println!(
                "VERIFICAR {} | horas {} | filas {} | origenes comprobados {} | hallazgos {}",
                cache.display(),
                inf.horas,
                inf.rows,
                inf.origenes_comprobados,
                inf.hallazgos.len()
            );
            if inf.hallazgos.is_empty() {
                Ok(())
            } else {
                Err("verificacion con hallazgos".into())
            }
        }
        "expulsar" => {
            let hasta: u64 = valor(args, "--hasta")
                .ok_or("falta --hasta <bytes>")?
                .parse()
                .map_err(|e| format!("--hasta: {e}"))?;
            let hechas = expulsar_hasta(&cache, hasta, "a mano (cache-columnas expulsar)")?;
            for c in &hechas {
                println!("EXPULSADA {c}");
            }
            println!("EXPULSAR {} | horas expulsadas {} | uso {} B", cache.display(), hechas.len(), leer_manifiesto(&cache)?.uso_bytes());
            Ok(())
        }
        "leer" => {
            let num = |n: &str| -> Result<i64, String> {
                valor(args, n).ok_or(format!("falta {n}"))?.parse().map_err(|e| format!("{n}: {e}"))
            };
            let consulta = Consulta {
                series: vec![(
                    valor(args, "--fuente").unwrap_or_else(|| "tgb-reales".to_string()),
                    valor(args, "--simbolo").ok_or("falta --simbolo")?,
                )],
                desde_pub_ms: num("--desde")?,
                hasta_pub_ms: num("--hasta")?,
            };
            let e = leer(&cache, &consulta)?;
            let primera = e.filas.first().map(|f| f.pub_ms);
            let ultima = e.filas.last().map(|f| f.pub_ms);
            println!(
                "LEER {} | horas montadas {} | filas {} | pub_ms primera {:?} ultima {:?} (rango sobre pub_ms, no sobre el evento)",
                cache.display(),
                e.horas.len(),
                e.filas.len(),
                primera,
                ultima
            );
            Ok(())
        }
        "medir" => {            let l = limite(args)?;
            let s = medir(&l.config, &l.raiz_medicion, &cache)?;
            println!("{}", s.texto(leer_manifiesto(&cache)?.uso_bytes()));
            println!("MEDIR limite_cache_bytes {} ({:.3} GB decimales)", s.limite_b, s.limite_b as f64 / BYTES_POR_GB);
            Ok(())
        }
        _ => Err(USO.into()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match principal(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cache-columnas: {e}");
            ExitCode::from(2)
        }
    }
}
