//! `cache-columnas construir|verificar` (pedido 20261116-2345-cachev3). Ver `src/lib.rs`.

use cache_columnas::construir::{construir_desde_cinta, simbolo_de_ruta, Opciones};
use cache_columnas::verificar::verificar;
use cache_columnas::RUTA_POR_DEFECTO;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USO: &str = "uso:
  cache-columnas construir <cinta.bin> [--cache DIR] [--fuente tgb-reales] [--simbolo S] [--sha256 HEX]
  cache-columnas verificar [--cache DIR] [--origen]
La cinta se lee por la cinta logica de tick-tape: X.bin, o X.bin.tgb si X.bin no existe (con las dos manda el .bin).";

fn valor(args: &[String], nombre: &str) -> Option<String> {
    args.iter().position(|a| a == nombre).and_then(|p| args.get(p + 1)).cloned()
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
                ..Opciones::default()
            };
            let inf = construir_desde_cinta(&cache, logica, &op)?;
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
