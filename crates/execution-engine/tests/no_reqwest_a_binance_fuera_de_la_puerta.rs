//! PUERTA-REST P5: ningún archivo `.rs` del workspace escribe un host de
//! Binance fuera de la puerta REST única (`binance_gateway.rs`), salvo los de
//! la lista blanca `tests/allowlist_rest_binance.md`, que sólo puede
//! achicarse: cada paso P6-P10 quita sus filas y su recibo trae el diff.
//!
//! Sin red: sólo lee archivos del repo.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Techo de la lista blanca: el inventario inicial de P5 (21 archivos, `rg -l`
/// de los cinco hosts en el commit de P4, fe3c23405). Bajarlo al quitar
/// filas es lo esperado; subirlo es abrir un acceso nuevo fuera de la puerta.
const TECHO_LISTA_BLANCA: usize = 21;

const LISTA_BLANCA: &str = include_str!("allowlist_rest_binance.md");

/// Los cinco hosts vigilados (pedido v3, §6.2). Se arman por partes para que
/// este archivo no sea él mismo un acceso.
fn hosts() -> Vec<String> {
    [
        ["fapi", "binance", "com"].join("."),
        ["testnet", "binancefuture", "com"].join("."),
        ["fstream", "binance", "com"].join("."),
        ["stream", "binancefuture", "com"].join("."),
        ["data", "binance", "vision"].join("."),
    ]
    .to_vec()
}

/// Carpetas del workspace que se escanean.
const CARPETAS: [&str; 5] = ["crates", "src", "tests", "benches", "examples"];

fn raiz() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn recorrer(dir: &Path, salida: &mut Vec<PathBuf>) {
    let Ok(entradas) = std::fs::read_dir(dir) else { return };
    for e in entradas.flatten() {
        let ruta = e.path();
        let nombre = e.file_name().to_string_lossy().to_string();
        if ruta.is_dir() {
            if nombre == "target" || nombre.starts_with('.') {
                continue;
            }
            recorrer(&ruta, salida);
        } else if nombre.ends_with(".rs") {
            salida.push(ruta);
        }
    }
}

/// Ruta relativa a la raíz, con `/`.
fn relativa(ruta: &Path, raiz: &Path) -> String {
    ruta.strip_prefix(raiz)
        .unwrap_or(ruta)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Los archivos `.rs` que escriben algún host vigilado.
fn accesos_directos() -> BTreeSet<String> {
    let raiz = raiz();
    let hosts = hosts();
    let mut archivos = Vec::new();
    for c in CARPETAS {
        recorrer(&raiz.join(c), &mut archivos);
    }
    assert!(!archivos.is_empty(), "control: el escaneo encontró archivos .rs");
    archivos
        .into_iter()
        .filter(|a| {
            std::fs::read_to_string(a)
                .map(|t| hosts.iter().any(|h| t.contains(h.as_str())))
                .unwrap_or(false)
        })
        .map(|a| relativa(&a, &raiz))
        .collect()
}

/// Una fila de la lista blanca: ruta, paso y deuda.
struct Fila {
    ruta: String,
    paso: String,
    deuda: String,
}

fn lista_blanca() -> Vec<Fila> {
    LISTA_BLANCA
        .replace("\r\n", "\n")
        .lines()
        .filter(|l| l.starts_with("| `"))
        .map(|l| {
            let c: Vec<&str> = l.split('|').map(str::trim).collect();
            Fila {
                ruta: c.get(1).unwrap_or(&"").trim_matches('`').to_string(),
                paso: c.get(4).unwrap_or(&"").to_string(),
                deuda: c.get(5).unwrap_or(&"").to_string(),
            }
        })
        .collect()
}

#[test]
fn no_reqwest_a_binance_fuera_de_la_puerta() {
    let accesos = accesos_directos();
    let filas = lista_blanca();
    let lista: BTreeSet<String> = filas.iter().map(|f| f.ruta.clone()).collect();

    // Control: el escaneo ve un acceso conocido.
    assert!(
        accesos.contains("crates/execution-engine/src/client.rs"),
        "control: el escaneo no ve client.rs: {accesos:?}"
    );
    // La puerta no escribe hosts: la base se le pasa.
    assert!(
        !accesos.contains("crates/execution-engine/src/binance_gateway.rs"),
        "la puerta no lleva hosts escritos a mano"
    );

    let nuevos: Vec<&String> = accesos.difference(&lista).collect();
    assert!(
        nuevos.is_empty(),
        "acceso directo a Binance fuera de la puerta (pasar por binance_gateway.rs): {nuevos:?}"
    );
    let viejos: Vec<&String> = lista.difference(&accesos).collect();
    assert!(
        viejos.is_empty(),
        "filas de la lista blanca que ya no escriben ningún host: bórralas (la lista sólo se achica): {viejos:?}"
    );

    assert_eq!(lista.len(), filas.len(), "filas repetidas en la lista blanca");
    assert!(
        filas.len() <= TECHO_LISTA_BLANCA,
        "la lista blanca creció: {} > {TECHO_LISTA_BLANCA}",
        filas.len()
    );
    for f in &filas {
        assert!(!f.paso.is_empty() && !f.deuda.is_empty(), "{}: sin paso o sin deuda", f.ruta);
    }
}
