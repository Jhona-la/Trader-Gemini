// IBKR-I3 (integrador, 2026-11-16): puerta de CI. El conector IBKR es de solo
// lectura DE VERDAD: falla si aparece un metodo de orden en src/ (o en tests/,
// salvo este mismo archivo, que tiene que nombrarlos para buscarlos) fuera de
// una allowlist VACIA. Sin red ni IB Gateway.
//
// Pedido junie_cli_pedido_conector_ibkr_solo_lectura_v2.md (I3) y revision de
// QODER-CLI (seccion 4: hueco de tests/, endurecimiento aplicado aqui).
use std::fs;
use std::path::{Path, PathBuf};

/// Metodos y tipos de orden de ibapi::Client que este crate no puede usar jamas.
/// Fuente: docs.rs/ibapi/2.7.0/ibapi/struct.Client.html (2026-11-16).
/// Las dos ultimas entradas son endurecimiento del integrador: el modulo de
/// ordenes de ibapi y su prelude (que reexporta los constructores de ordenes).
const PROHIBIDOS: &[&str] = &[
    "place_order",
    "submit_order",
    "cancel_order",
    "global_cancel",
    "submit_oca_orders",
    "exercise_options",
    "next_order_id",
    "order_update_stream",
    "OrderBuilder",
    "PlaceOrder",
    "order_builder",
    "ibapi::orders",
    "ibapi::prelude",
];

/// Apariciones permitidas: NINGUNA. La lista queda vacia a proposito.
const ALLOWLIST: &[&str] = &[];

/// Este archivo: el unico que puede nombrar los tokens (los busca).
const ESTE_ARCHIVO: &str = "ibkr_solo_lectura_contract.rs";

fn fuentes_rs(raiz: &Path, salida: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(raiz) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            fuentes_rs(&p, salida);
        } else if p.extension().is_some_and(|x| x == "rs") {
            salida.push(p);
        }
    }
}

fn escanear(texto: &str, nombre: &str) -> Vec<String> {
    let mut hallados = Vec::new();
    for (i, linea) in texto.lines().enumerate() {
        for tok in PROHIBIDOS {
            if linea.contains(tok) {
                hallados.push(format!("{nombre}:{} contiene {}", i + 1, tok));
            }
        }
    }
    hallados
}

fn violaciones_en(raiz: &str, excluir: Option<&str>) -> (usize, Vec<String>) {
    let mut archivos = Vec::new();
    fuentes_rs(Path::new(raiz), &mut archivos);
    let mut violaciones = Vec::new();
    for f in &archivos {
        let rel = f.to_string_lossy().replace('\\', "/");
        if ALLOWLIST.iter().any(|a| rel.contains(a)) {
            continue;
        }
        if excluir.is_some_and(|x| f.file_name().is_some_and(|n| n == x)) {
            continue;
        }
        let texto = fs::read_to_string(f).unwrap_or_else(|e| panic!("no pude leer {rel}: {e}"));
        violaciones.extend(escanear(&texto, &rel));
    }
    (archivos.len(), violaciones)
}

#[test]
fn i3_src_no_tiene_metodos_de_orden() {
    let (n, violaciones) = violaciones_en("src", None);
    assert!(n > 0, "no se encontro src/ del crate");
    assert!(violaciones.is_empty(), "metodos de orden fuera del allowlist (vacia): {violaciones:#?}");
}

#[test]
fn i3_tests_no_tienen_metodos_de_orden() {
    let (n, violaciones) = violaciones_en("tests", Some(ESTE_ARCHIVO));
    assert!(n > 1, "no se encontro tests/ del crate");
    assert!(violaciones.is_empty(), "metodos de orden en tests/ (allowlist vacia): {violaciones:#?}");
}

#[test]
fn i3_escaner_detecta_orden_sintetica() {
    let v = escanear("let r = cliente.place_order(1, &c, &o);", "sintetico.rs");
    assert_eq!(v, vec!["sintetico.rs:1 contiene place_order"]);
    let v = escanear("use ibapi::orders::Action;", "sintetico.rs");
    assert_eq!(v, vec!["sintetico.rs:1 contiene ibapi::orders"]);
    assert!(escanear("let t = cliente.server_time().await;", "limpio.rs").is_empty());
}

#[test]
fn i3_ibapi_solo_en_modulo_ibkr() {
    let mut archivos = Vec::new();
    fuentes_rs(Path::new("src"), &mut archivos);
    assert!(archivos.iter().any(|f| f.to_string_lossy().replace('\\', "/").contains("src/ibkr/")), "no existe src/ibkr/");
    for f in archivos {
        let rel = f.to_string_lossy().replace('\\', "/");
        if rel.contains("src/ibkr/") {
            continue;
        }
        let texto = fs::read_to_string(&f).unwrap_or_else(|e| panic!("no pude leer {rel}: {e}"));
        assert!(!texto.contains("ibapi"), "ibapi fuera de src/ibkr en {rel}");
    }
}
