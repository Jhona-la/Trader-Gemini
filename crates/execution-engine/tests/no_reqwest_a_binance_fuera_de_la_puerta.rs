//! PUERTA-REST P5: ningún archivo `.rs` del workspace escribe un host de
//! Binance fuera de la puerta REST única (`binance_gateway.rs`), salvo los de
//! la lista blanca `tests/allowlist_rest_binance.md`, que sólo puede
//! achicarse: cada paso P6-P10 quita sus filas y su recibo trae el diff.
//!
//! Sin red: sólo lee archivos del repo.
//!
//! C4 (dictamen R4): que la lista sólo se achique se impone por IDENTIDAD, no
//! por tamaño. `tests/allowlist_rest_binance_aprobaciones.md` es la cadena de
//! aprobaciones A0, A1, ...: A0 es el inventario de P5 (fijado por su huella
//! SHA-256), cada Ai es subconjunto de A(i-1) y declara su diff («Bajas»), y
//! la lista vigente es exactamente la última aprobación. Así fallan la
//! sustitución a igual tamaño (alta de una ruta nueva por baja de otra) y el
//! reingreso tras reducir. Límite: es un escaneo de literales por archivo; un
//! archivo de la lista puede sumar llamadas y un host armado por partes no se
//! ve. No aísla el transporte.

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Techo de la lista blanca: el inventario inicial de P5 (21 archivos, `rg -l`
/// de los cinco hosts en el commit de P4, fe3c23405). Bajarlo al quitar
/// filas es lo esperado; subirlo es abrir un acceso nuevo fuera de la puerta.
const TECHO_LISTA_BLANCA: usize = 21;

const LISTA_BLANCA: &str = include_str!("allowlist_rest_binance.md");

const APROBACIONES: &str = include_str!("allowlist_rest_binance_aprobaciones.md");

/// C4: huella SHA-256 de A0 (las 21 rutas del inventario de P5, da829673b,
/// ordenadas por bytes y unidas con `\n`). Cambiarla es reabrir el
/// inventario inicial: un cambio visible en el diff, que necesita su recibo.
const HUELLA_A0: &str = "c046e2cea96a8d0b24c9d6301fdb4af1ccd6e5c42d32a9dd9eae6ff9d0227d83";

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

    // C4: la lista vigente contra la cadena de aprobaciones, por identidad.
    let aprobaciones = leer_aprobaciones(APROBACIONES);
    assert!(!aprobaciones.is_empty(), "sin aprobaciones");
    assert_eq!(
        huella(&aprobaciones[0].rutas),
        HUELLA_A0,
        "A0 no es el inventario de P5: {:?}",
        aprobaciones[0].rutas
    );
    let veredicto = verificar_aprobaciones(&aprobaciones, &lista);
    assert!(veredicto.is_ok(), "{}", veredicto.err().unwrap_or_default());
}

// ---------------------------------------------------------------- C4 ----

/// Una aprobación de la cadena: su nombre, su recibo, las bajas que declara
/// frente a la anterior y el conjunto aprobado.
#[derive(Clone, Debug)]
struct Aprobacion {
    nombre: String,
    recibo: String,
    bajas: Option<BTreeSet<String>>,
    rutas: BTreeSet<String>,
}

/// Lee los bloques `## Ai ...` con sus líneas `Recibo:`, `Bajas:` (rutas
/// entre comillas invertidas, o `ninguna`) y `` - `ruta` ``.
fn leer_aprobaciones(texto: &str) -> Vec<Aprobacion> {
    let texto = texto.replace("\r\n", "\n");
    let mut salida: Vec<Aprobacion> = Vec::new();
    for l in texto.lines() {
        if let Some(nombre) = l.strip_prefix("## ") {
            salida.push(Aprobacion {
                nombre: nombre.trim().to_string(),
                recibo: String::new(),
                bajas: None,
                rutas: BTreeSet::new(),
            });
            continue;
        }
        let Some(actual) = salida.last_mut() else { continue };
        if let Some(r) = l.strip_prefix("Recibo:") {
            actual.recibo = r.trim().to_string();
        } else if let Some(b) = l.strip_prefix("Bajas:") {
            actual.bajas = Some(
                b.split('`')
                    .skip(1)
                    .step_by(2)
                    .map(str::to_string)
                    .collect(),
            );
        } else if let Some(r) = l.strip_prefix("- `") {
            actual.rutas.insert(r.trim_end().trim_end_matches('`').to_string());
        }
    }
    salida
}

/// SHA-256 en hexadecimal de las rutas ordenadas, unidas con `\n`.
fn huella(rutas: &BTreeSet<String>) -> String {
    let unidas = rutas.iter().cloned().collect::<Vec<_>>().join("\n");
    Sha256::digest(unidas.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn lista_de(c: &BTreeSet<String>) -> Vec<&String> {
    c.iter().collect()
}

/// La regla C4: cada aprobación es subconjunto de la anterior (sin altas ni
/// reingresos), declara exactamente sus bajas y trae su recibo; la lista
/// vigente es exactamente la última aprobación. El error lleva el diff.
fn verificar_aprobaciones(aprobaciones: &[Aprobacion], vigente: &BTreeSet<String>) -> Result<(), String> {
    let Some(ultima) = aprobaciones.last() else {
        return Err("sin aprobaciones".to_string());
    };
    let mut vistas: BTreeSet<String> = BTreeSet::new();
    for (i, a) in aprobaciones.iter().enumerate() {
        if a.recibo.is_empty() {
            return Err(format!("{}: sin recibo", a.nombre));
        }
        if i > 0 {
            let previa = &aprobaciones[i - 1];
            let altas: BTreeSet<String> = a.rutas.difference(&previa.rutas).cloned().collect();
            if !altas.is_empty() {
                let reingresos: BTreeSet<String> = altas.intersection(&vistas).cloned().collect();
                return Err(format!(
                    "{}: no es subconjunto de {}: altas {:?}, reingresos {:?}",
                    a.nombre,
                    previa.nombre,
                    lista_de(&altas),
                    lista_de(&reingresos)
                ));
            }
            let bajas: BTreeSet<String> = previa.rutas.difference(&a.rutas).cloned().collect();
            if a.bajas.as_ref() != Some(&bajas) {
                return Err(format!(
                    "{}: el diff declarado no coincide: declara {:?}, son {:?}",
                    a.nombre, a.bajas, lista_de(&bajas)
                ));
            }
        }
        vistas.extend(a.rutas.iter().cloned());
    }
    let altas: BTreeSet<String> = vigente.difference(&ultima.rutas).cloned().collect();
    let bajas: BTreeSet<String> = ultima.rutas.difference(vigente).cloned().collect();
    if altas.is_empty() && bajas.is_empty() {
        return Ok(());
    }
    let reingresos: BTreeSet<String> = altas.intersection(&vistas).cloned().collect();
    Err(format!(
        "la lista blanca no es la ultima aprobacion ({}): altas {:?} (reingresos {:?}), bajas sin aprobar {:?}. \
         Una baja se aprueba con un bloque nuevo y su recibo; una alta no se aprueba (va por la puerta)",
        ultima.nombre,
        lista_de(&altas),
        lista_de(&reingresos),
        lista_de(&bajas)
    ))
}

fn conjunto(rutas: &[&str]) -> BTreeSet<String> {
    rutas.iter().map(|r| r.to_string()).collect()
}

fn aprobacion(nombre: &str, bajas: Option<&[&str]>, rutas: &[&str]) -> Aprobacion {
    Aprobacion {
        nombre: nombre.to_string(),
        recibo: format!("recibo de {nombre}"),
        bajas: bajas.map(conjunto),
        rutas: conjunto(rutas),
    }
}

/// C4: los controles del dictamen sobre la regla, con conjuntos sintéticos.
#[test]
fn c4_la_lista_solo_se_achica_por_identidad() {
    let a0 = aprobacion("A0", None, &["a.rs", "b.rs", "c.rs"]);

    // La lista igual a A0 pasa.
    assert!(verificar_aprobaciones(&[a0.clone()], &conjunto(&["a.rs", "b.rs", "c.rs"])).is_ok());

    // Sustitución a igual tamaño: baja de c, alta de d, mismo tamaño.
    let r = verificar_aprobaciones(&[a0.clone()], &conjunto(&["a.rs", "b.rs", "d.rs"]));
    assert!(matches!(r, Err(ref e) if e.contains("d.rs") && e.contains("c.rs")), "sustitucion: {r:?}");

    // Una aprobación que sustituye a igual tamaño tampoco vale.
    let a1_sust = aprobacion("A1", Some(&["c.rs"]), &["a.rs", "b.rs", "d.rs"]);
    let r = verificar_aprobaciones(&[a0.clone(), a1_sust], &conjunto(&["a.rs", "b.rs", "d.rs"]));
    assert!(matches!(r, Err(ref e) if e.contains("no es subconjunto")), "aprobacion sustituta: {r:?}");

    // Reducir con su aprobación pasa.
    let a1 = aprobacion("A1", Some(&["c.rs"]), &["a.rs", "b.rs"]);
    assert!(verificar_aprobaciones(&[a0.clone(), a1.clone()], &conjunto(&["a.rs", "b.rs"])).is_ok());

    // Reingreso tras reducir: c vuelve a la lista (por debajo del techo).
    let r = verificar_aprobaciones(&[a0.clone(), a1.clone()], &conjunto(&["a.rs", "b.rs", "c.rs"]));
    assert!(matches!(r, Err(ref e) if e.contains("reingresos [\"c.rs\"]")), "reingreso: {r:?}");

    // Reingreso con una aprobación nueva que lo vuelve a admitir.
    let a2 = aprobacion("A2", Some(&[]), &["a.rs", "b.rs", "c.rs"]);
    let r = verificar_aprobaciones(&[a0.clone(), a1.clone(), a2], &conjunto(&["a.rs", "b.rs", "c.rs"]));
    assert!(matches!(r, Err(ref e) if e.contains("reingresos [\"c.rs\"]")), "reingreso aprobado: {r:?}");

    // Reducir sin aprobarla: falla, pide el bloque con su recibo.
    let r = verificar_aprobaciones(&[a0.clone()], &conjunto(&["a.rs", "b.rs"]));
    assert!(matches!(r, Err(ref e) if e.contains("bajas sin aprobar")), "baja sin aprobar: {r:?}");

    // El diff declarado tiene que ser el real.
    let a1_mal = aprobacion("A1", Some(&["b.rs"]), &["a.rs", "b.rs"]);
    let r = verificar_aprobaciones(&[a0.clone(), a1_mal], &conjunto(&["a.rs", "b.rs"]));
    assert!(matches!(r, Err(ref e) if e.contains("diff declarado")), "diff: {r:?}");

    // Sin recibo no hay aprobación.
    let mut a1_sin = a1;
    a1_sin.recibo.clear();
    assert!(verificar_aprobaciones(&[a0, a1_sin], &conjunto(&["a.rs", "b.rs"])).is_err());

    // El lector: el archivo real trae A0 con las 21 rutas y su recibo.
    let reales = leer_aprobaciones(APROBACIONES);
    assert_eq!(reales[0].rutas.len(), TECHO_LISTA_BLANCA);
    assert!(!reales[0].recibo.is_empty());
}
