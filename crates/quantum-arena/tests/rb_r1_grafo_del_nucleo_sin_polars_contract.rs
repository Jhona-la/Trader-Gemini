//! RB-R1 (Codex RB, 2026-10-19) — LA PILA POLARS FUERA DEL GRAFO DEL NÚCLEO.
//!
//! `polars` (19 crates `polars-*` con `parquet`/`lazy`; `polars_core` medido a
//! 1,2 GB de RSS en un solo rustc) entraba en `god-engine-core`,
//! `backtest-engine` y `evolution-engine` sólo por `data-pipeline`
//! (`macro_data.rs` y `market_context.rs`, que sólo usaba la herramienta
//! `src/bin/download_history.rs`), y `evolution-engine` lo declaraba sin
//! usarlo. RB-R1 retiró el MacroFetcher (duplicaba `macro_history_sync` y
//! nadie leía sus `MACRO_*.parquet`), mudó el fetcher de funding a la
//! herramienta y quitó las dependencias: la clausura de `god-engine-core` en
//! `Cargo.lock` bajó de 459 a 361 paquetes (también salieron
//! `yahoo_finance_api` y 12 dependencias exclusivas suyas).
//!
//! Este contrato impide que vuelvan: recorre la clausura de dependencias de
//! los paquetes del núcleo en `Cargo.lock` (que incluye las de dev y build de
//! los miembros: sobreaproxima, así que también cubre lo que compilan los
//! tests) y exige que ni `polars*` ni `yahoo_finance_api` aparezcan.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

fn raiz_workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// nombre de paquete → nombres de sus dependencias (todas las versiones
/// fusionadas, por eso sobreaproxima).
fn leer_lock(texto: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut grafo: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for bloque in texto.split("[[package]]").skip(1) {
        let mut nombre: Option<String> = None;
        let mut deps: BTreeSet<String> = BTreeSet::new();
        let mut en_deps = false;
        for linea in bloque.lines() {
            let l = linea.trim();
            if en_deps {
                if l.starts_with(']') {
                    en_deps = false;
                    continue;
                }
                // "nombre", "nombre 1.2.3" o "nombre 1.2.3 (registry+…)".
                let entrada = l.trim_end_matches(',').trim_matches('"');
                if let Some(n) = entrada.split_whitespace().next() {
                    if !n.is_empty() {
                        deps.insert(n.to_string());
                    }
                }
                continue;
            }
            if let Some(v) = l.strip_prefix("name = ") {
                if nombre.is_none() {
                    nombre = Some(v.trim_matches('"').to_string());
                }
            } else if l.starts_with("dependencies = [") {
                en_deps = !l.ends_with(']');
                // Lista en una sola línea: dependencies = ["a", "b 1.0"].
                if let (Some(a), Some(b)) = (l.find('['), l.rfind(']')) {
                    for parte in l[a + 1..b].split(',') {
                        let entrada = parte.trim().trim_matches('"');
                        if let Some(n) = entrada.split_whitespace().next() {
                            if !n.is_empty() {
                                deps.insert(n.to_string());
                            }
                        }
                    }
                }
            }
        }
        if let Some(n) = nombre {
            grafo.entry(n).or_default().extend(deps);
        }
    }
    grafo
}

/// Clausura de dependencias de `raiz`, con el padre de cada nodo para poder
/// explicar por dónde entra un paquete.
fn clausura(
    grafo: &BTreeMap<String, BTreeSet<String>>,
    raiz: &str,
) -> BTreeMap<String, Option<String>> {
    let mut padre: BTreeMap<String, Option<String>> = BTreeMap::new();
    let mut cola: VecDeque<String> = VecDeque::new();
    padre.insert(raiz.to_string(), None);
    cola.push_back(raiz.to_string());
    while let Some(x) = cola.pop_front() {
        if let Some(ds) = grafo.get(&x) {
            for d in ds {
                if !padre.contains_key(d) {
                    padre.insert(d.clone(), Some(x.clone()));
                    cola.push_back(d.clone());
                }
            }
        }
    }
    padre
}

fn camino(padre: &BTreeMap<String, Option<String>>, hoja: &str) -> String {
    let mut c = vec![hoja.to_string()];
    let mut x = hoja.to_string();
    while let Some(Some(p)) = padre.get(&x) {
        c.push(p.clone());
        x = p.clone();
    }
    c.reverse();
    c.join(" -> ")
}

fn es_prohibido(nombre: &str) -> bool {
    nombre == "polars" || nombre.starts_with("polars-") || nombre == "yahoo_finance_api"
}

#[test]
fn rb_r1_el_lector_del_lockfile_resuelve_una_clausura_conocida() {
    let lock = r#"
version = 4

[[package]]
name = "a"
version = "0.1.0"
dependencies = [
 "b",
 "c 1.0.0 (registry+https://github.com/rust-lang/crates.io-index)",
]

[[package]]
name = "b"
version = "0.1.0"
dependencies = ["d 2.0.0"]

[[package]]
name = "c"
version = "1.0.0"

[[package]]
name = "d"
version = "2.0.0"
dependencies = [
 "polars-core",
]

[[package]]
name = "polars-core"
version = "0.43.1"

[[package]]
name = "e"
version = "0.1.0"
dependencies = [
 "yahoo_finance_api",
]
"#;
    let g = leer_lock(lock);
    let c = clausura(&g, "a");
    let nombres: BTreeSet<&str> = c.keys().map(String::as_str).collect();
    let esperado: BTreeSet<&str> = ["a", "b", "c", "d", "polars-core"].into_iter().collect();
    assert_eq!(nombres, esperado);
    assert_eq!(camino(&c, "polars-core"), "a -> b -> d -> polars-core");
    assert!(es_prohibido("polars-core") && es_prohibido("polars") && es_prohibido("yahoo_finance_api"));
    assert!(!es_prohibido("polarity") && !es_prohibido("yahoo"));
    assert!(!clausura(&g, "e").contains_key("polars-core"));
}

#[test]
fn rb_r1_ningun_paquete_del_nucleo_compila_polars_ni_yahoo() {
    let ruta = raiz_workspace().join("Cargo.lock");
    let texto = match fs::read_to_string(&ruta) {
        Ok(t) => t,
        Err(e) => panic!("no se pudo leer {}: {e}", ruta.display()),
    };
    let g = leer_lock(&texto);
    let mut violaciones = Vec::new();
    for raiz in [
        "god-engine-core",
        "backtest-engine",
        "evolution-engine",
        "risk-engine",
        "signal-engine",
        "quantum-arena",
        "execution-engine",
        "audit-engine",
        "data-pipeline",
    ] {
        assert!(g.contains_key(raiz), "{raiz} no está en Cargo.lock");
        let c = clausura(&g, raiz);
        assert!(c.len() > 100, "clausura de {raiz} sospechosamente pequeña: {}", c.len());
        for nombre in c.keys().filter(|n| es_prohibido(n)) {
            violaciones.push(format!("{raiz}: {}", camino(&c, nombre)));
        }
    }
    assert!(
        violaciones.is_empty(),
        "RB-R1: dependencias pesadas de vuelta en el grafo del núcleo (cada build de \
         tests de todos los agentes las compilaría):\n{}",
        violaciones.join("\n")
    );
}

#[test]
fn rb_r1_el_fetcher_de_funding_vive_con_su_unico_llamador() {
    let raiz = raiz_workspace();
    let leer = |rel: &str| match fs::read_to_string(raiz.join(rel)) {
        Ok(t) => t.chars().filter(|c| !c.is_whitespace()).collect::<String>(),
        Err(e) => panic!("no se pudo leer {rel}: {e}"),
    };
    let herramienta = leer("src/bin/download_history.rs");
    assert!(herramienta.contains("#[path=\"download_history/market_context.rs\"]modmarket_context;"));
    assert!(herramienta.contains("market_fetcher.fetch_funding_history(symbol)"), "el fetcher se usa");
    assert!(
        !herramienta.contains("MacroFetcher::") && !herramienta.contains("data_pipeline::macro_data"),
        "el MacroFetcher duplicado quedó retirado (sólo puede nombrarse en comentarios)"
    );
    let modulo = leer("src/bin/download_history/market_context.rs");
    assert!(modulo.contains("pubasyncfnfetch_funding_history("));
    let pipeline = leer("crates/data-pipeline/src/lib.rs");
    assert!(!pipeline.contains("pubmodmacro_data;") && !pipeline.contains("pubmodmarket_context;"));
    assert!(!raiz.join("crates/data-pipeline/src/macro_data.rs").exists());
    assert!(!raiz.join("crates/data-pipeline/src/market_context.rs").exists());
}
