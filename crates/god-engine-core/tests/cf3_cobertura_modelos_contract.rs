//! C-F3.5 (Línea C), hallazgos F3.5-01 y F3.5-06: la cobertura de modelos
//! que se reporta y el inventario que se versiona deben describir lo que el
//! cargador acepta y lo que hay en el repositorio.
//!
//! Teoría que se prueba:
//! - F3.5-01. `ml_coverage::roster_coverage` decide «cubierto» por la sola
//!   PRESENCIA de `{SYM}_MOTOR.json` (o `.bin` sin hermano). El cargador
//!   (`NanoForest::from_data` / `load_model`) exige índices de hijos GLOBALES
//!   dentro de [inicio, fin) de su árbol (error «split has missing or
//!   cross-tree child»). `ml_registry::escanear_models` aplica ese mismo
//!   contrato a los mismos bytes, sin escribir caché, y lo publica en
//!   `valido_estructuralmente`. NEARUSDT_MOTOR.json guarda los índices de los
//!   hijos relativos a su árbol: cuenta como cubierto y el cargador lo
//!   rechaza. Propiedad: todo símbolo en `covered` tiene su `_MOTOR` .json
//!   con `valido_estructuralmente == Some(true)`.
//! - F3.5-06. El manifiesto versionado (`config_dir/models_manifest.json`,
//!   generado el 2026-10-02 en otra máquina) declara válidos y no inertes
//!   siete `_MOTOR`; el `models/` versionado sólo tiene ATOM y BNB válidos.
//!   Propiedad de inventario: todo artefacto que el manifiesto declara válido
//!   y no inerte existe en `models/`.
//!
//! Datos: el `models/` y el manifiesto versionados del repo (sin generador).
//! Son igualdades exactas; no hay tolerancia.
//!
//! Convención de la fase C-F3 (docs/audit/HALLAZGOS_C-F3_2026-10-11.md): un
//! test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto;
//! documenta el defecto y se activa con el arreglo. Para correrlos:
//! `cargo test ... -- --include-ignored`.

use god_engine_core::ml_coverage::roster_coverage;
use god_engine_core::ml_registry::escanear_models;
use std::path::PathBuf;

fn raiz_repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Verdad: la cobertura del roster sólo cuenta modelos que el cargador
/// acepta. Hoy NEARUSDT queda en `covered` con
/// `valido_estructuralmente = Some(false)`.
#[test]
#[ignore = "OPEN F3.5-01 (Línea C: que la cobertura signifique cargabilidad con el veredicto de ml_registry; pedido a Línea E: reexportar NEAR con índices globales o retirarlo): roster_coverage cuenta la sola presencia de NEARUSDT_MOTOR.json, que el cargador rechaza por hijos fuera de su árbol"]
fn cf3_cobertura_del_roster_coincide_con_el_cargador() {
    let dir = raiz_repo().join("models");
    let roster = ["BTCUSDT", "ATOMUSDT", "BNBUSDT", "NEARUSDT"];
    let cobertura = roster_coverage(&roster, &dir);
    // escanear_models es puro: no llama a load_model ni escribe la caché .bin.
    let manifiesto = escanear_models(&dir, 0);
    println!(
        "observado: covered = {:?}, missing = {:?}",
        cobertura.covered, cobertura.missing
    );
    for e in &manifiesto.entries {
        println!(
            "observado: {} ({}): valido_estructuralmente = {:?}, error = {:?}",
            e.key, e.archivo, e.valido_estructuralmente, e.error_estructural
        );
    }
    let mut rechazados = Vec::new();
    for sym in &cobertura.covered {
        let clave = format!("{sym}_MOTOR");
        let entrada = manifiesto
            .entries
            .iter()
            .find(|e| e.key == clave && e.archivo.ends_with(".json"))
            .unwrap_or_else(|| {
                panic!(
                    "{clave} cubierto pero ausente del inventario: {:?}",
                    manifiesto
                        .entries
                        .iter()
                        .map(|e| &e.key)
                        .collect::<Vec<_>>()
                )
            });
        if entrada.valido_estructuralmente != Some(true) {
            rechazados.push((clave, entrada.error_estructural.clone()));
        }
    }
    assert!(
        rechazados.is_empty(),
        "cuentan como cubiertos pero el cargador los rechaza: {rechazados:?}"
    );
}

/// Verdad: todo artefacto que el manifiesto versionado declara válido y no
/// inerte existe en el `models/` versionado. Hoy faltan ADAUSDT, BTCUSDT,
/// SOLUSDT, XLMUSDT y XRPUSDT `_MOTOR`.
///
/// Nota del escéptico de completitud: la prueba codifica una POLÍTICA
/// (versionar `models/` entero, o regenerar el manifiesto con lo versionado)
/// que decide Línea E; Línea C la deja como pedido. Se activa cuando el
/// manifiesto y `models/` describan la misma máquina, por cualquiera de los
/// dos caminos.
#[test]
#[ignore = "OPEN F3.5-06 (pedido a Línea E: decidir si models/ se versiona entero y regenerar el manifiesto; Línea C corrige ml_registry.rs y el heartbeat): el manifiesto de 2026-10-02 describe otra máquina y declara vivos cinco _MOTOR que el repo no tiene"]
fn cf3_models_versionado_coincide_con_manifiesto() {
    let raiz = raiz_repo();
    let texto = std::fs::read_to_string(raiz.join("config_dir/models_manifest.json"))
        .expect("manifiesto versionado");
    let v: serde_json::Value = serde_json::from_str(&texto).expect("manifiesto JSON");
    let entradas = v["entries"].as_array().expect("entries");
    let mut faltan = Vec::new();
    for e in entradas {
        let valido = e["valido_estructuralmente"].as_bool() == Some(true);
        let inerte = !e["inerte_en_roster"].is_null();
        let archivo = e["archivo"].as_str().unwrap_or_default();
        let existe = raiz.join("models").join(archivo).exists();
        println!(
            "observado: {archivo}: valido = {valido}, inerte = {inerte}, en models/ = {existe}"
        );
        if valido && !inerte && !existe {
            faltan.push(archivo.to_string());
        }
    }
    assert!(
        faltan.is_empty(),
        "el manifiesto declara modelos vivos que el repo no tiene: {faltan:?}"
    );
}
