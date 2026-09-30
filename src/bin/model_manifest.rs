//! Generador del manifest de modelos (Ola XLVIII·G).
//!
//! Binario de operación: escanea models/ y escribe el manifest JSON
//! commit-able (el inventario versionado con huellas SHA-256). Correr
//! tras cada promoción y commitar el resultado junto al modelo:
//!
//!   cargo run --bin model_manifest
//!
//! El diff identifica cambios en archivos, no confirma recarga ni habilidad:
//! un hash puede cambiar sólo por formato. MR-01 separa legibilidad JSON,
//! estructura del predictor y promoción (no acreditada por este inventario).

use god_engine_core::ml_registry::escanear_models;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let ahora = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let manifest = escanear_models(Path::new("models"), ahora);
    let legibles = manifest.entries.iter().filter(|e| e.legible).count();
    let validos = manifest.entries.iter()
        .filter(|e| e.valido_estructuralmente == Some(true)).count();
    println!(
        "📦 [MODEL-REGISTRY] {} archivos inventariados; {} JSON legibles; {} estructuras válidas; promoción y activación NO acreditadas",
        manifest.entries.len(), legibles, validos
    );
    for e in &manifest.entries {
        println!(
            "  {:<24} base={:<7.4} arboles={:<4} {} bytes  {}",
            e.key,
            e.base.unwrap_or(f64::NAN),
            e.n_arboles.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
            e.bytes,
            &e.sha256[..12]
        );
        if let Some(reason) = &e.error_estructural {
            println!("    estructura rechazada: {reason}");
        }
    }
    let salida = Path::new("config_dir/models_manifest.json");
    if let Err(e) = manifest.escribir(salida) {
        eprintln!("❌ escribiendo {}: {e}", salida.display());
        std::process::exit(1);
    }
    println!("✅ manifest → {}", salida.display());
}
