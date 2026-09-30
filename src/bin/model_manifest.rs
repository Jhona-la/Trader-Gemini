//! Generador del manifest de modelos (Ola XLVIII·G).
//!
//! Binario de operación: escanea models/ y escribe el manifest JSON
//! commit-able (el inventario versionado con huellas SHA-256). Correr
//! tras cada promoción y commitar el resultado junto al modelo:
//!
//!   cargo run --bin model_manifest
//!
//! El diff del manifest ES el changelog de modelos: nueva clave =
//! símbolo desbloqueado; hash cambiado = re-entrenamiento; base
//! cambiada = recalibración que el gate ML consume.

use god_engine_core::ml_registry::escanear_models;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let ahora = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let manifest = escanear_models(Path::new("models"), ahora);
    let promovidos = manifest.entries.iter().filter(|e| e.legible).count();
    println!(
        "📦 [MODEL-REGISTRY] {} modelos promovidos ({} entradas, {} ilegibles)",
        promovidos,
        manifest.entries.len(),
        manifest.entries.len() - promovidos
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
    }
    let salida = Path::new("config_dir/models_manifest.json");
    if let Err(e) = manifest.escribir(salida) {
        eprintln!("❌ escribiendo {}: {e}", salida.display());
        std::process::exit(1);
    }
    println!("✅ manifest → {}", salida.display());
}
