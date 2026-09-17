//! D-07 — RAÍZ DE DATOS centralizada para crates que no dependen del binario.
//! Resuelve TGM_DATA_DIR o "data" (default). Elimina CWD-dependency.

#[inline(always)]
pub fn data_root() -> String {
    std::env::var("TGM_DATA_DIR")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "data".to_string())
}

#[inline(always)]
pub fn data_join(sub: &str) -> String {
    format!("{}/{}", data_root(), sub)
}
