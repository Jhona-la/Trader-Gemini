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

/// CL-48: el proceso corre contra demo, testnet o shadow (`SHADOW_MODE`,
/// `USE_TESTNET` o `BINANCE_USE_DEMO` = "true"). Regla única: el
/// `EnvManager::is_demo_env` del binario delega aquí, y los crates que no
/// dependen del binario (diarios de `execution-engine`) la leen de aquí.
pub fn is_demo_env() -> bool {
    ["SHADOW_MODE", "USE_TESTNET", "BINANCE_USE_DEMO"]
        .iter()
        .any(|v| std::env::var(v).is_ok_and(|s| s.trim().eq_ignore_ascii_case("true")))
}

/// CL-48: ruta del estado aprendido y de los diarios de un entorno,
/// `data/{demo|prod}/<archivo>`. Antes la envolvente Kelly, el fee-breaker,
/// el diario de posiciones y el de fills vivían en `data/` sin separar
/// entornos: una sesión de testnet sembraba el posterior Kelly de
/// producción, suspendía símbolos reales y prestaba su τ y su precio de
/// entrada a la adopción y a los cierres de bracket reales.
/// Crea la carpeta del entorno, como `EnvManager::data_path`.
pub fn env_data_path(file_name: &str) -> String {
    let path = env_data_path_for(is_demo_env(), file_name);
    if let Some(dir) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    path
}

/// Variante pura de `env_data_path`: no lee el entorno ni toca el disco.
pub fn env_data_path_for(demo: bool, file_name: &str) -> String {
    format!("data/{}/{file_name}", if demo { "demo" } else { "prod" })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cl48_cada_entorno_tiene_su_carpeta() {
        assert_eq!(env_data_path_for(true, "kelly_envelope.json"), "data/demo/kelly_envelope.json");
        assert_eq!(env_data_path_for(false, "kelly_envelope.json"), "data/prod/kelly_envelope.json");
    }
}
