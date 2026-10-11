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

/// CL-48: el VENUE es de prueba: demo, testnet o shadow (`SHADOW_MODE`,
/// `USE_TESTNET` o `BINANCE_USE_DEMO` = "true"). Elige el exchange al que se
/// conecta el proceso (src/symbol_manager.rs, src/bin/god_engine.rs); el
/// `EnvManager::is_demo_env` del binario delega aquí. No decide por sí solo
/// la carpeta del estado: ver `estado_de_produccion` (CL-61).
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
    let modo = std::env::var("TG_GENOME_ENV").ok();
    let produccion = estado_de_produccion(modo.as_deref(), is_demo_env());
    let path = env_data_path_for(!produccion, file_name);
    if let Some(dir) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    path
}

/// CL-61 (LC-F1-01): el estado aprendido de PRODUCCIÓN sólo lo escribe una
/// sesión de dinero real contra el venue real. El modo es `TG_GENOME_ENV`,
/// que el host fija con la puerta de mainnet («E3 — ENTORNO DE GENOMA por
/// modo de operación» en src/bin/god_engine.rs) y que ya separa el linaje
/// del genoma: una sola regla para el genoma y para el estado. Antes la
/// carpeta seguía sólo al venue, y un arranque en papel sin banderas
/// aprendía de cierres simulados en `data/prod`; la sesión armada siguiente
/// restauraba ese posterior Kelly y dimensionaba con él. Sin modo declarado
/// no se presume producción (D-651: el entorno del genoma tampoco tiene
/// valor por defecto).
pub fn estado_de_produccion(modo: Option<&str>, venue_de_prueba: bool) -> bool {
    modo.is_some_and(|m| m.trim() == "prod") && !venue_de_prueba
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

    #[test]
    fn cl61_produccion_exige_modo_real_y_venue_real() {
        assert!(estado_de_produccion(Some("prod"), false));
        assert!(estado_de_produccion(Some(" prod "), false));
        assert!(!estado_de_produccion(Some("prod"), true));
        assert!(!estado_de_produccion(Some("demo"), false));
        assert!(!estado_de_produccion(Some("backtest"), false));
        assert!(!estado_de_produccion(None, false));
    }
}
