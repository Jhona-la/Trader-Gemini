use std::env;
use std::fs;

    /// Environment Manager for State Separation
    /// Resolves paths dynamically based on SHADOW_MODE or USE_TESTNET flags.
    pub struct EnvManager;

    impl EnvManager {
        /// D-07 — RAÍZ DE DATOS centralizada: resuelve TGM_DATA_DIR o el
        /// default "data" relativo al CWD. Todos los paths de datos deben
        /// usar esto para eliminar la dependencia del directorio de
        /// lanzamiento (Docker, Task Scheduler, servicio).
        pub fn data_root() -> String {
            std::env::var("TGM_DATA_DIR")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| "data".to_string())
        }

        /// Helper: unir la raíz con un subpath ("historical/BTCUSDT.parquet").
        pub fn data_join(sub: &str) -> String {
            let root = Self::data_root();
            format!("{}/{}", root, sub)
        }

        /// Determines if the system is currently running in a Demo/Shadow environment
        #[inline(always)]
        pub fn is_demo_env() -> bool {
            let shadow = env::var("SHADOW_MODE").unwrap_or_default().trim().to_lowercase() == "true";
            let testnet = env::var("USE_TESTNET").unwrap_or_default().trim().to_lowercase() == "true";
            let demo = env::var("BINANCE_USE_DEMO").unwrap_or_default().trim().to_lowercase() == "true";
            shadow || testnet || demo
        }

        /// Helper to resolve a path based on the environment and create directories if they don't exist
        fn resolve_path(base_dir: &str, file_name: Option<&str>) -> String {
            let env_folder = if Self::is_demo_env() { "demo" } else { "prod" };
            let dir_path = format!("{}/{}", base_dir, env_folder);
            
            // Ensure directory exists
            let _ = fs::create_dir_all(&dir_path);

            if let Some(file) = file_name {
                format!("{}/{}", dir_path, file)
            } else {
                dir_path
            }
        }

        pub fn data_path(file_name: &str) -> String {
            Self::resolve_path("data", Some(file_name))
        }

        pub fn model_path(file_name: &str) -> String {
            Self::resolve_path("models", Some(file_name))
        }

    pub fn log_path(file_name: &str) -> String {
        Self::resolve_path("logs", Some(file_name))
    }

    /// FIX #734: Extracción sanitizada con .trim() automático de variables de entorno
    #[inline(always)]
    pub fn get_var(key: &str) -> Option<String> {
        env::var(key).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_manager_paths_and_vars() {
        let p_data = EnvManager::data_path("test_state.bin");
        assert!(p_data.contains("data"));
        assert!(p_data.contains("test_state.bin"));

        let p_model = EnvManager::model_path("weights.bin");
        assert!(p_model.contains("models"));
        assert!(p_model.contains("weights.bin"));

        let p_log = EnvManager::log_path("app.log");
        assert!(p_log.contains("logs"));
        assert!(p_log.contains("app.log"));

        let var = EnvManager::get_var("NON_EXISTENT_VAR_12345");
        assert!(var.is_none());
    }
}
