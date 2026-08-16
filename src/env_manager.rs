use std::env;
use std::fs;

    /// Environment Manager for State Separation
    /// Resolves paths dynamically based on SHADOW_MODE or USE_TESTNET flags.
    pub struct EnvManager;

    impl EnvManager {
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
}
