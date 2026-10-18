/// Environment Manager for State Separation
/// Resolves paths dynamically based on SHADOW_MODE or USE_TESTNET flags.
pub struct EnvManager;

impl EnvManager {
    /// Determines if the system is currently running in a Demo/Shadow environment.
    /// CL-48: la regla vive en `quantum_arena::paths::is_demo_env`, compartida
    /// con los crates que no dependen del binario.
    #[inline(always)]
    pub fn is_demo_env() -> bool {
        quantum_arena::paths::is_demo_env()
    }

    /// CL-48: misma función que usan los crates sin acceso al binario
    /// (`quantum_arena::paths::env_data_path`): una sola regla de ruta.
    pub fn data_path(file_name: &str) -> String {
        quantum_arena::paths::env_data_path(file_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_manager_data_path() {
        let p_data = EnvManager::data_path("test_state.bin");
        assert!(p_data.contains("data"));
        assert!(p_data.contains("test_state.bin"));
    }
}
