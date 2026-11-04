use quantum_arena::paths::env_data_path_for;

const WS_CLIENT: &str = include_str!("../src/ws_client.rs");

#[test]
fn raw_ticks_has_separate_demo_and_production_paths() {
    assert_eq!(env_data_path_for(true, "raw_ticks"), "data/demo/raw_ticks");
    assert_eq!(env_data_path_for(false, "raw_ticks"), "data/prod/raw_ticks");
}

#[test]
fn ws_client_delegates_raw_archive_path_to_environment_policy() {
    assert!(
        WS_CLIENT.contains("quantum_arena::paths::env_data_path(\"raw_ticks\")"),
        "ws_client debe resolver raw_ticks mediante la política de entorno"
    );
    assert!(
        !WS_CLIENT.contains("data/raw_ticks"),
        "la ruta compartida sin entorno debe desaparecer de ws_client"
    );
}
